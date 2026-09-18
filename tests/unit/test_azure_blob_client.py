"""Unit tests for services.blob.azure_blob_client.

No real Azure calls are made: build_container_client is monkeypatched to
return an in-memory fake wherever list_blobs / download_to_path are exercised,
so these tests never depend on SSRF resolution or a real network.
"""
import socket
from types import SimpleNamespace

import pytest
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
    ServiceResponseError,
)

from services.blob import azure_blob_client as blob_client
from services.blob.azure_blob_client import (
    BlobConnectionError,
    BlobSourceConfig,
    build_container_client,
    classify_error,
    download_to_path,
    list_blobs,
    sanitize_azure_error,
)


def _fake_getaddrinfo_global(*_args, **_kwargs):
    """Resolves to a globally routable address (a real Azure Storage range)."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("20.60.72.1", 0))]


@pytest.fixture(autouse=True)
def _block_real_dns(monkeypatch):
    """Guard against any test in this module accidentally reaching the real network.

    build_container_client / download_to_path re-validate account_url via
    assert_allowed_account_url(resolve=True), which calls socket.getaddrinfo. Without
    this fixture, tests that don't explicitly stub the SSRF check would silently depend
    on real DNS resolution of 'acct.blob.core.windows.net'.
    """
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_global)


def _http_error(status_code: int, message: str) -> HttpResponseError:
    """Build an HttpResponseError with a status_code set without needing a real response object."""
    exc = HttpResponseError(message)
    exc.status_code = status_code
    return exc


def _cfg(**overrides) -> BlobSourceConfig:
    defaults = dict(
        account_url="https://acct.blob.core.windows.net",
        container="etiquetas",
        prefix=None,
        auth_mode="ANONYMOUS",
        sas_token=None,
    )
    defaults.update(overrides)
    return BlobSourceConfig(**defaults)


# ---------------------------------------------------------------------------
# classify_error
# ---------------------------------------------------------------------------

class TestClassifyError:
    @pytest.mark.parametrize(
        "exc, expected_code",
        [
            (ClientAuthenticationError("Server failed to authenticate the request."), "AUTH_FAILED"),
            (ResourceNotFoundError("The specified container does not exist."), "CONTAINER_NOT_FOUND"),
            (_http_error(403, "forbidden"), "LISTING_FORBIDDEN"),
            (_http_error(404, "not found"), "CONTAINER_NOT_FOUND"),
            (_http_error(401, "unauthorized"), "AUTH_FAILED"),
            (socket.timeout("timed out"), "TIMEOUT"),
            (ServiceRequestError("Connection aborted"), "ACCOUNT_UNREACHABLE"),
            (ServiceResponseError("Connection aborted"), "ACCOUNT_UNREACHABLE"),
            (ConnectionError("connection refused"), "ACCOUNT_UNREACHABLE"),
            (ValueError("some other unrelated failure"), "DOWNLOAD_FAILED"),
        ],
    )
    def test_error_classification_table(self, exc, expected_code):
        assert classify_error(exc) == expected_code

    def test_timeout_detected_by_message_when_type_is_generic(self):
        """azure-core sometimes wraps a socket timeout inside a generic
        ServiceResponseError whose message mentions the timeout — the type
        check alone (isinstance ServiceResponseError) would misclassify this
        as ACCOUNT_UNREACHABLE, so classify_error inspects status_code first
        and falls back to text heuristics only for truly generic exceptions."""
        assert classify_error(TimeoutError("Read timed out")) == "TIMEOUT"


# ---------------------------------------------------------------------------
# sanitize_azure_error
# ---------------------------------------------------------------------------

class TestSanitizeAzureError:
    def test_strips_sas_query_string_from_url_in_message(self):
        secret_query = "sv=2024-11-04&ss=b&srt=co&sp=rl&se=2030-01-01T00%3A00%3A00Z&sig=SUPERSECRETSIG"
        message = (
            f"Server failed to authenticate the request. Make sure the value of the "
            f"Authorization header ... https://acct.blob.core.windows.net/etiquetas?{secret_query} "
            f"is formed correctly."
        )
        sanitized = sanitize_azure_error(message, secret_query)

        assert "SUPERSECRETSIG" not in sanitized
        assert secret_query not in sanitized
        assert "https://acct.blob.core.windows.net/etiquetas" in sanitized

    def test_redacts_standalone_sig_and_se_params_outside_a_url(self):
        message = "Invalid token: sv=2024&sig=SUPERSECRETSIG&se=2030-01-01"
        sanitized = sanitize_azure_error(message, None)

        assert "SUPERSECRETSIG" not in sanitized
        assert "sig=***" in sanitized
        assert "se=***" in sanitized

    def test_redacts_shared_access_signature_pattern(self):
        message = "Auth failed for SharedAccessSignature=abcdef123456"
        sanitized = sanitize_azure_error(message, None)
        assert "abcdef123456" not in sanitized

    def test_caps_length_at_500_chars(self):
        sanitized = sanitize_azure_error("x" * 1000, None)
        assert len(sanitized) == 500

    def test_accepts_exception_instance_directly(self):
        exc = ValueError("token=SECRETVALUE leaked")
        sanitized = sanitize_azure_error(exc, "SECRETVALUE")
        assert "SECRETVALUE" not in sanitized


# ---------------------------------------------------------------------------
# build_container_client
# ---------------------------------------------------------------------------

class TestBuildContainerClient:
    def test_passes_permit_redirects_false(self, monkeypatch):
        """The SDK's default RedirectPolicy would follow a redirect without
        re-checking it against the anti-SSRF allowlist (only the first hop is
        validated) — redirects must be disabled outright."""
        captured_kwargs = {}

        class _FakeSdkContainerClient:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        monkeypatch.setattr(blob_client, "ContainerClient", _FakeSdkContainerClient)

        build_container_client(_cfg())

        assert captured_kwargs["permit_redirects"] is False
        assert captured_kwargs["retry_total"] == 2

    def test_sdk_value_error_wrapped_as_invalid_config(self, monkeypatch):
        """A bare ValueError from the SDK's own __init__ validation (e.g. an
        empty container name) must not propagate as an unhandled 500."""

        class _RaisingSdkContainerClient:
            def __init__(self, **kwargs):
                raise ValueError("container_name must not be empty")

        monkeypatch.setattr(blob_client, "ContainerClient", _RaisingSdkContainerClient)

        with pytest.raises(BlobConnectionError) as exc_info:
            build_container_client(_cfg(container=""))
        assert exc_info.value.code == "INVALID_CONFIG"

    def test_unsupported_auth_mode_rejected(self):
        with pytest.raises(BlobConnectionError) as exc_info:
            build_container_client(_cfg(auth_mode="CONNECTION_STRING"))
        assert exc_info.value.code == "INVALID_CONFIG"

    def test_sas_token_mode_without_token_rejected(self):
        with pytest.raises(BlobConnectionError) as exc_info:
            build_container_client(_cfg(auth_mode="SAS_TOKEN", sas_token=None))
        assert exc_info.value.code == "INVALID_CONFIG"


# ---------------------------------------------------------------------------
# list_blobs — lazy pagination
# ---------------------------------------------------------------------------

def _fake_blob(name, size=100, etag='"abc"', content_type=None):
    content_settings = SimpleNamespace(content_type=content_type) if content_type is not None else None
    return SimpleNamespace(name=name, size=size, etag=etag, last_modified=None, content_settings=content_settings)


class _CountingPages:
    """Wraps a list of pages, counting how many have actually been iterated."""

    def __init__(self, pages):
        self._pages = pages
        self.pages_yielded = 0

    def __iter__(self):
        for page in self._pages:
            self.pages_yielded += 1
            yield page


class _FakeListBlobsHandle:
    def __init__(self, pages):
        self._pages = pages
        self.last_by_page = None

    def by_page(self, continuation_token=None):
        counting = _CountingPages(self._pages)
        self.last_by_page = counting
        return counting


class _FakeContainerClient:
    def __init__(self, pages, url="https://acct.blob.core.windows.net/etiquetas"):
        # Faithful to the real SDK: ContainerClient.url re-appends the raw credential
        # for SAS_TOKEN auth (verified against installed azure-storage-blob 12.30.1). A
        # fake that always hardcodes a clean URL here would hide the class of bug where
        # production code reads this field instead of the already-sanitized account_url.
        self.url = url
        self._handle = _FakeListBlobsHandle(pages)

    def list_blobs(self, name_starts_with=None, results_per_page=None):
        return self._handle


class TestListBlobsLazyPagination:
    def test_validates_eagerly_before_returning_the_iterator(self, monkeypatch):
        """Regression: list_blobs used to be a generator function itself, so
        build_container_client (and therefore BlobUrlRejected/INVALID_CONFIG)
        didn't run until the first next() call. It must now raise from the
        call to list_blobs() directly, before any iteration."""
        # Bypass the anti-SSRF/DNS check itself (already covered by
        # test_blob_url_guard.py) so this test stays hermetic and isolates the
        # auth_mode validation this regression is actually about.
        monkeypatch.setattr(blob_client, "assert_allowed_account_url", lambda raw, **kwargs: raw)

        with pytest.raises(BlobConnectionError) as exc_info:
            list_blobs(_cfg(auth_mode="SAS_TOKEN", sas_token=None), extensions=set(), page_size=10, max_items=10)
        assert exc_info.value.code == "INVALID_CONFIG"

    def test_does_not_materialize_all_pages_when_max_items_stops_early(self, monkeypatch):
        pages = [[_fake_blob("a.pdf")], [_fake_blob("b.pdf")], [_fake_blob("c.pdf")]]
        fake_client = _FakeContainerClient(pages)
        monkeypatch.setattr(blob_client, "build_container_client", lambda cfg: fake_client)

        results = list(list_blobs(_cfg(), extensions=set(), page_size=1, max_items=1))

        assert [b.name for b in results] == ["a.pdf"]
        # Only the first page should ever have been pulled from the SDK's
        # auto-paging iterator — pages 2 and 3 must never be fetched.
        assert fake_client._handle.last_by_page.pages_yielded == 1

    def test_filters_by_extension(self, monkeypatch):
        pages = [[_fake_blob("a.pdf"), _fake_blob("b.txt"), _fake_blob("c.PDF")]]
        fake_client = _FakeContainerClient(pages)
        monkeypatch.setattr(blob_client, "build_container_client", lambda cfg: fake_client)

        results = list(list_blobs(_cfg(), extensions={".pdf"}, page_size=10, max_items=10))

        assert [b.name for b in results] == ["a.pdf", "c.PDF"]

    def test_wraps_azure_errors_as_blob_connection_error(self, monkeypatch):
        class _FailingContainerClient(_FakeContainerClient):
            def list_blobs(self, name_starts_with=None, results_per_page=None):
                raise ResourceNotFoundError("The specified container does not exist.")

        monkeypatch.setattr(blob_client, "build_container_client", lambda cfg: _FailingContainerClient([]))

        with pytest.raises(BlobConnectionError) as exc_info:
            list(list_blobs(_cfg(), extensions=set(), page_size=10, max_items=10))
        assert exc_info.value.code == "CONTAINER_NOT_FOUND"

    def test_wraps_generic_os_errors_too(self, monkeypatch):
        """Regression: list_blobs used to only catch AzureError, letting a bare
        ConnectionError (or any other OSError) escape completely unsanitized."""

        class _FailingContainerClient(_FakeContainerClient):
            def list_blobs(self, name_starts_with=None, results_per_page=None):
                raise ConnectionError("connection refused, token=SUPERSECRETSIG leaked")

        monkeypatch.setattr(blob_client, "build_container_client", lambda cfg: _FailingContainerClient([]))

        with pytest.raises(BlobConnectionError) as exc_info:
            list(
                list_blobs(
                    _cfg(auth_mode="SAS_TOKEN", sas_token="SUPERSECRETSIG"),
                    extensions=set(),
                    page_size=10,
                    max_items=10,
                )
            )
        assert exc_info.value.code == "ACCOUNT_UNREACHABLE"
        assert "SUPERSECRETSIG" not in exc_info.value.message

    def test_remote_blob_url_never_carries_the_sas_credential(self, monkeypatch):
        """Regression (CRITICAL): ContainerClient.url re-appends the raw SAS token for
        SAS_TOKEN auth. RemoteBlob.url must be built from the credential-free,
        already-validated account_url — never from container_client.url."""
        secret_query = "sv=2024-11-04&sig=SUPERSECRETSIG&se=2030-01-01"
        pages = [[_fake_blob("reports/a.pdf")]]
        # Deliberately mimics the real SDK's leaky .url so this test would fail if
        # production code ever went back to reading it.
        leaky_url = f"https://acct.blob.core.windows.net/etiquetas?{secret_query}"
        fake_client = _FakeContainerClient(pages, url=leaky_url)
        monkeypatch.setattr(blob_client, "build_container_client", lambda cfg: fake_client)

        cfg = _cfg(auth_mode="SAS_TOKEN", sas_token=secret_query)
        results = list(list_blobs(cfg, extensions=set(), page_size=10, max_items=10))

        assert len(results) == 1
        assert "sig=" not in results[0].url
        assert "SUPERSECRETSIG" not in results[0].url
        assert results[0].url == "https://acct.blob.core.windows.net/etiquetas/reports/a.pdf"

    def test_remote_blob_url_percent_encodes_blob_name(self, monkeypatch):
        pages = [[_fake_blob("a report (final).pdf")]]
        fake_client = _FakeContainerClient(pages)
        monkeypatch.setattr(blob_client, "build_container_client", lambda cfg: fake_client)

        results = list(list_blobs(_cfg(), extensions=set(), page_size=10, max_items=10))

        assert " " not in results[0].url
        assert results[0].url == "https://acct.blob.core.windows.net/etiquetas/a%20report%20%28final%29.pdf"


# ---------------------------------------------------------------------------
# download_to_path — streaming + cleanup
# ---------------------------------------------------------------------------

class _FakeDownloader:
    def __init__(self, size, content):
        self.properties = SimpleNamespace(size=size)
        self._content = content
        self.readinto_called = False

    def readinto(self, stream):
        self.readinto_called = True
        stream.write(self._content)


class _FakeBlobClientForDownload:
    def __init__(self, downloader=None, raise_on_download=None):
        self._downloader = downloader
        self._raise_on_download = raise_on_download

    def download_blob(self):
        if self._raise_on_download:
            raise self._raise_on_download
        return self._downloader


class _FakeContainerClientForDownload:
    def __init__(self, blob_client_instance, url="https://acct.blob.core.windows.net/etiquetas?sig=SUPERSECRETSIG"):
        # Faithful to the real SDK (see _FakeContainerClient above): .url may carry a raw
        # SAS credential. download_to_path must never read this field for anything.
        self.url = url
        self._blob_client_instance = blob_client_instance

    def get_blob_client(self, blob_name):
        return self._blob_client_instance


class TestDownloadToPath:
    def test_uses_readinto_not_readall(self, monkeypatch, tmp_path):
        downloader = _FakeDownloader(size=5, content=b"hello")
        assert not hasattr(downloader, "readall")
        fake_container_client = _FakeContainerClientForDownload(_FakeBlobClientForDownload(downloader))
        monkeypatch.setattr(blob_client, "build_container_client", lambda cfg: fake_container_client)

        dest = tmp_path / "out.pdf"
        size = download_to_path(_cfg(), "a.pdf", str(dest))

        assert downloader.readinto_called is True
        assert size == 5
        assert dest.read_bytes() == b"hello"

    def test_file_too_large_rejected_without_downloading(self, monkeypatch, tmp_path):
        downloader = _FakeDownloader(size=10_000_000, content=b"x" * 10)
        fake_container_client = _FakeContainerClientForDownload(_FakeBlobClientForDownload(downloader))
        monkeypatch.setattr(blob_client, "build_container_client", lambda cfg: fake_container_client)

        dest = tmp_path / "out.pdf"
        with pytest.raises(BlobConnectionError) as exc_info:
            download_to_path(_cfg(), "a.pdf", str(dest), max_bytes=1_000)

        assert exc_info.value.code == "FILE_TOO_LARGE"
        assert downloader.readinto_called is False
        assert not dest.exists()

    def test_file_too_large_rejected_after_download_when_reported_size_is_none(self, monkeypatch, tmp_path):
        """The server may report no size (or understate it). The pre-download
        check alone would silently skip the limit in that case, so the actual
        bytes written to disk must be checked too, after the download completes."""
        downloader = _FakeDownloader(size=None, content=b"x" * 2_000)
        fake_container_client = _FakeContainerClientForDownload(_FakeBlobClientForDownload(downloader))
        monkeypatch.setattr(blob_client, "build_container_client", lambda cfg: fake_container_client)

        dest = tmp_path / "out.pdf"
        with pytest.raises(BlobConnectionError) as exc_info:
            download_to_path(_cfg(), "a.pdf", str(dest), max_bytes=1_000)

        assert exc_info.value.code == "FILE_TOO_LARGE"
        assert downloader.readinto_called is True
        assert not dest.exists()

    def test_removes_partial_file_on_download_error(self, monkeypatch, tmp_path):
        class _RaisingDownloader(_FakeDownloader):
            def readinto(self, stream):
                stream.write(b"partial-content")
                raise ServiceResponseError("connection reset")

        downloader = _RaisingDownloader(size=100, content=b"unused")
        fake_container_client = _FakeContainerClientForDownload(_FakeBlobClientForDownload(downloader))
        monkeypatch.setattr(blob_client, "build_container_client", lambda cfg: fake_container_client)

        dest = tmp_path / "out.pdf"
        with pytest.raises(BlobConnectionError) as exc_info:
            download_to_path(_cfg(), "a.pdf", str(dest))

        assert exc_info.value.code == "ACCOUNT_UNREACHABLE"
        assert not dest.exists()

    def test_removes_partial_file_when_download_blob_itself_raises(self, monkeypatch, tmp_path):
        fake_container_client = _FakeContainerClientForDownload(
            _FakeBlobClientForDownload(raise_on_download=ResourceNotFoundError("blob not found"))
        )
        monkeypatch.setattr(blob_client, "build_container_client", lambda cfg: fake_container_client)

        dest = tmp_path / "out.pdf"
        with pytest.raises(BlobConnectionError) as exc_info:
            download_to_path(_cfg(), "missing.pdf", str(dest))

        assert exc_info.value.code == "CONTAINER_NOT_FOUND"
        assert not dest.exists()

    def test_sanitizes_sas_token_out_of_download_error_message(self, monkeypatch, tmp_path):
        secret = "sv=2024&sig=SUPERSECRETSIG"
        fake_container_client = _FakeContainerClientForDownload(
            _FakeBlobClientForDownload(
                raise_on_download=ClientAuthenticationError(f"Auth failed with token {secret}")
            )
        )
        monkeypatch.setattr(blob_client, "build_container_client", lambda cfg: fake_container_client)

        dest = tmp_path / "out.pdf"
        with pytest.raises(BlobConnectionError) as exc_info:
            download_to_path(_cfg(auth_mode="SAS_TOKEN", sas_token=secret), "a.pdf", str(dest))

        assert secret not in exc_info.value.message
        assert "SUPERSECRETSIG" not in exc_info.value.message
