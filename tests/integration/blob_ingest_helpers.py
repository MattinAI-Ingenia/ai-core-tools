"""Shared helpers for the Azure Blob ingestion integration tests.

Every test here exercises the REAL ``azure_blob_client.list_blobs`` /
``download_to_path`` / ``classify_error`` / sanitization code paths — only the
``ContainerClient`` itself is replaced by :class:`FakeContainerClient` (a
faithful stand-in for the exact SDK surface ``azure_blob_client`` uses), so
no test ever touches the real network.

DNS is stubbed too: ``assert_allowed_account_url(resolve=True)`` calls
``socket.getaddrinfo``, and letting it resolve the fake account host for real
would both slow the suite down and depend on the sandbox's network policy.
The pytest fixtures that wire the fakes in live in
``tests/integration/conftest.py``; this module only carries data structures
and call helpers.
"""
import logging
import socket
from types import SimpleNamespace

from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceNotFoundError

ACCOUNT_URL = "https://acct.blob.core.windows.net"
CONTAINER = "etiquetas"

# Recognizable request-scoped SAS token used by the secret-hygiene tests (AC-9):
# distinctive enough that a substring assert can never false-positive.
SAS_TOKEN = "sv=2024-11-04&ss=b&srt=co&sp=rl&se=2030-01-01T00%3A00%3A00Z&sig=SUPERSECRETSIG"

BLOB_INGEST_URL = "/internal/apps/{app_id}/repositories/{repository_id}/ingest-azure-blobs"
BLOB_PREVIEW_URL = "/internal/apps/{app_id}/repositories/{repository_id}/preview-azure-blobs"


def http_error(status_code: int, message: str) -> HttpResponseError:
    """Build an ``HttpResponseError`` with a status code without a real response."""
    exc = HttpResponseError(message)
    exc.status_code = status_code
    return exc


AUTH_FAILED_ERROR = ClientAuthenticationError(
    "Server failed to authenticate the request with token " + SAS_TOKEN
)
CONTAINER_NOT_FOUND_ERROR = ResourceNotFoundError("The specified container does not exist.")
LISTING_FORBIDDEN_ERROR = http_error(403, "Public access is not permitted on this container")


def fake_dns_global(*_args, **_kwargs):
    """Resolve every host to a globally routable address (a real Azure range)."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("20.60.72.1", 0))]


# ---------------------------------------------------------------------------
# FakeContainerClient
# ---------------------------------------------------------------------------


class FakeBlob:
    """One blob served by :class:`FakeContainerClient`."""

    def __init__(self, name, content=b"%PDF-1.4 fake", size=None, etag='"etag-1"',
                 content_type="application/pdf"):
        self.name = name
        self.content = content
        self.size = size if size is not None else len(content)
        self.etag = etag
        self.content_type = content_type
        self.download_error = None
        self.container = None

    def as_listing_item(self):
        content_settings = SimpleNamespace(content_type=self.content_type)
        return SimpleNamespace(
            name=self.name, size=self.size, etag=self.etag,
            last_modified=None, content_settings=content_settings,
        )


class _FakeListBlobsHandle:
    def __init__(self, pages):
        self._pages = pages

    def by_page(self, continuation_token=None):
        return iter(self._pages)


class FakeDownloader:
    """A blob downloader that streams via ``readinto`` — and has no ``readall``
    at all, so a production regression to eager reads fails loudly (AC-10)."""

    def __init__(self, blob):
        self.properties = SimpleNamespace(size=blob.size)
        self._blob = blob
        self.readinto_calls = 0

    def readinto(self, stream):
        self.readinto_calls += 1
        stream.write(self._blob.content)


class FakeBlobClient:
    def __init__(self, blob):
        self._blob = blob

    def download_blob(self):
        container = self._blob.container
        container.download_blob_calls.append(self._blob.name)
        if self._blob.download_error is not None:
            raise self._blob.download_error
        downloader = FakeDownloader(self._blob)
        container.last_downloaders[self._blob.name] = downloader
        return downloader


class FakeContainerClient:
    """In-memory stand-in for ``azure.storage.blob.ContainerClient``.

    Covers exactly the surface ``azure_blob_client`` uses:

    * ``list_blobs(name_starts_with=..., results_per_page=...)`` → ``.by_page()``
      (lazy-ish pages, honoring the prefix filter),
    * ``get_blob_client(name).download_blob()`` → downloader with
      ``properties.size`` and ``readinto(stream)``.

    Failure injection + call counters let tests assert idempotency (AC-2: zero
    downloads), SSRF (AC-4: zero outbound calls) and streaming (AC-10).
    """

    def __init__(self, blobs=None, list_error=None, sas_token=None):
        self._blobs = {}
        for blob in (blobs or []):
            self.add_blob(blob)
        self.list_error = list_error
        self.list_calls = 0
        self.download_blob_calls = []
        # Faithful to the real SDK: .url re-appends the raw SAS credential —
        # production code must never read it (the unit tests guard this too).
        self.url = f"{ACCOUNT_URL}/{CONTAINER}" + (f"?{sas_token}" if sas_token else "")
        # blob_name -> the FakeDownloader handed to azure_blob_client, so tests
        # can assert streaming behaviour (readinto vs readall).
        self.last_downloaders = {}

    def add_blob(self, blob):
        if isinstance(blob, str):
            blob = FakeBlob(blob)
        blob.container = self
        self._blobs[blob.name] = blob
        return blob

    def list_blobs(self, name_starts_with=None, results_per_page=None):
        self.list_calls += 1
        if self.list_error is not None:
            raise self.list_error
        blobs = [
            b.as_listing_item() for b in self._blobs.values()
            if not name_starts_with or b.name.startswith(name_starts_with)
        ]
        page_size = max(1, results_per_page or len(blobs) or 1)
        pages = [blobs[i:i + page_size] for i in range(0, len(blobs), page_size)]
        return _FakeListBlobsHandle(pages)

    def get_blob_client(self, blob_name):
        if blob_name not in self._blobs:
            raise ResourceNotFoundError(f"Blob {blob_name} not found")
        return FakeBlobClient(self._blobs[blob_name])


# ---------------------------------------------------------------------------
# Log capture (backend loggers set propagate=False, so pytest's caplog
# cannot see them — AC-9 needs a direct handler instead).
# ---------------------------------------------------------------------------


class _RecordingHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


class LogCapture:
    """Handler-based capture of the ingestion path's backend loggers."""

    INGESTION_LOGGERS = (
        "services.azure_blob_ingest_service",
        "services.blob.azure_blob_client",
        "services.resource_service",
        "routers.internal.repositories",
    )

    def __init__(self):
        self.handler = _RecordingHandler()
        self._attached = []

    def __enter__(self):
        for name in self.INGESTION_LOGGERS:
            lg = logging.getLogger(name)
            lg.addHandler(self.handler)
            self._attached.append(lg)
        return self

    def __exit__(self, *exc_info):
        for lg in self._attached:
            lg.removeHandler(self.handler)
        return False

    def all_text(self) -> str:
        """Every record, formatted — the closest thing to 'grep the logs'."""
        return "\n".join(
            self.handler.format(r) if r.exc_info else r.getMessage()
            for r in self.handler.records
        )


# ---------------------------------------------------------------------------
# Trigger helpers
# ---------------------------------------------------------------------------


def ingest_payload(**overrides) -> dict:
    """A valid request body for ``trigger_ingestion`` / the HTTP endpoint."""
    payload = {"account_url": ACCOUNT_URL, "container": CONTAINER}
    payload.update(overrides)
    return payload


def trigger_ingestion(repository, db, **payload_overrides) -> dict:
    """Run ``AzureBlobIngestService.trigger_ingestion`` synchronously."""
    from services.azure_blob_ingest_service import AzureBlobIngestService

    return AzureBlobIngestService.trigger_ingestion(
        repository.app_id, repository.repository_id, ingest_payload(**payload_overrides), db,
    )


def resources_of(db, repository):
    """The repository's Resource rows, ordered by id."""
    from models.resource import Resource

    return (
        db.query(Resource)
        .filter(Resource.repository_id == repository.repository_id)
        .order_by(Resource.resource_id)
        .all()
    )


def azure_blob_resources(db, repository, container=CONTAINER, account_url=ACCOUNT_URL):
    """The repository's Resources created by an Azure blob ingestion run."""
    return [
        r for r in resources_of(db, repository)
        if (r.extra_metadata or {}).get("source_type") == "azure_blob"
        and (r.extra_metadata or {}).get("container") == container
        and (r.extra_metadata or {}).get("account_url") == account_url
    ]
