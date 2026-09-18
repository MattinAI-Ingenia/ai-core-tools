"""Unit tests for utils.blob_url_guard — anti-SSRF checks on account_url.

No real network calls are made: an autouse fixture replaces socket.getaddrinfo
with a version that fails loudly unless a test explicitly monkeypatches it
again for its own scenario.
"""
import socket

import pytest

from utils.blob_url_guard import BlobUrlRejected, assert_allowed_account_url, normalize_account_url


def _fake_getaddrinfo_global(*_args, **_kwargs):
    """Resolves to a globally routable address (a real Azure Storage range)."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("20.60.72.1", 0))]


def _fake_getaddrinfo_private(*_args, **_kwargs):
    """Resolves to a loopback address, simulating DNS rebinding / SSRF."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]


@pytest.fixture(autouse=True)
def _block_real_dns(monkeypatch):
    """Guard against any test in this module accidentally reaching the real network."""

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("Unexpected real DNS resolution attempt in a unit test")

    monkeypatch.setattr(socket, "getaddrinfo", _forbidden)


class TestNormalizeAccountUrl:
    def test_valid_url_is_normalized(self):
        assert normalize_account_url("https://Foo.BLOB.core.windows.net/container?x=1") == (
            "https://foo.blob.core.windows.net"
        )

    def test_unparseable_value_rejected_as_malformed(self):
        with pytest.raises(BlobUrlRejected) as exc_info:
            normalize_account_url("not a url")
        assert exc_info.value.code == "MALFORMED_URL"

    def test_empty_string_rejected_as_malformed(self):
        with pytest.raises(BlobUrlRejected) as exc_info:
            normalize_account_url("")
        assert exc_info.value.code == "MALFORMED_URL"

    def test_non_https_scheme_rejected(self):
        with pytest.raises(BlobUrlRejected) as exc_info:
            normalize_account_url("http://foo.blob.core.windows.net")
        assert exc_info.value.code == "SCHEME_NOT_ALLOWED"

    def test_bracketed_ipv6_host_is_rebracketed_not_truncated(self):
        """Regression: rebuilding the URL without brackets around an IPv6 host
        made it unparseable (or silently truncated) on the next urlsplit() call."""
        assert normalize_account_url("https://[2603:1030::1]") == "https://[2603:1030::1]"

    def test_bracketed_ipv6_loopback_is_rebracketed(self):
        assert normalize_account_url("https://[::1]") == "https://[::1]"


class TestAssertAllowedAccountUrl:
    def test_valid_url_accepted(self, monkeypatch):
        monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_global)
        result = assert_allowed_account_url("https://etiquetas.blob.core.windows.net", resolve=True)
        assert result == "https://etiquetas.blob.core.windows.net"

    def test_http_scheme_rejected(self):
        with pytest.raises(BlobUrlRejected) as exc_info:
            assert_allowed_account_url("http://etiquetas.blob.core.windows.net/etiquetas", resolve=True)
        assert exc_info.value.code == "SCHEME_NOT_ALLOWED"

    def test_metadata_endpoint_rejected(self):
        with pytest.raises(BlobUrlRejected):
            assert_allowed_account_url("http://169.254.169.254/latest/meta-data", resolve=True)

    def test_host_outside_allowlist_rejected(self):
        with pytest.raises(BlobUrlRejected) as exc_info:
            assert_allowed_account_url("https://evil.com", resolve=True)
        assert exc_info.value.code == "HOST_NOT_ALLOWED"

    def test_suffix_without_leading_dot_does_not_allow_prefixed_lookalike_host(self, monkeypatch):
        """Regression: an operator adding 'mycorp.com' (no leading dot, the
        natural way to write it) to BLOB_ALLOWED_HOST_SUFFIXES must not also
        silently allow 'evilmycorp.com' via a bare string .endswith() match."""
        monkeypatch.setenv("BLOB_ALLOWED_HOST_SUFFIXES", "mycorp.com")
        with pytest.raises(BlobUrlRejected) as exc_info:
            assert_allowed_account_url("https://evilmycorp.com", resolve=True)
        assert exc_info.value.code == "HOST_NOT_ALLOWED"

    def test_suffix_without_leading_dot_still_allows_subdomain_and_exact_host(self, monkeypatch):
        monkeypatch.setenv("BLOB_ALLOWED_HOST_SUFFIXES", "mycorp.com")
        monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_global)

        assert assert_allowed_account_url("https://sub.mycorp.com", resolve=True) == "https://sub.mycorp.com"
        assert assert_allowed_account_url("https://mycorp.com", resolve=True) == "https://mycorp.com"

    def test_ip_literal_rejected(self):
        with pytest.raises(BlobUrlRejected) as exc_info:
            assert_allowed_account_url("https://20.60.72.1", resolve=True)
        assert exc_info.value.code == "IP_LITERAL_NOT_ALLOWED"

    def test_bracketed_ipv6_literal_rejected(self):
        with pytest.raises(BlobUrlRejected) as exc_info:
            assert_allowed_account_url("https://[2603:1030::1]", resolve=True)
        assert exc_info.value.code == "IP_LITERAL_NOT_ALLOWED"

    def test_bracketed_ipv6_loopback_rejected_without_crashing(self):
        """Regression: urlsplit(<unbracketed rebuilt IPv6 host>).hostname used to
        return None/a truncated fragment, which either raised an unhandled
        AttributeError or let the IP-literal check silently miss the address."""
        with pytest.raises(BlobUrlRejected) as exc_info:
            assert_allowed_account_url("https://[::1]", resolve=True)
        assert exc_info.value.code == "IP_LITERAL_NOT_ALLOWED"

    def test_bracketed_ipv4_mapped_ipv6_metadata_address_rejected(self):
        with pytest.raises(BlobUrlRejected) as exc_info:
            assert_allowed_account_url("https://[::ffff:169.254.169.254]", resolve=True)
        assert exc_info.value.code == "IP_LITERAL_NOT_ALLOWED"

    def test_ip_literal_matching_a_misconfigured_allowlist_suffix_still_rejected(self, monkeypatch):
        """Defense in depth (AD-4): even if BLOB_ALLOWED_HOST_SUFFIXES is ever
        misconfigured loosely enough to match an IP-shaped host, the dedicated
        IP-literal check still blocks it with its own code — it runs before the
        suffix check specifically so this stays true."""
        monkeypatch.setenv("BLOB_ALLOWED_HOST_SUFFIXES", ".1")
        with pytest.raises(BlobUrlRejected) as exc_info:
            assert_allowed_account_url("https://127.0.0.1", resolve=True)
        assert exc_info.value.code == "IP_LITERAL_NOT_ALLOWED"

    def test_allowlisted_host_resolving_to_loopback_rejected(self, monkeypatch):
        """An allowlisted hostname (passes scheme + suffix + not-an-IP-literal)
        that resolves to a private/loopback address must still be rejected."""
        monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_private)
        with pytest.raises(BlobUrlRejected) as exc_info:
            assert_allowed_account_url("https://etiquetas.blob.core.windows.net", resolve=True)
        assert exc_info.value.code == "PRIVATE_ADDRESS_NOT_ALLOWED"

    def test_dns_resolution_failure_rejected(self, monkeypatch):
        def _raise(*_args, **_kwargs):
            raise socket.gaierror("name resolution failed")

        monkeypatch.setattr(socket, "getaddrinfo", _raise)
        with pytest.raises(BlobUrlRejected) as exc_info:
            assert_allowed_account_url("https://etiquetas.blob.core.windows.net", resolve=True)
        assert exc_info.value.code == "DNS_RESOLUTION_FAILED"

    def test_resolve_false_skips_dns_check_entirely(self):
        """resolve=False must never touch socket.getaddrinfo (the autouse fixture
        would raise AssertionError if it did)."""
        result = assert_allowed_account_url("https://etiquetas.blob.core.windows.net", resolve=False)
        assert result == "https://etiquetas.blob.core.windows.net"
