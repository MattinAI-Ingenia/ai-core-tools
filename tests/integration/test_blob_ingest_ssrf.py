"""AC-4: anti-SSRF — an attacker-influenced ``account_url`` is rejected with
422 BEFORE any outbound request is made.

Every case here asserts the fake SDK client's call counters stay at zero: the
rejection happens in ``blob_url_guard.assert_allowed_account_url``, which runs
both in ``trigger_ingestion`` and (defense in depth, AD-4) again inside
``build_container_client`` — never letting a call reach the transport layer.
"""
import pytest

from tests.integration.blob_ingest_helpers import (
    ACCOUNT_URL,
    BLOB_INGEST_URL,
    CONTAINER,
    FakeContainerClient,
    ingest_payload,
)


@pytest.fixture
def loopback_dns(monkeypatch):
    """Resolve every host to 127.0.0.1 — the cloud-metadata scenario."""

    def _resolve_to_loopback(*_args, **_kwargs):
        return [(2, 1, 6, "", ("127.0.0.1", 0))]

    monkeypatch.setattr("socket.getaddrinfo", _resolve_to_loopback)


def _post(client, repository, owner_headers, **payload):
    return client.post(
        BLOB_INGEST_URL.format(app_id=repository.app_id, repository_id=repository.repository_id),
        json=ingest_payload(**payload),
        headers=owner_headers,
    )


class TestAccountUrlRejectedBeforeAnyOutboundCall:
    def test_http_scheme_rejected(self, client, repository, owner_headers, fake_blob_client):
        response = _post(client, repository, owner_headers,
                         account_url="http://169.254.169.254/latest/meta-data")

        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "SCHEME_NOT_ALLOWED"
        assert fake_blob_client.list_calls == 0

    def test_ip_literal_rejected_even_over_https(self, client, repository, owner_headers, fake_blob_client):
        response = _post(client, repository, owner_headers, account_url="https://169.254.169.254")

        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "IP_LITERAL_NOT_ALLOWED"
        assert fake_blob_client.list_calls == 0

    def test_host_outside_the_allowlist_rejected(self, client, repository, owner_headers, fake_blob_client):
        response = _post(client, repository, owner_headers,
                         account_url="https://evil.example.com", container=CONTAINER)

        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "HOST_NOT_ALLOWED"
        # No SDK call was ever made — the rejection happened before listing.
        assert fake_blob_client.list_calls == 0

    def test_allowlisted_host_resolving_to_loopback_rejected(
        self, client, repository, owner_headers, fake_blob_client, loopback_dns,
    ):
        """A host that passes the allowlist but resolves to a private/loopback
        address (169.254.169.254 et al.) is still caught — defense in depth."""
        assert isinstance(fake_blob_client, FakeContainerClient)
        response = _post(client, repository, owner_headers,
                         account_url=ACCOUNT_URL, container=CONTAINER)

        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "PRIVATE_ADDRESS_NOT_ALLOWED"
        assert fake_blob_client.list_calls == 0

    def test_service_level_rejection_matches_the_http_contract(self, db, repository, fake_blob_client):
        """The same guarantee at the service boundary — before the run lock is
        even considered."""
        from services.azure_blob_ingest_service import AzureBlobIngestService
        from utils.blob_url_guard import BlobUrlRejected

        with pytest.raises(BlobUrlRejected) as exc_info:
            AzureBlobIngestService.trigger_ingestion(
                repository.app_id, repository.repository_id,
                ingest_payload(account_url="https://evil.example.com"), db,
            )
        assert exc_info.value.code == "HOST_NOT_ALLOWED"
        assert fake_blob_client.list_calls == 0
