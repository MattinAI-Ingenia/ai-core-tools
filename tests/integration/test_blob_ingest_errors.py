"""AC-5/AC-6: classified Azure errors map to distinct, sanitized HTTP statuses,
and triggering into a busy silo is refused with an actionable 409.

AC-5 — an invalid/expired SAS → AUTH_FAILED; a missing container →
CONTAINER_NOT_FOUND; a container the credential cannot list → LISTING_FORBIDDEN.
None of the three response bodies may contain the configured ``sas_token``.

AC-6 — the silo's advisory indexing lock is checked read-only (never forced):
while another holder has it, the endpoint refuses immediately with 409.
"""
import pytest

from services import silo_indexing_lock

from tests.integration.blob_ingest_helpers import (
    AUTH_FAILED_ERROR,
    BLOB_INGEST_URL,
    CONTAINER_NOT_FOUND_ERROR,
    LISTING_FORBIDDEN_ERROR,
    SAS_TOKEN,
    ingest_payload,
)


def _post(client, repository, owner_headers, **payload):
    return client.post(
        BLOB_INGEST_URL.format(app_id=repository.app_id, repository_id=repository.repository_id),
        json=ingest_payload(**payload),
        headers=owner_headers,
    )


class TestClassifiedErrors:
    @pytest.mark.parametrize(
        "list_error, expected_code, expected_status",
        [
            (AUTH_FAILED_ERROR, "AUTH_FAILED", 422),
            (CONTAINER_NOT_FOUND_ERROR, "CONTAINER_NOT_FOUND", 422),
            (LISTING_FORBIDDEN_ERROR, "LISTING_FORBIDDEN", 422),
        ],
        ids=["auth_failed", "container_not_found", "listing_forbidden"],
    )
    def test_classified_error_is_a_sanitized_422(
        self, client, repository, owner_headers, fake_blob_client, fake_dns,
        list_error, expected_code, expected_status,
    ):
        fake = fake_blob_client
        fake.list_error = list_error

        response = _post(client, repository, owner_headers,
                         auth_mode="SAS_TOKEN", sas_token=SAS_TOKEN)

        assert response.status_code == expected_status
        detail = response.json()["detail"]
        assert detail["code"] == expected_code
        # AC-5/NFR-3: the configured sas_token never reaches the response.
        assert SAS_TOKEN not in response.text
        assert "SUPERSECRETSIG" not in response.text

    def test_unreachable_account_is_a_502_not_a_422(
        self, client, repository, owner_headers, fake_blob_client, fake_dns,
    ):
        """ACCOUNT_UNREACHABLE/TIMEOUT are upstream connectivity problems — the
        router maps them to 502 (Bad Gateway), everything else to 422."""
        from azure.core.exceptions import ServiceRequestError

        fake_blob_client.list_error = ServiceRequestError("connection refused")

        response = _post(client, repository, owner_headers)

        assert response.status_code == 502
        assert response.json()["detail"]["code"] == "ACCOUNT_UNREACHABLE"


class TestBusySiloConflict:
    def test_trigger_into_a_locked_silo_is_a_409_with_an_actionable_message(
        self, client, repository, owner_headers, fake_blob_client, fake_dns,
    ):
        fake_blob_client.add_blob("docs/a.pdf")

        holder = silo_indexing_lock.acquire(repository.silo_id)
        assert holder is not None
        try:
            response = _post(client, repository, owner_headers)
        finally:
            silo_indexing_lock.release(holder, repository.silo_id)

        assert response.status_code == 409
        detail = response.json()["detail"]
        assert "wait for it to complete" in detail
        # Read-only check: the listing never ran, nothing was downloaded.
        assert fake_blob_client.list_calls == 0
        assert fake_blob_client.download_blob_calls == []
