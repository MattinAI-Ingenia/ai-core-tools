"""The preview endpoint: a dry-run of the listing + idempotency diff behind
``ingest-azure-blobs``, so the UI can show how many documents the current
filters match BEFORE the user triggers a Load.

A preview must not ingest anything: no downloads, no ``Resource`` rows, no
remembered source (``Repository.azure_blob_source``) — and it never needs the
silo or run locks, so it is answerable even while a real ingestion runs.
"""
import pytest

from services import silo_indexing_lock
from services.azure_blob_ingest_service import AzureBlobIngestService

from tests.integration.blob_ingest_helpers import (
    BLOB_PREVIEW_URL,
    FakeBlob,
    azure_blob_resources,
    ingest_payload,
    trigger_ingestion,
)


def _post(client, repository, owner_headers, **payload):
    return client.post(
        BLOB_PREVIEW_URL.format(app_id=repository.app_id, repository_id=repository.repository_id),
        json=ingest_payload(**payload),
        headers=owner_headers,
    )


class TestPreviewEndpoint:
    def test_reports_counts_without_ingesting_anything(
        self, client, repository, owner_headers, fake_blob_client, fake_dns, db,
    ):
        fake_blob_client.add_blob(FakeBlob("CDOC000933.pdf", etag='"e1"'))
        fake_blob_client.add_blob(FakeBlob("DSAT001234.pdf", etag='"e2"'))
        fake_blob_client.add_blob(FakeBlob("OTRO000111.pdf", etag='"e3"'))

        response = _post(client, repository, owner_headers, prefixes=["CDOC", "DSAT"])

        assert response.status_code == 200
        assert response.json() == {"total_blobs": 2, "pending_blobs": 2}
        assert fake_blob_client.download_blob_calls == []
        assert azure_blob_resources(db, repository) == []
        # The remembered source is written by a successful ingest only.
        assert repository.azure_blob_source is None

    def test_prefixes_and_blob_name_stay_mutually_exclusive(
        self, client, repository, owner_headers, fake_blob_client, fake_dns,
    ):
        fake_blob_client.add_blob(FakeBlob("CDOC000933.pdf"))

        response = _post(client, repository, owner_headers, prefixes=["CDOC"], blob_name="CDOC000933.pdf")

        assert response.status_code == 422
        assert "mutually exclusive" in response.json()["detail"]

    def test_more_than_16_prefixes_is_a_422_not_a_listing_fan_out(
        self, client, repository, owner_headers, fake_blob_client, fake_dns,
    ):
        """The schema caps the prefix list: one server-side listing runs per
        prefix, so an unbounded list would let one request hammer the storage
        account with arbitrarily many listing calls."""
        response = _post(client, repository, owner_headers, prefixes=[f"p{i}" for i in range(17)])

        assert response.status_code == 422
        assert fake_blob_client.list_calls == 0

    def test_runs_even_while_the_silo_is_busy(
        self, client, repository, owner_headers, fake_blob_client, fake_dns,
    ):
        """The preview is read-only: it never acquires the silo or run locks,
        so a count is answerable while a real ingestion is running."""
        fake_blob_client.add_blob(FakeBlob("CDOC000933.pdf", etag='"e1"'))

        holder = silo_indexing_lock.acquire(repository.silo_id)
        assert holder is not None
        try:
            response = _post(client, repository, owner_headers)
        finally:
            silo_indexing_lock.release(holder, repository.silo_id)

        assert response.status_code == 200
        assert response.json() == {"total_blobs": 1, "pending_blobs": 1}


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content", "instant_indexing")
def test_pending_count_shrinks_after_an_ingest(db, repository, fake_blob_client):
    """``pending_blobs`` is the diff against already-ingested blobs — after a
    successful Load the same preview reports everything as up to date."""
    for i in range(3):
        fake_blob_client.add_blob(FakeBlob(f"CDOC000{i}.pdf", etag=f'"e{i}"'))

    first = trigger_ingestion(repository, db)
    assert first["queued"] == 3

    preview = AzureBlobIngestService.preview_ingestion(
        repository.app_id, repository.repository_id, ingest_payload(prefixes=["CDOC"]), db,
    )

    assert preview == {"total_blobs": 3, "pending_blobs": 0}
