"""Multi-prefix filtering for the Azure blob listing.

Why this exists: the real container holds thousands of blobs but only a
subset is relevant for a load (name prefixes like ``CDOC``/``DSAT``) — the
single ``prefix`` field could not express "CDOC *or* DSAT", so a flat Load
pulled the whole (capped at ``BLOB_LIST_MAX_ITEMS``) container.

The listing therefore runs one server-side ``name_starts_with`` call per
requested prefix — the only way Azure can filter — unioned by blob name so
overlapping prefixes never yield a blob twice, with the run's
``BLOB_LIST_MAX_ITEMS`` ceiling spent across the whole union.
"""
import pytest

from tests.integration.blob_ingest_helpers import (
    ACCOUNT_URL,
    CONTAINER,
    FakeBlob,
    azure_blob_resources,
    trigger_ingestion,
)


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content", "instant_indexing")
def test_multiple_prefixes_ingest_the_union_of_matching_blobs(db, repository, fake_blob_client):
    fake_blob_client.add_blob(FakeBlob("CDOC000933.pdf", etag='"e1"'))
    fake_blob_client.add_blob(FakeBlob("DSAT001234.pdf", etag='"e2"'))
    fake_blob_client.add_blob(FakeBlob("OTRO000111.pdf", etag='"e3"'))

    result = trigger_ingestion(repository, db, prefixes=["CDOC", "DSAT"])

    assert result["total_blobs"] == 2
    assert result["queued"] == 2
    assert sorted(r.extra_metadata["blob_name"] for r in azure_blob_resources(db, repository)) == [
        "CDOC000933.pdf", "DSAT001234.pdf",
    ]
    # One server-side listing per prefix — never one for the whole container.
    assert fake_blob_client.list_calls == 2


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content", "instant_indexing")
def test_overlapping_prefixes_yield_each_blob_once(db, repository, fake_blob_client):
    """``CDOC`` and ``CDOC0`` both match CDOC000933.pdf — the union is
    de-duplicated by blob name, so every matching blob is ingested exactly
    once, in first-match (prefix order) listing order."""
    fake_blob_client.add_blob(FakeBlob("CDOC000933.pdf", etag='"e1"'))
    fake_blob_client.add_blob(FakeBlob("CDOC000934.pdf", etag='"e2"'))
    fake_blob_client.add_blob(FakeBlob("DSAT001234.pdf", etag='"e3"'))

    result = trigger_ingestion(repository, db, prefixes=["CDOC", "CDOC0", "DSAT"])

    assert result["total_blobs"] == 3
    assert result["queued"] == 3
    assert len(azure_blob_resources(db, repository)) == 3
    # Order-insensitive: downloads run concurrently (ThreadPoolExecutor), so
    # completion order is not deterministic.
    assert sorted(fake_blob_client.download_blob_calls) == [
        "CDOC000933.pdf", "CDOC000934.pdf", "DSAT001234.pdf",
    ]


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content", "instant_indexing")
def test_prefixes_are_normalized_before_listing(db, repository, fake_blob_client):
    """Whitespace is stripped, empties dropped and literal duplicates
    de-duplicated — none of them may waste a server-side listing call."""
    fake_blob_client.add_blob(FakeBlob("CDOC000933.pdf", etag='"e1"'))

    result = trigger_ingestion(repository, db, prefixes=[" CDOC ", "CDOC", "", "  "])

    assert result["total_blobs"] == 1
    assert result["queued"] == 1
    assert fake_blob_client.list_calls == 1


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content", "instant_indexing")
def test_max_items_cap_is_shared_across_prefixes(db, repository, fake_blob_client, monkeypatch):
    """The run's ``BLOB_LIST_MAX_ITEMS`` ceiling applies to the union of the
    per-prefix listings, not to each call — a run can never exceed the cap no
    matter how many prefixes are requested."""
    for i in range(4):
        fake_blob_client.add_blob(FakeBlob(f"CDOC000{i}.pdf", etag=f'"c{i}"'))
    for i in range(4):
        fake_blob_client.add_blob(FakeBlob(f"DSAT000{i}.pdf", etag=f'"d{i}"'))

    monkeypatch.setenv("BLOB_LIST_MAX_ITEMS", "5")
    result = trigger_ingestion(repository, db, prefixes=["CDOC", "DSAT"])

    # 4 CDOC blobs plus the first DSAT one — the cap is spent in prefix order.
    assert result["total_blobs"] == 5
    assert len(azure_blob_resources(db, repository)) == 5


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content", "instant_indexing")
def test_run_persists_the_prefixes_for_the_update_flow(db, repository, fake_blob_client):
    fake_blob_client.add_blob(FakeBlob("CDOC000933.pdf", etag='"e1"'))

    trigger_ingestion(repository, db, prefixes=["CDOC", "DSAT"])

    assert repository.azure_blob_source == {
        "account_url": ACCOUNT_URL,
        "container": CONTAINER,
        "prefixes": ["CDOC", "DSAT"],
        "name_excludes": [],
        "auth_mode": "ANONYMOUS",
    }
