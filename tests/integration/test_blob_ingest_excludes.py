"""``name_excludes`` — the per-name blacklist filter.

Complements ``prefixes`` (which keeps only blobs that START with one of the
given substrings): a blob whose name CONTAINS any exclude is never listed,
counted, or ingested — e.g. ``["_"]`` drops generated duplicates like
``CDOC002817_2a67fdb9.pdf`` while keeping ``CDOC000933.pdf``. Azure cannot
filter by substring server-side, so the check runs client-side in
``azure_blob_client._iter_blobs``, before the ``max_items`` budget counter
touches the blob.
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
def test_excludes_drop_blobs_whose_name_contains_the_substring(db, repository, fake_blob_client):
    fake_blob_client.add_blob(FakeBlob("CDOC000933.pdf", etag='"keep"'))
    fake_blob_client.add_blob(FakeBlob("CDOC002817_2a67fdb9.pdf", etag='"drop"'))
    fake_blob_client.add_blob(FakeBlob("DSAT001234.pdf", etag='"keep2"'))

    result = trigger_ingestion(repository, db, prefixes=["CDOC", "DSAT"], name_excludes=["_"])

    assert result["total_blobs"] == 2
    assert result["queued"] == 2
    assert sorted(r.extra_metadata["blob_name"] for r in azure_blob_resources(db, repository)) == [
        "CDOC000933.pdf", "DSAT001234.pdf",
    ]


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content", "instant_indexing")
def test_excluded_blobs_never_consume_the_max_items_budget(db, repository, fake_blob_client, monkeypatch):
    """Exclusions run before the budget counter: with a cap of 2 and one
    excluded blob among the matches, both REMAINING blobs must fit within the
    cap — an excluded blob is never one of them."""
    fake_blob_client.add_blob(FakeBlob("CDOC_a.pdf", etag='"excluded"'))
    fake_blob_client.add_blob(FakeBlob("CDOC1.pdf", etag='"e1"'))
    fake_blob_client.add_blob(FakeBlob("CDOC2.pdf", etag='"e2"'))
    fake_blob_client.add_blob(FakeBlob("CDOC3.pdf", etag='"e3"'))

    monkeypatch.setenv("BLOB_LIST_MAX_ITEMS", "2")
    result = trigger_ingestion(repository, db, prefixes=["CDOC"], name_excludes=["_"])

    assert result["total_blobs"] == 2
    assert sorted(r.extra_metadata["blob_name"] for r in azure_blob_resources(db, repository)) == [
        "CDOC1.pdf", "CDOC2.pdf",
    ]


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content", "instant_indexing")
def test_excludes_apply_to_single_file_runs_too(db, repository, fake_blob_client):
    """Uniform rule: the blacklist filters every listing, including an exact
    ``blob_name`` request — a file whose name carries an excluded substring is
    by definition excluded."""
    from utils.error_handlers import ValidationError

    fake_blob_client.add_blob(FakeBlob("CDOC002817_2a67fdb9.pdf", etag='"e1"'))

    with pytest.raises(ValidationError, match="not found"):
        trigger_ingestion(repository, db, blob_name="CDOC002817_2a67fdb9.pdf", name_excludes=["_"])


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content", "instant_indexing")
def test_excludes_are_persisted_for_the_update_flow(db, repository, fake_blob_client):
    fake_blob_client.add_blob(FakeBlob("CDOC000933.pdf", etag='"e1"'))

    trigger_ingestion(repository, db, name_excludes=["_", "-"])

    assert repository.azure_blob_source == {
        "account_url": ACCOUNT_URL,
        "container": CONTAINER,
        "prefixes": [],
        "name_excludes": ["_", "-"],
        "auth_mode": "ANONYMOUS",
    }
