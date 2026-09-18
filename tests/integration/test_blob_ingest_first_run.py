"""AC-1: one POST creates one ``Resource`` per new blob, through the real
``create_multiple_resources`` pipeline.

The FR-1 architectural guard is asserted too: ``SiloService.index_single_content``
must never be called on this path — everything goes through the manual-upload
pipeline's resource creation, inheriting its validation, background indexing
and SSE progress.
"""
import os

import pytest

from tests.integration.blob_ingest_helpers import (
    ACCOUNT_URL,
    FakeBlob,
    azure_blob_resources,
    resources_of,
    trigger_ingestion,
)


def _repo_files_root(repository):
    from services.resource_service import REPO_BASE_FOLDER

    return os.path.join(REPO_BASE_FOLDER, str(repository.repository_id))


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content")
def test_n_blobs_create_n_resources_with_provenance_metadata(db, repository, fake_blob_client, instant_indexing):
    fake_blob_client.add_blob(FakeBlob("docs/etiqueta-a.pdf", etag='"etag-a"'))
    fake_blob_client.add_blob(FakeBlob("docs/etiqueta-b.pdf", etag='"etag-b"'))

    result = trigger_ingestion(repository, db)

    assert result["queued"] == 2
    assert result["skipped_unchanged"] == 0
    assert result["skipped_unsupported"] == 0
    assert result["failed"] == 0
    assert result["session_id"] == "sid"

    resources = azure_blob_resources(db, repository)
    assert len(resources) == 2
    assert len(instant_indexing) == 1  # one background run handed the whole batch

    by_blob_name = {(r.extra_metadata or {}).get("blob_name"): r for r in resources}
    for blob_name, etag in [("docs/etiqueta-a.pdf", '"etag-a"'), ("docs/etiqueta-b.pdf", '"etag-b"')]:
        resource = by_blob_name[blob_name]
        metadata = resource.extra_metadata
        assert metadata["source_type"] == "azure_blob"
        assert metadata["account_url"] == ACCOUNT_URL
        assert metadata["container"] == "etiquetas"
        assert metadata["blob_name"] == blob_name
        assert metadata["blob_url"] == f"{ACCOUNT_URL}/etiquetas/{blob_name}"
        assert metadata["etag"] == etag
        # The staged file was moved into the repository folder — the Resource's
        # backing file really exists on the pipeline's canonical path.
        assert resource.uri
        assert os.path.getsize(os.path.join(_repo_files_root(repository), resource.uri)) > 0

    # The listing was fetched exactly once; each blob downloaded exactly once.
    assert fake_blob_client.list_calls == 1
    assert sorted(fake_blob_client.download_blob_calls) == ["docs/etiqueta-a.pdf", "docs/etiqueta-b.pdf"]


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content")
def test_only_blobs_matching_the_extension_filter_are_ingested(db, repository, fake_blob_client, instant_indexing):
    fake_blob_client.add_blob(FakeBlob("docs/a.pdf"))
    fake_blob_client.add_blob(FakeBlob("docs/notes.txt"))
    fake_blob_client.add_blob(FakeBlob("docs/data.exe"))

    result = trigger_ingestion(repository, db, file_extension_filters=["pdf", "txt"])

    assert result["queued"] == 2
    # An extension outside the filter is dropped at listing time (never
    # downloaded); skipped_unsupported is only a defense-in-depth counter for
    # classify_blobs receiving a different set than the listing did.
    assert result["skipped_unsupported"] == 0
    assert sorted(r.extra_metadata["blob_name"] for r in azure_blob_resources(db, repository)) == [
        "docs/a.pdf", "docs/notes.txt",
    ]
    assert "docs/data.exe" not in fake_blob_client.download_blob_calls


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content")
def test_prefix_limits_the_listing(db, repository, fake_blob_client, instant_indexing):
    fake_blob_client.add_blob(FakeBlob("2025/old.pdf"))
    fake_blob_client.add_blob(FakeBlob("2026/new.pdf"))

    result = trigger_ingestion(repository, db, prefix="2026/")

    assert result["queued"] == 1
    assert [r.extra_metadata["blob_name"] for r in azure_blob_resources(db, repository)] == ["2026/new.pdf"]


@pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content")
def test_container_with_no_matching_blobs_is_a_noop_202(db, repository, fake_blob_client, instant_indexing):
    """Edge case (spec.md): empty/filtered container → all counters 0, and the
    Resource pipeline is never even entered."""
    fake_blob_client.add_blob(FakeBlob("archive.zip"))

    result = trigger_ingestion(repository, db)

    assert result == {
        "queued": 0, "skipped_unchanged": 0, "skipped_unsupported": 0, "failed": 0,
        "session_id": None, "total_blobs": 0, "pending_blobs": 0,
    }
    assert fake_blob_client.list_calls == 1
    assert fake_blob_client.download_blob_calls == []
    assert resources_of(db, repository) == []
