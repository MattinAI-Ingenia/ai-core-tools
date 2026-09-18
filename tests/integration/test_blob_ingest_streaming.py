"""AC-10: streaming and staging hygiene.

* The blob download goes through ``readinto`` (never an eager ``readall()``) —
  the fake downloader literally has no ``readall``, so a regression to eager
  reads fails loudly.
* The per-batch staging directory under ``TMP_BASE_FOLDER/blob_ingest_staging``
  is removed when the run finishes — success or per-blob failure — so the
  ``file_cleanup_worker`` sweep is only ever a crash backstop, not the norm.
"""
import os

import pytest

from tests.integration.blob_ingest_helpers import (
    FakeBlob,
    trigger_ingestion,
)


@pytest.mark.usefixtures("tmp_repo_base")
def test_blobs_stream_through_readinto_into_an_emptied_staging_dir(
    db, repository, fake_blob_client, instant_indexing, tmp_staging_root,
):
    fake_blob_client.add_blob(FakeBlob("docs/a.pdf", content=b"%PDF-1.4 stream-a"))
    fake_blob_client.add_blob(FakeBlob("docs/b.pdf", content=b"%PDF-1.4 stream-b"))

    result = trigger_ingestion(repository, db)

    assert result["queued"] == 2
    # Each blob was streamed exactly once, via readinto — never readall.
    assert sorted(fake_blob_client.download_blob_calls) == ["docs/a.pdf", "docs/b.pdf"]
    for blob_name in fake_blob_client.download_blob_calls:
        downloader = fake_blob_client.last_downloaders[blob_name]
        assert downloader.readinto_calls == 1

    # The staging root exists for the run but holds nothing afterwards: every
    # batch removes its own directory before returning (NFR-2).
    assert os.path.isdir(tmp_staging_root)
    assert os.listdir(tmp_staging_root) == []


@pytest.mark.usefixtures("tmp_repo_base")
def test_a_failed_download_leaves_no_staging_behind(
    db, repository, fake_blob_client, instant_indexing, tmp_staging_root,
):
    from azure.core.exceptions import ServiceResponseError

    good_blob = fake_blob_client.add_blob(FakeBlob("docs/ok.pdf"))
    bad_blob = fake_blob_client.add_blob(FakeBlob("docs/broken.pdf", etag='"etag-b"'))
    bad_blob.download_error = ServiceResponseError("connection reset mid-stream")

    result = trigger_ingestion(repository, db)

    # The good blob still went through; the broken one counts as failed.
    assert result["queued"] == 1
    assert result["failed"] == 1
    assert good_blob.name in fake_blob_client.download_blob_calls
    assert bad_blob.name in fake_blob_client.download_blob_calls
    # Staging is clean even when part of the batch failed.
    assert os.listdir(tmp_staging_root) == []
