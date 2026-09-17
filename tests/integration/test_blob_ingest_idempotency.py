"""AC-2/AC-3: idempotency across re-runs of the same container.

AC-2 — a second run against an unchanged container must not download anything,
must not create new Resources, and must never take the *silo* indexing lock
(the per-repository *run* lock, a negative advisory id, is a different lock and
is legitimately taken by every run).

AC-3 — a blob whose ETag changed is superseded index-then-swap: the new
Resource is created first, and only then is the stale one deleted, so the silo
never ends up with zero indexed Resources for that blob.
"""
import os

import pytest

from models.resource import Resource
from services import silo_indexing_lock
from services.resource_service import ResourceService

from tests.integration.blob_ingest_helpers import (
    FakeBlob,
    azure_blob_resources,
    resources_of,
    trigger_ingestion,
)


@pytest.fixture
def lock_spy(monkeypatch):
    """Record every ``silo_indexing_lock.acquire`` lock id."""
    calls = []
    real_acquire = silo_indexing_lock.acquire
    monkeypatch.setattr(
        silo_indexing_lock, "acquire",
        lambda silo_id: (calls.append(silo_id), real_acquire(silo_id))[1],
    )
    return calls


class TestSecondRunSkipsEverything:
    """AC-2."""

    @pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content")
    def test_unchanged_container_downloads_nothing_and_takes_no_silo_lock(
        self, db, repository, fake_blob_client, instant_indexing, lock_spy,
    ):
        fake_blob_client.add_blob(FakeBlob("docs/a.pdf", etag='"etag-1"'))
        fake_blob_client.add_blob(FakeBlob("docs/b.pdf", etag='"etag-2"'))

        first = trigger_ingestion(repository, db)
        assert first["queued"] == 2

        downloads_after_first_run = len(fake_blob_client.download_blob_calls)
        creates_after_first_run = len(instant_indexing)
        # The first run legitimately acquires the silo lock (through
        # create_multiple_resources); only the second run's behaviour matters.
        lock_uses_after_first_run = list(lock_spy)

        second = trigger_ingestion(repository, db)

        assert second["queued"] == 0
        assert second["skipped_unchanged"] == 2
        assert second["failed"] == 0
        assert second["session_id"] is None
        assert len(fake_blob_client.download_blob_calls) == downloads_after_first_run  # no downloads
        assert len(instant_indexing) == creates_after_first_run  # no create_multiple_resources call
        assert len(resources_of(db, repository)) == 2  # no new Resources

        # AC-2, second half: during the unchanged-only run the per-SILO lock
        # (positive advisory id) is never acquired — the run only ever touches
        # its own per-repository run lock (negative id).
        second_run_lock_uses = lock_spy[len(lock_uses_after_first_run):]
        assert [c for c in second_run_lock_uses if c == repository.silo_id] == []
        assert all(c < 0 for c in second_run_lock_uses), second_run_lock_uses

    @pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content")
    def test_etag_missing_on_the_remote_side_is_never_treated_as_unchanged(
        self, db, repository, fake_blob_client, instant_indexing,
    ):
        """Edge case (spec.md): a blob without an ETag is always re-ingested
        (worst case redundant work), never silently skipped as unchanged."""
        fake_blob_client.add_blob(FakeBlob("docs/a.pdf", etag=None))

        first = trigger_ingestion(repository, db)
        assert first["queued"] == 1
        second = trigger_ingestion(repository, db)

        assert second["queued"] == 1
        assert second["skipped_unchanged"] == 0


class TestChangedBlobSupersede:
    """AC-3: index-then-swap — create first, delete the stale Resource after."""

    @pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content")
    def test_changed_etag_replaces_the_resource_without_duplicating_chunks(
        self, db, repository, fake_blob_client, instant_indexing, monkeypatch,
    ):
        fake_blob_client.add_blob(FakeBlob("docs/a.pdf", etag='"etag-1"', content=b"%PDF-1.4 v1"))

        first = trigger_ingestion(repository, db)
        assert first["queued"] == 1
        old_resource = azure_blob_resources(db, repository)[0]
        old_resource_id = old_resource.resource_id

        # The replacement content is already on the same local path before the
        # supersede step runs — this is the exact file the old Resource's
        # delete_file=False guard protects.
        new_content = b"%PDF-1.4 v2"
        fake_blob_client.add_blob(FakeBlob("docs/a.pdf", etag='"etag-2"', content=new_content))

        events = []

        real_create = ResourceService.create_multiple_resources

        def recording_create(*args, **kwargs):
            outcome = real_create(*args, **kwargs)
            events.append(("create", [r.uri for r in outcome[0]]))
            return outcome

        real_delete = ResourceService.delete_resource

        def recording_delete(resource_id, session, **kwargs):
            outcome = real_delete(resource_id, session, **kwargs)
            events.append(("delete", resource_id))
            return outcome

        monkeypatch.setattr(ResourceService, "create_multiple_resources", recording_create)
        monkeypatch.setattr(ResourceService, "delete_resource", recording_delete)

        second = trigger_ingestion(repository, db)

        assert second["queued"] == 1
        assert second["skipped_unchanged"] == 0

        # Exactly one Resource survives for this blob, carrying the new ETag.
        survivors = azure_blob_resources(db, repository)
        assert len(survivors) == 1
        assert survivors[0].resource_id != old_resource_id
        assert survivors[0].extra_metadata["etag"] == '"etag-2"'

        # The stale Resource row is gone from the DB…
        assert db.get(Resource, old_resource_id) is None

        # …and the delete happened only AFTER the new content was created and
        # handed to the indexer (index-then-swap, never delete-before-create).
        assert ("create", [survivors[0].uri]) in events
        assert ("delete", old_resource_id) in events
        assert events.index(("create", [survivors[0].uri])) < events.index(("delete", old_resource_id))

        # The replacement's backing file was NOT destroyed by the supersede
        # (delete_file=False): the new content is still on disk.
        from services.resource_service import REPO_BASE_FOLDER
        file_path = os.path.join(REPO_BASE_FOLDER, str(repository.repository_id), survivors[0].uri)
        assert open(file_path, "rb").read() == new_content

    @pytest.mark.usefixtures("tmp_repo_base", "forbid_index_single_content")
    def test_same_basename_under_different_prefixes_yields_distinguishable_resources(
        self, db, repository, fake_blob_client, instant_indexing,
    ):
        """Edge case (spec.md): keyed by full blob path, not basename — and the
        staged filename is unique per blob (non-injective sanitizer regression)."""
        fake_blob_client.add_blob(FakeBlob("a/report.pdf", etag='"etag-a"'))
        fake_blob_client.add_blob(FakeBlob("b/report.pdf", etag='"etag-b"'))
        fake_blob_client.add_blob(FakeBlob("a__report.pdf", etag='"etag-c"'))

        result = trigger_ingestion(repository, db)

        assert result["queued"] == 3
        resources = azure_blob_resources(db, repository)
        blob_names = {r.extra_metadata["blob_name"] for r in resources}
        assert blob_names == {"a/report.pdf", "b/report.pdf", "a__report.pdf"}
        uris = [r.uri for r in resources]
        assert len(set(uris)) == 3, "each blob must own its own staged file/uri"
