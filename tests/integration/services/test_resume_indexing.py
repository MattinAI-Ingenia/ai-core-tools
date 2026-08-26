"""Resuming an ingestion that stopped (backend restart, LLM outage, closed laptop).

Resumption needs no checkpoint of ours: LightRAG skips documents already in
PROCESSED and resets interrupted ones to PENDING, so re-running a half-done
resource picks up where it stopped.
"""

from unittest.mock import patch

import pytest

from models.repository import Repository
from models.resource import Resource
from services import silo_indexing_lock
from services.resource_service import ResourceService


@pytest.fixture
def repo(db, fake_app, fake_silo):
    repository = Repository(
        name="resume-repo", app_id=fake_app.app_id, silo_id=fake_silo.silo_id,
    )
    db.add(repository)
    db.flush()
    return repository


def _resource(db, repo, status, uri="f.pdf"):
    r = Resource(
        name=uri, uri=uri, type=".pdf", status=status, repository_id=repo.repository_id,
    )
    db.add(r)
    db.flush()
    return r


@pytest.fixture
def started():
    """Capture what would have been indexed instead of spawning a thread."""
    with patch.object(
        ResourceService, "_index_resources_background", return_value="session-1"
    ) as spawn:
        yield spawn


def test_nothing_to_resume(db, repo, started):
    _resource(db, repo, 'ready')

    assert ResourceService.resume_indexing(db, repo.repository_id) == (None, 0)
    started.assert_not_called()


def test_resumes_unfinished_resources(db, repo, started):
    _resource(db, repo, 'ready', 'done.pdf')
    _resource(db, repo, 'pending', 'never-started.pdf')
    _resource(db, repo, 'error', 'interrupted.pdf')

    session_id, resumed = ResourceService.resume_indexing(db, repo.repository_id)

    assert (session_id, resumed) == ("session-1", 2)
    queued = {r.uri for r in started.call_args.args[0]}
    assert queued == {'never-started.pdf', 'interrupted.pdf'}
    silo_indexing_lock.release(started.call_args.kwargs['lock_conn'], repo.silo_id)


def test_refuses_while_a_run_is_alive(db, repo, started):
    _resource(db, repo, 'pending')
    held = silo_indexing_lock.acquire(repo.silo_id)
    try:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            ResourceService.resume_indexing(db, repo.repository_id)
        assert exc.value.status_code == 409
        started.assert_not_called()
    finally:
        silo_indexing_lock.release(held, repo.silo_id)


def test_liveness_is_the_lock_not_the_row_statuses(db, repo):
    """A killed run leaves rows unfinished; that must not read as "indexing"."""
    _resource(db, repo, 'indexing')
    _resource(db, repo, 'pending')

    dead = ResourceService.get_ingestion_liveness(db, repo.repository_id)
    assert dead == {"is_indexing": False, "resumable": 2}

    held = silo_indexing_lock.acquire(repo.silo_id)
    try:
        alive = ResourceService.get_ingestion_liveness(db, repo.repository_id)
        # Alive: nothing to resume, the run will get to them.
        assert alive == {"is_indexing": True, "resumable": 0}
    finally:
        silo_indexing_lock.release(held, repo.silo_id)


def test_liveness_counts_cancelled_rows_with_leftover_progress(db, repo):
    """A 'cancelled' row that still exists always carries real progress —
    _discard_unindexed_after_cancel deletes the zero-progress ones the moment
    cancel lands. Whatever is left is a file cancel meant to finish (it had
    already started) but didn't: a race in the pipeline, or the process dying
    mid-run. It must nudge the same as 'paused', not disappear from the count."""
    _resource(db, repo, 'cancelled', 'started-but-incomplete.pdf')

    live = ResourceService.get_ingestion_liveness(db, repo.repository_id)

    assert live == {"is_indexing": False, "resumable": 1}


def test_resume_picks_up_cancelled_resources_too(db, repo, started):
    _resource(db, repo, 'ready', 'done.pdf')
    _resource(db, repo, 'cancelled', 'started-but-incomplete.pdf')

    session_id, resumed = ResourceService.resume_indexing(db, repo.repository_id)

    assert (session_id, resumed) == ("session-1", 1)
    queued = {r.uri for r in started.call_args.args[0]}
    assert queued == {'started-but-incomplete.pdf'}
    silo_indexing_lock.release(started.call_args.kwargs['lock_conn'], repo.silo_id)


class TestLivenessSweepsStaleZeroProgressCancels:
    """A dead run leaves any zero-progress 'cancelled' row exactly where the
    live thread would have removed it, had it lived long enough to reach
    _discard_unindexed_after_cancel — reachability the thread does not get
    when a crash or a container restart kills it first. get_ingestion_liveness
    is the read path every poll and page load already goes through, so it
    doubles as the self-healing sweep: no live thread needed to finish the job."""

    def test_sweeps_when_no_run_is_alive(self, db, repo):
        with patch.object(ResourceService, "_discard_unindexed_after_cancel") as discard:
            ResourceService.get_ingestion_liveness(db, repo.repository_id)
        discard.assert_called_once_with(repo.repository_id)

    def test_does_not_sweep_while_a_run_is_alive(self, db, repo):
        held = silo_indexing_lock.acquire(repo.silo_id)
        try:
            with patch.object(ResourceService, "_discard_unindexed_after_cancel") as discard:
                ResourceService.get_ingestion_liveness(db, repo.repository_id)
            discard.assert_not_called()
        finally:
            silo_indexing_lock.release(held, repo.silo_id)
