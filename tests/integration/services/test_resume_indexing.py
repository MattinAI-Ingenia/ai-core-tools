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
