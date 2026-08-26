"""Resuming must not count last run's failures before this run retries them.

``get_indexing_progress`` counts ``failed_chunks`` by *current* status within
the active batch (see test_indexing_progress_from_rows.py). Resuming
re-stamps every queued resource — including ones left 'error' by the
previous run — into the new batch, but used to leave their status untouched.
The progress bar then reported them as already-failed at 0%, before a
single retry ran. They must read as 'pending' until the retry actually
happens, and 'error' only if it fails again.

This exercises ``ResourceService._index_resources_background``'s stamping
step, which deliberately opens its *own* DB session/connection (so the SSE
endpoint sees the stamp the instant the HTTP response returns, regardless of
whether the caller's transaction ever commits — see the comment in
resource_service.py). The project's shared ``db`` fixture wraps each test in
a connection-level transaction that is rolled back at teardown and never
truly committed (tests/conftest.py's `join_transaction_mode="create_savepoint"`
strategy), so a second, independently-opened session can never see rows the
`db` fixture only flushed. Exercising this cross-session behavior therefore
needs its own really-committed rows — hence the manual setup/teardown here
instead of the shared ``db``/``fake_app``/``fake_silo`` fixtures.
"""

from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from models.app import App
from models.repository import Repository
from models.resource import Resource
from models.silo import Silo
from models.user import User
from services.resource_service import ResourceService


@pytest.fixture
def committed_repo(test_engine):
    """A really-committed User → App → Silo → Repository chain, cleaned up after."""
    session = Session(bind=test_engine)
    user = User(email="resume-test@mattin-test.com", name="Resume Test", is_active=True, platform_role="editor")
    session.add(user)
    session.flush()
    app_obj = App(name="Resume Test App", slug="resume-test-app", owner_id=user.user_id,
                  agent_rate_limit=0, max_file_size_mb=10)
    session.add(app_obj)
    session.flush()
    silo = Silo(name="Resume Test Silo", description="", silo_type="DOMAIN", app_id=app_obj.app_id)
    session.add(silo)
    session.flush()
    repository = Repository(name="resume-repo", app_id=app_obj.app_id, silo_id=silo.silo_id)
    session.add(repository)
    session.commit()

    yield session, repository

    session.query(Resource).filter(Resource.repository_id == repository.repository_id).delete()
    session.query(Repository).filter(Repository.repository_id == repository.repository_id).delete()
    session.query(Silo).filter(Silo.silo_id == silo.silo_id).delete()
    session.query(App).filter(App.app_id == app_obj.app_id).delete()
    session.query(User).filter(User.user_id == user.user_id).delete()
    session.commit()
    session.close()


def _add(session, repo, *, name, status):
    resource = Resource(name=name, uri=name, type=".pdf", status=status, repository_id=repo.repository_id)
    session.add(resource)
    session.commit()
    return resource


@pytest.fixture
def no_op_background_thread():
    """Resuming starts a real daemon thread that calls into LightRAG/the vector
    store — none of which exist in this test. Only the synchronous stamping
    step (before the thread starts) is under test here, so the thread itself
    and the real cross-process lock are replaced with no-ops."""
    with patch("threading.Thread") as thread_cls, \
         patch("services.silo_indexing_lock.acquire", return_value=object()), \
         patch("services.silo_indexing_lock.release"):
        thread_cls.return_value.start.return_value = None
        yield


def test_resume_moves_error_resources_to_pending_not_error(committed_repo, no_op_background_thread):
    session, repo = committed_repo
    failed = _add(session, repo, name="failed.pdf", status="error")
    stuck_pending = _add(session, repo, name="stuck.pdf", status="pending")
    untouched_ready = _add(session, repo, name="ready.pdf", status="ready")

    session_id, resumed = ResourceService.resume_indexing(session, repo.repository_id)

    assert session_id is not None
    assert resumed == 2  # the ready one is not queued

    session.expire_all()  # the stamping step wrote through a different session/connection
    assert session.get(Resource, failed.resource_id).status == "pending"
    assert session.get(Resource, stuck_pending.resource_id).status == "pending"
    assert session.get(Resource, untouched_ready.resource_id).status == "ready"  # never touched


def test_resume_reports_zero_failed_before_any_retry_ran(committed_repo, no_op_background_thread):
    session, repo = committed_repo
    _add(session, repo, name="failed_1.pdf", status="error")
    _add(session, repo, name="failed_2.pdf", status="error")

    ResourceService.resume_indexing(session, repo.repository_id)

    session.expire_all()
    progress = ResourceService.get_indexing_progress(session, repo.repository_id)
    assert progress is not None
    assert progress["failed_chunks"] == 0


def test_resume_processes_resources_in_a_stable_order(committed_repo):
    """Without ORDER BY, which file resuming starts with — and which ones are
    still unfinished the next time the user stops — looked random from one
    resume to the next instead of steadily working through the same queue."""
    session, repo = committed_repo
    # Insert out of name order so a name-based sort would not accidentally pass.
    third = _add(session, repo, name="c.pdf", status="error")
    first = _add(session, repo, name="a.pdf", status="pending")
    second = _add(session, repo, name="b.pdf", status="error")

    captured = {}

    # **kwargs: this stub only cares about the order it is handed, so it must not
    # break every time the real function gains a flag (retry_failed, resumed).
    def fake_background(resources, silo_id=0, lock_conn=None, **kwargs):
        captured["ids"] = [r.resource_id for r in resources]
        return "fake-session-id"

    with patch("services.silo_indexing_lock.acquire", return_value=object()), \
         patch("services.silo_indexing_lock.release"), \
         patch.object(ResourceService, "_index_resources_background", side_effect=fake_background):
        ResourceService.resume_indexing(session, repo.repository_id)

    assert captured["ids"] == [third.resource_id, first.resource_id, second.resource_id]


# ---------------------------------------------------------------------------
# The accumulated ingestion clock (Repository.ingestion_elapsed_seconds)
# ---------------------------------------------------------------------------


def _bank(session, repo, seconds):
    """Pretend an earlier run already spent this long."""
    session.get(Repository, repo.repository_id).ingestion_elapsed_seconds = seconds
    session.commit()


def _clock(session, repo):
    session.expire_all()  # the stamping step writes through its own session
    return session.get(Repository, repo.repository_id).ingestion_elapsed_seconds


def test_a_fresh_upload_starts_the_clock_over(committed_repo, no_op_background_thread):
    session, repo = committed_repo
    _bank(session, repo, 900)
    new_file = _add(session, repo, name="new.pdf", status="pending")

    ResourceService._index_resources_background([new_file], silo_id=0)

    assert _clock(session, repo) == 0


def test_uploading_during_a_pause_keeps_the_banked_time(committed_repo, no_op_background_thread):
    """A paused run holds no silo lock, so an upload is accepted while files sit
    parked. That is not a fresh job — wiping the clock would cost the parked
    files the minutes they had already banked."""
    session, repo = committed_repo
    _bank(session, repo, 900)
    _add(session, repo, name="parked.pdf", status="paused")
    new_file = _add(session, repo, name="new.pdf", status="pending")

    ResourceService._index_resources_background([new_file], silo_id=0)

    assert _clock(session, repo) == 900


def test_a_cancelled_leftover_does_not_hold_the_clock(committed_repo, no_op_background_thread):
    """Cancelled rows never resume, so they must not keep a stale clock alive."""
    session, repo = committed_repo
    _bank(session, repo, 900)
    _add(session, repo, name="dropped.pdf", status="cancelled")
    new_file = _add(session, repo, name="new.pdf", status="pending")

    ResourceService._index_resources_background([new_file], silo_id=0)

    assert _clock(session, repo) == 0


def test_resuming_never_resets_the_clock(committed_repo, no_op_background_thread):
    session, repo = committed_repo
    _bank(session, repo, 900)
    _add(session, repo, name="parked.pdf", status="paused")

    ResourceService.resume_indexing(session, repo.repository_id)

    assert _clock(session, repo) == 900
