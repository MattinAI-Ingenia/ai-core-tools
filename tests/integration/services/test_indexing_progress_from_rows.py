"""``ResourceService.get_indexing_progress`` reads the bar from the Resource rows.

The in-memory tracker it replaces was invisible to half the requests (uvicorn
runs several workers), which made the bar appear at random and report
"complete" mid-run.  These tests pin the row-derived behaviour instead.
"""

from datetime import datetime, timedelta

import pytest

from models.repository import Repository
from models.resource import Resource
from services.resource_service import ResourceService


@pytest.fixture
def repo(db, fake_app, fake_silo):
    repository = Repository(
        name="progress-repo",
        app_id=fake_app.app_id,
        silo_id=fake_silo.silo_id,
    )
    db.add(repository)
    db.flush()
    return repository


def _add(db, repo, *, status, done, total, started_at):
    resource = Resource(
        name="f.pdf",
        uri="f.pdf",
        type=".pdf",
        status=status,
        repository_id=repo.repository_id,
        progress_done=done,
        progress_total=total,
        progress_started_at=started_at,
    )
    db.add(resource)
    db.flush()
    return resource


def test_returns_none_when_nothing_is_indexing(db, repo):
    _add(db, repo, status='ready', done=8, total=8, started_at=datetime.now())
    assert ResourceService.get_indexing_progress(db, repo.repository_id) is None


def test_finished_files_stay_in_the_batch_totals(db, repo):
    """A completed file must keep contributing, or the percentage jumps back."""
    started = datetime.now() - timedelta(seconds=100)
    _add(db, repo, status='ready', done=8, total=8, started_at=started)
    _add(db, repo, status='indexing', done=2, total=12, started_at=started)

    progress = ResourceService.get_indexing_progress(db, repo.repository_id)

    assert (progress['processed_chunks'], progress['total_chunks']) == (10, 20)
    assert progress['progress_percent'] == pytest.approx(50.0)
    assert progress['current_chunk_name'] == 'f.pdf'
    # 10 units in 100 s → 10 remaining ≈ 100 s more.
    assert progress['estimated_remaining_seconds'] == pytest.approx(100, abs=5)
    assert progress['estimated_total_time_seconds'] == pytest.approx(200, abs=5)


def test_older_batch_is_ignored(db, repo):
    old = datetime.now() - timedelta(hours=2)
    new = datetime.now() - timedelta(seconds=10)
    _add(db, repo, status='ready', done=5, total=5, started_at=old)
    _add(db, repo, status='indexing', done=1, total=4, started_at=new)

    progress = ResourceService.get_indexing_progress(db, repo.repository_id)

    assert progress['total_chunks'] == 4
    assert progress['session_id'] == new.isoformat()


def test_pending_batch_before_counting_reports_zero_not_complete(db, repo):
    """Between upload and chunk counting, progress_total is still NULL.

    It must not read as "finished" — that is what emitted a premature
    ``complete`` and made the bar vanish.
    """
    _add(db, repo, status='pending', done=0, total=None, started_at=datetime.now())

    progress = ResourceService.get_indexing_progress(db, repo.repository_id)

    assert progress is not None
    assert progress['total_chunks'] == 0
    assert progress['progress_percent'] == 0.0
    assert progress['estimated_remaining_seconds'] is None


def test_failed_files_are_counted(db, repo):
    started = datetime.now() - timedelta(seconds=30)
    _add(db, repo, status='error', done=3, total=3, started_at=started)
    _add(db, repo, status='indexing', done=1, total=3, started_at=started)

    progress = ResourceService.get_indexing_progress(db, repo.repository_id)

    assert progress['failed_chunks'] == 1
    assert progress['processed_chunks'] == 4
