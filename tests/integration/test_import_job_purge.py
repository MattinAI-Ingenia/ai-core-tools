"""Uses real SessionLocal commits (see test_import_job_download.py docstring) —
_purge_abandoned_import_jobs opens its own SessionLocal() session internally."""
import pytest
from datetime import datetime, timedelta

from db.database import SessionLocal
from services.file_cleanup_worker import _purge_abandoned_import_jobs
from models.import_job import ImportJob
from models.import_job_row import ImportJobRow
from models.enums.import_row_status import ImportRowStatus
from tests.integration.csv_import_helpers import setup_repository_and_app, cleanup_app


@pytest.fixture
def repo_setup(test_engine):
    db = SessionLocal()
    repository, app_id = setup_repository_and_app(db)
    yield db, repository
    db.close()
    cleanup_app(app_id)


def test_purges_job_with_no_activity_for_14_days(tmp_path, repo_setup):
    db, repository = repo_setup
    staged = tmp_path / "staged.pdf"
    staged.write_bytes(b"%PDF-1.4 old")

    job = ImportJob(
        repository_id=repository.repository_id,
        last_activity_at=datetime.utcnow() - timedelta(days=15),
    )
    db.add(job)
    db.commit()
    row = ImportJobRow(import_job_id=job.id, url='http://a.com/x.pdf', status=ImportRowStatus.DOWNLOADED,
                        staged_path=str(staged))
    db.add(row)
    db.commit()
    job_id = job.id

    removed = _purge_abandoned_import_jobs()

    assert removed == 1
    assert not staged.exists()
    assert db.query(ImportJob).filter(ImportJob.id == job_id).first() is None


def test_does_not_purge_job_with_recent_activity(repo_setup):
    db, repository = repo_setup
    job = ImportJob(
        repository_id=repository.repository_id,
        last_activity_at=datetime.utcnow() - timedelta(days=2),
    )
    db.add(job)
    db.commit()

    removed = _purge_abandoned_import_jobs()

    assert removed == 0
    assert db.query(ImportJob).filter(ImportJob.id == job.id).first() is not None
