"""Exercises import_job_download's SessionLocal()-based background functions.

Uses real commits (mirrors tests/integration/test_crawl_executor.py) since the
code under test opens its own SessionLocal() session, invisible to the
savepoint-isolated `db` fixture used elsewhere in the suite.
"""
import pytest
from unittest.mock import patch, AsyncMock

from db.database import SessionLocal
from models.import_job import ImportJob
from models.import_job_row import ImportJobRow
from models.enums.import_row_status import ImportRowStatus
from services.crawl.http_fetcher import FetchResult
from services.import_job_download import download_job_rows, download_one_row
from tests.integration.csv_import_helpers import setup_repository_and_app, cleanup_app


@pytest.fixture
def repo_setup(test_engine):
    db = SessionLocal()
    repository, app_id = setup_repository_and_app(db)
    yield db, repository
    db.close()
    cleanup_app(app_id)


def _make_row(db, job, url):
    row = ImportJobRow(import_job_id=job.id, url=url, row_metadata={'title': 'A'})
    db.add(row)
    db.commit()
    return row


def _make_job(db, repository):
    job = ImportJob(repository_id=repository.repository_id, source_filename='x.csv', link_column='link')
    db.add(job)
    db.commit()
    return job


@patch('services.import_job_download.fetch', new_callable=AsyncMock)
def test_successful_pdf_download_marks_row_downloaded(mock_fetch, repo_setup):
    db, repository = repo_setup
    job = _make_job(db, repository)
    row = _make_row(db, job, 'http://a.com/x.pdf')
    mock_fetch.return_value = FetchResult(status_code=200, content=b'%PDF-1.4 fake content')

    download_job_rows(job.id)

    db.refresh(row)
    assert row.status == ImportRowStatus.DOWNLOADED
    assert row.staged_path is not None
    assert row.failure_reason is None


@patch('services.import_job_download.fetch', new_callable=AsyncMock)
def test_non_pdf_content_marks_row_failed(mock_fetch, repo_setup):
    db, repository = repo_setup
    job = _make_job(db, repository)
    row = _make_row(db, job, 'http://a.com/x.pdf')
    mock_fetch.return_value = FetchResult(status_code=200, content=b'<html>not a pdf</html>')

    download_job_rows(job.id)

    db.refresh(row)
    assert row.status == ImportRowStatus.FAILED
    assert row.failure_reason == 'NOT_A_PDF'


@patch('services.import_job_download.fetch', new_callable=AsyncMock)
def test_404_marks_row_not_found(mock_fetch, repo_setup):
    db, repository = repo_setup
    job = _make_job(db, repository)
    row = _make_row(db, job, 'http://a.com/x.pdf')
    mock_fetch.return_value = FetchResult(status_code=404)

    download_job_rows(job.id)

    db.refresh(row)
    assert row.status == ImportRowStatus.FAILED
    assert row.failure_reason == 'NOT_FOUND'


@patch('services.import_job_download.fetch', new_callable=AsyncMock)
def test_shared_url_downloaded_once_across_rows(mock_fetch, repo_setup):
    db, repository = repo_setup
    job = _make_job(db, repository)
    row1 = _make_row(db, job, 'http://a.com/shared.pdf')
    row2 = _make_row(db, job, 'http://a.com/shared.pdf')
    mock_fetch.return_value = FetchResult(status_code=200, content=b'%PDF-1.4 fake content')

    download_job_rows(job.id)

    assert mock_fetch.call_count == 1
    db.refresh(row1)
    db.refresh(row2)
    assert row1.status == ImportRowStatus.DOWNLOADED
    assert row2.status == ImportRowStatus.DOWNLOADED
    assert row1.staged_path != row2.staged_path


@patch('services.import_job_download.fetch', new_callable=AsyncMock)
def test_job_moves_to_review_when_all_rows_settled(mock_fetch, repo_setup):
    db, repository = repo_setup
    job = _make_job(db, repository)
    _make_row(db, job, 'http://a.com/x.pdf')
    mock_fetch.return_value = FetchResult(status_code=200, content=b'%PDF-1.4 fake content')

    download_job_rows(job.id)

    db.refresh(job)
    assert job.status.value == 'REVIEW'


@pytest.mark.asyncio
@patch('services.import_job_download.fetch', new_callable=AsyncMock)
async def test_retry_one_row_only_touches_that_row(mock_fetch, repo_setup):
    db, repository = repo_setup
    job = _make_job(db, repository)
    row1 = _make_row(db, job, 'http://a.com/one.pdf')
    row2 = _make_row(db, job, 'http://a.com/two.pdf')
    row2.status = ImportRowStatus.FAILED
    row2.failure_reason = 'NOT_FOUND'
    db.commit()
    mock_fetch.return_value = FetchResult(status_code=200, content=b'%PDF-1.4 fake content')

    await download_one_row(row2.id)

    db.refresh(row1)
    db.refresh(row2)
    assert row1.status == ImportRowStatus.PENDING
    assert row2.status == ImportRowStatus.DOWNLOADED
