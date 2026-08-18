import os
import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from db.database import SessionLocal
from models.import_job import ImportJob
from models.import_job_row import ImportJobRow
from models.repository import Repository
from models.app import App
from models.enums.import_job_status import ImportJobStatus
from models.enums.import_row_status import ImportRowStatus
from services.crawl.http_fetcher import fetch, FetchResult
from utils.config import get_app_config
from utils.logger import get_logger

logger = get_logger(__name__)

_PDF_MAGIC = b'%PDF-'


def _staging_dir(import_job_id: int) -> str:
    cfg = get_app_config()
    path = os.path.join(cfg['TMP_BASE_FOLDER'], 'csv_import_staging', str(import_job_id))
    os.makedirs(path, exist_ok=True)
    return path


def _max_file_size_mb(db: Session, repository_id: int) -> int:
    repo = db.query(Repository).filter(Repository.repository_id == repository_id).first()
    if not repo:
        return 0
    app = db.query(App).filter(App.app_id == repo.app_id).first()
    return (app.max_file_size_mb or 0) if app else 0


def _classify_result(result: FetchResult, max_size_mb: int) -> tuple[bool, str | None]:
    """Returns (is_success, failure_reason)."""
    if result.status_code == 0:
        return False, 'TIMED_OUT' if result.error and 'Timeout' in result.error else 'NOT_FOUND'
    if result.status_code in (401, 403):
        return False, 'ACCESS_DENIED'
    if result.status_code != 200:
        return False, 'NOT_FOUND'
    content = result.content or b''
    if max_size_mb > 0 and len(content) / (1024 * 1024) > max_size_mb:
        return False, 'FILE_TOO_LARGE'
    if not content.startswith(_PDF_MAGIC):
        return False, 'NOT_A_PDF'
    return True, None


async def _download_one(url: str) -> FetchResult:
    return await fetch(url, timeout=30.0)


def _apply_result_to_row(db: Session, row: ImportJobRow, result: FetchResult, max_size_mb: int,
                          staging_dir: str) -> None:
    """Each row gets its own staged file, even when several rows share a URL —
    confirm/discard on one row must not affect a sibling row's file (confirm
    moves the file away via shutil.move, which would break a shared path)."""
    is_success, failure_reason = _classify_result(result, max_size_mb)
    if not is_success:
        row.status = ImportRowStatus.FAILED
        row.failure_reason = failure_reason
        return

    staged_path = os.path.join(staging_dir, f"{uuid.uuid4().hex}.pdf")
    with open(staged_path, 'wb') as f:
        f.write(result.content)

    row.status = ImportRowStatus.DOWNLOADED
    row.failure_reason = None
    row.staged_path = staged_path


def _maybe_close_to_review(db: Session, job: ImportJob) -> None:
    unsettled = db.query(ImportJobRow).filter(
        ImportJobRow.import_job_id == job.id,
        ImportJobRow.status.in_([ImportRowStatus.PENDING, ImportRowStatus.DOWNLOADING]),
    ).count()
    if unsettled == 0:
        job.status = ImportJobStatus.REVIEW
        db.commit()


def download_job_rows(import_job_id: int) -> None:
    """Download every PENDING row of a job. Same URL is fetched once and its
    result (staged file or failure) is copied to every row sharing that URL."""
    import asyncio

    db = SessionLocal()
    try:
        job = db.query(ImportJob).filter(ImportJob.id == import_job_id).first()
        if not job:
            logger.error(f"ImportJob {import_job_id} not found for download")
            return

        rows = db.query(ImportJobRow).filter(
            ImportJobRow.import_job_id == import_job_id,
            ImportJobRow.status == ImportRowStatus.PENDING,
        ).all()
        if not rows:
            return

        max_size_mb = _max_file_size_mb(db, job.repository_id)
        staging_dir = _staging_dir(import_job_id)

        rows_by_url: dict[str, list[ImportJobRow]] = {}
        for row in rows:
            rows_by_url.setdefault(row.url, []).append(row)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            for url, url_rows in rows_by_url.items():
                for row in url_rows:
                    row.status = ImportRowStatus.DOWNLOADING
                db.commit()

                result = loop.run_until_complete(_download_one(url))

                for row in url_rows:
                    _apply_result_to_row(db, row, result, max_size_mb, staging_dir)
                db.commit()
        finally:
            loop.close()

        job.last_activity_at = datetime.utcnow()
        db.commit()
        _maybe_close_to_review(db, job)
    except Exception:
        logger.error(f"Error downloading ImportJob {import_job_id}", exc_info=True)
    finally:
        db.close()


async def download_one_row(row_id: int) -> None:
    """Retry a single row in place. Called directly (awaited) from the retry
    endpoint, which is itself async — a single fetch is fast enough not to
    need a background thread, and awaiting in-place avoids spinning up a
    nested event loop while one is already running in this thread."""
    db = SessionLocal()
    try:
        row = db.query(ImportJobRow).filter(ImportJobRow.id == row_id).first()
        if not row:
            logger.error(f"ImportJobRow {row_id} not found for retry")
            return
        job = db.query(ImportJob).filter(ImportJob.id == row.import_job_id).first()

        max_size_mb = _max_file_size_mb(db, job.repository_id)
        staging_dir = _staging_dir(job.id)

        row.status = ImportRowStatus.DOWNLOADING
        db.commit()

        result = await _download_one(row.url)

        _apply_result_to_row(db, row, result, max_size_mb, staging_dir)
        job.last_activity_at = datetime.utcnow()
        db.commit()
    except Exception:
        logger.error(f"Error retrying ImportJobRow {row_id}", exc_info=True)
    finally:
        db.close()
