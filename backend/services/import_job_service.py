import threading
from typing import BinaryIO

from sqlalchemy.orm import Session

from models.import_job import ImportJob
from models.import_job_row import ImportJobRow
from models.enums.import_job_status import ImportJobStatus
from models.enums.import_row_status import ImportRowStatus
from services.csv_import_parser import read_csv_headers, parse_and_dedupe_csv
from services.import_job_download import download_job_rows
from utils.logger import get_logger

logger = get_logger(__name__)


class ConflictError(Exception):
    def __init__(self, job_id: int):
        self.job_id = job_id
        super().__init__(f"An import job is already active: {job_id}")


class ImportJobService:

    @staticmethod
    def get_active_job(repository_id: int, db: Session) -> ImportJob | None:
        return db.query(ImportJob).filter(
            ImportJob.repository_id == repository_id,
            ImportJob.status.in_([ImportJobStatus.DOWNLOADING, ImportJobStatus.REVIEW]),
        ).first()

    @staticmethod
    def preview_headers(file: BinaryIO) -> list[str]:
        return read_csv_headers(file)

    @staticmethod
    def create_job(repository_id: int, file: BinaryIO, link_column: str, db: Session,
                    source_filename: str | None = None) -> ImportJob:
        existing = ImportJobService.get_active_job(repository_id, db)
        if existing:
            raise ConflictError(existing.id)

        rows, no_link_count = parse_and_dedupe_csv(file, link_column)
        if no_link_count:
            logger.info(f"CSV import for repository {repository_id}: {no_link_count} row(s) skipped (no link)")

        job = ImportJob(repository_id=repository_id, source_filename=source_filename, link_column=link_column)
        db.add(job)
        db.commit()

        for parsed_row in rows:
            db.add(ImportJobRow(import_job_id=job.id, url=parsed_row.url, row_metadata=parsed_row.row_metadata))
        db.commit()
        db.refresh(job)

        if rows:
            threading.Thread(target=download_job_rows, args=(job.id,), daemon=True).start()
        else:
            job.status = ImportJobStatus.REVIEW
            db.commit()

        return job

    @staticmethod
    def get_job_with_counts(import_job_id: int, db: Session) -> dict:
        job = db.query(ImportJob).filter(ImportJob.id == import_job_id).first()
        if not job:
            return None
        rows = db.query(ImportJobRow).filter(ImportJobRow.import_job_id == import_job_id).all()
        counts = {status.value.lower(): 0 for status in ImportRowStatus}
        for row in rows:
            counts[row.status.value.lower()] += 1
        counts['total'] = len(rows)
        return {'job': job, 'rows': rows, 'counts': counts}
