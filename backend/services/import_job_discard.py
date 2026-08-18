import os
from datetime import datetime

from sqlalchemy.orm import Session

from models.import_job import ImportJob
from models.import_job_row import ImportJobRow
from models.enums.import_row_status import ImportRowStatus
from services.import_job_confirm import maybe_close_job
from utils.logger import get_logger

logger = get_logger(__name__)


def discard_rows(import_job_id: int, row_ids: list[int], db: Session) -> None:
    job = db.query(ImportJob).filter(ImportJob.id == import_job_id).first()
    rows = db.query(ImportJobRow).filter(
        ImportJobRow.import_job_id == import_job_id,
        ImportJobRow.id.in_(row_ids),
    ).all()

    for row in rows:
        if row.staged_path and os.path.exists(row.staged_path):
            try:
                os.remove(row.staged_path)
            except OSError:
                logger.warning(f"Could not remove staged file {row.staged_path}", exc_info=True)
        row.status = ImportRowStatus.DISCARDED
    db.commit()

    job.last_activity_at = datetime.utcnow()
    db.commit()

    maybe_close_job(job, db)
