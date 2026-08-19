from datetime import datetime

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from models.import_job import ImportJob
from models.import_job_row import ImportJobRow
from models.enums.import_row_status import ImportRowStatus
from repositories.repository_repository import RepositoryRepository
from services.resource_service import ResourceService


class _StagedFileAdapter:
    """Duck-types the file-like object `_process_single_file` expects
    (`.filename` + `.save(path)`), letting an already-downloaded PDF flow
    through the exact same resource creation path as a manual upload."""

    def __init__(self, staged_path: str, filename: str):
        self.filename = filename
        self._staged_path = staged_path

    def save(self, dest_path: str) -> None:
        import shutil
        shutil.move(self._staged_path, dest_path)


def _pdf_filename(url: str) -> str:
    """Last URL segment as a filename, guaranteed to end in ``.pdf`` exactly once."""
    base = url.rsplit('/', 1)[-1] or 'document'
    return base if base.lower().endswith('.pdf') else f"{base}.pdf"


def maybe_close_job(job: ImportJob, db: Session) -> None:
    remaining = db.query(ImportJobRow).filter(
        ImportJobRow.import_job_id == job.id,
        ImportJobRow.status.notin_([ImportRowStatus.CONFIRMED, ImportRowStatus.DISCARDED]),
    ).count()
    if remaining == 0:
        db.delete(job)
        db.commit()


def estimate_rows(import_job_id: int, row_ids: list[int], db: Session) -> dict:
    from services.silo_service import SiloService

    job = db.query(ImportJob).filter(ImportJob.id == import_job_id).first()
    repo = RepositoryRepository.get_by_id(db, job.repository_id)
    if not repo or not getattr(repo, 'silo_id', None):
        raise ValueError("Repository has no silo configured")

    rows = db.query(ImportJobRow).filter(
        ImportJobRow.import_job_id == import_job_id,
        ImportJobRow.id.in_(row_ids),
        ImportJobRow.status == ImportRowStatus.DOWNLOADED,
    ).all()

    extracted_documents = []
    for row in rows:
        base_metadata = {"repository_id": job.repository_id, "name": row.url, "file_type": ".pdf"}
        docs = SiloService.extract_documents_from_file(row.staged_path, ".pdf", base_metadata, split=False)
        for doc in docs:
            extracted_documents.append({
                "content": getattr(doc, "page_content", ""),
                "metadata": getattr(doc, "metadata", {}),
            })

    return SiloService.estimate_indexing_cost(repo.silo_id, extracted_documents, db)


def confirm_rows(import_job_id: int, row_ids: list[int], db: Session) -> dict:
    job = db.query(ImportJob).filter(ImportJob.id == import_job_id).first()

    # Same guard as manual upload (resource_service.upload_resources_to_repository):
    # LightRAG's asyncio locks are bound to a single event loop and cannot be
    # shared across the independent background threads each batch spawns.
    # The rejection is enforced by the advisory lock ResourceService takes in
    # create_multiple_resources (called below), which is visible to every worker.

    rows = db.query(ImportJobRow).filter(
        ImportJobRow.import_job_id == import_job_id,
        ImportJobRow.id.in_(row_ids),
        ImportJobRow.status == ImportRowStatus.DOWNLOADED,
    ).all()

    files = [_StagedFileAdapter(row.staged_path, _pdf_filename(row.url)) for row in rows]
    extra_metadata = {i: (row.row_metadata or {}) for i, row in enumerate(rows)}

    created_resources, failed_files, session_id = ResourceService.create_multiple_resources(
        files=files, repository_id=job.repository_id, db=db, extra_metadata=extra_metadata,
    )

    for row, resource in zip(rows, created_resources):
        row.status = ImportRowStatus.CONFIRMED
        row.resource_id = resource.resource_id
    db.commit()

    job.last_activity_at = datetime.utcnow()
    db.commit()

    maybe_close_job(job, db)

    return {
        'created_resources': [
            {
                "resource_id": r.resource_id,
                "uri": r.uri,
                "repository_id": r.repository_id,
                "create_date": r.create_date,
                "size": None,
                "content_type": r.type or "unknown",
            } for r in created_resources
        ],
        'failed_files': failed_files,
        'session_id': session_id,
    }
