import io
import pytest

from services.import_job_service import ImportJobService, ConflictError
from models.enums.import_job_status import ImportJobStatus


def _csv_file(text: str):
    return io.BytesIO(text.encode('utf-8'))


def test_create_job_persists_deduped_rows(db, repository):
    csv_text = "link,title\nhttp://a.com/x.pdf,A\nhttp://a.com/x.pdf,A\nhttp://a.com/y.pdf,B\n"
    job = ImportJobService.create_job(repository.repository_id, _csv_file(csv_text), 'link', db)

    assert job.id is not None
    assert job.status == ImportJobStatus.DOWNLOADING
    assert len(job.rows) == 2


def test_create_job_rejects_unknown_link_column(db, repository):
    with pytest.raises(ValueError):
        ImportJobService.create_job(repository.repository_id, _csv_file("a,b\n1,2\n"), 'nope', db)


def test_second_active_job_conflicts(db, repository):
    ImportJobService.create_job(
        repository.repository_id, _csv_file("link\nhttp://a.com/x.pdf\n"), 'link', db,
    )
    with pytest.raises(ConflictError):
        ImportJobService.create_job(
            repository.repository_id, _csv_file("link\nhttp://a.com/y.pdf\n"), 'link', db,
        )


def test_get_active_job_returns_none_when_no_job(db, repository):
    assert ImportJobService.get_active_job(repository.repository_id, db) is None
