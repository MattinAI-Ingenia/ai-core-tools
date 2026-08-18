from unittest.mock import patch

import pytest

from models.import_job import ImportJob
from models.import_job_row import ImportJobRow
from models.enums.import_row_status import ImportRowStatus
from models.resource import Resource
from models.ai_service import AIService
from models.embedding_service import EmbeddingService
from services.import_job_confirm import confirm_rows, estimate_rows
from services.resource_service import ResourceService


@pytest.fixture
def lightrag_repository(db, repository):
    """Upgrades the base `repository` fixture's Silo to a fully-configured
    LightRAG silo, required by estimate_indexing_cost."""
    silo = repository.silo
    ai_service = AIService(name="Test Indexing LLM", provider="OpenAI", api_key="sk-test",
                            app_id=repository.app_id)
    embedding_service = EmbeddingService(name="Test Embedding", provider="OpenAI", api_key="sk-test",
                                          app_id=repository.app_id)
    db.add(ai_service)
    db.add(embedding_service)
    db.flush()

    silo.vector_db_type = 'LIGHTRAG'
    silo.indexing_service_id = ai_service.service_id
    silo.embedding_service_id = embedding_service.service_id
    db.commit()
    db.refresh(repository)
    return repository


def _make_downloaded_row(db, job, tmp_path, url, row_metadata=None, name="x", real_pdf=False):
    staged = tmp_path / f"{name}.pdf"
    if real_pdf:
        import pymupdf
        doc = pymupdf.open()
        doc.new_page()
        doc.save(str(staged))
        doc.close()
    else:
        staged.write_bytes(b"%PDF-1.4 fake")
    row = ImportJobRow(import_job_id=job.id, url=url, row_metadata=row_metadata or {},
                        status=ImportRowStatus.DOWNLOADED, staged_path=str(staged))
    db.add(row)
    db.commit()
    return row


def test_confirm_creates_resource_with_metadata_and_closes_job(tmp_path, db, repository):
    job = ImportJob(repository_id=repository.repository_id)
    db.add(job)
    db.commit()
    row = _make_downloaded_row(db, job, tmp_path, 'http://a.com/x.pdf', {'title': 'Finance Report'})

    with patch.object(ResourceService, '_index_resources_background', return_value='sid'):
        result = confirm_rows(job.id, [row.id], db)

    assert len(result['created_resources']) == 1
    resource = db.query(Resource).filter(
        Resource.resource_id == result['created_resources'][0]['resource_id'],
    ).first()
    assert resource.extra_metadata == {'title': 'Finance Report'}

    assert db.query(ImportJob).filter(ImportJob.id == job.id).first() is None


def test_confirming_some_rows_keeps_job_open(tmp_path, db, repository):
    job = ImportJob(repository_id=repository.repository_id)
    db.add(job)
    db.commit()
    row1 = _make_downloaded_row(db, job, tmp_path, 'http://a.com/x.pdf', name="x")
    _make_downloaded_row(db, job, tmp_path, 'http://a.com/y.pdf', name="y")

    with patch.object(ResourceService, '_index_resources_background', return_value='sid'):
        confirm_rows(job.id, [row1.id], db)

    assert db.query(ImportJob).filter(ImportJob.id == job.id).first() is not None


def test_estimate_rows_returns_cost_estimation_shape(tmp_path, db, lightrag_repository):
    repository = lightrag_repository
    job = ImportJob(repository_id=repository.repository_id)
    db.add(job)
    db.commit()
    row = _make_downloaded_row(db, job, tmp_path, 'http://a.com/x.pdf', real_pdf=True)

    result = estimate_rows(job.id, [row.id], db)

    assert 'total_chunks' in result
