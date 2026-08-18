"""Verifies the csvimp001 migration's tables exist and Resource.extra_metadata round-trips."""
from sqlalchemy import inspect

from models.resource import Resource
from models.import_job import ImportJob
from models.import_job_row import ImportJobRow


def test_import_job_tables_exist(db):
    inspector = inspect(db.get_bind())
    tables = inspector.get_table_names()
    assert 'import_job' in tables
    assert 'import_job_row' in tables


def test_resource_has_extra_metadata_column(db, repository):
    resource = Resource(name='test', uri='test.pdf', type='.pdf', status='pending',
                         repository_id=repository.repository_id,
                         extra_metadata={'category': 'finance'})
    db.add(resource)
    db.flush()
    db.refresh(resource)
    assert resource.extra_metadata == {'category': 'finance'}
