"""Shared setup/teardown for CSV-import tests that exercise SessionLocal()-based
background code (import_job_download, import_job_service, file_cleanup_worker).

Mirrors the pattern in tests/integration/test_crawl_executor.py: real commits via
SessionLocal so the code under test (which opens its own SessionLocal() session)
can see the rows, with explicit cascade-delete cleanup at teardown.
"""
from datetime import datetime

from db.database import SessionLocal
from models.app import App
from models.user import User
from models.silo import Silo
from models.repository import Repository
from models.resource import Resource
from models.import_job import ImportJob


def setup_repository_and_app(db, max_file_size_mb: int = 0) -> tuple:
    """Create a User, App, Silo, and Repository with real commits.
    Returns (repository, app_id). Caller is responsible for cleanup via cleanup_app."""
    user = User(
        name="CSV Import Test User",
        email=f"csv-import-test-{datetime.utcnow().timestamp()}@test.com",
        is_active=True,
        platform_role="editor",
    )
    db.add(user)
    db.flush()

    app = App(name="CSV Import Test App", owner_id=user.user_id, max_file_size_mb=max_file_size_mb)
    db.add(app)
    db.flush()

    silo = Silo(name="CSV Import Test Silo", silo_type="REPO", app_id=app.app_id, vector_db_type="PGVECTOR")
    db.add(silo)
    db.flush()

    repository = Repository(name="CSV Import Test Repository", type="REPO", status="active",
                             app_id=app.app_id, silo_id=silo.silo_id)
    db.add(repository)
    db.flush()
    db.commit()  # real commit so SessionLocal() in the code under test can see these rows

    return repository, app.app_id


def cleanup_app(app_id: int) -> None:
    """Delete the test App and its Repository/Silo/Resource/ImportJob children.

    Repository/Silo have no DB-level ON DELETE CASCADE from App, so children
    are deleted explicitly in dependency order before the App itself."""
    db = SessionLocal()
    try:
        repo_ids = [r.repository_id for r in db.query(Repository).filter(Repository.app_id == app_id).all()]
        if repo_ids:
            db.query(ImportJob).filter(ImportJob.repository_id.in_(repo_ids)).delete(synchronize_session=False)
            db.query(Resource).filter(Resource.repository_id.in_(repo_ids)).delete(synchronize_session=False)
            db.query(Repository).filter(Repository.repository_id.in_(repo_ids)).delete(synchronize_session=False)
        db.query(Silo).filter(Silo.app_id == app_id).delete(synchronize_session=False)

        app = db.query(App).filter(App.app_id == app_id).first()
        if app:
            db.delete(app)
        db.commit()
    finally:
        db.close()
