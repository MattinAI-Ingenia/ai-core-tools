"""Retry-row endpoint test.

Uses a real SessionLocal-backed TestClient (not the savepoint `client`/`db`
fixtures) because the retry endpoint calls download_one_row(), which opens
its own SessionLocal() session — invisible to uncommitted savepoint data.
Mirrors the real-commit pattern in tests/integration/test_crawl_executor.py.
"""
import io
import pytest
from unittest.mock import patch, AsyncMock

from fastapi.testclient import TestClient
from db.database import SessionLocal, get_db
from models.import_job import ImportJob
from models.import_job_row import ImportJobRow
from services.crawl.http_fetcher import FetchResult
from services.import_job_service import ImportJobService
from tests.integration.csv_import_helpers import setup_repository_and_app, cleanup_app
from utils.local_auth_tokens import mint_access_token


@pytest.fixture
def real_client(test_engine):
    from main import app

    db = SessionLocal()
    repository, app_id = setup_repository_and_app(db)

    from models.user import User
    user = db.query(User).filter(User.email.like('csv-import-test-%')).order_by(User.user_id.desc()).first()
    token, _ = mint_access_token(user.user_id, user.email, user.name)
    headers = {"Authorization": f"Bearer {token}"}

    def override_get_db():
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client, repository, app_id, headers
    app.dependency_overrides.pop(get_db, None)

    db.close()
    cleanup_app(app_id)


@patch('services.import_job_download.fetch', new_callable=AsyncMock)
def test_retry_endpoint_updates_row(mock_fetch, real_client):
    client, repository, app_id, headers = real_client
    db = SessionLocal()
    try:
        mock_fetch.return_value = FetchResult(status_code=404)
        job = ImportJobService.create_job(
            repository.repository_id,
            io.BytesIO(b"link\nhttp://a.com/x.pdf\n"),
            'link', db,
        )
        row_id = job.rows[0].id
        job_id = job.id
        db.commit()
    finally:
        db.close()

    mock_fetch.return_value = FetchResult(status_code=200, content=b'%PDF-1.4 ok')
    response = client.post(
        f"/internal/apps/{app_id}/repositories/{repository.repository_id}"
        f"/csv-imports/{job_id}/rows/{row_id}/retry",
        headers=headers,
    )

    assert response.status_code == 200
    body = response.json()
    row = next(r for r in body['rows'] if r['id'] == row_id)
    assert row['status'] == 'DOWNLOADED'
