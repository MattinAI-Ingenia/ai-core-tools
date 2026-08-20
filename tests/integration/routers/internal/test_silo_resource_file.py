"""``GET /internal/apps/{app_id}/silos/{silo_id}/resources/{resource_id}/file``

Serves a resource by resource_id alone, scoped to its silo — the citation-chunk
"open source PDF" link only knows a silo_id and a resource_id (parsed from the
LightRAG chunk id), not which repository the resource lives in. Must not leak
a resource from a different silo just because the caller can name its id.
"""

from __future__ import annotations

import pytest

from models.repository import Repository
from models.resource import Resource
from models.silo import Silo

pytestmark = pytest.mark.integration


@pytest.fixture
def repo_in(fake_app, db):
    def _make(silo):
        repository = Repository(name="r", app_id=fake_app.app_id, silo_id=silo.silo_id)
        db.add(repository)
        db.flush()
        return repository
    return _make


@pytest.fixture
def resource_in(db):
    def _make(repository, name="f.pdf"):
        resource = Resource(name=name, uri=name, type=".pdf", status="ready", repository_id=repository.repository_id)
        db.add(resource)
        db.flush()
        return resource
    return _make


def test_returns_404_for_a_resource_in_a_different_silo(client, db, fake_app, fake_silo, auth_headers, repo_in, resource_in):
    other_silo = Silo(name="other-silo", app_id=fake_app.app_id, vector_db_type="LIGHTRAG")
    db.add(other_silo)
    db.flush()
    resource = resource_in(repo_in(other_silo))

    resp = client.get(
        f"/internal/apps/{fake_app.app_id}/silos/{fake_silo.silo_id}/resources/{resource.resource_id}/file",
        headers=auth_headers,
    )

    assert resp.status_code == 404


def test_returns_404_for_a_nonexistent_resource(client, fake_app, fake_silo, auth_headers):
    resp = client.get(
        f"/internal/apps/{fake_app.app_id}/silos/{fake_silo.silo_id}/resources/999999/file",
        headers=auth_headers,
    )

    assert resp.status_code == 404


def test_serves_the_file_for_a_resource_in_this_silo(client, db, fake_app, fake_silo, auth_headers, repo_in, resource_in, tmp_path, monkeypatch):
    repository = repo_in(fake_silo)
    resource = resource_in(repository, name="report.pdf")

    repo_dir = tmp_path / str(repository.repository_id)
    repo_dir.mkdir(parents=True)
    (repo_dir / "report.pdf").write_bytes(b"%PDF-1.4 fake content")
    monkeypatch.setattr("services.resource_service.REPO_BASE_FOLDER", str(tmp_path))

    resp = client.get(
        f"/internal/apps/{fake_app.app_id}/silos/{fake_silo.silo_id}/resources/{resource.resource_id}/file",
        headers=auth_headers,
    )

    assert resp.status_code == 200
    assert resp.content == b"%PDF-1.4 fake content"
    assert resp.headers["content-type"] == "application/pdf"
