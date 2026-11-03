"""Integration tests: DELETE /internal/apps/{app}/repositories/{repo}/resources/{resource}
for a LightRAG silo (graph first under silo_indexing_lock, then file + row). The vector
store is mocked at services.silo_service._get_vector_store."""

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from models.embedding_service import EmbeddingService
from models.repository import Repository
from models.resource import Resource
from models.silo import Silo
from repositories.resource_repository import ResourceRepository
from services import silo_indexing_lock
from services.resource_service import (
    LIGHTRAG_DELETE_FAILED_DETAIL,
    LIGHTRAG_DELETE_PARTIAL_DETAIL,
    LIGHTRAG_SILO_BUSY_DETAIL,
)


def _make_silo(db, app, vector_db_type):
    emb = EmbeddingService(
        name=f"{vector_db_type} Emb", provider="OpenAI", api_key="sk-test-fake",  # pragma: allowlist secret
        app_id=app.app_id,
    )
    db.add(emb)
    db.flush()
    silo = Silo(
        name=f"{vector_db_type} Silo", description="d", silo_type="REPO", status="active",
        app_id=app.app_id, vector_db_type=vector_db_type, embedding_service_id=emb.service_id,
    )
    db.add(silo)
    db.flush()
    return silo


def _make_repo_and_resource(db, app, silo, tmp_path):
    repo = Repository(
        name="Repo", type="default", status="active", app_id=app.app_id,
        silo_id=silo.silo_id, create_date=datetime.now(),
    )
    db.add(repo)
    db.flush()
    res = Resource(
        name="doc.txt", uri="doc.txt", type=".txt", status="active",
        repository_id=repo.repository_id, create_date=datetime.now(),
    )
    db.add(res)
    db.flush()
    folder = tmp_path / str(repo.repository_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "doc.txt"
    path.write_text("hello")
    return repo, res, str(path)


@pytest.fixture(autouse=True)
def repo_base(tmp_path):
    with patch("services.resource_service.REPO_BASE_FOLDER", str(tmp_path)):
        yield tmp_path


@pytest.fixture
def lr(db, fake_app, repo_base):
    silo = _make_silo(db, fake_app, "LIGHTRAG")
    repo, res, path = _make_repo_and_resource(db, fake_app, silo, repo_base)
    return silo, repo, res, path


def _url(app_id, repo_id, resource_id=None):
    base = f"/internal/apps/{app_id}/repositories/{repo_id}/resources"
    return base if resource_id is None else f"{base}/{resource_id}"


def _delete(client, fake_app, repo, res, headers):
    return client.delete(_url(fake_app.app_id, repo.repository_id, res.resource_id), headers=headers)


def _row_exists(db, resource_id):
    db.expire_all()
    return db.get(Resource, resource_id) is not None


def _store(outcome=None, side_effect=None):
    store = MagicMock()
    store.delete_resource_document.return_value = outcome
    if side_effect is not None:
        store.delete_resource_document.side_effect = side_effect
    return store


def _lock_free(silo_id):
    conn = silo_indexing_lock.acquire(silo_id)
    assert conn is not None
    silo_indexing_lock.release(conn, silo_id)


class TestLightRagSingleFileDelete:
    def test_success_deletes_file_and_row(self, client, db, fake_app, owner_headers, lr):
        silo, repo, res, path = lr
        rid = res.resource_id
        store = _store("success")
        with patch("services.silo_service._get_vector_store", return_value=store):
            resp = _delete(client, fake_app, repo, res, owner_headers)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"message": "Resource deleted successfully"}
        store.delete_resource_document.assert_called_once_with(f"silo_{silo.silo_id}", rid)
        assert not _row_exists(db, rid)
        assert not os.path.exists(path)

    def test_error_detail_is_not_the_old_unsupported_message(self, client, fake_app, owner_headers, lr):
        _, repo, res, _ = lr
        with patch("services.silo_service._get_vector_store", return_value=_store("fail")):
            resp = _delete(client, fake_app, repo, res, owner_headers)
        assert resp.status_code == 500
        assert resp.json()["detail"] == LIGHTRAG_DELETE_FAILED_DETAIL
        assert "do not support deleting individual files" not in resp.text

    def test_delete_all_still_409(self, client, fake_app, owner_headers, lr):
        _, repo, _, _ = lr
        resp = client.delete(_url(fake_app.app_id, repo.repository_id), headers=owner_headers)
        assert resp.status_code == 409
        assert "do not support deleting individual files" in resp.json()["detail"]

    def test_not_found_is_success(self, client, db, fake_app, owner_headers, lr):
        _, repo, res, path = lr
        rid = res.resource_id
        with patch("services.silo_service._get_vector_store", return_value=_store("not_found")):
            resp = _delete(client, fake_app, repo, res, owner_headers)
        assert resp.status_code == 200, resp.text
        assert not _row_exists(db, rid)
        assert not os.path.exists(path)

    def test_fail_keeps_file_and_row_then_retry_succeeds(self, client, db, fake_app, owner_headers, lr):
        _, repo, res, path = lr
        rid = res.resource_id
        with patch("services.silo_service._get_vector_store", return_value=_store("fail")):
            resp = _delete(client, fake_app, repo, res, owner_headers)
        assert resp.status_code == 500
        assert resp.json()["detail"] == LIGHTRAG_DELETE_FAILED_DETAIL
        assert _row_exists(db, rid) and os.path.exists(path)

        boom = _store(side_effect=RuntimeError("boom /secret/path"))
        with patch("services.silo_service._get_vector_store", return_value=boom):
            resp = _delete(client, fake_app, repo, res, owner_headers)
        assert resp.status_code == 500
        assert resp.json()["detail"] == LIGHTRAG_DELETE_FAILED_DETAIL
        assert "boom" not in resp.text and "/secret/path" not in resp.text
        assert _row_exists(db, rid) and os.path.exists(path)

        with patch("services.silo_service._get_vector_store", return_value=_store("success")):
            resp = _delete(client, fake_app, repo, res, owner_headers)
        assert resp.status_code == 200
        assert not _row_exists(db, rid) and not os.path.exists(path)

    def test_not_allowed_is_409(self, client, db, fake_app, owner_headers, lr):
        _, repo, res, path = lr
        with patch("services.silo_service._get_vector_store", return_value=_store("not_allowed")):
            resp = _delete(client, fake_app, repo, res, owner_headers)
        assert resp.status_code == 409
        assert resp.json()["detail"] == LIGHTRAG_SILO_BUSY_DETAIL
        assert _row_exists(db, res.resource_id) and os.path.exists(path)

    def test_graph_ok_but_file_removal_fails_is_partial_500(self, client, fake_app, owner_headers, lr):
        # delete_resource is stubbed: its real failure path rolls back the session, which would
        # also discard the uncommitted fixture rows and turn this into the 404 case.
        _, repo, res, _ = lr
        with patch("services.silo_service._get_vector_store", return_value=_store("success")), \
                patch("services.resource_service.ResourceService.delete_resource", return_value=False):
            resp = _delete(client, fake_app, repo, res, owner_headers)
        assert resp.status_code == 500
        assert resp.json()["detail"] == LIGHTRAG_DELETE_PARTIAL_DETAIL

    def test_row_gone_after_graph_step_is_404(self, client, fake_app, owner_headers, lr):
        _, repo, res, _ = lr
        real = ResourceRepository.get_by_id
        calls = []

        def real_first_then_gone(db, resource_id):
            calls.append(resource_id)
            return real(db, resource_id) if len(calls) == 1 else None

        with patch("services.silo_service._get_vector_store", return_value=_store("success")), \
                patch("services.resource_service.ResourceService.delete_resource", return_value=False), \
                patch("services.resource_service.ResourceRepository.get_by_id", side_effect=real_first_then_gone):
            resp = _delete(client, fake_app, repo, res, owner_headers)
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Resource not found or could not be deleted"

    def test_lock_held_returns_409_without_calling_store(self, client, db, fake_app, owner_headers, lr):
        silo, repo, res, path = lr
        store = _store("success")
        conn = silo_indexing_lock.acquire(silo.silo_id)
        assert conn is not None
        try:
            with patch("services.silo_service._get_vector_store", return_value=store):
                resp = _delete(client, fake_app, repo, res, owner_headers)
        finally:
            silo_indexing_lock.release(conn, silo.silo_id)
        assert resp.status_code == 409
        assert resp.json()["detail"] == LIGHTRAG_SILO_BUSY_DETAIL
        store.delete_resource_document.assert_not_called()
        assert _row_exists(db, res.resource_id) and os.path.exists(path)

    @pytest.mark.parametrize("kind", ["success", "not_allowed", "fail", "exception"])
    def test_lock_always_released(self, client, fake_app, owner_headers, lr, kind):
        silo, repo, res, _ = lr
        store = _store(side_effect=RuntimeError("x")) if kind == "exception" else _store(kind)
        with patch("services.silo_service._get_vector_store", return_value=store):
            resp = _delete(client, fake_app, repo, res, owner_headers)
        assert resp.status_code == {"success": 200, "not_allowed": 409, "fail": 500, "exception": 500}[kind]
        _lock_free(silo.silo_id)

    def test_lock_held_during_graph_step(self, client, fake_app, owner_headers, lr):
        silo, repo, res, _ = lr
        seen = []

        def during(collection, resource_id):
            c = silo_indexing_lock.acquire(silo.silo_id)
            seen.append(c)
            if c is not None:  # only if the production lock were broken: don't leak it
                silo_indexing_lock.release(c, silo.silo_id)
            return "success"

        store = MagicMock()
        store.delete_resource_document.side_effect = during
        with patch("services.silo_service._get_vector_store", return_value=store):
            resp = _delete(client, fake_app, repo, res, owner_headers)
        assert resp.status_code == 200, resp.text
        assert seen == [None]

    def test_retired_service_maps_to_409_and_releases_lock(self, client, db, fake_app, owner_headers, lr):
        from utils.service_lifecycle import ServiceRetiredError

        silo, repo, res, path = lr
        err = ServiceRetiredError(
            MagicMock(service_id=1, end_of_service_at=datetime(2020, 1, 1), deprecation_notice=None),
            "embedding",
        )
        with patch("services.silo_service._get_vector_store", side_effect=err):
            resp = _delete(client, fake_app, repo, res, owner_headers)
        assert resp.status_code == 409
        assert resp.json()["error_code"] == "SERVICE_RETIRED"
        assert _row_exists(db, res.resource_id) and os.path.exists(path)
        _lock_free(silo.silo_id)

    def test_pgvector_silo_unaffected(self, client, db, fake_app, owner_headers, repo_base):
        silo = _make_silo(db, fake_app, "PGVECTOR")
        repo, res, path = _make_repo_and_resource(db, fake_app, silo, repo_base)
        rid = res.resource_id
        conn = silo_indexing_lock.acquire(silo.silo_id)
        assert conn is not None
        try:
            with patch("services.resource_service.SiloService.delete_resource") as tolerant, \
                    patch("services.silo_service._get_vector_store") as gvs:
                resp = _delete(client, fake_app, repo, res, owner_headers)
        finally:
            silo_indexing_lock.release(conn, silo.silo_id)
        assert resp.status_code == 200, resp.text
        tolerant.assert_called_once()
        gvs.assert_not_called()
        assert not _row_exists(db, rid)
