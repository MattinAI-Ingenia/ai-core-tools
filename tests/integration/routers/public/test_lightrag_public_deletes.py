"""
Mocked integration tests for the public ``DELETE .../docs/delete/all`` endpoint on LightRAG silos (FR-11).

Auth, routing and DB are real; only the vector store is mocked.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from services.silo_service import (
    LIGHTRAG_SILO_BUSY_DETAIL,
    LIGHTRAG_UNSUPPORTED_DELETE_DETAIL,
    LIGHTRAG_WIPE_FAILED_DETAIL,
)
from utils.service_lifecycle import ServiceRetiredError

GET_STORE = "services.silo_service._get_vector_store"


def delete_all_url(app_id: int, silo_id: int) -> str:
    return f"/public/v1/app/{app_id}/silos/{silo_id}/docs/delete/all"


def _make_silo(db, fake_app, vector_db_type: str):
    from models.silo import Silo

    silo = Silo(
        name=f"{vector_db_type} Silo",
        description="d",
        app_id=fake_app.app_id,
        silo_type="CUSTOM",
        status="active",
        vector_db_type=vector_db_type,
    )
    db.add(silo)
    db.flush()
    return silo


@pytest.fixture
def lightrag_silo(db, fake_app):
    return _make_silo(db, fake_app, "LIGHTRAG")


def _delete(client, fake_app, silo, fake_api_key):
    return client.delete(
        delete_all_url(fake_app.app_id, silo.silo_id), headers={"X-API-KEY": fake_api_key.key}
    )


def _assert_lock_free(silo_id: int) -> None:
    from services import silo_indexing_lock

    c = silo_indexing_lock.acquire(silo_id)
    assert c is not None
    silo_indexing_lock.release(c, silo_id)


class TestPublicDeleteAllLightRAG:
    def test_wipe_failure_returns_500(self, client, fake_app, lightrag_silo, fake_api_key):
        store = MagicMock()
        store.wipe_workspace.side_effect = RuntimeError("boom")
        with patch(GET_STORE, return_value=store):
            resp = _delete(client, fake_app, lightrag_silo, fake_api_key)
        assert resp.status_code == 500
        assert resp.json()["detail"] == LIGHTRAG_WIPE_FAILED_DETAIL
        assert "successfully" not in resp.text
        _assert_lock_free(lightrag_silo.silo_id)

    def test_rag_cannot_be_built_returns_500(self, client, fake_app, lightrag_silo, fake_api_key):
        with patch(GET_STORE, side_effect=RuntimeError("no rag")):
            resp = _delete(client, fake_app, lightrag_silo, fake_api_key)
        assert resp.status_code == 500
        assert resp.json()["detail"] == LIGHTRAG_WIPE_FAILED_DETAIL
        _assert_lock_free(lightrag_silo.silo_id)

    def test_busy_returns_409_without_wipe(self, client, fake_app, lightrag_silo, fake_api_key):
        from services import silo_indexing_lock

        store = MagicMock()
        conn = silo_indexing_lock.acquire(lightrag_silo.silo_id)
        assert conn is not None
        try:
            with patch(GET_STORE, return_value=store):
                resp = _delete(client, fake_app, lightrag_silo, fake_api_key)
        finally:
            silo_indexing_lock.release(conn, lightrag_silo.silo_id)
        assert resp.status_code == 409
        assert resp.json()["detail"] == LIGHTRAG_SILO_BUSY_DETAIL
        store.wipe_workspace.assert_not_called()

    def test_retired_service_returns_409(self, client, fake_app, lightrag_silo, fake_api_key):
        err = ServiceRetiredError(
            MagicMock(service_id=1, end_of_service_at=datetime(2020, 1, 1), deprecation_notice=None), "embedding"
        )
        with patch(GET_STORE, side_effect=err):
            resp = _delete(client, fake_app, lightrag_silo, fake_api_key)
        assert resp.status_code == 409
        assert resp.json()["error_code"] == "SERVICE_RETIRED"
        _assert_lock_free(lightrag_silo.silo_id)

    def test_success_wipes_workspace(self, client, fake_app, lightrag_silo, fake_api_key):
        store = MagicMock()
        with patch(GET_STORE, return_value=store):
            resp = _delete(client, fake_app, lightrag_silo, fake_api_key)
        assert resp.status_code == 200
        store.wipe_workspace.assert_called_once_with(f"silo_{lightrag_silo.silo_id}")

    def test_success_marks_repository_files_as_not_indexed(self, client, db, fake_app, lightrag_silo, fake_api_key):
        from models.repository import Repository
        from models.resource import Resource

        repo = Repository(name="Repo", type="default", status="active", app_id=fake_app.app_id,
                          silo_id=lightrag_silo.silo_id, create_date=datetime.now())
        db.add(repo)
        db.flush()
        res = Resource(name="a.txt", type="txt", status="indexed", repository_id=repo.repository_id,
                       progress_done=5, progress_total=5)
        db.add(res)
        db.flush()
        with patch(GET_STORE, return_value=MagicMock()):
            resp = _delete(client, fake_app, lightrag_silo, fake_api_key)
        assert resp.status_code == 200
        db.expire_all()
        res = db.get(Resource, res.resource_id)
        assert (res.status, res.progress_done, res.progress_total) == ("pending", 0, None)


class TestPublicDeleteAllPgvectorUnchanged:
    def test_pgvector_ignores_lock_and_deletes_collection(self, client, db, fake_app, fake_api_key):
        from services import silo_indexing_lock

        silo = _make_silo(db, fake_app, "PGVECTOR")
        store = MagicMock()
        conn = silo_indexing_lock.acquire(silo.silo_id)
        assert conn is not None
        try:
            with patch(GET_STORE, return_value=store), patch(
                "services.silo_service.SiloService.check_silo_collection_exists", return_value=True
            ):
                resp = _delete(client, fake_app, silo, fake_api_key)
        finally:
            silo_indexing_lock.release(conn, silo.silo_id)
        assert resp.status_code == 200
        store.delete_collection.assert_called_once()


# ---------------------------------------------------------------------------
# FR-13: partial deletes are refused (409) on a LightRAG silo, untouched on PGVector
# ---------------------------------------------------------------------------


def _headers(fake_api_key):
    return {"X-API-KEY": fake_api_key.key}


def _media(db, fake_app, silo, tmp_path, with_repo: bool):
    from datetime import datetime

    from models.media import Media
    from models.repository import Repository

    repo = None
    if with_repo:
        repo = Repository(name="Repo", type="default", status="active", app_id=fake_app.app_id,
                          silo_id=silo.silo_id, create_date=datetime.now())
        db.add(repo)
        db.flush()
    path = tmp_path / "clip.mp4"
    path.write_text("x")
    media = Media(name="clip", silo_id=silo.silo_id, repository_id=repo.repository_id if repo else None,
                  source_type="upload", status="completed", file_path=str(path), create_date=datetime.now())
    db.add(media)
    db.flush()
    return repo, media, path


def _media_alive(db, media_id) -> bool:
    from models.media import Media

    db.expire_all()
    return db.get(Media, media_id) is not None


@pytest.mark.parametrize("kind", ["LIGHTRAG", "PGVECTOR"])
class TestPublicPartialDeletes:
    def test_docs_delete_by_ids(self, client, db, fake_app, fake_api_key, kind):
        silo = _make_silo(db, fake_app, kind)
        url = f"/public/v1/app/{fake_app.app_id}/silos/{silo.silo_id}/docs/delete"
        store = MagicMock()
        with patch(GET_STORE, return_value=store) as get_store, patch(
            "services.silo_service.SiloService.check_silo_collection_exists", return_value=True
        ):
            resp = client.request("DELETE", url, json={"ids": ["a"]}, headers=_headers(fake_api_key))
        if kind == "LIGHTRAG":
            assert resp.status_code == 409
            assert resp.json()["detail"] == LIGHTRAG_UNSUPPORTED_DELETE_DETAIL
            get_store.assert_not_called()
        else:
            assert resp.status_code == 200
            store.delete_documents.assert_called_once()

    def test_docs_delete_by_metadata(self, client, db, fake_app, fake_api_key, kind):
        silo = _make_silo(db, fake_app, kind)
        url = f"/public/v1/app/{fake_app.app_id}/silos/{silo.silo_id}/docs/delete-by-metadata"
        store = MagicMock()
        store.count_documents.return_value = 0
        with patch(GET_STORE, return_value=store) as get_store, patch(
            "services.silo_service.SiloService.check_silo_collection_exists", return_value=True
        ):
            resp = client.request(
                "DELETE", url, json={"filter_metadata": {"k": {"$eq": "v"}}}, headers=_headers(fake_api_key)
            )
        if kind == "LIGHTRAG":
            assert resp.status_code == 409
            assert resp.json()["detail"] == LIGHTRAG_UNSUPPORTED_DELETE_DETAIL
            get_store.assert_not_called()
        else:
            assert resp.status_code == 200  # no embedding service -> 0 deleted, same as before

    def test_repository_media_delete(self, client, db, fake_app, fake_api_key, tmp_path, kind):
        repo, media, path = _media(db, fake_app, _make_silo(db, fake_app, kind), tmp_path, with_repo=True)
        media_id = media.media_id
        url = f"/public/v1/app/{fake_app.app_id}/repositories/{repo.repository_id}/media/{media_id}"
        with patch(GET_STORE) as get_store, patch("services.silo_service.SiloService.delete_media") as delete_media:
            resp = client.delete(url, headers=_headers(fake_api_key))
        if kind == "LIGHTRAG":
            assert resp.status_code == 409
            assert resp.json()["detail"] == LIGHTRAG_UNSUPPORTED_DELETE_DETAIL
            assert _media_alive(db, media_id)
            assert path.exists()
            get_store.assert_not_called()
            delete_media.assert_not_called()
        else:
            assert resp.status_code == 200
            assert not _media_alive(db, media_id)
            delete_media.assert_called_once()

    def test_silo_media_delete(self, client, db, fake_app, fake_api_key, tmp_path, kind):
        silo = _make_silo(db, fake_app, kind)
        _, media, path = _media(db, fake_app, silo, tmp_path, with_repo=False)
        media_id = media.media_id
        url = f"/public/v1/app/{fake_app.app_id}/silos/{silo.silo_id}/media/{media_id}"
        with patch(GET_STORE) as get_store, patch("services.silo_service.SiloService.delete_media") as delete_media:
            resp = client.delete(url, headers=_headers(fake_api_key))
        if kind == "LIGHTRAG":
            assert resp.status_code == 409
            assert resp.json()["detail"] == LIGHTRAG_UNSUPPORTED_DELETE_DETAIL
            assert _media_alive(db, media_id)
            assert path.exists()
            get_store.assert_not_called()
            delete_media.assert_not_called()
        else:
            assert resp.status_code == 200
            assert not _media_alive(db, media_id)
            delete_media.assert_called_once()
