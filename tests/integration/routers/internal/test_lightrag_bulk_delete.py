"""Delete-all of a LightRAG repository and the folder-delete refusal (mocked store, no real LightRAG)."""

import os
from datetime import datetime
from unittest.mock import patch

import pytest

from models.folder import Folder
from models.repository import Repository
from models.resource import Resource
from tests.integration.routers.internal.test_lightrag_resource_delete import _make_silo, _row_exists


@pytest.fixture(autouse=True)
def repo_base(tmp_path):
    with patch("services.resource_service.REPO_BASE_FOLDER", str(tmp_path)):
        yield tmp_path


def _repo(db, app, silo):
    repo = Repository(
        name="Repo", type="default", status="active", app_id=app.app_id,
        silo_id=silo.silo_id, create_date=datetime.now(),
    )
    db.add(repo)
    db.flush()
    return repo


def _resource_in(db, repo, folder, base, uri):
    res = Resource(
        name=uri, uri=uri, type=".txt", status="active", repository_id=repo.repository_id,
        folder_id=folder.folder_id, create_date=datetime.now(),
    )
    db.add(res)
    db.flush()
    directory = base / str(repo.repository_id) / folder.name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / uri
    path.write_text("hello")
    return res, str(path)


class TestNonLightRagFolderDelete:
    def test_non_lightrag_folder_delete_characterization(self, client, db, fake_app, owner_headers, repo_base):
        """AC-26: tolerant folder delete: a failing resource does not stop the others."""
        silo = _make_silo(db, fake_app, "PGVECTOR")
        repo = _repo(db, fake_app, silo)
        folder = Folder(name="F", repository_id=repo.repository_id)
        db.add(folder)
        db.flush()
        made = [_resource_in(db, repo, folder, repo_base, f"r{i}.txt") for i in range(3)]
        rids = [r.resource_id for r, _ in made]
        folder_id = folder.folder_id
        folder_dir = repo_base / str(repo.repository_id) / "F"
        # delete_resource's failure path rolls the session back: persist the fixture rows first.
        db.commit()

        calls = []

        def flaky(resource):
            calls.append(resource.resource_id)
            if len(calls) == 2:
                raise RuntimeError("boom")

        with patch("services.resource_service.SiloService.delete_resource", side_effect=flaky):
            resp = client.delete(
                f"/internal/apps/{fake_app.app_id}/repositories/{repo.repository_id}/folders/{folder_id}",
                headers=owner_headers,
            )
        assert resp.status_code == 200, resp.text
        assert resp.json()["message"] == (
            f"Folder {folder_id} and all its contents have been deleted successfully"
        )
        assert len(calls) == 3
        db.expire_all()
        assert db.get(Folder, folder_id) is None
        for rid in rids:  # the failing one goes with the folder (cascade)
            assert not _row_exists(db, rid)
        assert not folder_dir.exists()


# --- step_008: delete all files of a LightRAG repository (FR-3) ---------------------------------

import logging  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

from fastapi import HTTPException  # noqa: E402

from models.media import Media  # noqa: E402
from services import silo_indexing_lock  # noqa: E402
from services.resource_service import (  # noqa: E402
    LIGHTRAG_SILO_BUSY_DETAIL,
    ResourceService,
)
from tests.integration.routers.internal.test_lightrag_resource_delete import _lock_free  # noqa: E402


def _all_url(app, repo):
    return f"/internal/apps/{app.app_id}/repositories/{repo.repository_id}/resources"


def _populate(db, repo, base, count=3):
    """Create resources with files on disk: [(id, path)]."""
    out = []
    for i in range(count):
        res = Resource(
            name=f"r{i}.txt", uri=f"r{i}.txt", type=".txt", status="active",
            repository_id=repo.repository_id, create_date=datetime.now(),
        )
        db.add(res)
        db.flush()
        directory = base / str(repo.repository_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"r{i}.txt").write_text("x")
        out.append((res, str(directory / f"r{i}.txt")))
    return [(r.resource_id, p) for r, p in out]


def _media(db, repo, base):
    path = base / "media" / "m1.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"v")
    media = Media(
        silo_id=repo.silo_id, repository_id=repo.repository_id, name="m1", source_type="upload",
        file_path=str(path),
    )
    db.add(media)
    db.flush()
    return media.media_id, str(path)


class TestLightRagDeleteAll:
    @pytest.fixture
    def lr(self, db, fake_app, repo_base):
        silo = _make_silo(db, fake_app, "LIGHTRAG")
        return silo, _repo(db, fake_app, silo)

    def test_ac10_ac16_wipes_then_deletes_rows_files_and_media(self, client, db, fake_app, owner_headers, repo_base, lr):
        silo, repo = lr
        items = _populate(db, repo, repo_base, 3)
        mid, mpath = _media(db, repo, repo_base)
        store = MagicMock()
        with patch("services.silo_service._get_vector_store", return_value=store):
            resp = client.delete(_all_url(fake_app, repo), headers=owner_headers)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"deleted_count": 4, "failed_count": 0}
        store.wipe_workspace.assert_called_once_with(f"silo_{silo.silo_id}")
        db.expire_all()
        for rid, path in items:
            assert not _row_exists(db, rid) and not os.path.exists(path)
        assert db.get(Media, mid) is None and not os.path.exists(mpath)
        assert "do not support deleting individual files" not in resp.text  # AC-15
        _lock_free(silo.silo_id)

    def test_ac11_lock_held_is_409(self, client, db, fake_app, owner_headers, repo_base, lr):
        silo, repo = lr
        items = _populate(db, repo, repo_base, 2)
        store = MagicMock()
        conn = silo_indexing_lock.acquire(silo.silo_id)
        assert conn is not None
        try:
            with patch("services.silo_service._get_vector_store", return_value=store):
                resp = client.delete(_all_url(fake_app, repo), headers=owner_headers)
        finally:
            silo_indexing_lock.release(conn, silo.silo_id)
        assert resp.status_code == 409
        assert resp.json()["detail"] == LIGHTRAG_SILO_BUSY_DETAIL
        store.wipe_workspace.assert_not_called()
        assert all(_row_exists(db, rid) and os.path.exists(p) for rid, p in items)

    def test_ac12_wipe_failure_is_500_and_deletes_nothing(self, client, db, fake_app, owner_headers, repo_base, lr):
        silo, repo = lr
        items = _populate(db, repo, repo_base, 2)
        mid, mpath = _media(db, repo, repo_base)
        store = MagicMock()
        store.wipe_workspace.side_effect = RuntimeError("boom /secret/path")
        with patch("services.silo_service._get_vector_store", return_value=store):
            resp = client.delete(_all_url(fake_app, repo), headers=owner_headers)
        assert resp.status_code == 500
        assert "Could not clear the knowledge graph" in resp.json()["detail"]
        assert "boom" not in resp.text
        assert all(_row_exists(db, rid) and os.path.exists(p) for rid, p in items)
        assert db.get(Media, mid) is not None and os.path.exists(mpath)
        _lock_free(silo.silo_id)

    def test_ac13_partial_is_200_with_failed_count_and_error_log(
        self, client, db, fake_app, owner_headers, repo_base, lr, caplog
    ):
        silo, repo = lr
        items = _populate(db, repo, repo_base, 4)
        bad = items[0][0]
        real = ResourceService.delete_resource

        def flaky(rid, db_, **kw):
            return False if rid == bad else real(rid, db_, **kw)

        with patch("services.silo_service._get_vector_store", return_value=MagicMock()), \
                patch("services.resource_service.ResourceService.delete_resource", side_effect=flaky), \
                caplog.at_level(logging.ERROR):
            resp = client.delete(_all_url(fake_app, repo), headers=owner_headers)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"deleted_count": 3, "failed_count": 1}
        assert any(r.levelno == logging.ERROR and str(bad) in r.getMessage() for r in caplog.records)
        assert _row_exists(db, bad)
        _lock_free(silo.silo_id)

        with patch("services.silo_service._get_vector_store", return_value=MagicMock()):
            resp = client.delete(_all_url(fake_app, repo), headers=owner_headers)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"deleted_count": 1, "failed_count": 0}
        assert not any(_row_exists(db, rid) for rid, _ in items)

    def test_ac14_empty_repo_still_wipes(self, client, fake_app, owner_headers, lr):
        silo, repo = lr
        store = MagicMock()
        with patch("services.silo_service._get_vector_store", return_value=store):
            resp = client.delete(_all_url(fake_app, repo), headers=owner_headers)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"deleted_count": 0, "failed_count": 0}
        store.wipe_workspace.assert_called_once_with(f"silo_{silo.silo_id}")

    def test_ac17_pgvector_takes_no_lock(self, client, db, fake_app, owner_headers, repo_base):
        silo = _make_silo(db, fake_app, "PGVECTOR")
        repo = _repo(db, fake_app, silo)
        items = _populate(db, repo, repo_base, 2)
        conn = silo_indexing_lock.acquire(silo.silo_id)
        assert conn is not None
        try:
            with patch("services.resource_service.SiloService.delete_resource") as per_resource, \
                    patch("services.silo_service._get_vector_store") as gvs:
                resp = client.delete(_all_url(fake_app, repo), headers=owner_headers)
        finally:
            silo_indexing_lock.release(conn, silo.silo_id)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"deleted_count": 2, "failed_count": 0}
        assert per_resource.call_count == 2
        gvs.assert_not_called()
        assert not any(_row_exists(db, rid) for rid, _ in items)

    def test_retired_service_is_409_service_retired(self, client, db, fake_app, owner_headers, repo_base, lr):
        from utils.service_lifecycle import ServiceRetiredError

        silo, repo = lr
        items = _populate(db, repo, repo_base, 1)
        err = ServiceRetiredError(
            MagicMock(service_id=1, end_of_service_at=datetime(2020, 1, 1), deprecation_notice=None), "embedding",
        )
        with patch("services.silo_service._get_vector_store", side_effect=err):
            resp = client.delete(_all_url(fake_app, repo), headers=owner_headers)
        assert resp.status_code == 409
        assert resp.json()["error_code"] == "SERVICE_RETIRED"
        assert _row_exists(db, items[0][0])
        _lock_free(silo.silo_id)

    @pytest.mark.parametrize("other", ["repository", "domain"])
    def test_shared_silo_is_409_and_touches_nothing(self, client, db, fake_app, owner_headers, repo_base, lr, other):
        from models.domain import Domain

        silo, repo = lr
        items = _populate(db, repo, repo_base, 2)
        if other == "repository":
            _repo(db, fake_app, silo)
        else:
            db.add(Domain(name="D", app_id=fake_app.app_id, silo_id=silo.silo_id))
            db.flush()
        store = MagicMock()
        with patch("services.silo_service._get_vector_store", return_value=store):
            resp = client.delete(_all_url(fake_app, repo), headers=owner_headers)
        assert resp.status_code == 409
        assert "shared with another repository or domain" in resp.json()["detail"]
        store.wipe_workspace.assert_not_called()
        assert all(_row_exists(db, rid) and os.path.exists(p) for rid, p in items)
        _lock_free(silo.silo_id)

    def test_media_delete_failure_is_counted(self, client, db, fake_app, owner_headers, repo_base, lr):
        silo, repo = lr
        mid, _ = _media(db, repo, repo_base)
        db.commit()
        with patch("services.silo_service._get_vector_store", return_value=MagicMock()), \
                patch("services.media_service.MediaService.delete_media_item", return_value=False):
            resp = client.delete(_all_url(fake_app, repo), headers=owner_headers)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"deleted_count": 0, "failed_count": 1}
        _lock_free(silo.silo_id)


class TestLightRagFolderDeleteRefused:
    def test_folder_delete_is_409_and_touches_nothing(self, client, db, fake_app, owner_headers, repo_base):
        from services.silo_service import LIGHTRAG_UNSUPPORTED_DELETE_DETAIL

        silo = _make_silo(db, fake_app, "LIGHTRAG")
        repo = _repo(db, fake_app, silo)
        folder = Folder(name="F", repository_id=repo.repository_id)
        db.add(folder)
        db.flush()
        res, path = _resource_in(db, repo, folder, repo_base, "doc.txt")
        rid, folder_id = res.resource_id, folder.folder_id
        db.commit()

        resp = client.delete(
            f"/internal/apps/{fake_app.app_id}/repositories/{repo.repository_id}/folders/{folder_id}",
            headers=owner_headers,
        )
        assert resp.status_code == 409, resp.text
        assert resp.json()["detail"] == LIGHTRAG_UNSUPPORTED_DELETE_DETAIL
        db.expire_all()
        assert db.get(Folder, folder_id) is not None
        assert _row_exists(db, rid)
        assert os.path.exists(path)
