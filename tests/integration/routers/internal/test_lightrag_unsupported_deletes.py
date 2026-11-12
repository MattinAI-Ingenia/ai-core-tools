"""
Integration tests (FR-13): internal partial-delete endpoints answer 409 on a LightRAG silo, before anything is
counted, deleted or built; the same endpoints on a PGVector silo are unchanged. Only the vector store is mocked.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from models.domain import Domain
from models.domain_url import DomainUrl
from models.enums.discovery_source import DiscoverySource
from models.enums.domain_url_status import DomainUrlStatus
from models.media import Media
from models.repository import Repository
from models.silo import Silo
from services.silo_service import LIGHTRAG_UNSUPPORTED_DELETE_DETAIL

GET_STORE = "services.silo_service._get_vector_store"


def _silo(db, app, vector_db_type):
    silo = Silo(name=f"{vector_db_type} Silo", description="d", app_id=app.app_id, silo_type="CUSTOM",
                status="active", vector_db_type=vector_db_type)
    db.add(silo)
    db.flush()
    return silo


def _domain_url(db, app, silo):
    domain = Domain(name="D", description="d", base_url="https://example.com", content_tag="body",
                    app_id=app.app_id, silo_id=silo.silo_id, create_date=datetime.now())
    db.add(domain)
    db.flush()
    row = DomainUrl(domain_id=domain.domain_id, url="https://example.com/p", normalized_url="https://example.com/p",
                    status=DomainUrlStatus.INDEXED, discovered_via=DiscoverySource.MANUAL, depth=0,
                    created_at=datetime.utcnow(), updated_at=datetime.utcnow())
    db.add(row)
    db.flush()
    return domain, row


def _media(db, app, silo, tmp_path):
    repo = Repository(name="Repo", type="default", status="active", app_id=app.app_id, silo_id=silo.silo_id,
                      create_date=datetime.now())
    db.add(repo)
    db.flush()
    path = tmp_path / "clip.mp4"
    path.write_text("x")
    media = Media(name="clip", silo_id=silo.silo_id, repository_id=repo.repository_id, source_type="upload",
                  status="completed", file_path=str(path), create_date=datetime.now())
    db.add(media)
    db.flush()
    return repo, media, path


def _alive(db, model, pk):
    db.expire_all()
    return db.get(model, pk) is not None


@pytest.mark.parametrize("kind", ["LIGHTRAG", "PGVECTOR"])
class TestInternalPartialDeletes:
    def test_documents_delete(self, client, db, fake_app, owner_headers, kind):
        silo = _silo(db, fake_app, kind)
        url = f"/internal/apps/{fake_app.app_id}/silos/{silo.silo_id}/documents"
        store = MagicMock()
        with patch(GET_STORE, return_value=store) as get_store, patch(
            "services.silo_service.SiloService.check_silo_collection_exists", return_value=True
        ):
            resp = client.request("DELETE", url, json={"document_ids": ["a"]}, headers=owner_headers)
        if kind == "LIGHTRAG":
            assert resp.status_code == 409
            assert resp.json()["detail"] == LIGHTRAG_UNSUPPORTED_DELETE_DETAIL
            get_store.assert_not_called()
        else:
            assert resp.status_code == 200
            store.delete_documents.assert_called_once()

    def test_domain_url_delete(self, client, db, fake_app, owner_headers, kind):
        domain, row = _domain_url(db, fake_app, _silo(db, fake_app, kind))
        url_id = row.id
        url = f"/internal/apps/{fake_app.app_id}/domains/{domain.domain_id}/urls/{url_id}"
        with patch(GET_STORE) as get_store, patch("services.silo_service.SiloService.delete_url") as delete_url:
            resp = client.delete(url, headers=owner_headers)
        if kind == "LIGHTRAG":
            assert resp.status_code == 409
            assert resp.json()["detail"] == LIGHTRAG_UNSUPPORTED_DELETE_DETAIL
            assert _alive(db, DomainUrl, url_id)
            get_store.assert_not_called()
            delete_url.assert_not_called()
        else:
            assert resp.status_code == 200
            assert not _alive(db, DomainUrl, url_id)
            delete_url.assert_called_once()

    def test_media_delete(self, client, db, fake_app, owner_headers, tmp_path, kind):
        repo, media, path = _media(db, fake_app, _silo(db, fake_app, kind), tmp_path)
        media_id = media.media_id
        url = f"/internal/apps/{fake_app.app_id}/repositories/{repo.repository_id}/media/{media_id}"
        with patch(GET_STORE) as get_store, patch("services.silo_service.SiloService.delete_media") as delete_media:
            resp = client.delete(url, headers=owner_headers)
        if kind == "LIGHTRAG":
            assert resp.status_code == 409
            assert resp.json()["detail"] == LIGHTRAG_UNSUPPORTED_DELETE_DETAIL
            assert _alive(db, Media, media_id)
            assert path.exists()
            get_store.assert_not_called()
            delete_media.assert_not_called()
        else:
            delete_media.assert_called_once()
            assert resp.status_code == 200
            assert not _alive(db, Media, media_id)
            assert not path.exists()
