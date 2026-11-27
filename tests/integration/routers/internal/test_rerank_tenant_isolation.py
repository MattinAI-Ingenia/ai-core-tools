"""Tenant isolation sweep for rerank services (NFR-3, NFR-4; rerank-services step_013).

App B (its OWNER, the strongest non-platform role) must not list / get / update / delete / test / export / use in
an agent / use in the playground / set as default any service of app A, and system services must not be writable
by anybody but an OMNIADMIN through any route. Nothing about the other tenant may leak in the answer.
"""
from datetime import datetime

import pytest

from models.agent import Agent
from models.app_collaborator import AppCollaborator, CollaborationRole, CollaborationStatus
from models.rerank_service import RerankService
from models.silo import Silo
from tests.factories import AppFactory, EmbeddingServiceFactory, UserFactory, configure_factories
from tests.integration.routers.internal._eos_helpers import headers_for, member_headers

pytestmark = pytest.mark.integration

ADMIN = "/internal/admin/system-rerank-services"
SECRET = "sk-tenant-a-secret"  # pragma: allowlist secret
BODY = {"name": "Hijack", "provider": "CohereCompatible", "model_name": "m", "api_key": "",
        "base_url": "https://rr.example.com"}


def _svc(db, app_id, name):
    row = RerankService(name=name, provider="CohereCompatible", model_name="m", endpoint="https://rr.example.com",
                        api_key=SECRET, app_id=app_id, create_date=datetime.now())
    db.add(row)
    db.flush()
    return row


@pytest.fixture
def tenants(db, fake_app, monkeypatch):
    monkeypatch.setattr("utils.ssrf_guard.resolve_host", lambda host: ["93.184.216.34"])
    configure_factories(db)
    owner_b = UserFactory()
    app_b = AppFactory(owner_id=owner_b.user_id)
    db.flush()
    db.add(AppCollaborator(app_id=app_b.app_id, user_id=owner_b.user_id, role=CollaborationRole.OWNER,
                           invited_by=owner_b.user_id, status=CollaborationStatus.ACCEPTED,
                           accepted_at=datetime.now()))
    db.flush()
    svc_a = _svc(db, fake_app.app_id, "tenant-a-service")
    svc_b = _svc(db, app_b.app_id, "tenant-b-service")
    system = _svc(db, None, "platform-service")
    return type("T", (), {"a": fake_app, "b": app_b, "svc_a": svc_a, "svc_b": svc_b, "system": system,
                          "b_headers": headers_for(owner_b)})


def _urls(app, tail=""):
    return f"/internal/apps/{app.app_id}/rerank-services/{tail}"


def _unchanged(db, row, name):
    db.expire_all()
    fresh = db.get(RerankService, row.service_id)
    assert fresh is not None and fresh.name == name and fresh.api_key == SECRET


class TestAppBCannotReachAppAServices:
    def test_through_its_own_routes_every_operation_is_404(self, client, db, tenants):
        sid, h, b = str(tenants.svc_a.service_id), tenants.b_headers, tenants.b
        calls = [
            client.get(_urls(b, sid), headers=h),
            client.post(_urls(b, sid), json=BODY, headers=h),
            client.delete(_urls(b, sid), headers=h),
            client.post(_urls(b, "test-connection") + f"?service_id={sid}", json=BODY, headers=h),
            client.post(_urls(b, f"{sid}/export"), headers=h),
            client.get(_urls(b, f"{sid}/used-by"), headers=h),
        ]
        assert [c.status_code for c in calls] == [404] * len(calls)
        assert all(SECRET not in c.text and "tenant-a-service" not in c.text for c in calls)
        _unchanged(db, tenants.svc_a, "tenant-a-service")

    def test_listing_and_options_do_not_include_it(self, client, tenants):
        for tail in ("", "options"):
            text = client.get(_urls(tenants.b, tail), headers=tenants.b_headers).text
            assert "tenant-a-service" not in text and SECRET not in text
        assert "tenant-b-service" in client.get(_urls(tenants.b), headers=tenants.b_headers).text

    def test_through_app_a_routes_b_is_not_a_member(self, client, db, tenants):
        sid, h, a = str(tenants.svc_a.service_id), tenants.b_headers, tenants.a
        calls = [
            client.get(_urls(a), headers=h),
            client.get(_urls(a, sid), headers=h),
            client.post(_urls(a, sid), json=BODY, headers=h),
            client.post(_urls(a, "0"), json=BODY, headers=h),
            client.delete(_urls(a, sid), headers=h),
            client.post(_urls(a, "test-connection") + f"?service_id={sid}", json=BODY, headers=h),
            client.post(_urls(a, f"{sid}/export"), headers=h),
            client.get(_urls(a, "options"), headers=h),
        ]
        assert all(c.status_code in (403, 404) for c in calls), [c.status_code for c in calls]
        _unchanged(db, tenants.svc_a, "tenant-a-service")
        assert db.query(RerankService).filter(RerankService.name == "Hijack").count() == 0

    def test_it_cannot_be_the_app_b_default(self, client, db, tenants):
        resp = client.put(f"/internal/apps/{tenants.b.app_id}",
                          json={"name": tenants.b.name, "default_rerank_service_id": tenants.svc_a.service_id},
                          headers=tenants.b_headers)
        assert resp.status_code in (400, 404, 422), resp.text
        db.expire_all()
        assert db.get(type(tenants.b), tenants.b.app_id).default_rerank_service_id is None
        assert "tenant-a-service" not in resp.text

    def test_it_cannot_be_the_app_a_default_by_b(self, client, db, tenants):
        resp = client.put(f"/internal/apps/{tenants.a.app_id}",
                          json={"name": tenants.a.name, "default_rerank_service_id": tenants.svc_b.service_id},
                          headers=tenants.b_headers)
        assert resp.status_code in (403, 404)
        db.expire_all()
        assert db.get(type(tenants.a), tenants.a.app_id).default_rerank_service_id is None

    def test_it_cannot_be_used_by_an_agent_of_b(self, client, db, tenants):
        for fields in ({"rag_rerank_enabled": True, "rag_rerank_service_id": tenants.svc_a.service_id},):
            resp = client.post(f"/internal/apps/{tenants.b.app_id}/agents/0", headers=tenants.b_headers, json={
                "name": "steal", "description": "", "system_prompt": "t", "prompt_template": "", "type": "agent",
                "is_tool": False, "has_memory": False, "service_id": None, "silo_id": None,
                "output_parser_id": None, "temperature": 0.7, "tool_ids": [], "mcp_config_ids": [],
                "skill_ids": [], "vision_service_id": None, "vision_system_prompt": "", "text_system_prompt": "",
                **fields})
            assert 400 <= resp.status_code < 500, resp.text
            assert "tenant-a-service" not in resp.text
        assert db.query(Agent).filter(Agent.app_id == tenants.b.app_id, Agent.name == "steal").count() == 0

    def test_it_cannot_be_used_in_the_playground_of_b(self, client, db, tenants):
        emb = EmbeddingServiceFactory(app=tenants.b)
        silo = Silo(name="b-silo", silo_type="REPO", app_id=tenants.b.app_id, vector_db_type="PGVECTOR",
                    embedding_service_id=emb.service_id)
        db.add(silo)
        db.flush()
        from unittest.mock import MagicMock, patch

        # The collection must exist for the validation to run (a missing one short-circuits to no results).
        with patch("services.silo_service.SiloService.check_silo_collection_exists", return_value=True), \
                patch("services.silo_service._get_vector_store", return_value=MagicMock()):
            resp = client.post(f"/internal/apps/{tenants.b.app_id}/silos/{silo.silo_id}/search",
                               headers=tenants.b_headers,
                               json={"query": "q", "rerank": True, "rerank_service_id": tenants.svc_a.service_id})
            missing = client.post(f"/internal/apps/{tenants.b.app_id}/silos/{silo.silo_id}/search",
                                  headers=tenants.b_headers,
                                  json={"query": "q", "rerank": True, "rerank_service_id": 987654})
        assert resp.status_code == 422, resp.text
        # a foreign id is indistinguishable from a missing one apart from the number the caller sent
        assert resp.json()["detail"].replace(str(tenants.svc_a.service_id), "N") == \
            missing.json()["detail"].replace("987654", "N")

    def test_a_silo_of_a_cannot_be_searched_by_b(self, client, db, tenants):
        emb = EmbeddingServiceFactory(app=tenants.a)
        silo = Silo(name="a-silo", silo_type="REPO", app_id=tenants.a.app_id, vector_db_type="PGVECTOR",
                    embedding_service_id=emb.service_id)
        db.add(silo)
        db.flush()
        resp = client.post(f"/internal/apps/{tenants.a.app_id}/silos/{silo.silo_id}/search", headers=tenants.b_headers,
                           json={"query": "q", "rerank": True, "rerank_service_id": tenants.svc_a.service_id})
        assert resp.status_code in (403, 404)


class TestSystemServicesAreOmniadminOnly:
    @pytest.mark.parametrize("role", [None, CollaborationRole.VIEWER, CollaborationRole.EDITOR,
                                      CollaborationRole.ADMINISTRATOR])
    def test_app_routes_never_write_a_system_service(self, client, db, tenants, role):
        headers = tenants.b_headers if role is None else member_headers(db, tenants.a, role)
        app = tenants.b if role is None else tenants.a
        sid = str(tenants.system.service_id)
        calls = [
            client.post(_urls(app, sid), json=BODY, headers=headers),
            client.delete(_urls(app, sid), headers=headers),
            client.post(_urls(app, "test-connection") + f"?service_id={sid}", json=BODY, headers=headers),
        ]
        assert all(c.status_code in (403, 404) for c in calls), [c.status_code for c in calls]
        db.expire_all()
        row = db.get(RerankService, tenants.system.service_id)
        assert row is not None and row.name == "platform-service" and row.app_id is None
        assert db.query(RerankService).filter(RerankService.name == "Hijack").count() == 0

    @pytest.mark.parametrize("role", [None, CollaborationRole.VIEWER, CollaborationRole.ADMINISTRATOR])
    def test_admin_routes_are_403_for_everybody_else(self, client, db, tenants, role):
        headers = tenants.b_headers if role is None else member_headers(db, tenants.a, role)
        sid = tenants.system.service_id
        calls = [
            client.get(ADMIN, headers=headers),
            client.post(ADMIN, json=BODY, headers=headers),
            client.put(ADMIN + "/platform-default", json={"service_id": sid}, headers=headers),
            client.post(ADMIN + "/test-connection", json=BODY, headers=headers),
            client.get(f"{ADMIN}/{sid}", headers=headers),
            client.get(f"{ADMIN}/{sid}/used-by", headers=headers),
            client.put(f"{ADMIN}/{sid}", json=BODY, headers=headers),
            client.delete(f"{ADMIN}/{sid}", headers=headers),
        ]
        assert [c.status_code for c in calls] == [403] * len(calls)
        db.expire_all()
        row = db.get(RerankService, sid)
        assert row is not None and row.name == "platform-service" and row.is_platform_default is False

    def test_an_app_created_service_is_never_system(self, client, tenants):
        resp = client.post(_urls(tenants.b, "0"), headers=tenants.b_headers, json={**BODY, "name": "mine",
                           "app_id": None, "is_system": True, "is_platform_default": True})
        assert resp.status_code in (200, 201), resp.text
        assert resp.json()["is_system"] is False
        assert resp.json().get("is_platform_default", False) is False

    def test_unauthenticated_requests_are_rejected(self, client, tenants):
        assert client.get(ADMIN).status_code in (401, 403)
        assert client.get(_urls(tenants.b)).status_code in (401, 403)
