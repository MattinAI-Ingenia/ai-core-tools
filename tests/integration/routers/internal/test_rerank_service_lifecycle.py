"""Rerank lifecycle: used-by (AC-16), delete protection (AC-20), app cascade (AC-19), health throttling."""
from datetime import datetime

import pytest

from models.agent import Agent
from models.app_collaborator import CollaborationRole
from models.rerank_service import RerankService
from tests.factories import AgentFactory, AIServiceFactory, configure_factories
from tests.integration.routers.internal._eos_helpers import future, make_app, member_headers, past

pytestmark = pytest.mark.integration



def rerank(db, app, name="rr", eos=None, platform_default=False):
    row = RerankService(
        name=name, provider="CohereCompatible", model_name="m", endpoint="https://rr.example.com", api_key="k",  # pragma: allowlist secret
        app_id=app.app_id if app else None, create_date=datetime.now(), end_of_service_at=eos,
        is_platform_default=platform_default,
    )
    db.add(row)
    db.flush()
    return row


def agent(db, app, ai, **kw):
    return AgentFactory(app=app, ai_service=ai, **kw)


@pytest.fixture
def ai(db, fake_app):
    configure_factories(db)
    return AIServiceFactory(app=fake_app)


def used_by_url(app, svc):
    return f"/internal/apps/{app.app_id}/rerank-services/{svc.service_id}/used-by"


class TestUsedBy:
    def test_explicit_inheriting_default_and_platform_flag(self, client, fake_app, owner_headers, db, ai):
        system = rerank(db, None, "rr-platform", platform_default=True)
        explicit = agent(db, fake_app, ai, name="explicit", rag_rerank_enabled=True,
                         rag_rerank_service_id=system.service_id)
        inherit_platform = agent(db, fake_app, ai, name="inherit-platform", rag_rerank_enabled=True)
        off = agent(db, fake_app, ai, name="off", rag_rerank_enabled=False)
        resp = client.get(used_by_url(fake_app, system), headers=owner_headers)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        roles = {a["agent_id"]: a["roles"] for a in body["agents"]}
        assert roles == {explicit.agent_id: ["rerank"], inherit_platform.agent_id: ["rerank_platform_default"]}
        assert off.agent_id not in roles
        assert body["is_platform_default"] is True and body["kind"] == "rerank"
        assert body["silos"] == [] and body["repositories"] == []

        # once the app has its own default the platform default is no longer inherited by its agents
        own = rerank(db, fake_app, "rr-own")
        fake_app.default_rerank_service_id = own.service_id
        db.flush()
        body = client.get(used_by_url(fake_app, system), headers=owner_headers).json()
        assert {a["agent_id"] for a in body["agents"]} == {explicit.agent_id}
        body = client.get(used_by_url(fake_app, own), headers=owner_headers).json()
        assert {a["agent_id"]: a["roles"] for a in body["agents"]} == {
            inherit_platform.agent_id: ["rerank_app_default"]}
        assert [a["app_id"] for a in body["apps_using_as_default"]] == [fake_app.app_id]
        assert body["is_platform_default"] is False

    def test_foreign_service_is_404_and_viewer_can_read(self, client, fake_app, owner_headers, db):
        foreign = rerank(db, make_app(db), "rr-foreign")
        assert client.get(used_by_url(fake_app, foreign), headers=owner_headers).status_code == 404
        own = rerank(db, fake_app, "rr-own")
        viewer = member_headers(db, fake_app, CollaborationRole.VIEWER)
        assert client.get(used_by_url(fake_app, own), headers=viewer).status_code == 200

    def test_admin_view_spans_apps_and_is_omniadmin_only(self, client, fake_app, owner_headers, admin_headers, db, ai):
        system = rerank(db, None, "rr-sys")
        other = make_app(db)
        other_ai = AIServiceFactory(app=other)
        a = agent(db, fake_app, ai, rag_rerank_enabled=True, rag_rerank_service_id=system.service_id)
        b = agent(db, other, other_ai, rag_rerank_enabled=True, rag_rerank_service_id=system.service_id)
        url = f"/internal/admin/system-rerank-services/{system.service_id}/used-by"
        assert client.get(url, headers=owner_headers).status_code == 403
        body = client.get(url, headers=admin_headers).json()
        assert {x["agent_id"] for x in body["agents"]} == {a.agent_id, b.agent_id}
        assert all(x["app_name"] for x in body["agents"])
        own = rerank(db, fake_app, "rr-app-owned")
        assert client.get(f"/internal/admin/system-rerank-services/{own.service_id}/used-by",
                          headers=admin_headers).status_code == 404


class TestDeleteProtection:
    def test_explicit_reference_blocks_with_usage(self, client, fake_app, owner_headers, db, ai):
        svc = rerank(db, fake_app, "rr-ref")
        a = agent(db, fake_app, ai, rag_rerank_enabled=True, rag_rerank_service_id=svc.service_id)
        resp = client.delete(f"/internal/apps/{fake_app.app_id}/rerank-services/{svc.service_id}", headers=owner_headers)
        assert resp.status_code == 409, resp.text
        detail = resp.json()["detail"]
        assert detail["message"] and detail["usage"]["agents"][0]["agent_id"] == a.agent_id
        assert db.get(RerankService, svc.service_id) is not None

    def test_app_default_blocks_even_without_agents(self, client, fake_app, owner_headers, db):
        svc = rerank(db, fake_app, "rr-default")
        fake_app.default_rerank_service_id = svc.service_id
        db.flush()
        resp = client.delete(f"/internal/apps/{fake_app.app_id}/rerank-services/{svc.service_id}", headers=owner_headers)
        assert resp.status_code == 409
        assert resp.json()["detail"]["usage"]["apps_using_as_default"][0]["app_id"] == fake_app.app_id

    def test_inheriting_agents_alone_do_not_block(self, client, fake_app, owner_headers, db, ai):
        svc = rerank(db, fake_app, "rr-inherit")
        agent(db, fake_app, ai, rag_rerank_enabled=True)  # inherits nothing: no default points at svc
        resp = client.delete(f"/internal/apps/{fake_app.app_id}/rerank-services/{svc.service_id}", headers=owner_headers)
        assert resp.status_code == 204, resp.text

    def test_unreferenced_is_deleted(self, client, fake_app, owner_headers, db):
        svc = rerank(db, fake_app, "rr-free")
        sid = svc.service_id
        resp = client.delete(f"/internal/apps/{fake_app.app_id}/rerank-services/{sid}", headers=owner_headers)
        assert resp.status_code == 204
        db.expire_all()
        assert db.get(RerankService, sid) is None

    def test_system_service_blocked_by_platform_flag_and_other_apps_agents(self, client, admin_headers, db, ai):
        url = "/internal/admin/system-rerank-services/"
        flagged = rerank(db, None, "rr-flag", platform_default=True)
        resp = client.delete(url + str(flagged.service_id), headers=admin_headers)
        assert resp.status_code == 409 and resp.json()["detail"]["usage"]["is_platform_default"] is True
        referenced = rerank(db, None, "rr-sys-ref")
        agent(db, ai_app(db, ai), ai, rag_rerank_enabled=True, rag_rerank_service_id=referenced.service_id)
        assert client.delete(url + str(referenced.service_id), headers=admin_headers).status_code == 409
        free = rerank(db, None, "rr-sys-free")
        assert client.delete(url + str(free.service_id), headers=admin_headers).status_code == 204


def ai_app(db, ai):
    from models.app import App

    return db.get(App, ai.app_id)


class TestAppCascade:
    def test_app_deletion_with_rerank_services_default_and_agents(self, db, fake_app, ai):
        from services.app_service import AppService

        system = rerank(db, None, "rr-system")
        own = rerank(db, fake_app, "rr-own")
        fake_app.default_rerank_service_id = own.service_id
        agent(db, fake_app, ai, name="a1", rag_rerank_enabled=True, rag_rerank_service_id=own.service_id)
        agent(db, fake_app, ai, name="a2", rag_rerank_enabled=True, rag_rerank_service_id=system.service_id)
        agent(db, fake_app, ai, name="a3", rag_rerank_enabled=True)
        db.flush()
        app_id, system_id = fake_app.app_id, system.service_id

        assert AppService(db).delete_app(app_id) is True
        db.expire_all()
        assert db.query(RerankService).filter(RerankService.app_id == app_id).count() == 0
        assert db.get(RerankService, system_id) is not None
        assert db.query(Agent).filter(Agent.app_id == app_id).count() == 0


    def test_a_failure_after_the_rerank_step_rolls_the_deletion_back(self, db, fake_app, ai, monkeypatch):
        from services.app_service import AppService

        own = rerank(db, fake_app, "rr-own")
        rerank(db, fake_app, "rr-own-2")
        fake_app.default_rerank_service_id = own.service_id
        agent(db, fake_app, ai, rag_rerank_enabled=True, rag_rerank_service_id=own.service_id)
        db.commit()  # persist the setup so the failed deletion's rollback cannot take it away
        app_id = fake_app.app_id

        def boom(self, _app_id):
            raise RuntimeError("forced failure after the rerank step")

        monkeypatch.setattr("repositories.app_collaboration_repository.AppCollaborationRepository."
                            "get_collaborations_by_app", boom)
        assert AppService(db).delete_app(app_id) is False
        db.expire_all()
        assert db.query(RerankService).filter(RerankService.app_id == app_id).count() == 2
        assert db.get(type(fake_app), app_id) is not None
        assert db.get(type(fake_app), app_id).default_rerank_service_id == own.service_id


class TestHealthThrottle:
    @pytest.fixture
    def session_factory(self, db, monkeypatch):
        """SessionLocal stand-in that reuses the test session (its commits are savepoints) and never closes it."""
        class Proxy:
            def __getattr__(self, name):
                return getattr(db, name)

            def close(self):
                pass

        monkeypatch.setattr("services.service_health_service.SessionLocal", lambda: Proxy())

    def test_same_message_is_throttled_but_a_new_one_or_an_old_row_is_written(self, db, session_factory):
        from datetime import timedelta

        from services.service_health_service import ServiceHealthService

        svc = rerank(db, None, "rr-health")
        record = ServiceHealthService.record_failure_message
        record(RerankService, svc.service_id, "Rerank request timed out after 5s")
        db.refresh(svc)
        first_at = svc.last_error_at
        assert svc.last_error == "Rerank request timed out after 5s" and first_at is not None

        record(RerankService, svc.service_id, "Rerank request timed out after 5s")  # throttled
        db.refresh(svc)
        assert svc.last_error_at == first_at

        record(RerankService, svc.service_id, "Reranker rejected the API key (HTTP 401)")  # different message
        db.refresh(svc)
        assert svc.last_error == "Reranker rejected the API key (HTTP 401)" and svc.last_error_at >= first_at

        old = svc.last_error_at - timedelta(minutes=5)
        db.query(RerankService).filter(RerankService.service_id == svc.service_id).update({"last_error_at": old})
        record(RerankService, svc.service_id, "Reranker rejected the API key (HTTP 401)")  # same but stale
        db.refresh(svc)
        assert svc.last_error_at > old

    def test_recorder_writes_failure_and_clears_on_success(self, db, session_factory):
        from services.rerank_health import DbRerankHealthRecorder
        from tools.rerank.types import RerankTarget

        svc = rerank(db, None, "rr-recorder")
        target = RerankTarget.from_service(svc)
        recorder = DbRerankHealthRecorder()
        recorder.record_success(target)  # healthy: nothing to clear
        recorder.record_failure(target, "Rerank request timed out after 5s")
        db.refresh(svc)
        assert svc.last_error == "Rerank request timed out after 5s"
        recorder.record_success(RerankTarget.from_service(svc))
        db.refresh(svc)
        assert svc.last_error is None and svc.last_error_at is None
