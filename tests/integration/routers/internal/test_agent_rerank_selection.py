"""Agent rerank configuration: save-time validation (AD-17, AC-11 storage, AC-13 backend)."""
from datetime import datetime

import pytest
from sqlalchemy import event

from models.agent import Agent
from models.rerank_service import RerankService
from tests.factories import AppFactory, configure_factories
from tests.integration.routers.internal._eos_helpers import make_app, past

pytestmark = pytest.mark.integration

NO_SERVICE_MSG = "no rerank service is available"


def rerank(db, app, name="rr", eos=None, platform_default=False):
    row = RerankService(
        name=name, provider="CohereCompatible", model_name="m", endpoint="https://rr.example.com", api_key="k",  # pragma: allowlist secret
        app_id=app.app_id if app else None, create_date=datetime.now(), end_of_service_at=eos,
        is_platform_default=platform_default,
    )
    db.add(row)
    db.flush()
    return row


@pytest.fixture(autouse=True)
def _no_platform_default(db):
    db.query(RerankService).filter(RerankService.is_platform_default.is_(True)).update(
        {"is_platform_default": False})
    db.flush()


def payload(**overrides) -> dict:
    base = {
        "name": "Rerank agent", "description": "", "system_prompt": "t", "prompt_template": "", "type": "agent",
        "is_tool": False, "has_memory": False, "service_id": None, "silo_id": None, "output_parser_id": None,
        "temperature": 0.7, "tool_ids": [], "mcp_config_ids": [], "skill_ids": [], "vision_service_id": None,
        "vision_system_prompt": "", "text_system_prompt": "",
    }
    base.update(overrides)
    return base


def post(client, app, headers, **fields):
    return client.post(f"/internal/apps/{app.app_id}/agents/0", json=payload(**fields), headers=headers)


class TestEnabledWithoutExplicitService:
    def test_rejected_when_no_default_exists(self, client, fake_app, owner_headers):
        resp = post(client, fake_app, owner_headers, rag_rerank_enabled=True)
        assert 400 <= resp.status_code < 500
        assert NO_SERVICE_MSG in resp.text

    def test_inherit_with_app_default_stores_null(self, client, fake_app, owner_headers, db):
        fake_app.default_rerank_service_id = rerank(db, fake_app, "rr-app").service_id
        db.flush()
        resp = post(client, fake_app, owner_headers, rag_rerank_enabled=True, rag_rerank_service_id=None)
        assert resp.status_code in (200, 201), resp.text
        body = resp.json()
        assert body["rag_rerank_enabled"] is True and body["rag_rerank_service_id"] is None
        agent = db.get(Agent, body["agent_id"])
        assert agent.rag_rerank_service_id is None

    def test_inherit_with_platform_default_is_accepted(self, client, fake_app, owner_headers, db):
        rerank(db, None, "rr-platform", platform_default=True)
        resp = post(client, fake_app, owner_headers, rag_rerank_enabled=True)
        assert resp.status_code in (200, 201), resp.text

    def test_disabled_needs_no_service(self, client, fake_app, owner_headers):
        resp = post(client, fake_app, owner_headers, rag_rerank_enabled=False)
        assert resp.status_code in (200, 201), resp.text
        assert resp.json()["rag_rerank_enabled"] is False


class TestExplicitService:
    def test_own_and_system_services_are_accepted(self, client, fake_app, owner_headers, db):
        own = rerank(db, fake_app, "rr-own")
        resp = post(client, fake_app, owner_headers, rag_rerank_enabled=True, rag_rerank_service_id=own.service_id,
                    rag_rerank_top_n=20)
        assert resp.status_code in (200, 201), resp.text
        assert resp.json()["rag_rerank_service_id"] == own.service_id
        assert resp.json()["rag_rerank_top_n"] == 20
        system = rerank(db, None, "rr-sys")
        resp = post(client, fake_app, owner_headers, rag_rerank_enabled=True, rag_rerank_service_id=system.service_id)
        assert resp.status_code in (200, 201), resp.text

    def test_foreign_and_missing_service_rejected_alike(self, client, fake_app, owner_headers, db):
        foreign = rerank(db, make_app(db), "rr-foreign")
        a = post(client, fake_app, owner_headers, rag_rerank_enabled=True, rag_rerank_service_id=foreign.service_id)
        b = post(client, fake_app, owner_headers, rag_rerank_enabled=True, rag_rerank_service_id=987654321)
        assert a.status_code == b.status_code == 400
        assert "does not exist or does not belong to this app" in a.text

    def test_retired_service_rejected_unless_already_selected(self, client, fake_app, owner_headers, db):
        retired = rerank(db, fake_app, "rr-retired", eos=past(2))
        resp = post(client, fake_app, owner_headers, rag_rerank_enabled=True,
                    rag_rerank_service_id=retired.service_id)
        assert resp.status_code == 400 and "retired" in resp.text

        ok = rerank(db, fake_app, "rr-ok")
        created = post(client, fake_app, owner_headers, rag_rerank_enabled=True, rag_rerank_service_id=ok.service_id)
        assert created.status_code in (200, 201), created.text
        agent_id = created.json()["agent_id"]
        ok.end_of_service_at = past(1)
        db.flush()
        resp = client.post(
            f"/internal/apps/{fake_app.app_id}/agents/{agent_id}",
            json=payload(rag_rerank_enabled=True, rag_rerank_service_id=ok.service_id), headers=owner_headers,
        )
        assert resp.status_code in (200, 201), resp.text


class TestUpdateKeepsOrClearsSelection:
    def _update(self, client, app, headers, agent_id, **fields):
        return client.post(f"/internal/apps/{app.app_id}/agents/{agent_id}", json=payload(**fields), headers=headers)

    def _created(self, client, app, headers, svc):
        resp = post(client, app, headers, rag_rerank_enabled=True, rag_rerank_service_id=svc.service_id,
                    rag_rerank_top_n=7)
        assert resp.status_code in (200, 201), resp.text
        return resp.json()["agent_id"]

    def test_update_omitting_rerank_fields_keeps_them(self, client, fake_app, owner_headers, db):
        svc = rerank(db, fake_app, "rr-keep")
        agent_id = self._created(client, fake_app, owner_headers, svc)
        resp = self._update(client, fake_app, owner_headers, agent_id, system_prompt="changed prompt")
        assert resp.status_code in (200, 201), resp.text
        db.expire_all()
        agent = db.get(Agent, agent_id)
        assert agent.system_prompt == "changed prompt"
        assert agent.rag_rerank_enabled is True
        assert agent.rag_rerank_service_id == svc.service_id and agent.rag_rerank_top_n == 7

    def test_explicit_null_service_means_inherit(self, client, fake_app, owner_headers, db):
        svc = rerank(db, fake_app, "rr-inherit")
        fake_app.default_rerank_service_id = svc.service_id
        db.flush()
        agent_id = self._created(client, fake_app, owner_headers, svc)
        resp = self._update(client, fake_app, owner_headers, agent_id, rag_rerank_service_id=None)
        assert resp.status_code in (200, 201), resp.text
        db.expire_all()
        agent = db.get(Agent, agent_id)
        assert agent.rag_rerank_service_id is None and agent.rag_rerank_top_n == 7


class TestInheritedRetiredDefault:
    def test_save_rejected_when_inherited_app_default_is_retired(self, client, fake_app, owner_headers, db):
        fake_app.default_rerank_service_id = rerank(db, fake_app, "rr-old", eos=past(2)).service_id
        db.flush()
        resp = post(client, fake_app, owner_headers, rag_rerank_enabled=True)
        assert resp.status_code == 400 and NO_SERVICE_MSG in resp.text


class TestResolveStatementCount:
    def test_resolve_is_one_statement_even_without_candidates(self, db, fake_app):
        from services.rerank_resolution_service import RerankResolutionService

        statements = []
        conn = db.get_bind()

        def _on(c, cursor, statement, *a):
            statements.append(statement)

        event.listen(conn, "before_cursor_execute", _on)
        try:
            result = RerankResolutionService.resolve(db, fake_app.app_id, None)
        finally:
            event.remove(conn, "before_cursor_execute", _on)
        assert result.service is None and len(statements) == 1


class TestUnrelatedEditsAreNotBlocked:
    def test_rename_of_enabled_inheriting_agent_succeeds_without_default(self, client, fake_app, owner_headers, db):
        default = rerank(db, fake_app, "rr-gone")
        fake_app.default_rerank_service_id = default.service_id
        db.flush()
        created = post(client, fake_app, owner_headers, rag_rerank_enabled=True)
        assert created.status_code in (200, 201), created.text
        agent_id = created.json()["agent_id"]
        fake_app.default_rerank_service_id = None
        db.flush()
        resp = client.post(f"/internal/apps/{fake_app.app_id}/agents/{agent_id}",
                           json=payload(name="Renamed"), headers=owner_headers)
        assert resp.status_code in (200, 201), resp.text
        # resending the rerank settings is an explicit change and is judged
        resp = client.post(f"/internal/apps/{fake_app.app_id}/agents/{agent_id}",
                           json=payload(name="Renamed", rag_rerank_enabled=True), headers=owner_headers)
        assert resp.status_code == 400 and NO_SERVICE_MSG in resp.text

    def test_enabling_with_no_default_is_still_rejected(self, client, fake_app, owner_headers):
        created = post(client, fake_app, owner_headers)
        agent_id = created.json()["agent_id"]
        resp = client.post(f"/internal/apps/{fake_app.app_id}/agents/{agent_id}",
                           json=payload(rag_rerank_enabled=True), headers=owner_headers)
        assert resp.status_code == 400 and NO_SERVICE_MSG in resp.text

    def test_changing_the_explicit_service_still_validates_ownership(self, client, fake_app, owner_headers, db):
        own = rerank(db, fake_app, "rr-own2")
        created = post(client, fake_app, owner_headers, rag_rerank_enabled=True, rag_rerank_service_id=own.service_id)
        agent_id = created.json()["agent_id"]
        foreign = rerank(db, make_app(db), "rr-foreign2")
        resp = client.post(f"/internal/apps/{fake_app.app_id}/agents/{agent_id}",
                           json=payload(rag_rerank_service_id=foreign.service_id), headers=owner_headers)
        assert resp.status_code == 400 and "does not belong to this app" in resp.text
