"""App default rerank service: API semantics, retired handling and role checks (FR-7, AC-7)."""
from datetime import datetime

import pytest

from models.app_collaborator import CollaborationRole
from models.rerank_service import RerankService
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


def put(client, app, headers, **fields):
    return client.put(f"/internal/apps/{app.app_id}", json={"name": app.name, **fields}, headers=headers)


class TestAppApi:
    def test_default_is_null_then_set_read_and_cleared(self, client, fake_app, owner_headers, db):
        assert client.get(f"/internal/apps/{fake_app.app_id}", headers=owner_headers).json()[
            "default_rerank_service_id"] is None
        svc = rerank(db, fake_app, "rr-default")
        resp = put(client, fake_app, owner_headers, default_rerank_service_id=svc.service_id)
        assert resp.status_code == 200, resp.text
        assert resp.json()["default_rerank_service_id"] == svc.service_id
        assert client.get(f"/internal/apps/{fake_app.app_id}", headers=owner_headers).json()[
            "default_rerank_service_id"] == svc.service_id
        # omitted keeps
        resp = put(client, fake_app, owner_headers, enable_openai_api=True)
        assert resp.status_code == 200 and resp.json()["default_rerank_service_id"] == svc.service_id
        # explicit null clears
        resp = put(client, fake_app, owner_headers, default_rerank_service_id=None)
        assert resp.status_code == 200 and resp.json()["default_rerank_service_id"] is None

    def test_a_system_service_is_allowed(self, client, fake_app, owner_headers, db):
        system = rerank(db, None, "rr-system")
        resp = put(client, fake_app, owner_headers, default_rerank_service_id=system.service_id)
        assert resp.status_code == 200 and resp.json()["default_rerank_service_id"] == system.service_id

    def test_another_apps_service_and_a_missing_one_are_rejected_alike(self, client, fake_app, owner_headers, db):
        foreign = rerank(db, make_app(db), "rr-foreign")
        a = put(client, fake_app, owner_headers, default_rerank_service_id=foreign.service_id)
        b = put(client, fake_app, owner_headers, default_rerank_service_id=987654321)
        assert a.status_code == b.status_code == 422
        assert a.json()["detail"].split("'")[2] == b.json()["detail"].split("'")[2]
        db.refresh(fake_app)
        assert fake_app.default_rerank_service_id is None

    def test_a_retired_service_cannot_be_newly_selected_but_can_be_kept(self, client, fake_app, owner_headers, db):
        retired = rerank(db, fake_app, "rr-retired", eos=past(2))
        resp = put(client, fake_app, owner_headers, default_rerank_service_id=retired.service_id)
        assert resp.status_code == 422, resp.text
        fake_app.default_rerank_service_id = retired.service_id
        db.flush()
        resp = put(client, fake_app, owner_headers, default_rerank_service_id=retired.service_id)
        assert resp.status_code == 200 and resp.json()["default_rerank_service_id"] == retired.service_id

    def test_only_administrators_and_up_can_change_it(self, client, fake_app, owner_headers, db):
        svc = rerank(db, fake_app, "rr-role")
        for role in (CollaborationRole.VIEWER, CollaborationRole.EDITOR):
            headers = member_headers(db, fake_app, role)
            assert put(client, fake_app, headers, default_rerank_service_id=svc.service_id).status_code == 403
        db.refresh(fake_app)
        assert fake_app.default_rerank_service_id is None
        headers = member_headers(db, fake_app, CollaborationRole.ADMINISTRATOR)
        resp = put(client, fake_app, headers, default_rerank_service_id=svc.service_id)
        assert resp.status_code == 200, resp.text
        assert resp.json()["default_rerank_service_id"] == svc.service_id

    def test_a_deprecated_default_shows_in_the_app_warnings(self, client, fake_app, owner_headers, db):
        svc = rerank(db, fake_app, "rr-soon", eos=future(3))
        fake_app.default_rerank_service_id = svc.service_id
        db.flush()
        body = client.get(f"/internal/apps/{fake_app.app_id}", headers=owner_headers).json()
        warnings = [(w["kind"], w["role"], w["via"], w["lifecycle_status"]) for w in body["lifecycle_warnings"]]
        assert ("rerank", "rerank", "app default", "deprecated") in warnings
