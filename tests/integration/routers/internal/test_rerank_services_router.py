"""Integration tests for the rerank service stack: app routes and OMNIADMIN system routes.

Requires a Postgres test DB (TEST_DATABASE_URL, database name test_db or test_db_*).
"""
import threading
from datetime import datetime, timedelta, timezone

import httpx
import pytest

import config
from models.rerank_service import RerankService
from tests.factories import AppFactory, UserFactory, configure_factories
from tools.rerank import http_clients

pytestmark = pytest.mark.integration

SECRET = "sk-secret-abcd"  # pragma: allowlist secret
BASE = "https://rr.example.com"


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch):
    monkeypatch.setattr("utils.ssrf_guard.resolve_host", lambda host: ["93.184.216.34"])
    monkeypatch.setattr(config, "RERANK_ALLOW_PRIVATE_ENDPOINTS", False)
    yield
    http_clients._set_clients_for_tests(None)


def _body(**over):
    data = {"name": "Reranker", "provider": "CohereCompatible", "model_name": "bge-reranker", "api_key": SECRET,
            "base_url": BASE + "/"}
    data.update(over)
    return data


def _svc(db, app_id=None, **over):
    data = dict(name="S", provider="CohereCompatible", model_name="m", endpoint=BASE, api_key=SECRET, app_id=app_id,
                create_date=datetime.now())
    data.update(over)
    row = RerankService(**data)
    db.add(row)
    db.flush()
    return row


@pytest.fixture
def other_app(db):
    configure_factories(db)
    return AppFactory(owner_id=UserFactory().user_id)


def _url(app, tail=""):
    return f"/internal/apps/{app.app_id}/rerank-services/{tail}"


class TestAppCrud:
    def test_create_get_update_delete_with_masking(self, client, fake_app, app_admin_headers):
        r = client.post(_url(fake_app, "0"), json=_body(), headers=app_admin_headers)
        assert r.status_code == 200, r.text
        detail = r.json()
        sid = detail["service_id"]
        assert detail["api_key"] == "****abcd" and SECRET not in r.text
        assert detail["base_url"] == BASE and detail["is_system"] is False

        # masked key preserved on update, other fields change
        r = client.post(_url(fake_app, str(sid)), json=_body(name="Renamed", api_key="****abcd"),
                        headers=app_admin_headers)
        assert r.status_code == 200 and r.json()["name"] == "Renamed"
        assert r.json()["api_key"] == "****abcd"

        # a new key replaces it
        r = client.post(_url(fake_app, str(sid)), json=_body(api_key="sk-other-wxyz"), headers=app_admin_headers)
        assert r.json()["api_key"] == "****wxyz"

        r = client.get(_url(fake_app, str(sid)), headers=app_admin_headers)
        assert r.status_code == 200 and "sk-other" not in r.text
        assert client.delete(_url(fake_app, str(sid)), headers=app_admin_headers).status_code == 204
        assert client.get(_url(fake_app, str(sid)), headers=app_admin_headers).status_code == 404

    def test_template_for_id_zero(self, client, fake_app, app_viewer_headers):
        r = client.get(_url(fake_app, "0"), headers=app_viewer_headers)
        assert r.status_code == 200
        assert r.json()["service_id"] == 0
        assert [p["value"] for p in r.json()["available_providers"]] == ["CohereCompatible"]

    def test_invalid_payload_422(self, client, fake_app, app_admin_headers):
        for over in ({"provider": "X"}, {"base_url": "ftp://a"}, {"model_name": " "}, {"name": ""}):
            r = client.post(_url(fake_app, "0"), json=_body(**over), headers=app_admin_headers)
            assert r.status_code == 422, over

    def test_private_endpoint_rejected_unless_flag(self, client, fake_app, app_admin_headers, monkeypatch):
        body = _body(base_url="http://10.0.0.5:8000", api_key="")
        assert client.post(_url(fake_app, "0"), json=body, headers=app_admin_headers).status_code == 422
        monkeypatch.setattr(config, "RERANK_ALLOW_PRIVATE_ENDPOINTS", True)
        assert client.post(_url(fake_app, "0"), json=body, headers=app_admin_headers).status_code == 200

    def test_key_over_plain_http_to_public_host_rejected(self, client, fake_app, app_admin_headers):
        r = client.post(_url(fake_app, "0"), json=_body(base_url="http://rr.example.com"), headers=app_admin_headers)
        assert r.status_code == 422
        assert SECRET not in r.text

    def test_needs_api_key_only_for_placeholder(self, client, fake_app, db, app_admin_headers):
        _svc(db, fake_app.app_id, name="a", api_key="CHANGE_ME")
        _svc(db, fake_app.app_id, name="b", api_key="")
        items = {i["name"]: i for i in client.get(_url(fake_app), headers=app_admin_headers).json()}
        assert items["a"]["needs_api_key"] is True and items["b"]["needs_api_key"] is False

    def test_masked_key_on_create_is_stored_as_empty(self, client, fake_app, app_admin_headers):
        r = client.post(_url(fake_app, "0"), json=_body(api_key="****abcd"), headers=app_admin_headers)
        assert r.status_code == 200 and r.json()["api_key"] == ""

    def test_masked_key_kept_only_for_same_host(self, client, fake_app, db, app_admin_headers):
        row = _svc(db, fake_app.app_id)
        other = _body(api_key="****abcd", base_url="https://evil.example.org")
        r = client.post(_url(fake_app, str(row.service_id)), json=other, headers=app_admin_headers)
        assert r.status_code == 422 and "Re-enter the API key" in r.json()["detail"]
        db.refresh(row)
        assert row.api_key == SECRET and row.endpoint == BASE
        # a freshly typed key may go to the new host
        r = client.post(_url(fake_app, str(row.service_id)), json={**other, "api_key": "sk-new-0000"},
                        headers=app_admin_headers)
        assert r.status_code == 200 and r.json()["base_url"] == "https://evil.example.org"

    def test_test_connection_other_host_with_masked_key_422(self, client, fake_app, db, app_admin_headers):
        row = _svc(db, fake_app.app_id)
        calls = []
        _transport(lambda req: calls.append(req) or _ok(req))
        r = client.post(_url(fake_app, "test-connection") + f"?service_id={row.service_id}",
                        json=_body(api_key="****abcd", base_url="https://evil.example.org"), headers=app_admin_headers)
        assert r.status_code == 422 and not calls

    def test_lifecycle_fields(self, client, fake_app, app_admin_headers):
        eos = (datetime.now(timezone.utc) + timedelta(days=5)).isoformat()
        r = client.post(_url(fake_app, "0"), json=_body(end_of_service_at=eos, deprecation_notice="moving"),
                        headers=app_admin_headers)
        assert r.status_code == 200
        assert r.json()["lifecycle_status"] == "deprecated" and r.json()["deprecation_notice"] == "moving"


class TestScopingAndRoles:
    def test_list_includes_system_flagged(self, client, fake_app, db, app_viewer_headers):
        _svc(db, fake_app.app_id, name="mine")
        _svc(db, None, name="platform")
        items = {i["name"]: i for i in client.get(_url(fake_app), headers=app_viewer_headers).json()}
        assert items["mine"]["is_system"] is False and items["platform"]["is_system"] is True
        assert "api_key" not in items["platform"]
        assert items["platform"]["base_url"] == "" and items["mine"]["base_url"] == BASE

    def test_system_base_url_hidden_from_app_owner_but_kept_in_admin_detail(
        self, client, fake_app, db, owner_headers, admin_headers
    ):
        sys_row = _svc(db, None, name="platform", endpoint="http://10.9.9.9:8000", api_key="")
        for item in client.get(_url(fake_app), headers=owner_headers).json():
            if item["is_system"]:
                assert item["base_url"] == ""
        assert "10.9.9.9" not in client.get(_url(fake_app, "options"), headers=owner_headers).text
        detail = client.get(f"{ADMIN}/{sys_row.service_id}", headers=admin_headers).json()
        assert detail["base_url"] == "http://10.9.9.9:8000"

    def test_system_service_read_only_in_app_routes(self, client, fake_app, db, app_admin_headers):
        sys_svc = _svc(db, None, name="platform")
        sid = str(sys_svc.service_id)
        assert client.get(_url(fake_app, sid), headers=app_admin_headers).status_code == 404
        assert client.post(_url(fake_app, sid), json=_body(), headers=app_admin_headers).status_code == 404
        assert client.delete(_url(fake_app, sid), headers=app_admin_headers).status_code == 404
        assert client.post(_url(fake_app, "test-connection") + f"?service_id={sid}", json=_body(),
                           headers=app_admin_headers).status_code == 404

    def test_other_app_cannot_touch_service(self, client, fake_app, other_app, db, app_admin_headers):
        foreign = _svc(db, other_app.app_id, name="foreign")
        sid = str(foreign.service_id)
        assert client.get(_url(fake_app, sid), headers=app_admin_headers).status_code == 404
        assert client.post(_url(fake_app, sid), json=_body(), headers=app_admin_headers).status_code == 404
        assert client.delete(_url(fake_app, sid), headers=app_admin_headers).status_code == 404
        r = client.post(_url(fake_app, "test-connection") + f"?service_id={sid}", json=_body(api_key=""),
                        headers=app_admin_headers)
        assert r.status_code == 404
        names = [i["name"] for i in client.get(_url(fake_app), headers=app_admin_headers).json()]
        assert "foreign" not in names

    def test_viewer_cannot_write(self, client, fake_app, db, app_viewer_headers):
        row = _svc(db, fake_app.app_id)
        assert client.post(_url(fake_app, "0"), json=_body(), headers=app_viewer_headers).status_code == 403
        assert client.post(_url(fake_app, str(row.service_id)), json=_body(), headers=app_viewer_headers).status_code == 403
        assert client.delete(_url(fake_app, str(row.service_id)), headers=app_viewer_headers).status_code == 403
        assert client.post(_url(fake_app, "test-connection"), json=_body(), headers=app_viewer_headers).status_code == 403
        assert client.get(_url(fake_app), headers=app_viewer_headers).status_code == 200


class TestOptionsAC8:
    def test_retired_app_default_has_no_platform_fallback(self, client, fake_app, db, app_viewer_headers):
        mine = _svc(db, fake_app.app_id, name="mine", end_of_service_at=datetime.now(timezone.utc) - timedelta(days=1))
        _svc(db, None, name="platform", is_platform_default=True)
        fake_app.default_rerank_service_id = mine.service_id
        db.flush()
        r = client.get(_url(fake_app, "options"), headers=app_viewer_headers).json()
        assert r["effective_default"] is None and r["platform_default"] is not None


class TestOptions:
    def test_options_and_effective_default(self, client, fake_app, db, app_viewer_headers):
        mine = _svc(db, fake_app.app_id, name="mine")
        plat = _svc(db, None, name="platform", is_platform_default=True)
        r = client.get(_url(fake_app, "options"), headers=app_viewer_headers).json()
        assert {o["name"] for o in r["options"]} == {"mine", "platform"}
        assert r["app_default"] is None and r["platform_default"]["service_id"] == plat.service_id
        assert r["effective_default"]["service_id"] == plat.service_id

        fake_app.default_rerank_service_id = mine.service_id
        db.flush()
        r = client.get(_url(fake_app, "options"), headers=app_viewer_headers).json()
        assert r["app_default"]["service_id"] == mine.service_id
        assert r["effective_default"]["service_id"] == mine.service_id

        mine.end_of_service_at = datetime.now(timezone.utc) - timedelta(days=1)
        plat.end_of_service_at = datetime.now(timezone.utc) - timedelta(days=1)
        db.flush()
        r = client.get(_url(fake_app, "options"), headers=app_viewer_headers).json()
        assert r["effective_default"] is None
        assert {o["lifecycle_status"] for o in r["options"]} == {"retired"}


def _transport(handler):
    http_clients._set_clients_for_tests(httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def _ok(request):
    return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}]})


class TestTestConnection:
    def test_success_and_stored_key_recovered(self, client, fake_app, db, app_admin_headers):
        row = _svc(db, fake_app.app_id)
        seen = {}

        def handler(request):
            seen["auth"] = request.headers.get("authorization")
            return _ok(request)

        _transport(handler)
        r = client.post(_url(fake_app, "test-connection") + f"?service_id={row.service_id}",
                        json=_body(api_key="****abcd"), headers=app_admin_headers)
        assert r.status_code == 200 and r.json()["status"] == "success"
        assert seen["auth"] == f"Bearer {SECRET}" and SECRET not in r.text

    @pytest.mark.parametrize("handler", [
        lambda req: httpx.Response(401, text=SECRET),
        lambda req: httpx.Response(200, text="<html>" + SECRET),
        lambda req: (_ for _ in ()).throw(httpx.ReadTimeout("t", request=req)),
    ])
    def test_errors_reported_without_key(self, client, fake_app, app_admin_headers, handler):
        _transport(handler)
        r = client.post(_url(fake_app, "test-connection"), json=_body(), headers=app_admin_headers)
        assert r.status_code == 200 and r.json()["status"] == "error"
        assert SECRET not in r.text

    def test_distinct_messages(self, client, fake_app, app_admin_headers):
        msgs = set()
        for h in (lambda q: httpx.Response(401), lambda q: httpx.Response(200, text="x"),
                  lambda q: (_ for _ in ()).throw(httpx.ReadTimeout("t", request=q))):
            _transport(h)
            msgs.add(client.post(_url(fake_app, "test-connection"), json=_body(), headers=app_admin_headers)
                     .json()["message"])
        assert len(msgs) == 3

    def test_private_endpoint_rejected_unless_flag(self, client, fake_app, app_admin_headers, monkeypatch):
        _transport(_ok)
        body = _body(base_url="https://10.0.0.5:8000", api_key="")
        assert client.post(_url(fake_app, "test-connection"), json=body, headers=app_admin_headers).status_code == 422
        monkeypatch.setattr(config, "RERANK_ALLOW_PRIVATE_ENDPOINTS", True)
        r = client.post(_url(fake_app, "test-connection"), json=body, headers=app_admin_headers)
        assert r.status_code == 200 and r.json()["status"] == "success"


ADMIN = "/internal/admin/system-rerank-services"


class TestAdminRoutes:
    def test_non_admin_403(self, client, auth_headers):
        assert client.get(ADMIN, headers=auth_headers).status_code == 403
        assert client.post(ADMIN, json=_body(), headers=auth_headers).status_code == 403
        assert client.put(ADMIN + "/platform-default", json={"service_id": None}, headers=auth_headers).status_code == 403
        assert client.post(ADMIN + "/test-connection", json=_body(), headers=auth_headers).status_code == 403

    def test_app_admin_not_platform_admin_403(self, client, app_admin_headers):
        assert client.get(ADMIN, headers=app_admin_headers).status_code == 403

    def test_crud_and_visible_read_only_in_apps(self, client, fake_app, admin_headers, app_viewer_headers):
        r = client.post(ADMIN, json=_body(base_url="http://10.1.2.3:8000", api_key=""), headers=admin_headers)
        assert r.status_code == 201, r.text  # system services are exempt from the SSRF policy
        sid = r.json()["service_id"]
        assert r.json()["is_system"] is True

        listing = client.get(ADMIN, headers=admin_headers).json()
        assert [s["service_id"] for s in listing] == [sid]

        r = client.put(f"{ADMIN}/{sid}", json=_body(name="Renamed", api_key=SECRET, base_url="https://rr.example.com"),
                       headers=admin_headers)
        assert r.status_code == 200 and r.json()["name"] == "Renamed"
        r = client.put(f"{ADMIN}/{sid}", json=_body(name="Again", api_key="****abcd", base_url="https://rr.example.com"),
                       headers=admin_headers)
        assert client.get(f"{ADMIN}/{sid}", headers=admin_headers).json()["api_key"] == "****abcd"

        in_app = client.get(_url(fake_app), headers=app_viewer_headers).json()
        assert [(s["service_id"], s["is_system"]) for s in in_app] == [(sid, True)]

        assert client.delete(f"{ADMIN}/{sid}", headers=admin_headers).status_code == 204
        assert client.delete(f"{ADMIN}/{sid}", headers=admin_headers).status_code == 404
        assert client.put(f"{ADMIN}/{sid}", json=_body(), headers=admin_headers).status_code == 404

    def test_system_routes_do_not_touch_app_services(self, client, fake_app, db, admin_headers):
        row = _svc(db, fake_app.app_id)
        assert client.get(f"{ADMIN}/{row.service_id}", headers=admin_headers).status_code == 404
        assert client.put(f"{ADMIN}/{row.service_id}", json=_body(), headers=admin_headers).status_code == 404
        assert client.delete(f"{ADMIN}/{row.service_id}", headers=admin_headers).status_code == 404

    def test_system_update_masked_key_other_host_422(self, client, db, admin_headers):
        row = _svc(db, None)
        r = client.put(f"{ADMIN}/{row.service_id}",
                       json=_body(api_key="****abcd", base_url="https://evil.example.org"), headers=admin_headers)
        assert r.status_code == 422
        r = client.post(f"{ADMIN}/test-connection?service_id={row.service_id}",
                        json=_body(api_key="", base_url="https://evil.example.org"), headers=admin_headers)
        assert r.status_code == 422

    def test_system_test_connection_recovers_stored_key(self, client, db, admin_headers, fake_app):
        sys_row = _svc(db, None)
        app_row = _svc(db, fake_app.app_id)
        seen = {}

        def handler(request):
            seen["auth"] = request.headers.get("authorization")
            return _ok(request)

        _transport(handler)
        r = client.post(f"{ADMIN}/test-connection?service_id={sys_row.service_id}", json=_body(api_key="****abcd"),
                        headers=admin_headers)
        assert r.json()["status"] == "success" and seen["auth"] == f"Bearer {SECRET}"
        r = client.post(f"{ADMIN}/test-connection?service_id={app_row.service_id}", json=_body(api_key="****abcd"),
                        headers=admin_headers)
        assert r.status_code == 404


class TestPlatformDefault:
    def test_switch_and_clear(self, client, db, admin_headers):
        a, b = _svc(db, None, name="A"), _svc(db, None, name="B")
        url = ADMIN + "/platform-default"

        r = client.put(url, json={"service_id": a.service_id}, headers=admin_headers)
        assert r.status_code == 200
        flags = {s["name"]: s["is_platform_default"] for s in r.json()}
        assert flags == {"A": True, "B": False}

        flags = {s["name"]: s["is_platform_default"]
                 for s in client.put(url, json={"service_id": b.service_id}, headers=admin_headers).json()}
        assert flags == {"A": False, "B": True}

        flags = {s["name"]: s["is_platform_default"]
                 for s in client.put(url, json={"service_id": None}, headers=admin_headers).json()}
        assert flags == {"A": False, "B": False}

    def test_rejects_unknown_app_owned_and_retired(self, client, fake_app, db, admin_headers):
        url = ADMIN + "/platform-default"
        app_row = _svc(db, fake_app.app_id)
        retired = _svc(db, None, name="old", end_of_service_at=datetime.now(timezone.utc) - timedelta(days=1))
        assert client.put(url, json={"service_id": 999999}, headers=admin_headers).status_code == 404
        assert client.put(url, json={"service_id": app_row.service_id}, headers=admin_headers).status_code == 404
        assert client.put(url, json={"service_id": retired.service_id}, headers=admin_headers).status_code == 422

    def test_concurrent_marking_yields_one_winner(self, test_engine):
        from sqlalchemy.orm import Session

        from services.rerank_service_service import RerankServiceError, RerankServiceService

        with Session(test_engine) as setup:
            rows = [RerankService(name=n, provider="CohereCompatible", model_name="m", endpoint=BASE, app_id=None,
                                  create_date=datetime.now()) for n in ("C1", "C2")]
            setup.add_all(rows)
            setup.commit()
            ids = [r.service_id for r in rows]
        outcomes, barrier = [], threading.Barrier(2)

        def worker(service_id):
            with Session(test_engine) as s:
                barrier.wait()
                try:
                    RerankServiceService.set_platform_default(s, service_id, None)
                    outcomes.append("ok")
                except RerankServiceError as exc:
                    outcomes.append(exc.status_code)

        try:
            threads = [threading.Thread(target=worker, args=(i,)) for i in ids]
            [t.start() for t in threads]
            [t.join(timeout=30) for t in threads]
            with Session(test_engine) as check:
                flagged = [r.service_id for r in check.query(RerankService).filter(RerankService.is_platform_default)]
            assert len(flagged) == 1  # the partial unique index guarantees a single winner
            assert all(o in ("ok", 409) for o in outcomes) and "ok" in outcomes and len(outcomes) == 2
        finally:
            with Session(test_engine) as cleanup:
                cleanup.query(RerankService).filter(RerankService.service_id.in_(ids)).delete(synchronize_session=False)
                cleanup.commit()
