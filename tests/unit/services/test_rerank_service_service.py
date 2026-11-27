"""Unit tests for RerankServiceService (no DB: repositories are stubbed, transport is mocked)."""
from types import SimpleNamespace

import httpx
import pytest
from datetime import datetime, timedelta, timezone

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

import config
from schemas.rerank_service_schemas import CreateUpdateRerankServiceSchema
from services import rerank_service_service as mod
from services.rerank_service_service import RerankServiceError, RerankServiceService
from tools.rerank import http_clients

PUBLIC_IP = "93.184.216.34"
SECRET = "sk-super-secret-1234"  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def _dns(monkeypatch):
    monkeypatch.setattr("utils.ssrf_guard.resolve_host", lambda host: [PUBLIC_IP])
    monkeypatch.setattr(config, "RERANK_ALLOW_PRIVATE_ENDPOINTS", False)


def _payload(**over):
    data = dict(name="R", provider="CohereCompatible", model_name="bge", api_key="", base_url="https://rr.example.com/")
    data.update(over)
    return CreateUpdateRerankServiceSchema(**data)


def _row(**over):
    base = dict(
        service_id=7, app_id=1, name="R", provider="CohereCompatible", model_name="bge", endpoint="https://rr.example.com",
        api_key=SECRET, create_date=None, is_platform_default=False, last_error=None, last_error_at=None,
        end_of_service_at=None, deprecation_notice=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _install(monkeypatch, handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    http_clients._set_clients_for_tests(client)
    return client


@pytest.fixture(autouse=True)
def _reset_client():
    yield
    http_clients._set_clients_for_tests(None)


class TestSchema:
    def test_normalises_base_url(self):
        assert _payload(base_url="https://rr.example.com/v1/rerank/").base_url == "https://rr.example.com/v1"

    @pytest.mark.parametrize("over", [
        {"provider": "Nope"}, {"name": ""}, {"model_name": "   "}, {"base_url": "ftp://x"},
        {"base_url": "https://u:p@x.com"}, {"name": "x" * 101}, {"model_name": "m" * 256},
    ])
    def test_rejects_invalid(self, over):
        with pytest.raises(ValidationError):
            _payload(**over)

    def test_api_key_optional(self):
        assert _payload(api_key=None).api_key is None


class TestMapping:
    def test_needs_api_key_only_for_placeholder(self):
        assert RerankServiceService.to_list_item(_row(api_key="CHANGE_ME"), warning_days=30).needs_api_key
        assert not RerankServiceService.to_list_item(_row(api_key=""), warning_days=30).needs_api_key
        assert not RerankServiceService.to_list_item(_row(api_key=None), warning_days=30).needs_api_key

    def test_detail_masks_key(self):
        detail = RerankServiceService.to_detail(_row(), 30)
        assert detail.api_key == "****1234"
        assert SECRET not in detail.model_dump_json()

    def test_is_system_from_app_id(self):
        assert RerankServiceService.to_list_item(_row(app_id=None), warning_days=30).is_system


class TestValidateEndpoint:
    def test_private_rejected_for_app(self):
        with pytest.raises(RerankServiceError) as e:
            RerankServiceService.validate_endpoint("https://10.0.0.5:8000", None, app_scoped=True)
        assert e.value.status_code == 422

    def test_private_allowed_for_system(self):
        RerankServiceService.validate_endpoint("http://10.0.0.5:8000", None, app_scoped=False)

    def test_private_allowed_with_env_flag(self, monkeypatch):
        monkeypatch.setattr(config, "RERANK_ALLOW_PRIVATE_ENDPOINTS", True)
        RerankServiceService.validate_endpoint("http://10.0.0.5:8000", SECRET, app_scoped=True)

    def test_key_over_plain_http_to_public_host_rejected(self):
        with pytest.raises(RerankServiceError):
            RerankServiceService.validate_endpoint("http://rr.example.com", SECRET, app_scoped=True)
        with pytest.raises(RerankServiceError):
            RerankServiceService.validate_endpoint("http://rr.example.com", SECRET, app_scoped=False)

    def test_http_without_key_or_masked_or_placeholder_ok(self):
        for key in (None, "", "CHANGE_ME", "****1234"):
            RerankServiceService.validate_endpoint("http://rr.example.com", key, app_scoped=True)

    def test_key_over_http_to_private_host_ok_for_system(self):
        RerankServiceService.validate_endpoint("http://10.0.0.5:8000", SECRET, app_scoped=False)


def _ok(request):
    return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.1}]})


class TestTestConnection:
    async def _run(self, monkeypatch, handler, payload=None, *, app_scoped=True, service_id=None, stored=None):
        _install(monkeypatch, handler)
        monkeypatch.setattr(mod.RerankServiceRepository, "get_by_id_and_app_id", lambda db, sid, aid: stored)
        monkeypatch.setattr(mod.RerankServiceRepository, "get_by_id", lambda db, sid: stored)
        return await RerankServiceService.test_connection_with_config(
            None, payload or _payload(), app_scoped=app_scoped, app_id=1 if app_scoped else None, service_id=service_id
        )

    async def test_success(self, monkeypatch):
        result = await self._run(monkeypatch, _ok)
        assert result["status"] == "success"

    async def test_failures_have_distinct_messages_and_never_echo_key(self, monkeypatch):
        def timeout(request):
            raise httpx.ReadTimeout("slow", request=request)

        cases = {
            "401": lambda r: httpx.Response(401, json={"error": SECRET}),
            "500": lambda r: httpx.Response(500, text=SECRET),
            "timeout": timeout,
            "malformed": lambda r: httpx.Response(200, text="not json " + SECRET),
            "shape": lambda r: httpx.Response(200, json={"nope": 1}),
        }
        messages = {}
        for name, handler in cases.items():
            result = await self._run(monkeypatch, handler, _payload(api_key="sk-typed"))
            assert result["status"] == "error"
            assert SECRET not in str(result) and "sk-typed" not in str(result)
            messages[name] = result["message"]
        assert len(set(messages.values())) == len(messages)

    async def test_stored_key_recovered_for_masked_input(self, monkeypatch):
        seen = {}

        def handler(request):
            seen["auth"] = request.headers.get("authorization")
            return _ok(request)

        for masked in ("****1234", "", "CHANGE_ME", None):
            seen.clear()
            result = await self._run(
                monkeypatch, handler, _payload(api_key=masked, base_url="https://rr.example.com"),
                service_id=7, stored=_row(),
            )
            assert result["status"] == "success"
            assert seen["auth"] == f"Bearer {SECRET}"

    async def test_typed_key_is_used_over_stored(self, monkeypatch):
        seen = {}

        def handler(request):
            seen["auth"] = request.headers.get("authorization")
            return _ok(request)

        await self._run(monkeypatch, handler, _payload(api_key="sk-new"), service_id=7, stored=_row())
        assert seen["auth"] == "Bearer sk-new"

    async def test_unknown_service_in_scope_is_404(self, monkeypatch):
        with pytest.raises(RerankServiceError) as e:
            await self._run(monkeypatch, _ok, service_id=7, stored=None)
        assert e.value.status_code == 404

    async def test_system_scope_ignores_app_owned_row(self, monkeypatch):
        with pytest.raises(RerankServiceError) as e:
            await self._run(monkeypatch, _ok, app_scoped=False, service_id=7, stored=_row(app_id=1))
        assert e.value.status_code == 404

    async def test_private_endpoint_rejected_for_app_scope(self, monkeypatch):
        with pytest.raises(RerankServiceError) as e:
            await self._run(monkeypatch, _ok, _payload(base_url="https://10.0.0.5:8000"))
        assert e.value.status_code == 422

    async def test_private_endpoint_allowed_with_env_flag(self, monkeypatch):
        monkeypatch.setattr(config, "RERANK_ALLOW_PRIVATE_ENDPOINTS", True)
        result = await self._run(monkeypatch, _ok, _payload(base_url="https://10.0.0.5:8000"))
        assert result["status"] == "success"


class TestKeyReuseOnHostChange:
    def test_same_origin_ok_and_default_ports_equal(self):
        from services.rerank_service_service import _check_key_reuse
        _check_key_reuse(_row(endpoint="https://rr.example.com"), "https://RR.example.com:443")

    @pytest.mark.parametrize("new", ["https://evil.example.org", "http://rr.example.com", "https://rr.example.com:8443"])
    def test_other_origin_rejected(self, new):
        from services.rerank_service_service import _check_key_reuse
        with pytest.raises(RerankServiceError) as e:
            _check_key_reuse(_row(), new)
        assert e.value.status_code == 422 and "Re-enter the API key" in e.value.message

    def test_no_stored_key_nothing_to_leak(self):
        from services.rerank_service_service import _check_key_reuse
        _check_key_reuse(_row(api_key=""), "https://evil.example.org")

    async def test_test_connection_other_host_422_and_nothing_sent(self, monkeypatch):
        calls = []
        with pytest.raises(RerankServiceError) as e:
            await TestTestConnection()._run(
                monkeypatch, lambda r: calls.append(r) or _ok(r),
                _payload(api_key="****1234", base_url="https://evil.example.org"), service_id=7, stored=_row(),
            )
        assert e.value.status_code == 422 and not calls

    async def test_test_connection_other_host_ok_with_typed_key(self, monkeypatch):
        result = await TestTestConnection()._run(
            monkeypatch, _ok, _payload(api_key="sk-new", base_url="https://other.example.org"),
            service_id=7, stored=_row(),
        )
        assert result["status"] == "success"

    def test_update_masked_key_other_host_rejected(self, monkeypatch):
        row = _row()
        monkeypatch.setattr(mod.RerankServiceRepository, "get_by_id_and_app_id", lambda db, sid, aid: row)
        monkeypatch.setattr(mod.RerankServiceRepository, "get_by_id", lambda db, sid: _row(app_id=None))
        with pytest.raises(RerankServiceError) as e:
            RerankServiceService.create_or_update_rerank_service(
                None, 1, 7, _payload(api_key="****1234", base_url="https://evil.example.org"))
        assert e.value.status_code == 422
        with pytest.raises(RerankServiceError):
            RerankServiceService.update_system_service(
                None, 7, _payload(api_key="****1234", base_url="https://evil.example.org"))
        assert row.api_key == SECRET


class TestDeleteGuard:
    def test_integrity_error_becomes_409(self, monkeypatch):
        db = SimpleNamespace(rollback=lambda: setattr(db, "rolled_back", True))
        scope = {"app_id": 1}
        monkeypatch.setattr(mod.RerankServiceRepository, "get_for_update", lambda d, s: _row(app_id=scope["app_id"]))
        monkeypatch.setattr(mod.ServiceLifecycleService, "get_usage", staticmethod(lambda *a, **k: None))

        def boom(d, s):
            raise IntegrityError("x", {}, Exception("fk"))

        monkeypatch.setattr(mod.RerankServiceRepository, "delete", boom)
        for app_id, call in ((1, lambda: RerankServiceService.delete_rerank_service(db, 1, 7)),
                             (None, lambda: RerankServiceService.delete_system_service(db, 7))):
            scope["app_id"] = app_id
            with pytest.raises(RerankServiceError) as e:
                call()
            assert e.value.status_code == 409 and e.value.message == "Rerank service is in use"
        assert db.rolled_back


class TestInUseError:
    def test_is_a_409_service_error_with_usage_detail(self):
        from schemas.service_lifecycle_schemas import ServiceUsageSchema

        usage = ServiceUsageSchema(service_id=1, service_name="s", kind="rerank", is_system=False,
                                   lifecycle_status="active")
        err = mod.RerankServiceInUseError(usage)
        assert isinstance(err, RerankServiceError) and err.status_code == 409
        assert err.detail()["usage"]["service_id"] == 1 and err.detail()["message"]

    def test_integrity_error_on_delete_recomputes_usage(self, monkeypatch):
        from schemas.service_lifecycle_schemas import ServiceUsageSchema

        usage = ServiceUsageSchema(service_id=7, service_name="s", kind="rerank", is_system=False,
                                   lifecycle_status="active")
        db = SimpleNamespace(rollback=lambda: None)
        monkeypatch.setattr(mod.RerankServiceRepository, "delete",
                            lambda d, s: (_ for _ in ()).throw(IntegrityError("x", {}, Exception("fk"))))
        monkeypatch.setattr(mod.ServiceLifecycleService, "get_usage", staticmethod(lambda *a, **k: usage))
        with pytest.raises(mod.RerankServiceInUseError) as e:
            mod.RerankServiceService._delete_row(db, _row())
        assert e.value.detail()["usage"]["service_id"] == 7


class TestEffectiveDefault:
    def _options(self, monkeypatch, app_default_row, platform_row):
        rows = [r for r in (app_default_row, platform_row) if r is not None]
        monkeypatch.setattr(mod.RerankServiceRepository, "get_by_app_id", lambda d, a: [r for r in rows if r.app_id])
        monkeypatch.setattr(mod.RerankServiceRepository, "get_system_services", lambda d: [r for r in rows if not r.app_id])
        monkeypatch.setattr(mod.ServiceLifecycleService, "warning_days", staticmethod(lambda d: 30))
        app = SimpleNamespace(default_rerank_service_id=app_default_row.service_id if app_default_row else None)
        monkeypatch.setattr(mod, "AppRepository", lambda d: SimpleNamespace(get_by_id=lambda a: app))
        return RerankServiceService.options(None, 1)

    def test_retired_app_default_skips_without_platform_fallback(self, monkeypatch):
        past = datetime.now(timezone.utc) - timedelta(days=1)
        res = self._options(monkeypatch, _row(service_id=1, end_of_service_at=past),
                            _row(service_id=2, app_id=None, is_platform_default=True))
        assert res.app_default.service_id == 1 and res.platform_default.service_id == 2
        assert res.effective_default is None

    def test_active_app_default_wins_and_platform_used_otherwise(self, monkeypatch):
        res = self._options(monkeypatch, _row(service_id=1), _row(service_id=2, app_id=None, is_platform_default=True))
        assert res.effective_default.service_id == 1
        res = self._options(monkeypatch, None, _row(service_id=2, app_id=None, is_platform_default=True))
        assert res.effective_default.service_id == 2

    def test_retired_platform_default_is_none(self, monkeypatch):
        past = datetime.now(timezone.utc) - timedelta(days=1)
        res = self._options(monkeypatch, None, _row(service_id=2, app_id=None, is_platform_default=True,
                                                    end_of_service_at=past))
        assert res.effective_default is None
