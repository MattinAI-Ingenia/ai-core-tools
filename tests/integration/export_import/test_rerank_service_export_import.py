"""Integration tests for Rerank Service export/import (single component and full app)."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import Session

from core.export_constants import PLACEHOLDER_API_KEY, get_component_type_from_file
from models.app import App
from models.rerank_service import RerankService
from models.user import User
from schemas.export_schemas import AppExportFileSchema, RerankServiceExportFileSchema
from schemas.import_schemas import ConflictMode
from services.full_app_export_service import FullAppExportService
from services.full_app_import_service import FullAppImportService
from services.rerank_service_export_service import RerankServiceExportService
from services.rerank_service_import_service import RerankServiceImportService

REAL_KEY = "sk-real-secret-key-123"  # pragma: allowlist secret
ENDPOINT = "https://rerank.example.com/v1"


@pytest.fixture
def setup(db_session: Session):
    ts = datetime.now().timestamp()
    owner = User(email=f"rr-owner-{ts}@test.com", name="Owner")
    db_session.add(owner)
    db_session.flush()
    app = App(name=f"RR Source {ts}", slug=f"rr-source-{ts}", owner_id=owner.user_id)
    db_session.add(app)
    db_session.flush()
    svc = RerankService(
        app_id=app.app_id, name=f"rr-{ts}", provider="CohereCompatible", model_name="rerank-x",
        endpoint=ENDPOINT, api_key=REAL_KEY, is_platform_default=False,
    )
    db_session.add(svc)
    db_session.flush()
    app.default_rerank_service_id = svc.service_id
    db_session.commit()
    created_apps = [app.app_id]
    yield {"owner": owner, "app": app, "svc": svc, "created_apps": created_apps, "ts": ts}
    db_session.rollback()
    for app_id in created_apps:
        a = db_session.get(App, app_id)
        if a is not None:
            a.default_rerank_service_id = None
            db_session.commit()
        db_session.query(RerankService).filter(RerankService.app_id == app_id).delete()
        a = db_session.get(App, app_id)
        if a is not None:
            db_session.delete(a)
        db_session.commit()
    db_session.query(RerankService).filter(RerankService.name.like("rr-sys-%")).delete()
    db_session.query(User).filter(User.user_id == owner.user_id).delete()
    db_session.commit()


def _import_full(db_session, export, owner, setup, **kw):
    result = FullAppImportService(db_session).import_full_app(
        export, owner.user_id, ConflictMode.RENAME, **kw
    )
    setup["created_apps"].append(result.app_id)
    return result


class TestSingleExportImport:
    def test_export_has_null_api_key(self, db_session, setup):
        export = RerankServiceExportService(db_session).export_rerank_service(
            setup["svc"].service_id, setup["app"].app_id
        )
        data = export.model_dump(mode="json")
        assert data["rerank_service"]["api_key"] is None
        assert REAL_KEY not in json.dumps(data)
        assert get_component_type_from_file(data) == "rerank_service"

    def test_export_other_tenant_not_found(self, db_session, setup):
        with pytest.raises(ValueError):
            RerankServiceExportService(db_session).export_rerank_service(setup["svc"].service_id, 0)

    def test_import_creates_placeholder_key(self, db_session, setup):
        export = RerankServiceExportService(db_session).export_rerank_service(
            setup["svc"].service_id, setup["app"].app_id
        )
        summary = RerankServiceImportService(db_session).import_rerank_service(
            export, setup["app"].app_id, ConflictMode.RENAME
        )
        created = db_session.get(RerankService, summary.component_id)
        assert created.api_key == PLACEHOLDER_API_KEY
        assert created.name != setup["svc"].name
        assert created.endpoint == ENDPOINT
        assert created.is_platform_default is False

    def test_import_conflict_fail(self, db_session, setup):
        export = RerankServiceExportService(db_session).export_rerank_service(
            setup["svc"].service_id, setup["app"].app_id
        )
        with pytest.raises(ValueError, match="already exists"):
            RerankServiceImportService(db_session).import_rerank_service(
                export, setup["app"].app_id, ConflictMode.FAIL
            )

    def test_import_rejects_private_endpoint(self, db_session, setup, monkeypatch):
        import config

        monkeypatch.setattr(config, "RERANK_ALLOW_PRIVATE_ENDPOINTS", False)
        export = RerankServiceExportService(db_session).export_rerank_service(
            setup["svc"].service_id, setup["app"].app_id
        )
        export.rerank_service.endpoint = "http://127.0.0.1:9000"
        export.rerank_service.name = "private-one"
        with pytest.raises(ValueError, match="private or internal"):
            RerankServiceImportService(db_session).import_rerank_service(
                export, setup["app"].app_id, ConflictMode.FAIL
            )


class TestOverrideAndRename:
    def _export(self, db_session, setup):
        return RerankServiceExportService(db_session).export_rerank_service(
            setup["svc"].service_id, setup["app"].app_id
        )

    def test_override_same_host_updates_fields_and_keeps_key(self, db_session, setup):
        export = self._export(db_session, setup)
        export.rerank_service.model_name = "rerank-y"
        export.rerank_service.endpoint = "https://rerank.example.com/other/rerank"
        export.rerank_service.deprecation_notice = "moving on"
        export.rerank_service.end_of_service_at = datetime(2099, 1, 1, tzinfo=timezone.utc)
        summary = RerankServiceImportService(db_session).import_rerank_service(
            export, setup["app"].app_id, ConflictMode.OVERRIDE
        )
        svc = db_session.get(RerankService, summary.component_id)
        assert summary.created is False
        assert svc.model_name == "rerank-y"
        assert svc.endpoint == "https://rerank.example.com/other"
        assert svc.deprecation_notice == "moving on"
        assert svc.end_of_service_at is not None
        assert svc.api_key == REAL_KEY

    def test_override_new_host_is_rejected_and_key_untouched(self, db_session, setup):
        export = self._export(db_session, setup)
        export.rerank_service.endpoint = "https://attacker.example.net/v1"
        with pytest.raises(ValueError, match="Re-enter the API key"):
            RerankServiceImportService(db_session).import_rerank_service(
                export, setup["app"].app_id, ConflictMode.OVERRIDE
            )
        db_session.rollback()
        svc = db_session.get(RerankService, setup["svc"].service_id)
        assert svc.endpoint == ENDPOINT
        assert svc.api_key == REAL_KEY

    def test_rename_with_new_name(self, db_session, setup):
        export = self._export(db_session, setup)
        summary = RerankServiceImportService(db_session).import_rerank_service(
            export, setup["app"].app_id, ConflictMode.RENAME, new_name="custom-name"
        )
        assert summary.component_name == "custom-name"
        assert db_session.get(RerankService, summary.component_id).api_key == PLACEHOLDER_API_KEY

    def test_fail_conflict_raises_typed_error(self, db_session, setup):
        from services.rerank_service_import_service import RerankServiceConflictError

        with pytest.raises(RerankServiceConflictError):
            RerankServiceImportService(db_session).import_rerank_service(
                self._export(db_session, setup), setup["app"].app_id, ConflictMode.FAIL
            )


class TestFullApp:
    def _export(self, db_session, setup) -> AppExportFileSchema:
        return FullAppExportService(db_session).export_full_app(setup["app"].app_id)

    def test_full_export_has_rerank_services_without_keys(self, db_session, setup):
        export = self._export(db_session, setup)
        assert [s.name for s in export.rerank_services] == [setup["svc"].name]
        assert export.rerank_services[0].api_key is None
        assert export.app.default_rerank_service_name == setup["svc"].name
        assert export.app.default_rerank_service_is_system is False
        assert REAL_KEY not in export.model_dump_json()

    def test_full_import_remaps_default(self, db_session, setup):
        export = self._export(db_session, setup)
        result = _import_full(db_session, export, setup["owner"], setup)
        assert not result.total_errors
        new_app = db_session.get(App, result.app_id)
        new_svc = db_session.query(RerankService).filter(RerankService.app_id == result.app_id).one()
        assert new_svc.service_id != setup["svc"].service_id
        assert new_svc.api_key == PLACEHOLDER_API_KEY
        assert new_app.default_rerank_service_id == new_svc.service_id

    def test_full_import_applies_supplied_api_key(self, db_session, setup):
        export = self._export(db_session, setup)
        result = _import_full(db_session, export, setup["owner"], setup, api_keys={setup["svc"].name: "user-key"})
        new_svc = db_session.query(RerankService).filter(RerankService.app_id == result.app_id).one()
        assert new_svc.api_key == "user-key"

    def test_system_default_absent_on_target_is_cleared_with_warning(self, db_session, setup):
        export = self._export(db_session, setup)
        export.rerank_services = []
        export.app.default_rerank_service_name = "rr-sys-missing"
        export.app.default_rerank_service_is_system = True
        result = _import_full(db_session, export, setup["owner"], setup)
        assert not result.total_errors
        assert db_session.get(App, result.app_id).default_rerank_service_id is None
        assert any("rr-sys-missing" in w for w in result.total_warnings)

    def test_system_default_present_on_target_is_linked(self, db_session, setup):
        system = RerankService(
            app_id=None, name=f"rr-sys-{setup['ts']}", provider="CohereCompatible", model_name="m",
            endpoint=ENDPOINT, api_key="k", is_platform_default=False,
        )
        db_session.add(system)
        db_session.commit()
        export = self._export(db_session, setup)
        export.rerank_services = []
        export.app.default_rerank_service_name = system.name
        export.app.default_rerank_service_is_system = True
        result = _import_full(db_session, export, setup["owner"], setup)
        assert db_session.get(App, result.app_id).default_rerank_service_id == system.service_id

    def test_old_export_without_rerank_services_still_imports(self, db_session, setup):
        data = self._export(db_session, setup).model_dump(mode="json")
        del data["rerank_services"]
        del data["app"]["default_rerank_service_name"]
        del data["app"]["default_rerank_service_is_system"]
        old = AppExportFileSchema(**data)
        assert old.rerank_services == []
        result = _import_full(db_session, old, setup["owner"], setup)
        assert not result.total_errors
        assert db_session.get(App, result.app_id).default_rerank_service_id is None

    def test_retired_system_default_is_not_linked(self, db_session, setup):
        system = RerankService(
            app_id=None, name=f"rr-sys-ret-{setup['ts']}", provider="CohereCompatible", model_name="m",
            endpoint=ENDPOINT, api_key="k", is_platform_default=False,
            end_of_service_at=datetime.now(timezone.utc) - timedelta(days=5),
        )
        db_session.add(system)
        db_session.commit()
        export = self._export(db_session, setup)
        export.rerank_services = []
        export.app.default_rerank_service_name = system.name
        export.app.default_rerank_service_is_system = True
        result = _import_full(db_session, export, setup["owner"], setup)
        assert db_session.get(App, result.app_id).default_rerank_service_id is None
        assert any("retired" in w for w in result.total_warnings)

    def test_deselected_rerank_services_leave_default_empty_with_warning(self, db_session, setup):
        export = self._export(db_session, setup)
        result = _import_full(db_session, export, setup["owner"], setup, component_selection={})
        assert not result.total_errors
        assert db_session.get(App, result.app_id).default_rerank_service_id is None
        assert any("was not imported" in w for w in result.total_warnings)

    def test_http_public_endpoint_with_supplied_key_is_rejected(self, db_session, setup):
        export = self._export(db_session, setup)
        export.rerank_services[0].endpoint = "http://rerank.example.com/v1"
        result = _import_full(db_session, export, setup["owner"], setup, api_keys={setup["svc"].name: "user-key"})
        assert result.total_errors
        assert "plain http" in result.total_errors[0]
        assert db_session.query(RerankService).filter(RerankService.app_id == result.app_id).count() == 0
