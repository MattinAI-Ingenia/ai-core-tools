"""Unit tests for `AppService._validate_memory_service_ids` (AC-38, AC-46).

Regression coverage for the security-audit MEDIUM finding on step_009 (App-level
memory service configuration): no test previously exercised this validator directly.
Pure-Python — the DB session is a `MagicMock`; every repository lookup is monkeypatched.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import config
from repositories.ai_service_repository import AIServiceRepository
from repositories.embedding_service_repository import EmbeddingServiceRepository
from services.app_service import AppService, MemoryServiceConfigError
import tools.memory.resolution as resolution_module


THIS_APP_ID = 1
OTHER_APP_ID = 2


def _service(service_id: int, app_id):
    return SimpleNamespace(service_id=service_id, app_id=app_id)


@pytest.fixture
def service():
    return AppService(MagicMock())


class TestExtractionServiceId:
    def test_id_belonging_to_another_app_is_rejected(self, service, monkeypatch):
        monkeypatch.setattr(
            AIServiceRepository, "get_by_id", staticmethod(lambda db, sid: _service(sid, OTHER_APP_ID))
        )

        with pytest.raises(MemoryServiceConfigError):
            service._validate_memory_service_ids(
                app_id=THIS_APP_ID, extraction_service_id=99, embedding_service_id=None
            )

    def test_nonexistent_id_is_rejected(self, service, monkeypatch):
        monkeypatch.setattr(AIServiceRepository, "get_by_id", staticmethod(lambda db, sid: None))

        with pytest.raises(MemoryServiceConfigError):
            service._validate_memory_service_ids(
                app_id=THIS_APP_ID, extraction_service_id=0, embedding_service_id=None
            )

    def test_system_level_id_is_accepted(self, service, monkeypatch):
        monkeypatch.setattr(
            AIServiceRepository, "get_by_id", staticmethod(lambda db, sid: _service(sid, None))
        )

        service._validate_memory_service_ids(
            app_id=THIS_APP_ID, extraction_service_id=7, embedding_service_id=None
        )  # must not raise

    def test_own_app_id_is_accepted(self, service, monkeypatch):
        monkeypatch.setattr(
            AIServiceRepository, "get_by_id", staticmethod(lambda db, sid: _service(sid, THIS_APP_ID))
        )

        service._validate_memory_service_ids(
            app_id=THIS_APP_ID, extraction_service_id=7, embedding_service_id=None
        )  # must not raise


class TestEmbeddingServiceIdOwnership:
    def test_id_belonging_to_another_app_is_rejected(self, service, monkeypatch):
        monkeypatch.setattr(
            EmbeddingServiceRepository, "get_by_id", staticmethod(lambda db, sid: _service(sid, OTHER_APP_ID))
        )

        with pytest.raises(MemoryServiceConfigError):
            service._validate_memory_service_ids(
                app_id=THIS_APP_ID, extraction_service_id=None, embedding_service_id=42
            )

    def test_nonexistent_id_is_rejected(self, service, monkeypatch):
        monkeypatch.setattr(EmbeddingServiceRepository, "get_by_id", staticmethod(lambda db, sid: None))

        with pytest.raises(MemoryServiceConfigError):
            service._validate_memory_service_ids(
                app_id=THIS_APP_ID, extraction_service_id=None, embedding_service_id=0
            )

    def test_system_level_id_is_accepted(self, service, monkeypatch):
        monkeypatch.setattr(
            EmbeddingServiceRepository, "get_by_id", staticmethod(lambda db, sid: _service(sid, None))
        )
        monkeypatch.setattr(
            resolution_module, "probe_embedding_dims_for_config", lambda svc: config.GLOBAL_MEMORY_EMBEDDING_DIMS
        )

        service._validate_memory_service_ids(
            app_id=THIS_APP_ID, extraction_service_id=None, embedding_service_id=42
        )  # must not raise


class TestEmbeddingServiceDimensionality:
    def test_mismatched_width_is_rejected(self, service, monkeypatch):
        monkeypatch.setattr(
            EmbeddingServiceRepository, "get_by_id", staticmethod(lambda db, sid: _service(sid, THIS_APP_ID))
        )
        monkeypatch.setattr(
            resolution_module,
            "probe_embedding_dims_for_config",
            lambda svc: config.GLOBAL_MEMORY_EMBEDDING_DIMS + 1,
        )

        with pytest.raises(MemoryServiceConfigError):
            service._validate_memory_service_ids(
                app_id=THIS_APP_ID, extraction_service_id=None, embedding_service_id=42
            )

    def test_matching_width_is_accepted(self, service, monkeypatch):
        monkeypatch.setattr(
            EmbeddingServiceRepository, "get_by_id", staticmethod(lambda db, sid: _service(sid, THIS_APP_ID))
        )
        monkeypatch.setattr(
            resolution_module, "probe_embedding_dims_for_config", lambda svc: config.GLOBAL_MEMORY_EMBEDDING_DIMS
        )

        service._validate_memory_service_ids(
            app_id=THIS_APP_ID, extraction_service_id=None, embedding_service_id=42
        )  # must not raise

    def test_undeterminable_width_is_accepted(self, service, monkeypatch):
        """A probe that cannot determine the width (`None`) must not block
        configuration — the runtime guard (`build_memory_clients`) is the backstop
        (AD-12)."""
        monkeypatch.setattr(
            EmbeddingServiceRepository, "get_by_id", staticmethod(lambda db, sid: _service(sid, THIS_APP_ID))
        )
        monkeypatch.setattr(resolution_module, "probe_embedding_dims_for_config", lambda svc: None)

        service._validate_memory_service_ids(
            app_id=THIS_APP_ID, extraction_service_id=None, embedding_service_id=42
        )  # must not raise


class TestSkipsProbeWhenEmbeddingIdUnchanged:
    def test_probe_is_not_called_when_id_matches_current_value(self, service, monkeypatch):
        monkeypatch.setattr(
            EmbeddingServiceRepository, "get_by_id", staticmethod(lambda db, sid: _service(sid, THIS_APP_ID))
        )
        probe = MagicMock()
        monkeypatch.setattr(resolution_module, "probe_embedding_dims_for_config", probe)

        service._validate_memory_service_ids(
            app_id=THIS_APP_ID,
            extraction_service_id=None,
            embedding_service_id=42,
            current_embedding_service_id=42,
        )

        probe.assert_not_called()

    def test_probe_still_runs_when_id_actually_changes(self, service, monkeypatch):
        monkeypatch.setattr(
            EmbeddingServiceRepository, "get_by_id", staticmethod(lambda db, sid: _service(sid, THIS_APP_ID))
        )
        probe = MagicMock(return_value=config.GLOBAL_MEMORY_EMBEDDING_DIMS)
        monkeypatch.setattr(resolution_module, "probe_embedding_dims_for_config", probe)

        service._validate_memory_service_ids(
            app_id=THIS_APP_ID,
            extraction_service_id=None,
            embedding_service_id=42,
            current_embedding_service_id=99,
        )

        probe.assert_called_once()
