"""Unit tests for `AppService._validate_memory_service_ids` (AC-38).

Regression coverage for the security-audit MEDIUM finding on step_009 (App-level
memory service configuration): no test previously exercised this validator directly.
Pure-Python — the DB session is a `MagicMock`; every repository lookup is monkeypatched.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from repositories.ai_service_repository import AIServiceRepository
from repositories.embedding_service_repository import EmbeddingServiceRepository
from services.app_service import AppService, MemoryServiceConfigError
import tools.embeddingTools as embedding_tools_module
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

        service._validate_memory_service_ids(
            app_id=THIS_APP_ID, extraction_service_id=None, embedding_service_id=42
        )  # must not raise


class TestEmbeddingServiceWidth:
    def test_any_width_is_accepted_without_contacting_the_provider(self, service, monkeypatch):
        """The memory store is dimension-agnostic, so validation is a local DB
        lookup only — it must never build (let alone call) an embeddings client."""
        monkeypatch.setattr(
            EmbeddingServiceRepository, "get_by_id", staticmethod(lambda db, sid: _service(sid, THIS_APP_ID))
        )
        build = MagicMock(side_effect=AssertionError("validation must not build an embeddings client"))
        monkeypatch.setattr(embedding_tools_module, "get_embeddings_model", build)
        monkeypatch.setattr(resolution_module, "get_embeddings_model", build)

        service._validate_memory_service_ids(
            app_id=THIS_APP_ID, extraction_service_id=None, embedding_service_id=42
        )  # must not raise

        build.assert_not_called()

