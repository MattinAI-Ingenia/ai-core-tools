"""AC-9: SiloService.wipe_lightrag_workspace resets derived state after a strict wipe.

Real DB through the ``db`` fixture; the vector store is mocked at services.silo_service._get_vector_store.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from models.embedding_service import EmbeddingService
from models.entity_name_variant import EntityNameVariant
from models.silo import Silo
from services.silo_service import SiloService
from tools.vector_stores import lightrag_store
from utils.service_lifecycle import ServiceRetiredError

_INVALIDATE = "services.metadata_values_cache_service.MetadataValuesCacheService.invalidate"


def _silo(db, app, name):
    emb = EmbeddingService(
        name=f"{name} Emb", provider="OpenAI", api_key="sk-test-fake",  # pragma: allowlist secret
        app_id=app.app_id,
    )
    db.add(emb)
    db.flush()
    silo = Silo(
        name=name, description="d", silo_type="REPO", status="active", app_id=app.app_id,
        vector_db_type="LIGHTRAG", embedding_service_id=emb.service_id,
    )
    db.add(silo)
    db.flush()
    db.add(EntityNameVariant(silo_id=silo.silo_id, canonical_name="mcf 40", variant="MCF-40", mention_count=2))
    db.flush()
    return silo


def _variants(db, silo_id):
    db.expire_all()
    return db.query(EntityNameVariant).filter(EntityNameVariant.silo_id == silo_id).count()


@pytest.fixture
def silos(db, fake_app):
    return _silo(db, fake_app, "S"), _silo(db, fake_app, "T")


@pytest.fixture
def session_proxy(db):
    """Make the service's own SessionLocal() use the test session (close is a no-op). Yields the proxy."""
    proxy = MagicMock(wraps=db)
    proxy.close = MagicMock()
    with patch("services.silo_service.SessionLocal", return_value=proxy):
        yield proxy


def _seed_variant_cache(monkeypatch, silo_id):
    monkeypatch.setitem(lightrag_store._VARIANT_NAME_CACHE, silo_id, (0.0, {"mcf 40": "MCF-40"}))


def test_wipe_resets_variants_and_caches(db, silos, session_proxy, monkeypatch):
    s, t = silos
    _seed_variant_cache(monkeypatch, s.silo_id)
    store = MagicMock()
    with patch("services.silo_service._get_vector_store", return_value=store), \
            patch(_INVALIDATE) as invalidate:
        SiloService.wipe_lightrag_workspace(s)

    store.wipe_workspace.assert_called_once_with(f"silo_{s.silo_id}")
    assert _variants(db, s.silo_id) == 0
    assert _variants(db, t.silo_id) == 1
    assert s.silo_id not in lightrag_store._VARIANT_NAME_CACHE
    invalidate.assert_called_once_with(s.silo_id)
    session_proxy.close.assert_called_once()


def test_failed_wipe_raises_and_keeps_variants(db, silos, session_proxy, monkeypatch):
    s, _ = silos
    _seed_variant_cache(monkeypatch, s.silo_id)
    store = MagicMock()
    store.wipe_workspace.side_effect = RuntimeError("boom")
    with patch("services.silo_service._get_vector_store", return_value=store), \
            patch(_INVALIDATE) as invalidate, \
            patch("services.silo_service.logger") as log, pytest.raises(RuntimeError):
        SiloService.wipe_lightrag_workspace(s)

    assert _variants(db, s.silo_id) == 1
    assert log.error.call_args.kwargs["exc_info"] is True
    invalidate.assert_not_called()
    assert s.silo_id in lightrag_store._VARIANT_NAME_CACHE
    session_proxy.close.assert_not_called()


def test_retired_service_propagates_without_error_log(db, silos, session_proxy, monkeypatch):
    s, _ = silos
    _seed_variant_cache(monkeypatch, s.silo_id)
    retired = ServiceRetiredError(SimpleNamespace(name="llm", end_of_service_at=datetime(2026, 1, 1)), "ai")
    with patch("services.silo_service._get_vector_store", side_effect=retired), \
            patch(_INVALIDATE) as invalidate, \
            patch("services.silo_service.logger") as log, pytest.raises(ServiceRetiredError):
        SiloService.wipe_lightrag_workspace(s)

    log.error.assert_not_called()
    assert _variants(db, s.silo_id) == 1
    invalidate.assert_not_called()
    assert s.silo_id in lightrag_store._VARIANT_NAME_CACHE
    session_proxy.close.assert_not_called()


def test_variant_cleanup_failure_is_logged_not_raised(db, silos, session_proxy):
    s, _ = silos
    session_proxy.commit = MagicMock(side_effect=RuntimeError("db down"))
    with patch("services.silo_service._get_vector_store", return_value=MagicMock()), \
            patch(_INVALIDATE) as invalidate, \
            patch("services.silo_service.logger") as log:
        SiloService.wipe_lightrag_workspace(s)

    log.error.assert_called_once()
    invalidate.assert_called_once_with(s.silo_id)
    session_proxy.rollback.assert_called_once()  # done by the repository itself
    session_proxy.close.assert_called_once()


def test_variant_cache_invalidation_failure_does_not_skip_metadata_cache(db, silos, session_proxy):
    s, _ = silos
    with patch("services.silo_service._get_vector_store", return_value=MagicMock()), \
            patch("tools.vector_stores.lightrag_store.invalidate_variant_name_cache", side_effect=RuntimeError("x")), \
            patch(_INVALIDATE) as invalidate, \
            patch("services.silo_service.logger") as log:
        SiloService.wipe_lightrag_workspace(s)

    log.error.assert_called_once()
    invalidate.assert_called_once_with(s.silo_id)
