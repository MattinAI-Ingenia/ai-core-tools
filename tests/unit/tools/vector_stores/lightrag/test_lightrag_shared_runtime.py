"""Regression tests for the process-wide LightRAG runtime shared by every store.

Bug being guarded against: ``VectorStoreFactory`` builds a fresh ``LightRAGStore``
per call (it is not cached for LIGHTRAG), and each store used to build its own
``LightRAG`` per collection (Neo4j driver, Qdrant client, an embedding probe
call, one 1-13 s handshake) plus a ``mkdtemp`` directory — none of it ever closed.
After a few hundred agent turns the backend held ~100 sockets and thousands of
temp directories, and the Postgres server started refusing connections.

The fix shares the built instances (and their dedicated loops) across stores
that carry the same configuration, closes them when the configuration changes
or the collection is deleted, and uses one working directory per process.
All LightRAG/Neo4j/Qdrant/PostgreSQL dependencies are mocked.
"""

from __future__ import annotations

import gc
import os
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.ai_service import AIService
from models.embedding_service import EmbeddingService
from tools.vector_store_factory import VectorStoreFactory
import tools.vector_stores.lightrag_store as lightrag_store
from tools.vector_stores.lightrag_store import LightRAGStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm(**overrides) -> AIService:
    fields = dict(service_id=1, name="llm", provider="Custom", endpoint="http://llm:8000/v1",
                  api_key="secret-llm-key", description="qwen")
    fields.update(overrides)
    return AIService(**fields)


def _emb(**overrides) -> EmbeddingService:
    fields = dict(service_id=2, name="emb", provider="OpenAI", endpoint=None,
                  api_key="secret-emb-key", description="text-embedding-3-small")
    fields.update(overrides)
    return EmbeddingService(**fields)


def _store(ai=None, emb=None, **kwargs) -> LightRAGStore:
    return LightRAGStore(
        db=MagicMock(),
        ai_service=ai or _llm(),
        embedding_service=emb or _emb(),
        shared_runtime=True,
        **kwargs,
    )


def _mock_rag():
    rag = MagicMock()
    rag.initialize_storages = AsyncMock()
    rag.finalize_storages = AsyncMock()
    return rag


def _reset_shared_runtime():
    runtime = lightrag_store._SHARED_RUNTIME
    lightrag_store._close_all_collection_loops(runtime.collection_loops)
    for registry in (runtime.rag_instances, runtime.collection_loops,
                     runtime.init_locks, runtime.fingerprints):
        registry.clear()


@pytest.fixture(autouse=True)
def _clean_shared_runtime(monkeypatch):
    """Every test starts and ends with an empty shared runtime and no grace delay."""
    monkeypatch.setattr(lightrag_store, "_RETIRE_GRACE_S", 0)
    _reset_shared_runtime()
    yield
    _reset_shared_runtime()


# ---------------------------------------------------------------------------
# Same configuration -> one instance, one loop, one working dir
# ---------------------------------------------------------------------------


def test_stores_with_the_same_configuration_share_one_instance_and_loop():
    built = []

    def build(self, collection_name):
        built.append(collection_name)
        return _mock_rag()

    with patch.object(LightRAGStore, "_build_rag", build):
        first, second = _store(), _store()
        rag1 = first._get_rag_instance("silo_1")
        rag2 = second._get_rag_instance("silo_1")

    assert rag1 is rag2
    assert built == ["silo_1"]
    assert first._collection_loops["silo_1"] is second._collection_loops["silo_1"]


def test_a_new_store_does_not_create_another_temp_dir():
    _store()  # warm the process-wide dir
    before = {d for d in os.listdir(os.path.dirname(_store()._working_dir)) if d.startswith("lightrag_")}
    stores = [_store() for _ in range(20)]
    after = {d for d in os.listdir(os.path.dirname(stores[0]._working_dir)) if d.startswith("lightrag_")}

    assert after == before
    assert len({s._working_dir for s in stores}) == 1


def test_a_private_store_removes_its_temp_dir_when_collected():
    store = LightRAGStore(db=MagicMock(), ai_service=_llm(), embedding_service=_emb())
    path = store._working_dir
    assert os.path.isdir(path)

    del store
    gc.collect()

    assert not os.path.exists(path)


def test_concurrent_stores_build_the_collection_exactly_once():
    built = []
    gate = threading.Barrier(8)

    def build(self, collection_name):
        built.append(collection_name)
        return _mock_rag()

    errors = []

    def worker():
        try:
            gate.wait(timeout=5)
            _store()._get_rag_instance("silo_race")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    with patch.object(LightRAGStore, "_build_rag", build):
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

    assert not errors
    assert built == ["silo_race"]


# ---------------------------------------------------------------------------
# Configuration change -> old instance is finalised and its loop stopped
# ---------------------------------------------------------------------------


def test_changing_the_configuration_replaces_and_closes_the_old_instance():
    rags = []

    def build(self, collection_name):
        rag = _mock_rag()
        rags.append(rag)
        return rag

    with patch.object(LightRAGStore, "_build_rag", build):
        old_store = _store()
        old_rag = old_store._get_rag_instance("silo_1")
        old_thread = old_store._collection_loops["silo_1"]._thread

        new_store = _store(emb=_emb(description="another-embedding-model"))
        new_rag = new_store._get_rag_instance("silo_1")

    assert new_rag is not old_rag
    assert len(rags) == 2
    old_rag.finalize_storages.assert_awaited_once()
    old_thread.join(timeout=5)
    assert not old_thread.is_alive()
    assert new_store._collection_loops["silo_1"]._thread.is_alive()


def test_changing_a_silo_setting_changes_the_fingerprint():
    base = _store()._fingerprint
    assert base is not None
    assert _store(lightrag_chunk_token_size=500)._fingerprint != base
    assert _store(lightrag_language="Spanish")._fingerprint != base
    assert _store(ai=_llm(endpoint="http://other:8000/v1"))._fingerprint != base
    assert _store()._fingerprint == base


def test_the_fingerprint_does_not_keep_service_secrets():
    fingerprint = _store()._fingerprint
    assert "secret-llm-key" not in repr(fingerprint)
    assert "secret-emb-key" not in repr(fingerprint)


# ---------------------------------------------------------------------------
# Things that must keep working as before
# ---------------------------------------------------------------------------


def test_services_that_cannot_be_fingerprinted_fall_back_to_a_private_runtime():
    plain = SimpleNamespace(provider="OpenAI", name="x", description="y", api_key="k", endpoint=None)
    first = LightRAGStore(db=MagicMock(), ai_service=plain, embedding_service=plain, shared_runtime=True)
    second = LightRAGStore(db=MagicMock(), ai_service=plain, embedding_service=plain, shared_runtime=True)

    assert first._fingerprint is None
    assert first._rag_instances is not second._rag_instances
    assert first._collection_loops is not second._collection_loops


def test_delete_collection_closes_the_shared_loop_and_allows_a_rebuild():
    built = []

    def build(self, collection_name):
        built.append(collection_name)
        return _mock_rag()

    with patch.object(LightRAGStore, "_build_rag", build), \
         patch.object(LightRAGStore, "_cleanup_neo4j"), \
         patch.object(LightRAGStore, "_cleanup_qdrant"), \
         patch.object(LightRAGStore, "_cleanup_postgres"):
        store = _store()
        store._get_rag_instance("silo_1")
        thread = store._collection_loops["silo_1"]._thread

        store.delete_collection("silo_1")
        thread.join(timeout=5)
        assert not thread.is_alive()

        _store()._get_rag_instance("silo_1")

    assert built == ["silo_1", "silo_1"]


def test_the_factory_builds_shared_stores_for_lightrag():
    with patch("tools.vector_stores.lightrag.adapters.is_lightrag_available", return_value=True):
        first = VectorStoreFactory.get_vector_store(
            MagicMock(), "LIGHTRAG", ai_service=_llm(), embedding_service=_emb())
        second = VectorStoreFactory.get_vector_store(
            MagicMock(), "LIGHTRAG", ai_service=_llm(), embedding_service=_emb())

    assert first._fingerprint is not None
    assert first._fingerprint == second._fingerprint
    assert first._rag_instances is second._rag_instances
