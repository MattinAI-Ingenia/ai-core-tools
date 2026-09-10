"""Regression tests for LightRAGStore's per-collection dedicated event loop.

Concurrency bug being guarded against: ``_get_rag_instance``/``_aget_rag_instance``
used to run every subsequent coroutine touching a cached ``rag``/Neo4j-driver
instance on a fresh THROWAWAY loop per call (via ``_run_async``), and the sync
``_get_rag_instance`` had no lock around its check-then-build-then-cache
section at all — two threads racing to first-touch the same collection could
both build+initialize it. See ``_CollectionEventLoop`` in ``lightrag_store.py``
for the fix: one persistent background thread+loop per collection, created
once (race-free) and reused for the collection's whole cache lifetime.

All LightRAG/Neo4j/Qdrant/PostgreSQL dependencies are mocked so these tests
run without network access.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.vector_stores.lightrag_store import LightRAGStore

# pytest.ini does not enable pytest-asyncio's auto mode, so mark every
# async test in this module explicitly (mirrors test_lightrag_store.py,
# which mixes plain `def` and `async def` tests under one module-level mark).
pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ai_service():
    return SimpleNamespace(
        provider="OpenAI", name="test-llm", description="gpt-4o",
        api_key="sk-test", endpoint=None,
    )


def _make_embedding_service():
    return SimpleNamespace(
        provider="OpenAI", name="test-embed", description="text-embedding-3-small",
        api_key="sk-test", endpoint=None, api_version=None,
    )


def _make_store() -> LightRAGStore:
    return LightRAGStore(
        db=MagicMock(),
        ai_service=_make_ai_service(),
        embedding_service=_make_embedding_service(),
    )


def _mock_rag():
    rag = MagicMock()
    rag.initialize_storages = AsyncMock()
    return rag


# ---------------------------------------------------------------------------
# Same collection, repeated access -> same instance AND same dedicated loop
# ---------------------------------------------------------------------------


def test_get_rag_instance_builds_once_and_reuses_the_same_loop_across_calls():
    store = _make_store()
    built = []

    def _fake_build_rag(collection_name):
        built.append(collection_name)
        return _mock_rag()

    with patch.object(store, "_build_rag", side_effect=_fake_build_rag):
        rag1 = store._get_rag_instance("silo_1")
        loop1 = store._collection_loops["silo_1"]
        rag2 = store._get_rag_instance("silo_1")
        loop2 = store._collection_loops["silo_1"]

    assert built == ["silo_1"]  # _build_rag ran exactly once
    assert rag1 is rag2
    assert loop1 is loop2


async def test_aget_rag_instance_shares_the_sync_paths_instance_and_loop():
    """The sync and async entry points are two doors into the SAME cached
    instance/loop for a given collection_name — never two different ones."""
    store = _make_store()

    with patch.object(store, "_build_rag", side_effect=lambda name: _mock_rag()):
        rag_sync = store._get_rag_instance("silo_1")
        loop_sync = store._collection_loops["silo_1"]

        rag_async = await store._aget_rag_instance("silo_1")
        loop_async = store._collection_loops["silo_1"]

    assert rag_async is rag_sync
    assert loop_async is loop_sync


def test_run_on_collection_loop_executes_on_the_collections_own_loop():
    """A coroutine dispatched via _run_on_collection_loop must actually run on
    the collection's dedicated loop — not a fresh throwaway one."""
    store = _make_store()
    with patch.object(store, "_build_rag", side_effect=lambda name: _mock_rag()):
        store._get_rag_instance("silo_1")

    expected_loop = store._collection_loops["silo_1"]._loop

    async def _capture_running_loop():
        return asyncio.get_running_loop()

    running_loop = store._run_on_collection_loop("silo_1", _capture_running_loop())
    assert running_loop is expected_loop


async def test_arun_on_collection_loop_executes_on_the_collections_own_loop():
    """Same guarantee from an ASYNC caller: the coroutine runs on the
    collection's dedicated loop, not on the calling coroutine's own loop."""
    store = _make_store()
    with patch.object(store, "_build_rag", side_effect=lambda name: _mock_rag()):
        store._get_rag_instance("silo_1")

    expected_loop = store._collection_loops["silo_1"]._loop
    caller_loop = asyncio.get_running_loop()
    assert caller_loop is not expected_loop  # sanity: genuinely a different loop/thread

    async def _capture_running_loop():
        return asyncio.get_running_loop()

    running_loop = await store._arun_on_collection_loop("silo_1", _capture_running_loop())
    assert running_loop is expected_loop


# ---------------------------------------------------------------------------
# Sync check-then-build-then-cache must be race-free (matches the async path)
# ---------------------------------------------------------------------------


def test_sync_first_access_is_race_free_under_concurrent_first_touch():
    """Two threads racing to first-touch the SAME collection must not both
    build+initialize it — only one build should win, matching
    _aget_rag_instance's pre-existing async guarantee.

    On the pre-fix code (no lock in _get_rag_instance) this reliably fails:
    both threads pass the initial "not yet cached" check before either one
    finishes building — the artificial delay below only widens that window
    to make the race deterministic instead of a rare flake.
    """
    store = _make_store()
    build_count = 0
    count_lock = threading.Lock()

    def _fake_build_rag(collection_name):
        nonlocal build_count
        with count_lock:
            build_count += 1
        time.sleep(0.05)
        return _mock_rag()

    results = []
    results_lock = threading.Lock()

    def _call():
        rag = store._get_rag_instance("silo_shared")
        with results_lock:
            results.append(rag)

    with patch.object(store, "_build_rag", side_effect=_fake_build_rag):
        t1 = threading.Thread(target=_call)
        t2 = threading.Thread(target=_call)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

    assert build_count == 1
    assert len(results) == 2
    assert results[0] is results[1]


# ---------------------------------------------------------------------------
# Different collections never share a loop
# ---------------------------------------------------------------------------


def test_different_collections_never_share_a_dedicated_loop():
    store = _make_store()

    with patch.object(store, "_build_rag", side_effect=lambda name: _mock_rag()):
        store._get_rag_instance("silo_A")
        store._get_rag_instance("silo_B")

    loop_a = store._collection_loops["silo_A"]
    loop_b = store._collection_loops["silo_B"]

    assert loop_a is not loop_b
    assert loop_a._loop is not loop_b._loop
    assert loop_a._thread is not loop_b._thread


# ---------------------------------------------------------------------------
# Teardown — no leaked thread/loop
# ---------------------------------------------------------------------------


def test_delete_collection_closes_the_dedicated_loop():
    store = _make_store()
    with patch.object(store, "_build_rag", side_effect=lambda name: _mock_rag()), \
         patch.object(store, "_cleanup_neo4j"), \
         patch.object(store, "_cleanup_qdrant"), \
         patch.object(store, "_cleanup_postgres"):
        store._get_rag_instance("silo_1")
        thread = store._collection_loops["silo_1"]._thread
        assert thread.is_alive()

        store.delete_collection("silo_1")

    thread.join(timeout=5)
    assert not thread.is_alive()
    assert "silo_1" not in store._collection_loops
