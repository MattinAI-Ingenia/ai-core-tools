"""Regression tests for LightRAG's process-wide Postgres client pool now
living on one shared, persistent event loop, instead of whichever
throwaway/per-collection loop happened to touch it first.

Concurrency bug being guarded against: ``lightrag.kg.postgres_impl.ClientManager``
keeps ONE asyncpg pool for the whole process, bound forever to whichever
event loop first created it. Indexing used to force-reset that pool at the
start of every job (``resource_service._run_and_unlock`` +
``reset_lightrag_postgres_pool``, serialized only against other *indexing*
jobs via a sentinel ``-1`` advisory lock) while queries -- a completely
separate ``LightRAGStore``/``_CollectionEventLoop``, since
``VectorStoreFactory`` never caches ``LightRAGStore`` instances -- never
coordinated with that reset at all. A query's in-flight Postgres operations
could have their pool yanked out from under them by an unrelated silo's
indexing job, crashing with "Future attached to a different loop" / asyncpg's
own cross-loop error.

Fix: ``_SharedPostgresLoop`` is a process-wide singleton loop; every entry
point that actually touches the pool (``ClientManager.get_client``,
``ClientManager.release_client``, ``PostgreSQLDB._run_with_retry`` -- the
sole chokepoint every ``query``/``execute``/storage-class DB call and pool
(re)build funnels through) is monkeypatched by
``_ensure_shared_postgres_loop_patch`` to always execute on it, regardless of
which collection/silo (and which of ITS OWN ``_CollectionEventLoop``)
initiated the call.

All lightrag/Postgres dependencies are faked so these tests run without a
real Postgres connection.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import patch

import pytest

from tools.vector_stores import lightrag_store
from tools.vector_stores.lightrag_store import (
    _CollectionEventLoop,
    _SharedPostgresLoop,
    _ensure_shared_postgres_loop_patch,
)

# pytest.ini does not enable pytest-asyncio's auto mode, so mark every
# async test in this module explicitly (mirrors the sibling
# test_lightrag_collection_loop.py in this same directory).
pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixture: undo the process-wide monkeypatch/singleton after each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_shared_postgres_patch_state():
    """``_ensure_shared_postgres_loop_patch`` mutates real classes imported
    from the installed ``lightrag`` package and a module-level singleton --
    both of which must not leak between tests (or into the rest of the
    suite, which may run real LightRAG code against a real Postgres test DB).
    """
    from lightrag.kg.postgres_impl import ClientManager, PostgreSQLDB

    orig_get_client = ClientManager.get_client
    orig_release_client = ClientManager.release_client
    orig_run_with_retry = PostgreSQLDB._run_with_retry
    orig_patch_applied = lightrag_store._postgres_impl_patch_applied
    orig_shared_instance = _SharedPostgresLoop._instance

    lightrag_store._postgres_impl_patch_applied = False
    _SharedPostgresLoop._instance = None

    yield

    ClientManager.get_client = orig_get_client
    ClientManager.release_client = orig_release_client
    PostgreSQLDB._run_with_retry = orig_run_with_retry

    # Stop whatever shared loop THIS test created, if any, before restoring
    # the pre-test singleton (which may be None, or another test's -- either
    # way this test's own thread must not keep running past the test).
    test_instance = _SharedPostgresLoop._instance
    if test_instance is not None and test_instance is not orig_shared_instance:
        test_instance._loop.call_soon_threadsafe(test_instance._loop.stop)
        test_instance._thread.join(timeout=5)

    lightrag_store._postgres_impl_patch_applied = orig_patch_applied
    _SharedPostgresLoop._instance = orig_shared_instance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_fake_client_manager():
    """Replace ``ClientManager.get_client``/``release_client`` with fakes
    that record the event loop they actually ran on and how many times the
    'pool' was (re)built -- standing in for asyncpg's real, genuinely
    cross-loop-sensitive connection pool, without touching a real database.

    Returns ``(build_calls, recorded_loops)``.
    """
    from lightrag.kg.postgres_impl import ClientManager

    build_calls: list[int] = []
    recorded_loops: list[asyncio.AbstractEventLoop] = []
    state = {"db": None}

    async def fake_get_client(cls, vector_storage=None):
        recorded_loops.append(asyncio.get_running_loop())
        if state["db"] is None:
            build_calls.append(1)
            state["db"] = object()
        return state["db"]

    async def fake_release_client(cls, db):
        recorded_loops.append(asyncio.get_running_loop())

    ClientManager.get_client = classmethod(fake_get_client)
    ClientManager.release_client = classmethod(fake_release_client)
    return build_calls, recorded_loops


def _install_fake_run_with_retry():
    """Replace ``PostgreSQLDB._run_with_retry`` with a fake that records the
    event loop it actually ran the operation on -- this is the single
    chokepoint every real ``query``/``execute``/storage DB call funnels
    through, so faking it here stands in for a query/insert touching the
    pool without needing a real connection.

    Returns ``recorded_loops``.
    """
    from lightrag.kg.postgres_impl import PostgreSQLDB

    recorded_loops: list[asyncio.AbstractEventLoop] = []

    async def fake_run_with_retry(self, operation, *, with_age=False, graph_name=None, timing_label=None):
        recorded_loops.append(asyncio.get_running_loop())
        return await operation(None)

    PostgreSQLDB._run_with_retry = fake_run_with_retry
    return recorded_loops


def _make_collection_loop(name: str) -> _CollectionEventLoop:
    """A dedicated loop standing in for one silo's own ``LightRAGStore``
    collection loop (built by a fresh, uncached store instance per the real
    ``VectorStoreFactory`` behaviour -- see its docstring)."""
    return _CollectionEventLoop(name)


# ---------------------------------------------------------------------------
# Every Postgres-touching call, from any collection, runs on the SAME loop
# ---------------------------------------------------------------------------


async def test_get_client_from_different_collections_always_uses_the_shared_loop():
    """Two different silos' indexing/query calls must land on the identical
    shared loop object -- never on either collection's own dedicated loop.

    Pre-fix (no monkeypatch at all), each call would simply run on whichever
    collection loop invoked it, so the two recorded loops would differ.
    """
    build_calls, recorded_loops = _install_fake_client_manager()
    _ensure_shared_postgres_loop_patch()

    from lightrag.kg.postgres_impl import ClientManager

    loop_a = _make_collection_loop("silo_A")
    loop_b = _make_collection_loop("silo_B")

    try:
        loop_a.run(ClientManager.get_client())
        loop_b.run(ClientManager.get_client())
    finally:
        loop_a.close()
        loop_b.close()

    assert len(recorded_loops) == 2
    assert recorded_loops[0] is recorded_loops[1], (
        "both collections' get_client() calls must execute on the same "
        "shared loop, not each collection's own loop"
    )
    shared_loop = _SharedPostgresLoop.instance()
    assert recorded_loops[0] is shared_loop._loop
    assert recorded_loops[0] is not loop_a._loop
    assert recorded_loops[0] is not loop_b._loop


async def test_run_with_retry_from_different_collections_always_uses_the_shared_loop():
    """Same guarantee for the query/insert chokepoint (``_run_with_retry``),
    which every real ``query``/``execute``/storage-class DB call funnels
    through -- not just pool (de)registration."""
    recorded_loops = _install_fake_run_with_retry()
    _ensure_shared_postgres_loop_patch()

    from lightrag.kg.postgres_impl import PostgreSQLDB

    db = PostgreSQLDB.__new__(PostgreSQLDB)  # no real connection needed

    async def _noop(_connection):
        return "ok"

    loop_a = _make_collection_loop("silo_A")
    loop_b = _make_collection_loop("silo_B")

    try:
        result_a = loop_a.run(db._run_with_retry(_noop))
        result_b = loop_b.run(db._run_with_retry(_noop))
    finally:
        loop_a.close()
        loop_b.close()

    assert result_a == result_b == "ok"
    assert recorded_loops[0] is recorded_loops[1]
    assert recorded_loops[0] is _SharedPostgresLoop.instance()._loop


# ---------------------------------------------------------------------------
# A concurrent "indexing" pool rebuild must not break a concurrent "query"
# ---------------------------------------------------------------------------


async def test_concurrent_indexing_and_query_do_not_cross_loops_or_race():
    """Simulate silo A indexing (repeatedly touching ``get_client``, which
    may rebuild the 'pool') fully concurrently with silo B querying (via
    ``_run_with_retry``), each from its own collection loop/thread.

    Before the fix, each collection ran Postgres calls on its OWN loop, so
    one job finishing (and, in the old design, force-resetting the
    process-wide pool) while the other was mid-flight would hand the second
    job a pool bound to a now-dead loop. After the fix both are always
    serialized onto the identical shared loop, so nothing ever executes on
    two different loops at once and neither job can invalidate the other's
    in-flight operation.
    """
    build_calls, get_client_loops = _install_fake_client_manager()
    run_with_retry_loops = _install_fake_run_with_retry()
    _ensure_shared_postgres_loop_patch()

    from lightrag.kg.postgres_impl import ClientManager, PostgreSQLDB

    db = PostgreSQLDB.__new__(PostgreSQLDB)

    async def _query_op(_connection):
        # Yield control so the two "jobs" genuinely interleave instead of
        # one finishing before the other starts.
        await asyncio.sleep(0.01)
        return "query-ok"

    indexing_loop = _make_collection_loop("silo_indexing")
    query_loop = _make_collection_loop("silo_query")

    errors: list[BaseException] = []
    results: dict[str, object] = {}

    def _run_indexing():
        try:
            for _ in range(5):
                indexing_loop.run(ClientManager.get_client())
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    def _run_query():
        try:
            results["query"] = query_loop.run(db._run_with_retry(_query_op))
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    t1 = threading.Thread(target=_run_indexing)
    t2 = threading.Thread(target=_run_query)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    try:
        assert not errors, f"concurrent indexing/query raised: {errors}"
        assert results.get("query") == "query-ok"
        assert build_calls == [1], "the fake pool must be built exactly once"
        shared_loop = _SharedPostgresLoop.instance()._loop
        assert all(loop is shared_loop for loop in get_client_loops)
        assert all(loop is shared_loop for loop in run_with_retry_loops)
    finally:
        indexing_loop.close()
        query_loop.close()


# ---------------------------------------------------------------------------
# The pool is built at most once across multiple calls from different silos
# ---------------------------------------------------------------------------


async def test_pool_built_at_most_once_across_multiple_silos_and_calls():
    """No repeated rebuild-and-orphan cycle: once the fake 'pool' exists,
    subsequent ``get_client`` calls -- from any collection -- must reuse it.

    This is as close as a unit test can get to proving the real asyncpg pool
    is never orphaned/rebuilt without a live Postgres server: the real
    ``ClientManager.get_client`` only calls ``PostgreSQLDB.initdb()`` (which
    is what actually creates the pool) when its cached ``db`` is ``None`` --
    exercised here by the same guard in the fake.
    """
    build_calls, _ = _install_fake_client_manager()
    _ensure_shared_postgres_loop_patch()

    from lightrag.kg.postgres_impl import ClientManager

    loops = [_make_collection_loop(f"silo_{i}") for i in range(4)]
    try:
        for loop in loops:
            loop.run(ClientManager.get_client())
            loop.run(ClientManager.get_client())  # second touch, same collection
    finally:
        for loop in loops:
            loop.close()

    assert build_calls == [1]


# ---------------------------------------------------------------------------
# The patch is idempotent
# ---------------------------------------------------------------------------


def test_ensure_patch_is_idempotent_and_singleton_loop_is_reused():
    _ensure_shared_postgres_loop_patch()
    first = _SharedPostgresLoop.instance()
    _ensure_shared_postgres_loop_patch()
    second = _SharedPostgresLoop.instance()
    assert first is second
