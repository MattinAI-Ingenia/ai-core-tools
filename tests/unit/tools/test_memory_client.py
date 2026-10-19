"""Unit tests for ``tools.memory.client`` (spec `user-memory-mem0`, step_005
security-audit fixes). Pure-Python — no real database; every store call is
faked or mocked.

Covers, without needing the real Postgres test DB (see
``tests/integration/tools/test_memory_client_namespace_isolation.py`` for the
end-to-end regression against a real store):

- ``_assert_identity``/``_assert_positive_int`` reject non-positive-int and
  ``bool`` app_id/user_id (a namespace built from either could contain a SQL
  `LIKE` metacharacter).
- ``_purge_sync``'s exact-namespace filtering and offset-advance pagination,
  including the max-iteration guard against a dominating neighbouring scope.
- ``list_user_scopes``'s exact-suffix re-check.
- ``GlobalMemoryPoolService``'s bounded setup()-convergence poll, its
  pool/`_is_setup_done` cleanup on a failed ``initialize_pool()``, and
  ``get_pool()`` requiring ``_is_setup_done``.
- The password-safe pool-init log line.
- ``_drain_all_clients`` leaving still-busy clients referenced instead of
  dropping them.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import psycopg
import pytest

import tools.memory.client as client_module
from tools.memory.client import (
    GlobalMemoryPoolService,
    PurgeResult,
    ScopeListResult,
    _assert_identity,
    _assert_positive_int,
    list_app_scopes,
    list_user_scopes,
    purge,
)
from tools.memory.identity import MemoryIdentity

_ROOT = "user_memory"


def _identity(app_id: int, user_id: int, agent_id: int = 1) -> MemoryIdentity:
    return MemoryIdentity(
        app_id=app_id, user_id=user_id, agent_id=agent_id,
        namespace=(_ROOT, str(app_id), str(user_id)),
    )


@contextlib.contextmanager
def _fake_sync_connection():
    yield None


class _FakeRow:
    __slots__ = ("namespace", "key", "value")

    def __init__(self, namespace, key):
        self.namespace = namespace
        self.key = key
        self.value = {"text": key}


# ---------------------------------------------------------------------------
# _assert_identity / _assert_positive_int
# ---------------------------------------------------------------------------


class TestAssertPositiveInt:
    def test_rejects_bool(self):
        with pytest.raises(AssertionError):
            _assert_positive_int(True, "app_id")

    def test_rejects_zero_and_negative(self):
        with pytest.raises(AssertionError):
            _assert_positive_int(0, "app_id")
        with pytest.raises(AssertionError):
            _assert_positive_int(-3, "app_id")

    def test_rejects_non_int(self):
        with pytest.raises(AssertionError):
            _assert_positive_int("5", "app_id")

    def test_accepts_positive_int(self):
        _assert_positive_int(5, "app_id")  # must not raise


class TestAssertIdentity:
    def test_rejects_non_memory_identity(self):
        with pytest.raises(AssertionError):
            _assert_identity(object())

    def test_rejects_bool_app_id_smuggled_past_the_dataclass(self):
        """A real `MemoryIdentity` can't normally hold a bool app_id (its own
        `__post_init__` would already reject the namespace mismatch), but a
        future caller manufacturing one via `object.__setattr__` must still be
        caught here — a `str(True) == "True"` namespace segment is at least
        shape-safe, but the *type* itself must never reach a namespace build.
        """
        identity = _identity(app_id=1, user_id=1)
        object.__setattr__(identity, "app_id", True)
        with pytest.raises(AssertionError):
            _assert_identity(identity)

    def test_rejects_non_positive_user_id(self):
        identity = _identity(app_id=1, user_id=1)
        object.__setattr__(identity, "user_id", -1)
        with pytest.raises(AssertionError):
            _assert_identity(identity)

    def test_accepts_well_formed_identity(self):
        _assert_identity(_identity(app_id=5, user_id=7))  # must not raise


# ---------------------------------------------------------------------------
# _purge_sync — exact-namespace filtering, offset-advance pagination, guard
# ---------------------------------------------------------------------------


class _FakeStore:
    """Mimics `PostgresStore.search`/`batch` over an in-memory row list:
    `search` returns a LIKE-prefix-matched (not exact) slice in stable order,
    `batch` deletes only rows whose (namespace, key) is in the given ops —
    exactly the same shape `_purge_sync` must defend against.
    """

    def __init__(self, rows: List[_FakeRow]):
        self.rows = rows

    def search(self, namespace, *, limit, offset):
        prefix_text = ".".join(namespace)
        matched = [row for row in self.rows if ".".join(row.namespace).startswith(prefix_text)]
        return matched[offset : offset + limit]

    def batch(self, ops):
        delete_keys = {(op.namespace, op.key) for op in ops}
        self.rows = [row for row in self.rows if (row.namespace, row.key) not in delete_keys]


class _InfiniteNeighbourStore:
    """A neighbouring scope that always fills the page and is never deleted —
    the exact shape that used to loop `_purge_sync` forever.
    """

    def search(self, namespace, *, limit, offset):
        return [_FakeRow((_ROOT, "5", "70"), f"n{offset}-{i}") for i in range(limit)]

    def batch(self, ops):
        raise AssertionError("must never attempt to delete a foreign-namespace row")


class TestPurgeSync:
    def test_deletes_only_exact_matches_interleaved_across_pages(self, monkeypatch):
        identity = _identity(app_id=5, user_id=7)
        rows = (
            [_FakeRow((_ROOT, "5", "70"), f"n{i}") for i in range(2)]
            + [_FakeRow((_ROOT, "5", "7"), f"e{i}") for i in range(3)]
            + [_FakeRow((_ROOT, "5", "70"), f"n{i}") for i in range(2, 4)]
        )
        fake_store = _FakeStore(rows)
        monkeypatch.setattr(client_module, "PostgresStore", lambda conn=None: fake_store)
        monkeypatch.setattr(client_module, "_sync_connection", _fake_sync_connection)
        monkeypatch.setattr(client_module, "_PURGE_BATCH_SIZE", 3)

        deleted = client_module._purge_sync(identity)

        assert deleted == 3
        remaining = {row.namespace for row in fake_store.rows}
        assert remaining == {(_ROOT, "5", "70")}

    def test_empty_namespace_is_a_no_op(self, monkeypatch):
        identity = _identity(app_id=5, user_id=7)
        fake_store = _FakeStore([])
        monkeypatch.setattr(client_module, "PostgresStore", lambda conn=None: fake_store)
        monkeypatch.setattr(client_module, "_sync_connection", _fake_sync_connection)

        assert client_module._purge_sync(identity) == 0

    def test_max_iteration_guard_raises_instead_of_looping_forever(self, monkeypatch):
        identity = _identity(app_id=5, user_id=7)
        monkeypatch.setattr(client_module, "PostgresStore", lambda conn=None: _InfiniteNeighbourStore())
        monkeypatch.setattr(client_module, "_sync_connection", _fake_sync_connection)
        monkeypatch.setattr(client_module, "_PURGE_BATCH_SIZE", 5)
        monkeypatch.setattr(client_module, "_PURGE_MAX_ITERATIONS", 3)

        with pytest.raises(client_module._PurgeIncomplete):
            client_module._purge_sync(identity)


class TestPurge:
    """`purge()`'s public contract: a `PurgeResult`, never a bare int, and
    `complete=False` — never a misleadingly "successful" `0` — whenever the
    walk was interrupted.
    """

    def test_returns_purge_result_with_complete_true_on_success(self, monkeypatch):
        identity = _identity(app_id=5, user_id=7)
        rows = [_FakeRow((_ROOT, "5", "7"), "e0"), _FakeRow((_ROOT, "5", "70"), "n0")]
        fake_store = _FakeStore(rows)
        monkeypatch.setattr(client_module, "PostgresStore", lambda conn=None: fake_store)
        monkeypatch.setattr(client_module, "_sync_connection", _fake_sync_connection)

        result = purge(identity)

        assert result == PurgeResult(deleted=1, complete=True)
        assert {row.namespace for row in fake_store.rows} == {(_ROOT, "5", "70")}

    def test_returns_complete_false_and_partial_count_on_interruption(self, monkeypatch):
        identity = _identity(app_id=5, user_id=7)
        monkeypatch.setattr(client_module, "PostgresStore", lambda conn=None: _InfiniteNeighbourStore())
        monkeypatch.setattr(client_module, "_sync_connection", _fake_sync_connection)
        monkeypatch.setattr(client_module, "_PURGE_BATCH_SIZE", 5)
        monkeypatch.setattr(client_module, "_PURGE_MAX_ITERATIONS", 3)

        result = purge(identity)

        assert result.complete is False
        assert result.deleted == 0

    def test_malformed_identity_returns_complete_false_not_a_silent_success(self):
        identity = _identity(app_id=5, user_id=7)
        object.__setattr__(identity, "app_id", True)

        result = purge(identity)

        assert result == PurgeResult(deleted=0, complete=False)


# ---------------------------------------------------------------------------
# list_user_scopes — exact-suffix re-check
# ---------------------------------------------------------------------------


class TestListUserScopes:
    def test_filters_out_suffix_collisions(self, monkeypatch):
        class _FakeListStore:
            def list_namespaces(self, *, prefix, suffix, max_depth, limit, offset):
                if offset > 0:
                    return []
                return [
                    (_ROOT, "5", "7"),
                    (_ROOT, "5", "70"),  # LIKE '%7' match, not an exact suffix
                    (_ROOT, "3", "17"),  # LIKE '%7' match (ends in digit 7), not user 7
                ]

        monkeypatch.setattr(client_module, "PostgresStore", lambda conn=None: _FakeListStore())
        monkeypatch.setattr(client_module, "_sync_connection", _fake_sync_connection)

        result = list_user_scopes(7)

        assert result.complete is True
        resolved = {(scope.app_id, scope.user_id) for scope in result.scopes}
        assert resolved == {(5, 7)}

    def test_rejects_non_positive_user_id(self):
        assert list_user_scopes(-1) == ScopeListResult(scopes=[], complete=False)
        # bool is an int subclass — must still be rejected
        assert list_user_scopes(True) == ScopeListResult(scopes=[], complete=False)

    def test_enumeration_failure_reports_incomplete_not_a_clean_empty_result(self, monkeypatch):
        """The regression this defect is named after: an enumeration that
        FAILS must never be indistinguishable from a genuinely empty scope
        set — both previously returned `[]`."""

        class _FailingListStore:
            def list_namespaces(self, *, prefix, suffix, max_depth, limit, offset):
                raise psycopg.OperationalError("store unreachable")

        monkeypatch.setattr(client_module, "PostgresStore", lambda conn=None: _FailingListStore())
        monkeypatch.setattr(client_module, "_sync_connection", _fake_sync_connection)

        result = list_user_scopes(7)

        assert result.complete is False
        assert result.scopes == []

    def test_enumeration_failure_on_a_later_page_keeps_scopes_already_collected(self, monkeypatch):
        class _FlakyListStore:
            def __init__(self):
                self.calls = 0

            def list_namespaces(self, *, prefix, suffix, max_depth, limit, offset):
                self.calls += 1
                if self.calls == 1:
                    return [(_ROOT, "5", "7")] * limit
                raise psycopg.OperationalError("store unreachable mid-page")

        monkeypatch.setattr(client_module, "_LIST_SCOPES_PAGE_SIZE", 1)
        monkeypatch.setattr(client_module, "PostgresStore", lambda conn=None: _FlakyListStore())
        monkeypatch.setattr(client_module, "_sync_connection", _fake_sync_connection)

        result = list_user_scopes(7)

        assert result.complete is False
        assert len(result.scopes) == 1


class TestListAppScopes:
    def test_success_returns_complete_true(self, monkeypatch):
        class _FakeListStore:
            def list_namespaces(self, *, prefix, max_depth, limit, offset):
                if offset > 0:
                    return []
                return [(_ROOT, "5", "7"), (_ROOT, "5", "8"), (_ROOT, "50", "1")]

        monkeypatch.setattr(client_module, "PostgresStore", lambda conn=None: _FakeListStore())
        monkeypatch.setattr(client_module, "_sync_connection", _fake_sync_connection)

        result = list_app_scopes(5)

        assert result.complete is True
        resolved = {(scope.app_id, scope.user_id) for scope in result.scopes}
        assert resolved == {(5, 7), (5, 8)}

    def test_rejects_non_positive_app_id(self):
        assert list_app_scopes(-1) == ScopeListResult(scopes=[], complete=False)
        assert list_app_scopes(True) == ScopeListResult(scopes=[], complete=False)

    def test_enumeration_failure_reports_incomplete_not_a_clean_empty_result(self, monkeypatch):
        class _FailingListStore:
            def list_namespaces(self, *, prefix, max_depth, limit, offset):
                raise psycopg.OperationalError("store unreachable")

        monkeypatch.setattr(client_module, "PostgresStore", lambda conn=None: _FailingListStore())
        monkeypatch.setattr(client_module, "_sync_connection", _fake_sync_connection)

        result = list_app_scopes(5)

        assert result.complete is False
        assert result.scopes == []


# ---------------------------------------------------------------------------
# GlobalMemoryPoolService — bounded setup() poll, failure cleanup, get_pool()
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_pool_service_state():
    """`GlobalMemoryPoolService` is a process-wide singleton (classvars) —
    every test that touches it must leave it exactly as it found it so other
    modules' tests (and the real app) never see leftover state.
    """
    yield
    GlobalMemoryPoolService._pool = None
    GlobalMemoryPoolService._is_setup_done = False


class TestRunSetupBoundedPoll:
    @pytest.mark.asyncio
    async def test_retries_through_a_transient_race_then_converges(self, monkeypatch):
        monkeypatch.setattr(client_module, "_SETUP_POLL_INTERVAL_S", 0.001)
        monkeypatch.setattr(client_module, "_SETUP_CONVERGENCE_TIMEOUT_S", 5.0)

        setup_store = AsyncMock()
        setup_store.setup = AsyncMock(side_effect=[psycopg.errors.UniqueViolation("cold-start race"), None])
        monkeypatch.setattr(client_module, "AsyncPostgresStore", MagicMock(return_value=setup_store))

        call_count = {"n": 0}

        async def fake_assert_migrations_complete(cls, store):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("not converged yet")

        monkeypatch.setattr(
            GlobalMemoryPoolService, "_assert_migrations_complete", classmethod(fake_assert_migrations_complete)
        )
        relax = AsyncMock()
        monkeypatch.setattr(GlobalMemoryPoolService, "_relax_vector_column", relax)
        GlobalMemoryPoolService._pool = MagicMock()

        await GlobalMemoryPoolService._run_setup(MagicMock())

        assert setup_store.setup.await_count == 2
        assert call_count["n"] == 2
        relax.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_failed_column_relax_is_retried_inside_the_poll(self, monkeypatch):
        """The post-setup DDL can lose a lock race (`lock_timeout`) on a busy,
        already-serving DB; that must be retried, not turn memory off."""
        monkeypatch.setattr(client_module, "_SETUP_POLL_INTERVAL_S", 0.001)
        monkeypatch.setattr(client_module, "_SETUP_CONVERGENCE_TIMEOUT_S", 5.0)

        setup_store = AsyncMock()
        monkeypatch.setattr(client_module, "AsyncPostgresStore", MagicMock(return_value=setup_store))
        monkeypatch.setattr(
            GlobalMemoryPoolService, "_assert_migrations_complete", classmethod(AsyncMock())
        )
        relax = AsyncMock(side_effect=[psycopg.errors.LockNotAvailable("lock timeout"), None])
        monkeypatch.setattr(GlobalMemoryPoolService, "_relax_vector_column", relax)
        GlobalMemoryPoolService._pool = MagicMock()

        await GlobalMemoryPoolService._run_setup(MagicMock())

        assert relax.await_count == 2

    @pytest.mark.asyncio
    async def test_gives_up_after_the_deadline_with_the_last_error_attached(self, monkeypatch):
        monkeypatch.setattr(client_module, "_SETUP_POLL_INTERVAL_S", 0.001)
        monkeypatch.setattr(client_module, "_SETUP_CONVERGENCE_TIMEOUT_S", 0.02)

        setup_store = AsyncMock()
        setup_store.setup = AsyncMock(side_effect=psycopg.errors.DuplicateTable("always racing"))
        monkeypatch.setattr(client_module, "AsyncPostgresStore", MagicMock(return_value=setup_store))

        async def always_incomplete(cls, store):
            raise RuntimeError("never converges")

        monkeypatch.setattr(
            GlobalMemoryPoolService, "_assert_migrations_complete", classmethod(always_incomplete)
        )
        GlobalMemoryPoolService._pool = MagicMock()

        with pytest.raises(RuntimeError, match="did not converge"):
            await GlobalMemoryPoolService._run_setup(MagicMock())


class TestSetupPoolHasNoStatementTimeout:
    @pytest.mark.asyncio
    async def test_main_pool_is_capped_but_setup_pool_is_not(self, monkeypatch):
        """The main pool caps statement duration so an abandoned turn cannot hold
        a connection; setup must NOT inherit that cap, because
        `CREATE INDEX CONCURRENTLY` on an upgraded client database legitimately
        outlives any per-turn budget, and an aborted one leaves an invalid index
        that `IF NOT EXISTS` then skips forever.
        """
        monkeypatch.setattr(client_module.settings, "is_global_memory_enabled", lambda: True)
        monkeypatch.setattr(client_module.settings, "write_timeout_s", lambda: 30.0)

        built_kwargs = []

        def fake_pool_factory(*args, **kwargs):
            built_kwargs.append(kwargs.get("kwargs", {}))
            pool = AsyncMock()
            pool.open = AsyncMock()
            pool.close = AsyncMock()
            return pool

        monkeypatch.setattr(client_module, "AsyncConnectionPool", MagicMock(side_effect=fake_pool_factory))
        monkeypatch.setattr(GlobalMemoryPoolService, "_run_setup", AsyncMock(return_value=None))

        try:
            await GlobalMemoryPoolService.initialize_pool()
        finally:
            GlobalMemoryPoolService._pool = None
            GlobalMemoryPoolService._is_setup_done = False

        assert len(built_kwargs) == 2, "expected one main pool and one dedicated setup pool"
        main_kwargs, setup_kwargs = built_kwargs
        assert main_kwargs["options"] == "-c statement_timeout=35000"
        assert "options" not in setup_kwargs


class TestInitializePoolFailureCleanup:
    @pytest.mark.asyncio
    async def test_closes_pool_and_resets_state_on_setup_failure(self, monkeypatch):
        monkeypatch.setattr(client_module.settings, "is_global_memory_enabled", lambda: True)

        fake_pool = AsyncMock()
        fake_pool.open = AsyncMock()
        fake_pool.close = AsyncMock()
        monkeypatch.setattr(client_module, "AsyncConnectionPool", MagicMock(return_value=fake_pool))
        # Patch the direct dependency: setup runs on its own dedicated pool
        # (no statement_timeout, so a long CREATE INDEX CONCURRENTLY is not
        # aborted), so `initialize_pool` no longer calls `_run_setup` itself.
        monkeypatch.setattr(
            GlobalMemoryPoolService,
            "_run_setup_on_dedicated_pool",
            AsyncMock(side_effect=RuntimeError("setup blew up")),
        )

        with pytest.raises(RuntimeError, match="setup blew up"):
            await GlobalMemoryPoolService.initialize_pool()

        assert GlobalMemoryPoolService._pool is None
        assert GlobalMemoryPoolService._is_setup_done is False
        fake_pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_log_line_never_includes_the_password(self, monkeypatch, caplog):
        monkeypatch.setattr(client_module.settings, "is_global_memory_enabled", lambda: True)
        monkeypatch.setenv("SQLALCHEMY_DATABASE_URI", "postgresql://user:p@ss@host:5432/db")

        fake_pool = AsyncMock()
        fake_pool.open = AsyncMock()
        monkeypatch.setattr(client_module, "AsyncConnectionPool", MagicMock(return_value=fake_pool))
        monkeypatch.setattr(GlobalMemoryPoolService, "_run_setup", AsyncMock(return_value=None))

        # `client_module.logger` is configured with `propagate=False` (see
        # `utils/logger.py`), so `caplog`'s root-attached handler never sees
        # its records — attach directly to this named logger instead.
        client_module.logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.INFO, logger=client_module.logger.name):
                await GlobalMemoryPoolService.initialize_pool()
        finally:
            client_module.logger.removeHandler(caplog.handler)

        assert "user:p@ss@host" not in caplog.text
        assert "host:5432/db" in caplog.text


class TestGetPool:
    def test_raises_when_pool_is_none(self):
        GlobalMemoryPoolService._pool = None
        GlobalMemoryPoolService._is_setup_done = False
        with pytest.raises(RuntimeError):
            GlobalMemoryPoolService.get_pool()

    def test_raises_when_pool_set_but_setup_not_done(self):
        """The regression this guards: a previous version left `_pool` set
        even when `setup()` had failed, so `get_pool()` kept succeeding
        against a possibly-incomplete schema."""
        GlobalMemoryPoolService._pool = MagicMock()
        GlobalMemoryPoolService._is_setup_done = False
        with pytest.raises(RuntimeError):
            GlobalMemoryPoolService.get_pool()

    def test_returns_pool_once_setup_is_done(self):
        fake_pool = MagicMock()
        GlobalMemoryPoolService._pool = fake_pool
        GlobalMemoryPoolService._is_setup_done = True
        assert GlobalMemoryPoolService.get_pool() is fake_pool


# ---------------------------------------------------------------------------
# _drain_all_clients — never strand a still-busy client by dropping it
# ---------------------------------------------------------------------------


class TestDrainAllClients:
    @pytest.mark.asyncio
    async def test_keeps_busy_clients_referenced_and_evicts_only_idle_ones(self, caplog):
        busy = MagicMock()
        busy.idle = False
        idle = MagicMock()
        idle.idle = True

        client_module._CLIENTS.clear()
        client_module._CLIENTS["busy"] = busy
        client_module._CLIENTS["idle"] = idle
        client_module.logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.WARNING, logger=client_module.logger.name):
                await client_module._drain_all_clients(timeout_s=0.05)

            assert "busy" in client_module._CLIENTS
            assert "idle" not in client_module._CLIENTS
            assert "still had in-flight operations" in caplog.text
        finally:
            client_module.logger.removeHandler(caplog.handler)
            client_module._CLIENTS.clear()
