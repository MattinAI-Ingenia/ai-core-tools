"""
Cross-session user memory — the single store seam (spec `user-memory-mem0`,
FR-9, AD-9, step_005).

**This is the ONLY module in the codebase that may import `langgraph.store`.**
Every other module reaches the store exclusively through `GlobalMemoryClient`
(the async read/write API), the module-level `purge` / `list_app_scopes` /
`list_user_scopes` functions (sync, cascade-triggered erasure/enumeration),
or through `GlobalMemoryPoolService` (pool lifecycle, wired from
`backend/main.py`'s FastAPI lifespan). Nothing here may issue raw SQL against
the store's own data tables (`store`, `store_vectors`) — the two bookkeeping
tables (`store_migrations`, `vector_migrations`) are read directly, once,
only to verify `setup()` actually completed (see
`GlobalMemoryPoolService._assert_migrations_complete`), which is a startup
health check, not a substitute for the store's own API. The one exception is
a single idempotent DDL statement after `setup()`
(`GlobalMemoryPoolService._relax_vector_column`) that makes
`store_vectors.embedding` dimension-agnostic — schema, never row data.

**Vector width is per embedding service, not per deployment.** The store's
`setup()` always creates `embedding vector(<dims>)` plus an HNSW index, which
would force every App onto one embedding width. The store never uses that
index for its own searches (its namespace/filter `WHERE` forces a sequential
scan — see `get_distance_operator` in `langgraph/store/postgres/base.py`) and
casts query vectors without a width (`%s::vector`), so after `setup()` the
index is dropped and the column relaxed to a plain `vector`, exactly like
LangChain's `langchain_pg_embedding`. pgvector still refuses to *compare* two
vectors of different widths, which a namespace can only contain if its App
switched embedding service: every fact therefore records the
`embedding_service_id` it was embedded with, and every vector search filters
on it, so facts from a previous model are simply not candidates (they remain
fully enumerable and erasable, which never touch vectors).

Two empirically-verified constraints from the FR-1 spike (`decisions.md`)
shape everything below:

- `AsyncPostgresStore.from_conn_string(...)` is an async context manager that
  opens and owns its own connection pool. Entering it per turn would open and
  tear down a pool every turn. Every `GlobalMemoryClient` instead shares one
  caller-owned `AsyncConnectionPool`, built once by `GlobalMemoryPoolService`.
- `AsyncPostgresStore` (via `AsyncBatchedBaseStore`) runs one background task
  per instance, bound to the event loop captured at construction time. It
  must therefore only ever be constructed inside a running event loop (never
  at import time), and an instance with operations still queued on that task
  must never be evicted or have its pool closed out from under it — both
  would strand an awaiting coroutine forever. `GlobalMemoryClient` tracks its
  own in-flight count so the module-level cache only evicts idle instances,
  and `GlobalMemoryPoolService.close_pool()` drains before closing the pool.
  Note the division of responsibility: `_inflight` only guards *eviction/
  shutdown* (never dropping an instance with queued work); it is
  `asyncio.wait_for` on each call, not `_inflight`, that bounds how long a
  single chat turn can block on the store — `wait_for` cancels the *waiter*
  on timeout, but the already-dispatched op keeps running to completion on
  the store's background task regardless (see `add()`'s docstring).

A namespace search/list is only ever safe when checked for **exact**
namespace equality after the store returns it. `PostgresStore`/
`AsyncPostgresStore` match every namespace operation (`search`,
`list_namespaces` prefix *and* suffix) with a SQL `LIKE` over the namespace's
text form, never equality — `store.prefix LIKE '{prefix}%'` for a prefix
match, `store.prefix LIKE '%{suffix}'` for a suffix match
(`langgraph/store/postgres/base.py`, `_get_batch_search_queries` /
`_get_batch_list_namespaces_queries`). A prefix search scoped to
`("user_memory", "5", "7")` therefore also matches `"user_memory.5.70"`,
`.71`, `.700`, ...; a suffix list scoped to user `7` also matches any user id
whose decimal form *ends* in `7` (`17`, `27`, `107`, ...). Every method below
that reads or deletes namespace-scoped rows re-checks the namespace
components it got back for exact equality before using them — this is the
seam's own isolation guarantee, not something the store provides.

The namespace root (`settings.namespace_root()`) is reserved for this feature
specifically because these four tables are shared by any other LangGraph
store this system might ever add.
"""
import asyncio
import hashlib
import os
import re
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, List, Optional

import psycopg
from langchain_core.embeddings import Embeddings
from langgraph.store.base import PutOp
from langgraph.store.postgres import AsyncPostgresStore, PostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from utils.logger import get_logger

from . import settings
from .identity import MemoryIdentity
from .settings import FAIL_ERROR, FAIL_TIMEOUT, log_memory_event

logger = get_logger(__name__)

# Bound on cached `GlobalMemoryClient` instances (AD-9). Instances share the
# one pool, so eviction never closes a connection — it only drops a Python
# object (and, per the in-flight guard below, only an idle one).
_MAX_CACHED_CLIENTS = 16

# Purge pages a namespace and deletes in one `abatch` per page rather than
# one unbounded pipelined statement (step_001 measured `abatch` at ~13x a
# per-key `adelete` loop, but a heavy scope must still be capped and paged —
# review-board action item 8).
_PURGE_BATCH_SIZE = 500

# Hard ceiling on `purge`'s pagination loop. Because a neighbouring scope's
# rows are never deleted (only exact-namespace matches are), the loop must
# advance past them explicitly rather than assuming every returned row shrinks
# the remaining set — see `_purge_sync`. This bounds how many pages a single
# purge call will ever walk, so a pathological/adversarial neighbouring scope
# raises instead of spinning forever holding a sync DB connection inside
# `AppService.delete_app` / `UserService.delete_user`.
_PURGE_MAX_ITERATIONS = 10_000

# `list_app_scopes`/`list_user_scopes` page `list_namespaces`; mirrors the
# store's own default `limit`.
_LIST_SCOPES_PAGE_SIZE = 100

# How long `close_pool()` waits for cached clients to drain their in-flight
# operations before closing the shared pool (AD-9). This is a best-effort
# grace period, not a hard guarantee — see `_drain_all_clients`.
_DRAIN_TIMEOUT_S = 5.0
_DRAIN_POLL_INTERVAL_S = 0.05

# Bounded poll for `GlobalMemoryPoolService._run_setup` convergence — see that
# method's docstring for why a single point-in-time re-check is wrong.
_SETUP_CONVERGENCE_TIMEOUT_S = 30.0
_SETUP_POLL_INTERVAL_S = 0.5

# `AsyncPostgresStore` requires `index["dims"]`, but only its `setup()` reads
# it — to size the `vector(<dims>)` column and HNSW index that
# `_relax_vector_column` immediately undoes. It must stay within pgvector's
# 2000-dim HNSW limit or a cold-DB `setup()` fails before it can be relaxed.
# It constrains nothing at runtime: real embeddings of any width are written
# and searched.
_STORE_PLACEHOLDER_DIMS = 1536

# Bounds how long the post-setup DDL waits for its ACCESS EXCLUSIVE lock on
# `store_vectors` on an upgraded, already-serving database; a timeout raises
# into `_run_setup`'s bounded poll and is retried there.
_RELAX_LOCK_TIMEOUT = "10s"

# Modest over-fetch applied to `search`/`similarity` before the exact-namespace
# filter is applied: since the store's own `limit`/`offset` are computed over
# the (buggy) LIKE-matched superset, asking for exactly `top_k` items and then
# discarding foreign-namespace rows can under-fill the result below `top_k`
# even though enough of the caller's *own* facts exist. Over-fetching a
# bounded multiple of the request size makes that under-fill unlikely without
# turning every read into an unbounded query.
_SEARCH_OVERFETCH_MULTIPLIER = 4
_SEARCH_OVERFETCH_MAX = 100
_SIMILARITY_OVERFETCH_LIMIT = 5


class _SetupOnlyEmbeddings(Embeddings):
    """Placeholder `Embeddings` used only to satisfy the store's index-config
    validation while running `setup()` at startup.

    `AsyncPostgresStore.__init__` raises unless `index["embed"]` is a real
    `Embeddings` instance (`langgraph.store.base.embed.ensure_embeddings`
    rejects `None` outright) — but `setup()` itself never calls `embed_query`
    / `embed_documents`; it only reads `index["dims"]` (and, for the ANN
    index migration, `index_config.get("ann_index_config", {})`) to size and
    configure the `store_vectors.embedding` column. Verified directly against
    `langgraph-checkpoint-postgres`'s installed `aio.py`/`base.py` (pinned to
    the `>=3.0` version this project depends on — a future major version
    bump must re-verify this). Calling either method on this placeholder is
    therefore always a programming error.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError(
            "_SetupOnlyEmbeddings must never be used for a real embedding call — "
            "it exists only to satisfy AsyncPostgresStore's setup()-time validation."
        )

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError(
            "_SetupOnlyEmbeddings must never be used for a real embedding call — "
            "it exists only to satisfy AsyncPostgresStore's setup()-time validation."
        )


def _get_db_uri() -> str:
    """The same DSN convention `CheckpointerCacheService` uses (`agent_cache_service.py:84`)."""
    return os.getenv(
        "SQLALCHEMY_DATABASE_URI",
        "postgresql://iacoretoolsdev:iacoretoolsdev@localhost:5432/iacoretoolsdev",
    )


def _assert_identity(identity: MemoryIdentity) -> None:
    """Re-check the AD-2 namespace invariant at the seam, defense in depth.

    `MemoryIdentity.__post_init__` already asserts this at construction time
    (identity.py), and the dataclass is frozen, so this can only fail if a
    future caller manages to hand the seam something that merely looks like a
    `MemoryIdentity`. Every public method below calls this first, inside its
    own fault-containment `try`, so a violation degrades to the method's
    neutral return value plus a `FAIL_ERROR` log line rather than propagating
    into a chat turn (NFR-4) — never as an unguarded assertion.

    Also asserts `app_id`/`user_id` are real positive `int`s (rejecting
    `bool`, which is an `int` subclass in Python) before trusting
    `identity.namespace` at all: a `str(app_id)` of `"%"` or `"_"` would
    otherwise make `("%", "%")` shape-valid, and both characters are SQL
    `LIKE` metacharacters — a namespace built from one would match far more
    than the caller's own scope, up to a whole-deployment read.
    """
    if not isinstance(identity, MemoryIdentity):
        raise AssertionError("GlobalMemoryClient methods accept only a MemoryIdentity instance.")
    _assert_positive_int(identity.app_id, "MemoryIdentity.app_id")
    _assert_positive_int(identity.user_id, "MemoryIdentity.user_id")
    expected = (settings.namespace_root(), str(identity.app_id), str(identity.user_id))
    if identity.namespace != expected:
        raise AssertionError(
            f"MemoryIdentity.namespace {identity.namespace!r} does not match the expected "
            f"{expected!r} — refusing to issue a store call with a mismatched namespace."
        )


def _assert_positive_int(value: Any, label: str) -> None:
    """Shared guard for `app_id`/`user_id`-shaped values (see `_assert_identity`).

    True only for a real positive `int` — never a `bool` (an `int` subclass),
    never a numeric string. Every namespace-scoped entry point in this module
    (the `MemoryIdentity` fields, and the raw `app_id`/`user_id` parameters
    `list_app_scopes`/`list_user_scopes` take directly) must pass this before
    the value is used to build a `LIKE` prefix/suffix — see the module
    docstring on why an unchecked value is a whole-deployment read risk.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AssertionError(f"{label} must be a positive int, got {value!r}")


def fact_key(fact: str) -> str:
    """Public name of `_fact_key`: the store key a fact's text maps to."""
    return _fact_key(fact)


def _fact_key(fact: str) -> str:
    """Deterministic per-fact key (AD-8 stage 2): a truncated SHA-256 of the
    normalised (lowercased, whitespace-collapsed) fact text, so writing an
    identical fact again overwrites the same row instead of accumulating.
    """
    normalized = " ".join(fact.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


@contextmanager
def _sync_connection() -> Iterator[psycopg.Connection]:
    """Open a short-lived sync `psycopg` connection for `purge`/`list_app_scopes`/
    `list_user_scopes`.

    Per the FR-1 spike (`decisions.md`): the sync `PostgresStore` and the
    async `AsyncPostgresStore` agree on the same SQL schema, but a sync
    `psycopg.Connection` cannot be pulled from the async `AsyncConnectionPool`
    `GlobalMemoryClient` otherwise uses — so these two sync call sites (both
    rare, cascade-triggered operations, never a per-turn hot path) simply open
    and close their own ordinary connection rather than sharing a pool.

    Bounded with `connect_timeout` (connection establishment) and a
    server-side `statement_timeout` (query execution): an unreachable or
    lock-contended database must not hang the delete cascade indefinitely,
    holding a threadpool worker and the caller's own open SQLAlchemy
    transaction hostage. Callers (`AppService.delete_app`,
    `UserService.delete_user`, and step_006's erasure endpoint) must invoke
    the functions built on this connection from **outside** their own DB
    transaction, and never from the async request path — this is a
    synchronous, blocking connection and would block the event loop if
    called from an `async def`.
    """
    conn = psycopg.connect(
        _get_db_uri(),
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
        connect_timeout=5,
        options="-c statement_timeout=30000",
    )
    try:
        yield conn
    finally:
        conn.close()


class GlobalMemoryPoolService:
    """Owns the single `AsyncConnectionPool` every `GlobalMemoryClient` shares (AD-9).

    Wired from `backend/main.py`'s FastAPI lifespan, alongside
    `CheckpointerCacheService`: `initialize_pool()` on startup,
    `close_pool()` on shutdown. Deliberately **not** a request-scoped
    dependency — the whole point of AD-9 is that this pool is created once
    and never entered/exited per turn.
    """

    _pool: Optional[AsyncConnectionPool] = None
    _is_setup_done = False

    @classmethod
    async def initialize_pool(cls) -> None:
        """Open the shared pool and run the store's `setup()` migrations, once.

        Sized small (min 1 / max 3) per AD-9: the batched store funnels every
        operation through one background task, so it needs roughly one
        concurrent connection, and this deployment's connection budget is
        already tight (`agent_cache_service.py` alone takes `max_size=10`
        against a default `max_connections=100` with two `uvicorn` workers).

        `autocommit=True` is not optional — copied verbatim from
        `agent_cache_service.py:96-109` — step_001 proved that without it the
        store's `CREATE INDEX CONCURRENTLY` migrations fail outright with
        `psycopg.errors.ActiveSqlTransaction`.

        Skips entirely when the deployment-wide kill switch is off
        (`settings.is_global_memory_enabled()`): if the feature is disabled
        for this process there is no reason to open a pool or run migrations
        against the database at all.

        On any failure opening the pool or completing setup, the pool is
        closed and `_pool`/`_is_setup_done` are reset before the exception is
        re-raised: `main.py`'s lifespan catches this and logs "memory
        inactive", but previously left `_pool` set to a pool whose schema
        might be incomplete — `get_pool()` would then keep succeeding, and
        memory would run against an incomplete schema (failing per turn)
        instead of being cleanly off for the process.
        """
        if not settings.is_global_memory_enabled():
            return
        if cls._pool is not None:
            return

        db_uri = _get_db_uri()
        logger.info(
            "Initializing global memory connection pool (DB: %s)",
            db_uri.rsplit("@", 1)[-1] if "@" in db_uri else "hidden",
        )

        # Server-side backstop for work the caller has already abandoned.
        # `asyncio.wait_for` cancels the *waiter*, not the query: the batched
        # store's background task still executes the op, so without this a slow
        # provider or a lock-blocked DB keeps one of only three connections busy
        # long after the turn gave up — and three such ops starve every
        # subsequent memory operation. Set above the longest app-side budget so
        # the app-side timeout normally fires first and this only reaps the
        # leftovers. `setup()` deliberately does not inherit it (see below).
        statement_timeout_ms = int(settings.write_timeout_s() * 1000) + 5000

        cls._pool = AsyncConnectionPool(
            conninfo=db_uri,
            min_size=1,
            max_size=3,
            max_lifetime=1800,
            max_idle=300,
            reconnect_timeout=30,
            open=False,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
                "options": f"-c statement_timeout={statement_timeout_ms}",
            },
        )
        try:
            await cls._pool.open(wait=True)
            await cls._run_setup_on_dedicated_pool(db_uri)
        except Exception:
            failed_pool = cls._pool
            cls._pool = None
            cls._is_setup_done = False
            if failed_pool is not None:
                try:
                    await failed_pool.close()
                except Exception:
                    logger.warning(
                        "Failed to close the global memory pool after a failed setup()",
                        exc_info=True,
                    )
            raise

        cls._is_setup_done = True
        logger.info("Global memory connection pool initialized successfully")

    @classmethod
    async def _run_setup_on_dedicated_pool(cls, db_uri: str) -> None:
        """Run `_run_setup` on a throwaway pool that has **no** `statement_timeout`.

        The main pool caps statement duration so an abandoned turn cannot hold a
        connection indefinitely — but the store's own migrations include
        `CREATE INDEX CONCURRENTLY`, and `_relax_vector_column`'s `ALTER TABLE`
        rewrites `store_vectors`; on an upgraded client database with a populated
        store either can legitimately run far longer than any per-turn budget,
        and a migration aborted mid-flight leaves the schema half-applied. So
        setup gets its own connection, closed as soon as it converges.
        """
        setup_pool = AsyncConnectionPool(
            conninfo=db_uri,
            min_size=1,
            max_size=1,
            open=False,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        )
        try:
            await setup_pool.open(wait=True)
            await cls._run_setup(setup_pool)
        finally:
            try:
                await setup_pool.close()
            except Exception:
                logger.warning("Failed to close the memory setup pool", exc_info=True)

    @classmethod
    async def _run_setup(cls, pool: AsyncConnectionPool) -> None:
        """Run `store.setup()` until the schema demonstrably converges, tolerating
        the two-worker cold-start race (FR-1 spike item 2).

        Two workers racing `setup()` on a cold DB commonly raise
        `psycopg.errors.UniqueViolation` on the version-bookkeeping INSERT —
        but a concurrent `CREATE INDEX CONCURRENTLY IF NOT EXISTS` of the same
        index name can just as easily surface as `DuplicateTable`, or block
        briefly on a lock, so this catches `psycopg.Error` broadly rather than
        one specific subclass.

        Critically, this is a **bounded poll**, not a single point-in-time
        re-check: catching the race once and immediately calling
        `_assert_migrations_complete` would see whatever version the *other*
        worker has reached so far (often still 0 or 1, not the final 3/2) and
        raise — permanently marking this worker's memory inactive over a
        transient cold-start race, the exact opposite of what this check
        exists to prevent. So: re-call `setup()` (a clean no-op once the
        schema is actually complete, per step_001) and re-verify
        `_assert_migrations_complete` in a loop, backing off
        `_SETUP_POLL_INTERVAL_S` between attempts, for up to
        `_SETUP_CONVERGENCE_TIMEOUT_S` before finally raising — by which point
        a merely-slow concurrent `setup()` has had ample time to finish, and a
        genuinely failed one is reported instead of silently retried forever.
        """
        setup_store = AsyncPostgresStore(
            conn=pool,
            index={
                "dims": _STORE_PLACEHOLDER_DIMS,
                "embed": _SetupOnlyEmbeddings(),
                "fields": ["text"],
            },
        )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + _SETUP_CONVERGENCE_TIMEOUT_S
        last_error: Optional[BaseException] = None

        while True:
            try:
                await setup_store.setup()
            except psycopg.Error as exc:
                last_error = exc
                logger.warning(
                    "Global memory store setup() raced with another worker (%s); "
                    "re-verifying migration completeness before retrying",
                    exc.__class__.__name__,
                )

            try:
                await cls._assert_migrations_complete(setup_store)
                await cls._relax_vector_column(pool)
                return
            except (RuntimeError, psycopg.Error) as exc:
                last_error = exc

            if loop.time() >= deadline:
                raise RuntimeError(
                    "Global memory store migrations did not converge within "
                    f"{_SETUP_CONVERGENCE_TIMEOUT_S}s of polling — another worker's "
                    f"setup() may be stalled or failed. Last observed error: {last_error}"
                ) from last_error

            await asyncio.sleep(_SETUP_POLL_INTERVAL_S)

    @classmethod
    async def _assert_migrations_complete(cls, setup_store: AsyncPostgresStore) -> None:
        """Verify `store_migrations`/`vector_migrations` both reached their last
        version. Reading these two bookkeeping tables directly (never
        `store`/`store_vectors`) is a one-time startup health check, not a
        substitute for the store's own API.

        Raises `RuntimeError` when incomplete. Called from within
        `_run_setup`'s bounded poll — a single raise here no longer means
        "give up", only "not converged yet, try again unless the deadline has
        passed".
        """
        expected_store_version = len(setup_store.MIGRATIONS) - 1
        expected_vector_version = len(setup_store.VECTOR_MIGRATIONS) - 1

        async with cls._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("SELECT max(v) AS v FROM store_migrations")
                row = await cur.fetchone()
                store_version = row["v"] if row else None

                await cur.execute("SELECT max(v) AS v FROM vector_migrations")
                row = await cur.fetchone()
                vector_version = row["v"] if row else None

        if store_version != expected_store_version or vector_version != expected_vector_version:
            raise RuntimeError(
                "Global memory store migrations are incomplete after setup(): "
                f"store_migrations at {store_version} (expected {expected_store_version}), "
                f"vector_migrations at {vector_version} (expected {expected_vector_version}). "
                "Another worker's setup() may still be applying migrations, or may have failed."
            )

    @classmethod
    async def _relax_vector_column(cls, pool: AsyncConnectionPool) -> None:
        """Make `store_vectors.embedding` accept vectors of any width (see the
        module docstring for why this is safe for the store's own queries).

        Idempotent and race-safe across workers: it is a no-op once the column
        is already a plain `vector` and the index is gone, and both statements
        run in one transaction under the same ACCESS EXCLUSIVE table lock, so
        two workers serialise rather than deadlock. Runs on the setup pool (no
        `statement_timeout`), bounded instead by `_RELAX_LOCK_TIMEOUT`.

        `store.setup()` never re-runs a recorded vector migration, so the
        dropped index and widened column are never recreated by later startups.
        """
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT format_type(a.atttypid, a.atttypmod) AS column_type, "
                    "to_regclass('store_vectors_embedding_idx') IS NOT NULL AS has_index "
                    "FROM pg_attribute a "
                    "WHERE a.attrelid = 'store_vectors'::regclass AND a.attname = 'embedding'"
                )
                row = await cur.fetchone()
            if row is None:
                raise RuntimeError("store_vectors.embedding not found after setup()")
            if row["column_type"] == "vector" and not row["has_index"]:
                return

            async with conn.transaction():
                await conn.execute(f"SET LOCAL lock_timeout = '{_RELAX_LOCK_TIMEOUT}'")
                await conn.execute("DROP INDEX IF EXISTS store_vectors_embedding_idx")
                await conn.execute("ALTER TABLE store_vectors ALTER COLUMN embedding TYPE vector")
        logger.info("Global memory store_vectors.embedding relaxed to accept any embedding width")

    @classmethod
    async def close_pool(cls) -> None:
        """Drain cached clients, then close the shared pool (AD-9 shutdown ordering)."""
        await _drain_all_clients()
        if cls._pool is not None:
            await cls._pool.close()
            logger.info("Global memory connection pool closed")
        cls._pool = None
        cls._is_setup_done = False

    @classmethod
    def get_pool(cls) -> AsyncConnectionPool:
        """Return the shared pool. Raises if `initialize_pool()` was never
        called (kill switch off), has not completed yet, or failed (in which
        case `_pool` is `None` again — see `initialize_pool`'s cleanup)."""
        if cls._pool is None or not cls._is_setup_done:
            raise RuntimeError(
                "Global memory pool not initialized. Call "
                "GlobalMemoryPoolService.initialize_pool() during application startup, "
                "or the kill switch is off for this process, or a previous setup() failed."
            )
        return cls._pool


@dataclass(frozen=True)
class RetrievedFact:
    """One fact returned by a read-path search, with the provenance the
    rendered memory block needs (when it was learned, and in which
    conversation) to let the model resolve conflicts by recency."""

    text: str
    created_at: Optional[str]
    conversation_id: Optional[int]
    key: Optional[str] = None


@dataclass(frozen=True)
class ClosestFact:
    """The caller's stored fact most similar to a candidate (dedup stage).
    `same_fact` is True when it is the very row the candidate would write
    (same normalised text, hence same `_fact_key`)."""

    score: float
    key: str
    same_fact: bool


def _to_retrieved(item: Any) -> Optional[RetrievedFact]:
    """A store item as a `RetrievedFact`, or `None` if it carries no text."""
    value = item.value or {}
    text = value.get("text")
    if not isinstance(text, str) or not text:
        return None
    created_at = value.get("created_at")
    conversation_id = value.get("conversation_id")
    return RetrievedFact(
        key=item.key,
        text=text,
        created_at=created_at if isinstance(created_at, str) else None,
        conversation_id=(
            conversation_id
            if isinstance(conversation_id, int) and not isinstance(conversation_id, bool)
            else None
        ),
    )


class GlobalMemoryClient:
    """The one wrapper around `AsyncPostgresStore` every memory read/write goes
    through (AD-1, AD-9). Every method takes a `MemoryIdentity` and nothing
    else identity-shaped (AD-2, AC-12), starts with the namespace invariant
    check, and catches every exception so a memory failure never reaches a
    chat turn (NFR-4).

    Construct only via `get_memory_client()` — never directly — so
    construction always happens inside a running event loop (required by
    `AsyncBatchedBaseStore`, which captures `asyncio.get_running_loop()` at
    `__init__`) and so the module-level cache is the only place instances are
    kept alive.

    `purge`/`list_app_scopes`/`list_user_scopes` are deliberately **not**
    methods on this class — see the module-level functions below. Erasure and
    enumeration must work even when the feature is disabled or its pool
    failed to initialize (facts already written before either of those things
    happened must still be erasable/discoverable), so they must never be
    reached only through `get_memory_client()`, which fails soft (`None`)
    exactly in those cases.
    """

    __slots__ = ("_store", "_embedding_service_id", "_inflight")

    def __init__(self, pool: AsyncConnectionPool, embeddings: Any, embedding_service_id: int) -> None:
        self._store = AsyncPostgresStore(
            conn=pool,
            index={
                "dims": _STORE_PLACEHOLDER_DIMS,
                "embed": embeddings,
                "fields": ["text"],
            },
        )
        self._embedding_service_id = embedding_service_id
        self._inflight = 0

    @property
    def idle(self) -> bool:
        """True when no operation issued through this instance is still in flight.

        Used by the module-level cache to decide whether an instance is safe
        to evict, and by shutdown to decide when draining is complete —
        evicting (or closing the pool under) an instance with queued
        operations would strand the awaiting coroutine forever, since the
        store's background task is cancelled with no way to resume it
        (AD-9, review-board action item 4). This guards *eviction/shutdown*
        only: the per-call `asyncio.wait_for` timeouts (see `search`/`add`/
        `similarity`) are what bound how long any single chat turn can block
        on the store — `_inflight` says nothing about per-call latency, only
        about whether it is currently safe to drop this instance.
        """
        return self._inflight == 0

    @property
    def _vector_filter(self) -> dict:
        """`asearch` filter restricting candidates to facts embedded by this
        client's embedding service — pgvector raises on comparing vectors of
        different widths, and vectors from different models are meaningless
        to compare even at equal width (D-16). See the module docstring."""
        return {"embedding_service_id": self._embedding_service_id}

    async def search(self, identity: MemoryIdentity, query: str, top_k: int) -> List[str]:
        """Text-only view of `search_facts` (same contract, same fault containment)."""
        return [fact.text for fact in await self.search_facts(identity, query, top_k)]

    async def session_facts(
        self, identity: MemoryIdentity, conversation_id: int, limit: int
    ) -> List[RetrievedFact]:
        """Facts learned in one conversation (`value.conversation_id`), most
        recently updated first, at most `limit`.

        A filter-only `asearch` (no query): a plain SQL filter on the stored
        value ordered by `updated_at`, so no embedding call. Same exact-
        namespace re-check as `search_facts`. Not restricted to this client's
        embedding service: these facts are shown to the extractor as text and
        only ever deleted by key. Fault-contained: returns `[]` on any failure.
        """
        if isinstance(conversation_id, bool) or not isinstance(conversation_id, int) or limit <= 0:
            return []
        self._inflight += 1
        try:
            _assert_identity(identity)
            items = await asyncio.wait_for(
                self._store.asearch(
                    identity.namespace, filter={"conversation_id": conversation_id},
                    limit=min(limit * _SEARCH_OVERFETCH_MULTIPLIER, _SEARCH_OVERFETCH_MAX),
                ),
                timeout=settings.read_timeout_s(),
            )
            own_items = [item for item in items if tuple(item.namespace) == tuple(identity.namespace)]
            return [fact for fact in (_to_retrieved(item) for item in own_items) if fact is not None][:limit]
        except asyncio.TimeoutError:
            log_memory_event(
                "session_facts", app_id=identity.app_id, agent_id=identity.agent_id,
                active=False, reason=FAIL_TIMEOUT, level="warning",
            )
            return []
        except Exception:
            log_memory_event(
                "session_facts", app_id=getattr(identity, "app_id", None), agent_id=getattr(identity, "agent_id", None),
                active=False, reason=FAIL_ERROR, level="warning",
            )
            return []
        finally:
            self._inflight -= 1

    async def search_facts(self, identity: MemoryIdentity, query: str, top_k: int) -> List[RetrievedFact]:
        """Return up to `top_k` facts from `identity`'s namespace, best match first,
        each with its `created_at` and originating `conversation_id` (either
        `None` for facts written before that provenance existed).

        Fault-contained: any error, timeout, or namespace-invariant violation
        returns `[]` rather than raising.

        `AsyncPostgresStore.asearch` matches namespaces with a SQL `LIKE`
        prefix, not equality (see the module docstring) — a search scoped to
        `("user_memory", "5", "7")` would also return rows from
        `("user_memory", "5", "70")`. Every returned item is therefore
        re-checked for exact namespace equality *before* its value is read,
        and over-fetched by `_SEARCH_OVERFETCH_MULTIPLIER` so that discarding
        foreign-namespace rows does not silently under-fill below `top_k`.
        """
        self._inflight += 1
        try:
            _assert_identity(identity)
            fetch_limit = min(max(top_k * _SEARCH_OVERFETCH_MULTIPLIER, top_k), _SEARCH_OVERFETCH_MAX)
            items = await asyncio.wait_for(
                self._store.asearch(
                    identity.namespace, query=query, filter=self._vector_filter, limit=fetch_limit
                ),
                timeout=settings.read_timeout_s(),
            )
            own_items = [item for item in items if tuple(item.namespace) == tuple(identity.namespace)]
            return [fact for fact in (_to_retrieved(item) for item in own_items[:top_k]) if fact is not None]
        except asyncio.TimeoutError:
            log_memory_event(
                "search", app_id=identity.app_id, agent_id=identity.agent_id,
                active=False, reason=FAIL_TIMEOUT, level="warning",
            )
            return []
        except Exception:
            log_memory_event(
                "search", app_id=getattr(identity, "app_id", None), agent_id=getattr(identity, "agent_id", None),
                active=False, reason=FAIL_ERROR, level="warning",
            )
            return []
        finally:
            self._inflight -= 1

    async def add(self, identity: MemoryIdentity, facts: List[str], conversation_id: Optional[int] = None) -> int:
        """Persist each fact string under `identity`'s namespace; return how many
        were written.

        No `infer` parameter — dedup is the write path's own stage (AD-8);
        this method only persists. Each fact's key is a deterministic hash of
        its normalised text (`_fact_key`), so writing an identical fact again
        overwrites rather than accumulates. The whole batch is bounded by
        `GLOBAL_MEMORY_WRITE_TIMEOUT_S` (the fire-and-forget write path's own
        budget) — any timeout or error returns `0` rather than raising.

        A returned `0` after a timeout does not guarantee nothing was
        written: `asyncio.wait_for` only cancels the *waiting* coroutine on
        timeout, it does not stop an already-dispatched `aput` from finishing
        on the store's background batch task — a mid-batch timeout can
        therefore return `0` after some facts already landed. This is benign
        *here* only because `_fact_key` makes every `aput` idempotent (a
        retry overwrites the same row rather than duplicating it) — callers
        (step_008's write path) must not read a `0` return as license to
        retry with newly-generated keys, or a timeout could accumulate
        duplicate rows under different keys for the same fact.
        """
        if not facts:
            return 0
        self._inflight += 1
        try:
            _assert_identity(identity)

            async def _write_all() -> int:
                written = 0
                for fact in facts:
                    key = _fact_key(fact)
                    value = {
                        "text": fact,
                        "agent_id": identity.agent_id,
                        "embedding_service_id": self._embedding_service_id,
                        "conversation_id": conversation_id,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    await self._store.aput(identity.namespace, key, value)
                    written += 1
                return written

            return await asyncio.wait_for(_write_all(), timeout=settings.write_timeout_s())
        except asyncio.TimeoutError:
            log_memory_event(
                "add", app_id=identity.app_id, agent_id=identity.agent_id,
                active=False, reason=FAIL_TIMEOUT, level="warning",
            )
            return 0
        except Exception:
            log_memory_event(
                "add", app_id=getattr(identity, "app_id", None), agent_id=getattr(identity, "agent_id", None),
                active=False, reason=FAIL_ERROR, level="warning",
            )
            return 0
        finally:
            self._inflight -= 1

    async def similarity(self, identity: MemoryIdentity, fact: str) -> Optional[float]:
        """Score-only view of `closest` (same contract, same fault containment)."""
        found = await self.closest(identity, fact)
        return found.score if found is not None else None

    async def closest(self, identity: MemoryIdentity, fact: str) -> Optional[ClosestFact]:
        """Return the stored fact most similar to `fact` (its score and key) within `identity`'s
        namespace, or `None` if there is no candidate at all (empty
        namespace) or the operation failed.

        This is AD-8 stage 2's single-candidate score: the store's
        `SearchItem.score` (confirmed present and populated by a real vector
        search — `decisions.md` item 5), not an aggregate.

        Same `LIKE`-prefix caveat as `search`: a namespace's neighbouring
        scope could otherwise rank above the caller's own top match and get
        returned as if it were the caller's candidate. `similarity` over-fetches
        a small number of candidates and discards anything outside
        `identity`'s exact namespace before taking the top score, rather than
        trusting the store's own `limit=1`.
        """
        self._inflight += 1
        try:
            _assert_identity(identity)
            items = await asyncio.wait_for(
                self._store.asearch(
                    identity.namespace, query=fact, filter=self._vector_filter, limit=_SIMILARITY_OVERFETCH_LIMIT
                ),
                timeout=settings.read_timeout_s(),
            )
            own_items = [item for item in items if tuple(item.namespace) == tuple(identity.namespace)]
            if not own_items or own_items[0].score is None:
                return None
            top = own_items[0]
            return ClosestFact(score=top.score, key=top.key, same_fact=top.key == _fact_key(fact))
        except asyncio.TimeoutError:
            log_memory_event(
                "similarity", app_id=identity.app_id, agent_id=identity.agent_id,
                active=False, reason=FAIL_TIMEOUT, level="warning",
            )
            return None
        except Exception:
            log_memory_event(
                "similarity", app_id=getattr(identity, "app_id", None), agent_id=getattr(identity, "agent_id", None),
                active=False, reason=FAIL_ERROR, level="warning",
            )
            return None
        finally:
            self._inflight -= 1

    async def delete_keys(self, identity: MemoryIdentity, keys: List[str]) -> int:
        """Delete the given fact keys from `identity`'s exact namespace; return
        how many deletes were issued (0 on any failure — never raises).

        Used by the write path to drop a near-duplicate older fact once its
        newer replacement is stored. The store's delete (`PutOp(..., None)`)
        matches the namespace with SQL equality, so a key can never reach a
        neighbouring scope; `store_vectors` rows go by `ON DELETE CASCADE`.
        """
        keys = [k for k in dict.fromkeys(keys) if isinstance(k, str) and _FACT_KEY_RE.match(k)]
        if not keys:
            return 0
        self._inflight += 1
        try:
            _assert_identity(identity)
            await asyncio.wait_for(
                self._store.abatch([PutOp(identity.namespace, key, None) for key in keys]),
                timeout=settings.write_timeout_s(),
            )
            return len(keys)
        except asyncio.TimeoutError:
            log_memory_event(
                "delete", app_id=identity.app_id, agent_id=identity.agent_id,
                active=False, reason=FAIL_TIMEOUT, level="warning",
            )
            return 0
        except Exception:
            log_memory_event(
                "delete", app_id=getattr(identity, "app_id", None), agent_id=getattr(identity, "agent_id", None),
                active=False, reason=FAIL_ERROR, level="warning",
            )
            return 0
        finally:
            self._inflight -= 1


@dataclass(frozen=True)
class PurgeResult:
    """Outcome of a `purge()` call — deliberately not a bare `int`.

    `deleted` alone cannot distinguish "nothing to purge" from "a failure
    interrupted the walk before anything (or everything) was removed" — both
    previously returned `0`. `complete` makes that distinction explicit:

    - `complete=True`: the namespace was walked to the end (an empty or
      short final page). `deleted` is the exact, final count of rows
      removed — `0` is a legitimate success here (AC-28, an
      already-empty/already-purged namespace).
    - `complete=False`: the walk was interrupted (identity validation
      failed, or an exception was raised mid-page). `deleted` reflects
      whatever was actually removed *before* the interruption — it must
      **not** be read as the total, and callers (step_006's GDPR erasure)
      must surface this as a failed/partial erasure, never as a `200`
      success with an inflated or misleading count.
    """

    deleted: int
    complete: bool


class _PurgeIncomplete(Exception):
    """Internal signal from `_purge_sync` that the walk was interrupted after
    deleting `deleted` rows. Never escapes `purge()` — it always translates
    this into a `PurgeResult(deleted, complete=False)`.
    """

    def __init__(self, deleted: int, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.deleted = deleted
        self.cause = cause


def purge(identity: MemoryIdentity) -> PurgeResult:
    """Delete every fact in `identity`'s namespace; return a `PurgeResult`.
    Sync — called from the sync `AppService.delete_app` /
    `UserService.delete_user` cascades (step_006).

    Deliberately a module-level function, not a `GlobalMemoryClient` method:
    erasure must work even when the feature is disabled or the shared pool
    failed to initialize (facts written before either of those things
    happened must still be erasable) — see the `GlobalMemoryClient` class
    docstring. It never touches `GlobalMemoryPoolService` or the kill switch.

    Uses the sync `PostgresStore` against its own short-lived connection
    (`_sync_connection`, never any `GlobalMemoryClient` instance's async
    store/pool), paging the namespace and deleting a page at a time via one
    `batch()` call (`store.batch([PutOp(ns, key, None), ...])`) rather than a
    per-key `adelete` loop: step_001 measured `abatch` at ~13x faster.
    `store_vectors` rows are removed by its `ON DELETE CASCADE` on `store` —
    no raw SQL against either table.

    `PostgresStore.search` matches namespaces with a `LIKE` prefix (see the
    module docstring), so a page can contain rows from a neighbouring scope
    (e.g. `("user_memory", "5", "70")` when purging `("user_memory", "5",
    "7")`). Only rows whose namespace is an **exact** match to `identity`'s
    are ever deleted — foreign rows are left untouched and never counted.
    Idempotent: an empty/already-purged namespace returns
    `PurgeResult(0, complete=True)`, a success (AC-28), not an error. Any
    failure returns `PurgeResult(<partial count>, complete=False)` — never
    conflated with a no-op success (NFR-5).
    """
    try:
        _assert_identity(identity)
    except Exception:
        log_memory_event(
            "purge", app_id=getattr(identity, "app_id", None), agent_id=getattr(identity, "agent_id", None),
            active=False, reason=FAIL_ERROR, level="warning",
        )
        return PurgeResult(deleted=0, complete=False)

    try:
        deleted = _purge_sync(identity)
        return PurgeResult(deleted=deleted, complete=True)
    except _PurgeIncomplete as exc:
        log_memory_event(
            "purge", app_id=identity.app_id, agent_id=identity.agent_id,
            active=False, count=exc.deleted, reason=FAIL_ERROR, level="warning",
        )
        return PurgeResult(deleted=exc.deleted, complete=False)
    except Exception:
        log_memory_event(
            "purge", app_id=identity.app_id, agent_id=identity.agent_id,
            active=False, reason=FAIL_ERROR, level="warning",
        )
        return PurgeResult(deleted=0, complete=False)


def _purge_sync(identity: MemoryIdentity) -> int:
    """Page `identity`'s `LIKE`-matched namespace bucket and delete only the
    rows whose namespace is an exact match, returning the exact deleted count.

    Pagination cannot assume "every returned row shrinks the remaining set at
    offset 0" (the previous design's bug): a neighbouring scope's rows are
    never deleted, so they would keep re-appearing at the same offset
    forever. Instead, `offset` only advances past the *foreign* rows seen so
    far in this page (rows we deliberately never delete); rows we do delete
    shrink the ranked set ahead of everything not yet advanced past, so the
    next page's `offset` correctly lands on the next not-yet-seen rows. A
    page shorter than `_PURGE_BATCH_SIZE` means there is nothing left to page
    through, and ends the walk. `_PURGE_MAX_ITERATIONS` is a hard backstop in
    case that invariant is ever violated (e.g. a future store version's
    pagination semantics change) — this raises rather than spinning forever
    holding the delete cascade's DB connection.

    Raises `_PurgeIncomplete(deleted_so_far, cause)` on any error so `purge()`
    can report a true partial count instead of silently returning `0`.
    """
    total = 0
    try:
        with _sync_connection() as conn:
            store = PostgresStore(conn=conn)
            offset = 0
            iterations = 0
            while True:
                iterations += 1
                if iterations > _PURGE_MAX_ITERATIONS:
                    raise RuntimeError(
                        f"purge() exceeded {_PURGE_MAX_ITERATIONS} pagination iterations for "
                        f"namespace {identity.namespace!r} — aborting rather than looping "
                        "indefinitely; a neighbouring scope may be unexpectedly large."
                    )

                items = store.search(identity.namespace, limit=_PURGE_BATCH_SIZE, offset=offset)
                if not items:
                    break

                exact = [item for item in items if tuple(item.namespace) == tuple(identity.namespace)]
                foreign_count = len(items) - len(exact)

                if exact:
                    ops = [PutOp(item.namespace, item.key, None) for item in exact]
                    store.batch(ops)
                    total += len(exact)

                offset += foreign_count

                if len(items) < _PURGE_BATCH_SIZE:
                    break
    except Exception as exc:
        raise _PurgeIncomplete(total, exc) from exc
    return total


# Upper bound on how many facts `list_facts` returns for one scope. A scope
# only grows by at most `max_facts_per_extraction()` per write-due turn and
# dedup keeps it from accumulating repeats, so a real user is far below this;
# it bounds the response for a pathological scope instead of streaming an
# unbounded list to the browser.
_LIST_FACTS_MAX = 500

# `_fact_key` output: the first 32 hex chars of a SHA-256.
_FACT_KEY_RE = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class StoredFact:
    """One remembered fact as shown to the user it is about."""

    key: str
    text: str
    agent_id: Optional[int]
    created_at: Optional[str]


def list_facts(identity: MemoryIdentity) -> List[StoredFact]:
    """Return the facts stored in `identity`'s exact namespace, newest first.

    Sync and module-level for the same reason as `purge`: a user must be able
    to see what is held about them even when the feature is disabled or its
    pool failed to start. Same `LIKE`-prefix caveat as everywhere else — only
    rows whose namespace is an exact match are returned. Raises on failure
    (unlike the turn-path methods) so the caller can answer 503 rather than
    show an empty list that looks like "nothing remembered".
    """
    _assert_identity(identity)
    facts: List[StoredFact] = []
    with _sync_connection() as conn:
        store = PostgresStore(conn=conn)
        offset = 0
        while len(facts) < _LIST_FACTS_MAX:
            items = store.search(identity.namespace, limit=_LIST_SCOPES_PAGE_SIZE, offset=offset)
            for item in items:
                if tuple(item.namespace) != tuple(identity.namespace):
                    continue
                value = item.value or {}
                text = value.get("text")
                if not isinstance(text, str) or not text:
                    continue
                agent_id = value.get("agent_id")
                facts.append(
                    StoredFact(
                        key=item.key,
                        text=text,
                        agent_id=agent_id if isinstance(agent_id, int) and not isinstance(agent_id, bool) else None,
                        created_at=value.get("created_at") if isinstance(value.get("created_at"), str) else None,
                    )
                )
            offset += len(items)
            if len(items) < _LIST_SCOPES_PAGE_SIZE:
                break
    facts.sort(key=lambda f: f.created_at or "", reverse=True)
    return facts[:_LIST_FACTS_MAX]


def delete_fact(identity: MemoryIdentity, key: str) -> bool:
    """Delete one fact from `identity`'s exact namespace; return whether it existed.

    The store's `get` and its delete (`PutOp(..., None)`) both match the
    namespace with SQL **equality** (`prefix = %s`), not the `LIKE` used by
    search/list, so a key can never reach a neighbouring scope's row — and
    `key` is additionally required to be a `_fact_key`-shaped value. Raises
    on store failure so the caller can answer 503.
    """
    _assert_identity(identity)
    if not isinstance(key, str) or not _FACT_KEY_RE.match(key):
        raise ValueError("Invalid fact key")
    with _sync_connection() as conn:
        store = PostgresStore(conn=conn)
        if store.get(identity.namespace, key) is None:
            return False
        store.batch([PutOp(identity.namespace, key, None)])
    return True


@dataclass(frozen=True)
class ScopeListResult:
    """Outcome of `list_app_scopes`/`list_user_scopes` — deliberately not a
    bare list, mirroring `PurgeResult` one level up the same cascade.

    `scopes` alone cannot distinguish "this app/user genuinely holds no
    memory scopes" from "enumeration was interrupted before it could find
    out" — both previously returned `[]` indistinguishably. `complete` makes
    that distinction explicit:

    - `complete=True`: the namespace prefix/suffix was walked to the end (an
      empty or short final page). `scopes` is the exact, final list — an
      empty list here is a legitimate result (a scope-less app/user, AC-28's
      sibling case at the enumeration level).
    - `complete=False`: the walk was interrupted — an invalid `app_id`/
      `user_id`, or an exception raised on any page, **including the
      first**. `scopes` reflects whatever was collected before the
      interruption and must never be read as the total. Callers
      (`delete_app_global_memory`/`delete_user_global_memory`) must fold
      this into their own aggregate `complete` in addition to each scope's
      own purge outcome — an enumeration that failed outright (zero scopes
      collected) must never be reported as a clean, complete erasure of
      nothing, which is exactly the defect this type exists to close
      (FR-8/NFR-5).
    """

    scopes: List[MemoryIdentity]
    complete: bool


class _ScopeListIncomplete(Exception):
    """Internal signal from `_list_app_scopes_sync`/`_list_user_scopes_sync`
    that the enumeration walk was interrupted after collecting `scopes` so
    far. Never escapes `list_app_scopes`/`list_user_scopes` — they always
    translate this into `ScopeListResult(scopes, complete=False)`.
    """

    def __init__(self, scopes: List[MemoryIdentity], cause: BaseException) -> None:
        super().__init__(str(cause))
        self.scopes = scopes
        self.cause = cause


def list_app_scopes(app_id: int) -> ScopeListResult:
    """Enumerate every memory scope under `app_id` that actually holds facts
    (AD-10), paged. Sync, module-level, for the same reason `purge` is (must
    work regardless of the kill switch/pool state).

    Uses `list_namespaces(prefix=(root, str(app_id)), max_depth=3)` — the
    namespace-prefix listing the FR-1 spike confirmed returns exactly the
    scopes holding facts, never an inferred candidate list from
    `AppCollaborator`/`Conversation` (which could silently miss a former
    collaborator whose memory outlived their membership).

    `list_namespaces`'s prefix match is the same `LIKE` match as `search`
    (see the module docstring), so a returned namespace tuple is re-derived
    from the store's own components and re-checked for an exact match to
    `root`/`app_id` (and a digit-only user-id segment) before it is trusted
    — this re-check is why this method was never affected by the leak found
    elsewhere in this file, and `list_user_scopes` mirrors it exactly.

    Each returned `MemoryIdentity` carries `agent_id=0`: a scope enumeration
    has no single originating agent (the whole point is it may span many),
    so `0` is a sentinel meaning "not applicable to a purge target" rather
    than real per-fact provenance.

    Returns a `ScopeListResult`, never a bare list: any failure — an invalid
    `app_id`, or an exception raised on any page of the walk — returns
    `ScopeListResult(scopes=<partial>, complete=False)` (NFR-4/NFR-5) rather
    than an indistinguishable `[]`. Callers must not read `scopes` from an
    incomplete result as the total.
    """
    try:
        _assert_positive_int(app_id, "app_id")
    except Exception:
        log_memory_event(
            "list_scopes", app_id=app_id if isinstance(app_id, int) else None, agent_id=None,
            active=False, reason=FAIL_ERROR, level="warning",
        )
        return ScopeListResult(scopes=[], complete=False)

    try:
        scopes = _list_app_scopes_sync(app_id)
        return ScopeListResult(scopes=scopes, complete=True)
    except _ScopeListIncomplete as exc:
        log_memory_event(
            "list_scopes", app_id=app_id, agent_id=None,
            active=False, count=len(exc.scopes), reason=FAIL_ERROR, level="warning",
        )
        return ScopeListResult(scopes=exc.scopes, complete=False)
    except Exception:
        log_memory_event(
            "list_scopes", app_id=app_id, agent_id=None,
            active=False, reason=FAIL_ERROR, level="warning",
        )
        return ScopeListResult(scopes=[], complete=False)


def _list_app_scopes_sync(app_id: int) -> List[MemoryIdentity]:
    """Raises `_ScopeListIncomplete(scopes_so_far, cause)` on any error mid-walk
    (including on the very first page) so `list_app_scopes` can report a true
    partial result instead of silently returning `[]`.
    """
    root = settings.namespace_root()
    prefix = (root, str(app_id))
    scopes: List[MemoryIdentity] = []
    try:
        offset = 0
        with _sync_connection() as conn:
            store = PostgresStore(conn=conn)
            while True:
                namespaces = store.list_namespaces(
                    prefix=prefix, max_depth=3, limit=_LIST_SCOPES_PAGE_SIZE, offset=offset
                )
                if not namespaces:
                    break
                for ns in namespaces:
                    if len(ns) != 3:
                        continue
                    ns_root, ns_app_id, ns_user_id = ns
                    if ns_root != root or ns_app_id != str(app_id) or not ns_user_id.isdigit():
                        continue
                    scopes.append(
                        MemoryIdentity(
                            app_id=int(ns_app_id),
                            user_id=int(ns_user_id),
                            agent_id=0,
                            namespace=ns,
                        )
                    )
                offset += len(namespaces)
                if len(namespaces) < _LIST_SCOPES_PAGE_SIZE:
                    break
    except Exception as exc:
        raise _ScopeListIncomplete(scopes, exc) from exc
    return scopes


def list_user_scopes(user_id: int) -> ScopeListResult:
    """Enumerate every memory scope belonging to `user_id`, across every App
    (AD-10). Sync, module-level, for the same reasons as `list_app_scopes`.

    Exists so step_006's cross-App erasure (`UserService.delete_user`) never
    has to import `langgraph.store` itself. Its spec'd approach —
    `alist_namespaces(prefix=(root,), suffix=(str(user_id),))` — inherits the
    same `LIKE`-match defect `search`/`list_app_scopes` do, except on the
    *suffix*: `prefix LIKE '%7'` matches any namespace whose full dotted text
    form ends in the digit `7`, i.e. **any user id ending in `7`** — `17`,
    `27`, `107`, not only user `7`. This function performs that same listing
    but re-derives each namespace's own last component and re-checks it for
    exact string equality to `str(user_id)` before trusting it, mirroring
    `list_app_scopes`'s defence exactly.

    Each returned `MemoryIdentity` carries `agent_id=0` (see
    `list_app_scopes`). Returns a `ScopeListResult`, never a bare list — see
    `list_app_scopes`'s docstring for why (NFR-4/NFR-5).
    """
    try:
        _assert_positive_int(user_id, "user_id")
    except Exception:
        log_memory_event(
            "list_scopes", app_id=None, agent_id=None, active=False, reason=FAIL_ERROR, level="warning"
        )
        return ScopeListResult(scopes=[], complete=False)

    try:
        scopes = _list_user_scopes_sync(user_id)
        return ScopeListResult(scopes=scopes, complete=True)
    except _ScopeListIncomplete as exc:
        log_memory_event(
            "list_scopes", app_id=None, agent_id=None,
            active=False, count=len(exc.scopes), reason=FAIL_ERROR, level="warning",
        )
        return ScopeListResult(scopes=exc.scopes, complete=False)
    except Exception:
        log_memory_event(
            "list_scopes", app_id=None, agent_id=None, active=False, reason=FAIL_ERROR, level="warning"
        )
        return ScopeListResult(scopes=[], complete=False)


def _list_user_scopes_sync(user_id: int) -> List[MemoryIdentity]:
    """Raises `_ScopeListIncomplete(scopes_so_far, cause)` on any error mid-walk
    — mirrors `_list_app_scopes_sync` exactly (see that function's docstring).
    """
    root = settings.namespace_root()
    suffix = (str(user_id),)
    scopes: List[MemoryIdentity] = []
    try:
        offset = 0
        with _sync_connection() as conn:
            store = PostgresStore(conn=conn)
            while True:
                namespaces = store.list_namespaces(
                    prefix=(root,), suffix=suffix, max_depth=3, limit=_LIST_SCOPES_PAGE_SIZE, offset=offset
                )
                if not namespaces:
                    break
                for ns in namespaces:
                    if len(ns) != 3:
                        continue
                    ns_root, ns_app_id, ns_user_id = ns
                    if ns_root != root or ns_user_id != str(user_id) or not ns_app_id.isdigit():
                        continue
                    scopes.append(
                        MemoryIdentity(
                            app_id=int(ns_app_id),
                            user_id=int(ns_user_id),
                            agent_id=0,
                            namespace=ns,
                        )
                    )
                offset += len(namespaces)
                if len(namespaces) < _LIST_SCOPES_PAGE_SIZE:
                    break
    except Exception as exc:
        raise _ScopeListIncomplete(scopes, exc) from exc
    return scopes


# Module-level bounded cache, keyed by `embedding_service_id` alone — the
# embeddings client is the store's only per-instance input (the vector width
# follows from it, see the module docstring) and the extraction LLM is not
# part of the store at all, so that is the only axis a `GlobalMemoryClient`
# varies on. FIFO eviction, but only of *idle* instances
# (AD-9): evicting one with queued operations would strand the coroutine
# awaiting them, since the store's background task is simply cancelled with
# no way to resume it.
_CLIENTS: "OrderedDict[int, GlobalMemoryClient]" = OrderedDict()


async def get_memory_client(embeddings: Any, embedding_service_id: Optional[int]) -> Optional[GlobalMemoryClient]:
    """Return the cached `GlobalMemoryClient` for `embedding_service_id`,
    building one if needed. Returns `None` when either input is missing (an
    unresolved embedding side, per `resolution.py`) or the shared pool was
    never initialized (kill switch off, or startup failed) — never raises.

    Must be called from within a running event loop (it always is, in
    practice — this is only ever awaited from `_prepare_turn`/the write path,
    both already inside the request's event loop), because
    `AsyncPostgresStore.__init__` captures `asyncio.get_running_loop()` for
    its background task and that capture must happen here, not at import
    time (AD-9).
    """
    if embeddings is None or embedding_service_id is None:
        return None

    cached = _CLIENTS.get(embedding_service_id)
    if cached is not None:
        _CLIENTS.move_to_end(embedding_service_id)
        return cached

    try:
        pool = GlobalMemoryPoolService.get_pool()
    except RuntimeError:
        logger.warning(
            "global_memory pool not initialized; cannot build a GlobalMemoryClient for embedding_service_id=%s",
            embedding_service_id,
        )
        return None

    client = GlobalMemoryClient(pool=pool, embeddings=embeddings, embedding_service_id=embedding_service_id)
    _CLIENTS[embedding_service_id] = client
    _evict_idle_if_over_cap()
    return client


def _evict_idle_if_over_cap() -> None:
    """FIFO-evict idle cached clients until the cache is back at the cap.

    If every cached instance currently has in-flight operations, the cache is
    left over the cap rather than stranding an awaiting coroutine — AD-9 is
    explicit that this is the correct trade-off ("evict only idle instances,
    or do not evict below a small cap").
    """
    while len(_CLIENTS) > _MAX_CACHED_CLIENTS:
        evicted = False
        for key, client in _CLIENTS.items():
            if client.idle:
                del _CLIENTS[key]
                evicted = True
                break
        if not evicted:
            break


async def _drain_all_clients(timeout_s: float = _DRAIN_TIMEOUT_S) -> None:
    """Best-effort wait for every cached client to go idle, then drop only the
    idle ones from the cache.

    Called by `GlobalMemoryPoolService.close_pool()` before closing the
    shared pool (AD-9 shutdown ordering): closing the pool while a client's
    background task still has queued operations is the same stranding
    failure as evicting it. This is a grace period, not a guarantee — by
    shutdown there should be no new turns starting, so any in-flight work is
    expected to finish quickly; if it does not within `timeout_s`, the pool
    is closed anyway rather than hanging application shutdown indefinitely.

    Clients still not idle after the deadline are deliberately **left
    referenced** in the cache rather than dropped: clearing the cache
    unconditionally previously dropped the last Python reference to a still-
    busy client, which lets it be garbage-collected — collection cancels its
    background batch task out from under whatever coroutine is still
    awaiting it, the exact AD-9 stranding this drain exists to prevent. A
    warning names how many clients were still busy so a stuck shutdown is
    visible in the logs instead of silently discarded.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    for client in list(_CLIENTS.values()):
        while not client.idle and loop.time() < deadline:
            await asyncio.sleep(_DRAIN_POLL_INTERVAL_S)

    still_busy_keys = [key for key, client in _CLIENTS.items() if not client.idle]
    if still_busy_keys:
        logger.warning(
            "Global memory shutdown: %d cached client(s) still had in-flight operations "
            "after the %.1fs drain deadline; leaving them referenced instead of evicting "
            "to avoid stranding an awaiting coroutine.",
            len(still_busy_keys), timeout_s,
        )

    for key in [key for key, client in _CLIENTS.items() if client.idle]:
        del _CLIENTS[key]
