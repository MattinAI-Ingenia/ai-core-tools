"""Regression tests for the cross-user memory namespace leak (spec
`user-memory-mem0`, step_005, security audit HIGH finding).

`AsyncPostgresStore`/`PostgresStore` match every namespace operation with a
SQL `LIKE`, never equality (`langgraph/store/postgres/base.py`): a search or
list scoped to `("user_memory", "5", "7")` also matches
`("user_memory", "5", "70")` (prefix match) and a suffix list scoped to user
`7` also matches user `17`/`27`/`107` (suffix match). `tools/memory/client.py`
must re-check every namespace it gets back from the store for exact equality
before trusting it — this module proves that against the real Postgres/
pgvector test database, not a mock.

Requires the test database on port 5433
(`docker compose -f docker/docker-compose.yaml --profile test up -d db_test`).
Creates the store's four tables (`store`, `store_vectors`, `store_migrations`,
`vector_migrations`) itself and drops them again once the module's tests
finish — this suite must never leave that schema behind for other tests.
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import List

import psycopg
import pytest
from langchain_core.embeddings import Embeddings
from langgraph.store.postgres import PostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

import config
from tools.memory import settings as memory_settings
from tools.memory.client import GlobalMemoryClient, list_user_scopes, purge
from tools.memory.identity import MemoryIdentity

pytestmark = pytest.mark.integration

_TEST_DB_URI = "postgresql://test_user:test_pass@localhost:5433/test_db"
_TEST_DIMS = 8


class _FakeEmbeddings(Embeddings):
    """Deterministic tiny-dim embeddings — good enough to rank real pgvector
    similarity queries without calling out to a real embedding provider.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255.0 for byte in digest[:_TEST_DIMS]]


def _make_identity(app_id: int, user_id: int, agent_id: int = 1) -> MemoryIdentity:
    root = memory_settings.namespace_root()
    return MemoryIdentity(
        app_id=app_id,
        user_id=user_id,
        agent_id=agent_id,
        namespace=(root, str(app_id), str(user_id)),
    )


def _raw_connection() -> psycopg.Connection:
    return psycopg.connect(_TEST_DB_URI, autocommit=True, prepare_threshold=0, row_factory=dict_row)


@pytest.fixture
def memory_schema(monkeypatch):
    """Create the store's four tables (small vector width for test speed),
    then drop all four afterward — this suite must never leave them behind.
    """
    monkeypatch.setattr(config, "GLOBAL_MEMORY_EMBEDDING_DIMS", _TEST_DIMS)

    with _raw_connection() as conn:
        setup_store = PostgresStore(
            conn, index={"dims": _TEST_DIMS, "embed": _FakeEmbeddings(), "fields": ["text"]}
        )
        setup_store.setup()

    yield

    with _raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS store_vectors CASCADE")
            cur.execute("DROP TABLE IF EXISTS store CASCADE")
            cur.execute("DROP TABLE IF EXISTS store_migrations")
            cur.execute("DROP TABLE IF EXISTS vector_migrations")


@pytest.fixture
async def memory_client(memory_schema):
    pool = AsyncConnectionPool(
        conninfo=_TEST_DB_URI,
        min_size=1,
        max_size=3,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open(wait=True)
    client = GlobalMemoryClient(pool=pool, embeddings=_FakeEmbeddings(), embedding_service_id=1)
    yield client

    # Drain before closing the pool — same ordering `GlobalMemoryPoolService.
    # close_pool()`/`_drain_all_clients` enforce in production (AD-9): closing
    # the pool while the store's background batch task still has queued work
    # strands it. (A "Task was destroyed but it is pending" warning from the
    # store's own background task can still appear at interpreter/loop
    # teardown even so — that task's lifecycle is owned by
    # `AsyncBatchedBaseStore`, not this seam, and production never tears a
    # single client down in isolation like this test does; it is cosmetic
    # here, not a sign either fixture or `client.py` failed to drain.)
    deadline = asyncio.get_running_loop().time() + 5.0
    while not client.idle and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)
    await pool.close()


class TestSearchNamespaceIsolation:
    """AC — a search scoped to one user must never surface a numerically
    adjacent user's facts, even though the store matches namespaces by LIKE
    prefix under the hood.
    """

    @pytest.mark.asyncio
    async def test_search_excludes_prefix_colliding_neighbour(self, memory_client):
        identity_7 = _make_identity(app_id=5, user_id=7)
        identity_70 = _make_identity(app_id=5, user_id=70)

        written_7 = await memory_client.add(identity_7, ["SECRETO DEL USUARIO 7"])
        written_70 = await memory_client.add(identity_70, ["SECRETO DEL USUARIO 70"])
        assert written_7 == 1
        assert written_70 == 1

        results = await memory_client.search(identity_7, query="secreto", top_k=10)

        assert results == ["SECRETO DEL USUARIO 7"]
        assert "SECRETO DEL USUARIO 70" not in results

    @pytest.mark.asyncio
    async def test_similarity_excludes_prefix_colliding_neighbour(self, memory_client):
        identity_7 = _make_identity(app_id=5, user_id=7)
        identity_70 = _make_identity(app_id=5, user_id=70)

        await memory_client.add(identity_70, ["a fact only user 70 owns"])

        # identity_7's namespace has no facts of its own yet — the store's LIKE
        # match would otherwise happily return user 70's item as if it were a
        # candidate within identity_7's own scope.
        score = await memory_client.similarity(identity_7, "a fact only user 70 owns")

        assert score is None


class TestPurgeNamespaceIsolation:
    """AC — purge must delete only the exact namespace, must never lose
    track of (or under/over-count) the exact rows it removes, and must never
    spin forever when a neighbouring scope shares its LIKE-matched bucket.
    """

    @pytest.mark.asyncio
    async def test_purge_deletes_only_exact_namespace_and_reports_true_count(self, memory_client):
        identity_7 = _make_identity(app_id=5, user_id=7)
        identity_70 = _make_identity(app_id=5, user_id=70)

        await memory_client.add(identity_7, ["user 7 fact one", "user 7 fact two"])
        await memory_client.add(identity_70, ["user 70's own fact — must survive"])

        result = purge(identity_7)

        assert result.complete is True
        assert result.deleted == 2

        root = memory_settings.namespace_root()
        with _raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT prefix FROM store WHERE prefix LIKE %s ORDER BY prefix",
                    (f"{root}.5.%",),
                )
                remaining_prefixes = {row["prefix"] for row in cur.fetchall()}

        # User 7's namespace is gone; user 70's is untouched by the purge.
        assert remaining_prefixes == {f"{root}.5.70"}

    @pytest.mark.asyncio
    async def test_purge_of_empty_namespace_is_a_no_op_success(self, memory_client):
        identity_7 = _make_identity(app_id=5, user_id=7)

        result = purge(identity_7)

        assert result == type(result)(deleted=0, complete=True)


class TestListUserScopesNamespaceIsolation:
    """AC — enumerating a user's scopes across every App must never surface a
    neighbour matched only because the store's suffix filter is also a LIKE.
    """

    def test_list_user_scopes_excludes_prefix_and_suffix_colliding_neighbours(self, memory_schema):
        root = memory_settings.namespace_root()
        with _raw_connection() as conn:
            store = PostgresStore(conn)
            # Same app, numeric *prefix* collision: "5.7" vs "5.70".
            store.put((root, "5", "7"), "k1", {"text": "user 7 fact"})
            store.put((root, "5", "70"), "k2", {"text": "user 70 fact"})
            # Different app, numeric *suffix* collision: "...3.17" ends in "7".
            store.put((root, "3", "17"), "k3", {"text": "user 17 fact"})

        result = list_user_scopes(7)
        assert result.complete is True
        resolved = {(scope.app_id, scope.user_id) for scope in result.scopes}

        assert resolved == {(5, 7)}
