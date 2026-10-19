"""The memory store accepts any embedding width (spec `user-memory-mem0`).

`AsyncPostgresStore.setup()` creates `store_vectors.embedding vector(<dims>)`
plus an HNSW index, which would pin every App to one embedding width.
`GlobalMemoryPoolService._relax_vector_column` drops the index and relaxes the
column to a plain `vector` right after `setup()`; every fact records the
`embedding_service_id` that embedded it and every vector search filters on it,
because pgvector raises ("different vector dimensions") on comparing vectors of
different widths. This module proves all of that against the real Postgres/
pgvector test database, not a mock.

Requires the test database on port 5433
(`docker compose -f docker/docker-compose.yaml --profile test up -d db_test`).
Creates the store's four tables itself and drops them again afterward.
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

from tools.memory import settings as memory_settings
from tools.memory.client import GlobalMemoryClient, GlobalMemoryPoolService, purge
from tools.memory.identity import MemoryIdentity

pytestmark = pytest.mark.integration

_TEST_DB_URI = "postgresql://test_user:test_pass@localhost:5433/test_db"


class _FakeEmbeddings(Embeddings):
    """Deterministic embeddings of a configurable width."""

    def __init__(self, dims: int) -> None:
        self.dims = dims

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)

    def _vector(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest() * (self.dims // 32 + 1)
        return [byte / 255.0 for byte in digest[: self.dims]]


def _identity(app_id: int, user_id: int) -> MemoryIdentity:
    root = memory_settings.namespace_root()
    return MemoryIdentity(app_id=app_id, user_id=user_id, agent_id=1, namespace=(root, str(app_id), str(user_id)))


def _raw_connection() -> psycopg.Connection:
    return psycopg.connect(_TEST_DB_URI, autocommit=True, prepare_threshold=0, row_factory=dict_row)


def _drop_store_tables() -> None:
    with _raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS store_vectors CASCADE")
            cur.execute("DROP TABLE IF EXISTS store CASCADE")
            cur.execute("DROP TABLE IF EXISTS store_migrations")
            cur.execute("DROP TABLE IF EXISTS vector_migrations")


def _embedding_column_state() -> dict:
    with _raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT format_type(a.atttypid, a.atttypmod) AS column_type, "
                "to_regclass('store_vectors_embedding_idx') IS NOT NULL AS has_index "
                "FROM pg_attribute a "
                "WHERE a.attrelid = 'store_vectors'::regclass AND a.attname = 'embedding'"
            )
            return cur.fetchone()


async def _open_pool() -> AsyncConnectionPool:
    pool = AsyncConnectionPool(
        conninfo=_TEST_DB_URI,
        min_size=1,
        max_size=3,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open(wait=True)
    return pool


async def _drain(*clients: GlobalMemoryClient) -> None:
    """Wait for the clients' queued store work before the fixture closes the
    pool — the same ordering `close_pool()` enforces in production (AD-9)."""
    deadline = asyncio.get_running_loop().time() + 5.0
    while any(not c.idle for c in clients) and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)


@pytest.fixture
async def pool(monkeypatch):
    """A pool with the production `_run_setup` (setup + relax) already applied.

    `_assert_migrations_complete` reads through `GlobalMemoryPoolService._pool`,
    which production sets before running setup — mirrored here.
    """
    _drop_store_tables()
    pool = await _open_pool()
    monkeypatch.setattr(GlobalMemoryPoolService, "_pool", pool)
    await GlobalMemoryPoolService._run_setup(pool)
    yield pool
    await pool.close()
    _drop_store_tables()


class TestSchemaIsRelaxed:
    @pytest.mark.asyncio
    async def test_setup_leaves_a_dimensionless_column_and_no_ann_index(self, pool):
        state = _embedding_column_state()
        assert state["column_type"] == "vector"
        assert state["has_index"] is False

    @pytest.mark.asyncio
    async def test_setup_is_idempotent(self, pool):
        await GlobalMemoryPoolService._run_setup(pool)
        state = _embedding_column_state()
        assert state["column_type"] == "vector"
        assert state["has_index"] is False

    @pytest.mark.asyncio
    async def test_existing_fixed_width_schema_is_relaxed_keeping_its_rows(self, monkeypatch):
        """A database set up before this change (`vector(1536)` + HNSW) is
        upgraded in place on the next startup, without losing stored facts."""
        _drop_store_tables()
        with _raw_connection() as conn:
            legacy = PostgresStore(conn, index={"dims": 8, "embed": _FakeEmbeddings(8), "fields": ["text"]})
            legacy.setup()
            legacy.put(("user_memory", "1", "7"), "k", {"text": "legacy fact"})
        assert _embedding_column_state() == {"column_type": "vector(8)", "has_index": True}

        pool = await _open_pool()
        try:
            monkeypatch.setattr(GlobalMemoryPoolService, "_pool", pool)
            await GlobalMemoryPoolService._run_setup(pool)

            assert _embedding_column_state() == {"column_type": "vector", "has_index": False}
            with _raw_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT count(*) AS n FROM store_vectors")
                    assert cur.fetchone()["n"] == 1
        finally:
            await pool.close()
            _drop_store_tables()


class TestAnyWidthPerApp:
    @pytest.mark.asyncio
    async def test_apps_with_different_widths_coexist(self, pool):
        """Two Apps whose embedding services differ in width (3072 vs 1536 —
        the case that used to be rejected) both write and read their facts."""
        wide = GlobalMemoryClient(pool=pool, embeddings=_FakeEmbeddings(3072), embedding_service_id=1)
        narrow = GlobalMemoryClient(pool=pool, embeddings=_FakeEmbeddings(1536), embedding_service_id=2)
        app_1, app_2 = _identity(app_id=1, user_id=7), _identity(app_id=2, user_id=7)

        assert await wide.add(app_1, ["prefers short answers"]) == 1
        assert await narrow.add(app_2, ["works at Ikerlan"]) == 1

        assert await wide.search(app_1, "short answers", top_k=5) == ["prefers short answers"]
        assert await narrow.search(app_2, "Ikerlan", top_k=5) == ["works at Ikerlan"]
        await _drain(wide, narrow)


class TestEmbeddingServiceChangeWithinOneNamespace:
    @pytest.mark.asyncio
    async def test_searches_only_see_facts_from_their_own_embedding_service(self, pool):
        """An App that switches embedding service ends up with mixed widths in
        one namespace. Without the per-service filter pgvector raises
        "different vector dimensions" and memory silently dies for that user;
        with it, each service sees only the facts it embedded."""
        old = GlobalMemoryClient(pool=pool, embeddings=_FakeEmbeddings(8), embedding_service_id=1)
        new = GlobalMemoryClient(pool=pool, embeddings=_FakeEmbeddings(12), embedding_service_id=2)
        identity = _identity(app_id=1, user_id=7)

        assert await old.add(identity, ["fact embedded by the old model"]) == 1
        assert await new.add(identity, ["fact embedded by the new model"]) == 1

        assert await new.search(identity, "fact", top_k=5) == ["fact embedded by the new model"]
        assert await old.search(identity, "fact", top_k=5) == ["fact embedded by the old model"]
        assert await new.similarity(identity, "fact embedded by the new model") is not None
        await _drain(old, new)

    @pytest.mark.asyncio
    async def test_relearning_a_fact_re_embeds_it_with_the_new_service(self, pool):
        """Fact keys hash the text, so the new service re-learning a fact
        overwrites the old row — re-embedded and re-tagged — instead of
        duplicating it."""
        old = GlobalMemoryClient(pool=pool, embeddings=_FakeEmbeddings(8), embedding_service_id=1)
        new = GlobalMemoryClient(pool=pool, embeddings=_FakeEmbeddings(12), embedding_service_id=2)
        identity = _identity(app_id=1, user_id=7)

        await old.add(identity, ["prefers Spanish"])
        await new.add(identity, ["prefers Spanish"])

        assert await new.search(identity, "Spanish", top_k=5) == ["prefers Spanish"]
        assert await old.search(identity, "Spanish", top_k=5) == []
        await _drain(old, new)

    @pytest.mark.asyncio
    async def test_erasure_removes_facts_of_every_width(self, pool):
        old = GlobalMemoryClient(pool=pool, embeddings=_FakeEmbeddings(8), embedding_service_id=1)
        new = GlobalMemoryClient(pool=pool, embeddings=_FakeEmbeddings(12), embedding_service_id=2)
        identity = _identity(app_id=1, user_id=7)
        await old.add(identity, ["old fact"])
        await new.add(identity, ["new fact"])
        await _drain(old, new)

        result = await asyncio.to_thread(purge, identity)

        assert result.complete is True
        assert result.deleted == 2
