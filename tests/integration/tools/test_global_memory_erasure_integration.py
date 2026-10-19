"""Integration tests for `services.global_memory_service`'s erasure functions
against the REAL LangGraph Postgres store (spec `user-memory-mem0`, FR-8,
AC-26/AC-27/AC-28/AC-42, step_006).

Requires the test database on port 5433
(`docker compose -f docker/docker-compose.yaml --profile test up -d db_test`).
Creates the store's four tables itself (small vector width for test speed)
and drops them again once the module's tests finish — mirrors
`tests/integration/tools/test_memory_client_namespace_isolation.py` exactly,
so this suite never leaves that schema behind for other tests.
"""
from __future__ import annotations

import hashlib
from typing import List
from unittest.mock import patch

import psycopg
import pytest
from langchain_core.embeddings import Embeddings
from langgraph.store.postgres import PostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from services.global_memory_service import (
    delete_app_global_memory,
    delete_global_memory,
    delete_user_global_memory,
)
from tools.memory import settings as memory_settings
from tools.memory.client import GlobalMemoryClient, PurgeResult
from tools.memory.identity import MemoryIdentity

pytestmark = pytest.mark.integration

_TEST_DB_URI = "postgresql://test_user:test_pass@localhost:5433/test_db"
_TEST_DIMS = 8


class _FakeEmbeddings(Embeddings):
    """Deterministic tiny-dim embeddings — good enough to rank real pgvector
    similarity queries without calling out to a real embedding provider."""

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
        app_id=app_id, user_id=user_id, agent_id=agent_id, namespace=(root, str(app_id), str(user_id))
    )


def _raw_connection() -> psycopg.Connection:
    return psycopg.connect(_TEST_DB_URI, autocommit=True, prepare_threshold=0, row_factory=dict_row)


@pytest.fixture
def memory_schema():
    """Create the store's four tables, then drop all four afterward."""
    with _raw_connection() as conn:
        setup_store = PostgresStore(conn, index={"dims": _TEST_DIMS, "embed": _FakeEmbeddings(), "fields": ["text"]})
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
    await pool.close()


class TestDeleteGlobalMemoryRemovesDataAndIsScoped:
    """AC-26/AC-27."""

    @pytest.mark.asyncio
    async def test_erasure_removes_only_the_exact_scope(self, memory_client):
        identity_1_7 = _make_identity(app_id=1, user_id=7)
        identity_2_7 = _make_identity(app_id=2, user_id=7)
        identity_1_8 = _make_identity(app_id=1, user_id=8)

        await memory_client.add(identity_1_7, ["fact about (1,7)"])
        await memory_client.add(identity_2_7, ["fact about (2,7)"])
        await memory_client.add(identity_1_8, ["fact about (1,8)"])

        result = delete_global_memory(db=None, app_id=1, user_id=7)

        assert result == PurgeResult(deleted=1, complete=True)
        assert await memory_client.search(identity_1_7, query="fact", top_k=10) == []
        assert await memory_client.search(identity_2_7, query="fact", top_k=10) == ["fact about (2,7)"]
        assert await memory_client.search(identity_1_8, query="fact", top_k=10) == ["fact about (1,8)"]

    def test_idempotent_second_call_and_empty_scope(self, memory_schema):
        """AC-28."""
        first = delete_global_memory(db=None, app_id=3, user_id=99)
        assert first == PurgeResult(deleted=0, complete=True)

        second = delete_global_memory(db=None, app_id=3, user_id=99)
        assert second == PurgeResult(deleted=0, complete=True)


class TestDeleteAppGlobalMemoryEnumeratesFromTheStore:
    """AC-29 (real-store half)."""

    @pytest.mark.asyncio
    async def test_purges_every_scope_the_app_holds(self, memory_client):
        identity_a1 = _make_identity(app_id=10, user_id=1)
        identity_a2 = _make_identity(app_id=10, user_id=2)
        identity_other_app = _make_identity(app_id=11, user_id=1)

        await memory_client.add(identity_a1, ["fact a1"])
        await memory_client.add(identity_a2, ["fact a2"])
        await memory_client.add(identity_other_app, ["fact other app"])

        result = delete_app_global_memory(db=None, app_id=10)

        assert result == PurgeResult(deleted=2, complete=True)
        assert await memory_client.search(identity_a1, query="fact", top_k=10) == []
        assert await memory_client.search(identity_a2, query="fact", top_k=10) == []
        assert await memory_client.search(identity_other_app, query="fact", top_k=10) == ["fact other app"]


class TestDeleteUserGlobalMemoryEnumeratesAcrossApps:
    """AC-30 / AC-61 (real-store half)."""

    @pytest.mark.asyncio
    async def test_purges_the_user_in_every_app(self, memory_client):
        identity_app1 = _make_identity(app_id=20, user_id=42)
        identity_app2 = _make_identity(app_id=21, user_id=42)
        identity_other_user = _make_identity(app_id=20, user_id=43)

        await memory_client.add(identity_app1, ["fact user 42 app 20"])
        await memory_client.add(identity_app2, ["fact user 42 app 21"])
        await memory_client.add(identity_other_user, ["fact user 43"])

        result = delete_user_global_memory(db=None, user_id=42)

        assert result == PurgeResult(deleted=2, complete=True)
        assert await memory_client.search(identity_app1, query="fact", top_k=10) == []
        assert await memory_client.search(identity_app2, query="fact", top_k=10) == []
        assert await memory_client.search(identity_other_user, query="fact", top_k=10) == ["fact user 43"]


class TestErasureNeverTouchesTheCheckpointer:
    """AC-42/D-8."""

    def test_no_checkpointer_call_is_made(self, memory_schema):
        with patch(
            "services.agent_cache_service.CheckpointerCacheService.invalidate_checkpointer_async"
        ) as mock_invalidate:
            delete_global_memory(db=None, app_id=99, user_id=99)
            delete_app_global_memory(db=None, app_id=99)
            delete_user_global_memory(db=None, user_id=99)

        mock_invalidate.assert_not_called()
