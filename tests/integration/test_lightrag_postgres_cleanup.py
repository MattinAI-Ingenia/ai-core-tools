"""AC-20/21: delete_collection must leave no PG-vector rows in model-suffixed tables.

Regression guard: lightrag-hku names its PostgreSQL vector tables
``lightrag_vdb_<base>_<model suffix>``, so for a silo with
``lightrag_vector_db_type="PGVECTOR"`` the rows must be dropped through the
library's own ``drop()``. Real LightRAG + Neo4j + the test DB; SKIPs where they are unavailable.
"""
import unittest.mock as mock

import pytest

lightrag = pytest.importorskip("lightrag")

from langchain_core.documents import Document
from sqlalchemy import text

from db.database import SessionLocal, db as database
from tests.integration.lightrag_helpers import (
    SimpleAiService,
    SimpleEmbeddingService,
    delete_lightrag_rows,
    neo4j_label_count,
    require_real_lightrag_backend,
    stub_embedding_func,
    stub_llm_calls_recorder,
)

TARGET = "silo_987654322"
BYSTANDER = "silo_987654323"
BASE_TABLES = {"lightrag_vdb_chunks", "lightrag_vdb_entity", "lightrag_vdb_relation"}


@pytest.fixture(scope="module", autouse=True)
def _require_real_lightrag_backend():
    require_real_lightrag_backend()


def _vdb_tables() -> list:
    session = SessionLocal()
    try:
        return list(session.execute(
            text("SELECT tablename FROM pg_tables WHERE tablename LIKE 'lightrag_vdb_%'")
        ).scalars().all())
    finally:
        session.close()


def _vdb_rows(workspace: str) -> dict:
    session = SessionLocal()
    try:
        return {
            table: session.execute(
                text(f'SELECT count(*) FROM "{table}" WHERE workspace = :ws'),  # noqa: S608 — table from pg_tables
                {"ws": workspace},
            ).scalar()
            for table in _vdb_tables()
        }
    finally:
        session.close()


def _teardown(store, patchers) -> None:
    """Every step always runs, even if an earlier one raises."""
    try:
        try:
            if store is not None:
                for ws in (TARGET, BYSTANDER):
                    try:
                        store.delete_collection(ws)
                    except Exception:  # noqa: BLE001 — cleanup must never mask the result
                        pass
        finally:
            for ws in (TARGET, BYSTANDER):
                try:
                    delete_lightrag_rows(ws)
                except Exception:  # noqa: BLE001
                    pass
    finally:
        try:
            if store is not None:
                for loop in list(store._collection_loops.values()):
                    loop.close()
        finally:
            for patcher in reversed(patchers):
                try:
                    patcher.stop()
                except RuntimeError:  # patcher never started
                    pass


class TestDeleteCollectionPgVector:
    def test_delete_collection_removes_pg_vector_rows(self, test_engine):  # noqa: ARG002
        from tools.vector_stores.lightrag import adapters
        from tools.vector_stores.lightrag_store import LightRAGStore

        for ws in (TARGET, BYSTANDER):
            if neo4j_label_count(ws) > 0:
                pytest.skip(f"Neo4j already contains workspace label '{ws}'; refusing to touch it.")

        stub_llm, _ = stub_llm_calls_recorder()
        stub_embedding, _ = stub_embedding_func()
        patchers = [
            mock.patch.object(adapters, "build_llm_model_func", lambda *a, **k: stub_llm),
            mock.patch.object(adapters, "build_embedding_func", lambda *a, **k: stub_embedding),
        ]
        store = None
        try:
            for patcher in patchers:
                patcher.start()
            store = LightRAGStore(
                db=database,
                ai_service=SimpleAiService(),
                embedding_service=SimpleEmbeddingService(),
                lightrag_vector_db_type="PGVECTOR",
            )
            for ws in (TARGET, BYSTANDER):
                store.index_documents(ws, [Document(
                    id="res1",
                    page_content="Ficha tecnica SOLARIUM-7 VMS-2210 TURBOX-9933 opera a 4 bar.",
                    metadata={"resource_id": 1},
                )])

            tables = _vdb_tables()
            assert set(tables) - BASE_TABLES, f"no model-suffixed vdb table exists: {tables}"
            assert sum(_vdb_rows(TARGET).values()) > 0, "precondition: target has no vdb rows"
            bystander_before = _vdb_rows(BYSTANDER)
            assert sum(bystander_before.values()) > 0

            store.delete_collection(TARGET)

            leftover = {t: n for t, n in _vdb_rows(TARGET).items() if n}
            assert not leftover, f"PG-vector rows survived delete_collection: {leftover}"
            assert _vdb_rows(BYSTANDER) == bystander_before
        finally:
            _teardown(store, patchers)
