"""AC-19: delete_resource_document removes exactly one document, nothing else.

Real LightRAG + Neo4j + the test DB (stubbed model adapters only); SKIPs where
they are unavailable.
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

TARGET = "silo_987654324"
BYSTANDER = "silo_987654325"
TEXT_101 = "Ficha tecnica ALPHA-1011 BRAVO-2022 opera a 4 bar."
TEXT_102 = "Ficha tecnica CHARLIE-3033 DELTA-4044 opera a 9 bar."


@pytest.fixture(scope="module", autouse=True)
def _require_real_lightrag_backend():
    require_real_lightrag_backend()


def _rows(workspace: str, doc_id: str) -> dict:
    """Timestamp-free snapshot of one document's PG rows."""
    session = SessionLocal()
    try:
        status = session.execute(
            text("SELECT id, status, chunks_count, content_length FROM lightrag_doc_status "
                 "WHERE workspace = :ws AND id = :d ORDER BY id"),
            {"ws": workspace, "d": doc_id},
        ).all()
        chunks = session.execute(
            text("SELECT id, full_doc_id, chunk_order_index, content FROM lightrag_doc_chunks "
                 "WHERE workspace = :ws AND full_doc_id = :d ORDER BY id"),
            {"ws": workspace, "d": doc_id},
        ).all()
        return {"status": [tuple(r) for r in status], "chunks": [tuple(r) for r in chunks]}
    finally:
        session.close()


def _vdb_rows(workspace: str) -> dict:
    """Row count per lightrag_vdb_* table; chunk tables also per full_doc_id."""
    session = SessionLocal()
    try:
        out = {}
        tables = session.execute(
            text("SELECT tablename FROM pg_tables WHERE tablename LIKE 'lightrag_vdb_%'")
        ).scalars().all()
        for table in tables:
            out[table] = session.execute(
                text(f'SELECT count(*) FROM "{table}" WHERE workspace = :ws'),  # noqa: S608 — table from pg_tables
                {"ws": workspace},
            ).scalar()
            has_doc_col = session.execute(
                text("SELECT 1 FROM information_schema.columns "
                     "WHERE table_name = :t AND column_name = 'full_doc_id'"),
                {"t": table},
            ).first()
            if table.startswith("lightrag_vdb_chunks") and has_doc_col:
                for doc_id in ("res101", "res102"):
                    out[(table, doc_id)] = session.execute(
                        text(f'SELECT count(*) FROM "{table}" '  # noqa: S608 — table from pg_tables
                             "WHERE workspace = :ws AND full_doc_id = :d"),
                        {"ws": workspace, "d": doc_id},
                    ).scalar()
        return out
    finally:
        session.close()


def _neo4j_entities(label: str, names: list) -> set:
    from neo4j import GraphDatabase

    import config

    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            return {r["id"] for r in session.run(
                f"MATCH (n:`{label}`) WHERE n.entity_id IN $names RETURN n.entity_id AS id",  # noqa: S608 — label from a test constant
                names=names,
            )}
    finally:
        driver.close()


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


GONE = ["alpha 1011", "bravo 2022"]  # entity ids are folded (NFKC + casefold + hyphen collapse)
KEPT = ["charlie 3033", "delta 4044"]


class TestDeleteResourceDocumentReal:
    def test_delete_removes_only_the_target_document(self, test_engine):  # noqa: ARG002
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
            store.index_documents(TARGET, [
                Document(id="res101", page_content=TEXT_101, metadata={"resource_id": 101}),
                Document(id="res102", page_content=TEXT_102, metadata={"resource_id": 102}),
            ])
            store.index_documents(BYSTANDER, [
                Document(id="res101", page_content=TEXT_101, metadata={"resource_id": 101}),
            ])

            target_101_before = _rows(TARGET, "res101")
            target_102_before = _rows(TARGET, "res102")
            bystander_before = _rows(BYSTANDER, "res101")
            for snap in (target_101_before, target_102_before, bystander_before):
                assert snap["status"] and snap["chunks"], f"precondition: missing rows {snap}"
            target_vdb_before = _vdb_rows(TARGET)
            bystander_vdb_before = _vdb_rows(BYSTANDER)
            assert any(n > 0 for n in target_vdb_before.values()), "precondition: no target vdb rows"
            assert _neo4j_entities(TARGET, GONE + KEPT) == set(GONE + KEPT), "precondition: entity ids differ"
            assert _neo4j_entities(BYSTANDER, GONE) == set(GONE)
            target_graph_before = neo4j_label_count(TARGET)
            bystander_graph_before = neo4j_label_count(BYSTANDER)
            assert target_graph_before > 0 and bystander_graph_before > 0

            assert store.delete_resource_document(TARGET, 101) == "success"

            assert _rows(TARGET, "res101") == {"status": [], "chunks": []}
            assert _rows(TARGET, "res102") == target_102_before
            assert _rows(BYSTANDER, "res101") == bystander_before

            assert neo4j_label_count(BYSTANDER) == bystander_graph_before
            target_graph_after = neo4j_label_count(TARGET)
            assert 0 < target_graph_after < target_graph_before  # res102's entities survive

            target_vdb_after = _vdb_rows(TARGET)
            for key, before in target_vdb_before.items():
                if not before:
                    continue
                if isinstance(key, tuple) and key[1] == "res102":
                    assert target_vdb_after[key] == before, f"res102 chunk vectors changed in {key[0]}"
                else:
                    assert target_vdb_after[key] < before, f"target vdb rows not reduced: {key}"
            assert _vdb_rows(BYSTANDER) == bystander_vdb_before

            assert _neo4j_entities(TARGET, GONE + KEPT) == set(KEPT)
            assert _neo4j_entities(BYSTANDER, GONE) == set(GONE)

            assert store.delete_resource_document(TARGET, 101) == "not_found"
        finally:
            _teardown(store, patchers)
