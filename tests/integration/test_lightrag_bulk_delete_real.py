"""Real-LightRAG tests for the strict workspace wipe, delete-all and legacy detection.

Real LightRAG + Neo4j + the test DB (+ Qdrant for the QDRANT bystander case);
only the two model adapters are stubbed. SKIPs where Neo4j is unavailable.
Workspaces use explicit out-of-range silo ids (9876543xx) and are wiped on teardown.
NEO4J_URI and QDRANT_URL must point at throwaway containers (the run wipes the
labels/points of silo ids 9876543xx).
"""
import os
import unittest.mock as mock
from datetime import datetime

import pytest

lightrag = pytest.importorskip("lightrag")

from langchain_core.documents import Document
from sqlalchemy import text

from db.database import SessionLocal, db as database
from tests.integration.lightrag_helpers import (
    SimpleAiService,
    SimpleEmbeddingService,
    delete_lightrag_rows,
    delete_neo4j_workspace,
    neo4j_label_count,
    require_qdrant,
    require_real_lightrag_backend,
    stub_embedding_func,
    stub_llm_calls_recorder,
)


@pytest.fixture(scope="module", autouse=True)
def _require_real_lightrag_backend():
    require_real_lightrag_backend()


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _Env:
    """Stub model adapters + instant retire; wipes every claimed workspace on exit."""

    def __init__(self):
        self.workspaces: dict = {}  # workspace -> lightrag_vector_db_type
        self.stores: list = []
        self._patchers: list = []

    def __enter__(self):
        import tools.vector_stores.lightrag_store as store_module
        from tools.vector_stores.lightrag import adapters

        stub_llm, _ = stub_llm_calls_recorder()
        stub_embedding, _ = stub_embedding_func()
        self._patchers = [
            mock.patch.object(store_module, "_RETIRE_GRACE_S", 0),
            mock.patch.object(adapters, "build_llm_model_func", lambda *a, **k: stub_llm),
            mock.patch.object(adapters, "build_embedding_func", lambda *a, **k: stub_embedding),
        ]
        for patcher in self._patchers:
            patcher.start()
        return self

    def __exit__(self, *exc):
        import tools.vector_stores.lightrag_store as store_module

        try:
            try:
                for ws, vdb_type in self.workspaces.items():
                    try:
                        self.store(vdb_type).delete_collection(ws)
                    except Exception:  # noqa: BLE001 — cleanup must never mask the result
                        pass
            finally:
                for ws, vdb_type in self.workspaces.items():
                    try:
                        delete_lightrag_rows(ws)
                    except Exception:  # noqa: BLE001
                        pass
                    _fallback_wipe(ws, vdb_type)
                runtime = store_module._SHARED_RUNTIME
                loops = [s._collection_loops.get(ws) for s in self.stores for ws in self.workspaces]
                loops += [runtime.collection_loops.pop(ws, None) for ws in self.workspaces]
                for ws in self.workspaces:
                    runtime.rag_instances.pop(ws, None)
                    runtime.fingerprints.pop(ws, None)
                for loop in {id(x): x for x in loops if x is not None}.values():
                    try:
                        loop.close()
                    except Exception:  # noqa: BLE001
                        pass
        finally:
            for patcher in reversed(self._patchers):
                patcher.stop()

    def claim(self, silo_id: int, vdb_type: str = "PGVECTOR") -> str:
        ws = f"silo_{silo_id}"
        if neo4j_label_count(ws) > 0:
            pytest.skip(f"Neo4j already contains workspace label '{ws}'; refusing to touch it.")
        self.workspaces[ws] = vdb_type
        return ws

    def store(self, vdb_type: str = "PGVECTOR"):
        from tools.vector_stores.lightrag_store import LightRAGStore

        store = LightRAGStore(
            db=database,
            ai_service=SimpleAiService(),
            embedding_service=SimpleEmbeddingService(),
            lightrag_vector_db_type=vdb_type,
        )
        self.stores.append(store)
        return store


@pytest.fixture
def env(test_engine):  # noqa: ARG001
    with _Env() as e:
        yield e


def _count(table: str, column: str, workspace: str, ids: list) -> int:
    session = SessionLocal()
    try:
        return session.execute(
            text(f"SELECT count(*) FROM {table} WHERE workspace = :ws AND {column} = ANY(:ids)"),  # noqa: S608 — constants
            {"ws": workspace, "ids": ids},
        ).scalar()
    finally:
        session.close()


def _graph_doc_rows(workspace: str, rids: list) -> dict:
    ids = [f"res{r}" for r in rids]
    return {
        "doc_status": _count("lightrag_doc_status", "id", workspace, ids),
        "doc_chunks": _count("lightrag_doc_chunks", "full_doc_id", workspace, ids),
    }


def _pg_snapshot(workspace: str) -> dict:
    """Row count of every workspace-bearing lightrag* table for one workspace."""
    session = SessionLocal()
    try:
        tables = session.execute(
            text("SELECT table_name FROM information_schema.columns "
                 "WHERE column_name = 'workspace' AND table_name ILIKE 'lightrag%'")
        ).scalars().all()
        return {
            t: session.execute(
                text(f'SELECT count(*) FROM "{t}" WHERE workspace = :ws'),  # noqa: S608 — table from information_schema
                {"ws": workspace},
            ).scalar()
            for t in sorted(tables)
        }
    finally:
        session.close()


def _qdrant_client_and_filter(workspace: str):
    import config
    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=config.QDRANT_URL, api_key=os.getenv("QDRANT_API_KEY") or None)
    flt = models.Filter(must=[models.FieldCondition(
        key="workspace_id", match=models.MatchValue(value=workspace))])
    return client, flt


def _qdrant_snapshot(workspace: str) -> dict:
    client, flt = _qdrant_client_and_filter(workspace)
    return {
        c.name: client.count(c.name, count_filter=flt, exact=True).count
        for c in client.get_collections().collections
        if c.name.startswith("lightrag_vdb")
    }


def _fallback_wipe(workspace: str, vdb_type: str) -> None:
    """Direct Neo4j (+ Qdrant) wipe, independent of delete_collection; never raises."""
    try:
        delete_neo4j_workspace(workspace)
    except Exception:  # noqa: BLE001 — cleanup must never mask the result
        pass
    if vdb_type == "QDRANT":
        try:
            client, flt = _qdrant_client_and_filter(workspace)
            for c in client.get_collections().collections:
                if c.name.startswith("lightrag_vdb"):
                    client.delete(c.name, points_selector=flt)
        except Exception:  # noqa: BLE001
            pass


def _doc(rid: int, token: str) -> Document:
    return Document(
        id=f"res{rid}",
        page_content=f"Ficha tecnica {token} opera a 4 bar en la instalacion principal.",
        metadata={"resource_id": rid, "name": f"file{rid}.txt"},
    )


# ---------------------------------------------------------------------------
# FR-10.1 / AC-1 and FR-10.2 / AC-2: HTTP-level reproduce-first
# ---------------------------------------------------------------------------


def _build_silo_repo(db, fake_app, silo_id: int):
    from models.ai_service import AIService
    from models.embedding_service import EmbeddingService
    from models.repository import Repository
    from models.silo import Silo

    emb = EmbeddingService(
        name="Real Emb", provider="Custom", api_key="x", app_id=fake_app.app_id,  # pragma: allowlist secret
    )
    llm = AIService(
        name="Real LLM", provider="Custom", api_key="x", app_id=fake_app.app_id,  # pragma: allowlist secret
    )
    db.add_all([emb, llm])
    db.flush()
    silo = Silo(
        silo_id=silo_id, name="Bulk Silo", description="d", silo_type="REPO", status="active",
        app_id=fake_app.app_id, vector_db_type="LIGHTRAG", lightrag_vector_db_type="PGVECTOR",
        embedding_service_id=emb.service_id, extract_service_id=llm.service_id,
    )
    db.add(silo)
    db.flush()
    repo = Repository(
        name="Repo", type="default", status="active", app_id=fake_app.app_id,
        silo_id=silo.silo_id, create_date=datetime.now(),
    )
    db.add(repo)
    db.flush()
    return silo, repo


@pytest.fixture
def base_dir(tmp_path):
    with mock.patch("services.resource_service.REPO_BASE_FOLDER", str(tmp_path)):
        yield tmp_path


class TestPublicDeleteAllLock:
    def test_public_delete_all_respects_indexing_lock(
        self, client, db, fake_app, fake_user, fake_api_key, env, base_dir,  # noqa: ARG002
    ):
        from models.resource import Resource
        from services import silo_indexing_lock
        from services.silo_service import _get_vector_store

        silo, repo = _build_silo_repo(db, fake_app, 987654332)
        ws = env.claim(silo.silo_id)
        res = Resource(
            name="a.txt", uri="a.txt", type=".txt", status="active",
            repository_id=repo.repository_id, create_date=datetime.now(),
        )
        db.add(res)
        db.flush()
        rid = res.resource_id
        db.commit()
        (base_dir / str(repo.repository_id)).mkdir(parents=True, exist_ok=True)
        (base_dir / str(repo.repository_id) / "a.txt").write_text("ECHO-5055")
        _get_vector_store(silo).index_documents(ws, [_doc(rid, "ECHO-5055")])
        before = _graph_doc_rows(ws, [rid])
        assert before["doc_status"] == 1

        conn = silo_indexing_lock.acquire(silo.silo_id)
        assert conn is not None
        try:
            resp = client.delete(
                f"/public/v1/app/{fake_app.app_id}/silos/{silo.silo_id}/docs/delete/all",
                headers={"X-API-KEY": fake_api_key.key},
            )
        finally:
            silo_indexing_lock.release(conn, silo.silo_id)
        assert resp.status_code == 409, f"{resp.status_code} {resp.text}"
        assert _graph_doc_rows(ws, [rid]) == before

        # AC-36: lock free -> the graph is wiped, Resource rows and files are kept.
        resp = client.delete(
            f"/public/v1/app/{fake_app.app_id}/silos/{silo.silo_id}/docs/delete/all",
            headers={"X-API-KEY": fake_api_key.key},
        )
        assert resp.status_code == 200, f"{resp.status_code} {resp.text}"
        assert _graph_doc_rows(ws, [rid]) == {"doc_status": 0, "doc_chunks": 0}
        db.expire_all()
        assert db.get(Resource, rid) is not None
        assert (base_dir / str(repo.repository_id) / "a.txt").exists()


# ---------------------------------------------------------------------------
# FR-10.3 / AC-3 and FR-10.4 / AC-4: store-level guards (expected to pass today)
# ---------------------------------------------------------------------------


# Wipe entry points under test: lenient (silo delete) and strict (bulk delete).
WIPE_ENTRIES = ("delete_collection", "wipe_workspace")


class TestWipeThenReindex:
    @pytest.mark.parametrize("entry", WIPE_ENTRIES)
    def test_wipe_then_reindex(self, env, entry):
        ws = env.claim(987654333)
        store = env.store()
        store.index_documents(ws, [_doc(7001, "FOXTROT-6066")])
        assert _graph_doc_rows(ws, [7001])["doc_status"] == 1

        getattr(store, entry)(ws)
        assert _graph_doc_rows(ws, [7001]) == {"doc_status": 0, "doc_chunks": 0}
        assert neo4j_label_count(ws) == 0

        store.index_documents(ws, [_doc(7002, "GOLF-7077"), _doc(7001, "FOXTROT-6066")])

        session = SessionLocal()
        try:
            for rid, token in ((7001, "FOXTROT-6066"), (7002, "GOLF-7077")):
                status = session.execute(
                    text("SELECT status FROM lightrag_doc_status WHERE workspace = :ws AND id = :i"),
                    {"ws": ws, "i": f"res{rid}"},
                ).scalar()
                assert str(status).lower().endswith("processed"), f"res{rid}: {status}"
                chunk = session.execute(
                    text("SELECT count(*) FROM lightrag_doc_chunks "
                         "WHERE workspace = :ws AND full_doc_id = :i AND content ILIKE :tok"),
                    {"ws": ws, "i": f"res{rid}", "tok": f"%{token}%"},
                ).scalar()
                assert chunk > 0, f"res{rid}: token {token} not retrievable from chunks"
        finally:
            session.close()


class TestLegacyDetection:
    def test_has_legacy_documents(self, env):
        """AC-29: legacy shapes (V-4) are detected; current docs and absent docs are not."""

        ws = env.claim(987654342)
        store = env.store()
        # (i) legacy non-PDF: no id -> doc-<hash>, file_path == basename(uri); '%' and '_' in the name
        store.index_documents(ws, [Document(
            page_content="Ficha legacy JULIET-1100 opera a 4 bar.",
            metadata={"name": "Informe_final 100%.txt", "resource_id": 1},
        )])
        # (ii) current resource, (iii) legacy per-page doc
        store.index_documents(ws, [_doc(2, "KILO-2200")])
        store.index_documents(ws, [Document(
            id="res3-p1", page_content="Pagina legacy LIMA-3300 opera a 4 bar.",
            metadata={"name": "manual.pdf", "resource_id": 3, "page": 1},
        )])
        # (vi) legacy non-PDF whose uri has a backslash path; (vii) non-ASCII name
        store.index_documents(ws, [Document(
            page_content="Ficha legacy MIKE-4400 opera a 4 bar.",
            metadata={"name": "dir\\Informe_final 100%.txt", "resource_id": 6},
        )])
        store.index_documents(ws, [Document(
            page_content="Ficha legacy NOVEMBER-5500 opera a 4 bar.",
            metadata={"name": "Año_ñandú.txt", "resource_id": 7},
        )])
        assert _count("lightrag_doc_status", "id", ws, ["res2", "res3-p1"]) == 2, "precondition: docs not indexed"

        def legacy(rid, uri):
            return store.has_legacy_documents(ws, rid, uri)

        assert legacy(1, "Informe_final 100%.txt") is True  # A: legacy non-PDF
        assert legacy(1, "dir\\Informe_final 100%.txt") is True  # backslash uri, same label as the indexer
        assert legacy(7, "x/Año_ñandú.txt") is True             # non-ASCII label
        assert legacy(2, "file2.txt") is False                  # B: current res{N}
        assert legacy(3, "manual.pdf") is True                  # C: legacy per-page
        assert legacy(4, "absent.txt") is False                 # D: no graph doc
        assert legacy(5, "Informe_final 1000.txt") is False     # '_'/'%' are not wildcards in the label match


def _bystander_scenario(env, vdb_type: str, x_id: int, y_id: int, entry: str) -> None:
    x = env.claim(x_id, vdb_type)
    y = env.claim(y_id, vdb_type)
    store = env.store(vdb_type)
    store.index_documents(x, [_doc(1, "HOTEL-8088")])
    store.index_documents(y, [_doc(1, "INDIA-9099")])

    def snapshot(ws):
        snap = {"pg": _pg_snapshot(ws), "neo4j": neo4j_label_count(ws)}
        if vdb_type == "QDRANT":
            snap["qdrant"] = _qdrant_snapshot(ws)
        return snap

    y_before = snapshot(y)
    x_before = snapshot(x)
    assert y_before["neo4j"] > 0 and sum(y_before["pg"].values()) > 0
    assert x_before["neo4j"] > 0 and sum(x_before["pg"].values()) > 0
    if vdb_type == "QDRANT":
        assert sum(y_before["qdrant"].values()) > 0 and sum(x_before["qdrant"].values()) > 0

    getattr(store, entry)(x)

    x_after = snapshot(x)
    assert x_after["neo4j"] == 0
    assert sum(x_after["pg"].values()) == 0, x_after["pg"]
    if vdb_type == "QDRANT":
        assert sum(x_after["qdrant"].values()) == 0, x_after["qdrant"]
    assert snapshot(y) == y_before


QDRANT_IDS = ((987654336, 987654337), (987654338, 987654339))


def _qdrant_bystander_child() -> None:
    """Entry point of the child process (see test below)."""
    with _Env() as env:
        for (x, y), entry in zip(QDRANT_IDS, WIPE_ENTRIES):
            _bystander_scenario(env, "QDRANT", x, y, entry)


class TestWipeSparesBystander:
    @pytest.mark.parametrize("entry", WIPE_ENTRIES)
    def test_wipe_spares_bystander_pgvector(self, env, entry):
        ids = {"delete_collection": (987654334, 987654335), "wipe_workspace": (987654340, 987654341)}
        _bystander_scenario(env, "PGVECTOR", *ids[entry], entry)

    def test_wipe_spares_bystander_qdrant(self, test_engine):  # noqa: ARG002
        """Runs in a child process: lightrag-hku's Postgres client pool is process-wide and
        refuses a QDRANT-configured instance once a PGVECTOR one initialised it (same
        process), so the two vector kinds cannot share this pytest session."""
        import subprocess
        import sys

        require_qdrant()
        for sid in sum(QDRANT_IDS, ()):
            if neo4j_label_count(f"silo_{sid}") > 0:
                pytest.skip(f"Neo4j already contains workspace label 'silo_{sid}'; refusing to touch it.")
        try:
            child = subprocess.run(
                [sys.executable, "-c",
                 "from tests.integration.test_lightrag_bulk_delete_real import _qdrant_bystander_child as c; c()"],
                env={**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)},
                capture_output=True, text=True, timeout=300,
            )
        except subprocess.TimeoutExpired:
            pytest.fail("Qdrant bystander child process timed out after 300 s")
        finally:
            for sid in sum(QDRANT_IDS, ()):
                _fallback_wipe(f"silo_{sid}", "QDRANT")
        assert child.returncode == 0, f"{child.stdout[-2000:]}\n{child.stderr[-3000:]}"
