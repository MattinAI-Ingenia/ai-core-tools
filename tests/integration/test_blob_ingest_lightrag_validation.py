"""AC-8 — the actual point of this spec: a real end-to-end validation of LightRAG.

Ingesting a small fixture set of real-shaped PDFs through the Azure Blob
ingestion path into a **LightRAG-backed** Silo must produce content LightRAG
can actually retrieve, with per-chunk provenance traceable back to the ingested
``Resource`` (FR-6) — not just ``Resource`` rows appearing in Postgres.

What runs for real here
-----------------------
* the full ``ResourceService.create_multiple_resources`` pipeline (real save,
  real DB rows, real background indexer),
* LightRAG's real chunking + pipeline (doc-status and KV in Postgres, knowledge
  graph in Neo4j, chunk vectors in PostgreSQL via its own PGVector storage), and
* the repo's own retrieval path (``SiloService.search_silo_documents_router``
  with a ``lightrag_query_mode``), which dispatches ``aquery_llm`` onto the
  collection's dedicated persistent event loop (commit 6fa7f59a) — ingestion
  and querying both go through that mechanism, never a throwaway loop.

What is stubbed (and why)
-------------------------
Only the two model adapters behind ``AIService``/``EmbeddingService``:
``build_llm_model_func`` returns canned entity/keyword JSON so no external LLM
API is called, and ``build_embedding_func`` returns a deterministic hashed
bag-of-words embedding so retrieval similarity is real (overlapping vocabulary
ranks first) yet fully offline. This proves the *pipeline* works end-to-end;
LLM/embedding *quality* is a separate benchmark
(docs/testing/lightrag_extraction_benchmark_corpus.md).

Environment requirements (validated before running, otherwise SKIP)
-------------------------------------------------------------------
* ``lightrag-hku`` importable (the backend image pins 1.5.6),
* ``LIGHTRAG_ENABLED=true``, ``NEO4J_URI``/``NEO4J_PASSWORD`` set,
* the Neo4j server reachable, and
* the test's workspace (``silo_<id>``) absent from Neo4j — the Neo4j server is
  shared with dev data, so the test uses an explicitly out-of-range silo id and
  REFUSES to run if that label already exists (it deletes only its own label).

Vector storage is LightRAG's PGVector backend against the test database, not
the shared Qdrant: nothing this test writes can touch another environment's
vectors, and ``pgvector/pgvector:pg17`` ships the ``vector`` extension LightRAG
enables itself.

Per the plan: if this environment cannot stand LightRAG up, the constraint is
reported here rather than silently downgrading the test to another vector
store — AC-8 is specifically about LightRAG.
"""
import asyncio
import hashlib
import json
import os
import re
import shutil
import socket
import tempfile
import time
import unittest.mock as mock
import urllib.parse

import pytest

lightrag = pytest.importorskip(
    "lightrag",
    reason=(
        "lightrag-hku is not installed in this environment. AC-8 requires real "
        "LightRAG (see docs/dependencies/lightrag.md §1); run this test where it "
        "is available — e.g. the backend image, which pins lightrag-hku==1.5.6."
    ),
)

from db.database import SessionLocal
from models.app import App
from models.ai_service import AIService
from models.embedding_service import EmbeddingService
from models.repository import Repository
from models.resource import Resource
from models.silo import Silo
from models.user import User
from services.silo_service import SiloService

from tests.integration.blob_ingest_helpers import (
    FakeBlob,
    FakeContainerClient,
    ingest_payload,
)

# Out-of-range workspace id: the Neo4j server is shared with dev data, whose
# silo ids are small. A label collision would make the test both read and
# DELETE non-test data — guarded below, never assumed.
_TEST_SILO_ID = 987_654_321
COLLECTION_NAME = f"silo_{_TEST_SILO_ID}"

INDEXING_TIMEOUT_SECONDS = 300
POLL_INTERVAL_SECONDS = 2

# The planted facts: each fixture PDF carries exactly one, worded so a query
# for it only matches that document's vocabulary.
FACTS = {
    "DTC-1140-bomba-circulacion.pdf": (
        "Ficha técnica DTC-1140. La bomba de circulación modelo SOLARIUM-7 "
        "opera a una presión nominal de 4,2 bar y una temperatura máxima de "
        "trabajo de 85 grados Celsius."
    ),
    "DTC-2210-valvula-seguridad.pdf": (
        "Ficha técnica DTC-2210. La válvula de seguridad VMS-2210 se abre a "
        "una presión de calibrado de 7,5 bar."
    ),
    "MAN-9933-mantenimiento.pdf": (
        "Manual MAN-9933. El plan de mantenimiento del motor TURBOX-9933 "
        "exige cambio de aceite cada 2000 horas de funcionamiento."
    ),
}

QUERY_FOR_FACT = {
    "DTC-1140-bomba-circulacion.pdf": "¿A qué presión nominal opera la bomba de circulación SOLARIUM-7?",
    "DTC-2210-valvula-seguridad.pdf": "¿A qué presión de calibrado se abre la válvula de seguridad VMS-2210?",
    "MAN-9933-mantenimiento.pdf": "¿Cada cuántas horas hay que cambiar el aceite del motor TURBOX-9933?",
}

# The distinctive code token of each fact (SOLARIUM-7, VMS-2210, TURBOX-9933):
# the string a retrieved chunk must contain to prove the document's content
# actually made it into LightRAG's storage.
FACT_CODES = {
    blob_name: re.findall(r"\b[A-Z][A-Z0-9-]{4,}\b", fact)[0]
    for blob_name, fact in FACTS.items()
}


def _write_fact_pdf(path: str, fact_text: str) -> None:
    """A one-page PDF with real extractable text carrying one planted fact."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(
        pymupdf.Rect(50, 72, 545, 700),
        fact_text + "\n\nDocumento de prueba generado para la validación LightRAG.",
        fontsize=12,
        fontname="helv",
    )
    doc.save(path)
    doc.close()


# ---------------------------------------------------------------------------
# Stubs for the two model adapters (no external LLM/embedding API calls)
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 256
_TOKEN_BUCKET_RE = re.compile(r"[a-záéíóúüñ0-9]{3,}")
_FACT_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9-]{4,}\b")


def _bucket(token: str) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % _EMBEDDING_DIM


def _embed_one(text: str):
    dense = [0.0] * _EMBEDDING_DIM
    for token in _TOKEN_BUCKET_RE.findall(text.lower()):
        dense[_bucket(token)] += 1.0
    norm = sum(d * d for d in dense) ** 0.5 or 1.0
    return [d / norm for d in dense]


def _stub_entities_for(input_text: str) -> dict:
    """Entity-extraction JSON: one TECHNICAL entity per distinctive token.

    LightRAG parses this with its real ``_process_json_extraction_result`` —
    the pipeline machinery (chunking, graph upserts, doc-status) runs for real.
    """
    names = list(dict.fromkeys(_FACT_TOKEN_RE.findall(input_text)))[:6]
    entities = [
        {"name": name, "type": "TECHNICAL",
         "description": f"{name} aparece mencionado en el documento ingerido."}
        for name in names
    ]
    relationships = [
        {"source": source, "target": target, "keywords": "ficha técnica",
         "description": f"{source} y {target} aparecen en el mismo documento."}
        for source, target in zip(names, names[1:])
    ]
    return {"entities": entities, "relationships": relationships}


def _stub_llm_calls_recorder():
    calls = []

    async def fake_llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        text = f"{system_prompt or ''}\n{prompt}"
        calls.append(text)
        if '"high_level_keywords"' in text:
            # LightRAG's keyword-extraction call: one prompt with the query at
            # the end (see extract_keywords_only, lightrag/operate.py, 1.5.6).
            query = (
                prompt.split("User Query:")[-1].split("---Output")[0]
                if "User Query:" in prompt else prompt
            )
            words = list(dict.fromkeys(
                re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ][\wÁÉÍÓÚÜÑáéíóúüñ-]{3,}", query)
            ))[:8]
            return json.dumps({"high_level_keywords": words[:4], "low_level_keywords": words[4:8]})
        if '"entities"' in text and '"relationships"' in text:
            # LightRAG's entity-extraction call: the chunk text sits inside the
            # ---Input Text--- section of the user prompt.
            input_text = prompt.split("---Input Text---")[-1].split("---Completion")[0]
            return json.dumps(_stub_entities_for(input_text))
        # Summary/other calls: plain text is fine.
        return "OK."

    return fake_llm_model_func, calls


def _stub_embedding_func():
    import threading

    import numpy as np

    calls = []

    async def fake_embed_func(texts):
        calls.append({
            "texts": list(texts),
            # The LightRAG coroutines must run on the collection's DEDICATED
            # persistent loop thread (commit 6fa7f59a) — record who called us.
            "thread": threading.current_thread().name,
        })
        if not texts:
            return np.zeros((0, _EMBEDDING_DIM), dtype=np.float32)
        return np.asarray([_embed_one(t) for t in texts], dtype=np.float32)

    from lightrag.utils import EmbeddingFunc

    return EmbeddingFunc(
        embedding_dim=_EMBEDDING_DIM,
        max_token_size=8192,
        func=fake_embed_func,
        model_name="stub-hashed-bow",
    ), calls


def _global_dns_stub(host, *args, **kwargs):
    """Resolve the fake blob host to a routable address; everything else (the
    Neo4j hostname included) goes to the REAL resolver — the LightRAG driver
    must reach the actual server, not this stub."""
    if isinstance(host, str) and host.endswith(".blob.core.windows.net"):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("20.60.72.1", 0))]
    return _real_getaddrinfo(host, *args, **kwargs)


_real_getaddrinfo = socket.getaddrinfo


# ---------------------------------------------------------------------------
# Environment gates — the test refuses to run where LightRAG cannot stand up,
# and never touches a Neo4j workspace it did not create itself.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _require_real_lightrag_backend():
    import config

    if not getattr(config, "LIGHTRAG_ENABLED", False):
        pytest.skip(
            "LIGHTRAG_ENABLED is false in this environment. AC-8 requires real "
            "LightRAG; set LIGHTRAG_ENABLED=true plus NEO4J_URI/NEO4J_PASSWORD "
            "to run it (e.g. the backend container)."
        )
    for required in ("NEO4J_URI", "NEO4J_PASSWORD"):
        if not getattr(config, required, None):
            pytest.skip(f"{required} is not configured; LightRAG cannot be stood up here.")

    parsed = urllib.parse.urlsplit(config.NEO4J_URI)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 7687), timeout=3):
            pass
    except OSError as exc:
        pytest.skip(f"Neo4j at {config.NEO4J_URI} is not reachable ({exc}); AC-8 cannot run here.")


def _neo4j_label_count(label: str) -> int:
    from neo4j import GraphDatabase

    import config

    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            return session.run(
                f"MATCH (n:`{label}`) RETURN count(n) AS c"  # noqa: S608 — label from a module constant
            ).single()["c"]
    finally:
        driver.close()


# ---------------------------------------------------------------------------
# One shared end-to-end run for the whole module: ingest once, then assert the
# different aspects of that same run (the real pipeline + LightRAG indexing is
# the expensive part; re-running it per test would triple the wall time).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lightrag_e2e(request, test_engine):  # noqa: ARG001 — test_engine ensures the schema exists
    from tools.vector_stores.lightrag import adapters
    from services import resource_service, silo_service
    from services.azure_blob_ingest_service import AzureBlobIngestService
    from services.blob import azure_blob_client

    # 1. Workspace guard: refuse to touch a shared backend that already has
    #    this label.
    if _neo4j_label_count(COLLECTION_NAME) > 0:
        pytest.skip(
            f"Neo4j already contains workspace label '{COLLECTION_NAME}' — refusing to "
            "run the LightRAG e2e against a shared backend with pre-existing data."
        )

    stub_llm, llm_calls = _stub_llm_calls_recorder()
    stub_embedding, embedding_calls = _stub_embedding_func()

    fake_client = FakeContainerClient()
    repo_base = tempfile.mkdtemp(prefix="lightrag-e2e-repos-")

    patchers = [
        mock.patch.object(resource_service, "REPO_BASE_FOLDER", repo_base),
        mock.patch.object(silo_service, "REPO_BASE_FOLDER", repo_base),
        mock.patch.object(azure_blob_client, "build_container_client", lambda cfg: fake_client),
        mock.patch.object(socket, "getaddrinfo", _global_dns_stub),
        mock.patch.object(adapters, "build_llm_model_func", lambda *a, **k: stub_llm),
        mock.patch.object(adapters, "build_embedding_func", lambda *a, **k: stub_embedding),
    ]
    for patcher in patchers:
        patcher.start()
    repository_id = None
    try:
        # 2. Real-committed entities (the background indexer opens its own
        #    SessionLocal sessions, so the savepoint-style `db` fixture cannot
        #    be used here).
        db = SessionLocal()
        try:
            user = User(
                name="Blob LightRAG E2E",
                email=f"blob-lightrag-e2e-{time.time_ns()}@test.com",
                is_active=True,
                platform_role="editor",
            )
            db.add(user)
            db.flush()

            app = App(name="Blob LightRAG E2E App", owner_id=user.user_id)
            db.add(app)
            db.flush()

            ai_service = AIService(
                name="Stub Extract LLM", provider="OpenAI", api_key="stub",  # pragma: allowlist secret
                app_id=app.app_id,
            )
            embedding_service = EmbeddingService(
                name="Stub Embedding", provider="OpenAI", api_key="stub",  # pragma: allowlist secret
                app_id=app.app_id,
            )
            db.add(ai_service)
            db.add(embedding_service)
            db.flush()

            # Explicit out-of-range silo id (see _TEST_SILO_ID) — keeps the
            # shared Neo4j/Qdrant namespaces collision-free.
            silo = Silo(
                silo_id=_TEST_SILO_ID,
                name="Blob LightRAG E2E Silo",
                silo_type="REPO",
                app_id=app.app_id,
                vector_db_type="LIGHTRAG",
                lightrag_vector_db_type="PGVECTOR",
                extract_service_id=ai_service.service_id,
                embedding_service_id=embedding_service.service_id,
                lightrag_language="Spanish",
            )
            db.add(silo)
            db.flush()

            repository = Repository(
                name="Blob LightRAG E2E Repository", type="REPO", status="active",
                app_id=app.app_id, silo_id=silo.silo_id,
            )
            db.add(repository)
            db.commit()
            repository_id, silo_id, app_id = (
                repository.repository_id, silo.silo_id, app.app_id,
            )
        finally:
            db.close()

        # 3. The fixture PDFs, served as Azure blobs.
        for blob_name, fact in FACTS.items():
            with tempfile.TemporaryDirectory() as tmp:
                pdf_path = os.path.join(tmp, "fixture.pdf")
                _write_fact_pdf(pdf_path, fact)
                with open(pdf_path, "rb") as fh:
                    fake_client.add_blob(FakeBlob(blob_name, content=fh.read(), etag=f'"etag-{blob_name}"'))

        # 4. One real ingestion run (the first batch runs synchronously inside
        #    trigger_ingestion; there is only one batch, so no background run
        #    thread is involved).
        db = SessionLocal()
        try:
            result = AzureBlobIngestService.trigger_ingestion(
                app_id, repository_id, ingest_payload(), db,
            )
        finally:
            db.close()

        statuses = _wait_for_indexing_completion(repository_id)

        yield {
            "result": result,
            "repository_id": repository_id,
            "silo_id": silo_id,
            "app_id": app_id,
            "statuses": statuses,
            "fake_client": fake_client,
            "llm_calls": llm_calls,
            "embedding_calls": embedding_calls,
            "collection_name": f"silo_{silo_id}",
            "repo_base": repo_base,
        }
    finally:
        # 5. Cleanup — every backend touched by the run, in dependency order.
        #    The stubs are STILL ACTIVE here, so delete_collection's internal
        #    _get_rag_instance builds with the same offline adapters.
        from tools.vector_stores.lightrag_store import LightRAGStore

        cleanup_store = LightRAGStore(
            db=SessionLocal(),
            ai_service=SimpleAiService(),
            embedding_service=SimpleEmbeddingService(),
            lightrag_vector_db_type="PGVECTOR",
        )
        try:
            cleanup_store.delete_collection(COLLECTION_NAME)
        except Exception:  # noqa: BLE001 — cleanup must never mask the test result
            pass
        finally:
            for loop in list(cleanup_store._collection_loops.values()):
                loop.close()

        _delete_lightrag_rows(COLLECTION_NAME)
        if repository_id is not None:
            _delete_db_rows(repository_id)
        shutil.rmtree(repo_base, ignore_errors=True)

        for patcher in reversed(patchers):
            patcher.stop()


class SimpleAiService:
    """Lightest stand-in satisfying LightRAGStore.__init__ for the cleanup run."""

    provider = "Custom"
    name = "cleanup"
    description = None
    api_key = ""  # pragma: allowlist secret
    endpoint = None


class SimpleEmbeddingService:
    provider = "Custom"
    name = "cleanup"
    description = None
    api_key = ""  # pragma: allowlist secret
    endpoint = None
    api_version = None


def _delete_lightrag_rows(workspace: str) -> None:
    """Remove this workspace's rows from every LightRAG PostgreSQL table.

    ``LightRAGStore.delete_collection`` leaves them behind — its
    ``_cleanup_postgres`` helper targets the JSON-backend table names instead
    of the real LightRAG PostgreSQL ones (documented bug in
    docs/dependencies/lightrag.md §8), so the test cleans up after itself.
    The tables are lowercase (``lightrag_*``, verified against lightrag-hku
    1.5.6) and their vector variants carry the embedding-model suffix, hence
    the case-insensitive prefix match.
    """
    from sqlalchemy import text

    db = SessionLocal()
    try:
        tables = db.execute(
            text("SELECT tablename FROM pg_tables WHERE tablename ILIKE 'lightrag%'")
        ).scalars().all()
        for table in tables:
            try:
                db.execute(
                    text(f'DELETE FROM "{table}" WHERE workspace = :workspace'),  # noqa: S608 — table from the LIGHTRAG% set
                    {"workspace": workspace},
                )
            except Exception:  # noqa: BLE001 — a table without a workspace column
                continue
        db.commit()
    finally:
        db.close()


def _delete_db_rows(repository_id: int) -> None:
    from models.indexing_metric import IndexingMetric

    db = SessionLocal()
    try:
        db.query(Resource).filter(Resource.repository_id == repository_id).delete(synchronize_session=False)
        db.query(IndexingMetric).filter(IndexingMetric.silo_id == _TEST_SILO_ID).delete(synchronize_session=False)
        db.query(Repository).filter(Repository.repository_id == repository_id).delete(synchronize_session=False)
        db.query(Silo).filter(Silo.silo_id == _TEST_SILO_ID).delete(synchronize_session=False)
        app_ids = [a.app_id for a in db.query(App).filter(App.name == "Blob LightRAG E2E App").all()]
        if app_ids:
            db.query(AIService).filter(AIService.app_id.in_(app_ids)).delete(synchronize_session=False)
            db.query(EmbeddingService).filter(EmbeddingService.app_id.in_(app_ids)).delete(synchronize_session=False)
            db.query(App).filter(App.app_id.in_(app_ids)).delete(synchronize_session=False)
            # User goes LAST: App.owner_id still references it until now.
            db.query(User).filter(User.name == "Blob LightRAG E2E").delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


def _wait_for_indexing_completion(repository_id: int):
    """Poll until every Resource left the pending/indexing states (or fail)."""
    deadline = time.monotonic() + INDEXING_TIMEOUT_SECONDS
    statuses = []
    while time.monotonic() < deadline:
        db = SessionLocal()
        try:
            statuses = [
                row.status for row in
                db.query(Resource.status).filter(Resource.repository_id == repository_id).all()
            ]
        finally:
            db.close()
        if statuses and all(s not in ("pending", "indexing") for s in statuses):
            return statuses
        time.sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError(f"Indexing did not complete within {INDEXING_TIMEOUT_SECONDS}s: {statuses}")


def _search(silo_id: int, query: str, mode: str = "naive", limit: int = 8):
    """The repo's own LightRAG retrieval path (only_need_context, no LLM)."""
    db = SessionLocal()
    try:
        return asyncio.run(SiloService.search_silo_documents_router(
            silo_id, query, lightrag_query_mode=mode, limit=limit, db=db,
        ))
    finally:
        db.close()


def _chunk_resource_id(chunk_metadata: dict):
    """Parse ``res<resource_id>-p<page>-chunk-<n>`` — LightRAG's chunk ids are
    built by this repo from Resource metadata (FR-6 provenance)."""
    match = re.match(r"^res(\d+)-p\d+-chunk-\d+$", (chunk_metadata or {}).get("chunk_id", ""))
    return int(match.group(1)) if match else None


def _label_prefix_of(file_path_label: str):
    """Strip LightRAG's `` (p. N)`` page suffix, then the extension — the
    prefix ``Resource.uri`` is staged under (see _sanitize_blob_filename)."""
    label = re.sub(r"\s*\(p\.\s*\d+\)\s*$", "", file_path_label or "")
    return os.path.splitext(os.path.basename(label))[0]


# ---------------------------------------------------------------------------
# The validation itself (AC-8)
# ---------------------------------------------------------------------------


class TestLightRagEndToEnd:
    def test_ingestion_indexes_every_fixture_into_light_rag(self, lightrag_e2e):
        run = lightrag_e2e
        result = run["result"]

        # The one-shot endpoint reported the three blobs as queued…
        assert result["queued"] == len(FACTS)
        assert result["failed"] == 0
        assert result["session_id"]

        # …and the background indexer actually finished them (real pipeline).
        assert run["statuses"], "no Resources were created"
        assert all(status == "ready" for status in run["statuses"]), run["statuses"]

        # The LightRAG pipeline really ran: entity extraction (the LLM stub)
        # was called once per chunk, embeddings too.
        assert run["llm_calls"], "LightRAG never invoked the LLM adapter"
        assert run["embedding_calls"], "LightRAG never invoked the embedding adapter"

    def test_retrieval_returns_chunks_traceable_to_the_ingested_blobs(self, lightrag_e2e):
        """AC-8's core assertion: for every fixture document, a query whose
        answer exists only there returns a chunk whose content carries the
        planted fact, whose chunk_id parses back to that Resource, and whose
        file-path label carries the ingested blob's staged filename."""
        run = lightrag_e2e

        db = SessionLocal()
        try:
            resources = db.query(Resource).filter(Resource.repository_id == run["repository_id"]).all()
            # file_path labels are built from Resource.uri; map label prefix ->
            # (resource_id, blob_name) for the provenance assertions below.
            by_uri_prefix = {
                os.path.splitext(r.uri)[0]: (r.resource_id, (r.extra_metadata or {}).get("blob_name"))
                for r in resources
            }
        finally:
            db.close()
        assert len(by_uri_prefix) == len(FACTS)
        assert all(blob for _, blob in by_uri_prefix.values())

        for blob_name, expected_code in FACT_CODES.items():
            search = _search(run["silo_id"], QUERY_FOR_FACT[blob_name])
            chunks = search["results"]
            assert chunks, f"LightRAG returned no chunks for the query about {blob_name}"

            with_fact = [
                chunk for chunk in chunks if expected_code in chunk["page_content"]
            ]
            assert with_fact, (
                f"no retrieved chunk contains the planted fact code {expected_code!r} "
                f"of {blob_name}; chunks: {[c['page_content'][:80] for c in chunks]}"
            )

            source = with_fact[0]["metadata"]
            provenance = by_uri_prefix.get(_label_prefix_of(source.get("file_path", "")))
            assert provenance, (
                f"chunk file_path {source.get('file_path')!r} does not map back to any "
                "ingested Resource (FR-6 provenance chain broken)"
            )
            resource_id, source_blob = provenance
            assert source_blob == blob_name, (
                f"the fact of {blob_name} was attributed to {source_blob}"
            )
            if source.get("chunk_id"):
                assert _chunk_resource_id(source) == resource_id

    def test_query_runs_on_the_collections_persistent_event_loop(self, lightrag_e2e):
        """Retrieval must go through the per-collection dedicated loop (commit
        6fa7f59a) — the bug this test guards: a throwaway loop per query crash-
        ing workers. Every embedding call the pipeline makes (indexing AND the
        retrieval that follows) must therefore land on one single thread: the
        collection's persistent ``lightrag-loop-silo_<id>`` one. A query that
        had run on a fresh throwaway loop would show a second thread here."""
        run = lightrag_e2e
        collection_thread = f"lightrag-loop-silo_{run['silo_id']}"

        _search(run["silo_id"], QUERY_FOR_FACT["DTC-2210-valvula-seguridad.pdf"])

        assert run["embedding_calls"], "no embedding call was ever made"
        threads = {call["thread"] for call in run["embedding_calls"]}
        assert threads == {collection_thread}, (
            f"LightRAG coroutines ran on {threads}; expected only the collection's "
            f"persistent loop thread {collection_thread!r}"
        )

    def test_second_run_skips_everything_and_lightrag_content_survives(self, lightrag_e2e):
        """Idempotency holds on a LightRAG silo too: a no-op re-run (same
        ETags) downloads nothing, creates nothing, and leaves what is already
        indexed retrievable."""
        from services.azure_blob_ingest_service import AzureBlobIngestService

        run = lightrag_e2e
        fake_client = run["fake_client"]
        downloads_before = len(fake_client.download_blob_calls)

        db = SessionLocal()
        try:
            result = AzureBlobIngestService.trigger_ingestion(
                run["app_id"], run["repository_id"], ingest_payload(), db,
            )
        finally:
            db.close()

        assert result["queued"] == 0
        assert result["skipped_unchanged"] == len(FACTS)
        assert len(fake_client.download_blob_calls) == downloads_before

        search = _search(run["silo_id"], QUERY_FOR_FACT["DTC-2210-valvula-seguridad.pdf"])
        assert search["results"], "LightRAG lost the indexed content after a no-op re-run"
        codes = [c["page_content"] for c in search["results"]]
        assert any(FACT_CODES["DTC-2210-valvula-seguridad.pdf"] in content for content in codes)
