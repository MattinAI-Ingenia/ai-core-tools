"""Shared offline stubs and environment gate for real-LightRAG integration tests.

Only the two model adapters behind ``AIService``/``EmbeddingService`` are
stubbed (canned entity JSON, deterministic hashed bag-of-words embeddings);
LightRAG itself, Neo4j and the PostgreSQL storages are real.
"""
import hashlib
import json
import re
import socket
import urllib.parse

import pytest

from db.database import SessionLocal

EMBEDDING_DIM = 256
TOKEN_BUCKET_RE = re.compile(r"[a-záéíóúüñ0-9]{3,}")
FACT_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9-]{4,}\b")


def require_real_lightrag_backend() -> None:
    """Skip unless LightRAG + a reachable Neo4j are configured."""
    import config

    if not getattr(config, "LIGHTRAG_ENABLED", False):
        pytest.skip(
            "LIGHTRAG_ENABLED is false in this environment. Real LightRAG is "
            "required; set LIGHTRAG_ENABLED=true plus NEO4J_URI/NEO4J_PASSWORD "
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
        pytest.skip(f"Neo4j at {config.NEO4J_URI} is not reachable ({exc}); cannot run here.")


def bucket(token: str) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % EMBEDDING_DIM


def embed_one(text: str):
    dense = [0.0] * EMBEDDING_DIM
    for token in TOKEN_BUCKET_RE.findall(text.lower()):
        dense[bucket(token)] += 1.0
    norm = sum(d * d for d in dense) ** 0.5 or 1.0
    return [d / norm for d in dense]


def stub_entities_for(input_text: str) -> dict:
    """Entity-extraction JSON: one TECHNICAL entity per distinctive token.

    LightRAG parses this with its real ``_process_json_extraction_result`` —
    the pipeline machinery (chunking, graph upserts, doc-status) runs for real.
    """
    names = list(dict.fromkeys(FACT_TOKEN_RE.findall(input_text)))[:6]
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


def stub_llm_calls_recorder():
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
            return json.dumps(stub_entities_for(input_text))
        # Summary/other calls: plain text is fine.
        return "OK."

    return fake_llm_model_func, calls


def stub_embedding_func():
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
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        return np.asarray([embed_one(t) for t in texts], dtype=np.float32)

    from lightrag.utils import EmbeddingFunc

    return EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=8192,
        func=fake_embed_func,
        model_name="stub-hashed-bow",
    ), calls


class SimpleAiService:
    """Lightest stand-in satisfying LightRAGStore.__init__."""

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


def neo4j_label_count(label: str) -> int:
    from neo4j import GraphDatabase

    import config

    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            return session.run(
                f"MATCH (n:`{label}`) RETURN count(n) AS c"  # noqa: S608 — label from a test constant
            ).single()["c"]
    finally:
        driver.close()


def delete_lightrag_rows(workspace: str) -> None:
    """Remove this workspace's rows from every LightRAG PostgreSQL table.

    Belt-and-braces test cleanup, independent of ``delete_collection``, so a
    failing test never leaves rows behind. The tables are lowercase (``lightrag_*``, verified against lightrag-hku
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
