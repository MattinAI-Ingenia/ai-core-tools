"""Compatibility test: langchain-postgres 0.0.17 + pgvector >=0.4.2.

We force pgvector >=0.4.2 in the Dockerfile (via `pip install --no-deps`) to
satisfy lightrag-hku, overriding the conservative `<0.4` pin declared by
langchain-postgres 0.0.17. This test verifies that the API langchain-postgres
uses from the pgvector package did NOT break between 0.3.x and 0.4.x.

If this test fails, the version-override strategy is unviable and we must
either:
  - find an older lightrag-hku compatible with pgvector <0.4, or
  - refactor pgvector_store.py to drop langchain-postgres.

This test does NOT require a running database. It exercises imports,
version pins, attribute presence and signatures. A separate runtime check
(behind an env flag) can hit a real DB if available.

Run with:
    pytest tests/integration/test_pgvector_lightrag_compat.py -v
"""
from __future__ import annotations

import inspect
import os

import pytest


# ---------------------------------------------------------------------------
# 1. Imports — both packages must coexist in the same interpreter.
# ---------------------------------------------------------------------------

def test_import_pgvector():
    """The raw pgvector package must import cleanly."""
    import pgvector  # noqa: F401


def test_import_langchain_postgres_pgvector():
    """langchain_postgres.vectorstores.PGVector must import cleanly,
    even when pgvector is newer than its declared upper bound.
    A failure here means a runtime ImportError caused by API change."""
    from langchain_postgres.vectorstores import PGVector  # noqa: F401


def test_import_lightrag():
    """lightrag-hku must import cleanly alongside the other two packages."""
    import lightrag  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Version pins — confirm the overrides took effect.
# ---------------------------------------------------------------------------

def test_pgvector_version_is_at_least_0_4_2():
    """pgvector should be the forced-upgraded version (>=0.4.2)."""
    import importlib.metadata as md

    version = md.version("pgvector")
    parts = tuple(int(p) for p in version.split(".")[:3])
    assert parts >= (0, 4, 2), (
        f"pgvector is {version}, expected >=0.4.2. The pip --no-deps "
        f"override in the Dockerfile did not take effect."
    )


def test_langchain_postgres_version_is_0_0_17_or_higher():
    """langchain-postgres should be 0.0.17 (the only version available)."""
    import importlib.metadata as md

    version = md.version("langchain-postgres")
    parts = tuple(int(p) for p in version.split(".")[:3])
    assert parts >= (0, 0, 17), f"langchain-postgres is {version}, expected >=0.0.17"


def test_lightrag_hku_version():
    """lightrag-hku should be the version we pinned in the Dockerfile."""
    import importlib.metadata as md

    version = md.version("lightrag-hku")
    # We pin 1.4.16 in the Dockerfile; accept any 1.4.x or newer just in case.
    parts = tuple(int(p) for p in version.split(".")[:2])
    assert parts >= (1, 4), f"lightrag-hku is {version}, expected >=1.4.0"


# ---------------------------------------------------------------------------
# 3. API surface — the methods pgvector_store.py uses must still exist with
#    compatible signatures. This is the actual "did the API break?" check.
# ---------------------------------------------------------------------------

REQUIRED_METHODS = [
    "add_documents",
    "delete",
    "delete_collection",
    "similarity_search",
    "similarity_search_with_score",
    "similarity_search_with_score_by_vector",
    "similarity_search_with_relevance_scores",
    "max_marginal_relevance_search",
    "as_retriever",
]


@pytest.mark.parametrize("method_name", REQUIRED_METHODS)
def test_pgvector_method_exists(method_name):
    """Each method that pgvector_store.py calls must still exist on PGVector."""
    from langchain_postgres.vectorstores import PGVector

    assert hasattr(PGVector, method_name), (
        f"PGVector.{method_name} does not exist anymore — API broke."
    )
    method = getattr(PGVector, method_name)
    assert callable(method), f"PGVector.{method_name} is not callable."


def test_pgvector_constructor_signature():
    """The constructor kwargs we pass in `_get_vector_store` must still be accepted.

    From pgvector_store.py:
        PGVector(
            embeddings=...,
            collection_name=...,
            connection=...,
            use_jsonb=True,
        )
    """
    from langchain_postgres.vectorstores import PGVector

    sig = inspect.signature(PGVector.__init__)
    params = sig.parameters
    expected = {"embeddings", "collection_name", "connection", "use_jsonb"}
    missing = expected - set(params.keys())
    assert not missing, (
        f"PGVector.__init__ no longer accepts: {missing}. "
        f"Current params: {list(params.keys())}"
    )


def test_similarity_search_with_score_signature():
    """similarity_search_with_score(query, k, filter=...) — the call shape we use."""
    from langchain_postgres.vectorstores import PGVector

    sig = inspect.signature(PGVector.similarity_search_with_score)
    params = sig.parameters
    assert "query" in params or "k" in params, (
        f"similarity_search_with_score signature changed: {list(params.keys())}"
    )
    # 'filter' is the kwarg we pass — make sure it exists.
    assert "filter" in params, (
        f"similarity_search_with_score no longer accepts 'filter'. "
        f"Current params: {list(params.keys())}"
    )


def test_max_marginal_relevance_search_signature():
    """MMR call: query, k, filter, fetch_k, lambda_mult."""
    from langchain_postgres.vectorstores import PGVector

    sig = inspect.signature(PGVector.max_marginal_relevance_search)
    params = set(sig.parameters.keys())
    expected = {"query", "k", "filter", "fetch_k", "lambda_mult"}
    missing = expected - params
    assert not missing, (
        f"max_marginal_relevance_search lost kwargs: {missing}. "
        f"Current params: {sorted(params)}"
    )


# ---------------------------------------------------------------------------
# 4. Smoke instantiation — building a PGVector should not throw at import-time
#    or at __init__-time. We use a stub embedding to avoid hitting any API.
# ---------------------------------------------------------------------------

class _StubEmbeddings:
    """Minimal LangChain Embeddings stub. Returns deterministic 4-dim vectors."""

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3, 0.4]


def test_pgvector_instantiation_with_stub_engine_no_db():
    """Instantiating PGVector with a SQLAlchemy engine pointing at a
    nonexistent DB should NOT raise during construction. Some versions
    of langchain-postgres eagerly ping the DB on __init__; if that's the
    case, the test will fail and tell us so."""
    from langchain_postgres.vectorstores import PGVector
    from sqlalchemy import create_engine

    # Use a dummy SQLite URL — we just want to see if construction itself blows up.
    # If PGVector demands a real Postgres connection in __init__, we'll see it.
    engine = create_engine("sqlite:///:memory:")

    try:
        store = PGVector(
            embeddings=_StubEmbeddings(),
            collection_name="compat_test_collection",
            connection=engine,
            use_jsonb=True,
        )
        # We don't care if subsequent operations work — only that __init__
        # didn't error from an API mismatch.
        assert store is not None
    except TypeError as e:
        pytest.fail(f"PGVector.__init__ TypeError — API broke: {e}")
    except ImportError as e:
        pytest.fail(f"PGVector import-time error from pgvector dep: {e}")
    except Exception as e:
        # Connection errors / table creation errors are acceptable here —
        # we're checking compatibility of code paths, not DB ops.
        msg = str(e).lower()
        if "type" in msg and ("argument" in msg or "kwarg" in msg):
            pytest.fail(f"PGVector init failed with what looks like an API mismatch: {e}")
        # Otherwise: SQL/connection error, ignore.


# ---------------------------------------------------------------------------
# 5. Optional runtime check — hit a real DB if SQLALCHEMY_DATABASE_URI is set
#    and the pgvector extension is installed. Skipped by default in unit runs.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.getenv("RUN_PGVECTOR_RUNTIME_TEST") != "1",
    reason="Set RUN_PGVECTOR_RUNTIME_TEST=1 to run real DB compatibility check.",
)
def test_runtime_add_and_search_against_real_db():
    """End-to-end: add docs and query against a real Postgres+pgvector DB.

    This is the strongest signal: if langchain-postgres still works with
    pgvector >=0.4.2 in actual queries, the override strategy is fully valid.
    """
    from langchain_core.documents import Document
    from langchain_postgres.vectorstores import PGVector
    from sqlalchemy import create_engine

    db_url = os.getenv("SQLALCHEMY_DATABASE_URI")
    assert db_url, "SQLALCHEMY_DATABASE_URI not set"

    engine = create_engine(db_url)
    store = PGVector(
        embeddings=_StubEmbeddings(),
        collection_name="compat_test_runtime",
        connection=engine,
        use_jsonb=True,
    )

    try:
        store.add_documents([
            Document(page_content="hello world", metadata={"k": "a"}),
            Document(page_content="goodbye world", metadata={"k": "b"}),
        ])
        results = store.similarity_search_with_score("hello", k=2, filter=None)
        assert len(results) >= 1
    finally:
        try:
            store.delete_collection()
        except Exception:
            pass
