"""Unit tests for ``tools.vector_stores.lightrag_store.LightRAGStore``.

All LightRAG / Neo4j / Qdrant / PostgreSQL dependencies are mocked so these
tests run without network access.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from tools.vector_stores.lightrag_store import LightRAGStore, LightRAGRetriever

# pytest.ini does not enable pytest-asyncio's auto mode, so mark every
# async test in this module explicitly.
pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ai_service():
    return SimpleNamespace(
        provider="OpenAI",
        name="test-llm",
        description="gpt-4o",
        api_key="sk-test",
        endpoint=None,
    )


def _make_embedding_service():
    return SimpleNamespace(
        provider="OpenAI",
        name="test-embed",
        description="text-embedding-3-small",
        api_key="sk-test",
        endpoint=None,
        api_version=None,
    )


@pytest.fixture
def store():
    db = MagicMock()
    return LightRAGStore(
        db=db,
        ai_service=_make_ai_service(),
        embedding_service=_make_embedding_service(),
    )


@pytest.fixture
def mock_rag():
    rag = MagicMock()
    rag.ainsert = AsyncMock()
    rag.aquery = AsyncMock(return_value="LightRAG response")
    rag.adelete_by_doc_id = AsyncMock()
    rag.initialize_storages = AsyncMock()
    return rag


# ---------------------------------------------------------------------------
# index_documents
# ---------------------------------------------------------------------------


def test_index_documents_calls_ainsert(store, mock_rag):
    store._get_rag_instance = MagicMock(return_value=mock_rag)

    docs = [
        Document(page_content="Hello world"),
        Document(page_content="Second doc"),
    ]
    with patch("tools.vector_stores.lightrag_store._ainsert_with_progress", new_callable=AsyncMock) as mock_insert:
        store.index_documents("silo_1", docs)

    mock_insert.assert_called_once()
    texts = mock_insert.call_args[0][1]
    assert texts == ["Hello world", "Second doc"]


def test_index_documents_skips_empty(store):
    store._get_rag_instance = MagicMock()

    store.index_documents("silo_1", [])

    store._get_rag_instance.assert_not_called()


def test_index_documents_skips_blank_content(store, mock_rag):
    store._get_rag_instance = MagicMock(return_value=mock_rag)

    # page_content="" is falsy → filtered out by `if doc.page_content`
    docs = [
        Document(page_content=""),
    ]
    store.index_documents("silo_1", docs)

    mock_rag.ainsert.assert_not_called()


# ---------------------------------------------------------------------------
# delete_documents
# ---------------------------------------------------------------------------


def test_delete_documents_is_noop(store, mock_rag):
    # delete_documents is a deliberate no-op (per-document graph deletion unsupported)
    store._get_rag_instance = MagicMock(return_value=mock_rag)

    store.delete_documents("silo_1", ["doc1", "doc2"])

    mock_rag.adelete_by_doc_id.assert_not_called()


# ---------------------------------------------------------------------------
# delete_collection
# ---------------------------------------------------------------------------


def test_delete_collection_removes_cached_instance(store, mock_rag):
    store._rag_instances["silo_1"] = mock_rag

    with patch.object(store, "_cleanup_neo4j"), \
         patch.object(store, "_cleanup_qdrant"), \
         patch.object(store, "_cleanup_postgres"):
        store.delete_collection("silo_1")

    assert "silo_1" not in store._rag_instances


# ---------------------------------------------------------------------------
# search_similar_documents
# ---------------------------------------------------------------------------


def test_search_similar_documents_calls_aquery(store, mock_rag):
    mock_rag.aquery.return_value = "Some response"
    store._get_rag_instance = MagicMock(return_value=mock_rag)

    with patch("tools.vector_stores.lightrag_store.QueryParam", create=True) as MockQP:
        # Patch the lazy import of QueryParam inside the method
        with patch.dict("sys.modules", {"lightrag.base": MagicMock(QueryParam=MockQP)}):
            results = store.search_similar_documents("silo_1", "test query", k=5)

    assert len(results) == 1
    assert results[0].page_content == "Some response"
    assert results[0].metadata == {"source": "lightrag", "query_mode": "hybrid", "lightrag_keywords": {}}


def test_search_similar_documents_bypass_returns_empty(store):
    store._get_rag_instance = MagicMock()

    with patch.dict("sys.modules", {"lightrag.base": MagicMock()}):
        results = store.search_similar_documents(
            "silo_1", "test query", search_type="bypass"
        )

    assert results == []
    store._get_rag_instance.assert_not_called()


# ---------------------------------------------------------------------------
# get_retriever
# ---------------------------------------------------------------------------


def test_get_retriever_returns_lightrag_retriever(store, mock_rag):
    store._get_rag_instance = MagicMock(return_value=mock_rag)

    retriever = store.get_retriever("silo_1")

    assert isinstance(retriever, LightRAGRetriever)
    assert retriever.query_mode == "hybrid"


def test_get_retriever_with_custom_query_mode(store, mock_rag):
    store._get_rag_instance = MagicMock(return_value=mock_rag)

    retriever = store.get_retriever(
        "silo_1", search_params={"lightrag_query_mode": "local"}
    )

    assert isinstance(retriever, LightRAGRetriever)
    assert retriever.query_mode == "local"


# ---------------------------------------------------------------------------
# count_documents
# ---------------------------------------------------------------------------


def test_count_documents_queries_postgres(store):
    mock_result = MagicMock()
    mock_result.scalar.return_value = 42
    store.db.execute.return_value = mock_result

    count = store.count_documents("silo_1")

    assert count == 42
    store.db.execute.assert_called_once()


# ---------------------------------------------------------------------------
# update_documents_metadata
# ---------------------------------------------------------------------------


def test_update_documents_metadata_returns_zero(store):
    result = store.update_documents_metadata(
        "silo_1",
        filter_metadata={"source": "test"},
        metadata_updates={"tag": "new"},
    )

    assert result == 0


# ---------------------------------------------------------------------------
# get_distinct_metadata_values
# ---------------------------------------------------------------------------


def test_get_distinct_metadata_values_returns_empty(store):
    result = store.get_distinct_metadata_values("silo_1", "source")

    assert result == []
