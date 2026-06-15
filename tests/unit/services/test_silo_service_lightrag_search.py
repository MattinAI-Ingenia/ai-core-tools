"""Unit tests for LightRAG retriever-based search path in SiloService."""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from backend.services.silo_service import SiloService


def _make_mock_doc(entities=None, relationships=None, chunks=None):
    graph_data = {
        "data": {
            "entities": entities or [{"id": "e1", "name": "Entity1"}],
            "relationships": relationships or [],
            "chunks": chunks or [{"id": "c1", "content": "chunk text"}],
        }
    }
    return Document(
        page_content="context string",
        metadata={"source": "lightrag", "query_mode": "hybrid", "lightrag_raw_data": graph_data},
    )


def _make_mock_silo(silo_id=1):
    mock_silo = MagicMock()
    mock_silo.id = silo_id
    mock_silo.embedding_service_id = 1
    mock_silo.embedding_service = MagicMock()
    mock_silo.vector_db_type = "LIGHTRAG"
    return mock_silo


@patch("backend.services.silo_service.SiloService.check_silo_collection_exists", return_value=True)
@patch("backend.services.silo_service._get_vector_store")
@patch("backend.services.silo_service.SiloService.get_silo")
def test_lightrag_query_mode_returns_chunks_and_graph(mock_get_silo, mock_get_store, _mock_exists):
    """When lightrag_query_mode is set, service uses retriever and returns chunks + lightrag_graph."""
    mock_silo = _make_mock_silo()
    mock_get_silo.return_value = mock_silo

    mock_doc = _make_mock_doc()
    mock_retriever = MagicMock()
    mock_retriever._get_relevant_documents.return_value = [mock_doc]
    mock_store = MagicMock()
    mock_store.get_retriever.return_value = mock_retriever
    mock_get_store.return_value = mock_store

    result = SiloService.search_silo_documents_router(
        silo_id=1,
        query="test query",
        lightrag_query_mode="hybrid",
        db=MagicMock(),
    )

    assert result is not None
    assert result["total_results"] == 1
    assert result["results"][0]["page_content"] == "chunk text"
    assert result["lightrag_graph"] is not None
    assert result["lightrag_graph"]["data"]["entities"][0]["name"] == "Entity1"
    mock_store.get_retriever.assert_called_once()
    mock_retriever._get_relevant_documents.assert_called_once_with("test query")


@patch("backend.services.silo_service.SiloService.check_silo_collection_exists", return_value=True)
@patch("backend.services.silo_service._get_vector_store")
@patch("backend.services.silo_service.SiloService.get_silo")
def test_lightrag_query_mode_empty_docs_returns_empty(mock_get_silo, mock_get_store, _mock_exists):
    """When retriever returns no docs, results and lightrag_graph are empty/None."""
    mock_get_silo.return_value = _make_mock_silo()
    mock_retriever = MagicMock()
    mock_retriever._get_relevant_documents.return_value = []
    mock_store = MagicMock()
    mock_store.get_retriever.return_value = mock_retriever
    mock_get_store.return_value = mock_store

    result = SiloService.search_silo_documents_router(
        silo_id=1,
        query="nothing",
        lightrag_query_mode="local",
        db=MagicMock(),
    )

    assert result is not None
    assert result["total_results"] == 0
    assert result["results"] == []
    assert result["lightrag_graph"] is None


@patch("backend.services.silo_service.SiloService.find_docs_in_collection")
@patch("backend.services.silo_service.SiloService.get_silo")
def test_no_lightrag_query_mode_uses_standard_path(mock_get_silo, mock_find_docs):
    """When lightrag_query_mode is None, falls through to standard find_docs_in_collection."""
    mock_get_silo.return_value = _make_mock_silo()
    mock_doc = Document(page_content="standard chunk", metadata={"_score": 0.9})
    mock_find_docs.return_value = [mock_doc]

    result = SiloService.search_silo_documents_router(
        silo_id=1,
        query="test",
        lightrag_query_mode=None,
        db=MagicMock(),
    )

    assert result is not None
    mock_find_docs.assert_called_once()
    assert "lightrag_graph" not in result
