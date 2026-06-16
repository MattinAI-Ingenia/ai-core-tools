"""Unit tests for LightRAG retriever-based search path in SiloService."""
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from langchain_core.documents import Document

from backend.services.silo_service import SiloService


def _make_mock_silo(silo_id=1):
    mock_silo = MagicMock()
    mock_silo.silo_id = silo_id
    mock_silo.embedding_service_id = 1
    mock_silo.embedding_service = MagicMock()
    mock_silo.vector_db_type = "LIGHTRAG"
    return mock_silo


def _make_raw_data(entities=None, chunks=None):
    return {
        "data": {
            "entities": entities or [{"id": "e1", "name": "Entity1"}],
            "relationships": [],
            "chunks": chunks or [{"id": "c1", "content": "chunk text"}],
        }
    }


@pytest.mark.asyncio
@patch("backend.services.silo_service._get_vector_store")
async def test_lightrag_query_mode_returns_chunks_and_graph(mock_get_store):
    """_search_via_lightrag_retriever returns chunks + lightrag_graph from raw data."""
    raw_data = _make_raw_data()
    mock_store = MagicMock()
    mock_store.aretrieve_graph_context = AsyncMock(return_value=raw_data)
    mock_get_store.return_value = mock_store

    result = await SiloService._search_via_lightrag_retriever(
        silo=_make_mock_silo(),
        query="test query",
        lightrag_query_mode="hybrid",
        limit=20,
        filter_metadata=None,
    )

    assert result["total_results"] == 1
    assert result["results"][0]["page_content"] == "chunk text"
    assert result["lightrag_graph"] is not None
    assert result["lightrag_graph"]["data"]["entities"][0]["name"] == "Entity1"
    mock_store.aretrieve_graph_context.assert_awaited_once()


@pytest.mark.asyncio
@patch("backend.services.silo_service._get_vector_store")
async def test_lightrag_query_mode_empty_raw_data_returns_empty(mock_get_store):
    """When aretrieve_graph_context returns empty dict, results and lightrag_graph are empty/None."""
    mock_store = MagicMock()
    mock_store.aretrieve_graph_context = AsyncMock(return_value={})
    mock_get_store.return_value = mock_store

    result = await SiloService._search_via_lightrag_retriever(
        silo=_make_mock_silo(),
        query="nothing",
        lightrag_query_mode="local",
        limit=20,
        filter_metadata=None,
    )

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
