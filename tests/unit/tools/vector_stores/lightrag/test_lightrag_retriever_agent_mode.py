"""Tests for LightRAGRetriever — only_need_context is always True."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from langchain_core.documents import Document

pytestmark = pytest.mark.asyncio


async def test_retriever_always_uses_only_need_context():
    """QueryParam must always have only_need_context=True and the Document
    must carry the raw_data in metadata."""
    from tools.vector_stores.lightrag_store import LightRAGRetriever

    # aquery_llm returns a dict: {llm_response: {content: ...}, data: {...}}
    query_result = {
        "llm_response": {"content": "context string"},
        "data": {
            "entities": [{"entity_name": "EntityA", "entity_type": "PERSON", "description": "desc", "file_path": "doc.pdf"}],
            "relationships": [],
            "chunks": [{"chunk_id": "c1", "content": "some text", "file_path": "doc.pdf"}],
            "references": [],
        },
    }

    rag_instance = MagicMock()
    rag_instance.aquery_llm = AsyncMock(return_value=query_result)

    mock_store = MagicMock()
    mock_store._aget_rag_instance = AsyncMock(return_value=rag_instance)

    captured_params = []

    def fake_query_param(**kwargs):
        p = SimpleNamespace(**kwargs)
        captured_params.append(p)
        return p

    with patch.dict("sys.modules", {"lightrag.base": MagicMock(QueryParam=fake_query_param)}):
        retriever = LightRAGRetriever(
            store=mock_store,
            collection_name="silo_1",
            query_mode="hybrid",
            top_k=5,
        )
        docs = await retriever._aget_relevant_documents("test query")

    assert len(captured_params) == 1
    assert captured_params[0].only_need_context is True

    assert len(docs) == 1
    assert docs[0].page_content == "context string"
    assert "lightrag_raw_data" in docs[0].metadata
    assert "data" in docs[0].metadata["lightrag_raw_data"]
