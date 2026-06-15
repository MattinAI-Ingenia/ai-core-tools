"""Tests for LightRAGRetriever — only_need_context is always True."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from langchain_core.documents import Document

pytestmark = pytest.mark.asyncio


def _make_query_result(context_str: str, raw_data: dict):
    result = MagicMock()
    result.content = context_str
    result.raw_data = raw_data
    return result


async def test_retriever_always_uses_only_need_context():
    """QueryParam must always have only_need_context=True and the Document
    must carry the raw_data in metadata."""
    from tools.vector_stores.lightrag_store import LightRAGRetriever

    fake_raw_data = {
        "data": {
            "entities": [{"id": "1", "name": "EntityA"}],
            "relationships": [{"id": "r1", "source": "1", "target": "2"}],
            "chunks": [{"id": "c1", "content": "some text", "file_path": "doc.pdf"}],
            "references": [{"reference_id": "1", "file_path": "doc.pdf"}],
        }
    }
    query_result = _make_query_result("context string", fake_raw_data)

    rag_instance = MagicMock()
    rag_instance.aquery = AsyncMock(return_value=query_result)

    # Capture QueryParam constructor calls
    captured_params = []

    def fake_query_param(**kwargs):
        p = SimpleNamespace(**kwargs)
        captured_params.append(p)
        return p

    with patch.dict("sys.modules", {"lightrag.base": MagicMock(QueryParam=fake_query_param)}):
        retriever = LightRAGRetriever(
            rag_instance=rag_instance,
            query_mode="hybrid",
            top_k=5,
        )
        docs = await retriever._aget_relevant_documents("test query")

    assert len(captured_params) == 1
    assert captured_params[0].only_need_context is True

    assert len(docs) == 1
    assert docs[0].page_content == "context string"
    assert docs[0].metadata["lightrag_raw_data"] == fake_raw_data
