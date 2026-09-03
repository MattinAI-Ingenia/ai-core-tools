"""Regression: when both skill-routed tools (retrieve_from_knowledge_base and
list_documents_mentioning) fire in the same turn, citation numbers must not
collide — each tool factory used to allocate its own independent counter, so
both tools numbered their sources starting at [1], and the frontend (which
resolves cite://N against a globally-merged chunk list) would show the wrong
source for the second tool's citations.

_resolve_and_build_retriever_tool now hoists ONE offset list and passes it
into both factories. This test exercises that shared-offset contract
directly, without going through _resolve_and_build_retriever_tool.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document

from tools.agentTools import _create_coverage_tool, _create_dynamic_lightrag_tool


def _fake_silo(silo_id=37):
    return SimpleNamespace(silo_id=silo_id, vector_db_type="LIGHTRAG")


@pytest.mark.asyncio
async def test_second_tool_continues_numbering_from_shared_offset():
    offset = [0]
    lightrag_tool = _create_dynamic_lightrag_tool(_fake_silo(), offset=offset)
    coverage_tool = _create_coverage_tool(_fake_silo(), app_id=1, offset=offset)

    retrieved_doc = Document(
        page_content="chunk content",
        metadata={"lightrag_raw_data": {"data": {"chunks": [
            {"file_path": "a.pdf p.1", "content": "primero"},
            {"file_path": "a.pdf p.2", "content": "segundo"},
        ]}}},
    )
    fake_retriever = SimpleNamespace(ainvoke=AsyncMock(return_value=[retrieved_doc]))

    with patch("services.silo_service.SiloService.get_silo_retriever", return_value=fake_retriever):
        first_content, _ = await lightrag_tool.coroutine(query="q", mode="hybrid")
    assert "[1] (source: a.pdf p.1)" in first_content
    assert "[2] (source: a.pdf p.2)" in first_content

    grouped = {"99": [("b.pdf p.5", "tercero")]}
    with patch("services.silo_service.SiloService.find_chunks_mentioning", return_value=(grouped, False)):
        second_content, _ = await coverage_tool.coroutine(term="tercero")
    # Numbering continues from where the first tool left off — [3], not [1].
    assert "[3] (source: b.pdf p.5)" in second_content
    assert "[1] (source:" not in second_content
    assert "[2] (source:" not in second_content
