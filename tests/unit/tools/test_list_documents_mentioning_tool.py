"""Pins list_documents_mentioning: the tool must return every matching
document (via SiloService.find_chunks_mentioning, no chunk_top_k) and must
resolve a named product to its resource_id (via resolve_document_by_name)
before searching, when the caller passes `doc`.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.agentTools import _create_coverage_tool


def _fake_silo(silo_id=37):
    return SimpleNamespace(silo_id=silo_id, vector_db_type="LIGHTRAG")


# NOTE: BaseTool.ainvoke() with a plain dict input runs without a tool_call_id,
# and langchain_core's _format_output collapses `(content, artifact)` down to
# just `content` in that case (see langchain_core.tools.base._format_output) —
# so calling through .ainvoke() here would silently drop the artifact rather
# than exercise the tool's actual return value. tests/unit/tools/test_get_retriever_tool.py
# hits the same langchain behavior and works around it the same way: invoke the
# underlying coroutine directly to get the real (content, artifact) tuple.


@pytest.mark.asyncio
async def test_returns_abstention_when_nothing_matches():
    tool = _create_coverage_tool(_fake_silo(), app_id=1)
    with patch("services.silo_service.SiloService.find_chunks_mentioning", return_value=({}, False)):
        content, artifact = await tool.coroutine(term="algo inexistente")
    assert "No se encontró ningún documento" in content
    assert artifact == []


@pytest.mark.asyncio
async def test_returns_one_citation_per_matched_document():
    tool = _create_coverage_tool(_fake_silo(), app_id=1)
    grouped = {
        "271": [("CDOC004043.pdf p.10", "aparece SG Ready aquí")],
        "275": [("CDOC004425.pdf p.5", "también menciona SG Ready")],
    }
    with patch("services.silo_service.SiloService.find_chunks_mentioning", return_value=(grouped, False)):
        content, artifact = await tool.coroutine(term="SG Ready")
    assert "2" in content  # "2 documento(s) mencionan..."
    assert len(artifact) == 1  # one wrapper Document carrying both citations
    chunks = artifact[0].metadata["lightrag_raw_data"]["data"]["chunks"]
    assert len(chunks) == 2
    assert {"CDOC004043.pdf p.10", "CDOC004425.pdf p.5"} == {c["file_path"] for c in chunks}


@pytest.mark.asyncio
async def test_resolves_doc_param_by_commercial_name_before_searching():
    tool = _create_coverage_tool(_fake_silo(), app_id=1)
    with (
        patch("services.silo_service.SiloService.find_chunks_mentioning", return_value=({}, False)) as find_mock,
        patch("services.silo_service.SiloService.resolve_document_by_name",
              return_value=[271]) as resolve_mock,
    ):
        await tool.coroutine(term="P20", doc="TERMAT")
    resolve_mock.assert_called_once()
    assert resolve_mock.call_args.args[0:2] == (1, "TERMAT")
    # the resolved resource_id(s) ([271]), not the raw name "TERMAT", must
    # reach find_chunks_mentioning's doc_filter
    assert find_mock.call_args.args[2] == [271]


@pytest.mark.asyncio
async def test_abstains_when_doc_name_unresolved():
    """An unresolvable `doc` (typo, or a name not in extra_metadata) must not
    silently search the whole corpus — it should abstain, naming the doc."""
    tool = _create_coverage_tool(_fake_silo(), app_id=1)
    with (
        patch("services.silo_service.SiloService.find_chunks_mentioning") as find_mock,
        patch("services.silo_service.SiloService.resolve_document_by_name",
              return_value=[]),
    ):
        content, artifact = await tool.coroutine(term="P20", doc="Modelo Inexistente")
    find_mock.assert_not_called()
    assert "Modelo Inexistente" in content
    assert artifact == []


@pytest.mark.asyncio
async def test_appends_caveat_when_row_cap_hit():
    tool = _create_coverage_tool(_fake_silo(), app_id=1)
    grouped = {"271": [("CDOC004043.pdf p.10", "aparece SG Ready aquí")]}
    with patch("services.silo_service.SiloService.find_chunks_mentioning", return_value=(grouped, True)):
        content, artifact = await tool.coroutine(term="SG Ready")
    assert "más resultados" in content
    assert len(artifact) == 1


@pytest.mark.asyncio
async def test_term_omitted_with_doc_returns_whole_document():
    """G07-style: 'list every parameter in manual Z' has no single literal
    term — omitting term and passing doc must return the whole document,
    not fail or require a term."""
    tool = _create_coverage_tool(_fake_silo(), app_id=1)
    grouped = {"271": [("CDOC004043.pdf p.9", "P00..."), ("CDOC004043.pdf p.15", "P81...")]}
    with (
        patch("services.silo_service.SiloService.find_chunks_mentioning", return_value=(grouped, False)) as find_mock,
        patch("services.silo_service.SiloService.resolve_document_by_name", return_value=[271]),
    ):
        content, artifact = await tool.coroutine(doc="Dual Clima R")
    assert find_mock.call_args.args[1] is None  # term
    assert find_mock.call_args.args[2] == [271]   # resolved doc_filter
    assert len(artifact) == 1
    chunks = artifact[0].metadata["lightrag_raw_data"]["data"]["chunks"]
    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_neither_term_nor_doc_is_a_clear_error():
    tool = _create_coverage_tool(_fake_silo(), app_id=1)
    with patch("services.silo_service.SiloService.find_chunks_mentioning") as find_mock:
        content, artifact = await tool.coroutine()
    find_mock.assert_not_called()
    assert artifact == []


@pytest.mark.asyncio
async def test_content_enumerates_every_matched_document_with_its_own_citation():
    """The model reliably drops entries when left to synthesize its own list
    from raw snippets (seen live: 32 documents found, 4 cited in prose) — the
    tool must build the enumerated, linked list itself, and instruct the model
    to reproduce it whole rather than select from it."""
    tool = _create_coverage_tool(_fake_silo(), app_id=1)
    grouped = {
        "271": [("CDOC004043.pdf p.10", "x")],
        "275": [("CDOC004425.pdf p.5", "y")],
        "280": [("CDOC001961.pdf p.2", "z")],
    }
    with patch("services.silo_service.SiloService.find_chunks_mentioning", return_value=(grouped, False)):
        content, artifact = await tool.coroutine(term="X")
    for label in ("CDOC004043.pdf p.10", "CDOC004425.pdf p.5", "CDOC001961.pdf p.2"):
        assert label in content
    assert "reproduce" in content.lower()
    for n in (1, 2, 3):
        assert f"(cite://{n})" in content


@pytest.mark.asyncio
async def test_returns_graceful_error_on_exception():
    tool = _create_coverage_tool(_fake_silo(), app_id=1)
    with patch(
        "services.silo_service.SiloService.find_chunks_mentioning",
        side_effect=RuntimeError("boom"),
    ):
        content, artifact = await tool.coroutine(term="SG Ready")
    assert artifact == []
    assert isinstance(content, str) and content
