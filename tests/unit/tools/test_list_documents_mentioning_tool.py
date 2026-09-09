"""Pins list_documents_mentioning: the tool must return every matching
document (via SiloService.find_chunks_mentioning, no chunk_top_k) and must
resolve a named product to its resource_id (via resolve_document_by_name)
before searching, when the caller passes `doc`.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.agentTools import _create_coverage_tool, _create_dynamic_lightrag_tool


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
        "271": [("CDOC004043.pdf p.10", "aparece SG Ready aquí", 10)],
        "275": [("CDOC004425.pdf p.5", "también menciona SG Ready", 5)],
    }
    with patch("services.silo_service.SiloService.find_chunks_mentioning", return_value=(grouped, False)):
        content, artifact = await tool.coroutine(term="SG Ready")
    assert "2" in content  # "2 documento(s) mencionan..."
    assert len(artifact) == 1  # one wrapper Document carrying both citations
    chunks = artifact[0].metadata["lightrag_raw_data"]["data"]["chunks"]
    assert len(chunks) == 2
    assert {"CDOC004043.pdf p.10", "CDOC004425.pdf p.5"} == {c["file_path"] for c in chunks}
    # resource_id/page must travel through so the frontend's "Open PDF" button
    # (which no-ops without them) works for coverage-tool citations too.
    by_file = {c["file_path"]: c for c in chunks}
    assert by_file["CDOC004043.pdf p.10"]["resource_id"] == 271
    assert by_file["CDOC004043.pdf p.10"]["page"] == 10


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
    grouped = {"271": [("CDOC004043.pdf p.10", "aparece SG Ready aquí", 10)]}
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
    grouped = {"271": [("CDOC004043.pdf p.9", "P00...", 9), ("CDOC004043.pdf p.15", "P81...", 15)]}
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
        "271": [("CDOC004043.pdf p.10", "x", 10)],
        "275": [("CDOC004425.pdf p.5", "y", 5)],
        "280": [("CDOC001961.pdf p.2", "z", 2)],
    }
    with patch("services.silo_service.SiloService.find_chunks_mentioning", return_value=(grouped, False)):
        content, artifact = await tool.coroutine(term="X")
    for label in ("CDOC004043.pdf p.10", "CDOC004425.pdf p.5", "CDOC001961.pdf p.2"):
        assert label in content
    assert "reproduce" in content.lower()
    for n in (1, 2, 3):
        assert f"(cite://{n})" in content


@pytest.mark.asyncio
async def test_expands_term_with_graph_entity_variants():
    """A literal term like "cenicero" must also search the real on-page
    spelling the graph knows about ("Cenicero Compresor Automatico") — this
    is what actually finds the answer to A20-style questions, since ILIKE
    alone misses phrasing differences."""
    tool = _create_coverage_tool(_fake_silo(), app_id=1)
    with (
        patch("services.silo_service.SiloService.find_chunks_mentioning", return_value=({}, False)) as find_mock,
        patch(
            "services.silo_service.SiloService.resolve_term_variants",
            return_value=["Cenicero Compresor Automatico"],
        ) as variants_mock,
    ):
        await tool.coroutine(term="cenicero")
    variants_mock.assert_called_once_with(37, "cenicero")
    assert find_mock.call_args.args[1] == ["cenicero", "Cenicero Compresor Automatico"]


@pytest.mark.asyncio
async def test_no_graph_variants_searches_the_plain_term():
    """No graph match (or Neo4j unavailable, degrading to []) must not change
    behavior — search with the literal term alone, same as before this."""
    tool = _create_coverage_tool(_fake_silo(), app_id=1)
    with (
        patch("services.silo_service.SiloService.find_chunks_mentioning", return_value=({}, False)) as find_mock,
        patch("services.silo_service.SiloService.resolve_term_variants", return_value=[]),
    ):
        await tool.coroutine(term="RAEE")
    assert find_mock.call_args.args[1] == "RAEE"


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


@pytest.mark.asyncio
async def test_shared_lock_serializes_sibling_lightrag_tool():
    """When _resolve_and_build_retriever_tool shares a lock between this tool
    and _create_dynamic_lightrag_tool, the two must never run concurrently —
    otherwise the shared citation offset (claimed in call order) can diverge
    from the frontend's chunk merge order (arrival order), see the call site
    comment in agentTools.py."""
    events: list[str] = []
    lock = asyncio.Lock()
    offset = [0]
    coverage_tool = _create_coverage_tool(_fake_silo(), app_id=1, offset=offset, lock=lock)
    lightrag_tool = _create_dynamic_lightrag_tool(_fake_silo(), offset=offset, lock=lock)

    def slow_find_chunks_mentioning(*args, **kwargs):
        # Runs inside asyncio.to_thread in the real tool — plain sync sleep.
        events.append("coverage:start")
        time.sleep(0.02)
        events.append("coverage:end")
        return {}, False

    async def slow_ainvoke(*args, **kwargs):
        events.append("lightrag:start")
        await asyncio.sleep(0.01)
        events.append("lightrag:end")
        return []

    with (
        patch("services.silo_service.SiloService.find_chunks_mentioning", side_effect=slow_find_chunks_mentioning),
        patch("services.silo_service.SiloService.resolve_term_variants", return_value=[]),
        patch("services.silo_service.SiloService.get_silo_retriever") as get_retriever_mock,
    ):
        get_retriever_mock.return_value.ainvoke = slow_ainvoke
        await asyncio.gather(
            coverage_tool.coroutine(term="algo"),
            lightrag_tool.coroutine(query="algo", mode="hybrid"),
        )

    # Whichever ran first must fully finish before the other starts.
    assert events in (
        ["coverage:start", "coverage:end", "lightrag:start", "lightrag:end"],
        ["lightrag:start", "lightrag:end", "coverage:start", "coverage:end"],
    )
