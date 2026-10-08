"""Knowledge router: specialists' citations renumbered onto one turn-wide
counter so cite://N keeps resolving to chunks[N-1] of the merged payload.
See docs/superpowers/specs/2026-10-07-knowledge-router-design.md."""
from tools.knowledge_router import chunk_count, graph_artifact, rebase_citations, tag_graph_with_silo


def test_rebase_citations_shifts_number_and_link():
    assert rebase_citations("a [1](cite://1) b [2](cite://2)", 3) == "a [4](cite://4) b [5](cite://5)"


def test_rebase_citations_normalizes_cjk_brackets():
    assert rebase_citations("a 【2】(cite://2)", 1) == "a [3](cite://3)"


def test_rebase_citations_zero_delta_is_identity():
    text = "a [1](cite://1) and [7] plain"
    assert rebase_citations(text, 0) == text


def test_rebase_citations_leaves_bare_numbers_alone():
    assert rebase_citations("step [1] then [1](cite://1)", 2) == "step [1] then [3](cite://3)"


def test_tag_graph_with_silo_stamps_every_element():
    graph = {"data": {"entities": [{"id": "E"}], "relationships": [{"id": "r"}], "chunks": [{"id": "c"}]}}
    tagged = tag_graph_with_silo(graph, 7, "Manuales ES")
    for key in ("entities", "relationships", "chunks"):
        assert tagged["data"][key][0]["silo_id"] == 7
        assert tagged["data"][key][0]["silo_name"] == "Manuales ES"
    assert "silo_id" not in graph["data"]["chunks"][0]  # input not mutated


def test_graph_artifact_shape_and_none():
    assert graph_artifact(None) is None
    graph = {"data": {"chunks": [{"id": "c"}]}}
    assert graph_artifact(graph)[0].metadata["lightrag_raw_data"] == graph


def test_chunk_count():
    assert chunk_count(None) == 0
    assert chunk_count({"data": {"chunks": [{"id": "a"}, {"id": "b"}]}}) == 2


import asyncio
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage

from models.agent import Agent
from tools import agentTools


class _CitingReactAgent:
    """Fake sub-agent: one LightRAG retrieval of *chunk_ids*, then an answer citing
    each chunk with the sub-agent's OWN counter, which keeps running across calls
    in the same turn (its tools are built once per turn, like the real ones)."""

    def __init__(self, chunk_ids, delay=0.0, stray_cite=None):
        self.chunk_ids = list(chunk_ids)
        self.delay = delay
        self.stray_cite = stray_cite  # a marker no chunk of this run backs (e.g. from a nested agent)
        self.cell = [0]  # replaced by IACTTool.run_counter, which the tool resets on every run

    def _turn(self):
        raw = {"data": {
            "entities": [{"id": "Quemador", "source_id": self.chunk_ids[0]}],
            "relationships": [],
            "chunks": [{"id": c, "content": c} for c in self.chunk_ids],
        }}
        tool_msg = ToolMessage(
            content="context", name="retrieve", tool_call_id="call-r",
            artifact=[Document(page_content="context", metadata={"lightrag_raw_data": raw})],
        )
        first = self.cell[0] + 1
        self.cell[0] += len(self.chunk_ids)
        cites = "".join(f"[{n}](cite://{n})" for n in range(first, self.cell[0] + 1))
        if self.stray_cite:
            cites += f"[{self.stray_cite}](cite://{self.stray_cite})"
        return tool_msg, AIMessage(content=f"Answer {cites}")

    async def ainvoke(self, payload):
        await asyncio.sleep(self.delay)
        tool_msg, final = self._turn()
        return {"messages": [*payload["messages"], tool_msg, final]}

    async def astream(self, _payload, stream_mode=None):
        await asyncio.sleep(self.delay)
        tool_msg, final = self._turn()
        yield ("updates", {"tools": {"messages": [tool_msg]}})
        yield ("updates", {"model": {"messages": [final]}})


def _router_tool(name, react_agent, offset, lock, silo_id=7, silo_name="Manuales ES"):
    agent = Agent(name=name, description=f"{name} manuals", system_prompt="")
    agent.agent_id = silo_id
    agent.silo_id = silo_id
    with patch("tools.agentTools.get_llm", return_value=MagicMock()):
        tool = agentTools.IACTTool(agent)
    tool.silo_name = silo_name
    tool.react_agent = react_agent
    if hasattr(react_agent, "cell"):
        react_agent.cell = tool.run_counter
    tool.enable_router_mode(offset, lock)
    return tool


@pytest.fixture
def no_writer():
    with patch.object(agentTools.IACTTool, "_get_stream_writer_or_none", return_value=None):
        yield


def _chunks(artifact):
    return artifact[0].metadata["lightrag_raw_data"]["data"]["chunks"]


@pytest.mark.asyncio
async def test_router_mode_returns_tagged_artifact(no_writer):
    offset, lock = [0], asyncio.Lock()
    tool = _router_tool("ES", _CitingReactAgent(["a", "b"]), offset, lock)

    text, artifact = await tool._arun("q")

    assert text == "Answer [1](cite://1)[2](cite://2)"
    assert [c["id"] for c in _chunks(artifact)] == ["a", "b"]
    assert {(c["silo_id"], c["silo_name"]) for c in _chunks(artifact)} == {(7, "Manuales ES")}
    assert offset == [2]
    assert tool.return_direct is True


@pytest.mark.asyncio
async def test_router_mode_streaming_keeps_graph_instead_of_forwarding_it():
    emitted = []
    tool = _router_tool("ES", _CitingReactAgent(["a", "b"]), [0], asyncio.Lock())

    with patch("langgraph.config.get_stream_writer", return_value=emitted.append):
        text, artifact = await tool._arun("q")

    assert text == "Answer [1](cite://1)[2](cite://2)"
    assert [c["id"] for c in _chunks(artifact)] == ["a", "b"]
    assert all(e.get("type") != "_lightrag_graph" for e in emitted)


@pytest.mark.asyncio
async def test_router_mode_rebases_onto_turn_offset(no_writer):
    offset = [3]
    tool = _router_tool("ES", _CitingReactAgent(["a", "b"]), offset, asyncio.Lock())

    text, _ = await tool._arun("q")

    assert text == "Answer [4](cite://4)[5](cite://5)"
    assert offset == [5]


@pytest.mark.asyncio
async def test_same_specialist_twice_in_a_turn_keeps_numbering(no_writer):
    offset, lock = [0], asyncio.Lock()
    es = _router_tool("ES", _CitingReactAgent(["a", "b"]), offset, lock, silo_id=7)
    en = _router_tool("EN", _CitingReactAgent(["c"]), offset, lock, silo_id=8, silo_name="Manuals EN")

    t1, _ = await es._arun("q1")  # ES local 1,2 -> 1,2
    t2, _ = await en._arun("q2")  # EN local 1   -> 3
    t3, _ = await es._arun("q3")  # ES local 3,4 (its counter kept running) -> 4,5

    assert (t1, t2, t3) == (
        "Answer [1](cite://1)[2](cite://2)",
        "Answer [3](cite://3)",
        "Answer [4](cite://4)[5](cite://5)",
    )
    assert offset == [5]


@pytest.mark.asyncio
async def test_classic_mode_still_returns_plain_text(no_writer):
    agent = Agent(name="Classic", description="", system_prompt="")
    with patch("tools.agentTools.get_llm", return_value=MagicMock()):
        tool = agentTools.IACTTool(agent)
    tool.react_agent = _CitingReactAgent(["a"])

    assert await tool._arun("q") == "Answer [1](cite://1)"
    assert tool.return_direct is False


from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from tools.knowledge_router import create_consultar_varios_tool


class _BrokenReactAgent:
    async def ainvoke(self, _payload):
        raise RuntimeError("down")


class _PlainReactAgent:
    """Specialist without LightRAG: answers, no artifact."""

    async def ainvoke(self, payload):
        return {"messages": [*payload["messages"], AIMessage(content="plain answer")]}


def _es_en(es_agent, en_agent):
    offset, lock = [0], asyncio.Lock()
    es = _router_tool("ES", es_agent, offset, lock, silo_id=7)
    en = _router_tool("EN", en_agent, offset, lock, silo_id=8, silo_name="Manuals EN")
    return create_consultar_varios_tool([es, en], lock), offset


@pytest.mark.asyncio
async def test_consultar_varios_numbers_in_list_order_not_finish_order(no_writer):
    # ES finishes last but is first in the list: its chunks must still be 1,2.
    tool, offset = _es_en(_CitingReactAgent(["a", "b"], delay=0.05), _CitingReactAgent(["c"]))

    text, artifact = await tool.coroutine(especialistas=["ES", "EN"], pregunta="q")

    assert text == "### ES\nAnswer [1](cite://1)[2](cite://2)\n\n### EN\nAnswer [3](cite://3)"
    assert [(c["id"], c["silo_id"]) for c in _chunks(artifact)] == [("a", 7), ("b", 7), ("c", 8)]
    entities = artifact[0].metadata["lightrag_raw_data"]["data"]["entities"]
    assert sorted((e["id"], e["silo_id"]) for e in entities) == [("Quemador", 7), ("Quemador", 8)]
    assert offset == [3]


@pytest.mark.asyncio
async def test_consultar_varios_survives_one_specialist_failing(no_writer):
    tool, offset = _es_en(_BrokenReactAgent(), _CitingReactAgent(["c"]))

    text, artifact = await tool.coroutine(especialistas=["ES", "EN"], pregunta="q")

    assert "### ES\n(error: down)" in text
    assert "### EN\nAnswer [1](cite://1)" in text
    assert [c["id"] for c in _chunks(artifact)] == ["c"]
    assert offset == [1]


@pytest.mark.asyncio
async def test_consultar_varios_ignores_unknown_and_duplicate_names(no_writer):
    tool, _ = _es_en(_CitingReactAgent(["a"]), _CitingReactAgent(["c"]))

    text, _ = await tool.coroutine(especialistas=["EN", "EN", "Nope"], pregunta="q")

    assert text == "### EN\nAnswer [1](cite://1)\n\n(ignored unknown specialists: Nope. Valid: ES, EN)"


@pytest.mark.asyncio
async def test_consultar_varios_matches_names_ignoring_case(no_writer):
    tool, _ = _es_en(_CitingReactAgent(["a"]), _CitingReactAgent(["c"]))

    text, _ = await tool.coroutine(especialistas=["en", "Es"], pregunta="q")

    assert text == "### EN\nAnswer [1](cite://1)\n\n### ES\nAnswer [2](cite://2)"


@pytest.mark.asyncio
async def test_consultar_varios_with_no_valid_names_lists_valid_ones(no_writer):
    tool, _ = _es_en(_CitingReactAgent(["a"]), _CitingReactAgent(["c"]))

    text, artifact = await tool.coroutine(especialistas=["Nope"], pregunta="q")

    assert artifact is None
    assert "ES" in text and "EN" in text


@pytest.mark.asyncio
async def test_consultar_varios_mixes_plain_and_lightrag_specialists(no_writer):
    tool, offset = _es_en(_PlainReactAgent(), _CitingReactAgent(["c"]))

    text, artifact = await tool.coroutine(especialistas=["ES", "EN"], pregunta="q")

    assert text == "### ES\nplain answer\n\n### EN\nAnswer [1](cite://1)"
    assert [c["id"] for c in _chunks(artifact)] == ["c"]
    assert offset == [1]


def _router_agent(is_router: bool, silo_id):
    return SimpleNamespace(
        agent_id=1, type="agent", ai_service=None, enable_code_interpreter=False,
        has_memory=False, memory_max_tokens=None, memory_max_messages=None,
        memory_summarize_threshold=None, output_parser_id=None, server_tools=[],
        tool_associations=[SimpleNamespace(tool=SimpleNamespace(type="agent"))],
        silo_id=silo_id, silo=None, skill_associations=[], mcp_associations=[],
        system_prompt="You route.", is_knowledge_router=is_router,
    )


async def _build(agent, specialist):
    with (
        patch("tools.agentTools.get_llm", return_value=MagicMock()),
        patch("tools.agentTools.get_output_parser", return_value=None),
        patch("tools.agentTools.create_langchain_agent", return_value=MagicMock()) as build,
        patch("tools.agentTools.MCPClientManager.get_client", new=AsyncMock(return_value=None)),
        patch("tools.agentTools.IACTTool.create", new=AsyncMock(return_value=specialist)),
        patch("tools.agentTools._resolve_and_build_retriever_tool", return_value=None) as own_silo,
    ):
        await agentTools.create_agent(agent)
    return build.call_args.kwargs, own_silo


def _specialist():
    specialist = MagicMock(spec=agentTools.IACTTool)
    specialist.name, specialist.description = "ES", "Spanish manuals"
    specialist.metadata = {}  # create_agent tags every tool with its metrics type
    specialist.agent = MagicMock(agent_id=1)
    return specialist


@pytest.mark.asyncio
async def test_create_agent_wires_knowledge_router():
    specialist = _specialist()

    kwargs, own_silo = await _build(_router_agent(True, silo_id=5), specialist)

    specialist.enable_router_mode.assert_called_once()
    assert "consultar_varios" in [getattr(t, "name", None) for t in kwargs["tools"]]
    assert "<knowledge_router>" in kwargs["system_prompt"]
    own_silo.assert_not_called()


@pytest.mark.asyncio
async def test_create_agent_without_flag_leaves_agent_tools_alone():
    specialist = _specialist()

    kwargs, _ = await _build(_router_agent(False, silo_id=None), specialist)

    specialist.enable_router_mode.assert_not_called()
    assert "consultar_varios" not in [getattr(t, "name", None) for t in kwargs["tools"]]
    assert "<knowledge_router>" not in kwargs["system_prompt"]


@pytest.mark.asyncio
async def test_router_mode_failure_shows_no_exception_details(no_writer):
    tool = _router_tool("ES", _BrokenReactAgent(), [0], asyncio.Lock())

    text, artifact = await tool._arun("q")

    assert "down" not in text and "RuntimeError" not in text
    assert artifact is None


# --- Through the real LangChain graph (create_agent + ToolNode) ---------------
# Pins what the router relies on from langchain: a return_direct tool whose
# response_format is set AFTER construction still ends the turn, and its
# ToolMessage keeps both the text and the artifact.

from langchain.agents import create_agent as _lc_create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import HumanMessage

from tools.knowledge_router import create_consultar_varios_tool as _make_consultar


class _ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


async def _run_graph(tools, *script):
    model = _ScriptedModel(messages=iter(script))
    result = await _lc_create_agent(model, tools=tools).ainvoke({"messages": [HumanMessage(content="q")]})
    return result["messages"]


@pytest.mark.asyncio
async def test_single_specialist_ends_the_turn_with_text_and_artifact(no_writer):
    es = _router_tool("ES", _CitingReactAgent(["a", "b"]), [0], asyncio.Lock())

    msgs = await _run_graph([es], _call("ES", {"query": "q"}, "c1"), AIMessage(content="MODEL MUST NOT RUN"))

    assert isinstance(msgs[-1], ToolMessage)  # no model call after the specialist
    assert msgs[-1].content == "Answer [1](cite://1)[2](cite://2)"
    assert [c["id"] for c in _chunks(msgs[-1].artifact)] == ["a", "b"]


@pytest.mark.asyncio
async def test_consultar_varios_hands_back_to_the_model_with_merged_artifact(no_writer):
    offset, lock = [0], asyncio.Lock()
    es = _router_tool("ES", _CitingReactAgent(["a"]), offset, lock, silo_id=7)
    en = _router_tool("EN", _CitingReactAgent(["c"]), offset, lock, silo_id=8, silo_name="Manuals EN")
    varios = _make_consultar([es, en], lock)

    msgs = await _run_graph(
        [es, en, varios],
        _call("consultar_varios", {"especialistas": ["ES", "EN"], "pregunta": "q"}, "c1"),
        AIMessage(content="Final [1](cite://1)[2](cite://2)"),
    )

    tool_msg = next(m for m in msgs if isinstance(m, ToolMessage))
    assert [(c["id"], c["silo_id"]) for c in _chunks(tool_msg.artifact)] == [("a", 7), ("c", 8)]
    assert msgs[-1].content == "Final [1](cite://1)[2](cite://2)"  # the model wrote the answer


# --- Per-run counter, out-of-range pruning, empty answers ---------------------


class _FailsOnceAfterRetrieving(_CitingReactAgent):
    """First run: the retriever already advanced the sub-agent's counter, then the run dies."""

    def __init__(self, chunk_ids):
        super().__init__(chunk_ids)
        self.failed = False

    async def ainvoke(self, payload):
        if not self.failed:
            self.failed = True
            self.cell[0] += len(self.chunk_ids)
            raise RuntimeError("provider down")
        return await super().ainvoke(payload)


@pytest.mark.asyncio
async def test_retry_after_a_mid_run_failure_keeps_numbering(no_writer):
    offset = [0]
    tool = _router_tool("ES", _FailsOnceAfterRetrieving(["a", "b"]), offset, asyncio.Lock())

    failed, artifact = await tool._arun("q1")
    retried, retried_artifact = await tool._arun("q2")

    assert artifact is None and offset == [2]
    assert retried == "Answer [1](cite://1)[2](cite://2)"
    assert [c["id"] for c in _chunks(retried_artifact)] == ["a", "b"]


@pytest.mark.asyncio
async def test_markers_no_chunk_of_this_run_backs_are_dropped(no_writer):
    # A nested agent-tool's classic answer (no artifact) cited [1]; after rebasing
    # onto the turn it would open ANOTHER specialist's chunk. It must not link.
    offset = [3]
    tool = _router_tool("ES", _CitingReactAgent(["a"], stray_cite=9), offset, asyncio.Lock())

    text, _ = await tool._arun("q")

    assert text == "Answer [4](cite://4)"


def test_rebase_citations_drops_markers_outside_the_valid_range():
    text = "x [1](cite://1) y [3](cite://3) z [0](cite://0)"
    assert rebase_citations(text, 10, valid=2) == "x [11](cite://11) y  z "
    assert rebase_citations(text, 0, valid=2) == "x [1](cite://1) y  z "


class _EmptyFinalReactAgent:
    """Final AI message comes back empty after a retrieval ran."""

    async def ainvoke(self, payload):
        tool_msg = ToolMessage(content="RAW CONTEXT [1] (source: x)", name="retrieve", tool_call_id="c")
        return {"messages": [*payload["messages"], tool_msg, AIMessage(content="")]}


@pytest.mark.asyncio
async def test_empty_specialist_answer_is_a_notice_not_the_raw_retrieval(no_writer):
    tool = _router_tool("ES", _EmptyFinalReactAgent(), [0], asyncio.Lock())

    text, _ = await tool._arun("q")

    assert text == "The specialist returned no answer."
    assert "RAW CONTEXT" not in text


def test_routing_instruction_does_not_infer_the_specialist_from_the_question_language():
    from tools.knowledge_router import KNOWLEDGE_ROUTER_INSTRUCTION as text

    # Questions can arrive in one language while the knowledge is split by another:
    # the language of the question says nothing, only an explicit restriction does.
    assert "language the question is written in" in text
    assert "explicitly" in text


# --- Per-specialist tool-call ceiling (latency) -------------------------------


@pytest.mark.asyncio
async def test_specialist_react_agent_gets_a_tool_call_limit():
    from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware

    agent = Agent(name="ES", description="d", system_prompt="")
    with (
        patch("tools.agentTools.get_llm", return_value=MagicMock()),
        patch("tools.agentTools.create_langchain_agent", return_value=MagicMock()) as build,
        patch.object(agentTools.MCPClientManager, "get_client", new=AsyncMock(return_value=None)),
    ):
        await agentTools.IACTTool.create(agent, max_tool_calls=3)
        limited = build.call_args.kwargs.get("middleware") or []
        await agentTools.IACTTool.create(agent)
        unlimited = build.call_args.kwargs.get("middleware") or []

    assert [type(m) for m in limited] == [ToolCallLimitMiddleware]
    assert limited[0].run_limit == 3  # create() honours whatever the router passes
    assert unlimited == []


async def _specialist_limit_for(agent):
    created = AsyncMock(return_value=_specialist())
    with (
        patch("tools.agentTools.get_llm", return_value=MagicMock()),
        patch("tools.agentTools.get_output_parser", return_value=None),
        patch("tools.agentTools.create_langchain_agent", return_value=MagicMock()),
        patch("tools.agentTools.MCPClientManager.get_client", new=AsyncMock(return_value=None)),
        patch("tools.agentTools.IACTTool.create", new=created),
        patch("tools.agentTools._resolve_and_build_retriever_tool", return_value=None),
    ):
        await agentTools.create_agent(agent)
    return created.call_args.kwargs["max_tool_calls"]


@pytest.mark.asyncio
async def test_router_limits_each_specialist_to_six_tool_calls_and_classic_does_not():
    assert await _specialist_limit_for(_router_agent(True, silo_id=None)) == 6
    assert await _specialist_limit_for(_router_agent(False, silo_id=None)) is None


@pytest.mark.asyncio
async def test_sub_agent_never_fans_out_parallel_tool_calls():
    """Retrieval and coverage tools number citations off one shared counter; running
    them in the same step lets the order they claim numbers differ from the order
    their chunks land in the payload (cite://N then opens another document). The
    top-level agent pre-binds parallel_tool_calls=False for this; sub-agents must too."""
    llm = MagicMock()
    agent = Agent(name="ES", description="d", system_prompt="")
    with (
        patch("tools.agentTools.get_llm", return_value=llm),
        patch("tools.agentTools.create_langchain_agent", return_value=MagicMock()) as build,
        patch.object(agentTools.MCPClientManager, "get_client", new=AsyncMock(return_value=None)),
    ):
        await agentTools.IACTTool.create(agent)

    assert llm.bind_tools.call_args.kwargs == {"parallel_tool_calls": False}
    assert build.call_args.kwargs["model"] is llm.bind_tools.return_value


@pytest.mark.asyncio
async def test_sub_agent_falls_back_when_provider_rejects_parallel_tool_calls():
    llm = MagicMock()
    llm.bind_tools.side_effect = [TypeError("unexpected keyword"), "bound-plain"]
    agent = Agent(name="ES", description="d", system_prompt="")
    with (
        patch("tools.agentTools.get_llm", return_value=llm),
        patch("tools.agentTools.create_langchain_agent", return_value=MagicMock()) as build,
        patch.object(agentTools.MCPClientManager, "get_client", new=AsyncMock(return_value=None)),
    ):
        await agentTools.IACTTool.create(agent)

    assert build.call_args.kwargs["model"] == "bound-plain"


def test_routing_instruction_requires_every_specialists_findings_in_the_final_answer():
    from tools.knowledge_router import KNOWLEDGE_ROUTER_INSTRUCTION as text

    # Measured 2026-10-07: with several specialists the router wrote its final answer
    # from only one of them although the other had returned its documents.
    assert "EVERY specialist" in text
    assert "one section per specialist" in text
    assert "Never drop" in text
