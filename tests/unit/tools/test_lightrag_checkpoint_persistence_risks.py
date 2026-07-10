"""Validates two risks for a "reconstruct lightrag_graph from the LangGraph
checkpoint" persistence design (sources/subgraph/citations survive a page refresh
by reading ToolMessage.artifact back out of history instead of a new DB write).

Risk 1 — SummarizationMiddleware is destructive: once triggered, it replaces the
whole message list with `RemoveMessage(REMOVE_ALL_MESSAGES) + summary + preserved`,
permanently dropping older ToolMessages (and their lightrag artifact) from the
checkpoint. Reconstruction-from-checkpoint only works for turns still inside the
`keep` window — never for turns already summarized away.

Risk 2 — a single AI turn can call multiple tools (multi-silo), producing several
*consecutive* ToolMessages before the next AIMessage. Any reconstruction must walk
back and collect all of them, not just "the ToolMessage right before the AIMessage".
"""
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain.agents.middleware.summarization import SummarizationMiddleware


class _FakeSummaryModel:
    """Minimal stand-in for the summarization LLM — no real API call."""

    _llm_type = "fake"

    async def ainvoke(self, prompt, config=None):
        return AIMessage(content="fake summary")


def _lightrag_tool_message(tool_call_id: str) -> ToolMessage:
    doc = Document(
        page_content="fused context",
        metadata={
            "source": "lightrag",
            "lightrag_raw_data": {"data": {"chunks": [{"id": "c1"}], "entities": [], "relationships": []}},
        },
    )
    return ToolMessage(content="tool text", tool_call_id=tool_call_id, artifact=[doc])


@pytest.mark.asyncio
async def test_summarization_permanently_drops_tool_message_artifacts():
    middleware = SummarizationMiddleware(
        model=_FakeSummaryModel(),
        trigger=("messages", 4),
        keep=("messages", 1),
    )

    old_turn_ai = AIMessage(
        content="", tool_calls=[{"name": "lightrag_search", "args": {}, "id": "call_1"}]
    )
    old_turn_tool = _lightrag_tool_message("call_1")
    old_turn_answer = AIMessage(content="Answer with [1](cite://1)")
    new_human = HumanMessage(content="follow-up question")

    messages = [
        HumanMessage(content="first question"),
        old_turn_ai,
        old_turn_tool,
        old_turn_answer,
        new_human,
    ]

    result = await middleware.abefore_model({"messages": messages}, runtime=None)

    assert result is not None, "expected summarization to trigger with this message count"
    tool_messages_left = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages_left == [], (
        "if this fails, langchain stopped dropping ToolMessages on summarization — "
        "re-evaluate whether checkpoint-reconstruction is viable across the keep window"
    )


def test_multi_tool_call_turn_has_multiple_consecutive_tool_messages():
    """One AI turn calling two silos in parallel leaves two consecutive
    ToolMessages before the next AIMessage — a reconstruction walk must collect
    both, matched by tool_call_id, not just take the single preceding message."""
    ai_with_two_calls = AIMessage(
        content="",
        tool_calls=[
            {"name": "lightrag_search_silo_1", "args": {}, "id": "call_1"},
            {"name": "lightrag_search_silo_2", "args": {}, "id": "call_2"},
        ],
    )
    tool_1 = _lightrag_tool_message("call_1")
    tool_2 = _lightrag_tool_message("call_2")
    final_answer = AIMessage(content="Combined answer [1](cite://1) [2](cite://2)")

    messages = [HumanMessage(content="q"), ai_with_two_calls, tool_1, tool_2, final_answer]

    consecutive_tool_msgs = []
    idx = len(messages) - 2  # message right before the final AIMessage
    while idx >= 0 and isinstance(messages[idx], ToolMessage):
        consecutive_tool_msgs.insert(0, messages[idx])
        idx -= 1

    assert len(consecutive_tool_msgs) == 2
    assert {m.tool_call_id for m in consecutive_tool_msgs} == {"call_1", "call_2"}
