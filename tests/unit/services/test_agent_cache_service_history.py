"""Unit tests for CheckpointerCacheService.get_conversation_history_async's
reconstruction of the LightRAG graph (sources/subgraph/citation data) from
ToolMessage.artifact — the checkpoint-reconstruction fix for sources/subgraph
disappearing on page refresh.

No real Postgres checkpointer needed: aget_tuple is mocked to return a canned
LangGraph channel_values state built from real LangChain message objects.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from services.agent_cache_service import CheckpointerCacheService


def _lightrag_tool_message(tool_call_id: str, chunk_id: str = "c1") -> ToolMessage:
    doc = Document(
        page_content="fused context",
        metadata={
            "source": "lightrag",
            "lightrag_raw_data": {
                "data": {"chunks": [{"id": chunk_id}], "entities": [{"id": "e1"}], "relationships": []}
            },
        },
    )
    return ToolMessage(content="tool text", tool_call_id=tool_call_id, artifact=[doc])


async def _history_for(messages):
    fake_state = SimpleNamespace(checkpoint={"channel_values": {"messages": messages}})
    fake_checkpointer = AsyncMock()
    fake_checkpointer.aget_tuple = AsyncMock(return_value=fake_state)
    with patch.object(
        CheckpointerCacheService, "get_async_checkpointer", AsyncMock(return_value=fake_checkpointer)
    ):
        return await CheckpointerCacheService.get_conversation_history_async(agent_id=1, session_id="s1")


async def _history_for_router(messages):
    fake_state = SimpleNamespace(checkpoint={"channel_values": {"messages": messages}})
    fake_checkpointer = AsyncMock()
    fake_checkpointer.aget_tuple = AsyncMock(return_value=fake_state)
    with patch.object(
        CheckpointerCacheService, "get_async_checkpointer", AsyncMock(return_value=fake_checkpointer)
    ):
        return await CheckpointerCacheService.get_conversation_history_async(
            agent_id=1, session_id="s1", tool_answers=True
        )


@pytest.mark.asyncio
async def test_final_ai_answer_gets_the_preceding_tool_messages_graph():
    messages = [
        HumanMessage(content="question"),
        AIMessage(content="", tool_calls=[{"name": "lightrag_search", "args": {}, "id": "call_1"}]),
        _lightrag_tool_message("call_1"),
        AIMessage(content="Answer [1](cite://1)"),
    ]

    history = await _history_for(messages)

    assert [h["role"] for h in history] == ["user", "agent"]
    assert history[1]["lightrag_graph"]["data"]["chunks"] == [{"id": "c1"}]


@pytest.mark.asyncio
async def test_multi_tool_call_turn_merges_both_tool_messages_graphs():
    """Matches the live streaming accumulator (agent_streaming_service.py merges
    every _lightrag_graph event of the turn via merge_lightrag_graph instead of
    overwriting): a turn that calls two silos must keep chunks from BOTH, in
    order, not just the last one."""
    messages = [
        HumanMessage(content="question"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "lightrag_search_1", "args": {}, "id": "call_1"},
                {"name": "lightrag_search_2", "args": {}, "id": "call_2"},
            ],
        ),
        _lightrag_tool_message("call_1", chunk_id="c1"),
        _lightrag_tool_message("call_2", chunk_id="c2"),
        AIMessage(content="Answer [1](cite://1)[2](cite://2)"),
    ]

    history = await _history_for(messages)

    assert history[1]["lightrag_graph"]["data"]["chunks"] == [{"id": "c1"}, {"id": "c2"}]


@pytest.mark.asyncio
async def test_followup_turn_without_tools_has_no_stale_graph():
    """A lightrag graph from an earlier turn must not leak onto a later AI
    answer that didn't call any tool itself."""
    messages = [
        HumanMessage(content="question"),
        AIMessage(content="", tool_calls=[{"name": "lightrag_search", "args": {}, "id": "call_1"}]),
        _lightrag_tool_message("call_1"),
        AIMessage(content="Answer [1](cite://1)"),
        HumanMessage(content="thanks, one more thing"),
        AIMessage(content="Sure, happy to help"),
    ]

    history = await _history_for(messages)

    assert [h["role"] for h in history] == ["user", "agent", "user", "agent"]
    assert "lightrag_graph" in history[1]
    assert "lightrag_graph" not in history[3]


@pytest.mark.asyncio
async def test_non_lightrag_tool_calls_do_not_attach_a_graph():
    messages = [
        HumanMessage(content="question"),
        AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {}, "id": "call_1"}]),
        ToolMessage(content="sunny", tool_call_id="call_1"),
        AIMessage(content="It's sunny"),
    ]

    history = await _history_for(messages)

    assert "lightrag_graph" not in history[1]


@pytest.mark.asyncio
async def test_turns_ending_on_tool_message_show_it_as_the_answer():
    """Knowledge-router pass-through (return_direct): the specialist's ToolMessage
    IS the answer — no AI message follows it."""
    messages = [
        HumanMessage(content="q1"),
        AIMessage(content="", tool_calls=[{"name": "ES", "args": {}, "id": "call_1"}]),
        _lightrag_tool_message("call_1"),
        HumanMessage(content="q2"),
        AIMessage(content="", tool_calls=[{"name": "EN", "args": {}, "id": "call_2"}]),
        ToolMessage(content="EN answer", tool_call_id="call_2"),
    ]

    history = await _history_for_router(messages)

    assert [(h["role"], h["content"]) for h in history] == [
        ("user", "q1"), ("agent", "tool text"), ("user", "q2"), ("agent", "EN answer"),
    ]
    assert history[1]["lightrag_graph"]["data"]["chunks"] == [{"id": "c1"}]
    assert "lightrag_graph" not in history[3]


@pytest.mark.asyncio
async def test_classic_agent_never_shows_a_dangling_tool_message_as_an_answer():
    """A turn cut off after a tool ran (disconnect, error) must not surface raw
    tool output as the agent's reply on reload."""
    messages = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[{"name": "lightrag_search", "args": {}, "id": "call_1"}]),
        _lightrag_tool_message("call_1"),
    ]

    history = await _history_for(messages)

    assert [h["role"] for h in history] == ["user"]


@pytest.mark.asyncio
async def test_router_turn_with_two_specialists_in_one_step_joins_both_answers():
    messages = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[
            {"name": "ES", "args": {}, "id": "c1"}, {"name": "EN", "args": {}, "id": "c2"},
        ]),
        ToolMessage(content="ES answer", tool_call_id="c1"),
        ToolMessage(content="EN answer", tool_call_id="c2"),
    ]

    history = await _history_for_router(messages)

    assert history[1]["content"] == "ES answer\n\nEN answer"


@pytest.mark.asyncio
async def test_router_tool_answer_from_an_earlier_step_is_not_carried_over():
    messages = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[{"name": "ES", "args": {}, "id": "c1"}]),
        ToolMessage(content="ES answer", tool_call_id="c1"),
        AIMessage(content="", tool_calls=[{"name": "EN", "args": {}, "id": "c2"}]),
        ToolMessage(content="EN answer", tool_call_id="c2"),
    ]

    history = await _history_for_router(messages)

    assert history[1]["content"] == "EN answer"
