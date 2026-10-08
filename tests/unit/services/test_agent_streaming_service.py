from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


async def _empty_stream():
    if False:
        yield None


async def _missing_tool_output_stream():
    raise RuntimeError(
        "Error code: 400 - No tool output found for function call call_stale"
    )
    if False:
        yield None


@pytest.mark.asyncio
async def test_streaming_agent_passes_sandbox_session_key_to_tool_builder():
    from services.agent_streaming_service import AgentStreamingService

    ctx = SimpleNamespace(
        effective_conv_id=297,
        conversation=None,
        agent=SimpleNamespace(name="Agent", has_memory=True),
        fresh_agent=SimpleNamespace(agent_id=1, has_memory=True),
        search_params={},
        session_id_for_cache="297",
        user_context={"user_id": "u1"},
        working_dir="/tmp/work",
        sandbox_handle=MagicMock(),
        sandbox_provider=MagicMock(),
        sandbox_session_key="conv_1_297",
        enhanced_message="hello",
        image_files=[],
        processed_files=[],
    )

    execution_service = MagicMock()
    execution_service._prepare_turn = AsyncMock(return_value=ctx)
    execution_service._finalize_turn = AsyncMock(
        return_value={
            "parsed_response": "done",
            "effective_conv_id": 297,
            "files_data": [],
        }
    )

    agent_chain = MagicMock()
    agent_chain.astream.return_value = _empty_stream()
    create_agent = AsyncMock(return_value=(agent_chain, None, None, None))

    service = AgentStreamingService()
    service.execution_service = execution_service
    db = MagicMock()

    with (
        patch("services.agent_streaming_service.create_agent", create_agent),
        patch("services.agent_streaming_service.prepare_agent_config", return_value={"configurable": {}}),
        patch("services.agent_streaming_service.build_human_message", return_value=SimpleNamespace(content="hello")),
    ):
        events = [
            event
            async for event in service.stream_agent_chat(
                agent_id=1,
                message="hello",
                file_references=[],
                user_context={"user_id": "u1"},
                conversation_id=297,
                db=db,
            )
        ]

    assert events
    assert create_agent.call_args.kwargs["sandbox_session_key"] == "conv_1_297"
    execution_service._begin_sandbox_turn.assert_called_once_with(
        ctx,
        db=db,
    )
    execution_service._end_sandbox_turn.assert_called_once_with(
        ctx,
        db=db,
    )


@pytest.mark.asyncio
async def test_streaming_resets_stale_tool_call_checkpoint_and_retries():
    from services.agent_streaming_service import AgentStreamingService

    ctx = SimpleNamespace(
        effective_conv_id=297,
        conversation=None,
        agent=SimpleNamespace(name="Agent", has_memory=True),
        fresh_agent=SimpleNamespace(agent_id=1, has_memory=True),
        search_params={},
        session_id_for_cache="297",
        user_context={"user_id": "u1"},
        working_dir="/tmp/work",
        sandbox_handle=MagicMock(),
        sandbox_provider=MagicMock(),
        sandbox_session_key="conv_1_297",
        enhanced_message="hello",
        image_files=[],
        processed_files=[],
    )

    execution_service = MagicMock()
    execution_service._prepare_turn = AsyncMock(return_value=ctx)
    execution_service._finalize_turn = AsyncMock(
        return_value={
            "parsed_response": "done",
            "effective_conv_id": 297,
            "files_data": [],
        }
    )

    first_chain = MagicMock()
    first_chain.astream.return_value = _missing_tool_output_stream()
    second_chain = MagicMock()
    second_chain.astream.return_value = _empty_stream()
    create_agent = AsyncMock(
        side_effect=[
            (first_chain, None, None, None),
            (second_chain, None, None, None),
        ]
    )

    service = AgentStreamingService()
    service.execution_service = execution_service

    with (
        patch("services.agent_streaming_service.create_agent", create_agent),
        patch("services.agent_streaming_service.prepare_agent_config", return_value={"configurable": {}}),
        patch("services.agent_streaming_service.build_human_message", return_value=SimpleNamespace(content="hello")),
        patch(
            "services.agent_streaming_service.CheckpointerCacheService.get_rollback_checkpoint_id",
            new=AsyncMock(return_value="01ARZ3NDEKTSV4RRFFQ69G5FAV"),
        ) as get_rollback_checkpoint_id,
    ):
        events = [
            event
            async for event in service.stream_agent_chat(
                agent_id=1,
                message="hello",
                file_references=[],
                user_context={"user_id": "u1"},
                conversation_id=297,
                db=MagicMock(),
            )
        ]

    assert len(create_agent.call_args_list) == 2
    get_rollback_checkpoint_id.assert_awaited_once_with(1, "297")
    # The retry must fork from the prior checkpoint, not delete anything.
    assert second_chain.astream.call_args.kwargs["config"]["configurable"]["checkpoint_id"] == (
        "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    )
    assert any('"done"' in event for event in events)


@pytest.mark.asyncio
async def test_streaming_fails_cleanly_when_no_checkpoint_to_roll_back_to():
    """When the broken checkpoint is the thread's first ever state, there is
    nothing to fork from — the turn must fail with a clean error instead of
    retrying (and must not fall back to deleting the thread)."""
    from services.agent_streaming_service import AgentStreamingService

    ctx = SimpleNamespace(
        effective_conv_id=297,
        conversation=None,
        agent=SimpleNamespace(name="Agent", has_memory=True),
        fresh_agent=SimpleNamespace(agent_id=1, has_memory=True),
        search_params={},
        session_id_for_cache="297",
        user_context={"user_id": "u1"},
        working_dir="/tmp/work",
        sandbox_handle=MagicMock(),
        sandbox_provider=MagicMock(),
        sandbox_session_key="conv_1_297",
        enhanced_message="hello",
        image_files=[],
        processed_files=[],
    )

    execution_service = MagicMock()
    execution_service._prepare_turn = AsyncMock(return_value=ctx)
    execution_service._finalize_turn = AsyncMock()

    first_chain = MagicMock()
    first_chain.astream.return_value = _missing_tool_output_stream()
    create_agent = AsyncMock(return_value=(first_chain, None, None, None))

    service = AgentStreamingService()
    service.execution_service = execution_service

    with (
        patch("services.agent_streaming_service.create_agent", create_agent),
        patch("services.agent_streaming_service.prepare_agent_config", return_value={"configurable": {}}),
        patch("services.agent_streaming_service.build_human_message", return_value=SimpleNamespace(content="hello")),
        patch(
            "services.agent_streaming_service.CheckpointerCacheService.get_rollback_checkpoint_id",
            new=AsyncMock(return_value=None),
        ),
    ):
        events = [
            event
            async for event in service.stream_agent_chat(
                agent_id=1,
                message="hello",
                file_references=[],
                user_context={"user_id": "u1"},
                conversation_id=297,
                db=MagicMock(),
            )
        ]

    assert len(create_agent.call_args_list) == 1
    assert any('"error"' in event and "resend" in event for event in events)
    execution_service._finalize_turn.assert_not_awaited()


async def _resumed_specialist_stream():
    from langchain_core.documents import Document
    from langchain_core.messages import ToolMessage

    graph = {"data": {"chunks": [{"id": "c1", "silo_id": 7}], "entities": [], "relationships": []}}
    yield (
        "updates",
        {"tools": {"messages": [ToolMessage(
            content="ES answer [1](cite://1)", tool_call_id="call_1", id="tm-1",
            artifact=[Document(page_content="", metadata={"lightrag_raw_data": graph})],
        )]}},
    )


@pytest.mark.asyncio
async def test_resumed_turn_ending_on_a_specialist_keeps_answer_and_graph():
    """HITL resume of a knowledge-router turn: the specialist's ToolMessage is the
    answer (no model tokens) and its graph must reach the client like on a normal
    turn, not leak as a raw internal event."""
    import json

    from services.agent_streaming_service import AgentStreamingService

    ctx = SimpleNamespace(
        effective_conv_id=5, conversation=None,
        agent=SimpleNamespace(name="Router", has_memory=True),
        fresh_agent=SimpleNamespace(agent_id=1, has_memory=True),
        search_params={}, session_id_for_cache="5", user_context={"user_id": "u1"}, working_dir=None,
    )
    execution_service = MagicMock()
    execution_service._prepare_turn = AsyncMock(return_value=ctx)
    execution_service._finalize_turn = AsyncMock(
        side_effect=lambda _ctx, raw, _db: {"parsed_response": raw, "effective_conv_id": 5, "files_data": []}
    )
    agent_chain = MagicMock()
    agent_chain.astream.return_value = _resumed_specialist_stream()
    agent_chain.aget_state = AsyncMock(return_value=SimpleNamespace(tasks=[]))

    service = AgentStreamingService()
    service.execution_service = execution_service

    with (
        patch("services.agent_streaming_service.create_agent", AsyncMock(return_value=(agent_chain, None, None, None))),
        patch("services.agent_streaming_service.prepare_agent_config", return_value={"configurable": {}}),
    ):
        events = [e async for e in service.stream_resume_agent_chat(
            agent_id=1, decisions=[{"type": "approve"}], user_context={"user_id": "u1"},
            conversation_id=5, db=MagicMock(),
        )]

    parsed = [json.loads(e.split("data: ", 1)[1]) for e in events]
    assert "_lightrag_graph" not in [p["type"] for p in parsed]
    payload = next(p["data"] for p in parsed if p["type"] == "done")
    assert payload["response"] == "ES answer [1](cite://1)"
    assert payload["lightrag_graph"]["data"]["chunks"] == [{"id": "c1", "silo_id": 7}]
