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
    create_agent = AsyncMock(return_value=(agent_chain, None, None))

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
            (first_chain, None, None),
            (second_chain, None, None),
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
    create_agent = AsyncMock(return_value=(first_chain, None, None))

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


@pytest.mark.asyncio
async def test_streaming_sets_turn_used_tool_call_on_ctx_before_finalize(monkeypatch):
    """Spec `user-memory-mem0`, D-23/step_008: the streaming path must set
    `ctx.turn_used_tool_call` BEFORE `_finalize_turn` runs — the write path
    (step_008) reads it off `ctx` to decide whether the assistant's reply may
    enter the extraction window.

    Deliberately derived from the raw ``updates`` chunk's messages directly
    (via `_response_included_tool_call`), NOT from `map_stream_event`'s SSE
    events — a security audit found the events path an unreliable gating
    signal (see `test_streaming_flags_tool_call_even_when_sse_mapping_fails`
    below). `map_stream_event` is mocked out here to isolate that this
    signal no longer depends on it at all.
    """
    from services.agent_streaming_service import AgentStreamingService

    ctx = SimpleNamespace(
        effective_conv_id=297,
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
        return_value={"parsed_response": "done", "effective_conv_id": 297, "files_data": []}
    )

    tool_call_message = SimpleNamespace(
        content="", tool_calls=[{"name": "search", "args": {}, "id": "1"}]
    )

    async def _one_chunk_stream():
        yield ("updates", {"agent": {"messages": [tool_call_message]}})

    agent_chain = MagicMock()
    agent_chain.astream.return_value = _one_chunk_stream()
    create_agent = AsyncMock(return_value=(agent_chain, None, None))

    service = AgentStreamingService()
    service.execution_service = execution_service

    with (
        patch("services.agent_streaming_service.create_agent", create_agent),
        patch("services.agent_streaming_service.prepare_agent_config", return_value={"configurable": {}}),
        patch("services.agent_streaming_service.build_human_message", return_value=SimpleNamespace(content="hello")),
        # `map_stream_event` returns nothing here — proves the tool-call
        # signal does not depend on it emitting `tool_start`/`tool_end`.
        patch("services.agent_streaming_service.map_stream_event", return_value=None),
    ):
        [
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

    assert ctx.turn_used_tool_call is True
    execution_service._finalize_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_flags_tool_call_even_when_sse_mapping_fails():
    """MEDIUM (step_008 audit): `map_stream_event` swallows exceptions and
    returns `None`, skips `tool_start` when the tool name is empty, and
    requires `"tool" in node_name` for `tool_end` — any of those is a false
    negative that would let tool/retrieved content leak into the global
    memory write path's extraction window if it were the only signal.

    Reproduces exactly that: a node name that does NOT contain "tool" (so
    the old `tool_end` heuristic would have missed it) carrying a
    `ToolMessage`-shaped message, with `map_stream_event` itself returning
    `None` (simulating any of its own failure/skip modes). The flag must
    still end up `True` because it is derived directly from the raw chunk's
    messages, independent of `map_stream_event`.
    """
    from services.agent_streaming_service import AgentStreamingService

    ctx = SimpleNamespace(
        effective_conv_id=297,
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
        return_value={"parsed_response": "done", "effective_conv_id": 297, "files_data": []}
    )

    tool_result_message = SimpleNamespace(type="tool", content="secret retrieved content")

    async def _one_chunk_stream():
        # Node name deliberately does not contain "tool" — the OLD
        # `_map_updates_chunk` tool_end heuristic would have skipped this.
        yield ("updates", {"some_other_node": {"messages": [tool_result_message]}})

    agent_chain = MagicMock()
    agent_chain.astream.return_value = _one_chunk_stream()
    create_agent = AsyncMock(return_value=(agent_chain, None, None))

    service = AgentStreamingService()
    service.execution_service = execution_service

    with (
        patch("services.agent_streaming_service.create_agent", create_agent),
        patch("services.agent_streaming_service.prepare_agent_config", return_value={"configurable": {}}),
        patch("services.agent_streaming_service.build_human_message", return_value=SimpleNamespace(content="hello")),
        # Simulate map_stream_event failing/skipping entirely for this chunk.
        patch("services.agent_streaming_service.map_stream_event", return_value=None),
    ):
        [
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

    assert ctx.turn_used_tool_call is True


@pytest.mark.asyncio
async def test_streaming_leaves_turn_used_tool_call_false_when_no_tool_ran():
    from services.agent_streaming_service import AgentStreamingService

    ctx = SimpleNamespace(
        effective_conv_id=297,
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
        return_value={"parsed_response": "done", "effective_conv_id": 297, "files_data": []}
    )

    agent_chain = MagicMock()
    agent_chain.astream.return_value = _empty_stream()
    create_agent = AsyncMock(return_value=(agent_chain, None, None))

    service = AgentStreamingService()
    service.execution_service = execution_service

    with (
        patch("services.agent_streaming_service.create_agent", create_agent),
        patch("services.agent_streaming_service.prepare_agent_config", return_value={"configurable": {}}),
        patch("services.agent_streaming_service.build_human_message", return_value=SimpleNamespace(content="hello")),
    ):
        [
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

    assert ctx.turn_used_tool_call is False
