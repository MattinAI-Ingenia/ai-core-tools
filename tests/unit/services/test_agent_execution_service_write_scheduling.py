"""Unit tests for the FR-5 write-path SCHEDULING wiring `_finalize_turn` adds
(spec `user-memory-mem0`, step_008): evaluating the D-15 batch trigger at
scheduling time using `ctx.memory_turn_index`, and forwarding the right
(cleaned) inputs to `services.global_memory_service.schedule_write_path`.

`schedule_write_path`/`run_write_path`/`extract_facts` themselves are unit
tested in `test_global_memory_write_path.py`; here `schedule_write_path` is
mocked so these tests focus purely on `_finalize_turn`'s own wiring: WHEN it
schedules, and WHAT it passes.

Because both the streaming and non-streaming chat paths delegate to the same
shared `_finalize_turn` (`agent_streaming_service.py` calls it directly), a
single set of tests against `_finalize_turn` covers both paths — see
`agent_streaming_service.py`'s own docstring for that delegation.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.agent_execution_context import AgentExecutionContext
from services.agent_execution_service import (
    AgentExecutionService,
    _response_included_tool_call,
)
from tools.memory import settings as memory_settings
from tools.memory.identity import MemoryIdentity


def _identity(app_id: int = 1, user_id: int = 7, agent_id: int = 3) -> MemoryIdentity:
    root = memory_settings.namespace_root()
    return MemoryIdentity(
        app_id=app_id, user_id=user_id, agent_id=agent_id,
        namespace=(root, str(app_id), str(user_id)),
    )


def _make_agent(agent_id: int = 1, agent_type: str = "agent", name: str = "Test Agent") -> MagicMock:
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.name = name
    agent.type = agent_type
    return agent


def _make_service() -> AgentExecutionService:
    svc = AgentExecutionService.__new__(AgentExecutionService)
    svc.agent_execution_repo = MagicMock()
    svc.session_service = MagicMock()
    svc.session_service.touch_session = AsyncMock()
    return svc


def _make_context(**overrides) -> AgentExecutionContext:
    agent = overrides.pop("agent", None) or _make_agent()
    fresh_agent = overrides.pop("fresh_agent", None) or MagicMock(ai_service=None)
    defaults = dict(
        agent_id=1,
        agent=agent,
        fresh_agent=fresh_agent,
        enhanced_message="hello",
        image_files=[],
        session=None,
        conversation=None,
        effective_conv_id=None,
        session_id_for_cache=None,
        working_dir=None,
        pre_existing_files=set(),
        processed_files=[],
        search_params=None,
        user_context=None,
        global_memory_identity=None,
        global_memory_services=None,
        global_memory_llm=None,
        global_memory_embeddings=None,
        memory_context=None,
        memory_turn_index=1,
        original_user_message="hello",
        turn_used_tool_call=False,
    )
    defaults.update(overrides)
    return AgentExecutionContext(**defaults)


async def _finalize(svc: AgentExecutionService, ctx: AgentExecutionContext, response_text: str = "Sure, noted."):
    with patch("tools.agentTools.parse_agent_response", return_value=response_text):
        return await svc._finalize_turn(ctx, response_text, db=MagicMock())


# ---------------------------------------------------------------------------
# AC-18: flag off (no active identity) -> zero write-path calls
# ---------------------------------------------------------------------------


class TestNoActiveIdentityNeverSchedules:
    @pytest.mark.asyncio
    async def test_inactive_memory_never_calls_schedule_write_path(self):
        """`ctx.global_memory_identity is None` covers every reason memory is
        inactive this turn (kill switch off, `has_global_memory=False`,
        ineligible identity, unresolved services) — `_prepare_turn` never
        sets it in any of those cases. Covers both chat paths, since both
        delegate to this same `_finalize_turn` (see module docstring)."""
        svc = _make_service()
        ctx = _make_context(global_memory_identity=None)

        with patch("services.global_memory_service.schedule_write_path") as mock_schedule:
            await _finalize(svc, ctx)

        mock_schedule.assert_not_called()


# ---------------------------------------------------------------------------
# AC-49: the batched trigger fires on the exact turn set, and no others
# ---------------------------------------------------------------------------


class TestBatchTriggerExactTurnSet:
    @pytest.mark.asyncio
    async def test_fires_on_turns_1_5_10_through_30_and_no_others(self):
        """With the default `write_every_n_turns=5`, driving turns 1..31
        must schedule a write on exactly {1, 5, 10, 15, 20, 25, 30} — this is
        `ctx.memory_turn_index` (as `_prepare_turn` would have resolved it,
        BEFORE `_finalize_turn`'s own message-count increment), never
        recomputed from `ctx.conversation` here. `ctx.conversation=None` in
        every case below so step 7's own increment never runs — irrelevant
        to this scheduling decision, since it reads the cached index, not
        `ctx.conversation`."""
        svc = _make_service()
        expected_fire_turns = {1, 5, 10, 15, 20, 25, 30}

        for turn in range(1, 32):
            with patch("services.global_memory_service.schedule_write_path") as mock_schedule:
                ctx = _make_context(global_memory_identity=_identity(), memory_turn_index=turn)
                await _finalize(svc, ctx)

            if turn in expected_fire_turns:
                assert mock_schedule.call_count == 1, f"turn {turn} should have scheduled a write"
            else:
                assert mock_schedule.call_count == 0, f"turn {turn} should NOT have scheduled a write"

    @pytest.mark.asyncio
    async def test_a_two_turn_conversation_still_extracts_once_on_turn_1(self):
        """R-16: the `turn_index == 1` clause is the only thing that
        captures a conversation that ends before the first threshold."""
        svc = _make_service()

        with patch("services.global_memory_service.schedule_write_path") as mock_schedule:
            ctx_turn_1 = _make_context(global_memory_identity=_identity(), memory_turn_index=1)
            await _finalize(svc, ctx_turn_1)
        assert mock_schedule.call_count == 1

        with patch("services.global_memory_service.schedule_write_path") as mock_schedule:
            ctx_turn_2 = _make_context(global_memory_identity=_identity(), memory_turn_index=2)
            await _finalize(svc, ctx_turn_2)
        assert mock_schedule.call_count == 0


# ---------------------------------------------------------------------------
# AC-13: scheduling is synchronous — `_finalize_turn` never awaits the
# scheduled task, and its own response is unaffected by it.
# ---------------------------------------------------------------------------


class TestSchedulingIsNonBlocking:
    @pytest.mark.asyncio
    async def test_finalize_turn_returns_normally_and_calls_schedule_synchronously(self):
        """`schedule_write_path` is a plain (non-async) function — if
        `_finalize_turn` ever mistakenly tried to `await` its return value,
        this test would fail with a TypeError (a `MagicMock()` result is not
        awaitable), so passing here structurally proves it is called, not
        awaited (AC-13)."""
        svc = _make_service()
        ctx = _make_context(global_memory_identity=_identity(), memory_turn_index=1)

        with patch("services.global_memory_service.schedule_write_path") as mock_schedule:
            result = await _finalize(svc, ctx, response_text="All good.")

        mock_schedule.assert_called_once()
        assert result["response"] == "All good."


# ---------------------------------------------------------------------------
# AC-14/scheduling failure containment: a bug in the scheduling call itself
# must never fail the turn.
# ---------------------------------------------------------------------------


class TestSchedulingFailureContainment:
    @pytest.mark.asyncio
    async def test_schedule_write_path_raising_does_not_fail_the_turn(self):
        svc = _make_service()
        ctx = _make_context(global_memory_identity=_identity(), memory_turn_index=1)

        with patch(
            "services.global_memory_service.schedule_write_path",
            side_effect=RuntimeError("boom"),
        ):
            result = await _finalize(svc, ctx, response_text="All good.")

        assert result["response"] == "All good."


# ---------------------------------------------------------------------------
# AC-60: the write path receives the ORIGINAL user message, never the
# attachment-enhanced one, plus the correct tool-call exclusion signal.
# ---------------------------------------------------------------------------


class TestSchedulingForwardsCleanInputs:
    @pytest.mark.asyncio
    async def test_forwards_original_user_message_not_the_enhanced_one(self):
        """`ctx.enhanced_message` carries the `[Attached files:]` segment and
        `Files base folder is:` line for any turn with a text attachment
        (`_prepare_message_with_files`); `ctx.original_user_message` never
        does. `_finalize_turn` must forward the latter to
        `schedule_write_path`."""
        svc = _make_service()
        enhanced = (
            "Please summarise this contract"
            "\n\nFiles base folder is: /tmp/mattin"
            "\n\n[Attached files:]"
            "\n\n--- File: contract.txt (Sandbox path: input/contract.txt) ---\n"
            "CONFIDENTIAL: SSN 123-45-6789\n"
            "--- End of contract.txt ---"
        )
        ctx = _make_context(
            global_memory_identity=_identity(),
            memory_turn_index=1,
            enhanced_message=enhanced,
            original_user_message="Please summarise this contract",
        )

        with patch("services.global_memory_service.schedule_write_path") as mock_schedule:
            await _finalize(svc, ctx, response_text="Here is a summary.")

        forwarded = mock_schedule.call_args.kwargs["original_user_message"]
        assert forwarded == "Please summarise this contract"
        assert "[Attached files:]" not in forwarded
        assert "Files base folder is:" not in forwarded
        assert "123-45-6789" not in forwarded

    @pytest.mark.asyncio
    async def test_forwards_turn_used_tool_call_flag(self):
        svc = _make_service()
        ctx = _make_context(
            global_memory_identity=_identity(), memory_turn_index=1, turn_used_tool_call=True,
        )

        with patch("services.global_memory_service.schedule_write_path") as mock_schedule:
            await _finalize(svc, ctx, response_text="Answer based on the document.")

        assert mock_schedule.call_args.kwargs["turn_used_tool_call"] is True

    @pytest.mark.asyncio
    async def test_forwards_turn_had_attachments_flag_from_processed_files(self):
        """HIGH (step_008 audit): uploading a file attachment is not a tool
        call — `_finalize_turn` must independently derive
        `turn_had_attachments` from `ctx.processed_files` and forward it,
        so `schedule_write_path` can gate the assistant's reply out of the
        extraction window even when `turn_used_tool_call` is `False`."""
        svc = _make_service()
        ctx = _make_context(
            global_memory_identity=_identity(),
            memory_turn_index=1,
            turn_used_tool_call=False,
            processed_files=[{"filename": "cv.pdf", "content": "..."}],
        )

        with patch("services.global_memory_service.schedule_write_path") as mock_schedule:
            await _finalize(svc, ctx, response_text="Resumen del CV adjunto.")

        assert mock_schedule.call_args.kwargs["turn_had_attachments"] is True
        assert mock_schedule.call_args.kwargs["turn_used_tool_call"] is False

    @pytest.mark.asyncio
    async def test_turn_had_attachments_is_false_without_processed_files(self):
        svc = _make_service()
        ctx = _make_context(
            global_memory_identity=_identity(), memory_turn_index=1, processed_files=[],
        )

        with patch("services.global_memory_service.schedule_write_path") as mock_schedule:
            await _finalize(svc, ctx, response_text="Plain reply.")

        assert mock_schedule.call_args.kwargs["turn_had_attachments"] is False

    @pytest.mark.asyncio
    async def test_never_passes_a_db_session_to_the_write_path(self):
        """AC-39's structural guarantee, exercised from the scheduling side:
        `_finalize_turn` never has anything DB-shaped to hand to
        `schedule_write_path` in the first place."""
        svc = _make_service()
        ctx = _make_context(global_memory_identity=_identity(), memory_turn_index=1)

        with patch("services.global_memory_service.schedule_write_path") as mock_schedule:
            await _finalize(svc, ctx)

        assert "db" not in mock_schedule.call_args.kwargs
        assert "session" not in mock_schedule.call_args.kwargs

    @pytest.mark.asyncio
    async def test_no_response_text_does_not_schedule_a_write(self):
        """A turn whose flattened response is empty (e.g. an image-only
        reply with no text) has nothing to extract from — no task should be
        created at all."""
        svc = _make_service()
        ctx = _make_context(global_memory_identity=_identity(), memory_turn_index=1)

        with patch("services.global_memory_service.schedule_write_path") as mock_schedule:
            await _finalize(svc, ctx, response_text="")

        mock_schedule.assert_not_called()


# ---------------------------------------------------------------------------
# `_response_included_tool_call` — the non-streaming tool-call detector
# `_execute_agent_async` feeds into `ctx.turn_used_tool_call`.
# ---------------------------------------------------------------------------


class TestResponseIncludedToolCall:
    def test_false_for_a_plain_text_only_response(self):
        from types import SimpleNamespace

        plain_ai_message = SimpleNamespace(content="hi")
        assert _response_included_tool_call([plain_ai_message]) is False

    def test_true_when_a_message_carries_tool_calls(self):
        from types import SimpleNamespace

        ai_message_with_tool_call = SimpleNamespace(
            content="", tool_calls=[{"name": "search", "args": {}, "id": "1"}]
        )
        assert _response_included_tool_call([ai_message_with_tool_call]) is True

    def test_true_for_a_tool_message_by_class_name(self):
        class ToolMessage:
            content = "tool output"

        assert _response_included_tool_call([ToolMessage()]) is True

    def test_true_for_a_message_with_type_tool_attribute(self):
        from types import SimpleNamespace

        generic_msg = SimpleNamespace(type="tool")
        assert _response_included_tool_call([generic_msg]) is True

    def test_false_for_empty_or_none_messages(self):
        assert _response_included_tool_call([]) is False
        assert _response_included_tool_call(None) is False


class TestExecuteAgentAsyncSetsToolCallFlagOnContext:
    @pytest.mark.asyncio
    async def test_sets_true_when_the_graph_response_included_a_tool_call(self):
        from types import SimpleNamespace

        agent = SimpleNamespace(agent_id=1, has_memory=False, app=None)
        svc = AgentExecutionService.__new__(AgentExecutionService)
        ctx = _make_context()

        tool_call_message = SimpleNamespace(content="", tool_calls=[{"name": "search", "args": {}, "id": "1"}])
        final_message = SimpleNamespace(content="Here's what I found.")
        chain = MagicMock()
        chain.ainvoke = AsyncMock(return_value={"messages": [tool_call_message, final_message]})

        with (
            patch("tools.agentTools.create_agent", AsyncMock(return_value=(chain, None))),
            patch("tools.agentTools.prepare_agent_config", return_value={"configurable": {}}),
            patch("tools.agentTools.build_human_message", return_value=SimpleNamespace(content="hi")),
        ):
            await svc._execute_agent_async(agent, "hi", ctx=ctx)

        assert ctx.turn_used_tool_call is True

    @pytest.mark.asyncio
    async def test_stays_false_when_the_graph_response_never_called_a_tool(self):
        from types import SimpleNamespace

        agent = SimpleNamespace(agent_id=1, has_memory=False, app=None)
        svc = AgentExecutionService.__new__(AgentExecutionService)
        ctx = _make_context()

        final_message = SimpleNamespace(content="Just a plain answer.")
        chain = MagicMock()
        chain.ainvoke = AsyncMock(return_value={"messages": [final_message]})

        with (
            patch("tools.agentTools.create_agent", AsyncMock(return_value=(chain, None))),
            patch("tools.agentTools.prepare_agent_config", return_value={"configurable": {}}),
            patch("tools.agentTools.build_human_message", return_value=SimpleNamespace(content="hi")),
        ):
            await svc._execute_agent_async(agent, "hi", ctx=ctx)

        assert ctx.turn_used_tool_call is False

    @pytest.mark.asyncio
    async def test_ctx_none_does_not_break_execution(self):
        """`ctx` is optional and additive — omitting it must behave exactly
        like before this feature (no attribute error, normal return)."""
        from types import SimpleNamespace

        agent = SimpleNamespace(agent_id=1, has_memory=False, app=None)
        svc = AgentExecutionService.__new__(AgentExecutionService)

        final_message = SimpleNamespace(content="Just a plain answer.")
        chain = MagicMock()
        chain.ainvoke = AsyncMock(return_value={"messages": [final_message]})

        with (
            patch("tools.agentTools.create_agent", AsyncMock(return_value=(chain, None))),
            patch("tools.agentTools.prepare_agent_config", return_value={"configurable": {}}),
            patch("tools.agentTools.build_human_message", return_value=SimpleNamespace(content="hi")),
        ):
            result = await svc._execute_agent_async(agent, "hi")

        assert result == "Just a plain answer."
