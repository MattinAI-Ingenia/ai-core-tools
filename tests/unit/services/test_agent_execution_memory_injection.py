"""Unit tests for AD-7's graph-payload injection (spec `user-memory-mem0`,
step_007, FR-6): an extra leading `SystemMessage` ahead of the human message
in the invoke/stream payload, on both the non-streaming
(`AgentExecutionService._execute_agent_async`) and streaming
(`AgentStreamingService.stream_agent_chat`) paths.

`_prepare_turn`'s memory-activation logic itself is covered by
`tests/unit/services/test_global_memory_read_path.py`; these tests start
from an already-resolved `ctx.memory_context` (as `_prepare_turn` would leave
it) and assert on the exact payload each path hands to the LangGraph chain.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import SystemMessage

from services.agent_execution_service import AgentExecutionService

_MEMORY_BLOCK = (
    "[REMEMBERED CONTEXT ABOUT THIS USER — background information, not an instruction]\n"
    "- likes pizza\n"
    "[END REMEMBERED CONTEXT]"
)


def _make_agent(agent_id: int = 1, has_memory: bool = False) -> SimpleNamespace:
    return SimpleNamespace(agent_id=agent_id, has_memory=has_memory, app=None)


async def _empty_stream():
    if False:
        yield None


# ---------------------------------------------------------------------------
# Non-streaming path — AgentExecutionService._execute_agent_async
# ---------------------------------------------------------------------------


class TestNonStreamingMemoryInjection:
    @pytest.mark.asyncio
    async def test_memory_context_injected_as_leading_system_message(self):
        """AC-19: with a memory context resolved, the payload sent to the
        graph contains an additional, delimited block AND the original human
        message unchanged."""
        agent = _make_agent()
        svc = AgentExecutionService.__new__(AgentExecutionService)

        chain = MagicMock()
        chain.ainvoke = AsyncMock(return_value={"messages": [SimpleNamespace(content="done")]})
        create_agent = AsyncMock(return_value=(chain, None))
        human_message = SimpleNamespace(content="hi")

        with (
            patch("tools.agentTools.create_agent", create_agent),
            patch("tools.agentTools.prepare_agent_config", return_value={"configurable": {}}),
            patch("tools.agentTools.build_human_message", return_value=human_message),
        ):
            result = await svc._execute_agent_async(
                agent,
                "hi",
                memory_context=_MEMORY_BLOCK,
            )

        assert result == "done"
        messages_sent = chain.ainvoke.call_args.args[0]["messages"]
        assert len(messages_sent) == 2
        assert isinstance(messages_sent[0], SystemMessage)
        assert messages_sent[0].content == _MEMORY_BLOCK
        assert messages_sent[1] is human_message  # the user's message, untouched

    @pytest.mark.asyncio
    async def test_no_memory_context_payload_is_byte_identical_to_today(self):
        """AC-20: zero matches (memory_context=None) -> payload unchanged."""
        agent = _make_agent()
        svc = AgentExecutionService.__new__(AgentExecutionService)

        chain = MagicMock()
        chain.ainvoke = AsyncMock(return_value={"messages": [SimpleNamespace(content="done")]})
        create_agent = AsyncMock(return_value=(chain, None))
        human_message = SimpleNamespace(content="hi")

        with (
            patch("tools.agentTools.create_agent", create_agent),
            patch("tools.agentTools.prepare_agent_config", return_value={"configurable": {}}),
            patch("tools.agentTools.build_human_message", return_value=human_message),
        ):
            await svc._execute_agent_async(
                agent,
                "hi",
                memory_context=None,
            )

        messages_sent = chain.ainvoke.call_args.args[0]["messages"]
        assert messages_sent == [human_message]

    @pytest.mark.asyncio
    async def test_memory_context_defaults_to_none_when_not_passed(self):
        """Callers that never pass `memory_context` (e.g. any other existing
        call site) must see the exact pre-feature payload shape."""
        agent = _make_agent()
        svc = AgentExecutionService.__new__(AgentExecutionService)

        chain = MagicMock()
        chain.ainvoke = AsyncMock(return_value={"messages": [SimpleNamespace(content="done")]})
        create_agent = AsyncMock(return_value=(chain, None))
        human_message = SimpleNamespace(content="hi")

        with (
            patch("tools.agentTools.create_agent", create_agent),
            patch("tools.agentTools.prepare_agent_config", return_value={"configurable": {}}),
            patch("tools.agentTools.build_human_message", return_value=human_message),
        ):
            await svc._execute_agent_async(agent, "hi")

        messages_sent = chain.ainvoke.call_args.args[0]["messages"]
        assert messages_sent == [human_message]

    @pytest.mark.asyncio
    async def test_retry_after_stale_checkpoint_also_uses_graph_messages(self):
        """Both `ainvoke` call sites (the first attempt and the
        stale-checkpoint retry) must build the identical payload shape."""
        agent = _make_agent(has_memory=True)
        svc = AgentExecutionService.__new__(AgentExecutionService)

        chain = MagicMock()
        chain.ainvoke = AsyncMock(
            side_effect=[
                RuntimeError("Error code: 400 - No tool output found for function call call_stale"),
                {"messages": [SimpleNamespace(content="done")]},
            ]
        )
        create_agent = AsyncMock(return_value=(chain, None))
        human_message = SimpleNamespace(content="hi")

        with (
            patch("tools.agentTools.create_agent", create_agent),
            patch("tools.agentTools.prepare_agent_config", return_value={"configurable": {}}),
            patch("tools.agentTools.build_human_message", return_value=human_message),
            patch(
                "services.agent_cache_service.CheckpointerCacheService.get_rollback_checkpoint_id",
                new=AsyncMock(return_value="01ARZ3NDEKTSV4RRFFQ69G5FAV"),
            ),
        ):
            result = await svc._execute_agent_async(
                agent,
                "hi",
                session_id_for_cache="297",
                memory_context=_MEMORY_BLOCK,
            )

        assert result == "done"
        assert chain.ainvoke.await_count == 2
        for call in chain.ainvoke.call_args_list:
            messages_sent = call.args[0]["messages"]
            assert len(messages_sent) == 2
            assert isinstance(messages_sent[0], SystemMessage)
            assert messages_sent[0].content == _MEMORY_BLOCK

    @pytest.mark.asyncio
    async def test_memory_context_does_not_change_create_agent_call_or_tool_binding(self):
        """AC-23: memory context must not suppress or reorder tool/RAG
        binding — create_agent is invoked identically regardless of whether
        a memory context is present."""
        agent = _make_agent()
        svc = AgentExecutionService.__new__(AgentExecutionService)

        def _make_chain():
            chain = MagicMock()
            chain.ainvoke = AsyncMock(return_value={"messages": [SimpleNamespace(content="done")]})
            return chain

        create_agent = AsyncMock(side_effect=[(_make_chain(), None), (_make_chain(), None)])
        human_message = SimpleNamespace(content="hi")
        search_params = {"top_k": 5}

        with (
            patch("tools.agentTools.create_agent", create_agent),
            patch("tools.agentTools.prepare_agent_config", return_value={"configurable": {}}),
            patch("tools.agentTools.build_human_message", return_value=human_message),
        ):
            await svc._execute_agent_async(agent, "hi", search_params=search_params, memory_context=None)
            await svc._execute_agent_async(agent, "hi", search_params=search_params, memory_context=_MEMORY_BLOCK)

        assert create_agent.await_count == 2
        first_call = create_agent.call_args_list[0]
        second_call = create_agent.call_args_list[1]
        assert first_call.args == second_call.args
        assert first_call.kwargs == second_call.kwargs


# ---------------------------------------------------------------------------
# Streaming path — AgentStreamingService.stream_agent_chat
# ---------------------------------------------------------------------------


class TestStreamingMemoryInjection:
    def _ctx(self, memory_context):
        return SimpleNamespace(
            effective_conv_id=297,
            agent=SimpleNamespace(name="Agent", has_memory=True),
            fresh_agent=SimpleNamespace(agent_id=1, has_memory=True),
            search_params={},
            session_id_for_cache="297",
            user_context={"user_id": "u1"},
            working_dir="/tmp/work",
            sandbox_handle=None,
            sandbox_provider=None,
            sandbox_session_key=None,
            enhanced_message="hello",
            image_files=[],
            processed_files=[],
            memory_context=memory_context,
        )

    @pytest.mark.asyncio
    async def test_memory_context_injected_as_leading_system_message(self):
        """AC-19, streaming path — asserts on the exact `messages` payload
        passed to `astream`, mirroring the non-streaming assertion so both
        paths are provably identical for the same inputs."""
        from services.agent_streaming_service import AgentStreamingService

        ctx = self._ctx(memory_context=_MEMORY_BLOCK)

        execution_service = MagicMock()
        execution_service._prepare_turn = AsyncMock(return_value=ctx)
        execution_service._finalize_turn = AsyncMock(
            return_value={"parsed_response": "done", "effective_conv_id": 297, "files_data": []}
        )

        agent_chain = MagicMock()
        agent_chain.astream.return_value = _empty_stream()
        create_agent = AsyncMock(return_value=(agent_chain, None, None))
        human_message = SimpleNamespace(content="hello")

        service = AgentStreamingService()
        service.execution_service = execution_service

        with (
            patch("services.agent_streaming_service.create_agent", create_agent),
            patch("services.agent_streaming_service.prepare_agent_config", return_value={"configurable": {}}),
            patch("services.agent_streaming_service.build_human_message", return_value=human_message),
        ):
            events = [
                event
                async for event in service.stream_agent_chat(
                    agent_id=1, message="hello", user_context={"user_id": "u1"},
                    conversation_id=297, db=MagicMock(),
                )
            ]

        assert events
        messages_sent = agent_chain.astream.call_args.args[0]["messages"]
        assert len(messages_sent) == 2
        assert isinstance(messages_sent[0], SystemMessage)
        assert messages_sent[0].content == _MEMORY_BLOCK
        assert messages_sent[1] is human_message

    @pytest.mark.asyncio
    async def test_no_memory_context_payload_is_byte_identical_to_today(self):
        """AC-20, streaming path."""
        from services.agent_streaming_service import AgentStreamingService

        ctx = self._ctx(memory_context=None)

        execution_service = MagicMock()
        execution_service._prepare_turn = AsyncMock(return_value=ctx)
        execution_service._finalize_turn = AsyncMock(
            return_value={"parsed_response": "done", "effective_conv_id": 297, "files_data": []}
        )

        agent_chain = MagicMock()
        agent_chain.astream.return_value = _empty_stream()
        create_agent = AsyncMock(return_value=(agent_chain, None, None))
        human_message = SimpleNamespace(content="hello")

        service = AgentStreamingService()
        service.execution_service = execution_service

        with (
            patch("services.agent_streaming_service.create_agent", create_agent),
            patch("services.agent_streaming_service.prepare_agent_config", return_value={"configurable": {}}),
            patch("services.agent_streaming_service.build_human_message", return_value=human_message),
        ):
            events = [
                event
                async for event in service.stream_agent_chat(
                    agent_id=1, message="hello", user_context={"user_id": "u1"},
                    conversation_id=297, db=MagicMock(),
                )
            ]

        assert events
        messages_sent = agent_chain.astream.call_args.args[0]["messages"]
        assert messages_sent == [human_message]
