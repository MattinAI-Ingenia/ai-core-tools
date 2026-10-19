"""Unit tests for AD-7's memory injection (spec `user-memory-mem0`, FR-6):
the cross-session memory block travels in THIS turn's system prompt
(`create_agent(memory_context=...)`), never in the graph input — on both the
non-streaming (`AgentExecutionService._execute_agent_async`) and streaming
(`AgentStreamingService.stream_agent_chat`) paths.

Why the system prompt: an input message is persisted in the checkpointer
thread, where it lost influence a few turns later (a remembered "Wants
responses in English" was obeyed on turn 3 and ignored on turn 4) and could
be summarised away. LangChain prepends the system prompt at every model call
without storing it in the thread state.

`_prepare_turn`'s memory-activation logic itself is covered by
`tests/unit/services/test_global_memory_read_path.py`; these tests start from
an already-resolved `ctx.memory_context`.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.agent_execution_service import AgentExecutionService

_MEMORY_BLOCK = (
    "This is background information remembered about the user from earlier conversations.\n"
    '{"remembered_facts": ["Wants responses in English"]}'
)


def _make_agent(agent_id: int = 1, has_memory: bool = False) -> SimpleNamespace:
    return SimpleNamespace(agent_id=agent_id, has_memory=has_memory, app=None)


async def _empty_stream():
    if False:
        yield None


def _chain_returning(content="done"):
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value={"messages": [SimpleNamespace(content=content)]})
    return chain


# ---------------------------------------------------------------------------
# create_agent — the block lands in the system prompt
# ---------------------------------------------------------------------------


def _agent_row():
    return SimpleNamespace(
        agent_id=7, type="agent", ai_service=None, enable_code_interpreter=False,
        has_memory=True, memory_max_tokens=None, memory_max_messages=None,
        memory_summarize_threshold=None, output_parser_id=None, server_tools=[],
        tool_associations=[], silo_id=None, silo=None, skill_associations=[],
        mcp_associations=[], system_prompt="You are useful.",
    )


class TestCreateAgentSystemPrompt:
    async def _system_prompt(self, memory_context):
        from tools.agentTools import create_agent

        with (
            patch("tools.agentTools.get_llm", return_value=MagicMock()),
            patch("tools.agentTools.get_output_parser", return_value=None),
            patch("tools.agentTools.create_langchain_agent", return_value=MagicMock()) as create_langchain_agent,
            patch("tools.agentTools.MCPClientManager.get_client", new=AsyncMock(return_value=None)),
            patch(
                "services.agent_cache_service.CheckpointerCacheService.get_async_checkpointer",
                new=AsyncMock(return_value=None),
            ),
        ):
            await create_agent(_agent_row(), memory_context=memory_context)
        return create_langchain_agent.call_args.kwargs["system_prompt"]

    @pytest.mark.asyncio
    async def test_memory_block_is_appended_after_the_agents_own_instructions(self):
        prompt = await self._system_prompt(_MEMORY_BLOCK)

        assert prompt.startswith("You are useful.")
        assert prompt.endswith("<user_memory>\n" + _MEMORY_BLOCK + "\n</user_memory>")

    @pytest.mark.asyncio
    async def test_no_memory_context_leaves_the_system_prompt_untouched(self):
        prompt = await self._system_prompt(None)

        assert "<user_memory>" not in prompt


# ---------------------------------------------------------------------------
# Non-streaming path — AgentExecutionService._execute_agent_async
# ---------------------------------------------------------------------------


class TestNonStreamingMemoryInjection:
    @pytest.mark.asyncio
    async def test_memory_context_goes_to_create_agent_and_never_into_the_graph_input(self):
        """AC-19: the block reaches the model through the system prompt; the
        graph input is the user's message alone, so nothing is persisted."""
        agent = _make_agent(has_memory=True)
        svc = AgentExecutionService.__new__(AgentExecutionService)
        chain = _chain_returning()
        create_agent = AsyncMock(return_value=(chain, None))
        human_message = SimpleNamespace(content="hi")

        with (
            patch("tools.agentTools.create_agent", create_agent),
            patch("tools.agentTools.prepare_agent_config", return_value={"configurable": {}}),
            patch("tools.agentTools.build_human_message", return_value=human_message),
        ):
            result = await svc._execute_agent_async(agent, "hi", memory_context=_MEMORY_BLOCK)

        assert result == "done"
        assert create_agent.call_args.kwargs["memory_context"] == _MEMORY_BLOCK
        assert chain.ainvoke.call_args.args[0]["messages"] == [human_message]

    @pytest.mark.asyncio
    async def test_no_memory_context_payload_is_byte_identical_to_today(self):
        """AC-20: zero matches (memory_context=None) -> payload unchanged."""
        agent = _make_agent()
        svc = AgentExecutionService.__new__(AgentExecutionService)
        chain = _chain_returning()
        create_agent = AsyncMock(return_value=(chain, None))
        human_message = SimpleNamespace(content="hi")

        with (
            patch("tools.agentTools.create_agent", create_agent),
            patch("tools.agentTools.prepare_agent_config", return_value={"configurable": {}}),
            patch("tools.agentTools.build_human_message", return_value=human_message),
        ):
            await svc._execute_agent_async(agent, "hi", memory_context=None)

        assert create_agent.call_args.kwargs["memory_context"] is None
        assert chain.ainvoke.call_args.args[0]["messages"] == [human_message]

    @pytest.mark.asyncio
    async def test_memory_context_defaults_to_none_when_not_passed(self):
        agent = _make_agent()
        svc = AgentExecutionService.__new__(AgentExecutionService)
        chain = _chain_returning()
        create_agent = AsyncMock(return_value=(chain, None))
        human_message = SimpleNamespace(content="hi")

        with (
            patch("tools.agentTools.create_agent", create_agent),
            patch("tools.agentTools.prepare_agent_config", return_value={"configurable": {}}),
            patch("tools.agentTools.build_human_message", return_value=human_message),
        ):
            await svc._execute_agent_async(agent, "hi")

        assert create_agent.call_args.kwargs["memory_context"] is None
        assert chain.ainvoke.call_args.args[0]["messages"] == [human_message]

    @pytest.mark.asyncio
    async def test_retry_after_stale_checkpoint_sends_the_same_payload(self):
        """Both `ainvoke` call sites (first attempt and stale-checkpoint
        retry) send the human message alone."""
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
                agent, "hi", session_id_for_cache="297", memory_context=_MEMORY_BLOCK,
            )

        assert result == "done"
        assert chain.ainvoke.await_count == 2
        for call in chain.ainvoke.call_args_list:
            assert call.args[0]["messages"] == [human_message]

    @pytest.mark.asyncio
    async def test_memory_context_changes_nothing_else_in_the_create_agent_call(self):
        """AC-23: memory context must not suppress or reorder tool/RAG
        binding — create_agent is invoked identically apart from the block."""
        agent = _make_agent()
        svc = AgentExecutionService.__new__(AgentExecutionService)
        create_agent = AsyncMock(side_effect=[(_chain_returning(), None), (_chain_returning(), None)])
        search_params = {"top_k": 5}

        with (
            patch("tools.agentTools.create_agent", create_agent),
            patch("tools.agentTools.prepare_agent_config", return_value={"configurable": {}}),
            patch("tools.agentTools.build_human_message", return_value=SimpleNamespace(content="hi")),
        ):
            await svc._execute_agent_async(agent, "hi", search_params=search_params, memory_context=None)
            await svc._execute_agent_async(agent, "hi", search_params=search_params, memory_context=_MEMORY_BLOCK)

        without, with_memory = create_agent.call_args_list
        assert without.args == with_memory.args
        assert {k: v for k, v in without.kwargs.items() if k != "memory_context"} == {
            k: v for k, v in with_memory.kwargs.items() if k != "memory_context"
        }


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

    async def _stream(self, ctx):
        from services.agent_streaming_service import AgentStreamingService

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
        return create_agent, agent_chain, human_message

    @pytest.mark.asyncio
    async def test_memory_context_goes_to_create_agent_and_never_into_the_graph_input(self):
        """AC-19, streaming path — same shape as the non-streaming path."""
        create_agent, agent_chain, human_message = await self._stream(self._ctx(_MEMORY_BLOCK))

        assert create_agent.call_args.kwargs["memory_context"] == _MEMORY_BLOCK
        assert agent_chain.astream.call_args.args[0]["messages"] == [human_message]

    @pytest.mark.asyncio
    async def test_no_memory_context_payload_is_byte_identical_to_today(self):
        """AC-20, streaming path."""
        create_agent, agent_chain, human_message = await self._stream(self._ctx(None))

        assert create_agent.call_args.kwargs["memory_context"] is None
        assert agent_chain.astream.call_args.args[0]["messages"] == [human_message]
