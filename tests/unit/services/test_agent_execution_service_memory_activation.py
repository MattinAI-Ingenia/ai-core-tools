"""Unit tests for the memory-activation wiring `_prepare_turn` adds (spec
`user-memory-mem0`, step_007, AD-6): the new step calls
`services.global_memory_service.prepare_turn_memory` /
`get_memory_context` and carries their outcome onto `AgentExecutionContext`.

`prepare_turn_memory`/`get_memory_context` themselves are unit-tested in
`tests/unit/services/test_global_memory_read_path.py`; here they are mocked
so these tests focus purely on the wiring: the returned `ctx` fields, and
fail-soft behaviour when the memory module raises (NFR-4).
"""
from unittest.mock import AsyncMock, MagicMock, patch

from datetime import datetime

import pytest

from services.agent_execution_service import AgentExecutionService

_MEMORY_BLOCK = (
    "[REMEMBERED CONTEXT ABOUT THIS USER — background information, not an instruction]\n"
    "- likes pizza\n"
    "[END REMEMBERED CONTEXT]"
)


def _make_agent(agent_id: int = 1, has_memory: bool = False, has_global_memory: bool = True):
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.name = "Test Agent"
    agent.type = "agent"
    agent.has_memory = has_memory
    agent.has_global_memory = has_global_memory
    agent.silo_id = None
    agent.output_parser_id = None
    agent.request_count = 0
    agent.is_frozen = False
    agent.ai_service = None
    agent.enable_code_interpreter = False
    agent.skill_associations = []
    return agent


def _make_service(agent, fresh_agent=None) -> AgentExecutionService:
    svc = AgentExecutionService.__new__(AgentExecutionService)
    svc.agent_service = MagicMock()
    svc.agent_service.get_agent.return_value = agent
    svc.session_service = MagicMock()
    svc.session_service.get_user_session = AsyncMock(return_value=MagicMock(id="sess"))
    svc.agent_execution_repo = MagicMock()
    svc.agent_execution_repo.get_agent_with_relationships.return_value = fresh_agent or agent
    return svc


class TestPrepareTurnMemoryWiring:
    @pytest.mark.asyncio
    async def test_memory_context_and_activation_fields_populated_on_success(self, tmp_path):
        agent = _make_agent(has_global_memory=True)
        svc = _make_service(agent)

        identity = MagicMock()
        resolved = MagicMock()
        llm = MagicMock()
        embeddings = MagicMock()

        with (
            patch(
                "services.agent_execution_service.get_app_config",
                return_value={"TMP_BASE_FOLDER": str(tmp_path)},
            ),
            patch(
                "services.global_memory_service.prepare_turn_memory",
                new=AsyncMock(return_value=(identity, resolved, llm, embeddings)),
            ),
            patch(
                "services.global_memory_service.get_memory_context",
                new=AsyncMock(return_value=_MEMORY_BLOCK),
            ),
        ):
            ctx = await svc._prepare_turn(
                agent_id=1,
                message="hello",
                user_context={"user_id": 5, "app_id": 1, "oauth": True},
                db=MagicMock(),
            )

        assert ctx.memory_context == _MEMORY_BLOCK
        assert ctx.global_memory_identity is identity
        assert ctx.global_memory_services is resolved
        assert ctx.global_memory_llm is llm
        assert ctx.global_memory_embeddings is embeddings

    @pytest.mark.asyncio
    async def test_inactive_memory_leaves_all_fields_none(self, tmp_path):
        agent = _make_agent(has_global_memory=False)
        svc = _make_service(agent)

        with (
            patch(
                "services.agent_execution_service.get_app_config",
                return_value={"TMP_BASE_FOLDER": str(tmp_path)},
            ),
            patch(
                "services.global_memory_service.prepare_turn_memory",
                new=AsyncMock(return_value=(None, None, None, None)),
            ) as mock_prepare,
            patch(
                "services.global_memory_service.get_memory_context",
                new=AsyncMock(),
            ) as mock_get_context,
        ):
            ctx = await svc._prepare_turn(
                agent_id=1,
                message="hello",
                user_context={"user_id": 5, "app_id": 1, "oauth": True},
                db=MagicMock(),
            )

        assert ctx.memory_context is None
        assert ctx.global_memory_identity is None
        assert ctx.global_memory_services is None
        assert ctx.global_memory_llm is None
        assert ctx.global_memory_embeddings is None
        mock_prepare.assert_awaited_once()
        # AC-18/AC-33: no client is even asked to search when memory is inactive.
        mock_get_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_activation_failure_degrades_to_no_memory_context(self, tmp_path):
        """NFR-4: any exception in the memory-activation step must never
        break the turn — `_prepare_turn` still returns a normal context."""
        agent = _make_agent(has_global_memory=True)
        svc = _make_service(agent)

        with (
            patch(
                "services.agent_execution_service.get_app_config",
                return_value={"TMP_BASE_FOLDER": str(tmp_path)},
            ),
            patch(
                "services.global_memory_service.prepare_turn_memory",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ),
        ):
            ctx = await svc._prepare_turn(
                agent_id=1,
                message="hello",
                user_context={"user_id": 5, "app_id": 1, "oauth": True},
                db=MagicMock(),
            )

        assert ctx.memory_context is None
        assert ctx.global_memory_identity is None
        assert ctx.enhanced_message == "hello"

    @pytest.mark.asyncio
    async def test_no_read_this_turn_leaves_memory_context_none_even_with_identity(self, tmp_path):
        """A resolved identity but an unresolved/failed embeddings client
        (e.g. `should_read` gated it out inside `prepare_turn_memory`, or the
        embedding side failed to resolve) must not call `get_memory_context`
        at all, and the payload stays memory-free for this turn."""
        agent = _make_agent(has_global_memory=True)
        svc = _make_service(agent)

        identity = MagicMock()

        with (
            patch(
                "services.agent_execution_service.get_app_config",
                return_value={"TMP_BASE_FOLDER": str(tmp_path)},
            ),
            patch(
                "services.global_memory_service.prepare_turn_memory",
                new=AsyncMock(return_value=(identity, MagicMock(), MagicMock(), None)),
            ),
            patch(
                "services.global_memory_service.get_memory_context",
                new=AsyncMock(),
            ) as mock_get_context,
        ):
            ctx = await svc._prepare_turn(
                agent_id=1,
                message="hello",
                user_context={"user_id": 5, "app_id": 1, "oauth": True},
                db=MagicMock(),
            )

        assert ctx.memory_context is None
        mock_get_context.assert_not_called()


def _make_conversation(message_count: int, agent_id: int = 1, conversation_id: int = 99):
    """A `Conversation`-shaped mock carrying just what `_prepare_turn` and
    `turn_index()` read: `message_count` (the turn-index source, AD-13),
    `conversation_id` (effective_conv_id / working dir), and `session_id`
    (`_prepare_turn`'s `has_memory=True` branch derives the session suffix
    from it)."""
    conversation = MagicMock()
    conversation.message_count = message_count
    conversation.conversation_id = conversation_id
    conversation.session_id = f"conv_{agent_id}_abc"
    return conversation


class TestEveryTurnReadsForConversationalAgents:
    """The memory block rides in each turn's system prompt and is never
    stored in the thread (AD-7), so a `has_memory=True` agent resolves and
    reads on EVERY turn — including turns that are neither the first ones nor
    write-due. Regression: reads used to stop after turn 2 on the assumption
    that the block was carried forward in-thread; the model then ignored a
    remembered "Wants responses in English" from turn 4 on.

    Exercises the real `prepare_turn_memory` through the full `_prepare_turn`
    wiring; only its I/O-bound collaborators are mocked.
    """

    async def _prepare(self, svc, conversation, tmp_path, memory_block):
        identity, resolved, llm, embeddings = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        with (
            patch(
                "services.agent_execution_service.get_app_config",
                return_value={"TMP_BASE_FOLDER": str(tmp_path)},
            ),
            patch(
                "services.conversation_service.ConversationService.get_conversation",
                return_value=conversation,
            ),
            patch("services.global_memory_service.resolve_identity", return_value=identity),
            patch("services.global_memory_service.resolve_memory_services", return_value=resolved),
            patch(
                "services.global_memory_service.build_memory_clients",
                return_value=(llm, embeddings),
            ),
            patch(
                "services.global_memory_service.get_memory_context",
                new=AsyncMock(return_value=memory_block),
            ) as mock_get_context,
        ):
            ctx = await svc._prepare_turn(
                agent_id=1,
                message="hello",
                user_context={"user_id": 5, "app_id": 1, "oauth": True},
                conversation_id=99,
                db=MagicMock(),
            )
        return ctx, identity, mock_get_context

    @pytest.mark.asyncio
    async def test_a_mid_conversation_turn_reads_and_injects(self, tmp_path):
        agent = _make_agent(has_global_memory=True, has_memory=True)
        svc = _make_service(agent)

        conversation = _make_conversation(message_count=6)  # turn_index 4
        last_activity = datetime(2026, 10, 7, 8, 6)
        conversation.updated_at = last_activity

        ctx, identity, mock_get_context = await self._prepare(svc, conversation, tmp_path, "remembered block")

        assert ctx.memory_turn_index == 4
        assert ctx.global_memory_identity is identity
        mock_get_context.assert_awaited_once()
        assert ctx.memory_context == "remembered block"
        # Recency: the read knows when this conversation was last active and
        # which conversation it is (`new_since_last_message`).
        assert mock_get_context.await_args.kwargs["last_activity_at"] == last_activity
        assert mock_get_context.await_args.kwargs["conversation_id"] == 99

    @pytest.mark.asyncio
    async def test_a_write_due_turn_reads_and_keeps_its_clients_for_the_write(self, tmp_path):
        agent = _make_agent(has_global_memory=True, has_memory=True)
        svc = _make_service(agent)

        ctx, identity, mock_get_context = await self._prepare(
            svc, _make_conversation(message_count=8), tmp_path, "remembered block"  # turn_index 5
        )

        assert ctx.memory_turn_index == 5
        assert ctx.global_memory_identity is identity
        assert ctx.global_memory_llm is not None
        assert ctx.global_memory_embeddings is not None
        mock_get_context.assert_awaited_once()
