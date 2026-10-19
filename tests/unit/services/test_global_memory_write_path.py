"""Unit tests for the FR-5 write path in `services.global_memory_service`
(spec `user-memory-mem0`, step_008): `extract_facts`, `_build_extraction_window`,
`run_write_path` and `schedule_write_path`.

Every store/LLM collaborator is mocked — per NFR-8 the store seam
(`tools.memory.client`) is the single mockable point, so none of this needs a
real Postgres/pgvector schema. `GlobalMemoryClient.add`'s own idempotent-key
behaviour ("same fact twice -> one record") is proven against the real store
in `tests/integration/tools/test_memory_client_namespace_isolation.py`; this
module covers the write path's OWN policy layered on top of that seam:
scheduling, extraction bounds, the dedup stage, and fault containment.
"""
import asyncio
from types import SimpleNamespace
import json
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import services.global_memory_service as global_memory_service
from services.global_memory_service import (
    EXTRACTION_PROMPT,
    _build_extraction_window,
    buffer_pending_user_message,
    take_pending_user_messages,
    extract_facts,
    run_write_path,
    schedule_write_path,
)
from tools.memory import settings as memory_settings
from tools.memory.client import ClosestFact
from tools.memory.identity import MemoryIdentity
from tools.memory.resolution import ResolvedMemoryServices


def _identity(app_id: int = 1, user_id: int = 7, agent_id: int = 3) -> MemoryIdentity:
    root = memory_settings.namespace_root()
    return MemoryIdentity(
        app_id=app_id, user_id=user_id, agent_id=agent_id,
        namespace=(root, str(app_id), str(user_id)),
    )


def _resolved(extraction_is_system: bool = False) -> ResolvedMemoryServices:
    return ResolvedMemoryServices(
        extraction_service_id=10,
        extraction_source="app_config",
        extraction_is_system=extraction_is_system,
        embedding_service_id=20,
        embedding_source="app_config",
        embedding_is_system=False,
    )


def _llm_returning(content: str) -> AsyncMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content=content))
    return llm


@pytest.fixture(autouse=True)
def _clear_write_tasks():
    """The module-level `_write_tasks` set must never leak state between
    tests — a task left over from a previous test would corrupt the
    back-pressure bound assertions below."""
    global_memory_service._write_tasks.clear()
    yield
    global_memory_service._write_tasks.clear()


# ---------------------------------------------------------------------------
# AC-51: the extraction prompt is a code constant, nothing else
# ---------------------------------------------------------------------------


class TestExtractionPromptIsAConstant:
    def test_prompt_is_a_plain_module_level_string(self):
        assert isinstance(EXTRACTION_PROMPT, str)
        assert len(EXTRACTION_PROMPT) > 0

    def test_extract_facts_never_accepts_a_prompt_override(self):
        """No parameter on the public write-path functions can substitute a
        different prompt — the only inputs are the LLM client and the
        (already-cleaned) window text."""
        import inspect

        sig = inspect.signature(extract_facts)
        assert list(sig.parameters) == ["llm", "window_text"]


# ---------------------------------------------------------------------------
# _build_extraction_window (D-23, AC-60)
# ---------------------------------------------------------------------------


class TestExtractionPromptContent:
    """Regression for a fact inferred from the language the user wrote in
    ("Communicates in Spanish") overriding their stated preference."""

    def test_records_standing_response_instructions_as_preferences(self):
        assert "always answer me in English" in EXTRACTION_PROMPT
        assert "Wants responses in English" in EXTRACTION_PROMPT

    def test_forbids_inferring_facts_from_the_language_the_user_writes_in(self):
        """Checked by meaning, not one exact sentence, so the prompt's wording
        can evolve: inferring from the language/style must stay excluded."""
        prompt = EXTRACTION_PROMPT.lower()
        assert "infer" in prompt
        assert "language" in prompt and ("writing style" in prompt or "tone" in prompt)

    def test_user_messages_are_the_authoritative_source(self):
        assert "The user's own message is the primary and authoritative source" in EXTRACTION_PROMPT

    def test_never_extracts_shared_document_content(self):
        assert "Never extract the content of documents" in EXTRACTION_PROMPT

    def test_carries_few_shot_examples_including_empty_outputs(self):
        assert EXTRACTION_PROMPT.count('"memory": []') >= 3
        assert '"text": "Wants responses in English"' in EXTRACTION_PROMPT

    def test_extends_and_updates_must_write_the_complete_combined_fact(self):
        """Regression: "Trabaja en IKERLAN" extended by "trabajo en proyectos de
        IA" came back as "Trabaja en proyectos de IA"; since extends replaces
        the linked memory, "Ikerlan" was lost."""
        assert "COMPLETE REPLACEMENT TEXT" in EXTRACTION_PROMPT
        assert '"Works on AI projects at Ikerlan"' in EXTRACTION_PROMPT
        assert "is REMOVED and replaced by your new memory" in EXTRACTION_PROMPT

    def test_never_carries_mem0s_deceptive_data_source_instruction(self):
        assert "publicly available sources" not in EXTRACTION_PROMPT


class TestBuildExtractionWindow:
    def test_includes_user_and_assistant_text_when_no_tool_call(self):
        window = _build_extraction_window("I like pizza", "Noted!", turn_used_tool_call=False)
        assert "User: I like pizza" in window
        assert "Assistant: Noted!" in window

    def test_excludes_assistant_text_when_turn_used_a_tool_call(self):
        """D-23: an assistant turn that followed a tool call routinely quotes
        retrieved/tool content verbatim and must never enter the window."""
        window = _build_extraction_window(
            "What's in the doc?", "The document says the secret is 42", turn_used_tool_call=True
        )
        assert "User: What's in the doc?" in window
        assert "secret is 42" not in window
        assert "Assistant:" not in window

    def test_never_receives_the_attached_files_segment_or_base_folder_line(self):
        """AC-60: the caller must pass the ORIGINAL user message
        (`ctx.original_user_message`), never `ctx.enhanced_message` — this
        function has no way to strip anything, so its output is only clean
        because its input already is. This test documents that contract at
        the unit level; the end-to-end proof (that `ctx.original_user_message`
        really is attachment-free) lives in
        `test_agent_execution_service_write_scheduling.py`.
        """
        clean_original_message = "Please review my notes"
        window = _build_extraction_window(clean_original_message, "Sure.", turn_used_tool_call=False)
        assert "[Attached files:]" not in window
        assert "Files base folder is:" not in window

    def test_empty_inputs_produce_an_empty_window(self):
        assert _build_extraction_window("", "", turn_used_tool_call=False) == ""
        assert _build_extraction_window(None, None, turn_used_tool_call=False) == ""

    def test_excludes_assistant_text_when_turn_had_attachments_even_without_a_tool_call(self):
        """HIGH (step_008 audit): uploading a file is NOT a tool call — its
        text is inlined into `enhanced_message` and answered inline, so
        `turn_used_tool_call` stays `False` even though the assistant's reply
        routinely restates the attachment's contents (a CV, a medical
        report, a contract). `turn_had_attachments` must independently gate
        the assistant half of the window."""
        window = _build_extraction_window(
            "Resume este CV",
            "Trabaja como ingeniero, diagnosticado de X, DNI 12345678X",
            turn_used_tool_call=False,
            turn_had_attachments=True,
        )
        assert "User: Resume este CV" in window
        assert "Assistant:" not in window
        assert "12345678X" not in window
        assert "diagnosticado" not in window

    def test_includes_assistant_text_when_no_tool_call_and_no_attachments(self):
        """The default (`turn_had_attachments=False`) must not change
        existing, already-safe behaviour for a plain text-only turn."""
        window = _build_extraction_window(
            "Resume este CV", "Trabaja como ingeniero", turn_used_tool_call=False,
        )
        assert "Assistant: Trabaja como ingeniero" in window

    def test_excludes_assistant_text_when_both_signals_are_true(self):
        window = _build_extraction_window(
            "Resume este CV y busca en la web",
            "El documento dice X",
            turn_used_tool_call=True,
            turn_had_attachments=True,
        )
        assert "Assistant:" not in window


# ---------------------------------------------------------------------------
# extract_facts (AD-8 stage 1, FR-5, R-20)
# ---------------------------------------------------------------------------


class TestExtractionWindowWithEarlierTurns:
    """D-15: a write-due turn extracts from every turn since the previous
    extraction — earlier turns contribute their user messages only."""

    def test_earlier_user_messages_precede_the_current_turn_oldest_first(self):
        window = _build_extraction_window(
            "Actually, give me detailed answers", "Sure.", False,
            earlier_user_messages=["turn two", "turn three", "turn four"],
        )

        assert window == (
            "User (earlier turn): turn two\n"
            "User (earlier turn): turn three\n"
            "User (earlier turn): turn four\n"
            "User: Actually, give me detailed answers\n"
            "Assistant: Sure."
        )

    def test_over_budget_drops_the_oldest_earlier_messages_never_the_current_turn(self, monkeypatch):
        monkeypatch.setattr(memory_settings, "extraction_input_max_chars", lambda: 80)
        window = _build_extraction_window(
            "current message", "", False,
            earlier_user_messages=["a" * 40, "newest earlier"],
        )

        assert window == "User (earlier turn): newest earlier\nUser: current message"

    def test_a_long_assistant_reply_never_pushes_out_the_users_messages(self, monkeypatch):
        """Regression: a 4.5k-char code answer used to consume the whole budget
        and drop every pending user message, so "prefiero que sean en python"
        never reached the extractor."""
        monkeypatch.setattr(memory_settings, "extraction_input_max_chars", lambda: 4000)
        window = _build_extraction_window(
            "hazme un ejemplo de api rest", "x" * 4500, False,
            earlier_user_messages=["trabajo en proyectos de IA", "prefiero que sean en python"],
        )

        assert "User (earlier turn): trabajo en proyectos de IA" in window
        assert "User (earlier turn): prefiero que sean en python" in window
        assert "User: hazme un ejemplo de api rest" in window
        assert len(window) <= 4000
        assistant_line = window.splitlines()[-1]
        assert assistant_line.startswith("Assistant: ") and assistant_line.endswith("…")

    def test_no_room_left_drops_the_assistant_reply_entirely(self, monkeypatch):
        monkeypatch.setattr(memory_settings, "extraction_input_max_chars", lambda: 60)
        window = _build_extraction_window("current message", "a long reply " * 20, False)

        assert window == "User: current message"

    def test_blank_earlier_messages_are_skipped(self):
        window = _build_extraction_window("now", "", False, earlier_user_messages=["", "  "])
        assert window == "User: now"

    def test_the_prompt_explains_earlier_turns_and_that_the_later_line_wins(self):
        assert "User (earlier turn):" in EXTRACTION_PROMPT
        assert "the later one wins" in EXTRACTION_PROMPT


class TestPendingUserMessageBuffer:
    def _conversation(self, pending=None):
        return SimpleNamespace(memory_pending_user_messages=pending)

    def test_buffers_messages_and_keeps_only_the_turns_between_two_extractions(self, monkeypatch):
        monkeypatch.setattr(memory_settings, "write_every_n_turns", lambda: 5)
        conversation, db = self._conversation(), MagicMock()

        for turn in range(2, 8):
            buffer_pending_user_message(db, conversation, f"turn {turn}")

        assert json.loads(conversation.memory_pending_user_messages) == ["turn 4", "turn 5", "turn 6", "turn 7"]
        assert db.commit.call_count == 6

    def test_take_returns_the_messages_and_clears_the_buffer(self):
        conversation, db = self._conversation(json.dumps(["turn 2", "turn 3"])), MagicMock()

        assert take_pending_user_messages(db, conversation) == ["turn 2", "turn 3"]
        assert conversation.memory_pending_user_messages is None
        db.commit.assert_called_once()

    def test_take_on_an_empty_buffer_does_not_touch_the_session(self):
        conversation, db = self._conversation(None), MagicMock()

        assert take_pending_user_messages(db, conversation) == []
        db.commit.assert_not_called()

    def test_a_corrupt_buffer_reads_as_empty(self):
        conversation = self._conversation("not json")
        assert take_pending_user_messages(MagicMock(), conversation) == []

    def test_buffering_failure_rolls_back_and_never_raises(self):
        conversation, db = self._conversation(), MagicMock()
        db.commit.side_effect = RuntimeError("db down")

        buffer_pending_user_message(db, conversation, "hello")

        db.rollback.assert_called_once()


class TestExtractFacts:
    @pytest.mark.asyncio
    async def test_no_llm_returns_empty_list_without_any_call(self):
        assert await extract_facts(None, "some window text") == []

    @pytest.mark.asyncio
    async def test_empty_window_returns_empty_list_without_calling_the_llm(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
        assert await extract_facts(llm, "   ") == []
        llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_parses_a_plain_json_array(self):
        llm = _llm_returning(json.dumps(["Likes concise answers", "Works as a backend engineer"]))
        facts = await extract_facts(llm, "User: I like short answers. I'm a backend engineer.")
        assert facts == ["Likes concise answers", "Works as a backend engineer"]

    @pytest.mark.asyncio
    async def test_tolerates_a_markdown_code_fence_around_the_json(self):
        llm = _llm_returning('```json\n["Prefers dark mode"]\n```')
        facts = await extract_facts(llm, "User: I prefer dark mode.")
        assert facts == ["Prefers dark mode"]

    @pytest.mark.asyncio
    async def test_empty_array_response_is_a_normal_no_facts_outcome(self):
        llm = _llm_returning("[]")
        assert await extract_facts(llm, "User: hi") == []

    @pytest.mark.asyncio
    async def test_unparseable_response_is_treated_as_no_facts_not_an_error(self):
        llm = _llm_returning("not json at all")
        assert await extract_facts(llm, "User: hi") == []

    @pytest.mark.asyncio
    async def test_non_string_array_items_are_dropped(self):
        llm = _llm_returning(json.dumps(["a real fact", 42, None, {"nested": "obj"}]))
        assert await extract_facts(llm, "User: hi") == ["a real fact"]

    @pytest.mark.asyncio
    async def test_llm_raising_returns_empty_list_never_raises(self):
        """AC-14 (extraction side)."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("provider is down"))
        assert await extract_facts(llm, "User: hi") == []

    @pytest.mark.asyncio
    async def test_caps_fact_count_at_max_facts_per_extraction(self, monkeypatch):
        monkeypatch.setattr(memory_settings, "max_facts_per_extraction", lambda: 2)
        llm = _llm_returning(json.dumps(["fact one", "fact two", "fact three", "fact four"]))
        facts = await extract_facts(llm, "User: lots of facts")
        assert facts == ["fact one", "fact two"]

    @pytest.mark.asyncio
    async def test_caps_each_fact_length_at_fact_max_chars(self, monkeypatch):
        monkeypatch.setattr(memory_settings, "fact_max_chars", lambda: 5)
        llm = _llm_returning(json.dumps(["a very long fact indeed"]))
        facts = await extract_facts(llm, "User: hi")
        assert facts == ["a ver"]

    @pytest.mark.asyncio
    async def test_input_is_truncated_before_reaching_the_llm(self, monkeypatch):
        monkeypatch.setattr(memory_settings, "extraction_input_max_chars", lambda: 10)
        llm = _llm_returning("[]")
        await extract_facts(llm, "0123456789ABCDEFGHIJ")

        sent_messages = llm.ainvoke.call_args.args[0]
        human_message = sent_messages[-1]
        assert human_message.content.endswith("\n\n0123456789")
        assert "ABCDEF" not in human_message.content

    @pytest.mark.asyncio
    async def test_input_carries_the_observation_date_for_relative_time_references(self):
        llm = _llm_returning('{"memory": []}')
        await extract_facts(llm, "User: next month I start a new job")

        human_message = llm.ainvoke.call_args.args[0][-1]
        assert re.match(r"^Observation Date: \d{4}-\d{2}-\d{2}\n\n", human_message.content)
        assert human_message.content.endswith("User: next month I start a new job")

    @pytest.mark.asyncio
    async def test_parses_the_mem0_style_memory_object(self):
        llm = _llm_returning(json.dumps({"memory": [{"text": "Wants responses in English"}, {"text": "Uses Python 3.11"}]}))
        assert await extract_facts(llm, "User: hi") == ["Wants responses in English", "Uses Python 3.11"]

    @pytest.mark.asyncio
    async def test_parses_a_memory_object_holding_plain_strings(self):
        llm = _llm_returning('```json\n{"memory": ["Prefers dark mode"]}\n```')
        assert await extract_facts(llm, "User: hi") == ["Prefers dark mode"]

    @pytest.mark.asyncio
    async def test_an_empty_memory_object_is_a_normal_no_facts_outcome(self):
        llm = _llm_returning('{"memory": []}')
        assert await extract_facts(llm, "User: hi") == []

    @pytest.mark.asyncio
    async def test_memory_items_without_usable_text_are_dropped(self):
        llm = _llm_returning(json.dumps({"memory": [{"text": "a real fact"}, {"text": ""}, {"id": "1"}, 7]}))
        assert await extract_facts(llm, "User: hi") == ["a real fact"]

    @pytest.mark.asyncio
    async def test_system_message_is_the_module_constant_prompt(self):
        llm = _llm_returning("[]")
        await extract_facts(llm, "User: hi")

        sent_messages = llm.ainvoke.call_args.args[0]
        system_message = sent_messages[0]
        assert system_message.content == EXTRACTION_PROMPT


# ---------------------------------------------------------------------------
# run_write_path (AD-8 stages 0-3, D-6, D-10, D-23)
# ---------------------------------------------------------------------------


class TestRunWritePathFactExtractionOutcomes:
    @pytest.mark.asyncio
    async def test_no_facts_means_zero_store_calls(self):
        """AC-15."""
        identity = _identity()
        resolved = _resolved()

        with (
            patch.object(global_memory_service, "extract_memories", new=AsyncMock(return_value=[])),
            patch.object(global_memory_service, "get_memory_client", new=AsyncMock()) as mock_get_client,
        ):
            await run_write_path(
                identity=identity, resolved=resolved, llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1, original_user_message="hi", response_text="hello",
                turn_used_tool_call=False,
            )

        # The client is fetched before extraction (to look up similar
        # memories), but nothing is written or deleted.
        client = mock_get_client.return_value
        client.add.assert_not_called()
        client.delete_keys.assert_not_called()

    @pytest.mark.asyncio
    async def test_only_fact_strings_cross_to_the_persistence_call(self):
        """AC-16."""
        identity = _identity()
        resolved = _resolved()
        client = MagicMock()
        client.closest = AsyncMock(return_value=None)
        client.add = AsyncMock(return_value=2)

        with (
            patch.object(
                global_memory_service, "extract_memories",
                new=AsyncMock(return_value=["Likes pizza", "Works remotely"]),
            ),
            patch.object(global_memory_service, "get_memory_client", new=AsyncMock(return_value=client)),
        ):
            await run_write_path(
                identity=identity, resolved=resolved, llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1,
                original_user_message="I like pizza and work remotely",
                response_text="Noted!",
                turn_used_tool_call=False,
            )

        client.add.assert_awaited_once_with(identity, ["Likes pizza", "Works remotely"], conversation_id=None)


class TestRunWritePathBuildsItsOwnClients:
    """When the turn's read-budgeted client construction timed out,
    `prepare_turn_memory` hands over no clients and the write builds them."""

    @pytest.mark.asyncio
    async def test_missing_clients_are_built_in_a_fresh_session_then_used(self):
        identity = _identity()
        resolved = _resolved()
        llm, embeddings = MagicMock(), MagicMock()
        client = MagicMock()
        client.closest = AsyncMock(return_value=None)
        client.add = AsyncMock(return_value=1)

        with (
            patch.object(
                global_memory_service, "_build_memory_clients_sync", return_value=(llm, embeddings)
            ) as mock_build,
            patch.object(
                global_memory_service, "extract_memories", new=AsyncMock(return_value=["Wants responses in English"])
            ) as mock_extract,
            patch.object(global_memory_service, "get_memory_client", new=AsyncMock(return_value=client)) as mock_get,
        ):
            await run_write_path(
                identity=identity, resolved=resolved, llm=None, embeddings=None,
                embedding_service_id=1, original_user_message="answer me always in English",
                response_text="Sure!", turn_used_tool_call=False,
            )

        mock_build.assert_called_once_with(resolved)
        assert mock_extract.await_args.args[0] is llm
        assert mock_get.await_args.args[0] is embeddings
        client.add.assert_awaited_once_with(identity, ["Wants responses in English"], conversation_id=None)

    @pytest.mark.asyncio
    async def test_clients_that_still_cannot_be_built_skip_the_write_without_raising(self):
        with (
            patch.object(global_memory_service, "_build_memory_clients_sync", return_value=(None, None)),
            patch.object(global_memory_service, "extract_memories", new=AsyncMock()) as mock_extract,
        ):
            await run_write_path(
                identity=_identity(), resolved=_resolved(), llm=None, embeddings=None,
                embedding_service_id=1, original_user_message="hi", response_text="hello",
                turn_used_tool_call=False,
            )

        mock_extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_clients_handed_over_by_the_turn_are_never_rebuilt(self):
        with (
            patch.object(global_memory_service, "_build_memory_clients_sync") as mock_build,
            patch.object(global_memory_service, "extract_memories", new=AsyncMock(return_value=[])),
        ):
            await run_write_path(
                identity=_identity(), resolved=_resolved(), llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1, original_user_message="hi", response_text="hello",
                turn_used_tool_call=False,
            )

        mock_build.assert_not_called()


class TestRunWritePathForwardsAttachmentSignal:
    @pytest.mark.asyncio
    async def test_turn_had_attachments_reaches_build_extraction_window(self):
        """HIGH (step_008 audit): `run_write_path` must forward
        `turn_had_attachments` through to `_build_extraction_window`, not
        only `turn_used_tool_call`."""
        identity = _identity()
        resolved = _resolved()

        with (
            patch.object(
                global_memory_service, "_build_extraction_window",
                wraps=global_memory_service._build_extraction_window,
            ) as mock_build_window,
            patch.object(global_memory_service, "extract_memories", new=AsyncMock(return_value=[])),
        ):
            await run_write_path(
                identity=identity, resolved=resolved, llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1,
                original_user_message="Resume este CV",
                response_text="Trabaja como ingeniero, DNI 12345678X",
                turn_used_tool_call=False,
                turn_had_attachments=True,
            )

        mock_build_window.assert_called_once_with(
            "Resume este CV",
            "Trabaja como ingeniero, DNI 12345678X",
            False,
            True,
            earlier_user_messages=(),
        )


class TestRunWritePathFailureContainment:
    @pytest.mark.asyncio
    async def test_extraction_raising_never_propagates(self):
        """AC-14."""
        identity = _identity()
        resolved = _resolved()

        with patch.object(
            global_memory_service, "extract_memories", new=AsyncMock(side_effect=RuntimeError("boom"))
        ):
            await run_write_path(
                identity=identity, resolved=resolved, llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1, original_user_message="hi", response_text="hi",
                turn_used_tool_call=False,
            )
        # No exception escaped — the assertion is simply that this line was reached.

    @pytest.mark.asyncio
    async def test_store_write_raising_never_propagates(self):
        """AC-14 (store side)."""
        identity = _identity()
        resolved = _resolved()
        client = MagicMock()
        client.closest = AsyncMock(return_value=None)
        client.add = AsyncMock(side_effect=RuntimeError("store unreachable"))

        with (
            patch.object(global_memory_service, "extract_memories", new=AsyncMock(return_value=["a fact"])),
            patch.object(global_memory_service, "get_memory_client", new=AsyncMock(return_value=client)),
        ):
            await run_write_path(
                identity=identity, resolved=resolved, llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1, original_user_message="hi", response_text="hi",
                turn_used_tool_call=False,
            )


class TestRunWritePathDedup:
    """AC-47 (revised): the most recent wording wins. A candidate scoring above
    the threshold against an existing fact is still written, and the older
    fact is deleted once that write lands."""

    async def _run(self, client, facts, conversation_id=None):
        with (
            patch.object(global_memory_service, "extract_memories", new=AsyncMock(return_value=facts)),
            patch.object(global_memory_service, "get_memory_client", new=AsyncMock(return_value=client)),
        ):
            await run_write_path(
                identity=_identity(), resolved=_resolved(), llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1, original_user_message="hi", response_text="hi",
                turn_used_tool_call=False, conversation_id=conversation_id,
            )

    def _client(self, closest, written=1):
        client = MagicMock()
        client.closest = AsyncMock(side_effect=closest) if isinstance(closest, (list, Exception)) else AsyncMock(return_value=closest)
        client.add = AsyncMock(return_value=written)
        client.delete_keys = AsyncMock(return_value=1)
        return client

    @pytest.mark.asyncio
    async def test_a_near_duplicate_replaces_the_older_fact(self, monkeypatch):
        monkeypatch.setattr(memory_settings, "dedup_threshold", lambda: 0.95)
        old = ClosestFact(score=0.955, key="a" * 32, same_fact=False)
        client = self._client([None, old], written=2)

        await self._run(client, ["new fact", "Wants responses in English."], conversation_id=76)

        client.add.assert_awaited_once_with(_identity(), ["new fact", "Wants responses in English."], conversation_id=76)
        client.delete_keys.assert_awaited_once_with(_identity(), ["a" * 32])

    @pytest.mark.asyncio
    async def test_identical_text_overwrites_in_place_without_a_delete(self, monkeypatch):
        monkeypatch.setattr(memory_settings, "dedup_threshold", lambda: 0.95)
        client = self._client(ClosestFact(score=1.0, key="b" * 32, same_fact=True))

        await self._run(client, ["Wants responses in English"])

        client.add.assert_awaited_once()
        client.delete_keys.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_below_the_threshold_nothing_is_replaced(self, monkeypatch):
        monkeypatch.setattr(memory_settings, "dedup_threshold", lambda: 0.95)
        client = self._client(ClosestFact(score=0.835, key="c" * 32, same_fact=False))

        await self._run(client, ["Wants responses in Spanish"])

        client.add.assert_awaited_once()
        client.delete_keys.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_older_fact_survives_when_the_write_fails(self, monkeypatch):
        """Never lose a fact: the delete only runs after a successful write."""
        monkeypatch.setattr(memory_settings, "dedup_threshold", lambda: 0.95)
        client = self._client(ClosestFact(score=0.99, key="d" * 32, same_fact=False), written=0)

        await self._run(client, ["new wording"])

        client.delete_keys.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dedup_stage_failure_writes_every_candidate_and_replaces_none(self):
        """'Losing a fact is worse than storing a near-duplicate' (AD-8)."""
        client = self._client(RuntimeError("embedding provider down"), written=2)

        await self._run(client, ["fact one", "fact two"])

        client.add.assert_awaited_once_with(_identity(), ["fact one", "fact two"], conversation_id=None)
        client.delete_keys.assert_not_awaited()


class TestRunWritePathUsageAndQuota:
    """AC-39."""

    @pytest.mark.asyncio
    async def test_uses_a_fresh_session_and_never_the_request_scoped_one(self, monkeypatch):
        quota_session = MagicMock(name="quota-session")
        usage_session = MagicMock(name="usage-session")
        session_factory = MagicMock(side_effect=[quota_session, usage_session])
        monkeypatch.setattr("db.database.SessionLocal", session_factory)

        identity = _identity()
        resolved = _resolved(extraction_is_system=True)
        client = MagicMock()
        client.closest = AsyncMock(return_value=None)
        client.add = AsyncMock(return_value=1)

        with (
            patch.object(global_memory_service, "extract_memories", new=AsyncMock(return_value=["a fact"])),
            patch.object(global_memory_service, "get_memory_client", new=AsyncMock(return_value=client)),
            patch(
                "services.tier_enforcement_service.TierEnforcementService.check_system_llm_quota"
            ) as mock_quota,
            patch(
                "services.usage_tracking_service.UsageTrackingService.record_system_llm_call"
            ) as mock_record,
        ):
            await run_write_path(
                identity=identity, resolved=resolved, llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1, original_user_message="hi", response_text="hi",
                turn_used_tool_call=False,
            )

        mock_quota.assert_called_once_with(quota_session, identity.user_id)
        mock_record.assert_called_once_with(usage_session, identity.user_id)
        quota_session.close.assert_called_once()
        usage_session.close.assert_called_once()
        # Two DISTINCT freshly-opened sessions were used — never a single
        # session held open across the LLM call, and never anything supplied
        # by the caller (this function accepts no `db`/session parameter at
        # all, so there is nothing for it to reuse even by mistake).
        assert quota_session is not usage_session

    @pytest.mark.asyncio
    async def test_quota_exhausted_skips_extraction_entirely_without_raising(self, monkeypatch):
        quota_session = MagicMock(name="quota-session")
        monkeypatch.setattr("db.database.SessionLocal", MagicMock(return_value=quota_session))

        identity = _identity()
        resolved = _resolved(extraction_is_system=True)

        with (
            patch(
                "services.tier_enforcement_service.TierEnforcementService.check_system_llm_quota",
                side_effect=HTTPException(status_code=429, detail="quota exhausted"),
            ),
            patch.object(global_memory_service, "extract_memories", new=AsyncMock()) as mock_extract,
            patch.object(global_memory_service, "get_memory_client", new=AsyncMock()) as mock_get_client,
        ):
            await run_write_path(
                identity=identity, resolved=resolved, llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1, original_user_message="hi", response_text="hi",
                turn_used_tool_call=False,
            )

        mock_extract.assert_not_called()
        mock_get_client.assert_not_called()
        quota_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_own_key_app_never_checks_quota_or_records_usage(self, monkeypatch):
        """A non-system (App-owned-key) extraction service must never touch
        the quota machinery at all — mirrors the existing `_finalize_turn`
        step 6 precedent for the main response LLM."""
        session_factory = MagicMock()
        monkeypatch.setattr("db.database.SessionLocal", session_factory)

        identity = _identity()
        resolved = _resolved(extraction_is_system=False)
        client = MagicMock()
        client.closest = AsyncMock(return_value=None)
        client.add = AsyncMock(return_value=1)

        with (
            patch.object(global_memory_service, "extract_memories", new=AsyncMock(return_value=["a fact"])),
            patch.object(global_memory_service, "get_memory_client", new=AsyncMock(return_value=client)),
            patch(
                "services.tier_enforcement_service.TierEnforcementService.check_system_llm_quota"
            ) as mock_quota,
            patch(
                "services.usage_tracking_service.UsageTrackingService.record_system_llm_call"
            ) as mock_record,
        ):
            await run_write_path(
                identity=identity, resolved=resolved, llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1, original_user_message="hi", response_text="hi",
                turn_used_tool_call=False,
            )

        mock_quota.assert_not_called()
        mock_record.assert_not_called()
        session_factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_zero_facts_still_records_usage(self, monkeypatch):
        """MEDIUM (step_008 audit, D-10): the extraction LLM call already
        happened — and was billed by the provider — whether or not it found
        any durable facts. Usage must be recorded on a `SKIP_NO_FACTS`
        outcome, not only after a successful `client.add`."""
        quota_session = MagicMock(name="quota-session")
        usage_session = MagicMock(name="usage-session")
        session_factory = MagicMock(side_effect=[quota_session, usage_session])
        monkeypatch.setattr("db.database.SessionLocal", session_factory)

        identity = _identity()
        resolved = _resolved(extraction_is_system=True)

        with (
            patch.object(global_memory_service, "extract_memories", new=AsyncMock(return_value=[])),
            patch.object(global_memory_service, "get_memory_client", new=AsyncMock()) as mock_get_client,
            patch(
                "services.tier_enforcement_service.TierEnforcementService.check_system_llm_quota"
            ) as mock_quota,
            patch(
                "services.usage_tracking_service.UsageTrackingService.record_system_llm_call"
            ) as mock_record,
        ):
            await run_write_path(
                identity=identity, resolved=resolved, llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1, original_user_message="hi", response_text="hi",
                turn_used_tool_call=False,
            )

        mock_quota.assert_called_once_with(quota_session, identity.user_id)
        mock_record.assert_called_once_with(usage_session, identity.user_id)
        usage_session.commit.assert_called_once()
        usage_session.close.assert_called_once()
        # No facts survived extraction, so nothing is written to the store.
        mock_get_client.return_value.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_usage_session_is_committed_before_being_closed(self, monkeypatch):
        """MEDIUM (step_008 audit, D-10): `UsageRecordRepository.increment`
        only `flush()`es — a background task that closes its session without
        committing silently rolls the row back. `run_write_path` must commit
        the usage session itself before closing it."""
        quota_session = MagicMock(name="quota-session")
        usage_session = MagicMock(name="usage-session")
        session_factory = MagicMock(side_effect=[quota_session, usage_session])
        monkeypatch.setattr("db.database.SessionLocal", session_factory)

        call_order = []
        usage_session.commit.side_effect = lambda: call_order.append("commit")
        usage_session.close.side_effect = lambda: call_order.append("close")

        identity = _identity()
        resolved = _resolved(extraction_is_system=True)
        client = MagicMock()
        client.closest = AsyncMock(return_value=None)
        client.add = AsyncMock(return_value=1)

        with (
            patch.object(global_memory_service, "extract_memories", new=AsyncMock(return_value=["a fact"])),
            patch.object(global_memory_service, "get_memory_client", new=AsyncMock(return_value=client)),
            patch("services.tier_enforcement_service.TierEnforcementService.check_system_llm_quota"),
            patch("services.usage_tracking_service.UsageTrackingService.record_system_llm_call"),
        ):
            await run_write_path(
                identity=identity, resolved=resolved, llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1, original_user_message="hi", response_text="hi",
                turn_used_tool_call=False,
            )

        usage_session.commit.assert_called_once()
        assert call_order == ["commit", "close"]

    @pytest.mark.asyncio
    async def test_usage_recording_failure_rolls_back_and_never_propagates(self, monkeypatch):
        """AC-14 mirrored for the usage-recording stage: a DB error here must
        degrade quietly, and must not leave the session dangling without a
        rollback."""
        quota_session = MagicMock(name="quota-session")
        usage_session = MagicMock(name="usage-session")
        session_factory = MagicMock(side_effect=[quota_session, usage_session])
        monkeypatch.setattr("db.database.SessionLocal", session_factory)

        identity = _identity()
        resolved = _resolved(extraction_is_system=True)

        with (
            patch.object(global_memory_service, "extract_memories", new=AsyncMock(return_value=[])),
            patch("services.tier_enforcement_service.TierEnforcementService.check_system_llm_quota"),
            patch(
                "services.usage_tracking_service.UsageTrackingService.record_system_llm_call",
                side_effect=RuntimeError("db unreachable"),
            ),
        ):
            await run_write_path(
                identity=identity, resolved=resolved, llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1, original_user_message="hi", response_text="hi",
                turn_used_tool_call=False,
            )
        # No exception escaped — reaching this line is the assertion.
        usage_session.rollback.assert_called_once()
        usage_session.close.assert_called_once()


# ---------------------------------------------------------------------------
# schedule_write_path (AD-8, D-6, R-8, NFR-7, AC-13/AC-17)
# ---------------------------------------------------------------------------


class TestScheduleWritePath:
    @pytest.mark.asyncio
    async def test_returns_immediately_without_waiting_on_the_task(self):
        """AC-13: `schedule_write_path` itself is synchronous and must not
        block on the scheduled coroutine."""
        started = asyncio.Event()
        may_finish = asyncio.Event()

        async def _slow_run_write_path(**kwargs):
            started.set()
            await may_finish.wait()

        identity = _identity()
        resolved = _resolved()

        with patch.object(global_memory_service, "run_write_path", new=_slow_run_write_path):
            schedule_write_path(
                identity=identity, resolved=resolved, llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1, original_user_message="hi", response_text="hi",
                turn_used_tool_call=False,
            )
            # The call above returned synchronously; the task exists but has
            # not necessarily started running yet on this event loop.
            assert len(global_memory_service._write_tasks) == 1

            await asyncio.wait_for(started.wait(), timeout=1.0)
            # The task is running (proven by `started` being set), but the
            # scheduling call above returned long before this point and
            # never itself waited for `may_finish` — proving it did not
            # block on the task it created.
            assert started.is_set()
            assert not may_finish.is_set()

            may_finish.set()
            # Let the task drain so it does not leak into the next test.
            await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_timeout_abandons_the_task_without_raising(self, monkeypatch):
        """AC-17."""
        monkeypatch.setattr(memory_settings, "write_timeout_s", lambda: 0.01)

        async def _hangs_forever(**kwargs):
            await asyncio.sleep(10)

        identity = _identity()
        resolved = _resolved()

        with patch.object(global_memory_service, "run_write_path", new=_hangs_forever):
            schedule_write_path(
                identity=identity, resolved=resolved, llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1, original_user_message="hi", response_text="hi",
                turn_used_tool_call=False,
            )
            task = next(iter(global_memory_service._write_tasks))
            await asyncio.wait_for(task, timeout=2.0)
            # No exception propagated out of the task itself.
            assert task.exception() is None

    @pytest.mark.asyncio
    async def test_backpressure_bound_skips_scheduling_when_at_capacity(self, monkeypatch):
        monkeypatch.setattr(memory_settings, "max_inflight_writes", lambda: 1)
        # Pre-fill the task set with a dummy in-flight task.
        dummy_task = asyncio.create_task(asyncio.sleep(10))
        global_memory_service._write_tasks.add(dummy_task)

        identity = _identity()
        resolved = _resolved()

        with patch.object(
            global_memory_service, "run_write_path", new=AsyncMock()
        ) as mock_run:
            schedule_write_path(
                identity=identity, resolved=resolved, llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1, original_user_message="hi", response_text="hi",
                turn_used_tool_call=False,
            )

        # No new task was added — only the dummy one remains.
        assert len(global_memory_service._write_tasks) == 1
        mock_run.assert_not_called()

        dummy_task.cancel()
        try:
            await dummy_task
        except asyncio.CancelledError:
            pass
