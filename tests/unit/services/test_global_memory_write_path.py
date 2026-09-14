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
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import services.global_memory_service as global_memory_service
from services.global_memory_service import (
    EXTRACTION_PROMPT,
    _build_extraction_window,
    extract_facts,
    run_write_path,
    schedule_write_path,
)
from tools.memory import settings as memory_settings
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
        assert human_message.content == "0123456789"

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
            patch.object(global_memory_service, "extract_facts", new=AsyncMock(return_value=[])),
            patch.object(global_memory_service, "get_memory_client", new=AsyncMock()) as mock_get_client,
        ):
            await run_write_path(
                identity=identity, resolved=resolved, llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1, original_user_message="hi", response_text="hello",
                turn_used_tool_call=False,
            )

        mock_get_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_only_fact_strings_cross_to_the_persistence_call(self):
        """AC-16."""
        identity = _identity()
        resolved = _resolved()
        client = MagicMock()
        client.similarity = AsyncMock(return_value=None)
        client.add = AsyncMock(return_value=2)

        with (
            patch.object(
                global_memory_service, "extract_facts",
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

        client.add.assert_awaited_once_with(identity, ["Likes pizza", "Works remotely"])


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
            patch.object(global_memory_service, "extract_facts", new=AsyncMock(return_value=[])),
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
        )


class TestRunWritePathFailureContainment:
    @pytest.mark.asyncio
    async def test_extraction_raising_never_propagates(self):
        """AC-14."""
        identity = _identity()
        resolved = _resolved()

        with patch.object(
            global_memory_service, "extract_facts", new=AsyncMock(side_effect=RuntimeError("boom"))
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
        client.similarity = AsyncMock(return_value=None)
        client.add = AsyncMock(side_effect=RuntimeError("store unreachable"))

        with (
            patch.object(global_memory_service, "extract_facts", new=AsyncMock(return_value=["a fact"])),
            patch.object(global_memory_service, "get_memory_client", new=AsyncMock(return_value=client)),
        ):
            await run_write_path(
                identity=identity, resolved=resolved, llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1, original_user_message="hi", response_text="hi",
                turn_used_tool_call=False,
            )


class TestRunWritePathDedup:
    """AC-47."""

    @pytest.mark.asyncio
    async def test_near_duplicate_above_threshold_is_dropped(self, monkeypatch):
        monkeypatch.setattr(memory_settings, "dedup_threshold", lambda: 0.9)
        identity = _identity()
        resolved = _resolved()
        client = MagicMock()
        # "new fact" is novel; "known fact" scores above threshold.
        client.similarity = AsyncMock(side_effect=[None, 0.95])
        client.add = AsyncMock(return_value=1)

        with (
            patch.object(
                global_memory_service, "extract_facts",
                new=AsyncMock(return_value=["new fact", "known fact"]),
            ),
            patch.object(global_memory_service, "get_memory_client", new=AsyncMock(return_value=client)),
        ):
            await run_write_path(
                identity=identity, resolved=resolved, llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1, original_user_message="hi", response_text="hi",
                turn_used_tool_call=False,
            )

        client.add.assert_awaited_once_with(identity, ["new fact"])

    @pytest.mark.asyncio
    async def test_all_candidates_deduped_away_writes_nothing(self, monkeypatch):
        monkeypatch.setattr(memory_settings, "dedup_threshold", lambda: 0.5)
        identity = _identity()
        resolved = _resolved()
        client = MagicMock()
        client.similarity = AsyncMock(return_value=0.99)
        client.add = AsyncMock(return_value=0)

        with (
            patch.object(global_memory_service, "extract_facts", new=AsyncMock(return_value=["known fact"])),
            patch.object(global_memory_service, "get_memory_client", new=AsyncMock(return_value=client)),
        ):
            await run_write_path(
                identity=identity, resolved=resolved, llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1, original_user_message="hi", response_text="hi",
                turn_used_tool_call=False,
            )

        client.add.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dedup_stage_failure_degrades_to_writing_every_candidate(self):
        """'Losing a fact is worse than storing a near-duplicate' (AD-8)."""
        identity = _identity()
        resolved = _resolved()
        client = MagicMock()
        client.similarity = AsyncMock(side_effect=RuntimeError("embedding provider down"))
        client.add = AsyncMock(return_value=2)

        with (
            patch.object(
                global_memory_service, "extract_facts",
                new=AsyncMock(return_value=["fact one", "fact two"]),
            ),
            patch.object(global_memory_service, "get_memory_client", new=AsyncMock(return_value=client)),
        ):
            await run_write_path(
                identity=identity, resolved=resolved, llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=1, original_user_message="hi", response_text="hi",
                turn_used_tool_call=False,
            )

        client.add.assert_awaited_once_with(identity, ["fact one", "fact two"])


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
        client.similarity = AsyncMock(return_value=None)
        client.add = AsyncMock(return_value=1)

        with (
            patch.object(global_memory_service, "extract_facts", new=AsyncMock(return_value=["a fact"])),
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
            patch.object(global_memory_service, "extract_facts", new=AsyncMock()) as mock_extract,
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
        client.similarity = AsyncMock(return_value=None)
        client.add = AsyncMock(return_value=1)

        with (
            patch.object(global_memory_service, "extract_facts", new=AsyncMock(return_value=["a fact"])),
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
            patch.object(global_memory_service, "extract_facts", new=AsyncMock(return_value=[])),
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
        # No facts survived extraction, so the store is never even reached.
        mock_get_client.assert_not_called()

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
        client.similarity = AsyncMock(return_value=None)
        client.add = AsyncMock(return_value=1)

        with (
            patch.object(global_memory_service, "extract_facts", new=AsyncMock(return_value=["a fact"])),
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
            patch.object(global_memory_service, "extract_facts", new=AsyncMock(return_value=[])),
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
