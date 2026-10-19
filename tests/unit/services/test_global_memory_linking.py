"""Extending / replacing memories through the extractor's links (spec
`user-memory-mem0`, revision s): before extraction the user's similar stored
memories are passed to the extractor as "Existing Memories" with prompt-local
ids; it may link a new fact to one with a `relationship`. extends / updates /
supersedes replace the linked memory (the more complete one wins); contradicts
keeps both; ids we did not hand out are ignored.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import services.global_memory_service as global_memory_service
from services.global_memory_service import (
    ExtractedFact,
    _find_existing_memories,
    _format_extraction_input,
    _parse_extracted,
    run_write_path,
)
from tools.memory import settings as memory_settings
from tools.memory.client import RetrievedFact, fact_key
from tools.memory.identity import MemoryIdentity
from tools.memory.resolution import ResolvedMemoryServices


def _identity() -> MemoryIdentity:
    root = memory_settings.namespace_root()
    return MemoryIdentity(app_id=1, user_id=7, agent_id=3, namespace=(root, "1", "7"))


def _resolved() -> ResolvedMemoryServices:
    return ResolvedMemoryServices(
        extraction_service_id=10, extraction_source="app_config", extraction_is_system=False,
        embedding_service_id=20, embedding_source="app_config", embedding_is_system=False,
    )


OLD = RetrievedFact(text="Works at Ikelran", created_at=None, conversation_id=None, key="a" * 32)


class TestParseExtracted:
    def test_reads_text_links_and_relationship(self):
        raw = json.dumps({"memory": [
            {"text": "Works on AI projects at Ikerlan", "linked_memory_ids": ["m1"], "relationship": "extends"},
            {"text": "Name is Itziar", "linked_memory_ids": [], "relationship": None},
        ]})

        assert _parse_extracted(raw) == [
            ExtractedFact("Works on AI projects at Ikerlan", ("m1",), "extends"),
            ExtractedFact("Name is Itziar"),
        ]

    def test_an_unknown_relationship_or_malformed_links_mean_no_link(self):
        raw = json.dumps({"memory": [
            {"text": "a", "linked_memory_ids": ["m1"], "relationship": "deletes_everything"},
            {"text": "b", "linked_memory_ids": "m1", "relationship": "extends"},
            {"text": "c", "linked_memory_ids": [], "relationship": "extends"},
        ]})

        assert _parse_extracted(raw) == [ExtractedFact("a"), ExtractedFact("b"), ExtractedFact("c")]

    def test_older_shapes_still_parse(self):
        assert _parse_extracted('["Uses Python"]') == [ExtractedFact("Uses Python")]
        assert _parse_extracted('{"memory": ["Uses Python"]}') == [ExtractedFact("Uses Python")]


class TestExtractionInput:
    def test_existing_memories_go_in_with_prompt_ids_and_no_store_keys(self):
        content = _format_extraction_input("User: trabajo en proyectos de ia en ikerlan", [("m1", "Works at Ikelran")])

        assert content.startswith("Observation Date: ")
        assert "Recently Extracted Memories: []" in content
        assert '"id": "m1"' in content and '"text": "Works at Ikelran"' in content
        assert "a" * 32 not in content
        assert content.endswith("User: trabajo en proyectos de ia en ikerlan")

    def test_no_existing_memories_is_an_empty_list(self):
        assert "Existing Memories:\n[]" in _format_extraction_input("User: hi", [])


class TestFindExistingMemories:
    @pytest.mark.asyncio
    async def test_one_search_per_message_deduplicated_and_capped(self, monkeypatch):
        monkeypatch.setattr(global_memory_service, "_EXISTING_MAX", 3)
        client = MagicMock()
        client.search_facts = AsyncMock(side_effect=[
            [RetrievedFact("A", None, None, "k1"), RetrievedFact("B", None, None, "k2")],
            [RetrievedFact("B", None, None, "k2"), RetrievedFact("C", None, None, "k3"), RetrievedFact("D", None, None, "k4")],
        ])

        found = await _find_existing_memories(client, _identity(), ["older message", "current message"])

        assert [f.key for f in found] == ["k1", "k2", "k3"]
        # Most recent message searched first, top 3 each.
        assert client.search_facts.await_args_list[0].args[1:] == ("current message", 3)

    @pytest.mark.asyncio
    async def test_a_failed_search_contributes_nothing(self):
        client = MagicMock()
        client.search_facts = AsyncMock(side_effect=RuntimeError("down"))

        assert await _find_existing_memories(client, _identity(), ["hi"]) == []


class TestSessionFactsAreAlwaysShown:
    """Client requirement: facts already learned in this conversation reach the
    extractor even when they are not similar to the messages under review, so
    they can be extended or updated."""

    @pytest.mark.asyncio
    async def test_session_facts_come_first_then_similar_ones_without_repeats(self):
        client = MagicMock()
        client.session_facts = AsyncMock(return_value=[
            RetrievedFact("Is named Itziar", None, 98, "s1"), RetrievedFact("Works at Ikerlan", None, 98, "s2"),
        ])
        client.search_facts = AsyncMock(return_value=[
            RetrievedFact("Works at Ikerlan", None, 98, "s2"), RetrievedFact("Uses Python", None, 70, "o1"),
        ])

        found = await _find_existing_memories(client, _identity(), ["dame un ejemplo"], conversation_id=98)

        assert [f.key for f in found] == ["s1", "s2", "o1"]
        client.session_facts.assert_awaited_once_with(_identity(), 98, global_memory_service._SESSION_FACTS_MAX)

    @pytest.mark.asyncio
    async def test_session_facts_do_not_eat_the_similar_budget(self, monkeypatch):
        monkeypatch.setattr(global_memory_service, "_EXISTING_MAX", 1)
        client = MagicMock()
        client.session_facts = AsyncMock(return_value=[RetrievedFact("a", None, 98, "s1"), RetrievedFact("b", None, 98, "s2")])
        client.search_facts = AsyncMock(return_value=[RetrievedFact("c", None, 70, "o1"), RetrievedFact("d", None, 70, "o2")])

        found = await _find_existing_memories(client, _identity(), ["x"], conversation_id=98)

        assert [f.key for f in found] == ["s1", "s2", "o1"]

    @pytest.mark.asyncio
    async def test_no_conversation_means_no_session_lookup(self):
        client = MagicMock()
        client.session_facts = AsyncMock()
        client.search_facts = AsyncMock(return_value=[])

        await _find_existing_memories(client, _identity(), ["x"])

        client.session_facts.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_session_fact_can_be_extended(self):
        """End to end through the write path: the session fact gets an id the
        extractor can link, and the more complete fact replaces it."""
        session_fact = RetrievedFact("Works at Ikerlan", None, 98, "b" * 32)
        client = MagicMock()
        client.session_facts = AsyncMock(return_value=[session_fact])
        client.search_facts = AsyncMock(return_value=[])
        client.closest = AsyncMock(return_value=None)
        client.add = AsyncMock(return_value=1)
        client.delete_keys = AsyncMock(return_value=1)

        with (
            patch.object(
                global_memory_service, "extract_memories",
                new=AsyncMock(return_value=[ExtractedFact("Works on AI projects at Ikerlan", ("m1",), "extends")]),
            ) as mock_extract,
            patch.object(global_memory_service, "get_memory_client", new=AsyncMock(return_value=client)),
        ):
            await run_write_path(
                identity=_identity(), resolved=_resolved(), llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=20, original_user_message="dame un ejemplo",
                response_text="OK", turn_used_tool_call=False,
                earlier_user_messages=("trabajo en proyectos de ia",), conversation_id=98,
            )

        assert mock_extract.await_args.args[2] == [("m1", "Works at Ikerlan")]
        client.delete_keys.assert_awaited_once_with(_identity(), ["b" * 32])


class TestApplyingRelationships:
    def _client(self, existing=(OLD,), closest=None):
        client = MagicMock()
        client.session_facts = AsyncMock(return_value=[])
        client.search_facts = AsyncMock(return_value=list(existing))
        client.closest = AsyncMock(return_value=closest)
        client.add = AsyncMock(return_value=1)
        client.delete_keys = AsyncMock(return_value=1)
        return client

    async def _run(self, client, extracted):
        with (
            patch.object(global_memory_service, "extract_memories", new=AsyncMock(return_value=extracted)) as mock_extract,
            patch.object(global_memory_service, "get_memory_client", new=AsyncMock(return_value=client)),
        ):
            await run_write_path(
                identity=_identity(), resolved=_resolved(), llm=MagicMock(), embeddings=MagicMock(),
                embedding_service_id=20, original_user_message="trabajo en proyectos de ia en ikerlan",
                response_text="Noted.", turn_used_tool_call=False,
            )
        return mock_extract

    @pytest.mark.asyncio
    async def test_the_extractor_sees_the_similar_memories_with_prompt_ids(self):
        client = self._client()
        mock_extract = await self._run(client, [])

        assert mock_extract.await_args.args[2] == [("m1", "Works at Ikelran")]

    @pytest.mark.parametrize("relationship", ["extends", "updates", "supersedes"])
    @pytest.mark.asyncio
    async def test_a_more_complete_fact_replaces_the_one_it_links(self, relationship):
        """"Works at Ikelran" + "trabajo en proyectos de IA en Ikerlan" must end
        as ONE memory: the more complete one."""
        client = self._client()

        await self._run(client, [ExtractedFact("Works on AI projects at Ikerlan", ("m1",), relationship)])

        client.add.assert_awaited_once()
        assert client.add.await_args.args[1] == ["Works on AI projects at Ikerlan"]
        client.delete_keys.assert_awaited_once_with(_identity(), [OLD.key])

    @pytest.mark.asyncio
    async def test_a_contradiction_keeps_both(self):
        client = self._client()

        await self._run(client, [ExtractedFact("Wants responses in Spanish", ("m1",), "contradicts")])

        client.add.assert_awaited_once()
        client.delete_keys.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ids_that_were_not_handed_out_are_ignored(self):
        """A manipulated reply cannot reach memories it was never shown."""
        client = self._client()

        await self._run(client, [ExtractedFact("x", ("m9", "aaaaaaaa"), "supersedes")])

        client.delete_keys.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_linked_facts_skip_the_vector_dedup(self):
        client = self._client()

        await self._run(client, [ExtractedFact("Works on AI projects at Ikerlan", ("m1",), "extends")])

        client.closest.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unlinked_facts_still_go_through_the_vector_dedup(self):
        client = self._client(existing=())

        await self._run(client, [ExtractedFact("Name is Itziar")])

        client.closest.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_never_deletes_the_row_the_new_fact_writes_to(self):
        """Same text = same key: the write overwrites it in place."""
        same = RetrievedFact("Works at Ikerlan", None, None, fact_key("Works at Ikerlan"))
        client = self._client(existing=(same,))

        await self._run(client, [ExtractedFact("Works at Ikerlan", ("m1",), "updates")])

        client.delete_keys.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_nothing_is_deleted_when_the_write_fails(self):
        client = self._client()
        client.add = AsyncMock(return_value=0)

        await self._run(client, [ExtractedFact("Works on AI projects at Ikerlan", ("m1",), "extends")])

        client.delete_keys.assert_not_awaited()
