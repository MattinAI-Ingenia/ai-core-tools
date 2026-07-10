"""Unit tests for streaming_utils — focuses on the chunk-mapping logic.

The critical regression these tests guard against is the SummarizationMiddleware
leak: middleware-internal LLM calls tag their config with
``{"metadata": {"lc_source": "summarization"}}``, and the streaming layer must
suppress those chunks so the summary text never reaches the SSE token stream.
"""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from tools.streaming_utils import (
    SSE_TOKEN,
    _map_messages_chunk,
    map_stream_event,
    merge_lightrag_graph,
)


def _ai_chunk(content: str) -> AIMessageChunk:
    return AIMessageChunk(content=content)


class TestMapMessagesChunk:
    def test_emits_token_for_plain_ai_chunk(self):
        chunk = (_ai_chunk("hello"), {"langgraph_node": "agent"})

        result = _map_messages_chunk(chunk)

        assert result == [{"type": SSE_TOKEN, "data": {"content": "hello"}}]

    def test_drops_summarization_chunk_top_level_metadata(self):
        # SummarizationMiddleware tags its model.ainvoke() with
        # config={"metadata": {"lc_source": "summarization"}}; LangGraph may
        # surface the value at the top level of the metadata dict.
        chunk = (
            _ai_chunk("## SESSION INTENT\nThe user is asking..."),
            {"langgraph_node": "agent", "lc_source": "summarization"},
        )

        assert _map_messages_chunk(chunk) is None

    def test_drops_summarization_chunk_nested_metadata(self):
        # LangGraph also surfaces config metadata under a nested "metadata" key.
        chunk = (
            _ai_chunk("## SUMMARY\n..."),
            {
                "langgraph_node": "agent",
                "metadata": {"lc_source": "summarization"},
            },
        )

        assert _map_messages_chunk(chunk) is None

    def test_drops_lightrag_internal_chunk(self):
        # LightRAG's keyword-extraction call runs inside the retriever tool;
        # tagged lc_source="lightrag", its JSON must never reach the stream.
        chunk = (
            _ai_chunk('{"high_level_keywords": ["paper authors"], "low_level_keywords": ["kb"]}'),
            {"langgraph_node": "tools", "lc_source": "lightrag"},
        )

        assert _map_messages_chunk(chunk) is None

    def test_unknown_lc_source_is_not_dropped(self):
        # Only known internal sources are filtered; unknown values pass through
        # so that a future, unrelated `lc_source` does not silently break the stream.
        chunk = (
            _ai_chunk("regular reply"),
            {"langgraph_node": "agent", "lc_source": "user_facing_thing"},
        )

        result = _map_messages_chunk(chunk)

        assert result == [{"type": SSE_TOKEN, "data": {"content": "regular reply"}}]

    def test_skips_tool_message(self):
        msg = ToolMessage(content="result", tool_call_id="abc")
        chunk = (msg, {"langgraph_node": "tools"})

        assert _map_messages_chunk(chunk) is None

    def test_skips_human_message(self):
        msg = HumanMessage(content="hi")
        chunk = (msg, {"langgraph_node": "agent"})

        assert _map_messages_chunk(chunk) is None

    def test_skips_pure_tool_call_chunk(self):
        ai = MagicMock()
        ai.__class__.__name__ = "AIMessageChunk"
        ai.tool_calls = [{"name": "search", "id": "1", "args": {}}]
        ai.content = ""

        # Patch isinstance check by routing through the public dispatcher.
        # Direct unit on _map_messages_chunk uses type().__name__, so a real
        # AIMessageChunk wrapper is more faithful here.
        chunk_obj = AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": "search", "id": "1", "args": "{}", "index": 0}],
        )
        chunk = (chunk_obj, {"langgraph_node": "agent"})

        assert _map_messages_chunk(chunk) is None

    def test_skips_empty_content(self):
        chunk = (_ai_chunk(""), {"langgraph_node": "agent"})

        assert _map_messages_chunk(chunk) is None

    def test_handles_malformed_chunk(self):
        assert _map_messages_chunk("not a tuple") is None
        assert _map_messages_chunk((_ai_chunk("x"),)) is None

    def test_handles_missing_metadata(self):
        # If metadata is None or non-dict, the filter must still pass through
        # legitimate tokens rather than crashing.
        chunk = (_ai_chunk("ok"), None)

        result = _map_messages_chunk(chunk)

        assert result == [{"type": SSE_TOKEN, "data": {"content": "ok"}}]


class TestMapStreamEventDispatcher:
    def test_messages_mode_routes_to_messages_handler(self):
        chunk = (_ai_chunk("hi"), {"langgraph_node": "agent"})

        events = map_stream_event("messages", chunk)

        assert events == [{"type": SSE_TOKEN, "data": {"content": "hi"}}]

    def test_messages_mode_drops_summarization(self):
        chunk = (
            _ai_chunk("## SESSION INTENT\nleak"),
            {"metadata": {"lc_source": "summarization"}},
        )

        assert map_stream_event("messages", chunk) is None


class TestMergeLightragGraph:
    """A turn calling the retrieval tool more than once (multi-silo, or the
    agent searching twice) must end up with ALL of it, not just the last call —
    this is the fix for citations/subgraph misaligning on multi-call turns."""

    def test_first_call_returned_as_is(self):
        graph = {"data": {"entities": [{"id": "e1"}], "relationships": [], "chunks": [{"id": "c1"}]}}

        assert merge_lightrag_graph(None, graph) == graph

    def test_second_call_concatenates_chunks_in_order(self):
        first = {"data": {"entities": [], "relationships": [], "chunks": [{"id": "a1"}, {"id": "a2"}]}}
        second = {"data": {"entities": [], "relationships": [], "chunks": [{"id": "b1"}]}}

        merged = merge_lightrag_graph(first, second)

        assert [c["id"] for c in merged["data"]["chunks"]] == ["a1", "a2", "b1"]

    def test_entities_deduped_by_id_across_calls(self):
        first = {"data": {"entities": [{"id": "e1", "name": "Empresa A"}], "relationships": [], "chunks": []}}
        second = {"data": {"entities": [{"id": "e1", "name": "Empresa A"}, {"id": "e2", "name": "Empresa B"}], "relationships": [], "chunks": []}}

        merged = merge_lightrag_graph(first, second)

        assert sorted(e["id"] for e in merged["data"]["entities"]) == ["e1", "e2"]

    def test_relationships_concatenated_without_dedup(self):
        # ponytail: LightRAG's fallback relationship id ("rel_0", "rel_1"...) is
        # call-local, not globally stable — deduping by id here could wrongly
        # merge two unrelated relationships that happen to share a fallback id.
        first = {"data": {"entities": [], "relationships": [{"id": "rel_0"}], "chunks": []}}
        second = {"data": {"entities": [], "relationships": [{"id": "rel_0"}], "chunks": []}}

        merged = merge_lightrag_graph(first, second)

        assert len(merged["data"]["relationships"]) == 2
