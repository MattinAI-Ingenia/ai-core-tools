"""Unit tests for _wrap_query_response display-map wiring.

Two decisions under test (docs/dependencies/lightrag.md §6.1):
1. the playground graph bubble (`lightrag_raw_data` → frontend
   `lightrag_graph`) shows DISPLAY names — required because the streaming
   layer's merge_lightrag_graph dedups entities by id, so without the map
   the bubble would show canonical keys while the explorer shows variants;
2. the LLM's own context string is NOT rewritten — it must keep matching
   what LightRAG actually indexed.
"""

from tools.vector_stores.lightrag_store import _collect_graph_names, _wrap_query_response


def _fake_response():
    return {
        "llm_response": {"content": "Context mentions MCF-40 and its burner."},
        "data": {
            "entities": [
                {"entity_name": "mcf 40", "entity_type": "equipment"},
                {"entity_name": "quemador mcf 40", "entity_type": "part"},
            ],
            "relationships": [
                {"src_id": "mcf 40", "tgt_id": "quemador mcf 40"},
            ],
        },
    }


class TestCollectGraphNames:
    def test_entities_and_both_endpoints(self):
        assert _collect_graph_names(_fake_response()) == {"mcf 40", "quemador mcf 40"}

    def test_none_and_malformed_are_safe(self):
        assert _collect_graph_names(None) == set()
        assert _collect_graph_names({"data": None}) == set()
        assert _collect_graph_names({"data": {"entities": "not-a-list"}}) == set()


class TestWrapQueryResponse:
    def test_bubble_gets_display_names_context_stays_canonical(self):
        name_map = {"mcf 40": "MCF-40", "quemador mcf 40": "Quemador MCF-40"}
        docs = _wrap_query_response(_fake_response(), "hybrid", name_map=name_map)
        assert len(docs) == 1
        # The LLM context keeps the canonical keys LightRAG indexed.
        assert "MCF-40 and its burner" in docs[0].page_content
        # The graph bubble shows the display variants.
        entities = docs[0].metadata["lightrag_raw_data"]["data"]["entities"]
        assert {e["id"] for e in entities} == {"MCF-40", "Quemador MCF-40"}
        edges = docs[0].metadata["lightrag_raw_data"]["data"]["relationships"]
        assert (edges[0]["source"], edges[0]["target"]) == ("MCF-40", "Quemador MCF-40")

    def test_no_name_map_keeps_canonical_everywhere(self):
        docs = _wrap_query_response(_fake_response(), "hybrid")
        entities = docs[0].metadata["lightrag_raw_data"]["data"]["entities"]
        assert {e["id"] for e in entities} == {"mcf 40", "quemador mcf 40"}

    def test_empty_response_returns_empty_list(self):
        assert _wrap_query_response(None, "hybrid") == []
        assert _wrap_query_response({}, "hybrid") == []

    def test_keyword_stripping_untouched_by_name_map(self):
        response = _fake_response()
        response["llm_response"]["content"] = (
            '{"high_level_keywords": ["mcf"], "low_level_keywords": ["mcf-40"]}\nContext'
        )
        docs = _wrap_query_response(response, "global")
        assert docs[0].page_content == "Context"
        assert docs[0].metadata["lightrag_keywords"]["low_level_keywords"] == ["mcf-40"]
