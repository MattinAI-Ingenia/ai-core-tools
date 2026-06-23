"""Unit tests for _normalize_lightrag_graph endpoint handling."""

from tools.vector_stores.lightrag_store import _normalize_lightrag_graph


def test_missing_relationship_endpoint_becomes_partial_node():
    """LightRAG truncates entities/relationships independently, so a relationship
    can reference an entity absent from the entity list. That endpoint must be
    added as a 'partial' node so the edge is still drawable."""
    raw = {
        "data": {
            "entities": [
                {"entity_name": "Statistics Flanders", "entity_type": "org"},
            ],
            "relationships": [
                {"src_id": "Statistics Flanders", "tgt_id": "Michael Reusens"},
            ],
        }
    }

    out = _normalize_lightrag_graph(raw)["data"]
    by_id = {e["id"]: e for e in out["entities"]}

    # The missing endpoint was added, marked partial; the full entity was not.
    assert by_id["Michael Reusens"]["partial"] is True
    assert "partial" not in by_id["Statistics Flanders"]
    # Both endpoints now exist, so the edge is renderable.
    assert {(r["source"], r["target"]) for r in out["relationships"]} == {
        ("Statistics Flanders", "Michael Reusens")
    }


def test_known_endpoints_add_no_partial_nodes():
    raw = {
        "data": {
            "entities": [
                {"entity_name": "A"},
                {"entity_name": "B"},
            ],
            "relationships": [{"src_id": "A", "tgt_id": "B"}],
        }
    }
    entities = _normalize_lightrag_graph(raw)["data"]["entities"]
    assert len(entities) == 2
    assert all("partial" not in e for e in entities)
