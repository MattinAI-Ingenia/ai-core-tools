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


def test_name_map_rewrites_ids_names_and_endpoints():
    """Display names (canonical → most-mentioned variant) must flow through
    entity ids AND relationship endpoints, and the partial-node builder must
    use the DISPLAY form so edges stay joinable."""
    raw = {
        "data": {
            "entities": [
                {"entity_name": "mcf 40", "entity_type": "equipment"},
                {"entity_name": "termico", "entity_type": "part"},
            ],
            "relationships": [
                {"src_id": "mcf 40", "tgt_id": "burner mcf 40"},
            ],
        }
    }
    name_map = {"mcf 40": "MCF-40", "termico": "Térmico", "burner mcf 40": "Quemador MCF-40"}

    out = _normalize_lightrag_graph(raw, name_map)["data"]
    by_id = {e["id"]: e for e in out["entities"]}

    assert set(by_id) == {"MCF-40", "Térmico", "Quemador MCF-40"}
    assert by_id["MCF-40"]["name"] == "MCF-40"
    assert out["relationships"][0]["source"] == "MCF-40"
    assert out["relationships"][0]["target"] == "Quemador MCF-40"


def test_name_map_unknown_names_fall_back_to_canonical():
    raw = {"data": {"entities": [{"entity_name": "mcf 40"}], "relationships": []}}
    out = _normalize_lightrag_graph(raw, {"unrelated": "X"})["data"]
    assert out["entities"][0]["id"] == "mcf 40"
    assert out["entities"][0]["name"] == "mcf 40"
