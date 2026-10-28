"""merge_lightrag_graph: real entities must never be downgraded by truncated stubs."""

import pytest

from tools.streaming_utils import merge_lightrag_graph


def _entity(name, *, partial=False, source_id="chunk_1"):
    e = {"id": name, "name": name, "entity_type": "Thing", "description": "d",
         "file_path": "f", "source_id": source_id}
    if partial:
        e.update({"entity_type": "", "description": "", "file_path": "", "partial": True,
                  "source_id": ""})
    return e


def _run(entities, relationships=()):
    return {"data": {"entities": entities,
                     "relationships": [{"id": f"rel_{i}", "source": s, "target": t}
                                       for i, (s, t) in enumerate(relationships)],
                     "chunks": []}}


def test_real_entity_survives_later_partial_stub():
    # Turn run 1 returns X real; run 2's relationship only references X (truncated
    # from its entity list -> synthesized stub). The stub must not replace the real X.
    real = _entity("X")
    stub = _entity("X", partial=True)
    merged = merge_lightrag_graph(_run([real], [("X", "Z")]), _run([stub], [("X", "W")]))
    by_id = {e["id"]: e for e in merged["data"]["entities"]}
    assert len(by_id) == 1
    assert by_id["X"]["source_id"] == "chunk_1"
    assert not by_id["X"].get("partial")
    assert {("X", "Z"), ("X", "W")} <= {(r["source"], r["target"]) for r in merged["data"]["relationships"]}


def test_later_real_entity_upgrades_earlier_stub():
    merged = merge_lightrag_graph(_run([_entity("X", partial=True)], [("X", "Z")]),
                                  _run([_entity("X")]))
    by_id = {e["id"]: e for e in merged["data"]["entities"]}
    assert len(by_id) == 1
    assert not by_id["X"].get("partial")


def test_same_name_in_different_silos_stays_two_nodes():
    a = {**_entity("X"), "silo_id": 37}
    b = {**_entity("X", partial=True), "silo_id": 39}
    merged = merge_lightrag_graph(_run([a]), _run([b]))
    assert len(merged["data"]["entities"]) == 2
