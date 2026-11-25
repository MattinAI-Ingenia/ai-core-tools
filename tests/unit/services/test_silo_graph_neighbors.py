"""``SiloGraphService.get_node_neighbors`` — rooted subgraph fetch for
expand-neighbors / center-here.

Neo4j isn't reachable from the test environment, so the driver is mocked;
these test canonicalization, local-first truncation, entity-only filtering
and the not-found contract — call arguments and result assembly, not a live
graph.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from services.silo_graph_service import SiloGraphService


class _FakeNode:
    """Minimal stand-in for a neo4j.graph.Node."""

    def __init__(self, entity_id: str, entity_type: str | None = "Machine"):
        self._data = {"entity_id": entity_id, "entity_type": entity_type}
        self.labels = []
        self.element_id = f"elem-{entity_id}"

    def get(self, key, default=None):
        return self._data.get(key, default)

    def items(self):
        return self._data.items()


class _ChunkNode(_FakeNode):
    """A workspace text-chunk node: no entity_id, never returned."""

    def __init__(self, chunk_id: str):
        super().__init__(chunk_id, entity_type=None)
        self._data = {}  # no entity_id, no entity_type

    def get(self, key, default=None):
        return self._data.get(key, default)


def _driver(session: MagicMock):
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = False
    return driver


def _run_get_node_neighbors(session, **kwargs):
    with (
        patch.object(SiloGraphService, "_neo4j_driver", return_value=_driver(session)),
        patch(
            "services.silo_graph_service.canonicalize_entity_name",
            side_effect=lambda s: f"canon({s})",
        ),
    ):
        result = SiloGraphService.get_node_neighbors(
            silo_id=1,
            max_nodes=kwargs.pop("max_nodes", 200),
            max_edges=kwargs.pop("max_edges", 1000),
            db=None,
            **kwargs,
        )
        return result, session


def test_incoming_display_name_canonicalized():
    """The frontend only knows DISPLAY ids (rewritten by the graph endpoint):
    the root lookup must run on the canonical merge key, not the raw string."""
    root = _FakeNode("canon(MCF-40)")
    session = MagicMock()
    session.run.side_effect = [
        [{"n": root}],   # root fetch
        [],              # depth-1 fetch
        [],              # deeper fetch (depth=1 → skipped; still bounded)
        [],              # edge fetch
    ]

    result, session = _run_get_node_neighbors(session, node_id="MCF-40", depth=2)

    assert session.run.call_args_list[0].kwargs["root"] == "canon(MCF-40)"
    assert result["nodes"][0]["id"] == "canon(MCF-40)"


def test_root_and_depth1_always_kept():
    """Locality guarantee: root + ALL depth-1 neighbors stay; the node cap
    only trims deeper layers (500 offered → budget-kept here)."""
    root = _FakeNode("canon(root)")
    depth1 = [_FakeNode(f"canon(n{i})") for i in range(2)]
    deeper = [_FakeNode(f"canon(d{i})") for i in range(500)]
    session = MagicMock()
    session.run.side_effect = [
        [{"n": root}],
        [{"m": n} for n in depth1],
        [{"m": n} for n in deeper],
        [],
    ]

    result, session = _run_get_node_neighbors(session, node_id="root", depth=2, max_nodes=12)

    # budget = 12 - 1 (root) - 2 (depth1) = 9 deeper nodes kept of the 500 offered
    deeper_call = session.run.call_args_list[2]
    assert deeper_call.kwargs["budget"] == 9
    ids = [n["id"] for n in result["nodes"]]
    assert "canon(root)" in ids
    for node in depth1:
        assert node._data["entity_id"] in ids
    assert len(result["nodes"]) == 12
    assert result["truncated"] is True


def test_entity_only_subgraph():
    """Chunk nodes (no entity_id) are never returned, and edges that touch
    them are dropped — same semantics as the global graph view."""
    root = _FakeNode("canon(root)")
    depth1 = [_FakeNode("canon(real)"), _ChunkNode("res7-p1-chunk-000")]
    deeper = [_FakeNode("canon(d)"), _ChunkNode("res7-p2-chunk-000")]
    session = MagicMock()
    session.run.side_effect = [
        [{"n": root}],
        [{"m": n} for n in depth1],
        [{"m": n} for n in deeper],
        [],
    ]

    result, _ = _run_get_node_neighbors(session, node_id="root", depth=2, max_nodes=200)

    ids = [n["id"] for n in result["nodes"]]
    assert ids == ["canon(root)", "canon(real)", "canon(d)"]


def test_root_not_found_raises():
    session = MagicMock()
    session.run.side_effect = [[]]  # root fetch → empty

    with pytest.raises(LookupError):
        _run_get_node_neighbors(session, node_id="ghost", depth=1)


def test_depth1_skips_deeper_query():
    """depth=1 has no deeper layer — only root, depth-1 and edge queries run."""
    session = MagicMock()
    session.run.side_effect = [[{"n": _FakeNode("canon(root)")}], [], []]

    result, session = _run_get_node_neighbors(session, node_id="root", depth=1, max_nodes=10)

    assert session.run.call_count == 3
    assert result["truncated"] is False
