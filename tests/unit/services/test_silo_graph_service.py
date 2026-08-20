"""``SiloGraphService._cypher_graph`` must cap edges by ``max_edges``, not a
multiple of ``max_nodes``, and report true totals for both nodes and edges.

Neo4j isn't reachable from the test environment (internal-network-only, no
published port), so the driver is mocked; these test the Cypher call
arguments and result assembly, not a live graph.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from services.silo_graph_service import SiloGraphService


class _FakeNode:
    """Minimal stand-in for a neo4j.graph.Node."""

    def __init__(self, entity_id: str):
        self._data = {"entity_id": entity_id}
        self.labels = ["Entity"]
        self.element_id = f"elem-{entity_id}"

    def get(self, key, default=None):
        return self._data.get(key, default)

    def items(self):
        return self._data.items()


def _run_cypher_graph(*, max_nodes=200, max_edges=1000, total_nodes=5000, total_edges=20000):
    """Drive ``_cypher_graph`` with a mocked driver, one fake node, no edges
    (edge *rows* aren't under test here — the LIMIT argument passed to the
    edge query is)."""
    session = MagicMock()
    session.run.side_effect = [
        MagicMock(single=lambda: {"total": total_nodes}),  # total node count
        MagicMock(single=lambda: {"total": total_edges}),  # total edge count
        [{"n": _FakeNode("bt-duo")}],  # node fetch
        [],  # edge fetch
    ]
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = False

    with patch.object(SiloGraphService, "_neo4j_driver", return_value=driver):
        result = SiloGraphService._cypher_graph(
            workspace="silo_1", max_nodes=max_nodes, max_edges=max_edges,
            node_label=None, search=None,
        )
    return result, session


def test_edge_query_is_capped_by_max_edges_not_a_multiple_of_max_nodes():
    result, session = _run_cypher_graph(max_nodes=200, max_edges=1000)

    edge_call = session.run.call_args_list[-1]
    assert edge_call.kwargs["limit"] == 1000
    assert result["total_edges"] == 20000


def test_reports_true_totals_alongside_the_fetched_counts():
    result, _ = _run_cypher_graph(total_nodes=5000, total_edges=20000)

    assert result["total_nodes"] == 5000
    assert result["total_edges"] == 20000
    assert result["node_count"] == 1  # the one fetched node, not the total


def test_truncated_when_edge_limit_is_hit_even_if_nodes_are_not():
    """A silo with few nodes but many edges between them must still show
    'truncated' — before this, truncated only looked at max_nodes."""
    session = MagicMock()
    session.run.side_effect = [
        MagicMock(single=lambda: {"total": 5}),
        MagicMock(single=lambda: {"total": 50_000}),
        [{"n": _FakeNode("hub")}],
        [
            {"src": "hub", "tgt": f"n{i}", "rel_type": "RELATED", "rid": i, "props": {}}
            for i in range(3)
        ],
    ]
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = False

    with patch.object(SiloGraphService, "_neo4j_driver", return_value=driver):
        result = SiloGraphService._cypher_graph(
            workspace="silo_1", max_nodes=200, max_edges=3, node_label=None, search=None,
        )

    assert result["truncated"] is True
    assert len(result["edges"]) == 3
