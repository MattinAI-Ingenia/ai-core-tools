"""Cypher-shape lint for every query ``SiloGraphService`` emits.

The unit mocks don't validate Cypher syntax — two real bugs shipped that way
(this Neo4j 5.26 build rejects the postfix ``NOT IN`` form, and a neighbor
query concluded with ``WITH`` without a ``RETURN``). This lint asserts the
syntax shape a real Neo4j enforces, without needing one:

- no postfix ``NOT IN`` (the parser rejects ``x NOT IN $l``; use ``NOT x IN $l``);
- every query contains a ``RETURN``;
- a query containing ``WITH`` must have a ``RETURN`` after its last ``WITH``
  (a query cannot conclude with ``WITH``).
"""

from __future__ import annotations

import re

from tests.unit.services.test_silo_graph_service import _run_cypher_graph
from tests.unit.services.test_silo_graph_neighbors import (
    _FakeNode,
    _run_get_node_neighbors,
)
from unittest.mock import MagicMock


def _assert_valid_cypher_shape(query: str) -> None:
    assert " NOT IN " not in query, f"postfix NOT IN is rejected by this Neo4j: {query}"
    assert re.search(r"\bRETURN\b", query), f"query never returns: {query}"
    with_matches = [m.start() for m in re.finditer(r"\bWITH\b", query)]
    if with_matches:
        tail = query[with_matches[-1]:]
        assert re.search(r"\bRETURN\b", tail), (
            f"query concludes with WITH (a real Neo4j rejects it): {query}"
        )


def _collect_graph_queries() -> list[str]:
    queries: list[str] = []
    variants = [
        {},
        {"entity_types": ["Machine", "Other"]},
        {"search": "pump"},
        {"entity_types": ["Machine", "Other"], "search": "pump"},
        {"entity_types": ["Machine"]},
    ]
    for kwargs in variants:
        _, session = _run_cypher_graph(**kwargs)
        queries.extend(call.args[0] for call in session.run.call_args_list)
    assert queries, "expected the graph service to emit queries"
    return queries


def _collect_neighbors_queries() -> list[str]:
    root = _FakeNode("canon(root)")
    queries: list[str] = []
    for depth, max_nodes in ((1, 10), (2, 10), (3, 200)):
        session = MagicMock()
        session.run.side_effect = [
            [{"n": root}],  # root fetch
            [],             # depth-1 fetch
            [],             # deeper layer 1
            [],             # deeper layer 2
            [],             # edge fetch
        ]
        _run_get_node_neighbors(session, node_id="root", depth=depth, max_nodes=max_nodes)
        queries.extend(call.args[0] for call in session.run.call_args_list)
    assert queries, "expected the neighbor service to emit queries"
    return queries


def test_every_graph_query_is_valid_cypher_shape():
    for query in _collect_graph_queries():
        _assert_valid_cypher_shape(query)


def test_every_neighbors_query_is_valid_cypher_shape():
    for query in _collect_neighbors_queries():
        _assert_valid_cypher_shape(query)
