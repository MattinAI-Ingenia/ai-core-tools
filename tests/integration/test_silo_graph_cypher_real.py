"""Real-Neo4j smoke for the silo graph service.

The unit mocks cannot validate Cypher syntax; two real bugs shipped that way
(postfix ``NOT IN`` and a WITH-terminated neighbor query). This runs the
service queries against a live Neo4j — skipped when none is reachable (same
pattern as ``lightrag_helpers``: Neo4j has no published port, so this only
runs where the operator points ``NEO4J_URI`` at one).

Seeds a synthetic workspace, exercises the graph and neighbors paths end to
end (no LightRAG, no LLM involved), then cleans the workspace up.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

# A silo id that cannot exist (workspaces are silo_{id}) — keeps the synthetic
# workspace away from real silo data.
SILO_ID = 998877
WORKSPACE = f"silo_{SILO_ID}"


@pytest.fixture
def neo_driver():
    try:
        import config
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"config unavailable: {exc}")
    uri = getattr(config, "NEO4J_URI", None)
    password = getattr(config, "NEO4J_PASSWORD", None)
    if not uri or not password:
        pytest.skip("NEO4J_URI/NEO4J_PASSWORD not set; cannot run real-Neo4j smoke")
    from neo4j import GraphDatabase

    try:
        driver = GraphDatabase.driver(
            uri,
            auth=(getattr(config, "NEO4J_USERNAME", "neo4j"), password),
        )
        driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001 — connectivity probing must not raise
        pytest.skip(f"Neo4j at {uri} is not reachable ({exc}); cannot run here.")
    yield driver
    driver.close()


@pytest.fixture
def seeded_workspace(neo_driver):
    """6 entity nodes (some untyped) chained e1→…→e6, plus one edge to a
    chunk-less hub — enough to exercise filter, counts, root and hops."""
    with neo_driver.session() as session:
        session.run(f"MATCH (n:`{WORKSPACE}`) DETACH DELETE n")
        session.run(
            f"UNWIND range(1, 6) AS i "
            f"CREATE (n:`{WORKSPACE}`) "
            "SET n.entity_id = 'e' + toString(i), "
            "    n.entity_type = CASE WHEN i % 3 = 0 THEN '' ELSE 'T' + toString(i % 3) END, "
            "    n.description = 'Node ' + toString(i)"
        )
        session.run(
            f"MATCH (a:`{WORKSPACE}`), (b:`{WORKSPACE}`) "
            "WHERE a.entity_id = 'e' + toString(toInteger(substring(b.entity_id, 1)) - 1) "
            "AND b.entity_id STARTS WITH 'e' AND toInteger(substring(b.entity_id, 1)) > 1 "
            "MERGE (a)-[:NEXT]->(b)"
        )
    yield
    with neo_driver.session() as session:
        session.run(f"MATCH (n:`{WORKSPACE}`) DETACH DELETE n")


def test_graph_path_runs_on_real_neo4j(seeded_workspace):
    from services.silo_graph_service import SiloGraphService

    result = SiloGraphService._cypher_graph(
        workspace=WORKSPACE, max_nodes=10, max_edges=100,
        entity_types=["T1", "Other"], search=None,
    )
    assert result["nodes"], "graph query returned no nodes"
    # Every node matches the filter: typed T1 or untyped (the 'Other' branch).
    for node in result["nodes"]:
        assert node["entity_type"] in ("T1", ""), node

    # And a search variant executes too (different WHERE shape).
    searched = SiloGraphService._cypher_graph(
        workspace=WORKSPACE, max_nodes=10, max_edges=100,
        entity_types=["T1", "Other"], search="node",
    )
    assert searched["nodes"], "searched graph query returned no nodes"


def test_neighbors_path_runs_on_real_neo4j(seeded_workspace):
    from services.silo_graph_service import SiloGraphService

    result = SiloGraphService.get_node_neighbors(
        silo_id=SILO_ID, node_id="e1", depth=2, max_nodes=10, max_edges=100, db=None,
    )
    ids = {n["id"] for n in result["nodes"]}
    assert "e1" in ids, "root missing from neighbors result"
    assert "e2" in ids, "depth-1 neighbor missing"
    assert "e3" in ids, "depth-2 neighbor missing"
    for edge in result["edges"]:
        assert edge["source"] in ids and edge["target"] in ids, edge


def test_neighbors_unknown_root_raises(seeded_workspace):
    from services.silo_graph_service import SiloGraphService

    with pytest.raises(LookupError):
        SiloGraphService.get_node_neighbors(
            silo_id=SILO_ID, node_id="ghost", depth=1, max_nodes=10, db=None,
        )
