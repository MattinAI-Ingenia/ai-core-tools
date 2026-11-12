"""``LightRAGStore._cleanup_neo4j`` drops the workspace's nodes AND its indexes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tools.vector_stores.lightrag_store import LightRAGStore


def test_cleanup_neo4j_deletes_nodes_and_workspace_indexes():
    session = MagicMock()
    # First run() is the node delete (result unused); second lists the indexes.
    session.run.side_effect = [
        MagicMock(),
        [{"name": "entity_id_fulltext_idx_silo_7"}, {"name": "index_abc"}],
        MagicMock(),
        MagicMock(),
    ]
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session

    store = LightRAGStore.__new__(LightRAGStore)
    with patch("config.NEO4J_URI", "bolt://x", create=True), \
         patch("config.NEO4J_PASSWORD", "pw", create=True), \
         patch("neo4j.GraphDatabase.driver", return_value=driver):
        store._cleanup_neo4j("silo_7")

    queries = [c.args[0] for c in session.run.call_args_list]
    assert queries[0] == "MATCH (n:`silo_7`) DETACH DELETE n"
    assert session.run.call_args_list[1].kwargs == {"label": "silo_7"}
    assert queries[2:] == [
        "DROP INDEX `entity_id_fulltext_idx_silo_7` IF EXISTS",
        "DROP INDEX `index_abc` IF EXISTS",
    ]
