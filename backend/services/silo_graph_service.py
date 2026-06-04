"""Service for fetching the LightRAG knowledge graph for a silo.

The graph is stored in Neo4j with every node/relationship carrying a
``workspace`` property equal to ``silo_{silo_id}``.  This service ALWAYS
filters by workspace — cross-silo data leakage is impossible by construction.

Two execution paths:
1. **LightRAG API** — ``rag.get_knowledge_graph(node_label, max_depth, max_nodes)``
   if the installed LightRAG version exposes it.
2. **Direct Cypher fallback** — raw Neo4j driver query with mandatory workspace
   filter.  Used when the LightRAG API is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SiloGraphService:
    """Read-only graph access, always scoped to a single silo workspace."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _workspace_name(silo_id: int) -> str:
        return f"silo_{silo_id}"

    @staticmethod
    def _neo4j_driver():
        """Return a connected Neo4j driver, or raise RuntimeError on misconfiguration."""
        try:
            import config  # noqa: WPS433
        except ImportError as exc:
            raise RuntimeError("Backend config module not found") from exc

        uri = getattr(config, "NEO4J_URI", None)
        password = getattr(config, "NEO4J_PASSWORD", None)
        username = getattr(config, "NEO4J_USERNAME", None) or "neo4j"

        if not uri or not password:
            raise RuntimeError(
                "NEO4J_URI and NEO4J_PASSWORD must be set to use the graph endpoint."
            )

        from neo4j import GraphDatabase  # noqa: WPS433

        return GraphDatabase.driver(uri, auth=(username, password))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def get_silo_graph(
        cls,
        silo_id: int,
        max_nodes: int = 200,
        max_depth: int = 2,
        node_label: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch the knowledge graph for *silo_id* and return a dict matching
        :class:`~schemas.silo_graph_schemas.SiloGraphResponse`.

        Raises:
            RuntimeError: When Neo4j is unreachable or not configured.
        """
        workspace = cls._workspace_name(silo_id)

        # --- Path 1: Try LightRAG get_knowledge_graph ---
        result = cls._try_lightrag_api(
            workspace=workspace,
            max_nodes=max_nodes,
            max_depth=max_depth,
            node_label=node_label,
            search=search,
        )
        if result is not None:
            return result

        # --- Path 2: Direct Cypher (fallback) ---
        return cls._cypher_graph(
            workspace=workspace,
            max_nodes=max_nodes,
            node_label=node_label,
            search=search,
        )

    # ------------------------------------------------------------------
    # Path 1 — LightRAG API
    # ------------------------------------------------------------------

    @classmethod
    def _try_lightrag_api(
        cls,
        workspace: str,
        max_nodes: int,
        max_depth: int,
        node_label: Optional[str],
        search: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Attempt to use LightRAG's built-in graph API.

        Returns None if the method does not exist, so the caller can fall back.
        """
        try:
            from lightrag import LightRAG  # noqa: WPS433

            if not hasattr(LightRAG, "get_knowledge_graph"):
                return None

            # We need a configured LightRAG instance to call the method.
            # Import the store so we can reuse _get_rag_instance.
            from tools.vector_stores.lightrag_store import LightRAGStore  # noqa: WPS433

            store = LightRAGStore()
            rag = store._get_rag_instance(workspace)

            kwargs: Dict[str, Any] = {"max_nodes": max_nodes, "max_depth": max_depth}
            if node_label:
                kwargs["node_label"] = node_label
            if search:
                kwargs["search"] = search

            raw = rag.get_knowledge_graph(**kwargs)
            return cls._normalise_lightrag_graph(raw, max_nodes)
        except Exception as exc:
            logger.debug("LightRAG get_knowledge_graph not available: %s", exc)
            return None

    @classmethod
    def _normalise_lightrag_graph(cls, raw: Any, max_nodes: int) -> Dict[str, Any]:
        """Convert whatever structure LightRAG returns into our schema dict."""
        nodes: List[Dict] = []
        edges: List[Dict] = []

        # LightRAG may return a dict or a KnowledgeGraph-like object
        if isinstance(raw, dict):
            raw_nodes = raw.get("nodes") or raw.get("vertices") or []
            raw_edges = raw.get("edges") or raw.get("relationships") or []
        else:
            raw_nodes = getattr(raw, "nodes", []) or getattr(raw, "vertices", [])
            raw_edges = getattr(raw, "edges", []) or getattr(raw, "relationships", [])

        for n in raw_nodes:
            if isinstance(n, dict):
                nodes.append({
                    "id": str(n.get("id") or n.get("node_id") or n.get("name", "")),
                    "labels": n.get("labels") or [n.get("type", "Entity")],
                    "properties": {k: v for k, v in n.items() if k not in ("id", "node_id", "labels", "type")},
                })
            else:
                nid = str(getattr(n, "id", "") or getattr(n, "node_id", "") or getattr(n, "name", ""))
                nodes.append({
                    "id": nid,
                    "labels": getattr(n, "labels", ["Entity"]) or ["Entity"],
                    "properties": {},
                })

        for e in raw_edges:
            if isinstance(e, dict):
                edges.append({
                    "id": str(e.get("id") or f"{e.get('source')}-{e.get('target')}"),
                    "source": str(e.get("source") or e.get("src") or ""),
                    "target": str(e.get("target") or e.get("tgt") or ""),
                    "type": str(e.get("type") or e.get("label") or "RELATED"),
                    "properties": {},
                })
            else:
                edges.append({
                    "id": str(getattr(e, "id", "") or ""),
                    "source": str(getattr(e, "source", "") or ""),
                    "target": str(getattr(e, "target", "") or ""),
                    "type": str(getattr(e, "type", "RELATED") or "RELATED"),
                    "properties": {},
                })

        truncated = len(nodes) >= max_nodes
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "truncated": truncated,
        }

    # ------------------------------------------------------------------
    # Path 2 — Direct Cypher (fallback)
    # ------------------------------------------------------------------

    @classmethod
    def _cypher_graph(
        cls,
        workspace: str,
        max_nodes: int,
        node_label: Optional[str],
        search: Optional[str],
    ) -> Dict[str, Any]:
        """Query Neo4j directly with mandatory workspace filter."""
        driver = cls._neo4j_driver()
        nodes: List[Dict] = []
        edges: List[Dict] = []

        try:
            with driver.session() as neo_session:
                # --- Nodes ---
                if node_label:
                    node_q = (
                        f"MATCH (n:`{node_label}`) "
                        "WHERE n.workspace = $ws "
                    )
                else:
                    node_q = "MATCH (n) WHERE n.workspace = $ws "

                if search:
                    node_q += "AND (toLower(n.entity_name) CONTAINS toLower($search) OR toLower(n.description) CONTAINS toLower($search)) "

                node_q += "RETURN n LIMIT $limit"

                params: Dict[str, Any] = {"ws": workspace, "limit": max_nodes}
                if search:
                    params["search"] = search

                result = neo_session.run(node_q, **params)
                node_ids = set()
                for record in result:
                    n = record["n"]
                    nid = str(n.get("entity_name") or n.get("id") or n.element_id)
                    props = dict(n.items())
                    props.pop("workspace", None)
                    nodes.append({
                        "id": nid,
                        "labels": list(n.labels),
                        "properties": props,
                    })
                    node_ids.add(nid)

                # --- Edges between fetched nodes ---
                if node_ids:
                    edge_q = (
                        "MATCH (a)-[r]->(b) "
                        "WHERE a.workspace = $ws AND b.workspace = $ws "
                        "AND (a.entity_name IN $ids OR b.entity_name IN $ids) "
                        "RETURN a.entity_name AS src, b.entity_name AS tgt, type(r) AS rel_type, id(r) AS rid, properties(r) AS props "
                        "LIMIT $limit"
                    )
                    edge_result = neo_session.run(
                        edge_q,
                        ws=workspace,
                        ids=list(node_ids),
                        limit=max_nodes * 3,
                    )
                    for rec in edge_result:
                        src = str(rec["src"] or "")
                        tgt = str(rec["tgt"] or "")
                        if not src or not tgt:
                            continue
                        rel_props = dict(rec["props"] or {})
                        rel_props.pop("workspace", None)
                        edges.append({
                            "id": str(rec["rid"]),
                            "source": src,
                            "target": tgt,
                            "type": str(rec["rel_type"]),
                            "properties": rel_props,
                        })
        finally:
            driver.close()

        truncated = len(nodes) >= max_nodes
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "truncated": truncated,
        }
