"""Service for fetching the LightRAG knowledge graph for a silo.

The graph is stored in Neo4j with every node/relationship carrying a
``workspace`` *label* (e.g. ``silo_14``), not a property — so it is read via
direct Cypher, matching on that label.  LightRAG's own ``get_knowledge_graph``
API filters by workspace property and returns empty results here, so it is
not used.  This service ALWAYS filters by workspace — cross-silo data leakage
is impossible by construction.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Labels cannot be parameterized in a Cypher MATCH — they are interpolated
# into a backtick-quoted label expression, so any backtick in the query param
# breaks out of it and can append arbitrary Cypher (cross-tenant reads, or
# DETACH DELETE across every workspace). Entity-type labels are identifiers,
# so this whitelist covers the real ones and rejects everything else.
_NODE_LABEL_RE = re.compile(r"^[A-Za-z0-9_\- ]{1,64}$")


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
        max_edges: int = 1000,
        max_depth: int = 2,
        node_label: Optional[str] = None,
        search: Optional[str] = None,
        db=None,
    ) -> Dict[str, Any]:
        """Fetch the knowledge graph for *silo_id* and return a dict matching
        :class:`~schemas.silo_graph_schemas.SiloGraphResponse`.

        ``db`` (optional request-scoped session) enables display-name mapping:
        LightRAG node names are the canonical merge keys (accent/case/space/
        hyphen-folded), so nodes are rewritten to their most-mentioned raw
        spelling via the ``entity_name_variant`` table — the same names the
        playground bubble shows. Without ``db``, canonical names are returned.

        Raises:
            RuntimeError: When Neo4j is unreachable or not configured.
            ValueError: When ``node_label`` is not a valid Cypher label.
        """
        if node_label and not _NODE_LABEL_RE.fullmatch(node_label):
            raise ValueError(f"Invalid node_label: {node_label!r}")

        workspace = cls._workspace_name(silo_id)
        graph = cls._cypher_graph(
            workspace=workspace,
            max_nodes=max_nodes,
            max_edges=max_edges,
            node_label=node_label,
            search=search,
        )

        # Presentation rewrite, AFTER the Cypher fetch: matching ran on the
        # canonical ids; only the returned strings carry the display variants.
        names = {n["id"] for n in graph["nodes"]}
        for edge in graph["edges"]:
            names.update((edge["source"], edge["target"]))
        if db is not None and names:
            try:
                from repositories.entity_name_variant_repository import (
                    EntityNameVariantRepository,
                )
                name_map = EntityNameVariantRepository.best_variant_map(
                    silo_id, sorted(names), db
                )
            except Exception as exc:  # noqa: WPS433 - rendering must not fail here
                logger.warning(
                    "Failed to resolve graph display names for silo %d: %s",
                    silo_id, exc,
                )
                name_map = {}
        else:
            name_map = {}
        if name_map:
            for node in graph["nodes"]:
                node["id"] = name_map.get(node["id"], node["id"])
                props = node.get("properties")
                if isinstance(props, dict) and props.get("entity_id"):
                    props["entity_id"] = name_map.get(props["entity_id"], props["entity_id"])
            for edge in graph["edges"]:
                edge["source"] = name_map.get(edge["source"], edge["source"])
                edge["target"] = name_map.get(edge["target"], edge["target"])

        return graph

    # ------------------------------------------------------------------
    # Direct Cypher query
    # ------------------------------------------------------------------

    @classmethod
    def _build_search_clause(cls, search: str, params: Dict[str, Any]) -> str:
        """Build the node-search WHERE clause.

        Two match layers:
        - raw spelling (``toLower(...) CONTAINS toLower($search)``): covers
          prose descriptions (kept raw, with accents) and legacy node names
          indexed before merge-key canonicalization;
        - canonical merge key (``n.entity_id CONTAINS $canonical_search``):
          since indexing folds names (case/accents/space/hyphen — see
          ``tools.vector_stores.lightrag.entity_name_normalization``), a
          user-typed "MCF-40" only reaches node ``mcf 40`` through the
          folded term. Skipped when the fold is empty (``"-"``, etc.)
          because ``CONTAINS ""`` matches everything.
        """
        from tools.vector_stores.lightrag.entity_name_normalization import (
            canonicalize_entity_name,
        )

        conditions = [
            "toLower(n.entity_id) CONTAINS toLower($search)",
            "toLower(n.description) CONTAINS toLower($search)",
        ]
        canonical_search = canonicalize_entity_name(search)
        if canonical_search:
            conditions.append("n.entity_id CONTAINS $canonical_search")
            params["canonical_search"] = canonical_search
        return "WHERE (" + " OR ".join(conditions) + ") "

    @classmethod
    def _cypher_graph(
        cls,
        workspace: str,
        max_nodes: int,
        max_edges: int,
        node_label: Optional[str],
        search: Optional[str],
    ) -> Dict[str, Any]:
        """Query Neo4j directly scoped to the workspace label.

        LightRAG stores the workspace as a node *label* (e.g. ``silo_14``),
        not as a ``workspace`` property.  All queries therefore match by label
        using backtick-escaped dynamic label syntax.
        """
        driver = cls._neo4j_driver()
        nodes: List[Dict] = []
        edges: List[Dict] = []
        total_nodes = 0
        total_edges = 0

        try:
            with driver.session() as neo_session:
                # --- Total node/edge counts (without limit, for the slider upper
                # bound and the "N of M" display) ---
                # Only count entity nodes (those with entity_id set) — LightRAG also
                # stores text-chunk nodes in the same workspace that don't have entity_id
                # and never appear in the UI, so the count must exclude them.
                count_q = f"MATCH (n:`{workspace}`) WHERE n.entity_id IS NOT NULL RETURN count(n) AS total"
                count_record = neo_session.run(count_q).single()
                total_nodes = count_record["total"] if count_record else 0

                edge_count_q = (
                    f"MATCH (a:`{workspace}`)-[r]->(b:`{workspace}`) "
                    "WHERE a.entity_id IS NOT NULL AND b.entity_id IS NOT NULL "
                    "RETURN count(r) AS total"
                )
                edge_count_record = neo_session.run(edge_count_q).single()
                total_edges = edge_count_record["total"] if edge_count_record else 0

                # --- Nodes ---
                # LightRAG tags every node with the workspace as a label.
                # When an additional entity-type label filter is requested,
                # both labels must match (workspace AND entity_type).
                if node_label:
                    node_q = f"MATCH (n:`{workspace}`:`{node_label}`) "
                else:
                    node_q = f"MATCH (n:`{workspace}`) "

                params: Dict[str, Any] = {"limit": max_nodes}
                if search:
                    node_q += cls._build_search_clause(search, params)
                    params["search"] = search

                node_q += (
                    "WITH n, size([(n)-[]-() | 1]) AS degree "
                    "ORDER BY degree DESC "
                    "RETURN n LIMIT $limit"
                )

                result = neo_session.run(node_q, **params)
                node_ids = set()
                for record in result:
                    n = record["n"]
                    # LightRAG stores the entity identifier in entity_id
                    nid = str(n.get("entity_id") or n.get("entity_name") or n.element_id)
                    props = dict(n.items())
                    nodes.append({
                        "id": nid,
                        # Expose only the non-workspace labels for the UI
                        "labels": [lb for lb in n.labels if lb != workspace] or list(n.labels),
                        "properties": props,
                    })
                    node_ids.add(nid)

                # --- Edges between fetched nodes ---
                # Match relationships where BOTH endpoints are in the fetched node set
                # to prevent dangling relationships that crash the renderer.
                if node_ids:
                    edge_q = (
                        f"MATCH (a:`{workspace}`)-[r]->(b) "
                        "WHERE a.entity_id IN $ids AND b.entity_id IN $ids "
                        "RETURN a.entity_id AS src, b.entity_id AS tgt, "
                        "type(r) AS rel_type, id(r) AS rid, properties(r) AS props "
                        "LIMIT $limit"
                    )
                    edge_result = neo_session.run(
                        edge_q,
                        ids=list(node_ids),
                        limit=max_edges,
                    )
                    for rec in edge_result:
                        src = str(rec["src"] or "")
                        tgt = str(rec["tgt"] or "")
                        if not src or not tgt:
                            continue
                        rel_props = dict(rec["props"] or {})
                        edges.append({
                            "id": str(rec["rid"]),
                            "source": src,
                            "target": tgt,
                            "type": str(rec["rel_type"]),
                            "properties": rel_props,
                        })
        finally:
            driver.close()

        truncated = len(nodes) >= max_nodes or len(edges) >= max_edges
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "truncated": truncated,
        }
