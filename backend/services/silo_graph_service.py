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
from typing import Any, Dict, List, Optional

from tools.vector_stores.lightrag.entity_name_normalization import (
    canonicalize_entity_name,
)

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
        max_edges: int = 1000,
        search: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
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
        """
        workspace = cls._workspace_name(silo_id)
        graph = cls._cypher_graph(
            workspace=workspace,
            max_nodes=max_nodes,
            max_edges=max_edges,
            entity_types=entity_types,
            search=search,
        )

        # Presentation rewrite, AFTER the Cypher fetch: matching ran on the
        # canonical ids; only the returned strings carry the display variants.
        names = {n["id"] for n in graph["nodes"]}
        for edge in graph["edges"]:
            names.update((edge["source"], edge["target"]))
        cls._rewrite_display(graph, silo_id, names, db)

        return graph

    @classmethod
    def get_node_neighbors(
        cls,
        silo_id: int,
        node_id: str,
        depth: int,
        max_nodes: int = 200,
        max_edges: int = 1000,
        db=None,
    ) -> Dict[str, Any]:
        """Fetch the subgraph rooted at one entity node (expand/center).

        The frontend only knows DISPLAY ids (``get_silo_graph`` rewrites them),
        so *node_id* is canonicalized before the root lookup. The root and its
        depth-1 neighbors are always kept; ``max_nodes`` only trims deeper
        layers. Raises :class:`LookupError` when the canonical id has no
        entity node (router maps it to 404).
        """
        workspace = cls._workspace_name(silo_id)
        graph = cls._cypher_neighbors(
            workspace=workspace,
            root=canonicalize_entity_name(node_id),
            depth=depth,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )
        names = {n["id"] for n in graph["nodes"]}
        for edge in graph["edges"]:
            names.update((edge["source"], edge["target"]))
        cls._rewrite_display(graph, silo_id, names, db)
        return graph

    # ------------------------------------------------------------------
    # Display rewrite (shared by every graph response)
    # ------------------------------------------------------------------

    @classmethod
    def _rewrite_display(
        cls,
        graph: Dict[str, Any],
        silo_id: int,
        names: set,
        db,
    ) -> None:
        """Rewrite canonical node ids to their best display variants in place.

        After rewriting, two distinct canonical ids can collide on the same
        display variant — NVL breaks on duplicate node ids, so the colliding
        nodes are merged (nodes arrive ordered by degree DESC: the first one
        wins) and their edges are kept, remapped onto the surviving id.
        """
        if db is None or not names:
            return
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
            return
        if not name_map:
            return

        for node in graph["nodes"]:
            node["id"] = name_map.get(node["id"], node["id"])
            props = node.get("properties")
            if isinstance(props, dict) and props.get("entity_id"):
                props["entity_id"] = name_map.get(props["entity_id"], props["entity_id"])
        for edge in graph["edges"]:
            edge["source"] = name_map.get(edge["source"], edge["source"])
            edge["target"] = name_map.get(edge["target"], edge["target"])

        # Colliding nodes share the same display id by construction — the first
        # occurrence wins (nodes are degree-ordered); edges need no remap, only
        # a dedup of the pairs the merge made identical.
        merged_nodes = []
        seen_ids = set()
        for node in graph["nodes"]:
            if node["id"] in seen_ids:
                continue
            seen_ids.add(node["id"])
            merged_nodes.append(node)
        deduped_edges = []
        seen_edges = set()
        for edge in graph["edges"]:
            key = (edge["source"], edge["target"], edge["type"])
            if key in seen_edges:
                continue
            seen_edges.add(key)
            deduped_edges.append(edge)
        if len(merged_nodes) != len(graph["nodes"]):
            graph["nodes"] = merged_nodes
            graph["edges"] = deduped_edges
            graph["node_count"] = len(merged_nodes)
            graph["edge_count"] = len(deduped_edges)

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
        entity_types: Optional[List[str]] = None,
        search: Optional[str] = None,
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

                # --- Per-type counts (for the legend/filter chips) ---
                type_counts: List[Dict[str, Any]] = []
                type_count_q = (
                    f"MATCH (n:`{workspace}`) WHERE n.entity_id IS NOT NULL "
                    "RETURN n.entity_type AS type, count(*) AS count ORDER BY count DESC"
                )
                for record in neo_session.run(type_count_q):
                    type_counts.append({
                        "type": record["type"] or "Other",
                        "count": record["count"],
                    })
                type_counts = cls._cap_type_counts(type_counts)

                # --- Nodes ---
                # LightRAG stores the workspace as a label and the entity type
                # as a PROPERTY (n.entity_type), so the type filter is a
                # parameterized WHERE — never a Cypher label interpolation.
                node_q = f"MATCH (n:`{workspace}`) "

                params: Dict[str, Any] = {"limit": max_nodes}
                if search:
                    node_q += cls._build_search_clause(search, params)
                    params["search"] = search
                if entity_types:
                    # "Other" (counts map NULL/'' to it, plus every type beyond
                    # the top-10 chip cap) must match untyped AND rare-typed
                    # nodes, or its chip would filter the graph down to nothing.
                    node_q += (
                        "AND (n.entity_type IN $types OR "
                        "('Other' IN $types AND (n.entity_type IS NULL OR n.entity_type = '' "
                        "OR NOT n.entity_type IN $top))) "
                        if search else
                        "WHERE (n.entity_type IN $types OR "
                        "('Other' IN $types AND (n.entity_type IS NULL OR n.entity_type = '' "
                        "OR NOT n.entity_type IN $top))) "
                    )
                    params["types"] = entity_types
                    params["top"] = [tc["type"] for tc in type_counts if tc["type"] != "Other"]

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
                        "entity_type": props.get("entity_type"),
                    })
                    node_ids.add(nid)

                # --- Edges between fetched nodes ---
                edges = cls._fetch_induced_edges(neo_session, workspace, node_ids, max_edges)
        finally:
            driver.close()

        truncated = len(nodes) >= max_nodes or len(edges) >= max_edges
        return {
            "nodes": nodes,
            "edges": edges,
            "entity_types": type_counts,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "truncated": truncated,
        }

    TYPE_CHIP_LIMIT = 10

    # Junk type names the extraction LLM invents beyond the silo's configured
    # vocabulary — folded into the "Other" bucket so they don't waste chips.
    _JUNK_TYPE_NAMES = frozenset({"other", "otro", "autre", "altro", "unknown"})

    @classmethod
    def _cap_type_counts(cls, type_counts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep the legend usable: the extraction LLM invents type variants
        (``other``, ``UNKNOWN``, per-language drift...) — those fold into
        ``Other`` along with every type beyond the top-N chips."""
        folded: List[Dict[str, Any]] = []
        other_count = 0
        for tc in type_counts:
            if tc["type"] and tc["type"].lower() in cls._JUNK_TYPE_NAMES:
                other_count += tc["count"]
            else:
                folded.append(tc)
        other_count += sum(tc["count"] for tc in folded[cls.TYPE_CHIP_LIMIT:])
        if other_count > 0:
            folded = folded[:cls.TYPE_CHIP_LIMIT]
            folded.append({"type": "Other", "count": other_count})
        return folded

    @staticmethod
    def _fetch_induced_edges(neo_session, workspace: str, node_ids: set, max_edges: int) -> List[Dict]:
        """Edges whose BOTH endpoints are in the fetched node set (prevents
        dangling relationships that crash the renderer). Chunk nodes are
        excluded by requiring entity_id on both endpoints."""
        edges: List[Dict] = []
        if not node_ids:
            return edges
        edge_q = (
            f"MATCH (a:`{workspace}`)-[r]->(b) "
            "WHERE a.entity_id IN $ids AND b.entity_id IN $ids "
            "RETURN a.entity_id AS src, b.entity_id AS tgt, "
            "type(r) AS rel_type, id(r) AS rid, properties(r) AS props "
            "LIMIT $limit"
        )
        edge_result = neo_session.run(edge_q, ids=list(node_ids), limit=max_edges)
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
        return edges

    # ------------------------------------------------------------------
    # Rooted (expand / center) query
    # ------------------------------------------------------------------

    @classmethod
    def _cypher_neighbors(
        cls,
        workspace: str,
        root: str,
        depth: int,
        max_nodes: int,
        max_edges: int,
    ) -> Dict[str, Any]:
        """Subgraph rooted at one entity: root + all depth-1 neighbors always,
        deeper layers capped by the remaining budget (highest degree first)."""
        driver = cls._neo4j_driver()
        nodes: List[Dict] = []
        node_ids = set()
        truncated = False
        try:
            with driver.session() as neo_session:
                root_recs = list(neo_session.run(
                    f"MATCH (n:`{workspace}`) WHERE n.entity_id = $root "
                    "AND n.entity_id IS NOT NULL RETURN n LIMIT 1",
                    root=root,
                ))
                if not root_recs:
                    raise LookupError(f"Node not found: {root}")

                def _node_dict(n) -> Dict[str, Any]:
                    props = dict(n.items())
                    return {
                        "id": str(n.get("entity_id") or n.get("entity_name") or n.element_id),
                        "labels": [lb for lb in n.labels if lb != workspace] or list(n.labels),
                        "properties": props,
                        "entity_type": props.get("entity_type"),
                    }

                root_node = _node_dict(root_recs[0]["n"])
                nodes.append(root_node)
                node_ids.add(root_node["id"])

                # Depth 1: bounded by the remaining budget too — an unbounded
                # hub fetch (10k+ neighbors) would blow the payload, the edge
                # `IN` list and the NVL render. Offered > kept → truncated.
                d1_q = (
                    f"MATCH (n:`{workspace}`)-[]-(m:`{workspace}`) "
                    "WHERE n.entity_id = $root AND m.entity_id IS NOT NULL "
                    "WITH DISTINCT m, size([(m)-[]-() | 1]) AS degree "
                    "RETURN m ORDER BY degree DESC LIMIT $budget"
                )
                budget = max_nodes - len(nodes)
                d1_offered = 0
                for rec in neo_session.run(d1_q, root=root, budget=budget):
                    node = _node_dict(rec["m"])
                    # Cypher already excluded chunk nodes; the client-side check
                    # keeps the entity-only contract even with a misbehaving store.
                    if not node["properties"].get("entity_id") or node["id"] in node_ids:
                        continue
                    d1_offered += 1
                    nodes.append(node)
                    node_ids.add(node["id"])
                truncated = truncated or d1_offered >= budget

                # Deeper layers, one hop at a time from the entity frontier:
                # per-layer LIMITs keep the work bounded (a single `*2..N`
                # match would enumerate every path before LIMIT — O(d^N) from
                # a hub), and chunk-mediation can never sneak an edge-less
                # entity in because each hop only links entity to entity.
                frontier = list(node_ids)
                for _ in range(1, depth):
                    remaining = max_nodes - len(nodes)
                    if remaining <= 0:
                        truncated = True
                        break
                    layer_q = (
                        f"MATCH (n:`{workspace}`)-[]-(m:`{workspace}`) "
                        "WHERE n.entity_id IN $frontier AND m.entity_id IS NOT NULL "
                        "AND NOT m.entity_id IN $known "
                        "WITH DISTINCT m, size([(m)-[]-() | 1]) AS degree "
                        "RETURN m ORDER BY degree DESC LIMIT $budget"
                    )
                    offered = 0
                    for rec in neo_session.run(
                        layer_q, frontier=frontier, known=list(node_ids), budget=remaining,
                    ):
                        node = _node_dict(rec["m"])
                        if not node["properties"].get("entity_id") or node["id"] in node_ids:
                            continue
                        offered += 1
                        if len(nodes) >= max_nodes:
                            continue
                        nodes.append(node)
                        node_ids.add(node["id"])
                    # The layer query hit its LIMIT and kept everything it
                    # offered → more nodes exist beyond the cap.
                    truncated = truncated or offered >= remaining
                    frontier = [
                        n["properties"]["entity_id"]
                        for n in nodes[-offered:] if n["properties"].get("entity_id")
                    ] or frontier

                edges = cls._fetch_induced_edges(neo_session, workspace, node_ids, max_edges)
        finally:
            driver.close()

        type_counts: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}
        for node in nodes:
            t = node["entity_type"] or "Other"
            counts[t] = counts.get(t, 0) + 1
        type_counts = cls._cap_type_counts([
            {"type": t, "count": c} for t, c in sorted(counts.items(), key=lambda kv: -kv[1])
        ])
        return {
            "nodes": nodes,
            "edges": edges,
            "entity_types": type_counts,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "truncated": truncated,
        }
