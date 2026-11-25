// Pure helpers for the silo knowledge-graph explorer. No React here, so the rules are testable.

/** Types for the silo graph API payload (mirrors schemas.silo_graph_schemas). */
export interface GraphNode {
  id: string;
  labels: string[];
  properties: Record<string, unknown>;
  entity_type?: string | null;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  properties: Record<string, unknown>;
}

export interface GraphEntityType {
  type: string;
  count: number;
}

export interface SiloGraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  entity_types: GraphEntityType[];
  node_count: number;
  edge_count: number;
  total_nodes: number;
  total_edges: number;
  truncated: boolean;
  max_nodes: number;
}

/** A source document a node/edge was extracted from (GET /graph/sources). */
export interface GraphSourceDoc {
  chunk_id: string;
  resource_id?: number | null;
  page?: number | null;
  file_name?: string | null;
}

/** LightRAG joins the chunk ids an element was extracted from with this separator. */
const SOURCE_SEP = '<SEP>';

export function splitSourceIds(sourceId: string | undefined | null): string[] {
  if (!sourceId) return [];
  return sourceId
    .split(SOURCE_SEP)
    .map((id) => id.trim())
    .filter(Boolean);
}

/** Merge a rooted-neighbors fetch into the loaded graph: dedupe by id, drop
 *  edges whose endpoints are missing, and never exceed the canvas cap. */
export function mergeGraphData(base: SiloGraphData, incoming: SiloGraphData): SiloGraphData {
  const nodes = [...base.nodes];
  const ids = new Set(nodes.map((n) => n.id));
  const cap = Math.max(base.max_nodes, nodes.length);
  for (const node of incoming.nodes) {
    if (ids.has(node.id) || nodes.length >= cap) continue;
    ids.add(node.id);
    nodes.push(node);
  }

  const edges = [...base.edges];
  const seenEdges = new Set(edges.map((e) => e.id));
  for (const edge of incoming.edges) {
    if (seenEdges.has(edge.id) || !ids.has(edge.source) || !ids.has(edge.target)) continue;
    seenEdges.add(edge.id);
    edges.push(edge);
  }

  return {
    ...base,
    nodes,
    edges,
    node_count: nodes.length,
    edge_count: edges.length,
  };
}

/** The neighbors of a node among the loaded edges, deduped, both directions. */
export function neighborsOf(
  nodeId: string,
  edges: GraphEdge[],
): Array<{ relatedId: string; type: string }> {
  const out: Array<{ relatedId: string; type: string }> = [];
  const seen = new Set<string>();
  for (const e of edges) {
    const other = e.source === nodeId ? e.target : e.target === nodeId ? e.source : null;
    if (other === null) continue;
    if (seen.has(other)) continue;
    seen.add(other);
    out.push({ relatedId: other, type: e.type });
  }
  return out;
}
