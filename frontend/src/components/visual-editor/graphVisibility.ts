import type { GraphEdge, GraphNode } from '../../hooks/useAppGraph';

export interface VisibleGraph {
  readonly nodes: readonly GraphNode[];
  readonly edges: readonly GraphEdge[];
}

/**
 * Derives the currently-visible subset of the graph from the full graph plus
 * the set of collapsed agent ids. Pure filtering - never mutates positions,
 * so collapsing/expanding never triggers a layout recompute.
 *
 * Visibility rule:
 * - Agent nodes are ALWAYS visible. Collapsing an agent only hides ITS
 *   satellite resources (and the edges into them), never the agent itself.
 * - A non-agent resource is visible if it is reachable from at least one
 *   currently-EXPANDED agent, following every edge except `tool` edges
 *   (so Silo -> EmbeddingService chains resolve correctly). A resource
 *   shared by several agents therefore stays visible as long as ANY
 *   referencing agent is expanded.
 * - A resource with no incoming edge at all (not referenced by any agent,
 *   e.g. a Silo just created and not yet attached) has no agent to gate its
 *   visibility - it is ALWAYS visible, regardless of collapse state, so it
 *   can be dragged onto an agent from the canvas. Anything IT in turn
 *   depends on (e.g. that orphan Silo's own EmbeddingService) is pulled in
 *   the same way as an expanded agent's resources are.
 * - A `tool` edge (agent -> agent, agent-as-tool composition) is visible
 *   only while its SOURCE agent is expanded - collapsing an agent hides the
 *   tool edge it draws, but never the target agent node, which is always
 *   visible regardless of any collapse state.
 */
export function computeVisibleGraph(
  nodes: readonly GraphNode[],
  edges: readonly GraphEdge[],
  collapsedAgentIds: ReadonlySet<string>,
): VisibleGraph {
  const agentIds = new Set(nodes.filter((node) => node.kind === 'agent').map((node) => node.id));
  const expandedAgentIds = new Set([...agentIds].filter((id) => !collapsedAgentIds.has(id)));

  // Adjacency over every non-`tool` edge, used to walk from an expanded
  // agent (or an orphan resource, see below) to its (possibly indirect,
  // e.g. Silo -> EmbeddingService) resources.
  const adjacency = new Map<string, string[]>();
  for (const edge of edges) {
    if (edge.kind === 'tool') continue;
    const targets = adjacency.get(edge.source);
    if (targets) {
      targets.push(edge.target);
    } else {
      adjacency.set(edge.source, [edge.target]);
    }
  }

  const referencedResourceIds = new Set(
    edges.filter((edge) => edge.kind !== 'tool').map((edge) => edge.target),
  );
  const orphanResourceIds = new Set(
    nodes
      .filter((node) => node.kind !== 'agent' && !referencedResourceIds.has(node.id))
      .map((node) => node.id),
  );

  const visibleResourceIds = new Set<string>(orphanResourceIds);
  for (const rootId of [...expandedAgentIds, ...orphanResourceIds]) {
    const queue: string[] = [rootId];
    const seen = new Set<string>([rootId]);
    while (queue.length > 0) {
      const current = queue.shift() as string;
      for (const next of adjacency.get(current) ?? []) {
        if (seen.has(next)) continue;
        seen.add(next);
        if (!agentIds.has(next)) {
          visibleResourceIds.add(next);
        }
        queue.push(next);
      }
    }
  }

  const visibleNodeIds = new Set<string>([...agentIds, ...visibleResourceIds]);
  const visibleNodes = nodes.filter((node) => visibleNodeIds.has(node.id));
  const visibleEdges = edges.filter((edge) => {
    if (!visibleNodeIds.has(edge.source) || !visibleNodeIds.has(edge.target)) return false;
    if (edge.kind === 'tool') return expandedAgentIds.has(edge.source);
    return true;
  });

  return { nodes: visibleNodes, edges: visibleEdges };
}
