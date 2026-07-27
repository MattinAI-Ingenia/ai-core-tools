import type { GraphEdge, GraphNode, GraphNodeKind } from '../../hooks/useAppGraph';

/**
 * Deterministic, agent-centric initial layout (no layout library - dagre/elk
 * are intentionally not used here).
 *
 * - Agents form the visual spine: a single vertical column at x = 0, one row
 *   per agent, sorted by numeric id so the same graph always produces the
 *   same row order.
 * - Every other resource (service/silo/embedding/skill/mcp/parser) is fanned
 *   out to the right of the spine in one column per kind. Within a column, a
 *   resource is placed near the vertical average of the agent row(s) that
 *   reference it, then nudged down just enough to avoid overlapping the
 *   resource above it in the same column - so the fan stays close to its
 *   owning agent(s) while remaining readable.
 * - A resource referenced by several agents (e.g. a Silo shared by two
 *   Agents, or an EmbeddingService shared by several Silos) is only ever
 *   placed once, near the average position of ALL its referencing agents -
 *   never duplicated.
 * - Agent-as-tool edges (`kind: 'tool'`) connect two agents that are both
 *   already on the spine, so they need no separate placement: they simply
 *   read as vertical spine-to-spine connections, visually distinct from the
 *   resource fan on the right.
 *
 * Pure function of `(nodes, edges)` - same input always yields the same
 * output map, no randomness, no DOM/React Flow dependency - so it stays
 * trivially unit-testable and swappable for a smarter layout later.
 */

const AGENT_COLUMN_X = 0;
const AGENT_ROW_HEIGHT = 160;
const SATELLITE_ROW_HEIGHT = 90;
const SATELLITE_COLUMN_GAP = 260;

/** Left-to-right column order for every non-agent node kind. */
const SATELLITE_KIND_ORDER: readonly Exclude<GraphNodeKind, 'agent'>[] = [
  'service',
  'silo',
  'embedding',
  'skill',
  'mcp',
  'parser',
];

export interface GraphPosition {
  readonly x: number;
  readonly y: number;
}

/** Extracts the numeric id suffix from a namespaced node id (e.g. `agent:12` -> 12). */
function numericSuffix(id: string): number {
  const match = /:(\d+)$/.exec(id);
  return match ? Number.parseInt(match[1], 10) : Number.NaN;
}

/** Stable comparator: numeric id suffix first, falling back to lexicographic id. */
function compareNodeIds(a: string, b: string): number {
  const na = numericSuffix(a);
  const nb = numericSuffix(b);
  if (!Number.isNaN(na) && !Number.isNaN(nb) && na !== nb) {
    return na - nb;
  }
  return a.localeCompare(b);
}

interface SatelliteEntry {
  readonly node: GraphNode;
  readonly targetY: number;
  readonly hasOwners: boolean;
}

/**
 * Computes a `{ x, y }` position per node id using the agent-centric layout
 * described above.
 */
export function computeGraphLayout(
  nodes: readonly GraphNode[],
  edges: readonly GraphEdge[],
): ReadonlyMap<string, GraphPosition> {
  const positions = new Map<string, GraphPosition>();

  const agentNodes = nodes.filter((node) => node.kind === 'agent');
  const sortedAgents = [...agentNodes].sort((a, b) => compareNodeIds(a.id, b.id));

  const agentRowById = new Map<string, number>();
  sortedAgents.forEach((agent, rowIndex) => {
    agentRowById.set(agent.id, rowIndex);
    positions.set(agent.id, { x: AGENT_COLUMN_X, y: rowIndex * AGENT_ROW_HEIGHT });
  });

  // Which agent row(s) "own" each resource id, collected from direct
  // agent -> resource edges first, then propagated one hop further for
  // resources only reachable indirectly (currently just Silo -> Embedding).
  const ownerRowsByResource = new Map<string, number[]>();
  function addOwnerRows(resourceId: string, rows: readonly number[]): void {
    const existing = ownerRowsByResource.get(resourceId);
    if (existing) {
      existing.push(...rows);
    } else {
      ownerRowsByResource.set(resourceId, [...rows]);
    }
  }

  for (const edge of edges) {
    if (edge.kind === 'tool') continue; // agent -> agent, no resource placement needed
    const sourceRow = agentRowById.get(edge.source);
    if (sourceRow === undefined) continue; // handled in the indirect pass below
    addOwnerRows(edge.target, [sourceRow]);
  }

  // Indirect pass: Silo -> EmbeddingService edges have a Silo (not an Agent)
  // as their source, so an embedding's owning agents are whichever agents
  // own the silo(s) that reference it.
  for (const edge of edges) {
    if (edge.kind !== 'embedding') continue;
    if (agentRowById.has(edge.source)) continue; // no such edge today, but stay defensive
    const siloOwnerRows = ownerRowsByResource.get(edge.source);
    if (siloOwnerRows && siloOwnerRows.length > 0) {
      addOwnerRows(edge.target, siloOwnerRows);
    }
  }

  const nodesByKind = new Map<Exclude<GraphNodeKind, 'agent'>, GraphNode[]>();
  for (const node of nodes) {
    if (node.kind === 'agent') continue;
    const bucket = nodesByKind.get(node.kind);
    if (bucket) {
      bucket.push(node);
    } else {
      nodesByKind.set(node.kind, [node]);
    }
  }

  SATELLITE_KIND_ORDER.forEach((kind, columnIndex) => {
    const columnX = AGENT_COLUMN_X + (columnIndex + 1) * SATELLITE_COLUMN_GAP;
    const bucket = nodesByKind.get(kind) ?? [];

    const entries: SatelliteEntry[] = bucket.map((node) => {
      const ownerRows = ownerRowsByResource.get(node.id);
      if (ownerRows && ownerRows.length > 0) {
        const averageRow = ownerRows.reduce((sum, row) => sum + row, 0) / ownerRows.length;
        return { node, targetY: averageRow * AGENT_ROW_HEIGHT, hasOwners: true };
      }
      // Orphan resource (not yet attached to any agent) - still needs a
      // stable slot; it sorts after every owned resource in its column.
      return { node, targetY: 0, hasOwners: false };
    });

    // Deterministic order: owned resources first (by their target row),
    // orphans last, ties broken by node id so re-running never reshuffles.
    entries.sort((a, b) => {
      if (a.hasOwners !== b.hasOwners) return a.hasOwners ? -1 : 1;
      if (a.targetY !== b.targetY) return a.targetY - b.targetY;
      return compareNodeIds(a.node.id, b.node.id);
    });

    let previousY: number | undefined;
    for (const entry of entries) {
      const y = previousY === undefined ? entry.targetY : Math.max(entry.targetY, previousY + SATELLITE_ROW_HEIGHT);
      positions.set(entry.node.id, { x: columnX, y });
      previousY = y;
    }
  });

  return positions;
}
