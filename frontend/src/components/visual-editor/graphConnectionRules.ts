import type { GraphEdgeKind, GraphNode, GraphNodeKind } from '../../hooks/useAppGraph';
import { parseNodeId } from '../../hooks/useAppGraph';

/** Node kinds a new connection may be dragged from/to. */
export const CONNECTABLE_NODE_KINDS: ReadonlySet<GraphNodeKind> = new Set([
  'agent',
  'silo',
  'skill',
  'mcp',
]);

/** Edge kinds that may be removed from the canvas (select + Delete/Backspace). */
export const DELETABLE_EDGE_KINDS: ReadonlySet<GraphEdgeKind> = new Set([
  'silo',
  'skill',
  'mcp',
  'tool',
]);

/** Non-agent node kinds that pair with an agent to form a valid connection. */
const AGENT_RESOURCE_PAIRS: ReadonlyArray<{ readonly kind: GraphEdgeKind; readonly resourceKind: GraphNodeKind }> = [
  { kind: 'silo', resourceKind: 'silo' },
  { kind: 'skill', resourceKind: 'skill' },
  { kind: 'mcp', resourceKind: 'mcp' },
];

export interface ResolvedConnection {
  readonly kind: GraphEdgeKind;
  /** The agent that owns the relationship (always the "using" agent for tool edges). */
  readonly agentNumericId: number;
  /** The connected resource, or the tool agent for tool edges. */
  readonly targetNumericId: number;
}

interface MinimalConnection {
  readonly source: string | null | undefined;
  readonly target: string | null | undefined;
}

function isToolAgent(node: GraphNode): boolean {
  const data = node.data as { is_tool?: unknown };
  return data.is_tool === true;
}

/**
 * Resolves a drag-to-connect gesture into the relationship it represents,
 * normalizing direction so the agent always ends up as `agentNumericId`
 * regardless of which handle the user dragged from. Returns `null` for any
 * pair this canvas doesn't support editing (including out-of-scope kinds
 * like agent<->service, and invalid tool connections).
 */
export function resolveConnection(
  nodes: readonly GraphNode[],
  connection: MinimalConnection,
): ResolvedConnection | null {
  const { source, target } = connection;
  if (!source || !target || source === target) return null;

  const sourceNode = nodes.find((node) => node.id === source);
  const targetNode = nodes.find((node) => node.id === target);
  if (!sourceNode || !targetNode) return null;

  if (sourceNode.kind === 'agent' && targetNode.kind === 'agent') {
    if (!isToolAgent(targetNode)) return null;
    return {
      kind: 'tool',
      agentNumericId: parseNodeId(source).numericId,
      targetNumericId: parseNodeId(target).numericId,
    };
  }

  const pair = AGENT_RESOURCE_PAIRS.find(
    ({ resourceKind }) =>
      (sourceNode.kind === 'agent' && targetNode.kind === resourceKind) ||
      (targetNode.kind === 'agent' && sourceNode.kind === resourceKind),
  );
  if (!pair) return null;

  const agentNode = sourceNode.kind === 'agent' ? sourceNode : targetNode;
  const resourceNode = sourceNode.kind === 'agent' ? targetNode : sourceNode;
  return {
    kind: pair.kind,
    agentNumericId: parseNodeId(agentNode.id).numericId,
    targetNumericId: parseNodeId(resourceNode.id).numericId,
  };
}

export function isValidGraphConnection(nodes: readonly GraphNode[], connection: MinimalConnection): boolean {
  return resolveConnection(nodes, connection) !== null;
}
