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

/** Node kinds that may be deleted from the canvas (select + Delete/Backspace). */
export const DELETABLE_NODE_KINDS: ReadonlySet<GraphNodeKind> = new Set([
  'agent',
  'silo',
  'skill',
]);

/** Non-agent node kinds that may form an agent-to-resource connection edge. */
const AGENT_RESOURCE_EDGE_KINDS: ReadonlySet<GraphEdgeKind> = new Set([
  'silo',
  'skill',
  'mcp',
]);

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

  let agentNode: GraphNode | null = null;
  let edgeKind: GraphEdgeKind | null = null;

  if (sourceNode.kind === 'agent' && AGENT_RESOURCE_EDGE_KINDS.has(targetNode.kind as GraphEdgeKind)) {
    agentNode = sourceNode;
    edgeKind = targetNode.kind as GraphEdgeKind;
  } else if (targetNode.kind === 'agent' && AGENT_RESOURCE_EDGE_KINDS.has(sourceNode.kind as GraphEdgeKind)) {
    agentNode = targetNode;
    edgeKind = sourceNode.kind as GraphEdgeKind;
  }

  if (!agentNode || !edgeKind) return null;

  const resourceNode = sourceNode.kind === 'agent' ? targetNode : sourceNode;
  return {
    kind: edgeKind,
    agentNumericId: parseNodeId(agentNode.id).numericId,
    targetNumericId: parseNodeId(resourceNode.id).numericId,
  };
}

export function isValidGraphConnection(nodes: readonly GraphNode[], connection: MinimalConnection): boolean {
  return resolveConnection(nodes, connection) !== null;
}
