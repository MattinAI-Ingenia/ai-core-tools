import type { GraphEdgeKind, GraphNode, GraphNodeKind } from '../../hooks/useAppGraph';
import { parseNodeId } from '../../hooks/useAppGraph';

/** Node kinds a new connection may be dragged from/to. */
export const CONNECTABLE_NODE_KINDS: ReadonlySet<GraphNodeKind> = new Set([
  'agent',
  'service',
  'silo',
  'skill',
  'mcp',
  'embedding',
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

/**
 * Non-agent node kinds that may form an agent-to-resource connection edge,
 * and the edge kind that connection produces. Usually the same string (a
 * `silo` node produces a `silo` edge), but `embedding` is the one exception:
 * an EmbeddingService node already produces a `embedding` edge kind for its
 * (immutable, non-editable) Silo -> EmbeddingService relationship, so the
 * agent-editable one needs its own distinct kind (`media_embedding`) to
 * avoid the two colliding under the same `DELETABLE_EDGE_KINDS`/mutation
 * routing.
 *
 * `service`/`media_embedding` are single-valued like `silo`: connecting a
 * new one doesn't need its own edge deleted first - `buildRelationshipChange`
 * overwrites the agent's `service_id`/`media_embedding_service_id` in place,
 * so dragging a replacement is an atomic swap rather than a remove-then-add.
 */
const AGENT_RESOURCE_PAIRS: ReadonlyArray<{ readonly resourceKind: GraphNodeKind; readonly edgeKind: GraphEdgeKind }> = [
  { resourceKind: 'service', edgeKind: 'service' },
  { resourceKind: 'silo', edgeKind: 'silo' },
  { resourceKind: 'skill', edgeKind: 'skill' },
  { resourceKind: 'mcp', edgeKind: 'mcp' },
  { resourceKind: 'embedding', edgeKind: 'media_embedding' },
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
 * pair this canvas doesn't support editing (e.g. silo<->skill, or an
 * agent<->agent drag where the target isn't a tool agent).
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

  if (sourceNode.kind === 'agent') {
    const pair = AGENT_RESOURCE_PAIRS.find((p) => p.resourceKind === targetNode.kind);
    if (pair) {
      agentNode = sourceNode;
      edgeKind = pair.edgeKind;
    }
  } else if (targetNode.kind === 'agent') {
    const pair = AGENT_RESOURCE_PAIRS.find((p) => p.resourceKind === sourceNode.kind);
    if (pair) {
      agentNode = targetNode;
      edgeKind = pair.edgeKind;
    }
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
