import { MarkerType, type Edge } from '@xyflow/react';
import type { GraphEdge, GraphEdgeKind, GraphNode } from '../../hooks/useAppGraph';
import type { AppFlowNode, AppFlowNodeData } from './EntityNodeCard';
import type { GraphPosition } from './graphLayout';
import { describeGraphNode } from './graphNodeDetail';
import { EDGE_KIND_VISUALS } from './edgeKindConfig';
import { CONNECTABLE_NODE_KINDS, DELETABLE_EDGE_KINDS, DELETABLE_NODE_KINDS } from './graphConnectionRules';

/** Edge data payload - `kind` is carried through for any future custom edge rendering. */
export interface AppFlowEdgeData extends Record<string, unknown> {
  readonly kind: GraphEdgeKind;
}

// Uses React Flow's built-in bezier ("default") edge renderer - visual
// distinction between relationship kinds comes from `style`/`markerEnd`
// below, not a custom edge component, so the React Flow `type` stays
// `'default'` while the domain `kind` travels in `data`.
export type AppFlowEdge = Edge<AppFlowEdgeData, 'default'>;

export interface ToFlowNodesOptions {
  /** Agent node ids currently collapsed (hiding their satellite resources). */
  readonly collapsedAgentIds: ReadonlySet<string>;
  /** Invoked with an agent node id when its collapse toggle is activated. */
  readonly onToggleCollapse: (agentId: string) => void;
  /**
   * Invoked with the original (framework-neutral) graph node when a card's
   * "Edit" affordance is activated. Optional - when omitted, no per-node
   * edit button is rendered. Deliberately untyped w.r.t. routes: this
   * module (and the canvas above it) never imports react-router, so the
   * caller (a page/thin wrapper) owns the navigation decision per entity
   * kind.
   */
  readonly onEditNode?: (node: GraphNode) => void;
}

/**
 * Adapts the framework-neutral graph nodes/edges (from `useAppGraph`) into
 * `@xyflow/react` nodes/edges. This is the ONLY place in the visual editor
 * that imports both the graph hook's types and `@xyflow/react` - keeps
 * `useAppGraph` itself framework-neutral per sub-issue 1.
 *
 * `positions` is a precomputed `nodeId -> {x,y}` map (the deterministic
 * default layout merged with any saved/dragged positions by the caller) -
 * this function never computes layout itself, so filtering the graph down
 * to its currently-visible subset (collapse/expand) never triggers a
 * layout recompute.
 */
export function toFlowNodes(
  nodes: readonly GraphNode[],
  positions: ReadonlyMap<string, GraphPosition>,
  { collapsedAgentIds, onToggleCollapse, onEditNode }: ToFlowNodesOptions,
): AppFlowNode[] {
  return nodes.map((node) => {
    const position = positions.get(node.id) ?? { x: 0, y: 0 };
    const data: AppFlowNodeData = {
      label: node.label,
      detail: describeGraphNode(node),
      ...(node.kind === 'agent'
        ? {
            collapsed: collapsedAgentIds.has(node.id),
            onToggleCollapse: () => onToggleCollapse(node.id),
          }
        : {}),
      ...(onEditNode ? { onEdit: () => onEditNode(node) } : {}),
    };

    return {
      id: node.id,
      type: node.kind,
      position,
      data,
      // Interactive canvas: only the 4 editable relationship kinds' node
      // kinds may originate a new connection. Agent/Silo/Skill nodes may
      // also be deleted outright (backed by their own delete endpoint, with
      // a confirmation dialog - see AppGraphCanvas's onBeforeDelete).
      connectable: CONNECTABLE_NODE_KINDS.has(node.kind),
      deletable: DELETABLE_NODE_KINDS.has(node.kind),
    };
  });
}

export function toFlowEdges(edges: readonly GraphEdge[]): AppFlowEdge[] {
  return edges.map((edge) => {
    const visual = EDGE_KIND_VISUALS[edge.kind];
    return {
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: 'default',
      data: { kind: edge.kind },
      reconnectable: false,
      deletable: DELETABLE_EDGE_KINDS.has(edge.kind),
      focusable: false,
      // Wider than React Flow's own default (20px) - these edges are thin
      // (1.5-2px) and easy to miss on a precise click.
      interactionWidth: 40,
      style: {
        stroke: visual.stroke,
        strokeWidth: visual.strokeWidth,
        strokeDasharray: visual.dashed ? '6 4' : undefined,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: visual.stroke,
        width: 16,
        height: 16,
      },
    };
  });
}
