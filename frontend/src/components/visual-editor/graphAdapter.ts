import { MarkerType, type Edge } from '@xyflow/react';
import type { AppGraph, GraphEdgeKind } from '../../hooks/useAppGraph';
import type { AppFlowNode } from './EntityNodeCard';
import { computeGraphLayout } from './graphLayout';
import { describeGraphNode } from './graphNodeDetail';
import { EDGE_KIND_VISUALS } from './edgeKindConfig';

/** Edge data payload - `kind` is carried through for any future custom edge rendering. */
export interface AppFlowEdgeData extends Record<string, unknown> {
  readonly kind: GraphEdgeKind;
}

// Uses React Flow's built-in bezier ("default") edge renderer - visual
// distinction between relationship kinds comes from `style`/`markerEnd`
// below, not a custom edge component, so the React Flow `type` stays
// `'default'` while the domain `kind` travels in `data`.
export type AppFlowEdge = Edge<AppFlowEdgeData, 'default'>;

/**
 * Adapts the framework-neutral `AppGraph` (from `useAppGraph`) into
 * `@xyflow/react` nodes/edges. This is the ONLY place in the visual editor
 * that imports both the graph hook's types and `@xyflow/react` - keeps
 * `useAppGraph` itself framework-neutral per sub-issue 1.
 */
export function toFlowNodes(graph: AppGraph): AppFlowNode[] {
  const positions = computeGraphLayout(graph.nodes);

  return graph.nodes.map((node) => {
    const position = positions.get(node.id) ?? { x: 0, y: 0 };
    return {
      id: node.id,
      type: node.kind,
      position,
      data: {
        label: node.label,
        detail: describeGraphNode(node),
      },
      // Read-only canvas: nodes may still be dragged for a nicer look, but
      // are never connectable/deletable from the UI.
      connectable: false,
      deletable: false,
    };
  });
}

export function toFlowEdges(graph: AppGraph): AppFlowEdge[] {
  return graph.edges.map((edge) => {
    const visual = EDGE_KIND_VISUALS[edge.kind];
    return {
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: 'default',
      data: { kind: edge.kind },
      reconnectable: false,
      focusable: false,
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
