import { useEffect, useMemo, useState } from 'react';
import { applyNodeChanges, type OnNodesChange } from '@xyflow/react';
import { useAppGraph } from '../../hooks/useAppGraph';
import { GraphFlowView } from './GraphFlowView';
import { toFlowNodes, toFlowEdges } from './graphAdapter';
import type { AppFlowNode } from './EntityNodeCard';

export interface AppGraphCanvasProps {
  readonly appId: string | number | undefined;
  readonly className?: string;
}

/**
 * Read-only React Flow canvas for an App's resource graph. Fetches the graph
 * via `useAppGraph` and adapts it to `@xyflow/react` nodes/edges; all actual
 * rendering lives in the pure `GraphFlowView` so that component stays easy
 * to unit test with plain fixtures.
 *
 * No routing/page wiring and no drag-position persistence here - both are
 * later sub-issues. Dragging nodes only updates in-memory canvas state.
 */
export function AppGraphCanvas({ appId, className }: AppGraphCanvasProps) {
  const { nodes: graphNodes, edges: graphEdges, loading, error, refetch } = useAppGraph(appId);

  const flowNodes = useMemo(
    () => toFlowNodes({ nodes: graphNodes, edges: graphEdges }),
    [graphNodes, graphEdges],
  );
  const flowEdges = useMemo(
    () => toFlowEdges({ nodes: graphNodes, edges: graphEdges }),
    [graphNodes, graphEdges],
  );

  // Local, draggable copy of the adapted nodes. React Flow owns node
  // position while the user drags; this is intentionally NOT derived via
  // useMemo because it must survive re-renders that don't touch the graph
  // data itself (e.g. React Flow's own dimension-measurement changes).
  // It re-seeds from `flowNodes` only when the underlying graph data
  // actually changes (new fetch/refetch) - a documented React Flow pattern
  // for controlled canvases fed by async data, not simple derived UI state.
  const [nodes, setNodes] = useState<AppFlowNode[]>(flowNodes);

  useEffect(() => {
    setNodes(flowNodes);
  }, [flowNodes]);

  const onNodesChange: OnNodesChange<AppFlowNode> = (changes) => {
    setNodes((current) => applyNodeChanges(changes, current));
  };

  return (
    <GraphFlowView
      nodes={nodes}
      edges={flowEdges}
      loading={loading}
      error={error}
      onNodesChange={onNodesChange}
      onRetry={refetch}
      className={className}
    />
  );
}
