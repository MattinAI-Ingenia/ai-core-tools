import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { applyEdgeChanges, applyNodeChanges, type OnEdgesChange, type OnNodeDrag, type OnNodesChange } from '@xyflow/react';
import { toast } from 'sonner';
import { useAppGraph, type GraphNode } from '../../hooks/useAppGraph';
import { GraphFlowView } from './GraphFlowView';
import { toFlowNodes, toFlowEdges } from './graphAdapter';
import { computeGraphLayout, type GraphPosition } from './graphLayout';
import { computeVisibleGraph } from './graphVisibility';
import { useGraphLayoutStorage } from './useGraphLayoutStorage';
import { useGraphMutations } from './useGraphMutations';
import { isValidGraphConnection } from './graphConnectionRules';
import type { AppFlowNode } from './EntityNodeCard';
import type { AppFlowEdge } from './graphAdapter';

export interface AppGraphCanvasProps {
  readonly appId: string | number | undefined;
  readonly className?: string;
  /**
   * Invoked with the original graph node when a card's "Edit" button is
   * activated. Optional - omit to render the canvas without any edit
   * affordance. Deliberately generic (no react-router import here): the
   * page hosting this canvas owns the actual navigation decision per
   * entity kind.
   */
  readonly onEditNode?: (node: GraphNode) => void;
}

/**
 * Read-only React Flow canvas for an App's resource graph. Fetches the graph
 * via `useAppGraph`, adapts it to `@xyflow/react` nodes/edges, and layers on
 * top of that:
 * - a deterministic agent-centric default layout (`computeGraphLayout`),
 * - agent-centric collapse/expand of satellite resources (`computeVisibleGraph`),
 * - `localStorage` persistence of dragged positions + collapsed agents
 *   (`useGraphLayoutStorage`), scoped per app.
 *
 * All actual rendering lives in the pure `GraphFlowView` so that component
 * stays easy to unit test with plain fixtures. Relationship editing is backed
 * by `useGraphMutations` (connect/disconnect) which syncs 4 editable relationship
 * kinds—silo/skill/tool/mcp—to the backend via drag-to-connect and select+delete;
 * layout and collapse state remain client-side only (localStorage).
 */
export function AppGraphCanvas({ appId, className, onEditNode }: AppGraphCanvasProps) {
  const { nodes: graphNodes, edges: graphEdges, loading, error, refetch } = useAppGraph(appId);
  const { load, save } = useGraphLayoutStorage(appId);
  const { pendingEdges, hiddenEdgeIds, connect, disconnectMany } = useGraphMutations({
    appId,
    graphNodes,
    onSettled: refetch,
  });

  // Only show the full-screen loading state on the genuinely-empty initial
  // load. `loading` also flips true during every background refetch that
  // `onSettled` triggers after a mutation - if that raw value reached
  // `GraphFlowView` it would unmount/remount `<ReactFlow>` on every
  // connect/disconnect (resetting `fitView`'s viewport), defeating this
  // canvas's optimistic-update intent. Once there's already a rendered
  // graph, a background refetch should never unmount it.
  const showFullScreenLoading = loading && graphNodes.length === 0;

  const [collapsedAgentIds, setCollapsedAgentIds] = useState<ReadonlySet<string>>(() => new Set());

  // Accumulates every node id's last-known position, INCLUDING ids that are
  // currently hidden by a collapsed agent - so re-expanding an agent (or a
  // later refetch) restores a drag that happened before the node was hidden,
  // instead of snapping back to its default layout position. Seeded from
  // `localStorage` on mount/app-change; a plain ref because it must survive
  // renders without itself triggering one, and is only ever mutated inside
  // event handlers/effects below, never during render.
  const savedPositionsRef = useRef<Readonly<Record<string, GraphPosition>>>({});

  // Hydrate from localStorage whenever the app changes (covers both the
  // initial mount and navigating the canvas to a different app id). This
  // never WRITES to storage, only reads - persistence itself only happens
  // from the drag-stop/collapse-toggle handlers below, in response to an
  // actual user action.
  useEffect(() => {
    const initial = load();
    savedPositionsRef.current = initial.positions;
    setCollapsedAgentIds(initial.collapsedAgentIds);
  }, [load]);

  // Deterministic default layout, computed over the FULL graph (not just
  // the currently-visible subset) so collapsing/expanding an agent never
  // shifts any other node's default position.
  const defaultPositions = useMemo(
    () => computeGraphLayout(graphNodes, graphEdges),
    [graphNodes, graphEdges],
  );

  const visibleGraph = useMemo(
    () => computeVisibleGraph(graphNodes, graphEdges, collapsedAgentIds),
    [graphNodes, graphEdges, collapsedAgentIds],
  );

  // Local, draggable copy of the adapted (currently-visible) nodes. React
  // Flow owns node position while the user drags; this is intentionally NOT
  // derived via useMemo because it must survive re-renders that don't touch
  // the graph data itself (e.g. React Flow's own dimension-measurement
  // changes).
  const [nodes, setNodes] = useState<AppFlowNode[]>([]);

  // Mirrors `nodes` for the drag-stop/collapse-toggle handlers below, so
  // THEIR identities stay stable while dragging (every mouse-move updates
  // `nodes` state via `onNodesChange`) instead of depending on `nodes`
  // directly, which would otherwise recreate them - and, transitively,
  // `flowNodes` and the merge effect above - on every pointer move.
  const nodesRef = useRef<readonly AppFlowNode[]>(nodes);
  useEffect(() => {
    nodesRef.current = nodes;
  }, [nodes]);

  /** Merges `currentNodes`' live positions into the accumulator and persists it. */
  const persistLayout = useCallback(
    (nextCollapsedAgentIds: ReadonlySet<string>, currentNodes: readonly AppFlowNode[]) => {
      const mergedPositions: Record<string, GraphPosition> = { ...savedPositionsRef.current };
      for (const node of currentNodes) {
        mergedPositions[node.id] = node.position;
      }
      savedPositionsRef.current = mergedPositions;

      const knownNodeIds = new Set(graphNodes.map((node) => node.id));
      save({ positions: mergedPositions, collapsedAgentIds: nextCollapsedAgentIds }, knownNodeIds);
    },
    [graphNodes, save],
  );

  const handleToggleCollapse = useCallback(
    (agentId: string) => {
      const next = new Set(collapsedAgentIds);
      if (next.has(agentId)) {
        next.delete(agentId);
      } else {
        next.add(agentId);
      }
      setCollapsedAgentIds(next);
      persistLayout(next, nodesRef.current);
    },
    [collapsedAgentIds, persistLayout],
  );

  const flowNodes = useMemo(
    () =>
      toFlowNodes(visibleGraph.nodes, defaultPositions, {
        collapsedAgentIds,
        onToggleCollapse: handleToggleCollapse,
        onEditNode,
      }),
    [visibleGraph.nodes, defaultPositions, collapsedAgentIds, handleToggleCollapse, onEditNode],
  );
  const flowEdges = useMemo(() => toFlowEdges(visibleGraph.edges), [visibleGraph.edges]);
  const displayedEdges = useMemo(
    () => [...flowEdges.filter((edge) => !hiddenEdgeIds.has(edge.id)), ...pendingEdges],
    [flowEdges, hiddenEdgeIds, pendingEdges],
  );

  // Local, selectable copy of `displayedEdges`, mirroring the `nodes` state
  // above. React Flow computes click-to-select (and every other pointer
  // interaction) as a "change" it hands to `onEdgesChange` - without owned
  // state to apply that change to, a click has nowhere to record `selected`,
  // so edges could never be selected (and therefore never deleted via the
  // canvas's own delete-key handling, which acts on the currently-selected
  // edges).
  const [edges, setEdges] = useState<AppFlowEdge[]>([]);

  // Re-seeds from `displayedEdges` whenever the visible/optimistic edge set
  // changes, preserving each edge's live `selected` state across that resync
  // (same rationale as the node position merge above).
  useEffect(() => {
    setEdges((current) => {
      const currentById = new Map(current.map((edge) => [edge.id, edge]));
      return displayedEdges.map((edge) => {
        const existing = currentById.get(edge.id);
        return existing ? { ...edge, selected: existing.selected } : edge;
      });
    });
  }, [displayedEdges]);

  const onEdgesChange: OnEdgesChange<AppFlowEdge> = (changes) => {
    setEdges((current) => applyEdgeChanges(changes, current));
  };

  // React Flow's own delete-key handling silently drops any selected edge
  // whose `deletable` is false before `onEdgesDelete` ever fires - so a
  // read-only relationship (e.g. Silo -> EmbeddingService) can be selected
  // and Delete pressed with zero feedback. This listener runs independently
  // of that pipeline purely to surface a toast for the read-only edges in
  // the current selection; deletable edges in the same selection are still
  // removed normally via `onEdgesDelete`/`disconnectMany`.
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key !== 'Delete' && event.key !== 'Backspace') return;
      if (event.target instanceof HTMLElement && ['INPUT', 'TEXTAREA'].includes(event.target.tagName)) return;

      const hasBlockedSelection = edges.some((edge) => edge.selected && edge.deletable === false);
      if (hasBlockedSelection) {
        toast.error("This relationship is read-only and can't be deleted from the canvas.");
      }
    }
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [edges]);

  // `toFlowEdges` sets a fixed per-kind stroke/width so relationship kinds
  // stay visually distinct - but that inline `style` also outranks React
  // Flow's default CSS for `.selected` edges (inline style beats a CSS
  // class), so a selected edge otherwise looks identical to an unselected
  // one. Overlaying a bold, consistent "selected" style here (applied at
  // render time, over the `edges` state itself) gives the user visible
  // confirmation of what select+Delete is about to remove.
  const styledEdges = useMemo(
    () =>
      edges.map((edge) =>
        edge.selected
          ? {
              ...edge,
              style: { ...edge.style, stroke: '#2563eb', strokeWidth: 3 },
              markerEnd:
                typeof edge.markerEnd === 'object' && edge.markerEnd !== null
                  ? { ...edge.markerEnd, color: '#2563eb' }
                  : edge.markerEnd,
            }
          : edge,
      ),
    [edges],
  );

  // Re-seeds from `flowNodes` whenever the visible node set changes (new
  // fetch/refetch OR a collapse/expand toggle) - but preserves the
  // in-memory position of every node id already present in the current
  // state (so a refetch never drops a drag), and falls back to the
  // persisted position (if any) before the layout default for genuinely new
  // ids, e.g. a resource just re-appearing after its owning agent expanded.
  useEffect(() => {
    setNodes((current) => {
      const currentById = new Map(current.map((node) => [node.id, node]));
      return flowNodes.map((flowNode) => {
        const existing = currentById.get(flowNode.id);
        if (existing) {
          return { ...flowNode, position: existing.position };
        }
        const saved = savedPositionsRef.current[flowNode.id];
        return saved ? { ...flowNode, position: saved } : flowNode;
      });
    });
  }, [flowNodes]);

  const onNodesChange: OnNodesChange<AppFlowNode> = (changes) => {
    setNodes((current) => applyNodeChanges(changes, current));
  };

  const handleNodeDragStop: OnNodeDrag<AppFlowNode> = useCallback(
    (_event, draggedNode) => {
      const updatedNodes = nodesRef.current.map((node) =>
        node.id === draggedNode.id ? { ...node, position: draggedNode.position } : node,
      );
      persistLayout(collapsedAgentIds, updatedNodes);
    },
    [collapsedAgentIds, persistLayout],
  );

  const handleIsValidConnection = useCallback(
    (connection: Parameters<typeof isValidGraphConnection>[1]) => isValidGraphConnection(graphNodes, connection),
    [graphNodes],
  );

  const handleEdgesDelete = useCallback(
    (edges: readonly AppFlowEdge[]) => {
      disconnectMany(edges);
    },
    [disconnectMany],
  );

  return (
    <GraphFlowView
      nodes={nodes}
      edges={styledEdges}
      loading={showFullScreenLoading}
      error={error}
      onNodesChange={onNodesChange}
      onNodeDragStop={handleNodeDragStop}
      onEdgesChange={onEdgesChange}
      onConnect={connect}
      onEdgesDelete={handleEdgesDelete}
      isValidConnection={handleIsValidConnection}
      onRetry={refetch}
      className={className}
    />
  );
}
