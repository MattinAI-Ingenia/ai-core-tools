import { useCallback, useEffect, useMemo, useRef, useState, type ComponentType } from 'react';
import {
  Background,
  BackgroundVariant,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlow,
  applyEdgeChanges,
  applyNodeChanges,
  type Edge,
  type EdgeMouseHandler,
  type NodeMouseHandler,
  type NodeProps,
  type OnBeforeDelete,
  type OnConnect,
  type OnEdgesChange,
  type OnEdgesDelete,
  type OnNodeDrag,
  type OnNodesChange,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { AlertTriangle, Loader2, RefreshCw } from 'lucide-react';
import { toast } from 'sonner';
import { apiService, type Agent } from '../../../services/api';
import { MESSAGES, errorMessage } from '../../../constants/messages';
import { PIECE_VISUALS } from '../nodeKindConfig';
import {
  buildPortPatch,
  connectedPieces,
  portsFor,
  type AgentType,
  type PieceKind,
  type PortId,
} from './agentPorts';
import { canConnect, canDisconnect } from './connectionRules';
import { loadLayout, saveLayout, type CanvasLayout, type StorageLike } from './canvasLayout';
import { createSaveQueue, planConnect, type SaveStep } from './savePlan';
import { AgentPortNode, type CanvasFlowNode } from './AgentPortNode';
import { PieceNode } from './PieceNode';
import { useAgentCanvasData, type CanvasPiece } from './useAgentCanvasData';

/** What the inspector panel (later task) receives when something is selected. */
export type CanvasSelection =
  | { type: 'agent' }
  | { type: 'piece'; piece: CanvasPiece }
  | { type: 'edge'; port: PortId; piece: CanvasPiece };

export interface AgentCanvasProps {
  readonly appId: number;
  readonly agentId: number;
  /** False for roles below EDITOR: no connecting, no disconnecting (panning/dragging still works). */
  readonly canEdit: boolean;
  /** Invoked with the selection on click; null when the pane is clicked. */
  readonly onSelect?: (selection: CanvasSelection | null) => void;
}

interface CanvasEdgeData extends Record<string, unknown> {
  readonly port: PortId;
  readonly piece: CanvasPiece;
}

type CanvasEdge = Edge<CanvasEdgeData, 'default'>;

const NODE_TYPES: Record<string, ComponentType<NodeProps<CanvasFlowNode>>> = {
  agent: AgentPortNode,
  piece: PieceNode,
};

/** Column of connected pieces to the left of the agent, where all port handles sit. */
const PIECE_COLUMN_X = -320;
const PIECE_ROW_HEIGHT = 90;

/** localStorage access itself can throw (blocked storage); the canvas works without it. */
function safeLocalStorage(): StorageLike | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

/** Splits a piece node id (`${kind}:${id}`) back into its parts. */
function parsePieceNodeId(nodeId: string): { kind: PieceKind; id: number } | null {
  const separator = nodeId.indexOf(':');
  if (separator <= 0) return null;
  const kind = nodeId.slice(0, separator) as PieceKind;
  const id = Number.parseInt(nodeId.slice(separator + 1), 10);
  return Number.isNaN(id) ? null : { kind, id };
}

/**
 * Default positions: the agent sits at the origin and its connected pieces
 * line up in one column to its left, grouped by port in `portsFor` order.
 * Positions saved in localStorage (from an earlier drag) override the default.
 */
function buildFlowNodes(
  agent: Agent,
  agentType: AgentType,
  connected: ReadonlyArray<{ port: PortId; kind: PieceKind; id: number }>,
  pieces: readonly CanvasPiece[],
  activePortIds: readonly PortId[],
  savedPositions: Readonly<Record<string, { x: number; y: number }>>,
  canEdit: boolean,
): CanvasFlowNode[] {
  const ports = portsFor(agentType);
  const portOrder = new Map(ports.map((port, index) => [port.id, index]));
  const byPieceId = new Map(pieces.map((piece) => [`${piece.kind}:${piece.id}`, piece]));

  const nodes: CanvasFlowNode[] = [
    {
      id: `agent:${agent.agent_id}`,
      type: 'agent',
      position: savedPositions[`agent:${agent.agent_id}`] ?? { x: 0, y: 0 },
      data: {
        label: agent.name,
        typeCaption: agentType === 'ocr_agent' ? 'OCR agent' : 'Agent',
        ports: ports.filter((port) => !port.secondary),
        secondaryPorts: ports.filter((port) => port.secondary),
        connectedPortIds: activePortIds,
      },
      // Nodes are never deletable here: resources are managed from their own pages.
      deletable: false,
      connectable: canEdit,
    },
  ];

  const ordered = [...connected].sort(
    (a, b) => (portOrder.get(a.port) ?? 0) - (portOrder.get(b.port) ?? 0) || a.id - b.id,
  );
  ordered.forEach((entry, index) => {
    const piece = byPieceId.get(`${entry.kind}:${entry.id}`);
    if (!piece) return;
    nodes.push({
      id: `${entry.kind}:${entry.id}`,
      type: 'piece',
      position:
        savedPositions[`${entry.kind}:${entry.id}`] ??
        { x: PIECE_COLUMN_X, y: (index - (ordered.length - 1) / 2) * PIECE_ROW_HEIGHT },
      data: { label: piece.name, piece, portLabel: ports[portOrder.get(entry.port) ?? 0].label },
      deletable: false,
      connectable: canEdit,
    });
  });
  return nodes;
}

function makeCanvasEdge(args: {
  port: PortId;
  piece: CanvasPiece;
  targetId: string;
  id?: string;
  pending?: boolean;
}): CanvasEdge {
  const color = PIECE_VISUALS[args.piece.kind].accentHex;
  return {
    id: args.id ?? `${args.port}:${args.piece.kind}:${args.piece.id}`,
    source: `${args.piece.kind}:${args.piece.id}`,
    target: args.targetId,
    targetHandle: args.port,
    type: 'default',
    deletable: true,
    focusable: false,
    reconnectable: false,
    interactionWidth: 40,
    data: { port: args.port, piece: args.piece },
    style: { stroke: color, strokeWidth: 1.5, ...(args.pending ? { strokeDasharray: '6 4' } : {}) },
    markerEnd: { type: MarkerType.ArrowClosed, color, width: 16, height: 16 },
  };
}

function buildFlowEdges(
  agentNodeId: string,
  connected: ReadonlyArray<{ port: PortId; kind: PieceKind; id: number }>,
  pieces: readonly CanvasPiece[],
): CanvasEdge[] {
  const byPieceId = new Map(pieces.map((piece) => [`${piece.kind}:${piece.id}`, piece]));
  const edges: CanvasEdge[] = [];
  for (const entry of connected) {
    const piece = byPieceId.get(`${entry.kind}:${entry.id}`);
    if (!piece) continue;
    edges.push(makeCanvasEdge({ port: entry.port, piece, targetId: agentNodeId }));
  }
  return edges;
}

function CanvasLoading() {
  return (
    <div
      className="flex h-[70vh] w-full min-h-[400px] flex-col items-center justify-center gap-3 rounded-xl border border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-800/50"
      role="status"
      aria-live="polite"
    >
      <Loader2
        className="h-8 w-8 animate-spin motion-reduce:animate-none text-indigo-500 dark:text-indigo-400"
        aria-hidden="true"
      />
      <p className="text-sm text-gray-500 dark:text-gray-400">{MESSAGES.LOADING('agent canvas')}</p>
    </div>
  );
}

function CanvasError({ error, onRetry }: { readonly error: string; readonly onRetry: () => void }) {
  return (
    <div
      className="flex h-[70vh] w-full min-h-[400px] flex-col items-center justify-center gap-3 rounded-xl border border-red-200 bg-red-50 px-6 text-center dark:border-red-800/50 dark:bg-red-900/20"
      role="alert"
      aria-live="assertive"
    >
      <AlertTriangle className="h-8 w-8 text-red-500 dark:text-red-400" aria-hidden="true" />
      <p className="max-w-md text-sm text-red-700 dark:text-red-300">{error}</p>
      <button
        type="button"
        onClick={onRetry}
        className="inline-flex items-center gap-1.5 rounded-lg border border-red-300 bg-white px-3 py-1.5 text-sm font-medium text-red-700 transition-colors hover:bg-red-50 dark:border-red-700 dark:bg-gray-800 dark:text-red-300 dark:hover:bg-red-900/30"
      >
        <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
        Try again
      </button>
    </div>
  );
}

/**
 * Per-agent visual canvas: the edited agent in the middle with one named
 * port per field it connects through, connected pieces around it, and
 * drag-to-connect/select+Delete editing backed by the pure ports model,
 * connection rules and save queue. Layout (positions + loose pieces) lives
 * only in localStorage; connections save immediately, one step per gesture.
 */
export function AgentCanvas({ appId, agentId, canEdit, onSelect }: AgentCanvasProps) {
  const { agent, pieces, toolEdges, loading, error, refetch } = useAgentCanvasData(appId, agentId);
  const storage = useMemo(safeLocalStorage, []);
  const queue = useMemo(() => createSaveQueue(), []);

  const agentType: AgentType = agent?.type === 'ocr_agent' ? 'ocr_agent' : 'agent';
  const agentNodeId = agent ? `agent:${agent.agent_id}` : '';

  const connected = useMemo(
    () => (agent ? connectedPieces(agentType, agent) : []),
    [agent, agentType],
  );

  // Saved layout scoped to this app+agent. Node ids include every existing
  // piece (not just connected ones) so a disconnected piece keeps its
  // position in storage and `loose` ids survive the round trip.
  const layout = useMemo<CanvasLayout>(() => {
    if (!agent) return { positions: {}, loose: [] };
    const known = new Set<string>([agentNodeId, ...pieces.map((piece) => `${piece.kind}:${piece.id}`)]);
    return loadLayout(storage, appId, agentId, known);
  }, [agent, agentNodeId, pieces, storage, appId, agentId]);

  const [pendingEdges, setPendingEdges] = useState<CanvasEdge[]>([]);

  // Ports with a pending connection also render their handle, so the dashed
  // optimistic edge has somewhere to land until the refetch resolves.
  const activePortIds = useMemo(() => {
    const ids = new Set<PortId>(connected.map((entry) => entry.port));
    for (const edge of pendingEdges) if (edge.data) ids.add(edge.data.port);
    return [...ids];
  }, [connected, pendingEdges]);

  const flowNodes = useMemo(
    () =>
      agent
        ? buildFlowNodes(agent, agentType, connected, pieces, activePortIds, layout.positions, canEdit)
        : [],
    [agent, agentType, connected, pieces, activePortIds, layout.positions, canEdit],
  );

  const flowEdges = useMemo(
    () => (agent ? buildFlowEdges(agentNodeId, connected, pieces) : []),
    [agent, agentNodeId, connected, pieces],
  );

  // Local, draggable copies of the adapted nodes/edges - same pattern as the
  // old app-wide canvas: React Flow owns position/selection between
  // re-renders, and a resync preserves the live values.
  const [nodes, setNodes] = useState<CanvasFlowNode[]>([]);
  const nodesRef = useRef<readonly CanvasFlowNode[]>(nodes);
  useEffect(() => {
    nodesRef.current = nodes;
  }, [nodes]);

  useEffect(() => {
    setNodes((current) => {
      const byId = new Map(current.map((node) => [node.id, node]));
      return flowNodes.map((node) => {
        const existing = byId.get(node.id);
        return existing ? { ...node, position: existing.position } : node;
      });
    });
  }, [flowNodes]);

  const [edges, setEdges] = useState<CanvasEdge[]>([]);
  useEffect(() => {
    setEdges((current) => {
      const byId = new Map(current.map((edge) => [edge.id, edge]));
      return [...flowEdges, ...pendingEdges].map((edge) => {
        const existing = byId.get(edge.id);
        return existing ? { ...edge, selected: existing.selected } : edge;
      });
    });
  }, [flowEdges, pendingEdges]);

  const onNodesChange: OnNodesChange<CanvasFlowNode> = useCallback((changes) => {
    setNodes((current) => applyNodeChanges(changes, current));
  }, []);

  const onEdgesChange: OnEdgesChange<CanvasEdge> = useCallback((changes) => {
    setEdges((current) => applyEdgeChanges(changes, current));
  }, []);

  const handleNodeDragStop: OnNodeDrag<CanvasFlowNode> = useCallback(
    (_event, draggedNode) => {
      const positions: Record<string, { x: number; y: number }> = { ...layout.positions };
      for (const node of nodesRef.current) {
        positions[node.id] = node.id === draggedNode.id ? draggedNode.position : node.position;
      }
      saveLayout(storage, appId, agentId, { positions, loose: layout.loose });
    },
    [layout.positions, layout.loose, storage, appId, agentId],
  );

  /** Saves `steps` one at a time (fetch → merge patch → update), stopping at the first failure; then refetches. */
  const runSteps = useCallback(
    async (steps: readonly SaveStep[]): Promise<void> => {
      for (const step of steps) {
        try {
          await queue.run(step.agentId, async () => {
            const fresh = await apiService.getAgent(appId, step.agentId);
            await apiService.updateAgent(appId, step.agentId, { ...fresh, ...step.patch });
            if (step.toast) toast.success(step.toast);
          });
        } catch (err) {
          toast.error(errorMessage(err, MESSAGES.UPDATE_FAILED('connection')));
          break;
        }
      }
      await refetch();
    },
    [appId, queue, refetch],
  );

  const handleConnect: OnConnect = useCallback(
    (connection) => {
      if (!agent) return;
      // The agent body is always the edge target; normalize either drag direction.
      const toAgent = connection.target === agentNodeId;
      const fromAgent = connection.source === agentNodeId;
      const pieceNodeId = toAgent ? connection.source : fromAgent ? connection.target : null;
      const rawPortId = toAgent ? connection.targetHandle : fromAgent ? connection.sourceHandle : null;
      if (!pieceNodeId || typeof rawPortId !== 'string') return;
      const parsed = parsePieceNodeId(pieceNodeId);
      const piece = parsed
        ? pieces.find((entry) => entry.kind === parsed.kind && entry.id === parsed.id)
        : undefined;
      const port = portsFor(agentType).find((entry) => entry.id === (rawPortId as PortId));
      if (!parsed || !piece || !port) return;

      const result = canConnect({
        canEdit,
        agentId: agent.agent_id,
        agentType,
        hasMemory: agent.has_memory === true,
        port,
        kind: piece.kind,
        pieceId: piece.id,
        lifecycleStatus: piece.lifecycleStatus,
        toolEdges,
      });
      if (result.ok === false) {
        toast.error(result.reason);
        return;
      }
      if (result.warning) toast.warning(result.warning);

      const patch = buildPortPatch(agent, port.id, piece.kind, piece.id, 'add');
      const replaced = port.multiple
        ? null
        : connected.find((entry) => entry.port === port.id && entry.id !== piece.id);
      const replacedName = replaced
        ? (pieces.find((entry) => entry.kind === replaced.kind && entry.id === replaced.id)?.name ?? null)
        : null;
      const steps = planConnect({
        agentId: agent.agent_id,
        port: port.id,
        kind: piece.kind,
        pieceId: piece.id,
        patch,
        targetIsTool: piece.isTool,
        codeInterpreterOn: agent.enable_code_interpreter === true,
        replacedName,
        pieceName: piece.name,
      });

      const pendingId = `pending:${port.id}:${piece.kind}:${piece.id}`;
      setPendingEdges((current) => [
        ...current.filter((edge) => edge.id !== pendingId),
        makeCanvasEdge({ port: port.id, piece, targetId: agentNodeId, id: pendingId, pending: true }),
      ]);

      void runSteps(steps).finally(() => {
        setPendingEdges((current) => current.filter((edge) => edge.id !== pendingId));
      });
    },
    [agent, agentNodeId, agentType, pieces, canEdit, toolEdges, connected, runSteps],
  );

  const handleBeforeDelete: OnBeforeDelete<CanvasFlowNode, CanvasEdge> = useCallback(
    async ({ edges: candidateEdges }) => {
      const ports = agent ? portsFor(agentType) : [];
      const allowed: CanvasEdge[] = [];
      const blockedReasons = new Set<string>();
      for (const edge of candidateEdges) {
        // A pending (not yet saved) edge must not be disconnected: on a
        // single-valued port the remove patch would null the field and wipe
        // the connection that is already there. It disappears on its own
        // once the gesture's save settles.
        if (edge.id.startsWith('pending:')) continue;
        const port = edge.data ? ports.find((entry) => entry.id === edge.data?.port) : undefined;
        if (!agent || !port) continue;
        const result = canDisconnect({ canEdit, agentType, hasMemory: agent.has_memory === true, port });
        if (result.ok === false) {
          blockedReasons.add(result.reason);
          continue;
        }
        allowed.push(edge);
      }
      for (const reason of blockedReasons) toast.error(reason);
      return { nodes: [], edges: allowed };
    },
    [agent, agentType, canEdit],
  );

  const handleEdgesDelete: OnEdgesDelete<CanvasEdge> = useCallback(
    (removed) => {
      if (!agent) return;
      const ports = portsFor(agentType);
      const removedData = removed.flatMap((edge) => (edge.data ? [edge.data] : []));
      if (removedData.length === 0) return;
      let succeeded = false;
      void (async () => {
        for (const data of removedData) {
          const port = ports.find((entry) => entry.id === data.port);
          if (!port) continue;
          try {
            await queue.run(agent.agent_id, async () => {
              const fresh = await apiService.getAgent(appId, agent.agent_id);
              await apiService.updateAgent(appId, agent.agent_id, {
                ...fresh,
                ...buildPortPatch(fresh, port.id, data.piece.kind, data.piece.id, 'remove'),
              });
            });
            succeeded = true;
          } catch (err) {
            toast.error(errorMessage(err, MESSAGES.UPDATE_FAILED('connection')));
          }
        }
        if (succeeded) toast.success(MESSAGES.UPDATED('connection'));
        await refetch();
      })();
    },
    [agent, agentType, appId, queue, refetch],
  );

  const handleNodeClick: NodeMouseHandler<CanvasFlowNode> = useCallback(
    (_event, node) => {
      if (!onSelect) return;
      if (node.type === 'agent') {
        onSelect({ type: 'agent' });
        return;
      }
      if (node.data.piece) onSelect({ type: 'piece', piece: node.data.piece });
    },
    [onSelect],
  );

  const handleEdgeClick: EdgeMouseHandler<CanvasEdge> = useCallback(
    (_event, edge) => {
      if (edge.data) onSelect?.({ type: 'edge', port: edge.data.port, piece: edge.data.piece });
    },
    [onSelect],
  );

  const handlePaneClick = useCallback(() => {
    onSelect?.(null);
  }, [onSelect]);

  // Inline `style` outranks React Flow's `.selected` CSS class, so overlay a
  // bold consistent "selected" look at render time (same as the old canvas).
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

  // Only show the full-screen loading state on the genuinely-empty initial
  // load; a background refetch must never unmount the flow (same as the old
  // canvas - it would reset the viewport on every connect/disconnect).
  if (loading && !agent) return <CanvasLoading />;
  if (error) return <CanvasError error={error} onRetry={refetch} />;
  if (!agent) return <CanvasLoading />;

  return (
    <div className="h-[70vh] w-full min-h-[400px] overflow-hidden rounded-xl border border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-900">
      <h2 className="sr-only">Agent canvas</h2>
      <ReactFlow<CanvasFlowNode, CanvasEdge>
        nodes={nodes}
        edges={styledEdges}
        nodeTypes={NODE_TYPES}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeDragStop={handleNodeDragStop}
        onConnect={handleConnect}
        onBeforeDelete={handleBeforeDelete}
        onEdgesDelete={handleEdgesDelete}
        onNodeClick={handleNodeClick}
        onEdgeClick={handleEdgeClick}
        onPaneClick={handlePaneClick}
        nodesConnectable={canEdit}
        elementsSelectable
        nodesDraggable
        // react-flow's default (0px) requires the pointer to not move AT ALL
        // between mousedown and mouseup to register a click.
        nodeClickDistance={10}
        nodesFocusable={false}
        edgesReconnectable={false}
        edgesFocusable={false}
        deleteKeyCode={['Delete', 'Backspace']}
        proOptions={{ hideAttribution: true }}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.2}
        maxZoom={1.5}
      >
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} className="dark:opacity-30" />
        <MiniMap
          pannable
          zoomable
          className="!bg-white dark:!bg-gray-800 [&_.react-flow__minimap-mask]:!fill-gray-300/60 dark:[&_.react-flow__minimap-mask]:!fill-gray-600/40"
        />
        <Controls showInteractive={false} className="[&_button]:!bg-white [&_button]:dark:!bg-gray-800 [&_button]:dark:!fill-gray-300 [&_button]:dark:!border-gray-700" />
      </ReactFlow>
    </div>
  );
}
