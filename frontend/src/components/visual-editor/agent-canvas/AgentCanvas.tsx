import { useCallback, useEffect, useId, useMemo, useRef, useState, type ComponentType, type DragEvent } from 'react';
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
  type OnInit,
  type OnNodeDrag,
  type OnNodesChange,
  type OnNodesDelete,
  type ReactFlowInstance,
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
  resolveDropPort,
  type AgentType,
  type PieceKind,
  type PortDef,
  type PortId,
} from './agentPorts';
import { canConnect, canDisconnect } from './connectionRules';
import { layoutKey, loadLayout, saveLayout, type CanvasLayout, type StorageLike } from './canvasLayout';
import { createSaveQueue, planConnect, type SaveStep } from './savePlan';
import { AgentPortNode, type CanvasFlowNode } from './AgentPortNode';
import { PieceNode } from './PieceNode';
import { CanvasPalette, PIECE_DRAG_MIME, parsePieceDragPayload, pieceKindPhrase } from './CanvasPalette';
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

/**
 * A save step plus, when it targets the edited agent, the connection intent
 * behind it. The port patch is rebuilt from the freshly fetched agent inside
 * the queued task, so two rapid connects on the same port can't overwrite
 * each other's addition (`step.patch` still carries the plan's extras such
 * as `enable_code_interpreter`).
 */
interface CanvasSaveStep {
  readonly step: SaveStep;
  readonly connect?: { readonly port: PortId; readonly kind: PieceKind; readonly pieceId: number };
}

/** A piece dropped on the agent where several ports accept it: the user picks one. */
interface PendingChoice {
  readonly piece: CanvasPiece;
  readonly ports: readonly PortDef[];
}

/** Layout written during this session; wins over the stored copy for the same app+agent. */
interface EditedLayout {
  readonly scope: string;
  readonly layout: CanvasLayout;
}

const NODE_TYPES: Record<string, ComponentType<NodeProps<CanvasFlowNode>>> = {
  agent: AgentPortNode,
  piece: PieceNode,
};

/** Column of connected pieces to the left of the agent, where all port handles sit. */
const PIECE_COLUMN_X = -320;
/** Default column for loose pieces (no saved position), to the right of the agent. */
const LOOSE_COLUMN_X = 320;
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

/** Node id of a piece card (`${kind}:${id}`), the same key the layout stores. */
function pieceNodeId(piece: Pick<CanvasPiece, 'kind' | 'id'>): string {
  return `${piece.kind}:${piece.id}`;
}

/**
 * Default positions: the agent sits at the origin and its connected pieces
 * line up in one column to its left, grouped by port in `portsFor` order.
 * Loose pieces (on the canvas but unconnected) sit in a column to the right.
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
  looseNodeIds: readonly string[],
): CanvasFlowNode[] {
  const ports = portsFor(agentType);
  const portOrder = new Map(ports.map((port, index) => [port.id, index]));
  const byPieceId = new Map(pieces.map((piece) => [pieceNodeId(piece), piece]));

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
  // One card per piece even when it feeds several ports (e.g. one AI service
  // on Model and Transcription); the port labels aggregate in port order.
  const deduped: Array<{ nodeId: string; piece: CanvasPiece; portLabels: string[] }> = [];
  const dedupeIndex = new Map<string, number>();
  for (const entry of ordered) {
    const piece = byPieceId.get(`${entry.kind}:${entry.id}`);
    if (!piece) continue;
    const nodeId = `${entry.kind}:${entry.id}`;
    const label = ports[portOrder.get(entry.port) ?? 0].label;
    const seenAt = dedupeIndex.get(nodeId);
    if (seenAt !== undefined) {
      deduped[seenAt].portLabels.push(label);
      continue;
    }
    dedupeIndex.set(nodeId, deduped.length);
    deduped.push({ nodeId, piece, portLabels: [label] });
  }
  deduped.forEach((entry, index) => {
    nodes.push({
      id: entry.nodeId,
      type: 'piece',
      position:
        savedPositions[entry.nodeId] ??
        { x: PIECE_COLUMN_X, y: (index - (deduped.length - 1) / 2) * PIECE_ROW_HEIGHT },
      data: { label: entry.piece.name, piece: entry.piece, portLabels: entry.portLabels },
      deletable: false,
      connectable: canEdit,
    });
  });
  looseNodeIds.forEach((nodeId, index) => {
    const piece = byPieceId.get(nodeId);
    if (!piece) return;
    nodes.push({
      id: nodeId,
      type: 'piece',
      position: savedPositions[nodeId] ?? { x: LOOSE_COLUMN_X, y: index * PIECE_ROW_HEIGHT },
      data: { label: piece.name, piece, loose: true },
      // Loose pieces only leave the canvas (layout); they are deletable when editable.
      deletable: canEdit,
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
  const byPieceId = new Map(pieces.map((piece) => [pieceNodeId(piece), piece]));
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

interface PortChooserProps {
  readonly name: string;
  readonly ports: readonly PortDef[];
  readonly onPick: (port: PortDef) => void;
  readonly onCancel: () => void;
}

/** Small card listing the ports a dropped piece can connect through. Escape or a click outside cancels. */
function PortChooser({ name, ports, onPick, onCancel }: PortChooserProps) {
  const headingId = useId();
  const containerRef = useRef<HTMLDivElement>(null);
  const firstPortRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    firstPortRef.current?.focus();
  }, []);

  useEffect(() => {
    const handlePointerDown = (event: MouseEvent) => {
      if (event.target instanceof Node && containerRef.current?.contains(event.target)) return;
      onCancel();
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onCancel();
    };
    document.addEventListener('mousedown', handlePointerDown);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handlePointerDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [onCancel]);

  return (
    <div
      ref={containerRef}
      role="dialog"
      aria-labelledby={headingId}
      className="absolute left-1/2 top-3 z-20 w-60 -translate-x-1/2 rounded-xl border border-gray-200 bg-white p-3 shadow-lg dark:border-gray-700 dark:bg-gray-800"
    >
      <p id={headingId} className="text-sm font-medium text-gray-900 dark:text-gray-100">
        Connect {name} to:
      </p>
      <div className="mt-2 flex flex-col gap-1">
        {ports.map((port, index) => (
          <button
            key={port.id}
            ref={index === 0 ? firstPortRef : undefined}
            type="button"
            onClick={() => onPick(port)}
            className="rounded-lg px-3 py-1.5 text-left text-sm text-gray-800 transition-colors hover:bg-indigo-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 dark:text-gray-100 dark:hover:bg-indigo-900/30"
          >
            {port.label}
          </button>
        ))}
      </div>
    </div>
  );
}

/**
 * Per-agent visual canvas: the edited agent in the middle with one named
 * port per field it connects through, connected pieces around it, and
 * drag-to-connect/select+Delete editing backed by the pure ports model,
 * connection rules and save queue. Pieces dragged from the palette and
 * dropped on the agent connect; dropped elsewhere they stay as loose cards.
 * Layout (positions + loose pieces) lives only in localStorage; connections
 * save immediately, one step per gesture.
 */
export function AgentCanvas({ appId, agentId, canEdit, onSelect }: AgentCanvasProps) {
  const { agent, pieces, toolEdges, loading, error, refetch } = useAgentCanvasData(appId, agentId);
  const storage = useMemo(safeLocalStorage, []);
  const queue = useMemo(() => createSaveQueue(), []);
  const scope = layoutKey(appId, agentId);

  const agentType: AgentType = agent?.type === 'ocr_agent' ? 'ocr_agent' : 'agent';
  const agentNodeId = agent ? `agent:${agent.agent_id}` : '';

  const connected = useMemo(
    () => (agent ? connectedPieces(agentType, agent) : []),
    [agent, agentType],
  );
  const connectedNodeIds = useMemo(
    () => new Set(connected.map((entry) => `${entry.kind}:${entry.id}`)),
    [connected],
  );

  const [editedLayout, setEditedLayout] = useState<EditedLayout | null>(null);

  // Saved layout scoped to this app+agent. Node ids include every existing
  // piece (not just connected ones) so a disconnected piece keeps its
  // position in storage and `loose` ids survive the round trip.
  const layout = useMemo<CanvasLayout>(() => {
    if (editedLayout?.scope === scope) return editedLayout.layout;
    if (!agent) return { positions: {}, loose: [] };
    const known = new Set<string>([agentNodeId, ...pieces.map((piece) => `${piece.kind}:${piece.id}`)]);
    return loadLayout(storage, appId, agentId, known);
  }, [agent, agentNodeId, pieces, storage, appId, agentId, scope, editedLayout]);

  /** Loose pieces that are still unconnected (a piece that got connected leaves the canvas as loose). */
  const looseNodeIds = useMemo(
    () => layout.loose.filter((nodeId) => !connectedNodeIds.has(nodeId)),
    [layout.loose, connectedNodeIds],
  );

  // Latest layout for callbacks that finish after an await (a connect's
  // save, a drag stop) so they never write back a stale copy.
  const layoutRef = useRef<CanvasLayout>(layout);
  useEffect(() => {
    layoutRef.current = layout;
  }, [layout]);

  /** Applies `mutate` to the latest layout, saves it and renders it. */
  const updateLayout = useCallback(
    (mutate: (current: CanvasLayout) => CanvasLayout) => {
      const next = mutate(layoutRef.current);
      layoutRef.current = next;
      saveLayout(storage, appId, agentId, next);
      setEditedLayout({ scope: layoutKey(appId, agentId), layout: next });
    },
    [storage, appId, agentId],
  );

  const [pendingEdges, setPendingEdges] = useState<CanvasEdge[]>([]);
  /** Gesture token for pending-edge ids, so one gesture's cleanup can't cross into another's. */
  const pendingSeq = useRef(0);

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
        ? buildFlowNodes(
            agent,
            agentType,
            connected,
            pieces,
            activePortIds,
            layout.positions,
            canEdit,
            looseNodeIds,
          )
        : [],
    [agent, agentType, connected, pieces, activePortIds, layout.positions, canEdit, looseNodeIds],
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

  // A failed background refetch (canvas already populated) keeps the live
  // view and reports through a toast; the full error screen is only for the
  // initial load.
  useEffect(() => {
    if (error && agent) toast.error(error);
  }, [error, agent]);

  /** Flow instance, captured on init: screen-to-flow conversion and hit tests for drops. */
  const flowRef = useRef<ReactFlowInstance<CanvasFlowNode, CanvasEdge> | null>(null);
  const handleInit: OnInit<CanvasFlowNode, CanvasEdge> = useCallback((instance) => {
    flowRef.current = instance;
  }, []);

  const [choice, setChoice] = useState<PendingChoice | null>(null);
  const cancelChoice = useCallback(() => setChoice(null), []);
  // The chooser belongs to the agent it was opened for.
  useEffect(() => {
    setChoice(null);
  }, [agentId]);

  /**
   * Removes a node from the live copy so the next sync re-adds it at its
   * layout position: the sync keeps live positions for nodes it already has.
   */
  const reseatNode = useCallback((nodeId: string) => {
    setNodes((current) => current.filter((node) => node.id !== nodeId));
  }, []);

  const onNodesChange: OnNodesChange<CanvasFlowNode> = useCallback((changes) => {
    setNodes((current) => applyNodeChanges(changes, current));
  }, []);

  const onEdgesChange: OnEdgesChange<CanvasEdge> = useCallback((changes) => {
    setEdges((current) => applyEdgeChanges(changes, current));
  }, []);

  /** Saves `steps` one at a time (fetch → merge patch → update), stopping at the first failure; then refetches. Resolves true when every step saved. */
  const runSteps = useCallback(
    async (steps: readonly CanvasSaveStep[]): Promise<boolean> => {
      let saved = true;
      for (const { step, connect } of steps) {
        try {
          await queue.run(step.agentId, async () => {
            const fresh = await apiService.getAgent(appId, step.agentId);
            // Rebuild the port patch from the fresh agent so rapid gestures on
            // the same port don't overwrite each other; step.patch still
            // carries the plan's extras (enable_code_interpreter).
            const patch = connect
              ? { ...step.patch, ...buildPortPatch(fresh, connect.port, connect.kind, connect.pieceId, 'add') }
              : step.patch;
            await apiService.updateAgent(appId, step.agentId, { ...fresh, ...patch });
            if (step.toast) toast.success(step.toast);
          });
        } catch (err) {
          toast.error(errorMessage(err, MESSAGES.UPDATE_FAILED('connection')));
          saved = false;
          break;
        }
      }
      await refetch();
      return saved;
    },
    [appId, queue, refetch],
  );

  /**
   * The one connect path: every gesture that connects a piece (a handle drag
   * or a drop on the agent) ends here. Rule check, then the planned saves are
   * queued and refetched; a loose piece leaves `loose` once its save succeeds.
   */
  const connectPiece = useCallback(
    (port: PortDef, piece: CanvasPiece): void => {
      if (!agent) return;
      if (connected.some((entry) => entry.port === port.id && entry.kind === piece.kind && entry.id === piece.id)) {
        toast.info(`${piece.name} is already connected`);
        return;
      }
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
      // planConnect decides the steps and toasts; steps for the edited agent
      // carry the connection intent so runSteps rebuilds the port patch from
      // the fresh agent inside the queue (the is_tool step saves another
      // agent and keeps its patch as is).
      const steps: CanvasSaveStep[] = planConnect({
        agentId: agent.agent_id,
        port: port.id,
        kind: piece.kind,
        pieceId: piece.id,
        patch,
        targetIsTool: piece.isTool,
        codeInterpreterOn: agent.enable_code_interpreter === true,
        replacedName,
        pieceName: piece.name,
      }).map((step) =>
        step.agentId === agent.agent_id
          ? { step, connect: { port: port.id, kind: piece.kind, pieceId: piece.id } }
          : { step },
      );

      // Unique id per gesture: one gesture's cleanup must never remove
      // another gesture's still-in-flight pending edge.
      pendingSeq.current += 1;
      const pendingId = `pending:${port.id}:${piece.kind}:${piece.id}:${pendingSeq.current}`;
      setPendingEdges((current) => [
        ...current,
        makeCanvasEdge({ port: port.id, piece, targetId: agentNodeId, id: pendingId, pending: true }),
      ]);

      const nodeId = pieceNodeId(piece);
      void runSteps(steps)
        .then((saved) => {
          if (saved && layoutRef.current.loose.includes(nodeId)) {
            updateLayout((current) => ({
              positions: current.positions,
              loose: current.loose.filter((id) => id !== nodeId),
            }));
          }
        })
        .finally(() => {
          setPendingEdges((current) => current.filter((edge) => edge.id !== pendingId));
        });
    },
    [agent, agentNodeId, agentType, pieces, canEdit, toolEdges, connected, runSteps, updateLayout],
  );

  /** A piece dropped on the agent body: one matching port connects directly, several open the chooser, none is refused. */
  const dropOnAgent = useCallback(
    (piece: CanvasPiece): void => {
      const target = resolveDropPort(agentType, piece.kind);
      if (target === null) {
        toast.error(`${pieceKindPhrase(piece.kind)} can't be connected to this agent`);
        return;
      }
      const ports = portsFor(agentType);
      if (target === 'ask') {
        setChoice({ piece, ports: ports.filter((port) => port.accepts.includes(piece.kind)) });
        return;
      }
      const port = ports.find((entry) => entry.id === target);
      if (port) connectPiece(port, piece);
    },
    [agentType, connectPiece],
  );

  const handleChoosePort = useCallback(
    (port: PortDef) => {
      if (!choice) return;
      setChoice(null);
      connectPiece(port, choice.piece);
    },
    [choice, connectPiece],
  );

  /** True when `point` (flow coordinates) lies on the agent node. */
  const isOverAgent = useCallback(
    (point: { x: number; y: number }): boolean => {
      const instance = flowRef.current;
      if (!instance || !agentNodeId) return false;
      return instance
        .getIntersectingNodes({ x: point.x, y: point.y, width: 1, height: 1 })
        .some((node) => node.id === agentNodeId);
    },
    [agentNodeId],
  );

  const handleNodeDragStop: OnNodeDrag<CanvasFlowNode> = useCallback(
    (_event, draggedNode) => {
      const positions: Record<string, { x: number; y: number }> = {};
      for (const node of nodesRef.current) {
        positions[node.id] = node.id === draggedNode.id ? draggedNode.position : node.position;
      }
      // A loose piece dragged over the agent connects. Its saved position is
      // dropped so it lands in the default column once connected.
      const piece = draggedNode.data.piece;
      const center = {
        x: draggedNode.position.x + (draggedNode.measured?.width ?? 0) / 2,
        y: draggedNode.position.y + (draggedNode.measured?.height ?? 0) / 2,
      };
      const droppedOnAgent =
        canEdit && piece !== undefined && looseNodeIds.includes(draggedNode.id) && isOverAgent(center);
      updateLayout((current) => {
        const merged = { ...current.positions, ...positions };
        if (droppedOnAgent) delete merged[draggedNode.id];
        return { positions: merged, loose: current.loose };
      });
      if (droppedOnAgent) reseatNode(draggedNode.id);
      if (droppedOnAgent && piece) dropOnAgent(piece);
    },
    [canEdit, looseNodeIds, isOverAgent, reseatNode, dropOnAgent, updateLayout],
  );

  const handleConnect: OnConnect = useCallback(
    (connection) => {
      if (!agent) return;
      // The agent body is always the edge target; normalize either drag direction.
      const toAgent = connection.target === agentNodeId;
      const fromAgent = connection.source === agentNodeId;
      const pieceNode = toAgent ? connection.source : fromAgent ? connection.target : null;
      const rawPortId = toAgent ? connection.targetHandle : fromAgent ? connection.sourceHandle : null;
      if (!pieceNode || typeof rawPortId !== 'string') return;
      const parsed = parsePieceNodeId(pieceNode);
      const piece = parsed
        ? pieces.find((entry) => entry.kind === parsed.kind && entry.id === parsed.id)
        : undefined;
      const port = portsFor(agentType).find((entry) => entry.id === (rawPortId as PortId));
      if (!parsed || !piece || !port) return;
      connectPiece(port, piece);
    },
    [agent, agentNodeId, agentType, pieces, connectPiece],
  );

  const handleBeforeDelete: OnBeforeDelete<CanvasFlowNode, CanvasEdge> = useCallback(
    async ({ nodes: candidateNodes, edges: candidateEdges }) => {
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
      // Only loose pieces may be removed, and only from the canvas: no API call.
      const looseNodes = canEdit
        ? candidateNodes.filter((node) => node.type === 'piece' && looseNodeIds.includes(node.id))
        : [];
      return { nodes: looseNodes, edges: allowed };
    },
    [agent, agentType, canEdit, looseNodeIds],
  );

  const handleNodesDelete: OnNodesDelete<CanvasFlowNode> = useCallback(
    (removed) => {
      const removedIds = new Set(removed.map((node) => node.id));
      updateLayout((current) => ({
        positions: current.positions,
        loose: current.loose.filter((id) => !removedIds.has(id)),
      }));
    },
    [updateLayout],
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

  const handleDragOver = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      if (!canEdit || !event.dataTransfer.types.includes(PIECE_DRAG_MIME)) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = 'copy';
    },
    [canEdit],
  );

  const handleDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      const payload = parsePieceDragPayload(event.dataTransfer.getData(PIECE_DRAG_MIME));
      if (!canEdit || !agent || !payload) return;
      event.preventDefault();
      // A new drop replaces any chooser still open from an earlier one.
      setChoice(null);
      const piece = pieces.find((entry) => entry.kind === payload.kind && entry.id === payload.id);
      const instance = flowRef.current;
      if (!piece || !instance) return;
      const point = instance.screenToFlowPosition({ x: event.clientX, y: event.clientY });
      if (isOverAgent(point)) {
        dropOnAgent(piece);
        return;
      }
      // Dropped elsewhere: the piece stays on the canvas as a loose card. A
      // piece that is already connected is already on the canvas.
      const nodeId = pieceNodeId(piece);
      if (connectedNodeIds.has(nodeId)) {
        toast.info(`${piece.name} is already on the canvas`);
        return;
      }
      reseatNode(nodeId);
      updateLayout((current) => ({
        positions: { ...current.positions, [nodeId]: point },
        loose: current.loose.includes(nodeId) ? current.loose : [...current.loose, nodeId],
      }));
    },
    [canEdit, agent, pieces, connectedNodeIds, isOverAgent, reseatNode, dropOnAgent, updateLayout],
  );

  // Placeholder until the creation modals arrive (Task 9 replaces it).
  const handleCreate = useCallback(() => toast.info('Coming soon'), []);

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
  if (error && !agent) return <CanvasError error={error} onRetry={refetch} />;
  if (!agent) return <CanvasLoading />;

  return (
    <div className="flex w-full flex-col gap-3 md:flex-row">
      <CanvasPalette pieces={pieces} agentType={agentType} canEdit={canEdit} onCreate={handleCreate} />
      <div
        className="relative h-[70vh] w-full min-h-[400px] min-w-0 flex-1 overflow-hidden rounded-xl border border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-900"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <h2 className="sr-only">Agent canvas</h2>
        <ReactFlow<CanvasFlowNode, CanvasEdge>
          nodes={nodes}
          edges={styledEdges}
          nodeTypes={NODE_TYPES}
          onInit={handleInit}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeDragStop={handleNodeDragStop}
          onConnect={handleConnect}
          onBeforeDelete={handleBeforeDelete}
          onNodesDelete={handleNodesDelete}
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
        {choice && (
          <PortChooser
            name={choice.piece.name}
            ports={choice.ports}
            onPick={handleChoosePort}
            onCancel={cancelChoice}
          />
        )}
      </div>
    </div>
  );
}
