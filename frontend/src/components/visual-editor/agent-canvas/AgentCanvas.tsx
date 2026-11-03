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
import { AlertTriangle, Loader2, Plus, RefreshCw } from 'lucide-react';
import { toast } from 'sonner';
import { apiService, ApiError, type AIService, type Agent } from '../../../services/api';
import { MESSAGES, errorMessage, validationDetailMessage } from '../../../constants/messages';
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
import { CanvasPalette, PIECE_DRAG_MIME, parsePieceDragPayload, PIECE_SINGULAR, pieceKindPhrase } from './CanvasPalette';
import { piecesForPort, type PortPickerEntry } from './portPicker';
import { CanvasInspector } from './CanvasInspector';
import { CreatePieceModal } from './CreatePieceModal';
import CompactServiceEditor from '../../services/CompactServiceEditor';
import { useServicesManager } from '../../../hooks/useServicesManager';
import { useSettingsCache } from '../../../contexts/SettingsCacheContext';
import type { LifecycleStatus, SandboxServiceFormData, ServiceFormData } from '../../../types/services';
import { useAgentCanvasData, type CanvasPiece } from './useAgentCanvasData';

/** What the inspector panel receives when something is selected. */
export type CanvasSelection =
  | { type: 'agent' }
  | { type: 'piece'; piece: CanvasPiece }
  | { type: 'edge'; port: PortId; piece: CanvasPiece };

/** Toast text for a failed canvas save: a 422's first validation message, else the error message or the fallback. */
function saveErrorMessage(err: unknown, fallback: string): string {
  // A 422 carries the backend's validation message in `detail`, not in `message`.
  const validation = err instanceof ApiError && err.status === 422 ? validationDetailMessage(err.detail) : null;
  return validation ?? errorMessage(err, fallback);
}

export interface AgentCanvasProps {
  readonly appId: number;
  readonly agentId: number;
  /** False for roles below EDITOR: no connecting, no disconnecting (panning/dragging still works). */
  readonly canEdit: boolean;
  /**
   * True for ADMINISTRATOR and up: enables the palette `+` for the kinds that
   * create shared resources (AI services, embedding, sandbox, skills, MCP,
   * output parsers, middlewares). Silos and agents only need `canEdit`.
   */
  readonly canCreateShared?: boolean;
  /** Invoked with the selection on click; null when the pane is clicked. */
  readonly onSelect?: (selection: CanvasSelection | null) => void;
  /** Invoked after each successful save (connect, disconnect or inspector field), e.g. to refresh the page header. */
  readonly onSaved?: () => void;
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

/** Pointer position of a React Flow drag event (mouse or touch); null when a touch drag has no coordinates. */
function dragPointerPoint(event: MouseEvent | TouchEvent): { x: number; y: number } | null {
  if (!('touches' in event)) return { x: event.clientX, y: event.clientY };
  const touch = event.changedTouches[0] ?? event.touches[0];
  return touch ? { x: touch.clientX, y: touch.clientY } : null;
}

/**
 * Default positions: the agent sits at the origin and its connected pieces
 * line up in one column to its left, grouped by port in `portsFor` order.
 * Loose pieces (on the canvas but unconnected) sit in a column to the right.
 * Positions saved in localStorage (from an earlier drag) override the default.
 */
export function buildFlowNodes(
  agent: Agent,
  agentType: AgentType,
  connected: ReadonlyArray<{ port: PortId; kind: PieceKind; id: number }>,
  pieces: readonly CanvasPiece[],
  activePortIds: readonly PortId[],
  savedPositions: Readonly<Record<string, { x: number; y: number }>>,
  canEdit: boolean,
  looseNodeIds: readonly string[],
  draggingKind: PieceKind | null,
  draggingSupportsVideo: boolean,
  onOpenPortPicker: (port: PortDef, row: DOMRect) => void,
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
        isTool: agent.is_tool === true,
        canEdit,
        draggingKind,
        draggingSupportsVideo,
        onOpenPortPicker,
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
      // Deletable only to disconnect: handleBeforeDelete maps the card to its
      // agent edges and never returns it as a removed node.
      deletable: canEdit,
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

/**
 * Edges a Delete gesture must consider: the ones React Flow collected plus,
 * for every selected connected piece card, each edge that card holds to the
 * agent (React Flow's default path collects those too; the card's delete
 * must not depend on it). Deduped by edge id; loose cards hold no edges and
 * the agent body is never collected as a piece.
 */
export function deleteCandidateEdges(
  candidateNodes: CanvasFlowNode[],
  candidateEdges: CanvasEdge[],
  edges: CanvasEdge[],
  agentNodeId: string,
  looseNodeIds: readonly string[],
): CanvasEdge[] {
  const connectedPieces = new Set(
    candidateNodes
      .filter((node) => node.type === 'piece' && !looseNodeIds.includes(node.id))
      .map((node) => node.id),
  );
  if (connectedPieces.size === 0) return candidateEdges;
  const byId = new Map(candidateEdges.map((edge) => [edge.id, edge]));
  for (const edge of edges) {
    if (connectedPieces.has(edge.source) && edge.target === agentNodeId) byId.set(edge.id, edge);
  }
  return [...byId.values()];
}

function CanvasLoading() {
  return (
    <div
      className="flex h-full w-full min-h-[400px] flex-col items-center justify-center gap-3 rounded-xl border border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-800/50"
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
      className="flex h-full w-full min-h-[400px] flex-col items-center justify-center gap-3 rounded-xl border border-red-200 bg-red-50 px-6 text-center dark:border-red-800/50 dark:bg-red-900/20"
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

interface PortPickerProps {
  readonly port: PortDef;
  /** Existing pieces the port accepts, already filtered by `piecesForPort`. */
  readonly entries: readonly PortPickerEntry[];
  /** Accepted kinds the user may create here (the palette's role rules). */
  readonly createKinds: readonly PieceKind[];
  /** Container-relative position of the row's «+». */
  readonly x: number;
  readonly y: number;
  readonly onPick: (piece: CanvasPiece) => void;
  readonly onCreate: (kind: PieceKind) => void;
  readonly onClose: () => void;
}

/**
 * Small popover anchored to a port row's «+»: connect one of the pieces the
 * port accepts, or start creating one. Escape or a click outside closes it,
 * and focus moves to the first pickable entry.
 */
function PortPicker({ port, entries, createKinds, x, y, onPick, onCreate, onClose }: PortPickerProps) {
  const headingId = useId();
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Disabled (retired) entries cannot take focus; the first pickable one does.
    containerRef.current?.querySelector<HTMLButtonElement>('button:not(:disabled)')?.focus();
  }, []);

  useEffect(() => {
    const handlePointerDown = (event: MouseEvent) => {
      if (event.target instanceof Node && containerRef.current?.contains(event.target)) return;
      onClose();
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    document.addEventListener('mousedown', handlePointerDown);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handlePointerDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [onClose]);

  return (
    <div
      ref={containerRef}
      role="dialog"
      aria-labelledby={headingId}
      style={{ left: x, top: y }}
      className="absolute z-20 w-56 rounded-xl border border-gray-200 bg-white p-2 shadow-lg dark:border-gray-700 dark:bg-gray-800"
    >
      <p id={headingId} className="px-1 pb-1 text-sm font-medium text-gray-900 dark:text-gray-100">
        Connect to {port.label}:
      </p>
      <div className="flex max-h-60 flex-col gap-0.5 overflow-y-auto">
        {entries.map((entry) => (
          <button
            key={`${entry.piece.kind}:${entry.piece.id}`}
            type="button"
            disabled={entry.disabled}
            title={entry.disabled ? 'This service is retired' : undefined}
            onClick={() => onPick(entry.piece)}
            className="flex min-w-0 items-center justify-between gap-2 rounded-lg px-2 py-1 text-left text-sm text-gray-800 transition-colors hover:bg-indigo-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:cursor-not-allowed disabled:opacity-60 dark:text-gray-100 dark:hover:bg-indigo-900/30"
          >
            <span className="min-w-0 truncate" title={entry.piece.name}>
              {entry.piece.name}
            </span>
            {entry.disabled && (
              <span className="shrink-0 rounded bg-red-100 px-1.5 py-0.5 text-[10px] font-medium text-red-700 dark:bg-red-900/40 dark:text-red-200">
                Retired
              </span>
            )}
          </button>
        ))}
        {createKinds.map((kind) => (
          <button
            key={`create:${kind}`}
            type="button"
            onClick={() => onCreate(kind)}
            className="flex items-center gap-2 rounded-lg px-2 py-1 text-left text-sm font-medium text-indigo-700 transition-colors hover:bg-indigo-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 dark:text-indigo-300 dark:hover:bg-indigo-900/30"
          >
            <Plus className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            Create new {PIECE_SINGULAR[kind]}
          </button>
        ))}
        {entries.length === 0 && createKinds.length === 0 && (
          <p className="px-2 py-1 text-xs text-gray-500 dark:text-gray-400">
            Everything this port accepts is already connected.
          </p>
        )}
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
 * save immediately, one step per gesture. The inspector on the right edits
 * the selected agent, piece or connection through the same save queue.
 */
export function AgentCanvas({ appId, agentId, canEdit, canCreateShared = false, onSelect, onSaved }: AgentCanvasProps) {
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

  /** Piece being dragged toward the canvas (palette or a loose card), for the port drop highlights. */
  const [draggingPiece, setDraggingPiece] = useState<CanvasPiece | null>(null);
  const draggingKind = draggingPiece?.kind ?? null;
  const draggingSupportsVideo = draggingPiece?.supportsVideo === true;

  const canvasRef = useRef<HTMLDivElement>(null);

  /** Opens the port's piece picker, anchored to the row's «+» and clamped inside the canvas. */
  const openPortPicker = useCallback((port: PortDef, row: DOMRect) => {
    const bounds = canvasRef.current?.getBoundingClientRect();
    if (!bounds) return;
    setChoice(null);
    const width = 232;
    const height = 272;
    setPicker({
      port,
      x: Math.min(Math.max(row.left - bounds.left + 8, 8), bounds.width - width),
      y: Math.min(Math.max(row.top - bounds.top, 8), bounds.height - height),
    });
  }, []);

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
            draggingKind,
            draggingSupportsVideo,
            openPortPicker,
          )
        : [],
    [
      agent,
      agentType,
      connected,
      pieces,
      activePortIds,
      layout.positions,
      canEdit,
      looseNodeIds,
      draggingKind,
      draggingSupportsVideo,
      openPortPicker,
    ],
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
  /** Node id of the piece set by a connect: once that node is on the canvas the view re-fits so it isn't left off screen. */
  const fitOnNextNodes = useRef<string | null>(null);
  useEffect(() => {
    const nodeId = fitOnNextNodes.current;
    // Not a boolean: clearing drag state on drop also changes `nodes` before
    // the refetch adds the piece node, so the fit must wait for that node itself.
    if (!nodeId || !nodes.some((node) => node.id === nodeId)) return;
    fitOnNextNodes.current = null;
    // Wait a beat so React Flow has measured the new node before fitting; an earlier fit ignores it.
    setTimeout(() => void flowRef.current?.fitView({ padding: 0.2, duration: 300 }), 150);
  }, [nodes]);

  const [choice, setChoice] = useState<PendingChoice | null>(null);
  const cancelChoice = useCallback(() => setChoice(null), []);
  /**
   * Creation flow in progress: the kind plus, when it was opened from a port
   * row's «+», the port the new piece must connect to.
   */
  const [creating, setCreating] = useState<{ readonly kind: PieceKind; readonly port?: PortDef } | null>(null);
  const closeCreate = useCallback(() => setCreating(null), []);
  /** AI service editing reuses the same manager (api, cache, toasts, save) the settings page uses, so an edit behaves identically in both. */
  const settingsCache = useSettingsCache();
  const servicesApi = {
    getAll: (id: number) => apiService.getAIServices(id),
    getOne: (id: number, sid: number) => apiService.getAIService(id, sid),
    create: (id: number, data: any) => apiService.createAIService(id, data),
    update: (id: number, sid: number, data: any) => apiService.updateAIService(id, sid, data),
    delete: (id: number, sid: number) => apiService.deleteAIService(id, sid),
  };
  const servicesCache = {
    get: (id: string) => settingsCache.getAIServices(id),
    set: (id: string, data: AIService[]) => settingsCache.setAIServices(id, data),
    invalidate: (id: string) => settingsCache.invalidateAIServices(id),
  };
  const {
    services,
    isModalOpen,
    editingService,
    handleEdit,
    handleSave,
    handleClose,
    setIsModalOpen,
    setEditingService,
  } = useServicesManager<AIService>(String(appId), servicesApi, servicesCache, { entity: 'AI service' });
  /** Opens the AI service editor for a canvas piece; the manager loads the full service detail first. */
  const handleEditService = useCallback(
    (piece: CanvasPiece) => {
      if (piece.kind !== 'ai_service') return;
      void handleEdit(piece.id);
    },
    [handleEdit],
  );
  /** A service edit saved: the manager closed the editor, the canvas reloads its pieces. */
  const handleSaveService = useCallback(
    async (data: ServiceFormData | SandboxServiceFormData): Promise<void> => {
      await handleSave(data);
      void refetch();
    },
    [handleSave, refetch],
  );
  /** Port row whose «+» popover is open, anchored at container-relative coordinates. */
  const [picker, setPicker] = useState<{ readonly port: PortDef; readonly x: number; readonly y: number } | null>(
    null,
  );
  // The chooser, the picker, the creation and the edit modals belong to the agent they were opened for.
  useEffect(() => {
    setChoice(null);
    setCreating(null);
    setIsModalOpen(false);
    setEditingService(null);
    setPicker(null);
    setDraggingPiece(null);
  }, [agentId]);

  // The selection belongs to the agent it was made for; the inspector only
  // shows it while its piece is still on the canvas and its connection still exists.
  const [selectionState, setSelectionState] = useState<{ agentId: number; selection: CanvasSelection } | null>(null);
  const selectCanvas = useCallback(
    (selection: CanvasSelection | null) => {
      setSelectionState(selection ? { agentId, selection } : null);
      onSelect?.(selection);
    },
    [agentId, onSelect],
  );
  const resolvedSelection = useMemo<CanvasSelection | null>(() => {
    if (!selectionState || selectionState.agentId !== agentId || !agent) return null;
    const selected = selectionState.selection;
    if (selected.type === 'agent') return selected;
    const piece = pieces.find((entry) => entry.kind === selected.piece.kind && entry.id === selected.piece.id);
    if (!piece) return null;
    if (selected.type === 'piece') return { type: 'piece', piece };
    const stillConnected = connected.some(
      (entry) => entry.port === selected.port && entry.kind === piece.kind && entry.id === piece.id,
    );
    return stillConnected ? { type: 'edge', port: selected.port, piece } : null;
  }, [selectionState, agentId, agent, pieces, connected]);

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

  /**
   * Saves `steps` one at a time (fetch → merge patch → update), stopping at the first failure; then refetches.
   * Resolves true when every step saved. `failureLabel` names what failed in the error toast.
   */
  const runSteps = useCallback(
    async (steps: readonly CanvasSaveStep[], failureLabel = 'connection'): Promise<boolean> => {
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
          toast.error(saveErrorMessage(err, MESSAGES.UPDATE_FAILED(failureLabel)));
          saved = false;
          break;
        }
      }
      await refetch();
      if (saved) onSaved?.();
      return saved;
    },
    [appId, queue, refetch, onSaved],
  );

  /** Inspector field saves: one step through the same queue and executor as the connects. */
  const saveAgentFields = useCallback(
    (patch: Record<string, unknown>): Promise<boolean> => runSteps([{ step: { agentId, patch } }], 'agent'),
    [agentId, runSteps],
  );

  /**
   * The one connect path: every gesture that connects a piece (a handle drag
   * or a drop on the agent) ends here. Rule check, then the planned saves are
   * queued and refetched; a loose piece leaves `loose` once its save succeeds.
   * `toastSuffix` extends the success toast (a drop hint for other ports).
   */
  const connectPiece = useCallback(
    (port: PortDef, piece: CanvasPiece, toastSuffix?: string): void => {
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
        supportsVideo: piece.supportsVideo,
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
        toastSuffix,
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
      fitOnNextNodes.current = nodeId;
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
      if (!port) return;
      // An AI service landing on the default Model port still has to be
      // dropped on Transcription/Video explicitly; nudge when either is empty.
      const toastSuffix =
        target === 'model' && piece.kind === 'ai_service' && !connected.some((entry) => entry.port === 'transcription' || entry.port === 'video')
          ? ' — to use it for transcription or video, drop it on that port or use its +'
          : undefined;
      connectPiece(port, piece, toastSuffix);
    },
    [agentType, connected, connectPiece],
  );

  const handleChoosePort = useCallback(
    (port: PortDef) => {
      if (!choice) return;
      setChoice(null);
      connectPiece(port, choice.piece);
    },
    [choice, connectPiece],
  );

  /**
   * A piece saved from a «+»: refetch so it is in the pieces list, then place
   * it the way a drop would. A piece no port takes stays as a loose card; any
   * other piece goes through the same drop path (direct connect or chooser) —
   * unless the «+» sat on a port row, which connects the piece to that port.
   */
  const handlePieceCreated = useCallback(
    async (
      kind: PieceKind,
      id: number,
      name: string,
      lifecycleStatus?: LifecycleStatus,
      supportsVideo?: boolean,
      port?: PortDef,
    ): Promise<void> => {
      setCreating(null);
      await refetch();
      // A new agent is never a tool yet, so connecting it through Tools adds is_tool first.
      const piece: CanvasPiece = {
        kind,
        id,
        name,
        ...(kind === 'agent' ? { isTool: false } : {}),
        ...(lifecycleStatus ? { lifecycleStatus } : {}),
        ...(kind === 'ai_service' && supportsVideo !== undefined ? { supportsVideo } : {}),
      };
      if (port) {
        connectPiece(port, piece);
        return;
      }
      if (resolveDropPort(agentType, kind) === null) {
        const nodeId = pieceNodeId(piece);
        updateLayout((current) => ({
          positions: current.positions,
          loose: current.loose.includes(nodeId) ? current.loose : [...current.loose, nodeId],
        }));
        return;
      }
      dropOnAgent(piece);
    },
    [agentType, refetch, dropOnAgent, updateLayout, connectPiece],
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

  /**
   * Port row under the pointer, scoped to the agent node. `hidden` is the
   * piece card being dragged by the pointer: it sits under the cursor at drag
   * stop and would shadow the row underneath, so it is lifted out of the hit
   * test for the duration of the call.
   */
  const portRowAtPoint = useCallback((x: number, y: number, hidden?: HTMLElement | null): PortId | null => {
    if (hidden) hidden.style.pointerEvents = 'none';
    try {
      const row = document.elementFromPoint(x, y)?.closest<HTMLElement>('[data-agent-node] [data-port-id]');
      return row ? ((row.dataset.portId as PortId) ?? null) : null;
    } finally {
      if (hidden) hidden.style.pointerEvents = '';
    }
  }, []);

  /** A loose card drag starts: the ports that would take it light up. */
  const handleNodeDragStart: OnNodeDrag<CanvasFlowNode> = useCallback(
    (_event, node) => {
      if (looseNodeIds.includes(node.id) && node.data.piece) setDraggingPiece(node.data.piece);
    },
    [looseNodeIds],
  );

  const handleNodeDragStop: OnNodeDrag<CanvasFlowNode> = useCallback(
    (event, draggedNode) => {
      setDraggingPiece(null);
      const positions: Record<string, { x: number; y: number }> = {};
      for (const node of nodesRef.current) {
        positions[node.id] = node.id === draggedNode.id ? draggedNode.position : node.position;
      }
      // A loose piece dragged over the agent connects. Its saved position is
      // dropped so it lands in the default column once connected.
      const piece = draggedNode.data.piece;
      const isLoosePiece = canEdit && piece !== undefined && looseNodeIds.includes(draggedNode.id);
      const center = {
        x: draggedNode.position.x + (draggedNode.measured?.width ?? 0) / 2,
        y: draggedNode.position.y + (draggedNode.measured?.height ?? 0) / 2,
      };
      const droppedOnAgent = isLoosePiece && isOverAgent(center);
      // A drop aimed at a specific port row connects there directly.
      let rowPort: PortDef | null = null;
      if (isLoosePiece) {
        const point = dragPointerPoint(event);
        if (point) {
          const draggedEl = document.querySelector<HTMLElement>(
            `.react-flow__node[data-id="${draggedNode.id}"]`,
          );
          const portId = portRowAtPoint(point.x, point.y, draggedEl);
          if (portId) rowPort = portsFor(agentType).find((entry) => entry.id === portId) ?? null;
        }
      }
      updateLayout((current) => {
        const merged = { ...current.positions, ...positions };
        if (droppedOnAgent || rowPort) delete merged[draggedNode.id];
        return { positions: merged, loose: current.loose };
      });
      if (droppedOnAgent || rowPort) reseatNode(draggedNode.id);
      if (rowPort && piece) {
        connectPiece(rowPort, piece);
        return;
      }
      if (droppedOnAgent && piece) dropOnAgent(piece);
    },
    [canEdit, looseNodeIds, isOverAgent, reseatNode, dropOnAgent, updateLayout, agentType, portRowAtPoint, connectPiece],
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
      // A connected piece card must delete exactly like its connection line:
      // every edge it holds to the agent joins the candidates (deduped), so
      // the same checks and save path apply.
      const candidates = deleteCandidateEdges(candidateNodes, candidateEdges, edges, agentNodeId, looseNodeIds);
      const allowed: CanvasEdge[] = [];
      const blockedReasons = new Set<string>();
      for (const edge of candidates) {
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
    [agent, agentType, canEdit, looseNodeIds, edges, agentNodeId],
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
            const saved = await queue.run(agent.agent_id, async () => {
              const fresh = await apiService.getAgent(appId, agent.agent_id);
              const patch = buildPortPatch(fresh, port.id, data.piece.kind, data.piece.id, 'remove');
              // Nothing to clear: the piece was already replaced or removed on the fresh agent.
              if (Object.keys(patch).length === 0) return false;
              await apiService.updateAgent(appId, agent.agent_id, { ...fresh, ...patch });
              return true;
            });
            if (saved) succeeded = true;
          } catch (err) {
            toast.error(saveErrorMessage(err, MESSAGES.UPDATE_FAILED('connection')));
          }
        }
        if (succeeded) {
          toast.success(MESSAGES.UPDATED('connection'));
          onSaved?.();
        }
        await refetch();
      })();
    },
    [agent, agentType, appId, queue, refetch, onSaved],
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
      // A new drop replaces any chooser or picker still open from an earlier gesture.
      setChoice(null);
      setPicker(null);
      setDraggingPiece(null);
      const piece = pieces.find((entry) => entry.kind === payload.kind && entry.id === payload.id);
      const instance = flowRef.current;
      if (!piece || !instance) return;
      // A drop on a specific port row connects there; the agent body falls
      // back to the default port or the chooser, anything else stays loose.
      const portId = portRowAtPoint(event.clientX, event.clientY);
      if (portId) {
        const port = portsFor(agentType).find((entry) => entry.id === portId);
        if (port) {
          connectPiece(port, piece);
          return;
        }
      }
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
    [canEdit, agent, pieces, connectedNodeIds, isOverAgent, reseatNode, dropOnAgent, updateLayout, agentType, portRowAtPoint, connectPiece],
  );

  const handleCreate = useCallback((kind: PieceKind) => setCreating({ kind }), []);

  const closePicker = useCallback(() => setPicker(null), []);

  const handlePickerPick = useCallback(
    (piece: CanvasPiece) => {
      if (!picker) return;
      setPicker(null);
      connectPiece(picker.port, piece);
    },
    [picker, connectPiece],
  );

  const handlePickerCreate = useCallback(
    (kind: PieceKind) => {
      if (!picker) return;
      setCreating({ kind, port: picker.port });
      setPicker(null);
    },
    [picker],
  );

  const pickerEntries = useMemo(
    () => (picker ? piecesForPort(picker.port, pieces, connected) : []),
    [picker, pieces, connected],
  );
  /** One "Create new …" per accepted kind, with the palette's role rules. */
  const pickerCreateKinds = useMemo(
    () =>
      picker
        ? picker.port.accepts.filter((kind) => kind === 'silo' || kind === 'agent' || canCreateShared)
        : [],
    [picker, canCreateShared],
  );

  const handleNodeClick: NodeMouseHandler<CanvasFlowNode> = useCallback(
    (_event, node) => {
      if (node.type === 'agent') {
        selectCanvas({ type: 'agent' });
        return;
      }
      if (node.data.piece) selectCanvas({ type: 'piece', piece: node.data.piece });
    },
    [selectCanvas],
  );

  const handleEdgeClick: EdgeMouseHandler<CanvasEdge> = useCallback(
    (_event, edge) => {
      if (edge.data) selectCanvas({ type: 'edge', port: edge.data.port, piece: edge.data.piece });
    },
    [selectCanvas],
  );

  const handlePaneClick = useCallback(() => {
    selectCanvas(null);
  }, [selectCanvas]);

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
    <div className="flex w-full flex-col gap-3 md:h-full md:flex-row">
      <CanvasPalette
        pieces={pieces}
        agentType={agentType}
        canEdit={canEdit}
        canCreateShared={canCreateShared}
        onCreate={handleCreate}
        onDraggingChange={setDraggingPiece}
      />
      <div
        ref={canvasRef}
        className="relative h-[70vh] w-full min-h-[400px] min-w-0 flex-1 md:h-full overflow-hidden rounded-xl border border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-900"
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
          onNodeDragStart={handleNodeDragStart}
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
        {picker && (
          <PortPicker
            port={picker.port}
            x={picker.x}
            y={picker.y}
            entries={pickerEntries}
            createKinds={pickerCreateKinds}
            onPick={handlePickerPick}
            onCreate={handlePickerCreate}
            onClose={closePicker}
          />
        )}
      </div>
      <CanvasInspector
        appId={appId}
        agent={agent}
        agentType={agentType}
        selection={resolvedSelection}
        canEdit={canEdit}
        canCreateShared={canCreateShared}
        onEditService={handleEditService}
        onSaveFields={saveAgentFields}
      />
      {creating && (
        <CreatePieceModal
          appId={appId}
          kind={creating.kind}
          onClose={closeCreate}
          onCreated={(kind, id, name, lifecycleStatus, supportsVideo) =>
            void handlePieceCreated(kind, id, name, lifecycleStatus, supportsVideo, creating.port)
          }
        />
      )}
      {isModalOpen && editingService && (
        <CompactServiceEditor
          isOpen
          kind="ai"
          scope="app"
          appId={appId}
          service={editingService}
          existingNames={services.map((s) => s.name)}
          onClose={handleClose}
          onSave={handleSaveService}
        />
      )}
    </div>
  );
}
