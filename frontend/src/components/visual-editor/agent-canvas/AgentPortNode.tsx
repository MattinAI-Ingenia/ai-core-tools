import { useState } from 'react';
import { ChevronDown, ChevronUp, Plus } from 'lucide-react';
import { Handle, Position, type Node, type NodeProps } from '@xyflow/react';
import { NODE_KIND_VISUALS } from '../nodeKindConfig';
import type { PieceKind, PortDef, PortId } from './agentPorts';
import type { CanvasPiece } from './useAgentCanvasData';

/**
 * Data payload carried by every node on the per-agent canvas. Fields are
 * optional and only the ones for the node's `type` are set, so the agent
 * body and the piece cards share one data shape.
 */
export interface CanvasNodeData extends Record<string, unknown> {
  readonly label: string;
  /** Caption under the label ("Agent" / "OCR agent"; piece cards use the kind caption). */
  readonly typeCaption?: string;
  /** Agent body only: primary port rows, in `portsFor` order. */
  readonly ports?: readonly PortDef[];
  /** Agent body only: secondary ports, folded behind the "More" toggle. */
  readonly secondaryPorts?: readonly PortDef[];
  /** Ports that currently have a connection (pending ones included); their rows always render. */
  readonly connectedPortIds?: readonly PortId[];
  /** Agent body only: whether the edited agent is usable as a tool (shows the Tool badge). */
  readonly isTool?: boolean;
  /** Agent body only: whether the current user may edit (drives the per-port «+» buttons). */
  readonly canEdit?: boolean;
  /** Agent body only: piece kind currently being dragged toward the canvas, if any. */
  readonly draggingKind?: PieceKind | null;
  /** Agent body only, AI services only: whether the dragged service supports video. */
  readonly draggingSupportsVideo?: boolean;
  /** Agent body only: opens the port's piece picker, anchored to the row's «+». */
  readonly onOpenPortPicker?: (port: PortDef, row: DOMRect) => void;
  /** Piece cards only: the connected piece. */
  readonly piece?: CanvasPiece;
  /** Piece cards only: labels of the ports the piece feeds, in port order (one card per piece even when it connects to several ports). */
  readonly portLabels?: readonly string[];
  /** Piece cards only: on the canvas but not connected to any port yet. */
  readonly loose?: boolean;
}

export type CanvasFlowNode = Node<CanvasNodeData>;

function PortRow({
  port,
  isConnected,
  isConnectable,
  canEdit,
  dragging,
  acceptsDrag,
  onOpenPicker,
}: {
  readonly port: PortDef;
  readonly isConnected: boolean;
  readonly isConnectable: boolean;
  readonly canEdit: boolean;
  readonly dragging: boolean;
  readonly acceptsDrag: boolean;
  readonly onOpenPicker: (port: PortDef, row: DOMRect) => void;
}) {
  // Empty secondary ports hint at how to fill them.
  const showHint = port.secondary === true && !isConnected;
  return (
    <div
      data-port-id={port.id}
      className={`relative flex items-center rounded-lg py-0.5 transition ${
        acceptsDrag
          ? 'ring-2 ring-indigo-400 bg-indigo-50 dark:bg-indigo-900/40'
          : dragging
            ? 'opacity-40'
            : ''
      }`}
    >
      <Handle
        type="target"
        position={Position.Left}
        id={port.id}
        isConnectable={isConnectable}
        className="!h-2.5 !w-2.5 !rounded-full !border-2 !border-white dark:!border-gray-900 !bg-indigo-400 dark:!bg-indigo-500"
      />
      <div className="ml-4 min-w-0 flex-1">
        <span className="block truncate text-xs text-gray-600 dark:text-gray-300" title={port.label}>
          {port.label}
        </span>
        {showHint && (
          <span className="block truncate text-[10px] italic text-gray-400 dark:text-gray-500">
            Drop here or click +
          </span>
        )}
      </div>
      {canEdit && (
        <button
          type="button"
          onClick={(event) => onOpenPicker(port, event.currentTarget.getBoundingClientRect())}
          aria-label={`Add to ${port.label}`}
          title={`Add to ${port.label}`}
          className="nodrag nopan mr-1 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-md text-indigo-600 transition-colors hover:bg-indigo-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 dark:text-indigo-300 dark:hover:bg-indigo-900/40"
        >
          <Plus className="h-3 w-3" aria-hidden="true" />
        </button>
      )}
    </div>
  );
}

/**
 * The edited agent's body: one target handle per port of `portsFor(type)`,
 * with the secondary ports folded behind a "More" toggle. Secondary ports
 * that already have a connection stay visible while folded, so their edges
 * always have a handle to land on. While a piece is being dragged, the
 * ports that would take it light up (and hidden ones unfold), the rest dim.
 */
export function AgentPortNode({ data, isConnectable }: NodeProps<CanvasFlowNode>) {
  const visual = NODE_KIND_VISUALS.agent;
  const Icon = visual.icon;
  const ports = data.ports ?? [];
  const secondary = data.secondaryPorts ?? [];
  const connected = new Set(data.connectedPortIds ?? []);
  const isTool = data.isTool === true;
  const canEdit = data.canEdit === true;
  const draggingKind = data.draggingKind ?? null;
  const draggingSupportsVideo = data.draggingSupportsVideo === true;
  const acceptsDrag = (port: PortDef): boolean =>
    draggingKind !== null &&
    port.accepts.includes(draggingKind) &&
    // The Video port only takes AI services that can analyse video.
    !(port.id === 'video' && draggingKind === 'ai_service' && !draggingSupportsVideo);
  const [expanded, setExpanded] = useState(false);
  const visibleSecondary = secondary.filter(
    (port) => expanded || connected.has(port.id) || acceptsDrag(port),
  );
  // No toggle when every secondary port is already visible (all connected,
  // or every hidden one is a drop target for the piece being dragged).
  const hasHiddenPorts = secondary.some((port) => !connected.has(port.id) && !acceptsDrag(port));
  const onOpenPicker = data.onOpenPortPicker ?? (() => undefined);

  return (
    <div
      data-agent-node=""
      className={`relative min-w-[230px] rounded-xl border-2 px-3 py-2.5 shadow-sm transition-shadow duration-150 hover:shadow-md ${visual.borderClass} ${visual.bgClass}`}
    >
      <div className="flex items-center gap-3">
        <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ${visual.chipClass}`}>
          <Icon className={`h-5 w-5 ${visual.iconClass}`} aria-hidden="true" />
        </div>
        <div className="min-w-0 flex-1">
          {/* Keep the name clear of the absolute Tool badge on the top-right corner. */}
          <p
            className={`truncate text-sm font-semibold text-gray-900 dark:text-gray-100 ${isTool ? 'pr-12' : ''}`}
            title={data.label}
          >
            {data.label}
          </p>
          <p className={`truncate text-xs ${visual.captionClass}`}>{data.typeCaption ?? visual.typeCaption}</p>
        </div>
      </div>
      {isTool && (
        <span className="absolute right-2 top-2 rounded bg-emerald-100 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-200">
          Tool
        </span>
      )}

      {/* Full-bleed rows so the port handles sit on the card's left border. */}
      <div className="-mx-3 mt-2 space-y-0.5 border-t border-gray-100 px-3 pt-2 dark:border-gray-800">
        {ports.map((port) => (
          <PortRow
            key={port.id}
            port={port}
            isConnected={connected.has(port.id)}
            isConnectable={isConnectable}
            canEdit={canEdit}
            dragging={draggingKind !== null}
            acceptsDrag={acceptsDrag(port)}
            onOpenPicker={onOpenPicker}
          />
        ))}
        {visibleSecondary.map((port) => (
          <PortRow
            key={port.id}
            port={port}
            isConnected={connected.has(port.id)}
            isConnectable={isConnectable}
            canEdit={canEdit}
            dragging={draggingKind !== null}
            acceptsDrag={acceptsDrag(port)}
            onOpenPicker={onOpenPicker}
          />
        ))}
      </div>

      {hasHiddenPorts && (
        <button
          type="button"
          onClick={() => setExpanded((current) => !current)}
          aria-expanded={expanded}
          className="nodrag nopan mt-1.5 inline-flex items-center gap-1 rounded-md px-1 py-0.5 text-xs text-indigo-600 hover:bg-indigo-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 dark:text-indigo-300 dark:hover:bg-indigo-900/40"
        >
          {expanded ? (
            <ChevronUp className="h-3.5 w-3.5" aria-hidden="true" />
          ) : (
            <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
          )}
          {expanded ? 'Less' : 'More'}
        </button>
      )}
    </div>
  );
}
