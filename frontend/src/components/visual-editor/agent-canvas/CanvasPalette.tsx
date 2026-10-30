import { useId, useState, type DragEvent } from 'react';
import { Plus, Search } from 'lucide-react';
import { PIECE_VISUALS } from '../nodeKindConfig';
import { ServiceOptionStatus, serviceStatusId } from '../../ui/ServiceOptionStatus';
import type { ServiceOptionLike } from '../../../utils/serviceLifecycle';
import { portsFor, type AgentType, type PieceKind } from './agentPorts';
import { KIND_PHRASES } from './connectionRules';
import type { CanvasPiece } from './useAgentCanvasData';

/** Drag payload type for palette items dropped on the canvas (JSON `{kind, id}`). */
export const PIECE_DRAG_MIME = 'application/x-mattin-piece';

const MEMORY_ROW_LABEL = 'Memory — coming soon';

interface PaletteCategory {
  readonly kind: PieceKind;
  readonly title: string;
  /** Lower-case noun for the "Create ..." button label. */
  readonly singular: string;
}

const PALETTE_CATEGORIES: readonly PaletteCategory[] = [
  { kind: 'ai_service', title: 'AI services', singular: 'AI service' },
  { kind: 'silo', title: 'Knowledge', singular: 'silo' },
  { kind: 'skill', title: 'Skills', singular: 'skill' },
  { kind: 'mcp', title: 'MCP servers', singular: 'MCP server' },
  { kind: 'agent', title: 'Agents', singular: 'agent' },
  { kind: 'parser', title: 'Output parsers', singular: 'output parser' },
  { kind: 'middleware', title: 'Middlewares', singular: 'middleware' },
  { kind: 'sandbox', title: 'Sandbox', singular: 'sandbox service' },
  { kind: 'embedding', title: 'Embedding', singular: 'embedding service' },
];

export function pieceKindPhrase(kind: PieceKind): string {
  return KIND_PHRASES[kind];
}

function isPieceKind(value: unknown): value is PieceKind {
  return typeof value === 'string' && Object.keys(KIND_PHRASES).includes(value);
}

/** Reads a palette drag payload; null for anything that is not a `{kind, id}` piece. */
export function parsePieceDragPayload(raw: string): { readonly kind: PieceKind; readonly id: number } | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (parsed === null || typeof parsed !== 'object') return null;
  const { kind, id } = parsed as { kind?: unknown; id?: unknown };
  if (!isPieceKind(kind) || typeof id !== 'number' || !Number.isInteger(id)) return null;
  return { kind, id };
}

export interface CanvasPaletteProps {
  readonly pieces: readonly CanvasPiece[];
  readonly agentType: AgentType;
  /** False for roles below EDITOR: items are not draggable and the `+` buttons are hidden. */
  readonly canEdit: boolean;
  /**
   * True for roles that may create shared resources (ADMINISTRATOR and up): the
   * `+` of AI services, embedding, sandbox, skills, MCP, output parsers and
   * middlewares. Silos and agents only need `canEdit`.
   */
  readonly canCreateShared: boolean;
  /** Opens the creation flow for a kind (wired by the creation modals). */
  readonly onCreate: (kind: PieceKind) => void;
}

interface PaletteItemProps {
  readonly piece: CanvasPiece;
  readonly canEdit: boolean;
}

function PaletteItem({ piece, canEdit }: PaletteItemProps) {
  const statusBaseId = useId();
  const visual = PIECE_VISUALS[piece.kind];
  const Icon = visual.icon;
  // Retired pieces cannot be connected, so they cannot be dragged either.
  const draggable = canEdit && piece.lifecycleStatus !== 'retired';
  const statusOptions: ServiceOptionLike[] = piece.lifecycleStatus
    ? [{ service_id: piece.id, name: piece.name, lifecycle_status: piece.lifecycleStatus }]
    : [];
  const describedBy = serviceStatusId(statusOptions, piece.id, statusBaseId);

  const handleDragStart = (event: DragEvent<HTMLDivElement>) => {
    event.dataTransfer.setData(PIECE_DRAG_MIME, JSON.stringify({ kind: piece.kind, id: piece.id }));
    event.dataTransfer.effectAllowed = 'copy';
  };

  return (
    <li>
      <div
        draggable={draggable}
        onDragStart={draggable ? handleDragStart : undefined}
        aria-describedby={describedBy}
        className={`flex items-start gap-2 rounded-lg border px-2 py-1.5 transition-shadow ${visual.borderClass} ${visual.bgClass} ${
          draggable ? 'cursor-grab hover:shadow-sm active:cursor-grabbing' : 'cursor-not-allowed opacity-60'
        }`}
      >
        <div className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-md ${visual.chipClass}`}>
          <Icon className={`h-3.5 w-3.5 ${visual.iconClass}`} aria-hidden="true" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="flex min-w-0 items-center gap-1.5 text-sm font-medium text-gray-900 dark:text-gray-100">
            <span className="truncate" title={piece.name}>
              {piece.name}
            </span>
            {piece.isSystem && (
              <span className="shrink-0 rounded bg-blue-100 px-1.5 py-0.5 text-[10px] font-medium text-blue-700 dark:bg-blue-900/40 dark:text-blue-200">
                System
              </span>
            )}
          </p>
          <ServiceOptionStatus options={statusOptions} selectedId={piece.id} selectId={statusBaseId} />
        </div>
      </div>
    </li>
  );
}

/**
 * Left-hand palette of the per-agent canvas: every piece the app offers,
 * grouped by kind. Items are dragged onto the canvas; the category `+` buttons
 * create new ones. OCR agents only see the categories their ports accept.
 */
export function CanvasPalette({ pieces, agentType, canEdit, canCreateShared, onCreate }: CanvasPaletteProps) {
  const baseId = useId();
  const [filter, setFilter] = useState('');
  const needle = filter.trim().toLowerCase();
  const ports = portsFor(agentType);
  /** Silos and agents need EDITOR; the other kinds create shared resources and need ADMINISTRATOR. */
  const canCreateFor = (kind: PieceKind): boolean =>
    kind === 'silo' || kind === 'agent' ? canEdit : canCreateShared;

  const groups = PALETTE_CATEGORIES.filter((category) => ports.some((port) => port.accepts.includes(category.kind)))
    .map((category) => ({
      category,
      items: pieces.filter(
        (piece) => piece.kind === category.kind && piece.name.toLowerCase().includes(needle),
      ),
    }))
    .filter(({ items }) => needle === '' || items.length > 0);
  const showMemory = agentType === 'agent' && (needle === '' || MEMORY_ROW_LABEL.toLowerCase().includes(needle));
  const noMatches = needle !== '' && groups.length === 0 && !showMemory;

  return (
    <aside
      aria-label="Pieces palette"
      className="flex max-h-72 w-full shrink-0 flex-col overflow-hidden rounded-xl border border-gray-200 bg-white md:max-h-none md:w-64 dark:border-gray-700 dark:bg-gray-800"
    >
      <div className="border-b border-gray-200 p-2 dark:border-gray-700">
        <label htmlFor={`${baseId}-filter`} className="sr-only">
          Filter pieces
        </label>
        <div className="relative">
          <Search
            className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-gray-400 dark:text-gray-500"
            aria-hidden="true"
          />
          <input
            id={`${baseId}-filter`}
            type="search"
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
            placeholder="Filter pieces"
            className="w-full rounded-lg border border-gray-300 bg-white py-1.5 pl-7 pr-2 text-sm text-gray-900 placeholder:text-gray-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 dark:border-gray-600 dark:bg-gray-900 dark:text-gray-100 dark:placeholder:text-gray-500"
          />
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        {groups.map(({ category, items }) => {
          const headingId = `${baseId}-${category.kind}`;
          return (
            <section key={category.kind} aria-labelledby={headingId} className="mb-3">
              <div className="flex items-center justify-between gap-2 px-1 pb-1">
                <h3
                  id={headingId}
                  className="text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400"
                >
                  {category.title}
                </h3>
                {canCreateFor(category.kind) && (
                  <button
                    type="button"
                    onClick={() => onCreate(category.kind)}
                    aria-label={`Create ${category.singular}`}
                    title={`Create ${category.singular}`}
                    className="inline-flex h-6 w-6 items-center justify-center rounded-md text-indigo-600 transition-colors hover:bg-indigo-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 dark:text-indigo-300 dark:hover:bg-indigo-900/40"
                  >
                    <Plus className="h-4 w-4" aria-hidden="true" />
                  </button>
                )}
              </div>
              <ul className="space-y-1">
                {items.map((piece) => (
                  <PaletteItem key={`${piece.kind}:${piece.id}`} piece={piece} canEdit={canEdit} />
                ))}
                {items.length === 0 && (
                  <li className="px-1 py-1 text-xs text-gray-400 dark:text-gray-500">None yet</li>
                )}
              </ul>
            </section>
          );
        })}

        {showMemory && (
          <button
            type="button"
            disabled
            className="mt-1 w-full rounded-lg border border-dashed border-gray-300 px-2 py-1.5 text-left text-sm text-gray-500 disabled:cursor-not-allowed disabled:opacity-60 dark:border-gray-600 dark:text-gray-400"
          >
            {MEMORY_ROW_LABEL}
          </button>
        )}

        {noMatches && (
          <p className="px-1 py-2 text-xs text-gray-500 dark:text-gray-400">No pieces match this filter.</p>
        )}
      </div>
    </aside>
  );
}
