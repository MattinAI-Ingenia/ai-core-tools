/**
 * Pure persistence of the per-agent canvas layout (node positions and ids
 * pulled out of ports) in the browser's localStorage. Storage is injected so
 * the module stays testable and free of any DOM access.
 */

/** Minimal storage surface the canvas needs (localStorage-compatible). */
export interface StorageLike {
  getItem(k: string): string | null;
  setItem(k: string, v: string): void;
}

/** Saved layout; node ids are `${kind}:${id}` strings. */
export interface CanvasLayout {
  readonly positions: Record<string, { x: number; y: number }>;
  readonly loose: string[];
}

/** localStorage key scoped to one agent of one app. */
export function layoutKey(appId: number, agentId: number): string {
  return `mattin.agentCanvas.v1.${appId}.${agentId}`;
}

function emptyLayout(): CanvasLayout {
  return { positions: {}, loose: [] };
}

/** Stored layout without entries for nodes the agent no longer has; malformed values are dropped. */
export function loadLayout(
  storage: StorageLike | null,
  appId: number,
  agentId: number,
  knownNodeIds: ReadonlySet<string>,
): CanvasLayout {
  if (!storage) return emptyLayout();
  let raw: string | null;
  try {
    raw = storage.getItem(layoutKey(appId, agentId));
  } catch {
    return emptyLayout();
  }
  if (raw === null) return emptyLayout();
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return emptyLayout();
  }
  if (parsed === null || typeof parsed !== 'object') return emptyLayout();
  const stored = parsed as { positions?: unknown; loose?: unknown };
  const positions: Record<string, { x: number; y: number }> = {};
  if (stored.positions && typeof stored.positions === 'object' && !Array.isArray(stored.positions)) {
    for (const [id, value] of Object.entries(stored.positions)) {
      if (!knownNodeIds.has(id)) continue;
      const point = value as { x: unknown; y: unknown } | null;
      if (
        point &&
        typeof point.x === 'number' &&
        typeof point.y === 'number' &&
        Number.isFinite(point.x) &&
        Number.isFinite(point.y)
      ) {
        positions[id] = { x: point.x, y: point.y };
      }
    }
  }
  const loose = Array.isArray(stored.loose)
    ? stored.loose.filter((id): id is string => typeof id === 'string' && knownNodeIds.has(id))
    : [];
  return { positions, loose };
}

/** Persist the layout; storage failures are swallowed (persistence is best-effort). */
export function saveLayout(
  storage: StorageLike | null,
  appId: number,
  agentId: number,
  layout: CanvasLayout,
): void {
  if (!storage) return;
  try {
    storage.setItem(layoutKey(appId, agentId), JSON.stringify(layout));
  } catch {
    // Quota exceeded or storage blocked: the canvas keeps working without saved layout.
  }
}
