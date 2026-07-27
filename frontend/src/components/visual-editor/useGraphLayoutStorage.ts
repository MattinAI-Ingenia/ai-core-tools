import { useCallback, useMemo } from 'react';
import type { GraphPosition } from './graphLayout';

/**
 * Versioned on-disk shape for the persisted per-app visual editor layout.
 * Bumping `v` lets a future shape change be detected and ignored gracefully
 * (falls back to defaults) instead of throwing on a stale/foreign payload.
 */
interface VisualEditorLayoutV1 {
  readonly v: 1;
  readonly positions: Readonly<Record<string, GraphPosition>>;
  readonly collapsedAgentIds: readonly string[];
}

export interface GraphLayoutStorageState {
  readonly positions: Readonly<Record<string, GraphPosition>>;
  readonly collapsedAgentIds: ReadonlySet<string>;
}

export interface UseGraphLayoutStorageResult {
  /** Reads the persisted layout for the current app, or an empty state if none/invalid. */
  readonly load: () => GraphLayoutStorageState;
  /**
   * Persists `state`, pruning any entry whose node id is not in
   * `knownNodeIds` first - this is what drops stale positions/collapsed ids
   * for resources that no longer exist in the app graph.
   */
  readonly save: (state: GraphLayoutStorageState, knownNodeIds: ReadonlySet<string>) => void;
}

const EMPTY_STATE: GraphLayoutStorageState = { positions: {}, collapsedAgentIds: new Set() };

function storageKeyFor(appId: string | number | undefined): string | undefined {
  if (appId === undefined || appId === '') return undefined;
  return `mattinai:visual-editor:${appId}`;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isGraphPosition(value: unknown): value is GraphPosition {
  if (!isPlainObject(value)) return false;
  return typeof value.x === 'number' && typeof value.y === 'number';
}

function isVisualEditorLayoutV1(value: unknown): value is VisualEditorLayoutV1 {
  if (!isPlainObject(value)) return false;
  if (value.v !== 1) return false;

  const { positions, collapsedAgentIds } = value;
  if (!isPlainObject(positions)) return false;
  if (!Object.values(positions).every(isGraphPosition)) return false;

  if (!Array.isArray(collapsedAgentIds)) return false;
  if (!collapsedAgentIds.every((id): id is string => typeof id === 'string')) return false;

  return true;
}

/**
 * Persistence for the read-only visual editor's client-side layout state:
 * dragged node positions and collapsed-agent ids, scoped to one App via
 * `localStorage` key `mattinai:visual-editor:<appId>`.
 *
 * Every `localStorage` access is wrapped in try/catch - private-mode
 * browsers and quota errors must never crash the canvas, they just mean the
 * layout resets to defaults next time.
 */
export function useGraphLayoutStorage(appId: string | number | undefined): UseGraphLayoutStorageResult {
  const storageKey = useMemo(() => storageKeyFor(appId), [appId]);

  const load = useCallback((): GraphLayoutStorageState => {
    if (!storageKey) return EMPTY_STATE;
    try {
      const raw = window.localStorage.getItem(storageKey);
      if (!raw) return EMPTY_STATE;

      const parsed = JSON.parse(raw) as unknown;
      if (!isVisualEditorLayoutV1(parsed)) return EMPTY_STATE;

      return {
        positions: parsed.positions,
        collapsedAgentIds: new Set(parsed.collapsedAgentIds),
      };
    } catch {
      return EMPTY_STATE;
    }
  }, [storageKey]);

  const save = useCallback(
    (state: GraphLayoutStorageState, knownNodeIds: ReadonlySet<string>): void => {
      if (!storageKey) return;
      try {
        const positions: Record<string, GraphPosition> = {};
        for (const [nodeId, position] of Object.entries(state.positions)) {
          if (knownNodeIds.has(nodeId)) {
            positions[nodeId] = position;
          }
        }

        const collapsedAgentIds = [...state.collapsedAgentIds].filter((id) => knownNodeIds.has(id));

        const payload: VisualEditorLayoutV1 = { v: 1, positions, collapsedAgentIds };
        window.localStorage.setItem(storageKey, JSON.stringify(payload));
      } catch {
        // Storage unavailable (private mode/quota) - drags/collapses still
        // work for the current session, they just won't survive a reload.
      }
    },
    [storageKey],
  );

  return { load, save };
}
