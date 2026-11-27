import type { RerankServiceOption, RerankServiceOptions } from '../types/services';
import { selectableServiceOptions } from './serviceLifecycle';

/** Mirrors backend DEFAULT_RERANK_TOP_N (used when no top N is stored). Candidates reranked = max(rag_k, top N). */
export const RERANK_TOP_N_DEFAULT = 30;
export const RERANK_TOP_N_MIN = 1;
export const RERANK_TOP_N_MAX = 50;

/** Clamp a raw input to 1..50; an empty or non-numeric input falls back to `fallback`. */
export function clampRerankTopN(raw: string | number, fallback: number = RERANK_TOP_N_DEFAULT): number {
  const n = typeof raw === 'number' ? Math.trunc(raw) : Number.parseInt(raw, 10);
  if (Number.isNaN(n)) return fallback;
  return Math.min(RERANK_TOP_N_MAX, Math.max(RERANK_TOP_N_MIN, n));
}

/** Services that can be picked: retired ones are hidden unless one is the current selection. */
export function rerankSelectableOptions(
  options: RerankServiceOptions | null,
  currentId: number | null,
): RerankServiceOption[] {
  return selectableServiceOptions(options?.options, currentId);
}

/**
 * FR-11: nothing can rerank. True only once the options are loaded, with no effective default and no
 * non-retired service to pick. While the options are unknown (loading or failed) it is false, so the
 * backend stays the final authority.
 */
export function isRerankUnavailable(options: RerankServiceOptions | null): boolean {
  if (!options) return false;
  return options.effective_default == null && rerankSelectableOptions(options, null).length === 0;
}

/** True when reranking would resolve a service: an explicit pick or an inherited default (mirrors the backend 422). */
export function rerankResolves(serviceId: number | null, options: RerankServiceOptions | null): boolean {
  if (!options) return true;
  return serviceId != null || options.effective_default != null;
}
