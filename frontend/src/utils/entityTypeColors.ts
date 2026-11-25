// Color coding of knowledge-graph entities by type. Pure functions, so the rules are testable.
// Shared by the silo graph view (global graph) and the playground subgraph bubble.

/** Fixed 12-color accessible palette (Tailwind 600 tones). A type keeps its color whatever the order. */
export const TYPE_PALETTE = [
  '#2563eb', // blue-600
  '#dc2626', // red-600
  '#059669', // emerald-600
  '#d97706', // amber-600
  '#7c3aed', // violet-600
  '#0891b2', // cyan-600
  '#db2777', // pink-600
  '#65a30d', // lime-600
  '#4f46e5', // indigo-600
  '#ea580c', // orange-600
  '#0d9488', // teal-600
  '#9333ea', // purple-600
] as const;

/** Fill for entities with no type (or when coloring is off). */
export const NEUTRAL_COLOR = '#6366f1'; // indigo-500 — matches the retrieved color of the bubble

/** The partial-node gray (relationship endpoint LightRAG truncated). */
export const PARTIAL_COLOR = '#94a3b8'; // slate-400

/** Ring colors: cited in the answer (green) vs merely retrieved (indigo). */
export const RING_CITED = '#22c55e'; // green-500
export const RING_RETRIEVED = '#6366f1'; // indigo-500

/** Fill of the element the user clicked (selected node / clicked edge) —
 *  deliberately NOT in TYPE_PALETTE so it can't be mistaken for a type. */
export const PRIMARY_COLOR = '#facc15'; // yellow-400

/** Ring around the elements ADJACENT to the selection (neighbors / endpoints),
 *  so they read as highlighted without stealing the yellow from the chosen one. */
export const HIGHLIGHT_BLUE = '#3b82f6'; // blue-500

/** FNV-1a over the UTF-16 code units — stable across sessions and silos. */
function fnv1a(text: string): number {
  let hash = 0x811c9dc5;
  for (const ch of text) {
    hash ^= ch.codePointAt(0)!;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

/** A stable color per entity type (hash of the type name); null → neutral gray. */
export function colorForType(type: string | null | undefined): string {
  if (!type) return PARTIAL_COLOR;
  return TYPE_PALETTE[fnv1a(type) % TYPE_PALETTE.length];
}

/**
 * NVL `overlayIcon` payload drawing a ring around a node: a circle stroked in
 * `color`, centered on the node, ~115% of its size (NVL has no native outline).
 * The SVG is a data-URI so no asset file is needed; NVL caches it per URL.
 */
export function typeRingIcon(color: string): { url: string; position: number[]; size: number } {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="45" fill="none" stroke="${color}" stroke-width="6"/></svg>`;
  return {
    url: `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`,
    position: [0, 0],
    size: 1.15,
  };
}
