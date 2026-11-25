// Node styling for the playground's LightRAG subgraph bubble — pure functions,
// so the semantics are testable (see tests/LightRAGGraphBubble.test.cjs).
import { colorForType, typeRingIcon, NEUTRAL_COLOR, PARTIAL_COLOR, RING_CITED, RING_RETRIEVED } from './entityTypeColors';

// True when any chunk the element was extracted from is among the cited chunks.
export function fromCitedChunk(sourceId: string | undefined, cited: Set<string>): boolean {
  if (!sourceId || cited.size === 0) return false;
  return sourceId.split('<SEP>').some((id) => cited.has(id.trim()));
}

/**
 * Node fill:
 * - partial nodes (relationship endpoints LightRAG truncated) keep their gray,
 *   whatever the coloring mode — they have no reliable type;
 * - type coloring ON → the entity type's palette color (gray when untyped);
 * - OFF → the old semantics: green when cited, indigo otherwise.
 */
export function bubbleNodeFill(
  entity: { entity_type?: unknown; partial?: boolean; source_id?: string },
  typeColoring: boolean,
  cited: Set<string> = new Set(),
): string {
  if (entity.partial) return PARTIAL_COLOR;
  const type = typeof entity.entity_type === 'string' ? entity.entity_type : undefined;
  if (typeColoring) return colorForType(type);
  return fromCitedChunk(entity.source_id, cited) ? RING_CITED : NEUTRAL_COLOR;
}

/**
 * NVL `overlayIcon` ring for a node, or undefined when it must not show one:
 * only when type coloring is ON — cited → green ring, merely retrieved → indigo.
 * Partial nodes never get a ring (their cited/retrieved status is unknown).
 */
export function bubbleNodeIcon(
  entity: { entity_type?: unknown; partial?: boolean; source_id?: string },
  cited: Set<string>,
  typeColoring: boolean,
): { url: string; position: number[]; size: number } | undefined {
  if (entity.partial || !typeColoring) return undefined;
  const color = fromCitedChunk(entity.source_id, cited) ? RING_CITED : RING_RETRIEVED;
  return typeRingIcon(color);
}
