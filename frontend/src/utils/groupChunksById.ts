import type { LightRAGChunk } from '../types/streaming';

export interface ChunkGroup {
  chunk: LightRAGChunk;
  /** 0-based positions of every copy in the original chunk list. Citations are positional
   * (`cite://N` is `chunks[N-1]`), so each position + 1 is a number the answer may cite. */
  positions: number[];
}

/** One group per distinct chunk id, in first-seen order; the group keeps its first copy.
 * A turn's payload repeats a page whenever several tool calls return it, and the bubble
 * should show it once. A chunk without an id is never merged with another. */
export function groupChunksById(chunks: LightRAGChunk[]): ChunkGroup[] {
  const byId = new Map<string, ChunkGroup>();
  const groups: ChunkGroup[] = [];
  chunks.forEach((chunk, position) => {
    const existing = chunk.id ? byId.get(chunk.id) : undefined;
    if (existing) {
      existing.positions.push(position);
      return;
    }
    const group: ChunkGroup = { chunk, positions: [position] };
    groups.push(group);
    if (chunk.id) byId.set(chunk.id, group);
  });
  return groups;
}
