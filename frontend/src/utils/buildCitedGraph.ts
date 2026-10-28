import type { LightRAGChunk, LightRAGGraphData } from '../types/streaming';

// Same separator LightRAG joins the chunk ids of an entity/relationship
// `source_id` with — kept in sync with LightRAGGraphBubble.
const GRAPH_FIELD_SEP = '<SEP>';

export interface CitedGraph {
  graph: LightRAGGraphData;
  /** Display number (1..X, citation order) of each chunk in graph.data.chunks —
   *  matches the renumbered [N] badges MessageContent renders. */
  chunkNumbers: number[];
}

/** Extract the "cited evidence" view of a turn's LightRAG payload: only the
 *  chunks the answer cited ([N](cite://N)), in first-citation order, plus the
 *  entities/relationships extracted from them. Endpoints of cited relationships
 *  are kept even when their own source chunks were not cited, so a cited edge
 *  never loses an endpoint — they still render in their normal (non-green) color. */
export function buildCitedGraph(
  graph: LightRAGGraphData,
  citedChunkIds: string[],
): CitedGraph | null {
  if (citedChunkIds.length === 0) return null;
  const data = graph.data ?? {};
  const chunks = data.chunks ?? [];
  const cited = new Set(citedChunkIds);

  const fromCited = (sourceId: string | undefined) =>
    !!sourceId && sourceId.split(GRAPH_FIELD_SEP).some((id) => cited.has(id.trim()));

  // First copy of each chunk (a payload repeats a page across tool calls).
  const byId = new Map<string, LightRAGChunk>();
  chunks.forEach((chunk) => {
    if (chunk.id && !byId.has(chunk.id)) byId.set(chunk.id, chunk);
  });

  const citedIds = citedChunkIds.filter((id) => byId.has(id));
  if (citedIds.length === 0) return null;

  const relationships = (data.relationships ?? []).filter((r) => fromCited(r.source_id));
  const endpointIds = new Set<string>();
  relationships.forEach((r) => {
    if (r.source) endpointIds.add(String(r.source));
    if (r.target) endpointIds.add(String(r.target));
  });

  return {
    graph: {
      ...graph,
      data: {
        ...data,
        chunks: citedIds.map((id) => byId.get(id)!),
        relationships,
        entities: (data.entities ?? []).filter(
          (e) => fromCited(e.source_id) || endpointIds.has(String(e.id)),
        ),
      },
    },
    chunkNumbers: citedIds.map((_, i) => i + 1),
  };
}

/** Display renumbering for citations. The LLM cites global positions in the
 *  chunk payload it was given (e.g. [33] of 66 chunks); this renumbers badges
 *  1..X over the DISTINCT cited sources, in first-citation order. The
 *  cite://N href keeps the machine pointer to the payload — only the shown
 *  number changes, so history replays stay deterministic. */
export interface CitationNumbering {
  /** Distinct cited chunk ids, in first-citation order. */
  citedChunkIds: string[];
  /** cite://N → display number (1..X). */
  numbersByCite: Map<number, number>;
  /** chunk id → display number (1..X). */
  numberByChunkId: Map<string, number>;
}

const EMPTY_NUMBERING: CitationNumbering = {
  citedChunkIds: [],
  numbersByCite: new Map(),
  numberByChunkId: new Map(),
};

export function numberingFromContent(content: string, chunks: LightRAGChunk[]): CitationNumbering {
  const citedChunkIds: string[] = [];
  const numbersByCite = new Map<number, number>();
  const numberByChunkId = new Map<string, number>();
  // Array.from, not for..of: the ES5 downlevel of for..of indexes the matchAll
  // iterator with .length (undefined), silently iterating nothing.
  Array.from(content.matchAll(/\(cite:\/\/(\d+)\)/g)).forEach((match) => {
    const cite = Number(match[1]);
    const id = chunks[cite - 1]?.id;
    // Unresolvable citation (not in the payload): leave it unnumbered — its
    // badge keeps the raw number and degrades to a plain marker anyway.
    if (!id) return;
    let display = numberByChunkId.get(id);
    if (display === undefined) {
      display = citedChunkIds.length + 1;
      citedChunkIds.push(id);
      numberByChunkId.set(id, display);
    }
    numbersByCite.set(cite, display);
  });
  return { citedChunkIds, numbersByCite, numberByChunkId };
}

export { EMPTY_NUMBERING };
