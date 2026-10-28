import type { LightRAGGraphData } from '../types/streaming';

export interface SiloGraph {
  siloId?: number;
  siloName?: string;
  graph: LightRAGGraphData;
}

/** Split a turn's merged LightRAG payload into one graph per source silo
 * (knowledge-router turns). Untagged payloads come back as a single group. */
export function splitGraphBySilo(graph: LightRAGGraphData): SiloGraph[] {
  const data = graph.data ?? {};
  const entities = data.entities ?? [];
  const relationships = data.relationships ?? [];
  const chunks = data.chunks ?? [];
  const all = [...entities, ...relationships, ...chunks];
  const siloIds = [...new Set(all.map((x) => x.silo_id))];

  if (siloIds.length <= 1) {
    return [{ siloId: siloIds[0], siloName: all[0]?.silo_name, graph }];
  }
  return siloIds.map((siloId) => ({
    siloId,
    siloName: all.find((x) => x.silo_id === siloId)?.silo_name,
    graph: {
      ...graph,
      data: {
        ...data,
        entities: entities.filter((e) => e.silo_id === siloId),
        relationships: relationships.filter((r) => r.silo_id === siloId),
        chunks: chunks.filter((c) => c.silo_id === siloId),
      },
    },
  }));
}
