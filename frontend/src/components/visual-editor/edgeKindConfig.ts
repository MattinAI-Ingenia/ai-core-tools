import type { GraphEdgeKind } from '../../hooks/useAppGraph';

/**
 * Per-relationship-kind edge styling. Agent-as-tool composition edges are
 * dashed + slightly thicker so they read as structurally different from
 * plain "agent uses resource" edges, which are solid and thinner.
 */
export interface EdgeKindVisual {
  readonly stroke: string;
  readonly dashed: boolean;
  readonly strokeWidth: number;
}

export const EDGE_KIND_VISUALS: Record<GraphEdgeKind, EdgeKindVisual> = {
  tool: { stroke: '#6366f1', dashed: true, strokeWidth: 2 },
  service: { stroke: '#2563eb', dashed: false, strokeWidth: 1.5 },
  silo: { stroke: '#d97706', dashed: false, strokeWidth: 1.5 },
  embedding: { stroke: '#0891b2', dashed: false, strokeWidth: 1.5 },
  skill: { stroke: '#9333ea', dashed: false, strokeWidth: 1.5 },
  mcp: { stroke: '#e11d48', dashed: false, strokeWidth: 1.5 },
  parser: { stroke: '#475569', dashed: false, strokeWidth: 1.5 },
};
