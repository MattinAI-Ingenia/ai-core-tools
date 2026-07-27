import type { GraphNode, GraphNodeKind } from '../../hooks/useAppGraph';

/**
 * Deterministic placeholder layout: one column per node kind (in a fixed,
 * read-left-to-right order that roughly follows the direction edges are
 * drawn in `useAppGraph`), rows stacked top-to-bottom within a column.
 *
 * This is intentionally the ONLY place that knows about node coordinates so
 * sub-issue 3 (real layout + drag persistence) can swap this function out
 * without touching the adapter or the node components.
 */
const COLUMN_ORDER: readonly GraphNodeKind[] = [
  'agent',
  'service',
  'silo',
  'embedding',
  'skill',
  'mcp',
  'parser',
];

const COLUMN_WIDTH = 260;
const ROW_HEIGHT = 110;

export interface GraphPosition {
  readonly x: number;
  readonly y: number;
}

/**
 * Computes a `{ x, y }` position per node id, grouped by kind into columns.
 */
export function computeGraphLayout(nodes: readonly GraphNode[]): ReadonlyMap<string, GraphPosition> {
  const nodesByKind = new Map<GraphNodeKind, GraphNode[]>();
  for (const node of nodes) {
    const bucket = nodesByKind.get(node.kind);
    if (bucket) {
      bucket.push(node);
    } else {
      nodesByKind.set(node.kind, [node]);
    }
  }

  const positions = new Map<string, GraphPosition>();
  COLUMN_ORDER.forEach((kind, columnIndex) => {
    const bucket = nodesByKind.get(kind) ?? [];
    // Center shorter columns vertically against the tallest one so the
    // graph reads as a balanced grid instead of top-aligned staircases.
    const tallestColumn = Math.max(
      1,
      ...COLUMN_ORDER.map((otherKind) => nodesByKind.get(otherKind)?.length ?? 0),
    );
    const verticalOffset = ((tallestColumn - bucket.length) * ROW_HEIGHT) / 2;

    bucket.forEach((node, rowIndex) => {
      positions.set(node.id, {
        x: columnIndex * COLUMN_WIDTH,
        y: verticalOffset + rowIndex * ROW_HEIGHT,
      });
    });
  });

  return positions;
}
