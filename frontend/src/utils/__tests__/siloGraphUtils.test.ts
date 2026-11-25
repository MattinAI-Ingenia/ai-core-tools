import { describe, expect, it } from 'vitest';
import { mergeGraphData, neighborsOf, splitSourceIds } from '../siloGraphUtils';
import type { GraphNode, GraphEdge, SiloGraphData } from '../siloGraphUtils';

function node(id: string, entityType: string | null = null): GraphNode {
  return { id, labels: [], properties: { entity_id: id, entity_type: entityType }, entity_type: entityType };
}

function edge(id: string, source: string, target: string, type = 'RELATED'): GraphEdge {
  return { id, source, target, type, properties: {} };
}

function data(nodes: GraphNode[], edges: GraphEdge[], maxNodes = 200): SiloGraphData {
  return {
    nodes, edges, entity_types: [],
    node_count: nodes.length, edge_count: edges.length,
    total_nodes: 5000, total_edges: 20000, truncated: false,
    max_nodes: maxNodes,
  };
}

describe('splitSourceIds', () => {
  it('splits a SEP-joined source_id and trims whitespace', () => {
    expect(splitSourceIds('res7-p1-chunk-000<SEP> res7-p2-chunk-004 ')).toEqual([
      'res7-p1-chunk-000', 'res7-p2-chunk-004',
    ]);
  });

  it('returns [] for missing/empty source ids', () => {
    expect(splitSourceIds(undefined)).toEqual([]);
    expect(splitSourceIds('')).toEqual([]);
  });
});

describe('mergeGraphData', () => {
  it('adds new nodes/edges and keeps existing ones (dedupe by id)', () => {
    const base = data([node('a'), node('b')], [edge('1', 'a', 'b')]);
    const incoming = data([node('a'), node('c')], [edge('1', 'a', 'b'), edge('2', 'b', 'c')]);

    const merged = mergeGraphData(base, incoming);

    expect(merged.nodes.map((n) => n.id).sort()).toEqual(['a', 'b', 'c']);
    expect(merged.edges.map((e) => e.id).sort()).toEqual(['1', '2']);
  });

  it('drops edges whose endpoints are missing after the merge', () => {
    const base = data([node('a')], []);
    const incoming = data([node('a'), node('c')], [edge('2', 'a', 'ghost')]);

    expect(mergeGraphData(base, incoming).edges).toEqual([]);
  });

  it('respects the canvas cap: at max nodes the base is returned unchanged', () => {
    const base = data([node('a'), node('b')], [], 2);
    const incoming = data([node('a'), node('c')], [edge('2', 'a', 'c')]);

    const merged = mergeGraphData(base, incoming);

    expect(merged.nodes).toHaveLength(2);
    expect(merged.edges).toEqual([]); // 'c' could not be added, so its edge is dropped
  });

  it('keeps the cap when the base payload has no max_nodes (backend responses do not)', () => {
    // A payload without max_nodes would otherwise make the cap NaN — disabled.
    const base = { ...data([node('a')], []), max_nodes: undefined as unknown as number };
    const incoming = data(
      Array.from({ length: 250 }, (_, i) => node(`n${i}`)),
      [],
    );

    expect(mergeGraphData(base, incoming).nodes).toHaveLength(200);
  });
});

describe('neighborsOf', () => {
  it('lists both directions, deduped, with the relationship type', () => {
    const edges = [edge('1', 'a', 'b'), edge('2', 'b', 'a'), edge('3', 'a', 'a')];

    expect(neighborsOf('a', edges)).toEqual([
      { relatedId: 'b', type: 'RELATED' },
      { relatedId: 'a', type: 'RELATED' },
    ]);
  });
});
