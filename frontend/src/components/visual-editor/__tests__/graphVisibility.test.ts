import { describe, expect, it } from 'vitest';
import type { GraphEdge, GraphNode } from '../../../hooks/useAppGraph';
import { computeVisibleGraph } from '../graphVisibility';

function node(id: string, kind: GraphNode['kind']): GraphNode {
  return { id, kind, label: id, data: {} as GraphNode['data'] };
}

function edge(kind: GraphEdge['kind'], source: string, target: string): GraphEdge {
  return { id: `${kind}:${source}->${target}`, source, target, kind };
}

describe('computeVisibleGraph - orphan resources', () => {
  it('always shows a silo with no agent referencing it, agent expanded or not', () => {
    const nodes = [node('agent:1', 'agent'), node('silo:9', 'silo')];
    const edges: GraphEdge[] = [];

    const expanded = computeVisibleGraph(nodes, edges, new Set());
    expect(expanded.nodes.map((n) => n.id)).toEqual(expect.arrayContaining(['agent:1', 'silo:9']));

    const collapsed = computeVisibleGraph(nodes, edges, new Set(['agent:1']));
    expect(collapsed.nodes.map((n) => n.id)).toEqual(expect.arrayContaining(['agent:1', 'silo:9']));
  });

  it('still hides a resource attached only to a collapsed agent (existing behavior preserved)', () => {
    const nodes = [node('agent:1', 'agent'), node('silo:9', 'silo')];
    const edges = [edge('silo', 'agent:1', 'silo:9')];

    const collapsed = computeVisibleGraph(nodes, edges, new Set(['agent:1']));
    expect(collapsed.nodes.map((n) => n.id)).not.toContain('silo:9');

    const expanded = computeVisibleGraph(nodes, edges, new Set());
    expect(expanded.nodes.map((n) => n.id)).toContain('silo:9');
  });

  it('pulls in an orphan silo own dependency (its embedding service) too', () => {
    const nodes = [node('agent:1', 'agent'), node('silo:9', 'silo'), node('embedding:3', 'embedding')];
    const edges = [edge('embedding', 'silo:9', 'embedding:3')];

    const result = computeVisibleGraph(nodes, edges, new Set(['agent:1']));
    expect(result.nodes.map((n) => n.id)).toEqual(
      expect.arrayContaining(['agent:1', 'silo:9', 'embedding:3']),
    );
    expect(result.edges.map((e) => e.id)).toContain('embedding:silo:9->embedding:3');
  });

  it('shows an orphan skill/mcp/service/parser the same way (generalized, not silo-only)', () => {
    const nodes = [
      node('agent:1', 'agent'),
      node('skill:1', 'skill'),
      node('mcp:1', 'mcp'),
      node('service:1', 'service'),
      node('parser:1', 'parser'),
    ];
    const result = computeVisibleGraph(nodes, [], new Set(['agent:1']));
    expect(result.nodes.map((n) => n.id)).toEqual(
      expect.arrayContaining(['skill:1', 'mcp:1', 'service:1', 'parser:1']),
    );
  });

  it('does not duplicate a resource that is both orphan-reachable and agent-reachable', () => {
    // silo:9 has no agent edge (orphan by that definition) but is also
    // reachable via its own embedding edge from another orphan silo chain -
    // exercising that the BFS seed set doesn't double-add nodes.
    const nodes = [node('agent:1', 'agent'), node('silo:9', 'silo')];
    const edges = [edge('silo', 'agent:1', 'silo:9')];

    const result = computeVisibleGraph(nodes, edges, new Set());
    const siloOccurrences = result.nodes.filter((n) => n.id === 'silo:9');
    expect(siloOccurrences).toHaveLength(1);
  });
});
