import { describe, expect, it } from 'vitest';
import type { GraphEdge, GraphNode } from '../../../hooks/useAppGraph';
import { toFlowEdges, toFlowNodes } from '../graphAdapter';

function node(id: string, kind: GraphNode['kind']): GraphNode {
  return { id, kind, label: id, data: {} as GraphNode['data'] };
}

const noopOptions = { collapsedAgentIds: new Set<string>(), onToggleCollapse: () => {} };

describe('toFlowNodes connectable flag', () => {
  it('marks agent, silo, skill, mcp nodes connectable', () => {
    const nodes = toFlowNodes(
      [node('agent:1', 'agent'), node('silo:1', 'silo'), node('skill:1', 'skill'), node('mcp:1', 'mcp')],
      new Map(),
      noopOptions,
    );
    for (const flowNode of nodes) {
      expect(flowNode.connectable).toBe(true);
    }
  });

  it('leaves service, embedding, parser nodes non-connectable', () => {
    const nodes = toFlowNodes(
      [node('service:1', 'service'), node('embedding:1', 'embedding'), node('parser:1', 'parser')],
      new Map(),
      noopOptions,
    );
    for (const flowNode of nodes) {
      expect(flowNode.connectable).toBe(false);
    }
  });

});

describe('toFlowNodes deletable flag', () => {
  it('marks agent, silo, skill nodes deletable', () => {
    const nodes = toFlowNodes(
      [node('agent:1', 'agent'), node('silo:1', 'silo'), node('skill:1', 'skill')],
      new Map(),
      noopOptions,
    );
    for (const flowNode of nodes) {
      expect(flowNode.deletable).toBe(true);
    }
  });

  it('leaves mcp, service, embedding, parser nodes non-deletable', () => {
    const nodes = toFlowNodes(
      [node('mcp:1', 'mcp'), node('service:1', 'service'), node('embedding:1', 'embedding'), node('parser:1', 'parser')],
      new Map(),
      noopOptions,
    );
    for (const flowNode of nodes) {
      expect(flowNode.deletable).toBe(false);
    }
  });
});

describe('toFlowEdges deletable flag', () => {
  function edge(kind: GraphEdge['kind']): GraphEdge {
    return { id: `${kind}:a->b`, source: 'a', target: 'b', kind };
  }

  it('marks silo, skill, mcp, tool edges deletable', () => {
    const edges = toFlowEdges([edge('silo'), edge('skill'), edge('mcp'), edge('tool')]);
    for (const flowEdge of edges) {
      expect(flowEdge.deletable).toBe(true);
    }
  });

  it('leaves service, embedding, parser edges non-deletable', () => {
    const edges = toFlowEdges([edge('service'), edge('embedding'), edge('parser')]);
    for (const flowEdge of edges) {
      expect(flowEdge.deletable).toBe(false);
    }
  });
});
