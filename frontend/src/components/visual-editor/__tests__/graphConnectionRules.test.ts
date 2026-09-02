import { describe, expect, it } from 'vitest';
import type { GraphNode } from '../../../hooks/useAppGraph';
import { isValidGraphConnection, resolveConnection } from '../graphConnectionRules';

function node(id: string, kind: GraphNode['kind'], data: Record<string, unknown> = {}): GraphNode {
  return { id, kind, label: id, data: data as GraphNode['data'] };
}

const agentA = node('agent:1', 'agent', { is_tool: false });
const agentB = node('agent:2', 'agent', { is_tool: true });
const agentC = node('agent:3', 'agent', { is_tool: false });
const silo = node('silo:5', 'silo');
const skill = node('skill:7', 'skill');
const mcp = node('mcp:9', 'mcp');
const service = node('service:2', 'service');

const nodes = [agentA, agentB, agentC, silo, skill, mcp, service];

describe('resolveConnection', () => {
  it('resolves agent -> silo', () => {
    expect(resolveConnection(nodes, { source: 'agent:1', target: 'silo:5' })).toEqual({
      kind: 'silo',
      agentNumericId: 1,
      targetNumericId: 5,
    });
  });

  it('resolves silo -> agent (reversed drag direction) the same way', () => {
    expect(resolveConnection(nodes, { source: 'silo:5', target: 'agent:1' })).toEqual({
      kind: 'silo',
      agentNumericId: 1,
      targetNumericId: 5,
    });
  });

  it('resolves agent -> skill', () => {
    expect(resolveConnection(nodes, { source: 'agent:1', target: 'skill:7' })).toEqual({
      kind: 'skill',
      agentNumericId: 1,
      targetNumericId: 7,
    });
  });

  it('resolves agent -> mcp', () => {
    expect(resolveConnection(nodes, { source: 'agent:1', target: 'mcp:9' })).toEqual({
      kind: 'mcp',
      agentNumericId: 1,
      targetNumericId: 9,
    });
  });

  it('resolves agent -> agent when the target is a tool agent', () => {
    expect(resolveConnection(nodes, { source: 'agent:1', target: 'agent:2' })).toEqual({
      kind: 'tool',
      agentNumericId: 1,
      targetNumericId: 2,
    });
  });

  it('rejects agent -> agent when the target is not a tool agent', () => {
    expect(resolveConnection(nodes, { source: 'agent:1', target: 'agent:3' })).toBeNull();
  });

  it('rejects a self connection', () => {
    expect(resolveConnection(nodes, { source: 'agent:1', target: 'agent:1' })).toBeNull();
  });

  it('rejects an out-of-scope pair (agent -> service)', () => {
    expect(resolveConnection(nodes, { source: 'agent:1', target: 'service:2' })).toBeNull();
  });

  it('rejects an unrelated pair (silo -> skill)', () => {
    expect(resolveConnection(nodes, { source: 'silo:5', target: 'skill:7' })).toBeNull();
  });

  it('rejects when a node id is unknown', () => {
    expect(resolveConnection(nodes, { source: 'agent:1', target: 'silo:999' })).toBeNull();
  });

  it('rejects null source/target', () => {
    expect(resolveConnection(nodes, { source: null, target: 'silo:5' })).toBeNull();
  });
});

describe('isValidGraphConnection', () => {
  it('mirrors resolveConnection as a boolean', () => {
    expect(isValidGraphConnection(nodes, { source: 'agent:1', target: 'silo:5' })).toBe(true);
    expect(isValidGraphConnection(nodes, { source: 'silo:5', target: 'skill:7' })).toBe(false);
  });
});
