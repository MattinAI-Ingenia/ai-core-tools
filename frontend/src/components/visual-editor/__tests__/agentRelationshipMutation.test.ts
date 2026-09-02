import { describe, expect, it } from 'vitest';
import type { Agent } from '../../../services/api';
import { buildRelationshipChange } from '../agentRelationshipMutation';

function agent(overrides: Partial<Agent> = {}): Agent {
  return {
    agent_id: 1,
    name: 'Test agent',
    system_prompt: '',
    prompt_template: '',
    type: 'AGENT',
    is_tool: false,
    has_memory: false,
    enable_code_interpreter: false,
    memory_max_messages: 20,
    memory_max_tokens: 4000,
    memory_summarize_threshold: 10,
    temperature: 0.7,
    created_at: '2026-01-01T00:00:00Z',
    request_count: 0,
    ai_services: [],
    ...overrides,
  } as Agent;
}

describe('buildRelationshipChange', () => {
  it('sets silo_id on add', () => {
    expect(buildRelationshipChange(agent(), { kind: 'silo', targetNumericId: 5 }, 'add')).toEqual({
      silo_id: 5,
    });
  });

  it('clears silo_id on remove', () => {
    expect(
      buildRelationshipChange(agent({ silo_id: 5 }), { kind: 'silo', targetNumericId: 5 }, 'remove'),
    ).toEqual({ silo_id: null });
  });

  it('appends to skill_ids on add without duplicating', () => {
    expect(
      buildRelationshipChange(agent({ skill_ids: [1, 2] }), { kind: 'skill', targetNumericId: 2 }, 'add'),
    ).toEqual({ skill_ids: [1, 2] });
    expect(
      buildRelationshipChange(agent({ skill_ids: [1] }), { kind: 'skill', targetNumericId: 2 }, 'add'),
    ).toEqual({ skill_ids: [1, 2] });
  });

  it('removes from skill_ids on remove', () => {
    expect(
      buildRelationshipChange(agent({ skill_ids: [1, 2] }), { kind: 'skill', targetNumericId: 2 }, 'remove'),
    ).toEqual({ skill_ids: [1] });
  });

  it('appends to mcp_config_ids on add', () => {
    expect(
      buildRelationshipChange(agent({ mcp_config_ids: [3] }), { kind: 'mcp', targetNumericId: 4 }, 'add'),
    ).toEqual({ mcp_config_ids: [3, 4] });
  });

  it('removes from mcp_config_ids on remove', () => {
    expect(
      buildRelationshipChange(agent({ mcp_config_ids: [3, 4] }), { kind: 'mcp', targetNumericId: 4 }, 'remove'),
    ).toEqual({ mcp_config_ids: [3] });
  });

  it('appends to tool_ids on add', () => {
    expect(
      buildRelationshipChange(agent({ tool_ids: [] }), { kind: 'tool', targetNumericId: 8 }, 'add'),
    ).toEqual({ tool_ids: [8] });
  });

  it('removes from tool_ids on remove', () => {
    expect(
      buildRelationshipChange(agent({ tool_ids: [8, 9] }), { kind: 'tool', targetNumericId: 8 }, 'remove'),
    ).toEqual({ tool_ids: [9] });
  });

  it('treats a missing array as empty when adding', () => {
    expect(buildRelationshipChange(agent(), { kind: 'skill', targetNumericId: 1 }, 'add')).toEqual({
      skill_ids: [1],
    });
  });

  it('treats a missing array as empty when removing (no-op)', () => {
    expect(buildRelationshipChange(agent(), { kind: 'skill', targetNumericId: 1 }, 'remove')).toEqual({
      skill_ids: [],
    });
  });
});
