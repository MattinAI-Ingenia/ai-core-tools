import type { GraphEdgeKind } from '../../hooks/useAppGraph';
import type { Agent } from '../../services/api';

export interface RelationshipChange {
  readonly kind: GraphEdgeKind;
  readonly targetNumericId: number;
}

function toggleId(ids: readonly number[] | undefined, targetId: number, mode: 'add' | 'remove'): number[] {
  const current = ids ?? [];
  if (mode === 'add') {
    return current.includes(targetId) ? [...current] : [...current, targetId];
  }
  return current.filter((id) => id !== targetId);
}

/**
 * Computes the `Agent` fields that change when a `silo`/`skill`/`mcp`/`tool`
 * edge is added or removed on the canvas. Returns only the changed field(s)
 * - the caller spreads this over the agent's current full record before
 * calling `apiService.updateAgent`, since that endpoint expects the full
 * agent payload rather than a partial patch.
 */
export function buildRelationshipChange(
  agent: Agent,
  change: RelationshipChange,
  mode: 'add' | 'remove',
): Partial<Agent> & { silo_id?: number | null } {
  switch (change.kind) {
    case 'silo':
      return { silo_id: mode === 'add' ? change.targetNumericId : null };
    case 'skill':
      return { skill_ids: toggleId(agent.skill_ids, change.targetNumericId, mode) };
    case 'mcp':
      return { mcp_config_ids: toggleId(agent.mcp_config_ids, change.targetNumericId, mode) };
    case 'tool':
      return { tool_ids: toggleId(agent.tool_ids, change.targetNumericId, mode) };
    default:
      throw new Error(`Edge kind "${change.kind}" is not editable from the canvas`);
  }
}
