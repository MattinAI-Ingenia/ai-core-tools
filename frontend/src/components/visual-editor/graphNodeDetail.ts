import type { GraphNode } from '../../hooks/useAppGraph';
import type { Agent, AIService, EmbeddingService, Silo, DataStructure } from '../../services/api';
import type { Skill, MCPConfig } from '../../core/types';

/**
 * Builds the short secondary detail line shown under a node's type caption
 * (e.g. provider/model for services, doc count for silos). Returns
 * `undefined` when the entity has nothing meaningful to show, in which case
 * the node card renders just the label + type caption.
 *
 * Narrows `GraphNode.data` by `node.kind` via assertion - safe because
 * `useAppGraph` only ever pairs a given `kind` with its matching payload
 * type (see `GraphNodeData`), it just doesn't encode that link at the type
 * level (`data` is a flat union).
 */
export function describeGraphNode(node: GraphNode): string | undefined {
  switch (node.kind) {
    case 'agent': {
      const agent = node.data as Agent;
      return agent.ai_service ? `${agent.ai_service.provider} · ${agent.ai_service.model_name}` : undefined;
    }
    case 'service': {
      const service = node.data as AIService;
      return `${service.provider} · ${service.model_name}`;
    }
    case 'embedding': {
      const embedding = node.data as EmbeddingService;
      return `${embedding.provider} · ${embedding.model_name}`;
    }
    case 'silo': {
      const silo = node.data as Silo;
      return `${silo.docs_count} doc${silo.docs_count === 1 ? '' : 's'}`;
    }
    case 'skill': {
      const skill = node.data as Skill;
      return skill.description;
    }
    case 'mcp': {
      const mcpConfig = node.data as MCPConfig;
      return mcpConfig.transport_type;
    }
    case 'parser': {
      const parser = node.data as DataStructure;
      return parser.field_count !== undefined
        ? `${parser.field_count} field${parser.field_count === 1 ? '' : 's'}`
        : undefined;
    }
    default:
      return undefined;
  }
}
