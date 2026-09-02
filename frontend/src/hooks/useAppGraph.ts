import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/api';
import type {
  Agent,
  AIService,
  EmbeddingService,
  Silo,
  DataStructure,
} from '../services/api';
// Skill and MCPConfig are declared in core/types.ts; api.ts imports them as
// `import type` without re-exporting, so they must be sourced from here too.
import type { Skill, MCPConfig } from '../core/types';

/**
 * Discriminator for the kind of resource a graph node represents.
 * Framework-neutral - React Flow-specific node types are derived from this
 * by the canvas layer (sub-issue 2), never imported here.
 */
export type GraphNodeKind =
  | 'agent'
  | 'service'
  | 'silo'
  | 'embedding'
  | 'skill'
  | 'mcp'
  | 'parser';

/** Union of the raw entity payloads a graph node can wrap. */
export type GraphNodeData =
  | Agent
  | AIService
  | Silo
  | EmbeddingService
  | Skill
  | MCPConfig
  | DataStructure;

/**
 * A single node in the app resource graph.
 * `id` is namespaced by kind (e.g. `agent:12`, `service:3`) so ids stay
 * unique across entity types that may share numeric primary keys.
 */
export interface GraphNode<T extends GraphNodeData = GraphNodeData> {
  readonly id: string;
  readonly kind: GraphNodeKind;
  readonly label: string;
  readonly data: T;
}

/** Relationship type carried by a graph edge. */
export type GraphEdgeKind =
  | 'tool'
  | 'silo'
  | 'service'
  | 'skill'
  | 'mcp'
  | 'parser'
  | 'embedding';

/**
 * A directed relationship between two graph nodes.
 * `source`/`target` reference `GraphNode.id` values.
 */
export interface GraphEdge {
  readonly id: string;
  readonly source: string;
  readonly target: string;
  readonly kind: GraphEdgeKind;
}

export interface AppGraph {
  readonly nodes: GraphNode[];
  readonly edges: GraphEdge[];
}

interface UseAppGraphResult extends AppGraph {
  readonly loading: boolean;
  readonly error: string | null;
  readonly refetch: () => Promise<void>;
}

/**
 * Relationship fields sourced from `apiService.getAgent` (AgentDetailSchema).
 * `getAgents` (list) returns AgentListItemSchema server-side, which does NOT
 * carry `silo_id`/`output_parser_id`/`tool_ids`/`mcp_config_ids`/`skill_ids` -
 * reading those off a list item is silently `undefined` at runtime with no
 * compile error. This distinct, narrower type exists so edge-building code
 * can only be fed detail-shaped objects, and any future attempt to pass a
 * list item here is a type error instead of a silent empty edge set.
 */
interface AgentDetailGraphItem {
  readonly agent_id: number;
  readonly service_id?: number;
  readonly silo_id?: number;
  readonly output_parser_id?: number;
  readonly skill_ids?: number[];
  readonly mcp_config_ids?: number[];
  readonly tool_ids?: number[];
}

/**
 * Relationship field sourced from `apiService.getSilo` (SiloDetailSchema).
 * `getSilos` (list) returns SiloListItemSchema, which has no
 * `embedding_service_id` - same rationale as `AgentDetailGraphItem` above.
 */
interface SiloDetailGraphItem {
  readonly silo_id: number;
  readonly embedding_service_id?: number;
}

function nodeId(kind: GraphNodeKind, id: number): string {
  return `${kind}:${id}`;
}

export function parseNodeId(id: string): { kind: GraphNodeKind; numericId: number } {
  const [kind, rawId] = id.split(':') as [GraphNodeKind, string];
  return { kind, numericId: Number.parseInt(rawId, 10) };
}

function edgeId(kind: GraphEdgeKind, source: string, target: string): string {
  return `${kind}:${source}->${target}`;
}

/**
 * Builds an agent-centric resource graph for an App: agents plus every
 * AIService, Silo, EmbeddingService, Skill, MCPConfig and OutputParser
 * they reference, deduplicated so a shared resource yields a single node
 * with one edge per referencing agent.
 *
 * Read-only data layer for the visual app graph editor. Deliberately
 * framework-neutral (no @xyflow/react types) so it can be consumed by any
 * canvas implementation and stays tree-shakeable for library consumers
 * that never render the graph.
 */
export function useAppGraph(appId: string | number | undefined): UseAppGraphResult {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchGraph = useCallback(async () => {
    if (appId === undefined || appId === '') {
      setNodes([]);
      setEdges([]);
      setLoading(false);
      return;
    }

    const numericAppId = typeof appId === 'number' ? appId : Number.parseInt(appId, 10);
    if (Number.isNaN(numericAppId)) {
      setNodes([]);
      setEdges([]);
      setLoading(false);
      setError('Invalid app id');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const [agents, silos, aiServices, embeddingServices, mcpConfigs, skills, outputParsers] =
        await Promise.all([
          apiService.getAgents(numericAppId),
          apiService.getSilos(numericAppId),
          apiService.getAIServices(numericAppId),
          apiService.getEmbeddingServices(numericAppId),
          apiService.getMCPConfigs(numericAppId),
          apiService.getSkills(numericAppId),
          apiService.getOutputParsers(numericAppId),
        ]);

      // ponytail: N+1 detail fetch, bounded by (#agents + #silos) for this app and
      // fully parallelized via Promise.all/Promise.all. Needed because the list
      // endpoints above only return id/label-level fields (AgentListItemSchema /
      // SiloListItemSchema) - the relationship fields used for edges only exist on
      // the detail schemas (AgentDetailSchema / SiloDetailSchema). Upgrade path if
      // this ever gets slow for apps with many agents/silos: add the relationship
      // fields to the list schemas server-side and drop this second round-trip.
      const [agentDetails, siloDetails] = await Promise.all([
        Promise.all(
          agents.map((agent): Promise<AgentDetailGraphItem> => apiService.getAgent(numericAppId, agent.agent_id)),
        ),
        Promise.all(
          silos.map((silo): Promise<SiloDetailGraphItem> => apiService.getSilo(numericAppId, silo.silo_id)),
        ),
      ]);

      const graphNodes: GraphNode[] = [];
      const graphEdges: GraphEdge[] = [];

      // One node per resource - dedup happens naturally because every
      // entity here is added exactly once, regardless of how many agents
      // reference it. Edges (added below, per agent) are what fan out.
      const agentNodeIds = new Set<number>();
      for (const agent of agents) {
        agentNodeIds.add(agent.agent_id);
        graphNodes.push({
          id: nodeId('agent', agent.agent_id),
          kind: 'agent',
          label: agent.name,
          data: agent,
        });
      }

      const serviceNodeIds = new Set<number>();
      for (const service of aiServices) {
        serviceNodeIds.add(service.service_id);
        graphNodes.push({
          id: nodeId('service', service.service_id),
          kind: 'service',
          label: service.name,
          data: service,
        });
      }

      const siloNodeIds = new Set<number>();
      for (const silo of silos) {
        siloNodeIds.add(silo.silo_id);
        graphNodes.push({
          id: nodeId('silo', silo.silo_id),
          kind: 'silo',
          label: silo.name,
          data: silo,
        });
      }

      const embeddingNodeIds = new Set<number>();
      for (const embedding of embeddingServices) {
        embeddingNodeIds.add(embedding.service_id);
        graphNodes.push({
          id: nodeId('embedding', embedding.service_id),
          kind: 'embedding',
          label: embedding.name,
          data: embedding,
        });
      }

      const skillNodeIds = new Set<number>();
      for (const skill of skills) {
        skillNodeIds.add(skill.skill_id);
        graphNodes.push({
          id: nodeId('skill', skill.skill_id),
          kind: 'skill',
          label: skill.name,
          data: skill,
        });
      }

      const mcpNodeIds = new Set<number>();
      for (const mcpConfig of mcpConfigs) {
        mcpNodeIds.add(mcpConfig.config_id);
        graphNodes.push({
          id: nodeId('mcp', mcpConfig.config_id),
          kind: 'mcp',
          label: mcpConfig.name,
          data: mcpConfig,
        });
      }

      const parserNodeIds = new Set<number>();
      for (const parser of outputParsers) {
        parserNodeIds.add(parser.parser_id);
        graphNodes.push({
          id: nodeId('parser', parser.parser_id),
          kind: 'parser',
          label: parser.name,
          data: parser,
        });
      }

      // Edges: one per (agent, referenced resource) pair, sourced from the
      // agent DETAIL objects (see AgentDetailGraphItem above) - never from the
      // list items iterated for nodes further up. Shared resources naturally
      // produce multiple edges into the single dedup'd node created above.
      for (const agent of agentDetails) {
        const agentId = nodeId('agent', agent.agent_id);

        if (agent.service_id !== undefined && serviceNodeIds.has(agent.service_id)) {
          const target = nodeId('service', agent.service_id);
          graphEdges.push({ id: edgeId('service', agentId, target), source: agentId, target, kind: 'service' });
        }

        if (agent.silo_id !== undefined && siloNodeIds.has(agent.silo_id)) {
          const target = nodeId('silo', agent.silo_id);
          graphEdges.push({ id: edgeId('silo', agentId, target), source: agentId, target, kind: 'silo' });
        }

        if (agent.output_parser_id !== undefined && parserNodeIds.has(agent.output_parser_id)) {
          const target = nodeId('parser', agent.output_parser_id);
          graphEdges.push({ id: edgeId('parser', agentId, target), source: agentId, target, kind: 'parser' });
        }

        for (const skillId of agent.skill_ids ?? []) {
          if (!skillNodeIds.has(skillId)) continue;
          const target = nodeId('skill', skillId);
          graphEdges.push({ id: edgeId('skill', agentId, target), source: agentId, target, kind: 'skill' });
        }

        for (const mcpConfigId of agent.mcp_config_ids ?? []) {
          if (!mcpNodeIds.has(mcpConfigId)) continue;
          const target = nodeId('mcp', mcpConfigId);
          graphEdges.push({ id: edgeId('mcp', agentId, target), source: agentId, target, kind: 'mcp' });
        }

        for (const toolAgentId of agent.tool_ids ?? []) {
          if (!agentNodeIds.has(toolAgentId)) continue;
          const target = nodeId('agent', toolAgentId);
          graphEdges.push({ id: edgeId('tool', agentId, target), source: agentId, target, kind: 'tool' });
        }
      }

      // Silo -> EmbeddingService links, sourced from silo DETAIL objects (see
      // SiloDetailGraphItem above) - the silo list items have no embedding_service_id.
      for (const silo of siloDetails) {
        if (silo.embedding_service_id === undefined || !embeddingNodeIds.has(silo.embedding_service_id)) {
          continue;
        }
        const source = nodeId('silo', silo.silo_id);
        const target = nodeId('embedding', silo.embedding_service_id);
        graphEdges.push({ id: edgeId('embedding', source, target), source, target, kind: 'embedding' });
      }

      setNodes(graphNodes);
      setEdges(graphEdges);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load app graph');
      setNodes([]);
      setEdges([]);
    } finally {
      setLoading(false);
    }
  }, [appId]);

  useEffect(() => {
    fetchGraph();
  }, [fetchGraph]);

  return { nodes, edges, loading, error, refetch: fetchGraph };
}
