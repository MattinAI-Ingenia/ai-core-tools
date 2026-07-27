import type { ComponentType } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { Handle, Position, type NodeProps, type Node } from '@xyflow/react';
import type { GraphNodeKind } from '../../hooks/useAppGraph';
import { NODE_KIND_VISUALS } from './nodeKindConfig';

/** Data payload carried by every node rendered on the app graph canvas. */
export interface AppFlowNodeData extends Record<string, unknown> {
  readonly label: string;
  readonly detail?: string;
  /**
   * Agent nodes only: whether this agent's satellite resources are
   * currently hidden. `undefined` for every non-agent node kind.
   */
  readonly collapsed?: boolean;
  /** Agent nodes only: toggles `collapsed` for this agent. */
  readonly onToggleCollapse?: () => void;
}

export type AppFlowNode = Node<AppFlowNodeData, GraphNodeKind>;

/**
 * Shared presentational card used by every per-kind node component below.
 * Not registered directly in `nodeTypes` - each kind gets its own thin
 * wrapper so React Flow's `nodeTypes` map has one distinct component per
 * `GraphNodeKind`, matching the entity kinds from `useAppGraph`.
 */
function EntityNodeCard({ kind, data }: { readonly kind: GraphNodeKind; readonly data: AppFlowNodeData }) {
  const visual = NODE_KIND_VISUALS[kind];
  const Icon = visual.icon;
  const isCollapsibleAgent = kind === 'agent' && data.onToggleCollapse !== undefined;

  return (
    <div
      className={`group relative flex items-center gap-3 rounded-xl border shadow-sm transition-shadow duration-150 hover:shadow-md ${visual.borderClass} ${visual.bgClass} ${
        visual.emphasize ? 'min-w-[220px] px-4 py-3 border-2' : 'min-w-[190px] px-3 py-2.5'
      }`}
    >
      <Handle
        type="target"
        position={Position.Left}
        className="!h-2.5 !w-2.5 !border-2 !border-white dark:!border-gray-900 !bg-gray-400 dark:!bg-gray-500"
      />

      <div className={`flex shrink-0 items-center justify-center rounded-lg ${visual.chipClass} ${visual.emphasize ? 'h-10 w-10' : 'h-8 w-8'}`}>
        <Icon className={`${visual.iconClass} ${visual.emphasize ? 'h-5 w-5' : 'h-4 w-4'}`} aria-hidden="true" />
      </div>

      <div className="min-w-0 flex-1">
        <p
          className={`truncate text-gray-900 dark:text-gray-100 ${visual.emphasize ? 'text-sm font-semibold' : 'text-sm font-medium'}`}
          title={data.label}
        >
          {data.label}
        </p>
        <p className={`truncate text-xs ${visual.captionClass}`}>{visual.typeCaption}</p>
        {data.detail && (
          <p className="truncate text-[11px] text-gray-600 dark:text-gray-300" title={data.detail}>
            {data.detail}
          </p>
        )}
      </div>

      {isCollapsibleAgent && (
        <button
          type="button"
          onClick={data.onToggleCollapse}
          aria-label={data.collapsed ? `Expand ${data.label}` : `Collapse ${data.label}`}
          aria-expanded={!data.collapsed}
          className="nodrag nopan shrink-0 rounded-md p-1 text-indigo-500 hover:bg-indigo-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 dark:text-indigo-300 dark:hover:bg-indigo-900/40 dark:focus-visible:ring-offset-gray-900"
        >
          {data.collapsed ? (
            <ChevronRight className="h-4 w-4" aria-hidden="true" />
          ) : (
            <ChevronDown className="h-4 w-4" aria-hidden="true" />
          )}
        </button>
      )}

      <Handle
        type="source"
        position={Position.Right}
        className="!h-2.5 !w-2.5 !border-2 !border-white dark:!border-gray-900 !bg-gray-400 dark:!bg-gray-500"
      />
    </div>
  );
}

/**
 * One thin wrapper component per `GraphNodeKind` so `nodeTypes` registers a
 * genuinely distinct React component per entity kind (agent/service/silo/
 * embedding/skill/mcp/parser), even though they all delegate to the shared
 * `EntityNodeCard` renderer above.
 */
export function AgentNode({ data }: NodeProps<AppFlowNode>) {
  return <EntityNodeCard kind="agent" data={data} />;
}

export function ServiceNode({ data }: NodeProps<AppFlowNode>) {
  return <EntityNodeCard kind="service" data={data} />;
}

export function SiloNode({ data }: NodeProps<AppFlowNode>) {
  return <EntityNodeCard kind="silo" data={data} />;
}

export function EmbeddingNode({ data }: NodeProps<AppFlowNode>) {
  return <EntityNodeCard kind="embedding" data={data} />;
}

export function SkillNode({ data }: NodeProps<AppFlowNode>) {
  return <EntityNodeCard kind="skill" data={data} />;
}

export function McpNode({ data }: NodeProps<AppFlowNode>) {
  return <EntityNodeCard kind="mcp" data={data} />;
}

export function ParserNode({ data }: NodeProps<AppFlowNode>) {
  return <EntityNodeCard kind="parser" data={data} />;
}

/** React Flow `nodeTypes` map - one entry per `GraphNodeKind`. */
export const APP_GRAPH_NODE_TYPES: Record<GraphNodeKind, ComponentType<NodeProps<AppFlowNode>>> = {
  agent: AgentNode,
  service: ServiceNode,
  silo: SiloNode,
  embedding: EmbeddingNode,
  skill: SkillNode,
  mcp: McpNode,
  parser: ParserNode,
};
