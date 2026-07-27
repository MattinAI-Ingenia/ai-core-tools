import { ReactFlow, MiniMap, Controls, Background, BackgroundVariant, type OnNodesChange } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { AlertTriangle, Loader2, RefreshCw, Workflow } from 'lucide-react';
import { APP_GRAPH_NODE_TYPES, type AppFlowNode } from './EntityNodeCard';
import type { AppFlowEdge } from './graphAdapter';

export interface GraphFlowViewProps {
  readonly nodes: readonly AppFlowNode[];
  readonly edges: readonly AppFlowEdge[];
  readonly loading: boolean;
  readonly error: string | null;
  readonly onNodesChange?: OnNodesChange<AppFlowNode>;
  readonly onRetry?: () => void;
  readonly className?: string;
}

/**
 * Pure, presentational read-only React Flow canvas for the app resource
 * graph. Takes already-adapted `@xyflow/react` nodes/edges plus the loading
 * and error state from `useAppGraph` - no data fetching of its own, so it
 * can be rendered/tested with plain fixture props.
 */
export function GraphFlowView({
  nodes,
  edges,
  loading,
  error,
  onNodesChange,
  onRetry,
  className = '',
}: GraphFlowViewProps) {
  if (loading) {
    return (
      <div
        className={`flex h-[70vh] w-full min-h-[400px] flex-col items-center justify-center gap-3 rounded-xl border border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-800/50 ${className}`}
        role="status"
        aria-live="polite"
      >
        <Loader2 className="h-8 w-8 animate-spin motion-reduce:animate-none text-indigo-500 dark:text-indigo-400" aria-hidden="true" />
        <p className="text-sm text-gray-500 dark:text-gray-400">Loading app graph…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div
        className={`flex h-[70vh] w-full min-h-[400px] flex-col items-center justify-center gap-3 rounded-xl border border-red-200 bg-red-50 px-6 text-center dark:border-red-800/50 dark:bg-red-900/20 ${className}`}
        role="alert"
        aria-live="assertive"
      >
        <AlertTriangle className="h-8 w-8 text-red-500 dark:text-red-400" aria-hidden="true" />
        <p className="max-w-md text-sm text-red-700 dark:text-red-300">{error}</p>
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="inline-flex items-center gap-1.5 rounded-lg border border-red-300 bg-white px-3 py-1.5 text-sm font-medium text-red-700 transition-colors hover:bg-red-50 dark:border-red-700 dark:bg-gray-800 dark:text-red-300 dark:hover:bg-red-900/30"
          >
            <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
            Try again
          </button>
        )}
      </div>
    );
  }

  if (nodes.length === 0) {
    return (
      <div
        className={`flex h-[70vh] w-full min-h-[400px] flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-gray-300 bg-gray-50 px-6 text-center dark:border-gray-700 dark:bg-gray-800/50 ${className}`}
        role="status"
        aria-live="polite"
      >
        <Workflow className="h-8 w-8 text-gray-400 dark:text-gray-500" aria-hidden="true" />
        <p className="text-sm text-gray-500 dark:text-gray-400">
          This app has no agents or resources to visualize yet.
        </p>
      </div>
    );
  }

  return (
    <div
      className={`h-[70vh] w-full min-h-[400px] overflow-hidden rounded-xl border border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-900 ${className}`}
    >
      {/* Visually-hidden heading gives the canvas an accessible name without
          hiding the interactive Controls/MiniMap buttons behind role="img". */}
      <h2 className="sr-only">App resource graph</h2>
      <ReactFlow<AppFlowNode, AppFlowEdge>
        nodes={nodes as AppFlowNode[]}
        edges={edges as AppFlowEdge[]}
        onNodesChange={onNodesChange}
        nodeTypes={APP_GRAPH_NODE_TYPES}
        nodesConnectable={false}
        elementsSelectable
        nodesDraggable
        nodesFocusable={false}
        edgesReconnectable={false}
        edgesFocusable={false}
        deleteKeyCode={null}
        proOptions={{ hideAttribution: true }}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.2}
        maxZoom={1.5}
      >
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} className="dark:opacity-30" />
        <MiniMap
          pannable
          zoomable
          className="!bg-white dark:!bg-gray-800 [&_.react-flow__minimap-mask]:!fill-gray-300/60 dark:[&_.react-flow__minimap-mask]:!fill-gray-600/40"
        />
        <Controls showInteractive={false} className="[&_button]:!bg-white [&_button]:dark:!bg-gray-800 [&_button]:dark:!fill-gray-300 [&_button]:dark:!border-gray-700" />
      </ReactFlow>
    </div>
  );
}
