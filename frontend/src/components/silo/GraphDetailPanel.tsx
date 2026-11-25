import React, { useEffect, useState } from 'react';
import { Expand, Crosshair, Loader2, X } from 'lucide-react';
import { apiService } from '../../services/api';
import { openSourcePdf } from '../../utils/openSourcePdf';
import { colorForType } from '../../utils/entityTypeColors';
import { neighborsOf, splitSourceIds } from '../../utils/siloGraphUtils';
import type { GraphEdge, GraphNode, GraphSourceDoc } from '../../utils/siloGraphUtils';

interface GraphDetailPanelProps {
  appId: number;
  siloId: number;
  selectedNode: GraphNode | null;
  selectedEdge: GraphEdge | null;
  edges: GraphEdge[];
  atCap: boolean;
  onClose: () => void;
  onSelectNode: (nodeId: string) => void;
  /** Returns # of added nodes, 0 = none added, null = the fetch failed. */
  onExpandNeighbors: (nodeId: string) => Promise<number | null>; // returns # of added nodes
  onCenter: (nodeId: string, depth: number) => void;
}

const DEPTH_OPTIONS = [1, 2, 3];

/**
 * Detail panel of the silo graph explorer: full properties of the selected
 * node (or the selected relationship), its loaded connections, its source
 * documents (clickable → PDF at page), and the expand / center-here controls.
 */
const GraphDetailPanel: React.FC<GraphDetailPanelProps> = ({
  appId, siloId, selectedNode, selectedEdge, edges, atCap,
  onClose, onSelectNode, onExpandNeighbors, onCenter,
}) => {
  const [sources, setSources] = useState<GraphSourceDoc[] | null>(null);
  const [sourcesLoading, setSourcesLoading] = useState(false);
  const [expandState, setExpandState] = useState<'idle' | 'busy' | 'no-new' | 'at-cap'>('idle');
  const [centerDepth, setCenterDepth] = useState(2);

  const sourceIds = selectedNode
    ? splitSourceIds(selectedNode.properties?.source_id as string | undefined)
    : splitSourceIds(selectedEdge?.properties?.source_id as string | undefined);

  // Sources are best-effort: a failed lookup just hides the chips.
  useEffect(() => {
    setSources(null);
    setExpandState('idle');
    if (sourceIds.length === 0) return;
    setSourcesLoading(true);
    let cancelled = false;
    apiService
      .getSiloGraphSources(appId, siloId, sourceIds)
      .then((data: { sources: GraphSourceDoc[] }) => {
        if (!cancelled) setSources(data.sources);
      })
      .catch(() => {
        /* best-effort */
      })
      .finally(() => {
        if (!cancelled) setSourcesLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedNode?.id, selectedEdge?.id, appId, siloId, sourceIds.join(',')]);

  if (!selectedNode && !selectedEdge) return null;

  const isNode = selectedNode !== null;
  const relation = isNode ? null : selectedEdge!;
  const connections = isNode ? neighborsOf(selectedNode!.id, edges) : [];

  const expand = async () => {
    if (atCap) {
      setExpandState('at-cap');
      return;
    }
    setExpandState('busy');
    const added = await onExpandNeighbors(selectedNode!.id);
    // null = the fetch itself failed (already surfaced by the parent view)
    setExpandState(added == null ? 'idle' : added > 0 ? 'idle' : 'no-new');
  };

  return (
    <div className="absolute top-3 right-3 bg-white dark:bg-gray-800 shadow-md border border-gray-200 dark:border-gray-700 rounded-lg p-3 text-xs w-[320px] max-h-[85%] overflow-y-auto space-y-2 z-10">
      <div className="flex items-start justify-between gap-2">
        {isNode ? (
          <span className="flex items-center gap-1.5 min-w-0">
            <span
              className="w-2.5 h-2.5 rounded-full shrink-0"
              style={{ background: colorForType(selectedNode!.entity_type) }}
              aria-hidden="true"
            />
            <span className="font-semibold text-gray-800 dark:text-gray-100 break-all" title={selectedNode!.id}>
              {selectedNode!.id}
            </span>
          </span>
        ) : (
          <span className="font-semibold text-gray-800 dark:text-gray-100 break-all">
            {relation!.type}
          </span>
        )}
        <button onClick={onClose} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 shrink-0 mt-0.5" aria-label="Close details">
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Properties (full list, scrollable panel) */}
      {(() => {
        const props = isNode
          ? selectedNode!.properties
          : { type: relation!.type, ...relation!.properties };
        const entries = Object.entries(props).filter(([k]) => k !== 'workspace');
        if (entries.length === 0) return null;
        return (
          <div className="border-t border-gray-100 dark:border-gray-700 pt-2 space-y-1">
            {entries.map(([k, v]) =>
              v == null || String(v) === '' ? null : (
                <div key={k} className="flex gap-1">
                  <span className="text-gray-500 dark:text-gray-400 shrink-0">{k}:</span>
                  <span className="text-gray-800 dark:text-gray-200 break-all whitespace-pre-wrap">{String(v)}</span>
                </div>
              ),
            )}
          </div>
        );
      })()}

      {/* Connections (loaded neighbors) */}
      {isNode && connections.length > 0 && (
        <div className="border-t border-gray-100 dark:border-gray-700 pt-2">
          <p className="text-gray-500 dark:text-gray-400 mb-1">
            Connections ({connections.length})
          </p>
          <div className="flex flex-wrap gap-1">
            {connections.map((c) => (
              <button
                key={c.relatedId}
                onClick={() => onSelectNode(c.relatedId)}
                title={c.type}
                className="max-w-full truncate rounded bg-indigo-50 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 px-1.5 py-0.5 hover:bg-indigo-100 dark:hover:bg-indigo-900/60"
              >
                {c.relatedId}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Source documents */}
      {(sourcesLoading || (sources && sources.length > 0)) && (
        <div className="border-t border-gray-100 dark:border-gray-700 pt-2">
          <p className="text-gray-500 dark:text-gray-400 mb-1">Source documents</p>
          {sourcesLoading ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin text-gray-400" />
          ) : (
            <div className="flex flex-wrap gap-1">
              {(sources ?? []).map((s) => {
                const label = s.file_name
                  ? s.page != null
                    ? `${s.file_name} (p. ${s.page})`
                    : s.file_name
                  : 'Unknown source';
                const openable = s.resource_id != null && s.page != null;
                return openable ? (
                  <button
                    key={s.chunk_id}
                    onClick={() => openSourcePdf(appId, siloId, s.resource_id!, s.page!)}
                    title={`Open ${s.file_name} at page ${s.page}`}
                    className="max-w-full truncate rounded-full bg-green-50 dark:bg-green-900/40 text-green-700 dark:text-green-300 px-1.5 py-0.5 hover:bg-green-100 dark:hover:bg-green-900/60"
                  >
                    {label}
                  </button>
                ) : (
                  <span
                    key={s.chunk_id}
                    title="Source file not available"
                    className="max-w-full truncate rounded-full bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-300 px-1.5 py-0.5"
                  >
                    {label}
                  </span>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Expand / center-here (nodes only) */}
      {isNode && (
        <div className="border-t border-gray-100 dark:border-gray-700 pt-2">
          <button
            onClick={expand}
            disabled={expandState === 'busy'}
            className="inline-flex items-center gap-1.5 text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-200 font-medium disabled:opacity-50"
          >
            {expandState === 'busy'
              ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
              : <Expand className="w-3.5 h-3.5" />}
            Expand neighbors
          </button>
          {expandState === 'no-new' && (
            <p className="text-gray-400 mt-1">No new neighbors to add.</p>
          )}
          {expandState === 'at-cap' && (
            <p className="text-amber-600 dark:text-amber-400 mt-1">
              Canvas is at the node cap — raise the Nodes slider first.
            </p>
          )}

          <div className="flex items-center gap-1.5 mt-2">
            <Crosshair className="w-3.5 h-3.5 text-indigo-500 shrink-0" />
            <span className="text-gray-500 dark:text-gray-400">Center graph:</span>
            {DEPTH_OPTIONS.map((d) => (
              <button
                key={d}
                onClick={() => setCenterDepth(d)}
                title={d === 1 ? 'Direct neighbors only' : `Up to ${d - 1} hop${d - 1 > 1 ? 's' : ''} beyond`}
                className={`rounded px-1.5 py-0.5 ${centerDepth === d
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'}`}
              >
                {d}
              </button>
            ))}
            <button
              onClick={() => onCenter(selectedNode!.id, centerDepth)}
              className="ml-auto rounded bg-indigo-600 hover:bg-indigo-700 text-white px-2 py-0.5 font-medium"
            >
              Go
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default GraphDetailPanel;
