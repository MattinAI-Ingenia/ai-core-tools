import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AlertTriangle, Loader2, RefreshCw, Search, X } from 'lucide-react';
import { InteractiveNvlWrapper } from '@neo4j-nvl/react';
import type { MouseEventCallbacks } from '@neo4j-nvl/react';
import type { Node as NvlNode, NvlOptions, Relationship as NvlRelationship } from '@neo4j-nvl/base';
import { apiService } from '../../services/api';

// ── API types ────────────────────────────────────────────────────────────────

interface GraphNode {
    id: string;
    labels: string[];
    properties: Record<string, unknown>;
}

interface GraphEdge {
    id: string;
    source: string;
    target: string;
    type: string;
    properties: Record<string, unknown>;
}

interface SiloGraphData {
    nodes: GraphNode[];
    edges: GraphEdge[];
    node_count: number;
    edge_count: number;
    total_nodes: number;
    truncated: boolean;
}

interface SiloGraphViewProps {
    appId: number;
    siloId: number;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function toNvlNodes(nodes: GraphNode[]): NvlNode[] {
    return nodes.map(n => ({
        id: n.id,
        captions: [{ value: n.id }],
        color: '#6366f1',
    }));
}

function toNvlRels(edges: GraphEdge[], nodes: GraphNode[]): NvlRelationship[] {
    const nodeIds = new Set(nodes.map(n => n.id));
    return edges
        .filter(e => nodeIds.has(e.source) && nodeIds.has(e.target))
        .map(e => {
            const props = e.properties as Record<string, unknown>;
            const label = String(
                props['keywords'] ?? props['description'] ?? e.type ?? ''
            ).slice(0, 40);
            return {
                id: e.id,
                from: e.source,
                to: e.target,
                captions: label ? [{ value: label }] : [],
                color: '#94a3b8',
            };
        });
}

// ── Component ────────────────────────────────────────────────────────────────

/**
 * Knowledge graph viewer for a LightRAG silo, powered by @neo4j-nvl/react.
 * Falls back gracefully when Neo4j is unreachable (503) or silo is non-LightRAG (409).
 */
const SiloGraphView: React.FC<SiloGraphViewProps> = ({ appId, siloId }) => {
    const [graphData, setGraphData] = useState<SiloGraphData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [search, setSearch] = useState('');
    const [sliderValue, setSliderValue] = useState(500);
    const [sliderMax, setSliderMax] = useState(500);
    const maxNodesRef = useRef(500);
    const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
    const lastMouse = useRef({ x: 0, y: 0 });
    const isFirstLoad = useRef(true);

    const fetchGraph = useCallback(
        async (searchQuery = '') => {
            setLoading(true);
            setError(null);
            setSelectedNode(null);
            try {
                const data = (await apiService.getSiloGraph(appId, siloId, {
                    maxNodes: maxNodesRef.current,
                    maxDepth: 2,
                    search: searchQuery || undefined,
                })) as SiloGraphData;
                setGraphData(data);
                if (data.total_nodes > 0) {
                    setSliderMax(data.total_nodes);
                    if (isFirstLoad.current) {
                        isFirstLoad.current = false;
                        maxNodesRef.current = data.total_nodes;
                        setSliderValue(data.total_nodes);
                    }
                }
            } catch (err: unknown) {
                const status = (err as { status?: number })?.status;
                if (status === 409) {
                    setError('This silo does not use LightRAG and has no knowledge graph.');
                } else if (status === 503) {
                    setError('Neo4j is not available. Check your configuration.');
                } else {
                    setError('Failed to load graph data.');
                }
            } finally {
                setLoading(false);
            }
        },
        [appId, siloId],
    );

    useEffect(() => {
        fetchGraph();
    }, [fetchGraph]);

    // NVL interaction callbacks
    const mouseEventCallbacks: MouseEventCallbacks = {
        onNodeClick: (node: NvlNode) => {
            const original = graphData?.nodes.find(n => n.id === node.id) ?? null;
            setSelectedNode(prev => (prev?.id === node.id ? null : original));
        },
        onCanvasClick: () => setSelectedNode(null),
        onPan: true,
        onZoom: true,
        onDrag: true,
    };

    const nvlOptions: NvlOptions = {
        layout: 'forceDirected',
        allowDynamicMinZoom: true,
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64 text-gray-500">
                <Loader2 className="w-6 h-6 animate-spin mr-2" />
                Loading knowledge graph…
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex items-center gap-3 p-4 bg-yellow-50 border border-yellow-200 rounded text-yellow-800">
                <AlertTriangle className="w-5 h-5 shrink-0" />
                <span className="text-sm">{error}</span>
            </div>
        );
    }

    if (!graphData || graphData.nodes.length === 0) {
        return (
            <div className="flex items-center justify-center h-48 text-gray-400 text-sm">
                No graph data indexed yet.
            </div>
        );
    }

    const nvlNodes = toNvlNodes(graphData.nodes);
    const nvlRels = toNvlRels(graphData.edges, graphData.nodes);

    return (
        <div className="flex flex-col" style={{ height: 'calc(100vh - 200px)', minHeight: 500 }}>
            {/* Toolbar */}
            <div className="flex items-center gap-2 flex-wrap p-3 shrink-0">
                <div className="relative flex-1 min-w-[200px]">
                    <Search className="absolute left-2 top-2 w-4 h-4 text-gray-400" />
                    <input
                        type="text"
                        value={search}
                        onChange={e => setSearch(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && fetchGraph(search)}
                        placeholder="Filter entities…"
                        className="pl-8 pr-3 py-1.5 text-sm border border-gray-300 rounded w-full focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                </div>
                <label className="flex items-center gap-2 text-xs text-gray-500 min-w-[180px]">
                    <span className="shrink-0">Nodes</span>
                    <input
                        type="range"
                        min={5}
                        max={sliderMax}
                        step={5}
                        value={sliderValue}
                        onChange={e => setSliderValue(Number(e.target.value))}
                        onMouseUp={e => {
                            maxNodesRef.current = Number((e.target as HTMLInputElement).value);
                            fetchGraph(search);
                        }}
                        onPointerUp={e => {
                            maxNodesRef.current = Number((e.target as HTMLInputElement).value);
                            fetchGraph(search);
                        }}
                        className="flex-1 accent-indigo-600"
                    />
                    <span className="w-10 text-right tabular-nums">{sliderValue}</span>
                </label>
                <button
                    onClick={() => fetchGraph(search)}
                    className="p-2 text-gray-500 hover:text-indigo-600 border border-gray-200 rounded"
                    title="Refresh"
                >
                    <RefreshCw className="w-4 h-4" />
                </button>
                <span className="text-xs text-gray-400">
                    {graphData.node_count} nodes · {graphData.edge_count} edges
                    {graphData.truncated && ' · (truncated)'}
                </span>
            </div>

            {graphData.truncated && (
                <div className="text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded px-3 py-1.5">
                    Graph truncated to {graphData.node_count} nodes. Use search to filter.
                </div>
            )}

            {/* Graph + side panel */}
            <div className="relative flex-1 border-t border-gray-200 overflow-hidden">
                <InteractiveNvlWrapper
                    nodes={nvlNodes}
                    rels={nvlRels}
                    nvlOptions={nvlOptions}
                    mouseEventCallbacks={mouseEventCallbacks}
                    style={{ width: '100%', height: '100%' }}
                />

                {/* Node detail panel */}
                {selectedNode && (
                    <div className="absolute top-3 right-3 bg-white shadow-md border border-gray-200 rounded p-3 text-xs max-w-[260px] space-y-1.5 z-10">
                        <div className="flex items-start justify-between gap-2">
                            <span className="font-semibold text-gray-800 break-all" title={selectedNode.id}>
                                {selectedNode.id}
                            </span>
                            <button
                                onClick={() => setSelectedNode(null)}
                                className="text-gray-400 hover:text-gray-600 shrink-0 mt-0.5"
                            >
                                <X className="w-3 h-3" />
                            </button>
                        </div>
                        {selectedNode.labels.length > 0 && (
                            <div className="flex flex-wrap gap-1">
                                {selectedNode.labels.map(l => (
                                    <span key={l} className="bg-indigo-100 text-indigo-700 rounded px-1.5 py-0.5">
                                        {l}
                                    </span>
                                ))}
                            </div>
                        )}
                        {Object.keys(selectedNode.properties).length > 0 && (
                            <div className="border-t border-gray-100 pt-1.5 space-y-1">
                                {Object.entries(selectedNode.properties)
                                    .filter(([k]) => k !== 'workspace')
                                    .slice(0, 8)
                                    .map(([k, v]) => (
                                        <div key={k} className="flex gap-1">
                                            <span className="text-gray-500 shrink-0">{k}:</span>
                                            <span className="text-gray-800 break-all">
                                                {String(v).length > 60 ? String(v).slice(0, 58) + '…' : String(v)}
                                            </span>
                                        </div>
                                    ))}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default SiloGraphView;
