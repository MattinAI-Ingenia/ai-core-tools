import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { AlertTriangle, Info, Loader2, RefreshCw, Search } from 'lucide-react';
import { InteractiveNvlWrapper } from '@neo4j-nvl/react';
import type { MouseEventCallbacks } from '@neo4j-nvl/react';
import type { Node as NvlNode, NvlOptions, Relationship as NvlRelationship } from '@neo4j-nvl/base';
import { apiService } from '../../services/api';
import { HIGHLIGHT_BLUE, NEUTRAL_COLOR, PARTIAL_COLOR, PRIMARY_COLOR, colorForType, typeRingIcon } from '../../utils/entityTypeColors';
import type { GraphNode, GraphEdge, SiloGraphData } from '../../utils/siloGraphUtils';
import { mergeGraphData } from '../../utils/siloGraphUtils';
import GraphDetailPanel from './GraphDetailPanel';

interface SiloGraphViewProps {
    appId: number;
    siloId: number;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function toNvlNodes(nodes: GraphNode[], typeColoring: boolean, chosenIds: Set<string>, neighborIds: Set<string>): NvlNode[] {
    return nodes.map(n => ({
        id: n.id,
        captions: [{ value: n.id }],
        color: typeColoring ? colorForType(n.entity_type) : NEUTRAL_COLOR,
        // Chosen element → yellow NVL selection ring; adjacent ones → blue ring
        // drawn as an overlayIcon (NVL only supports ONE selected color).
        selected: chosenIds.has(n.id),
        overlayIcon: neighborIds.has(n.id) ? typeRingIcon(HIGHLIGHT_BLUE) : undefined,
    }));
}

function toNvlRels(edges: GraphEdge[], nodes: GraphNode[], chosenIds: Set<string>, neighborIds: Set<string>): NvlRelationship[] {
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
                color: neighborIds.has(e.id) ? HIGHLIGHT_BLUE : '#94a3b8',
                selected: chosenIds.has(e.id),
            };
        });
}

// ── Component ────────────────────────────────────────────────────────────────

/**
 * Knowledge graph explorer for a LightRAG silo, powered by @neo4j-nvl/react.
 * Falls back gracefully when Neo4j is unreachable (503) or silo is non-LightRAG (409).
 */
const SiloGraphView: React.FC<SiloGraphViewProps> = ({ appId, siloId }) => {
    const { t } = useTranslation('common');
    const [graphData, setGraphData] = useState<SiloGraphData | null>(null);
    const [loading, setLoading] = useState(true);
    // `error` holds an i18n key (translated at render) so a language switch updates it.
    const [error, setError] = useState<string | null>(null);
    const [notice, setNotice] = useState<{ count: number; node: string } | null>(null);
    const [search, setSearch] = useState('');
    const DEFAULT_MAX_NODES = 50;
    // Slider ceiling — the renderer chokes well beyond this on a large silo,
    // so there is no "show me everything" option.
    const SLIDER_MAX_NODES = 100;
    const [sliderValue, setSliderValue] = useState(DEFAULT_MAX_NODES);
    const [sliderMax, setSliderMax] = useState(DEFAULT_MAX_NODES);
    const maxNodesRef = useRef(DEFAULT_MAX_NODES);
    // The centered node (center-here view) — slider/refresh re-issue the
    // centering instead of falling back to the whole-silo graph.
    const centeredRef = useRef<{ id: string; depth: number } | null>(null);
    const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
    const [selectedEdge, setSelectedEdge] = useState<GraphEdge | null>(null);
    const [activeTypes, setActiveTypes] = useState<Set<string>>(new Set());
    const [typeColoring, setTypeColoring] = useState(true);

    const fetchGraph = useCallback(
        async (searchQuery = '', types: Set<string> | null = null) => {
            setLoading(true);
            setError(null);
            setSelectedNode(null);
            setSelectedEdge(null);
            try {
                centeredRef.current = null;
                setNotice(null);
                const data = (await apiService.getSiloGraph(appId, siloId, {
                    maxNodes: maxNodesRef.current,
                    search: searchQuery || undefined,
                    entityTypes: types && types.size > 0 ? Array.from(types) : undefined,
                })) as SiloGraphData;
                setGraphData({ ...data, max_nodes: maxNodesRef.current });
                // Slider tops out at SLIDER_MAX_NODES regardless of how big the graph really is.
                if (data.total_nodes > 0) {
                    setSliderMax(Math.min(data.total_nodes, SLIDER_MAX_NODES));
                }
            } catch (err: unknown) {
                const status = (err as { status?: number })?.status;
                if (status === 409) {
                    setError('lightrag.siloGraphView.errors.notLightrag');
                } else if (status === 503) {
                    setError('lightrag.siloGraphView.errors.neo4jUnavailable');
                } else {
                    setError('lightrag.siloGraphView.errors.loadFailed');
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

    const expandNeighbors = useCallback(
        async (nodeId: string): Promise<number | null> => {
            const before = new Set((graphData?.nodes ?? []).map(n => n.id));
            try {
                const data = (await apiService.getSiloGraphNeighbors(appId, siloId, {
                    nodeId,
                    depth: 1,
                    maxNodes: maxNodesRef.current,
                })) as SiloGraphData;
                const merged = mergeGraphData(
                    graphData ?? { nodes: [], edges: [], entity_types: [], node_count: 0, edge_count: 0, total_nodes: 0, total_edges: 0, truncated: false, max_nodes: maxNodesRef.current },
                    data,
                );
                const added = merged.nodes.length - before.size;
                setGraphData(merged);
                return added;
            } catch (err: unknown) {
                const status = (err as { status?: number })?.status;
                if (status === 503) {
                    setError('lightrag.siloGraphView.errors.neo4jUnavailable');
                } else if (status === 404) {
                    setError('lightrag.siloGraphView.errors.nodeGone');
                } else {
                    setError('lightrag.siloGraphView.errors.neighborsFailed');
                }
                return null;
            }
        },
        [appId, siloId, graphData],
    );

    const centerOnNode = useCallback(
        async (nodeId: string, depth: number) => {
            setSelectedNode(null);
            setSelectedEdge(null);
            try {
                const data = (await apiService.getSiloGraphNeighbors(appId, siloId, {
                    nodeId,
                    depth,
                    maxNodes: maxNodesRef.current,
                })) as SiloGraphData;
                centeredRef.current = { id: nodeId, depth };
                setGraphData({ ...data, max_nodes: maxNodesRef.current });
                // A truncated result means the node cap dominated: raising the
                // depth can't add nodes, so say so instead of silently
                // returning an identical view.
                setNotice(data.truncated ? { count: data.nodes.length, node: nodeId } : null);
            } catch (err: unknown) {
                const status = (err as { status?: number })?.status;
                if (status === 404) {
                    setError('lightrag.siloGraphView.errors.nodeGone');
                } else if (status === 503) {
                    setError('lightrag.siloGraphView.errors.neo4jUnavailable');
                } else {
                    setError('lightrag.siloGraphView.errors.loadFailed');
                }
            }
        },
        [appId, siloId],
    );

    const toggleType = (type: string) => {
        const next = new Set(activeTypes);
        if (next.has(type)) {
            next.delete(type);
        } else {
            next.add(type);
        }
        setActiveTypes(next);
        fetchGraph(search, next);
    };

    // Slider / refresh: keep the centered view if one is active — re-issue the
    // centering with the new node budget instead of the whole-silo graph.
    const reloadView = () => {
        const centered = centeredRef.current;
        if (centered) {
            centerOnNode(centered.id, centered.depth);
        } else {
            fetchGraph(search, activeTypes);
        }
    };

    // NVL interaction callbacks
    const mouseEventCallbacks: MouseEventCallbacks = {
        onNodeClick: (node: NvlNode) => {
            const original = graphData?.nodes.find(n => n.id === node.id) ?? null;
            setSelectedEdge(null);
            setSelectedNode(prev => (prev?.id === node.id ? null : original));
        },
        onRelationshipClick: (rel: NvlRelationship) => {
            const original = graphData?.edges.find(e => e.id === rel.id) ?? null;
            setSelectedNode(null);
            setSelectedEdge(prev => (prev?.id === rel.id ? null : original));
        },
        onCanvasClick: () => {
            setSelectedNode(null);
            setSelectedEdge(null);
        },
        onPan: true,
        onZoom: true,
        onDrag: true,
    };

    const nvlOptions: NvlOptions = {
        layout: 'forceDirected',
        allowDynamicMinZoom: true,
        // Selection ring AROUND the element (fill untouched) — bright yellow so
        // the clicked node/edge stands out from the type palette.
        styling: {
            selectedBorderColor: PRIMARY_COLOR,
            selectedInnerBorderColor: PRIMARY_COLOR,
        },
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64 text-gray-500">
                <Loader2 className="w-6 h-6 animate-spin mr-2" />
                {t('lightrag.siloGraphView.loading')}
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex items-center gap-3 p-4 bg-yellow-50 border border-yellow-200 rounded text-yellow-800">
                <AlertTriangle className="w-5 h-5 shrink-0" />
                <span className="text-sm">{t(error)}</span>
            </div>
        );
    }

    const nodes = graphData?.nodes ?? [];
    const edges = graphData?.edges ?? [];
    // Highlight: clicked node → yellow ring on it, blue rings on its neighbors,
    // blue incident edges; clicked edge → yellow on it, blue on its endpoints.
    const chosenNodes = new Set<string>();
    const chosenEdges = new Set<string>();
    const neighborNodes = new Set<string>();
    const neighborEdges = new Set<string>();
    if (selectedNode) {
        chosenNodes.add(selectedNode.id);
        for (const e of edges) {
            if (e.source === selectedNode.id || e.target === selectedNode.id) {
                neighborNodes.add(e.source === selectedNode.id ? e.target : e.source);
                neighborEdges.add(e.id);
            }
        }
    } else if (selectedEdge) {
        chosenEdges.add(selectedEdge.id);
        neighborNodes.add(selectedEdge.source);
        neighborNodes.add(selectedEdge.target);
    }
    const nvlNodes = toNvlNodes(nodes, typeColoring, chosenNodes, neighborNodes);
    const nvlRels = toNvlRels(edges, nodes, chosenEdges, neighborEdges);
    const isEmpty = nvlNodes.length === 0;
    const entityTypes = graphData?.entity_types ?? [];
    const atCap = nodes.length >= Math.min(graphData?.max_nodes ?? maxNodesRef.current, sliderMax);

    return (
        <div className="flex flex-col" style={{ height: 'calc(100vh - 200px)', minHeight: 500 }}>
            {notice && (
                <div className="flex items-center gap-2 p-2 bg-indigo-50 dark:bg-indigo-900/30 border-b border-indigo-100 dark:border-indigo-900 text-indigo-700 dark:text-indigo-300 text-xs shrink-0">
                    <Info className="w-4 h-4 shrink-0" />
                    <span>{t('lightrag.siloGraphView.nodeCapNotice', { count: notice.count, node: notice.node })}</span>
                </div>
            )}
            {/* Toolbar */}
            <div className="flex items-center gap-2 flex-wrap p-3 shrink-0">
                <div className="relative flex-1 min-w-[200px]">
                    <Search className="absolute left-2 top-2 w-4 h-4 text-gray-400" />
                    <input
                        type="text"
                        value={search}
                        onChange={e => setSearch(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && fetchGraph(search, activeTypes)}
                        placeholder={t('lightrag.siloGraphView.filterPlaceholder')}
                        className="pl-8 pr-3 py-1.5 text-sm border border-gray-300 rounded w-full focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                </div>
                <label className="flex items-center gap-2 text-xs text-gray-500 min-w-[180px]">
                    <span className="shrink-0">{t('lightrag.siloGraphView.nodes')}</span>
                    <input
                        type="range"
                        min={5}
                        max={sliderMax}
                        step={5}
                        value={sliderValue}
                        onChange={e => setSliderValue(Number(e.target.value))}
                        onMouseUp={e => {
                            maxNodesRef.current = Number((e.target as HTMLInputElement).value);
                            reloadView();
                        }}
                        onPointerUp={e => {
                            maxNodesRef.current = Number((e.target as HTMLInputElement).value);
                            reloadView();
                        }}
                        className="flex-1 accent-indigo-600"
                    />
                    <span className="w-10 text-right tabular-nums">{sliderValue}</span>
                </label>
                <label className="flex items-center gap-1.5 text-xs text-gray-600 dark:text-gray-300" title={t('lightrag.siloGraphView.colorByTypeHint')}>
                    <input
                        type="checkbox"
                        checked={typeColoring}
                        onChange={e => setTypeColoring(e.target.checked)}
                        className="accent-indigo-600"
                    />
                    {t('lightrag.siloGraphView.colorByType')}
                </label>
                <button
                    onClick={() => reloadView()}
                    className="p-2 text-gray-500 hover:text-indigo-600 border border-gray-200 rounded"
                    title={t('lightrag.siloGraphView.refresh')}
                >
                    <RefreshCw className="w-4 h-4" />
                </button>
                <span className="text-xs text-gray-400">
                    {t('lightrag.siloGraphView.showing', {
                        nodes: graphData?.node_count ?? 0,
                        totalNodes: graphData?.total_nodes ?? 0,
                        edges: graphData?.edge_count ?? 0,
                        totalEdges: graphData?.total_edges ?? 0,
                    })}
                    {graphData?.truncated && ` · ${t('lightrag.siloGraphView.truncatedSuffix')}`}
                </span>
            </div>

            {/* Legend = type filter */}
            {entityTypes.length > 0 && (
                <div className="flex items-center gap-1.5 flex-wrap px-3 pb-2 shrink-0">
                    {entityTypes.map(({ type, count }) => {
                        const active = activeTypes.has(type);
                        // "Other" groups untyped + rare types — render it in the
                        // untyped gray, not a palette color.
                        const chipColor = type === 'Other' ? PARTIAL_COLOR : colorForType(type);
                        return (
                            <button
                                key={type}
                                onClick={() => toggleType(type)}
                                title={active
                                    ? t('lightrag.siloGraphView.filteringByTypes', { count: activeTypes.size })
                                    : t('lightrag.siloGraphView.filterBy', { type })}
                                className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs border transition-colors ${active
                                    ? 'border-transparent text-white'
                                    : 'border-gray-200 dark:border-gray-600 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'}`}
                                style={active ? { background: chipColor } : undefined}
                            >
                                <span
                                    className="w-2.5 h-2.5 rounded-full shrink-0"
                                    style={{ background: chipColor }}
                                    aria-hidden="true"
                                />
                                {type} · {count}
                            </button>
                        );
                    })}
                    {activeTypes.size > 0 && (
                        <button
                            onClick={() => {
                                setActiveTypes(new Set());
                                fetchGraph(search, new Set());
                            }}
                            className="text-xs text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-200"
                        >
                            {t('lightrag.siloGraphView.all')}
                        </button>
                    )}
                </div>
            )}

            {graphData?.truncated && (
                <div className="text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded px-3 py-1.5">
                    {t('lightrag.siloGraphView.truncatedBanner', { nodes: graphData.node_count, edges: graphData.edge_count })}
                </div>
            )}

            {/* Graph + side panel */}
            <div className="relative flex-1 border-t border-gray-200 overflow-hidden">
                {isEmpty ? (
                    <div className="flex items-center justify-center h-full text-gray-400 text-sm">
                        {search
                            ? t('lightrag.siloGraphView.noResults', { search })
                            : t('lightrag.siloGraphView.noGraphData')}
                    </div>
                ) : (
                    <InteractiveNvlWrapper
                        nodes={nvlNodes}
                        rels={nvlRels}
                        nvlOptions={nvlOptions}
                        mouseEventCallbacks={mouseEventCallbacks}
                        style={{ width: '100%', height: '100%' }}
                    />
                )}

                <GraphDetailPanel
                    appId={appId}
                    siloId={siloId}
                    selectedNode={selectedNode}
                    selectedEdge={selectedEdge}
                    edges={edges}
                    nodes={graphData?.nodes ?? []}
                    atCap={atCap}
                    onClose={() => {
                        setSelectedNode(null);
                        setSelectedEdge(null);
                    }}
                    onSelectNode={(nodeId) => {
                        const next = graphData?.nodes.find(n => n.id === nodeId) ?? null;
                        setSelectedNode(next);
                        setSelectedEdge(null);
                    }}
                    onExpandNeighbors={expandNeighbors}
                    onCenter={centerOnNode}
                />
            </div>
        </div>
    );
};

export default SiloGraphView;
