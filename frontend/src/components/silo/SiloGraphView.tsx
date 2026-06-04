import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Loader2, AlertTriangle, Search, ZoomIn, ZoomOut, RefreshCw } from 'lucide-react';
import { apiService } from '../../services/api';

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
    truncated: boolean;
}

interface SelectedNode {
    node: GraphNode;
    x: number;
    y: number;
}

interface SiloGraphViewProps {
    appId: number;
    siloId: number;
}

const NODE_RADIUS = 8;
const NODE_COLOR = '#6366f1';
const EDGE_COLOR = '#94a3b8';
const SELECTED_COLOR = '#f59e0b';
const BG_COLOR = '#f8fafc';

/**
 * Force-directed knowledge graph viewer for a LightRAG silo.
 *
 * Uses a lightweight canvas-based renderer (no external graph library required).
 * Falls back gracefully when Neo4j is unreachable (503) or silo is non-LightRAG (409).
 */
const SiloGraphView: React.FC<SiloGraphViewProps> = ({ appId, siloId }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [graphData, setGraphData] = useState<SiloGraphData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [search, setSearch] = useState('');
    const [selectedNode, setSelectedNode] = useState<SelectedNode | null>(null);
    const [scale, setScale] = useState(1);
    const [offset, setOffset] = useState({ x: 0, y: 0 });

    // Node positions (computed by simple force simulation)
    const positionsRef = useRef<Record<string, { x: number; y: number }>>({});
    const animFrameRef = useRef<number>(0);
    const isDragging = useRef(false);
    const lastMouse = useRef({ x: 0, y: 0 });

    const fetchGraph = useCallback(
        async (searchQuery = '') => {
            setLoading(true);
            setError(null);
            setSelectedNode(null);
            try {
                const data = await apiService.getSiloGraph(appId, siloId, {
                    maxNodes: 200,
                    maxDepth: 2,
                    search: searchQuery || undefined,
                });
                setGraphData(data as SiloGraphData);
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
        return () => cancelAnimationFrame(animFrameRef.current);
    }, [fetchGraph]);

    // Initialise node positions when graph data changes
    useEffect(() => {
        if (!graphData) return;
        const positions: Record<string, { x: number; y: number }> = {};
        const cx = 400;
        const cy = 300;
        graphData.nodes.forEach((node, i) => {
            const angle = (i / Math.max(graphData.nodes.length, 1)) * 2 * Math.PI;
            const r = Math.min(180, graphData.nodes.length * 10);
            positions[node.id] = {
                x: cx + r * Math.cos(angle) + (Math.random() - 0.5) * 20,
                y: cy + r * Math.sin(angle) + (Math.random() - 0.5) * 20,
            };
        });
        positionsRef.current = positions;
        runSimulation(graphData, positionsRef);
    }, [graphData]);

    // Force-directed simulation (simple Fruchterman-Reingold approximation)
    function runSimulation(
        data: SiloGraphData,
        posRef: React.MutableRefObject<Record<string, { x: number; y: number }>>,
    ) {
        let iter = 0;
        const MAX_ITER = 120;
        const W = 800;
        const H = 600;
        const k = Math.sqrt((W * H) / Math.max(data.nodes.length, 1));

        function tick() {
            if (iter >= MAX_ITER) {
                draw(data, posRef.current);
                return;
            }
            iter++;
            const temp = k * (1 - iter / MAX_ITER);
            const disp: Record<string, { x: number; y: number }> = {};
            data.nodes.forEach(n => (disp[n.id] = { x: 0, y: 0 }));

            // Repulsion
            data.nodes.forEach(u => {
                data.nodes.forEach(v => {
                    if (u.id === v.id) return;
                    const pu = posRef.current[u.id];
                    const pv = posRef.current[v.id];
                    if (!pu || !pv) return;
                    const dx = pu.x - pv.x;
                    const dy = pu.y - pv.y;
                    const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 0.01);
                    const force = (k * k) / dist;
                    disp[u.id].x += (dx / dist) * force;
                    disp[u.id].y += (dy / dist) * force;
                });
            });

            // Attraction
            data.edges.forEach(e => {
                const pu = posRef.current[e.source];
                const pv = posRef.current[e.target];
                if (!pu || !pv) return;
                const dx = pu.x - pv.x;
                const dy = pu.y - pv.y;
                const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 0.01);
                const force = (dist * dist) / k;
                disp[e.source].x -= (dx / dist) * force;
                disp[e.source].y -= (dy / dist) * force;
                disp[e.target].x += (dx / dist) * force;
                disp[e.target].y += (dy / dist) * force;
            });

            // Apply with temperature and bounds
            data.nodes.forEach(n => {
                const d = disp[n.id];
                const dlen = Math.max(Math.sqrt(d.x * d.x + d.y * d.y), 0.01);
                const pos = posRef.current[n.id];
                if (!pos) return;
                pos.x += (d.x / dlen) * Math.min(dlen, temp);
                pos.y += (d.y / dlen) * Math.min(dlen, temp);
                pos.x = Math.max(20, Math.min(W - 20, pos.x));
                pos.y = Math.max(20, Math.min(H - 20, pos.y));
            });

            draw(data, posRef.current);
            animFrameRef.current = requestAnimationFrame(tick);
        }

        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = requestAnimationFrame(tick);
    }

    function draw(
        data: SiloGraphData,
        positions: Record<string, { x: number; y: number }>,
    ) {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = BG_COLOR;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.save();
        ctx.translate(offset.x, offset.y);
        ctx.scale(scale, scale);

        // Edges
        ctx.strokeStyle = EDGE_COLOR;
        ctx.lineWidth = 1;
        data.edges.forEach(e => {
            const pu = positions[e.source];
            const pv = positions[e.target];
            if (!pu || !pv) return;
            ctx.beginPath();
            ctx.moveTo(pu.x, pu.y);
            ctx.lineTo(pv.x, pv.y);
            ctx.stroke();
        });

        // Nodes
        data.nodes.forEach(n => {
            const p = positions[n.id];
            if (!p) return;
            const isSelected = selectedNode?.node.id === n.id;
            ctx.beginPath();
            ctx.arc(p.x, p.y, NODE_RADIUS, 0, 2 * Math.PI);
            ctx.fillStyle = isSelected ? SELECTED_COLOR : NODE_COLOR;
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Label
            ctx.fillStyle = '#1e293b';
            ctx.font = '9px system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(n.id.length > 20 ? n.id.slice(0, 18) + '…' : n.id, p.x, p.y + NODE_RADIUS + 10);
        });

        ctx.restore();
    }

    // Redraw on scale/offset/selection change
    useEffect(() => {
        if (graphData) draw(graphData, positionsRef.current);
    }, [scale, offset, selectedNode, graphData]);

    // Canvas click → select node
    const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (!graphData) return;
        const canvas = canvasRef.current!;
        const rect = canvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left - offset.x) / scale;
        const my = (e.clientY - rect.top - offset.y) / scale;

        let hit: GraphNode | null = null;
        for (const n of graphData.nodes) {
            const p = positionsRef.current[n.id];
            if (!p) continue;
            const dx = mx - p.x;
            const dy = my - p.y;
            if (Math.sqrt(dx * dx + dy * dy) <= NODE_RADIUS + 4) {
                hit = n;
                break;
            }
        }

        if (hit) {
            setSelectedNode(prev =>
                prev?.node.id === hit!.id ? null : { node: hit!, x: e.clientX, y: e.clientY },
            );
        } else {
            setSelectedNode(null);
        }
    };

    const handleMouseDown = (e: React.MouseEvent) => {
        isDragging.current = true;
        lastMouse.current = { x: e.clientX, y: e.clientY };
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (!isDragging.current) return;
        setOffset(prev => ({
            x: prev.x + e.clientX - lastMouse.current.x,
            y: prev.y + e.clientY - lastMouse.current.y,
        }));
        lastMouse.current = { x: e.clientX, y: e.clientY };
    };

    const handleMouseUp = () => {
        isDragging.current = false;
    };

    const handleWheel = (e: React.WheelEvent) => {
        e.preventDefault();
        setScale(s => Math.max(0.2, Math.min(4, s - e.deltaY * 0.001)));
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

    return (
        <div className="flex flex-col gap-3">
            {/* Toolbar */}
            <div className="flex items-center gap-2 flex-wrap">
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
                <button
                    onClick={() => fetchGraph(search)}
                    className="p-2 text-gray-500 hover:text-indigo-600 border border-gray-200 rounded"
                    title="Refresh"
                >
                    <RefreshCw className="w-4 h-4" />
                </button>
                <button
                    onClick={() => setScale(s => Math.min(4, s + 0.2))}
                    className="p-2 text-gray-500 hover:text-indigo-600 border border-gray-200 rounded"
                    title="Zoom in"
                >
                    <ZoomIn className="w-4 h-4" />
                </button>
                <button
                    onClick={() => setScale(s => Math.max(0.2, s - 0.2))}
                    className="p-2 text-gray-500 hover:text-indigo-600 border border-gray-200 rounded"
                    title="Zoom out"
                >
                    <ZoomOut className="w-4 h-4" />
                </button>
                <span className="text-xs text-gray-400">
                    {graphData.node_count} nodes · {graphData.edge_count} edges
                    {graphData.truncated && ' · (truncated)'}
                </span>
            </div>

            {graphData.truncated && (
                <div className="text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded px-3 py-1.5">
                    Graph was truncated to {graphData.node_count} nodes. Use search to filter the view.
                </div>
            )}

            {/* Canvas */}
            <div className="relative border border-gray-200 rounded overflow-hidden" style={{ cursor: 'grab' }}>
                <canvas
                    ref={canvasRef}
                    width={800}
                    height={560}
                    className="w-full"
                    onClick={handleCanvasClick}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                    onWheel={handleWheel}
                    style={{ display: 'block', background: BG_COLOR }}
                />

                {/* Node detail panel */}
                {selectedNode && (
                    <div className="absolute top-3 right-3 bg-white shadow-md border border-gray-200 rounded p-3 text-xs max-w-[240px] space-y-1.5">
                        <div className="font-semibold text-gray-800 truncate" title={selectedNode.node.id}>
                            {selectedNode.node.id}
                        </div>
                        {selectedNode.node.labels.length > 0 && (
                            <div className="flex flex-wrap gap-1">
                                {selectedNode.node.labels.map(l => (
                                    <span key={l} className="bg-indigo-100 text-indigo-700 rounded px-1.5 py-0.5">{l}</span>
                                ))}
                            </div>
                        )}
                        {Object.entries(selectedNode.node.properties)
                            .filter(([k]) => k !== 'workspace')
                            .slice(0, 6)
                            .map(([k, v]) => (
                                <div key={k} className="flex gap-1">
                                    <span className="text-gray-500 shrink-0">{k}:</span>
                                    <span className="text-gray-800 truncate">{String(v)}</span>
                                </div>
                            ))}
                    </div>
                )}
            </div>
        </div>
    );
};

export default SiloGraphView;
