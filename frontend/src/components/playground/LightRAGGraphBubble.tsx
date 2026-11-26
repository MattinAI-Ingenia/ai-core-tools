import React, { useState, useMemo, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { InteractiveNvlWrapper } from '@neo4j-nvl/react';
import type { MouseEventCallbacks } from '@neo4j-nvl/react';
import type { Node as NvlNode, NvlOptions, Relationship as NvlRelationship } from '@neo4j-nvl/base';
import type NVL from '@neo4j-nvl/base';
import type { LightRAGGraphData, LightRAGEntity, LightRAGRelationship } from '../../types/streaming';
import OpenPdfButton from './OpenPdfButton';
import { groupChunksById } from '../../utils/groupChunksById';
import { bubbleNodeFill, bubbleNodeIcon, fromCitedChunk } from '../../utils/bubbleGraphStyle';
import { PARTIAL_COLOR, RING_CITED, RING_RETRIEVED, PRIMARY_COLOR, HIGHLIGHT_BLUE, colorForType, typeRingIcon } from '../../utils/entityTypeColors';
import { splitSourceIds } from '../../utils/siloGraphUtils';
import type { GraphSourceDoc } from '../../utils/siloGraphUtils';
import { apiService } from '../../services/api';
import { openSourcePdf } from '../../utils/openSourcePdf';
import { Loader2 } from 'lucide-react';

interface Props {
  graphData: LightRAGGraphData;
  /** Chunk ids the answer actually cited ([N](cite://N)). Entities/relationships
   *  extracted from one of these are highlighted green ("used"). */
  citedChunkIds?: string[];
  appId?: number;
  siloId?: number;
  /** Silo name shown before the counts when the turn spans several silos. */
  title?: string;
  /** Display number of each chunk at its position (0 = not cited → hidden).
   *  Omitted → fall back to 1-based positions (silo search view). */
  chunkNumbers?: number[];
}

// Node/edge styling lives in utils/bubbleGraphStyle (pure, tested) — the
// cited/retrieved marks are RINGS (overlayIcon) when type coloring is on, and
// the old green/indigo FILL when it is off.

function toNvlNodes(entities: LightRAGEntity[], cited: Set<string>, typeColoring: boolean, chosenIds: Set<string>, neighborIds: Set<string>): NvlNode[] {
  return entities.map((e) => ({
    id: String(e.id),
    captions: [{ value: String(e.name ?? e.id) }],
    color: bubbleNodeFill(e, typeColoring, cited),
    // Chosen → yellow NVL selection ring; adjacent → blue ring overlay
    // (replaces the cited/retrieved ring while the selection is active).
    overlayIcon: neighborIds.has(String(e.id))
      ? typeRingIcon(HIGHLIGHT_BLUE)
      : chosenIds.has(String(e.id)) ? undefined : bubbleNodeIcon(e, cited, typeColoring),
    selected: chosenIds.has(String(e.id)),
  }));
}

function toNvlRels(rels: LightRAGRelationship[], nodeIds: Set<string>, cited: Set<string>, chosenIds: Set<string>, neighborIds: Set<string>): NvlRelationship[] {
  return rels
    .filter((r) => r.source && r.target && nodeIds.has(String(r.source)) && nodeIds.has(String(r.target)))
    .map((r) => {
      const label = String(r.keywords ?? r.description ?? '').slice(0, 40);
      return {
        id: String(r.id),
        from: String(r.source),
        to: String(r.target),
        captions: label ? [{ value: label }] : [],
        color: neighborIds.has(String(r.id)) ? HIGHLIGHT_BLUE : fromCitedChunk(r.source_id, cited) ? RING_CITED : PARTIAL_COLOR,
        selected: chosenIds.has(String(r.id)),
      };
    });
}

export default function LightRAGGraphBubble({ graphData, citedChunkIds, appId, siloId, title, chunkNumbers }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [openChunk, setOpenChunk] = useState<number | null>(null);
  const [popoverPos, setPopoverPos] = useState<{ top: number; left: number; placement: 'top' | 'bottom' } | null>(null);
  const [selectedEntity, setSelectedEntity] = useState<LightRAGEntity | null>(null);
  const [selectedRel, setSelectedRel] = useState<LightRAGRelationship | null>(null);
  const [entitySources, setEntitySources] = useState<GraphSourceDoc[] | null>(null);
  const [entitySourcesLoading, setEntitySourcesLoading] = useState(false);
  const [typeColoring, setTypeColoring] = useState(true);
  const popoverRef = useRef<HTMLDivElement>(null);
  const nvlRef = useRef<NVL>(null);

  // Close the (portal-rendered) chunk popover on outside scroll/resize, since
  // its fixed coordinates would otherwise drift away from the chip. Scrolling
  // *inside* the popover must NOT close it.
  useEffect(() => {
    if (openChunk === null) return;
    const close = (e?: Event) => {
      if (e && popoverRef.current && e.target instanceof Node && popoverRef.current.contains(e.target)) {
        return; // scroll happened inside the popover — keep it open
      }
      setOpenChunk(null);
      setPopoverPos(null);
    };
    window.addEventListener('scroll', close, true);
    window.addEventListener('resize', close);
    return () => {
      window.removeEventListener('scroll', close, true);
      window.removeEventListener('resize', close);
    };
  }, [openChunk]);

  const POPOVER_WIDTH = 384; // w-96
  const POPOVER_MAX_HEIGHT = 320;

  // Source documents of the selected entity — best-effort: a failed lookup
  // just hides the chips (same as the silo graph's detail panel).
  useEffect(() => {
    setEntitySources(null);
    const silo = selectedEntity?.silo_id ?? siloId;
    const ids = splitSourceIds(selectedEntity?.source_id);
    if (!selectedEntity || !appId || !silo || ids.length === 0) return;
    setEntitySourcesLoading(true);
    let cancelled = false;
    apiService
      .getSiloGraphSources(appId, silo, ids)
      .then((data: { sources: GraphSourceDoc[] }) => {
        if (!cancelled) setEntitySources(data.sources);
      })
      .catch(() => {
        /* best-effort */
      })
      .finally(() => {
        if (!cancelled) setEntitySourcesLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedEntity?.id, appId, siloId]);

  const toggleChunk = (idx: number, e: React.MouseEvent<HTMLButtonElement>) => {
    if (openChunk === idx) {
      setOpenChunk(null);
      setPopoverPos(null);
      return;
    }
    const rect = e.currentTarget.getBoundingClientRect();
    const spaceBelow = window.innerHeight - rect.bottom;
    const placeBelow = spaceBelow >= POPOVER_MAX_HEIGHT || spaceBelow >= rect.top;
    const left = Math.min(
      Math.max(8, rect.left),
      window.innerWidth - POPOVER_WIDTH - 8,
    );
    setPopoverPos({
      top: placeBelow ? rect.bottom + 4 : rect.top - 4,
      left,
      placement: placeBelow ? 'bottom' : 'top',
    });
    setOpenChunk(idx);
  };

  const data = graphData?.data ?? {};
  const entities = data.entities ?? [];
  const relationships = data.relationships ?? [];
  const chunks = data.chunks ?? [];
  const references = data.references ?? [];

  // Partial nodes are synthesized endpoints, not real returned entities — exclude
  // them from the count so the header reflects the entities LightRAG actually returned.
  const entityCount = entities.filter((e) => !e.partial).length;
  // A turn's payload repeats a page whenever several tool calls return it; show each chunk once.
  const chunkGroups = useMemo(() => groupChunksById(chunks), [chunks]);
  const chunkCount = chunkGroups.length;
  const openGroup = openChunk !== null ? chunkGroups[openChunk] : undefined;
  // Every [N] the answer can use for this chunk (one per copy in the payload).
  // With renumbered citations the caller passes 0 for chunks the answer did not
  // cite — hide those; without any numbering fall back to 1-based positions.
  const openNumbers = chunkNumbers
    ? [...new Set((openGroup?.positions ?? []).map((p) => chunkNumbers?.[p]).filter((n): n is number => !!n))]
    : (openGroup?.positions ?? []).map((p) => p + 1);

  // Memoize the cited-chunk set on a stable key so identity churn in the parent
  // doesn't force the graph to re-layout every render.
  const citedKey = (citedChunkIds ?? []).join(',');
  const citedSet = useMemo(() => new Set(citedChunkIds ?? []), [citedKey]);
  const hasCited = citedSet.size > 0;

  const { nvlNodes, nvlRels } = useMemo(() => {
    // Highlight: clicked node → itself + its neighbors + incident edges;
    // clicked relationship → itself + its two endpoints.
    const chosenNodes = new Set<string>();
    const chosenEdges = new Set<string>();
    const neighborNodes = new Set<string>();
    const neighborEdges = new Set<string>();
    if (selectedEntity) {
      chosenNodes.add(String(selectedEntity.id));
      for (const r of relationships) {
        if (String(r.source) === String(selectedEntity.id)) {
          neighborNodes.add(String(r.target));
          neighborEdges.add(String(r.id));
        } else if (String(r.target) === String(selectedEntity.id)) {
          neighborNodes.add(String(r.source));
          neighborEdges.add(String(r.id));
        }
      }
    } else if (selectedRel) {
      chosenEdges.add(String(selectedRel.id));
      neighborNodes.add(String(selectedRel.source));
      neighborNodes.add(String(selectedRel.target));
    }
    const nodes = toNvlNodes(entities, citedSet, typeColoring, chosenNodes, neighborNodes);
    const ids = new Set(nodes.map((n) => n.id));
    return { nvlNodes: nodes, nvlRels: toNvlRels(relationships, ids, citedSet, chosenEdges, neighborEdges) };
  }, [entities, relationships, citedSet, typeColoring, selectedEntity, selectedRel]);

  // Count only the relationships actually drawn (both endpoints present) so the
  // header matches the canvas instead of LightRAG's raw relationship total.
  const relCount = nvlRels.length;

  // naive mode returns chunks but no graph — then the banner shows only the chunk
  // count, no "subgraph"/entities wording.
  const hasGraph = entityCount > 0 || relCount > 0;

  if (entityCount === 0 && relCount === 0 && chunkCount === 0) return null;


  // Interaction config — mirrors SiloGraphView so the bubble graph is
  // pannable/zoomable/draggable instead of a static render.
  const nvlOptions: NvlOptions = {
    layout: 'forceDirected',
    allowDynamicMinZoom: true,
    // Selection ring AROUND the element (fill untouched) — bright yellow.
    styling: {
      selectedBorderColor: PRIMARY_COLOR,
      selectedInnerBorderColor: PRIMARY_COLOR,
    },
  };
  const mouseEventCallbacks: MouseEventCallbacks = {
    onNodeClick: (node: NvlNode) => {
      setSelectedRel(null);
      // Recentre the view on the clicked node (also a zoom-to-fit on it).
      nvlRef.current?.fit([node.id]);
      const original = entities.find((e) => String(e.id) === node.id) ?? null;
      setSelectedEntity((prev) => (prev?.id === node.id ? null : original));
    },
    onRelationshipClick: (rel: NvlRelationship) => {
      setSelectedEntity(null);
      const original = relationships.find((r) => String(r.id) === String(rel.id)) ?? null;
      setSelectedRel((prev) => (prev?.id === original?.id ? null : original));
    },
    onCanvasClick: () => {
      setSelectedEntity(null);
      setSelectedRel(null);
    },
    onPan: true,
    onZoom: true,
    onDrag: true,
  };

  const closeModal = () => {
    setModalOpen(false);
    setSelectedEntity(null);
  };

  return (
    <>
      <div className="mt-2 border border-indigo-200 dark:border-indigo-700/50 rounded-xl overflow-hidden">
        <button
          onClick={() => setExpanded((v) => !v)}
          className="w-full flex items-center gap-2 px-3 py-2 bg-indigo-50 dark:bg-indigo-900/20 hover:bg-indigo-100 dark:hover:bg-indigo-900/30 transition-colors text-left"
          aria-expanded={expanded}
        >
          <svg className="w-4 h-4 text-indigo-500 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
            {hasGraph ? (
              <>
                <circle cx="12" cy="12" r="3" />
                <path d="M12 2v3M12 19v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M2 12h3M19 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12" />
              </>
            ) : (
              // naive/chunks-only → document icon (matches the chunk chips below)
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            )}
          </svg>
          <span className="text-xs font-medium text-indigo-700 dark:text-indigo-300 flex-1">
            {title ? `${title} · ` : ''}
            {hasGraph
              ? `Knowledge subgraph · ${entityCount} ${entityCount === 1 ? 'entity' : 'entities'} · ${relCount} ${relCount === 1 ? 'relationship' : 'relationships'}${chunkCount > 0 ? ` · ${chunkCount} ${chunkCount === 1 ? 'chunk' : 'chunks'}` : ''}`
              : `${chunkCount} ${chunkCount === 1 ? 'chunk' : 'chunks'}`}
          </span>
          <svg
            className={`w-4 h-4 text-indigo-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
            viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {expanded && (
          <div className="px-3 py-2 bg-white dark:bg-gray-800/50 space-y-2">
            {chunkCount > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {chunkGroups.map(({ chunk: c }, idx) => {
                  const isOpen = openChunk === idx;
                  const isCited = citedSet.has(c.id);
                  const srcLabel = c.file_path && c.file_path !== 'unknown_source'
                    ? c.file_path
                    : 'Unknown source';
                  return (
                    <button
                      key={`${c.id ?? 'chunk'}-${idx}`}
                      onClick={(e) => toggleChunk(idx, e)}
                      aria-expanded={isOpen}
                      title={isCited ? `${srcLabel} (citado en la respuesta)` : srcLabel}
                      className={`inline-flex items-center gap-1 text-xs rounded-full px-2.5 py-1 transition-colors ${isOpen
                        ? isCited
                          ? 'bg-green-200 dark:bg-green-900/60 text-green-800 dark:text-green-200'
                          : 'bg-indigo-200 dark:bg-indigo-900/60 text-indigo-800 dark:text-indigo-200'
                        : isCited
                          ? 'bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300 hover:bg-green-200 dark:hover:bg-green-900/60'
                          : 'bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 hover:bg-indigo-200 dark:hover:bg-indigo-900/60'
                        }`}
                    >
                      <svg className="w-3 h-3 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      <span className="max-w-[180px] truncate" title={srcLabel}>{srcLabel}</span>
                    </button>
                  );
                })}
              </div>
            )}
            {(entityCount > 0 || relCount > 0) && (
              <div className="flex items-center justify-between gap-2 flex-wrap">
                <button
                  onClick={() => setModalOpen(true)}
                  className="inline-flex items-center gap-1.5 text-xs text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-200 font-medium transition-colors"
                >
                  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
                    <circle cx="12" cy="12" r="3" />
                    <path d="M19.07 4.93a10 10 0 010 14.14M4.93 4.93a10 10 0 000 14.14" />
                  </svg>
                  View subgraph
                </button>
                <label
                  className="inline-flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400"
                  title="Color nodes by entity type; when off, cited nodes are green and retrieved ones indigo"
                >
                  <input
                    type="checkbox"
                    checked={typeColoring}
                    onChange={(e) => setTypeColoring(e.target.checked)}
                    className="accent-indigo-600"
                  />
                  Color by type
                </label>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Chunk popover — rendered in a portal with fixed positioning so it
          escapes the bubble's and the chat scroll container's overflow. */}
      {openChunk !== null && popoverPos && openGroup && createPortal(
        <>
          {/* click-outside catcher */}
          <div
            className="fixed inset-0 z-[60]"
            onClick={() => { setOpenChunk(null); setPopoverPos(null); }}
            aria-hidden="true"
          />
          <div
            ref={popoverRef}
            className="fixed z-[61] w-96 max-w-[90vw] bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg shadow-xl text-xs text-gray-700 dark:text-gray-200"
            style={{
              left: popoverPos.left,
              ...(popoverPos.placement === 'bottom'
                ? { top: popoverPos.top }
                : { bottom: window.innerHeight - popoverPos.top }),
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between gap-2 px-3 py-2 border-b border-gray-100 dark:border-gray-600">
              <span className="font-semibold text-gray-800 dark:text-gray-100 break-all" title={openGroup.chunk.file_path ?? ''}>
                {openGroup.chunk.file_path && openGroup.chunk.file_path !== 'unknown_source'
                  ? openGroup.chunk.file_path
                  : 'Unknown source'}
                {openNumbers.length > 0 && (
                  <span className="ml-1 font-normal text-gray-400">· {openNumbers.length === 1 ? 'Chunk' : 'Chunks'} {openNumbers.join(', ')}</span>
                )}
              </span>
              <button
                onClick={() => { setOpenChunk(null); setPopoverPos(null); }}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 shrink-0"
                aria-label="Close"
              >
                <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="max-h-80 overflow-y-auto p-3">
              <p className="whitespace-pre-wrap leading-relaxed">
                {openGroup.chunk.content ?? '(no content)'}
              </p>
            </div>
            <OpenPdfButton
              appId={appId}
              // The cited-evidence bubble spans silos, so fall back to the
              // chunk's own silo when the caller doesn't pass one.
              siloId={siloId ?? openGroup.chunk.silo_id}
              resourceId={openGroup.chunk.resource_id}
              page={openGroup.chunk.page}
            />
          </div>
        </>,
        document.body,
      )}

      {modalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
          onClick={closeModal}
          role="dialog"
          aria-modal="true"
          aria-label="Knowledge subgraph"
        >
          <div
            className="bg-white dark:bg-gray-900 rounded-2xl shadow-2xl w-[90vw] h-[80vh] max-w-5xl flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700">
              <span className="font-semibold text-gray-800 dark:text-gray-100 text-sm">
                Knowledge subgraph · {entityCount} {entityCount === 1 ? 'entity' : 'entities'} · {relCount} {relCount === 1 ? 'relationship' : 'relationships'}
              </span>
              <button
                onClick={closeModal}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors"
                aria-label="Close"
              >
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-4 py-2 border-b border-gray-200 dark:border-gray-700 text-xs text-gray-600 dark:text-gray-300">
              {hasCited && (
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: RING_CITED }} aria-hidden="true" />
                  In cited sources
                </span>
              )}
              <span className="inline-flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: RING_RETRIEVED }} aria-hidden="true" />
                Retrieved
              </span>
              {entities.some((e) => e.partial) && (
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: PARTIAL_COLOR }} aria-hidden="true" />
                  Partial (relationship endpoint)
                </span>
              )}
            </div>
            <div className="flex-1 relative">
              {nvlNodes.length > 0 ? (
                <>
                  <InteractiveNvlWrapper
                    ref={nvlRef}
                    nodes={nvlNodes}
                    rels={nvlRels}
                    nvlOptions={nvlOptions}
                    mouseEventCallbacks={mouseEventCallbacks}
                    style={{ width: '100%', height: '100%' }}
                  />

                  {(selectedEntity || selectedRel) && (
                    <div className="absolute top-3 right-3 bg-white dark:bg-gray-800 shadow-md border border-gray-200 dark:border-gray-700 rounded p-3 text-xs max-w-[280px] max-h-[85%] overflow-y-auto space-y-1.5 z-10">
                      {selectedEntity ? (
                        <>
                          <div className="flex items-start justify-between gap-2">
                            <span className="flex items-center gap-1.5 min-w-0">
                              <span
                                className="w-2.5 h-2.5 rounded-full shrink-0"
                                style={{ background: colorForType(selectedEntity.entity_type as string | undefined) }}
                                aria-hidden="true"
                              />
                              <span className="font-semibold text-gray-800 dark:text-gray-100 break-all" title={String(selectedEntity.name ?? selectedEntity.id)}>
                                {String(selectedEntity.name ?? selectedEntity.id)}
                              </span>
                            </span>
                            <button
                              onClick={() => setSelectedEntity(null)}
                              className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 shrink-0 mt-0.5"
                              aria-label="Close details"
                            >
                              <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                              </svg>
                            </button>
                          </div>
                          <div className="border-t border-gray-100 dark:border-gray-700 pt-1.5 space-y-1">
                            {Object.entries(selectedEntity)
                              .filter(([k, v]) => k !== 'id' && k !== 'name' && v != null && String(v) !== '')
                              .slice(0, 8)
                              .map(([k, v]) => (
                                <div key={k} className="flex gap-1">
                                  <span className="text-gray-500 dark:text-gray-400 shrink-0">{k}:</span>
                                  <span className="text-gray-800 dark:text-gray-200 break-all">
                                    {String(v).length > 60 ? String(v).slice(0, 58) + '…' : String(v)}
                                  </span>
                                </div>
                              ))}
                          </div>
                          {(entitySourcesLoading || (entitySources && entitySources.length > 0)) && (
                            <div className="border-t border-gray-100 dark:border-gray-700 pt-1.5">
                              <p className="text-gray-500 dark:text-gray-400 mb-1">Source documents</p>
                              {entitySourcesLoading ? (
                                <Loader2 className="w-3 h-3 animate-spin text-gray-400" />
                              ) : (
                                <div className="flex flex-wrap gap-1">
                                  {(entitySources ?? []).map((s) => {
                                    const label = s.file_name
                                      ? s.page != null
                                        ? `${s.file_name} (p. ${s.page})`
                                        : s.file_name
                                      : 'Unknown source';
                                    const openable = s.resource_id != null && s.page != null && appId != null;
                                    return openable ? (
                                      <button
                                        key={s.chunk_id}
                                        onClick={() => openSourcePdf(appId!, selectedEntity.silo_id ?? siloId!, s.resource_id!, s.page!)}
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
                        </>
                      ) : (
                        (() => {
                          const rel = selectedRel!;
                          const endpoints = [rel.source, rel.target]
                            .map((id) => entities.find((e) => String(e.id) === String(id)))
                            .filter((e): e is LightRAGEntity => !!e);
                          return (
                            <>
                              <div className="flex items-start justify-between gap-2">
                                <span className="font-semibold text-gray-800 dark:text-gray-100 break-all">
                                  {String(rel.type ?? rel.keywords ?? rel.description ?? 'relationship')}
                                </span>
                                <button
                                  onClick={() => setSelectedRel(null)}
                                  className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 shrink-0 mt-0.5"
                                  aria-label="Close details"
                                >
                                  <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                                  </svg>
                                </button>
                              </div>
                              <div className="border-t border-gray-100 dark:border-gray-700 pt-1.5 space-y-1">
                                <p className="text-gray-500 dark:text-gray-400">Connects</p>
                                <div className="flex items-center gap-1.5 flex-wrap">
                                  {endpoints.map((e, i) => (
                                    <React.Fragment key={String(e.id)}>
                                      {i > 0 && <span className="text-gray-400">→</span>}
                                      <button
                                        onClick={() => {
                                          setSelectedRel(null);
                                          setSelectedEntity(e);
                                        }}
                                        title={String(e.name ?? e.id)}
                                        className="inline-flex items-center gap-1 max-w-full truncate rounded-full bg-gray-50 dark:bg-gray-700/60 text-gray-700 dark:text-gray-200 px-1.5 py-0.5 hover:bg-gray-100 dark:hover:bg-gray-700"
                                      >
                                        <span
                                          className="w-2 h-2 rounded-full shrink-0"
                                          style={{ background: colorForType(e.entity_type as string | undefined) }}
                                          aria-hidden="true"
                                        />
                                        {String(e.name ?? e.id).slice(0, 24)}
                                      </button>
                                    </React.Fragment>
                                  ))}
                                </div>
                              </div>
                              <div className="border-t border-gray-100 dark:border-gray-700 pt-1.5 space-y-1">
                                {Object.entries(rel)
                                  .filter(([k, v]) => !['id', 'source', 'target', 'type', 'source_id', 'silo_id', 'silo_name'].includes(k) && v != null && String(v) !== '')
                                  .slice(0, 8)
                                  .map(([k, v]) => (
                                    <div key={k} className="flex gap-1">
                                      <span className="text-gray-500 dark:text-gray-400 shrink-0">{k}:</span>
                                      <span className="text-gray-800 dark:text-gray-200 break-all">
                                        {String(v).length > 60 ? String(v).slice(0, 58) + '…' : String(v)}
                                      </span>
                                    </div>
                                  ))}
                              </div>
                            </>
                          );
                        })()
                      )}
                    </div>
                  )}
                </>
              ) : (
                <div className="flex items-center justify-center h-full text-gray-400 text-sm">
                  No entities to display
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
