import React, { useState, useMemo, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { InteractiveNvlWrapper } from '@neo4j-nvl/react';
import type { MouseEventCallbacks } from '@neo4j-nvl/react';
import type { Node as NvlNode, NvlOptions, Relationship as NvlRelationship } from '@neo4j-nvl/base';
import type { LightRAGGraphData, LightRAGEntity, LightRAGRelationship, LightRAGChunk } from '../../types/streaming';

interface Props {
  graphData: LightRAGGraphData;
}

function toNvlNodes(entities: LightRAGEntity[]): NvlNode[] {
  return entities.map((e) => ({
    id: String(e.id),
    captions: [{ value: String(e.name ?? e.id) }],
    color: '#6366f1',
  }));
}

function toNvlRels(rels: LightRAGRelationship[], nodeIds: Set<string>): NvlRelationship[] {
  return rels
    .filter((r) => r.source && r.target && nodeIds.has(String(r.source)) && nodeIds.has(String(r.target)))
    .map((r) => {
      const label = String(r.keywords ?? r.description ?? '').slice(0, 40);
      return {
        id: String(r.id),
        from: String(r.source),
        to: String(r.target),
        captions: label ? [{ value: label }] : [],
        color: '#94a3b8',
      };
    });
}

export default function LightRAGGraphBubble({ graphData }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [openChunk, setOpenChunk] = useState<number | null>(null);
  const [popoverPos, setPopoverPos] = useState<{ top: number; left: number; placement: 'top' | 'bottom' } | null>(null);
  const [selectedEntity, setSelectedEntity] = useState<LightRAGEntity | null>(null);
  const popoverRef = useRef<HTMLDivElement>(null);

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

  const entityCount = entities.length;
  const relCount = relationships.length;
  const chunkCount = chunks.length;

  const { nvlNodes, nvlRels } = useMemo(() => {
    const nodes = toNvlNodes(entities);
    const ids = new Set(nodes.map((n) => n.id));
    return { nvlNodes: nodes, nvlRels: toNvlRels(relationships, ids) };
  }, [entities, relationships]);

  if (entityCount === 0 && relCount === 0 && chunkCount === 0) return null;


  // Interaction config — mirrors SiloGraphView so the bubble graph is
  // pannable/zoomable/draggable instead of a static render.
  const nvlOptions: NvlOptions = { layout: 'forceDirected', allowDynamicMinZoom: true };
  const mouseEventCallbacks: MouseEventCallbacks = {
    onNodeClick: (node: NvlNode) => {
      const original = entities.find((e) => String(e.id) === node.id) ?? null;
      setSelectedEntity((prev) => (prev?.id === node.id ? null : original));
    },
    onCanvasClick: () => setSelectedEntity(null),
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
            <circle cx="12" cy="12" r="3" />
            <path d="M12 2v3M12 19v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M2 12h3M19 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12" />
          </svg>
          <span className="text-xs font-medium text-indigo-700 dark:text-indigo-300 flex-1">
            Subgrafo · {entityCount} entidades · {relCount} relaciones
            {chunkCount > 0 && ` · ${chunkCount} fragmento${chunkCount > 1 ? 's' : ''}`}
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
                {chunks.map((c, idx) => {
                  const isOpen = openChunk === idx;
                  const srcLabel = c.file_path && c.file_path !== 'unknown_source'
                    ? c.file_path
                    : 'Unknown source';
                  return (
                    <button
                      key={c.id ?? idx}
                      onClick={(e) => toggleChunk(idx, e)}
                      aria-expanded={isOpen}
                      className={`inline-flex items-center gap-1 text-xs rounded-full px-2.5 py-1 transition-colors ${isOpen
                        ? 'bg-indigo-200 dark:bg-indigo-900/60 text-indigo-800 dark:text-indigo-200'
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
              <button
                onClick={() => setModalOpen(true)}
                className="inline-flex items-center gap-1.5 text-xs text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-200 font-medium transition-colors"
              >
                <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
                  <circle cx="12" cy="12" r="3" />
                  <path d="M19.07 4.93a10 10 0 010 14.14M4.93 4.93a10 10 0 000 14.14" />
                </svg>
                Ver subgrafo
              </button>
            )}
          </div>
        )}
      </div>

      {/* Chunk popover — rendered in a portal with fixed positioning so it
          escapes the bubble's and the chat scroll container's overflow. */}
      {openChunk !== null && popoverPos && chunks[openChunk] && createPortal(
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
              <span className="font-semibold text-gray-800 dark:text-gray-100 break-all" title={chunks[openChunk].file_path ?? ''}>
                {chunks[openChunk].file_path && chunks[openChunk].file_path !== 'unknown_source'
                  ? chunks[openChunk].file_path
                  : 'Unknown source'}
                <span className="ml-1 font-normal text-gray-400">· Fragmento {openChunk + 1}</span>
              </span>
              <button
                onClick={() => { setOpenChunk(null); setPopoverPos(null); }}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 shrink-0"
                aria-label="Cerrar"
              >
                <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="max-h-80 overflow-y-auto p-3">
              <p className="whitespace-pre-wrap leading-relaxed">
                {chunks[openChunk].content ?? '(sin contenido)'}
              </p>
            </div>
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
          aria-label="Subgrafo de conocimiento"
        >
          <div
            className="bg-white dark:bg-gray-900 rounded-2xl shadow-2xl w-[90vw] h-[80vh] max-w-5xl flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700">
              <span className="font-semibold text-gray-800 dark:text-gray-100 text-sm">
                Subgrafo · {entityCount} entidades · {relCount} relaciones
              </span>
              <button
                onClick={closeModal}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors"
                aria-label="Cerrar"
              >
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="flex-1 relative">
              {nvlNodes.length > 0 ? (
                <>
                  <InteractiveNvlWrapper
                    nodes={nvlNodes}
                    rels={nvlRels}
                    nvlOptions={nvlOptions}
                    mouseEventCallbacks={mouseEventCallbacks}
                    style={{ width: '100%', height: '100%' }}
                  />

                  {selectedEntity && (
                    <div className="absolute top-3 right-3 bg-white dark:bg-gray-800 shadow-md border border-gray-200 dark:border-gray-700 rounded p-3 text-xs max-w-[260px] space-y-1.5 z-10">
                      <div className="flex items-start justify-between gap-2">
                        <span className="font-semibold text-gray-800 dark:text-gray-100 break-all" title={String(selectedEntity.name ?? selectedEntity.id)}>
                          {String(selectedEntity.name ?? selectedEntity.id)}
                        </span>
                        <button
                          onClick={() => setSelectedEntity(null)}
                          className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 shrink-0 mt-0.5"
                          aria-label="Cerrar detalle"
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
                    </div>
                  )}
                </>
              ) : (
                <div className="flex items-center justify-center h-full text-gray-400 text-sm">
                  Sin entidades para visualizar
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
