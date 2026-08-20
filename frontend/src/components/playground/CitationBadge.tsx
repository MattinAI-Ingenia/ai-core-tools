import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { LightRAGChunk } from '../../types/streaming';
import OpenPdfButton from './OpenPdfButton';

interface Props {
  index: number;             // 1-based source number the LLM cited (cite://N)
  chunk?: LightRAGChunk;     // resolved from message chunks[index - 1]
  appId?: number;
  siloId?: number;
}

const POPOVER_WIDTH = 384;   // w-96
const POPOVER_MAX_HEIGHT = 320;

/**
 * Inline citation blob rendered at the end of a sentence: a superscript [N]
 * linking back to the LightRAG source chunk. Clicking opens a popover with the
 * chunk's file path and full text — the same content the subgraph bubble shows.
 *
 * When no chunk resolves (e.g. a multi-retrieval turn where the frontend only
 * kept the last call's chunks), it degrades to a plain, non-interactive marker
 * instead of a dead link.
 */
export default function CitationBadge({ index, chunk, appId, siloId }: Props) {
  const [pos, setPos] = useState<{ top: number; left: number; placement: 'top' | 'bottom' } | null>(null);
  const popoverRef = useRef<HTMLDivElement>(null);

  // Close on outside scroll/resize (fixed coords would drift); keep open when
  // scrolling inside the popover itself.
  useEffect(() => {
    if (!pos) return;
    const close = (e?: Event) => {
      if (e && popoverRef.current && e.target instanceof Node && popoverRef.current.contains(e.target)) return;
      setPos(null);
    };
    window.addEventListener('scroll', close, true);
    window.addEventListener('resize', close);
    return () => {
      window.removeEventListener('scroll', close, true);
      window.removeEventListener('resize', close);
    };
  }, [pos]);

  if (!chunk) {
    return <sup className="text-[0.7em] text-gray-400 dark:text-gray-500 mx-0.5">[{index}]</sup>;
  }

  const srcLabel = chunk.file_path && chunk.file_path !== 'unknown_source' ? chunk.file_path : 'Unknown source';

  const toggle = (e: React.MouseEvent<HTMLButtonElement>) => {
    if (pos) { setPos(null); return; }
    const rect = e.currentTarget.getBoundingClientRect();
    const spaceBelow = window.innerHeight - rect.bottom;
    const placeBelow = spaceBelow >= POPOVER_MAX_HEIGHT || spaceBelow >= rect.top;
    setPos({
      top: placeBelow ? rect.bottom + 4 : rect.top - 4,
      left: Math.min(Math.max(8, rect.left), window.innerWidth - POPOVER_WIDTH - 8),
      placement: placeBelow ? 'bottom' : 'top',
    });
  };

  return (
    <>
      <sup className="mx-0.5">
        <button
          type="button"
          onClick={toggle}
          aria-expanded={pos !== null}
          title={srcLabel}
          className={`inline-flex items-center justify-center align-baseline text-[0.7em] font-medium leading-none min-w-[1.3em] px-1 py-0.5 rounded transition-colors ${pos
            ? 'bg-indigo-200 dark:bg-indigo-900/60 text-indigo-800 dark:text-indigo-200'
            : 'bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 hover:bg-indigo-200 dark:hover:bg-indigo-900/60'
            }`}
        >
          {index}
        </button>
      </sup>

      {pos && createPortal(
        <>
          <div className="fixed inset-0 z-[60]" onClick={() => setPos(null)} aria-hidden="true" />
          <div
            ref={popoverRef}
            className="fixed z-[61] w-96 max-w-[90vw] bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg shadow-xl text-xs text-gray-700 dark:text-gray-200"
            style={{
              left: pos.left,
              ...(pos.placement === 'bottom' ? { top: pos.top } : { bottom: window.innerHeight - pos.top }),
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between gap-2 px-3 py-2 border-b border-gray-100 dark:border-gray-600">
              <span className="font-semibold text-gray-800 dark:text-gray-100 break-all" title={srcLabel}>
                {srcLabel}
                <span className="ml-1 font-normal text-gray-400">· Chunk {index}</span>
              </span>
              <button
                onClick={() => setPos(null)}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 shrink-0"
                aria-label="Close"
              >
                <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="max-h-80 overflow-y-auto p-3">
              <p className="whitespace-pre-wrap leading-relaxed">{chunk.content ?? '(no content)'}</p>
            </div>
            <OpenPdfButton appId={appId} siloId={siloId} resourceId={chunk.resource_id} page={chunk.page} />
          </div>
        </>,
        document.body,
      )}
    </>
  );
}
