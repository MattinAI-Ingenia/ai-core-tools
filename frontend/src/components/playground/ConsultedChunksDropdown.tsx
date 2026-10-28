import React, { useState } from 'react';

interface Props {
  /** Number of distinct consulted chunks, shown in the header. */
  count?: number;
  children: React.ReactNode;
}

/** Collapsible container holding the per-silo chunks + subgraphs LightRAG
 *  consulted for a turn, cited or not. */
export default function ConsultedChunksDropdown({ count, children }: Props) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-2 border border-indigo-200 dark:border-indigo-700/50 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 bg-indigo-50 dark:bg-indigo-900/20 hover:bg-indigo-100 dark:hover:bg-indigo-900/30 transition-colors text-left"
        aria-expanded={open}
      >
        <svg className="w-4 h-4 text-indigo-500 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <span className="text-xs font-medium text-indigo-700 dark:text-indigo-300 flex-1">
          Chunks consultados{typeof count === 'number' && count > 0 ? ` · ${count}` : ''}
        </span>
        <svg
          className={`w-4 h-4 text-indigo-400 transition-transform ${open ? 'rotate-180' : ''}`}
          viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {open && <div className="p-2 space-y-2">{children}</div>}
    </div>
  );
}
