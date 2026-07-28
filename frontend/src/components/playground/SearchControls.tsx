import { useEffect, useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';

export interface SearchControlsValue {
  limit: number;
  searchType: 'similarity' | 'similarity_score_threshold' | 'mmr';
  scoreThreshold: number;
  fetchK: number;
  lambdaMult: number;
  searchMethod: 'dense' | 'bm25' | 'hybrid';
  strategy: '' | 'rerank';
  topN: number;
  similarityThreshold: number | null;
}

export const DEFAULT_SEARCH_CONTROLS: SearchControlsValue = {
  limit: 20,
  searchType: 'similarity',
  scoreThreshold: 0.7,
  fetchK: 100,
  lambdaMult: 0.5,
  searchMethod: 'dense',
  strategy: '',
  topN: 5,
  similarityThreshold: null,
};

const SEARCH_METHODS: Array<{ value: SearchControlsValue['searchMethod']; label: string }> = [
  { value: 'dense', label: 'Dense (embeddings similarity)' },
  { value: 'bm25', label: 'BM25 (lexical / keyword)' },
];

const STRATEGIES: Array<{ value: SearchControlsValue['strategy']; label: string }> = [
  { value: '', label: 'None' },
  { value: 'rerank', label: 'Rerank (embeddings-based)' },
];

interface SearchControlsProps {
  siloId: string | number;
  value: SearchControlsValue;
  onChange: (v: SearchControlsValue) => void;
  disabled?: boolean;
}

function storageKey(siloId: string | number): string {
  return `silo-search-controls-${siloId}`;
}

function searchTypeLabel(searchType: SearchControlsValue['searchType']): string {
  switch (searchType) {
    case 'similarity_score_threshold':
      return 'Score Threshold';
    case 'mmr':
      return 'MMR';
    default:
      return 'Cosine Similarity';
  }
}

export default function SearchControls({ siloId, value, onChange, disabled }: Readonly<SearchControlsProps>) {
  const [isOpen, setIsOpen] = useState(false);

  // Load saved state on mount
  useEffect(() => {
    try {
      const raw = localStorage.getItem(storageKey(siloId));
      if (raw) {
        const saved = JSON.parse(raw) as Partial<SearchControlsValue>;
        onChange({ ...DEFAULT_SEARCH_CONTROLS, ...saved });
      }
    } catch {
      // ignore parse errors
    }
  }, [siloId]);

  function handleChange(patch: Partial<SearchControlsValue>) {
    const next = { ...value, ...patch };
    try {
      localStorage.setItem(storageKey(siloId), JSON.stringify(next));
    } catch {
      // ignore storage errors
    }
    onChange(next);
  }

  function handleSearchTypeChange(newType: SearchControlsValue['searchType']) {
    handleChange({
      searchType: newType,
      scoreThreshold: DEFAULT_SEARCH_CONTROLS.scoreThreshold,
      fetchK: DEFAULT_SEARCH_CONTROLS.fetchK,
      lambdaMult: DEFAULT_SEARCH_CONTROLS.lambdaMult,
    });
  }

  const summary = `Top K: ${value.limit} · ${searchTypeLabel(value.searchType)}`;

  return (
    <div className="border border-yellow-200 rounded-lg bg-yellow-50 overflow-hidden">
      {/* Header row */}
      <button
        type="button"
        onClick={() => setIsOpen((prev) => !prev)}
        disabled={disabled}
        className="w-full flex items-center justify-between px-4 py-2 text-left hover:bg-yellow-100 transition-colors disabled:opacity-60"
      >
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-gray-700">Search Controls</span>
          {!isOpen && (
            <span className="text-xs text-gray-500">{summary}</span>
          )}
        </div>
        {isOpen ? (
          <ChevronUp className="w-4 h-4 text-gray-500 shrink-0" />
        ) : (
          <ChevronDown className="w-4 h-4 text-gray-500 shrink-0" />
        )}
      </button>

      {/* Expanded content */}
      {isOpen && (
        <div className="px-4 pb-4 pt-2 space-y-4 border-t border-yellow-200">
          {/* Top-K slider */}
          <div>
            <label className="block text-sm text-gray-700 mb-1">
              Top K results: <span className="font-semibold">{value.limit}</span>
            </label>
            <input
              type="range"
              min={1}
              max={200}
              step={1}
              value={value.limit}
              disabled={disabled}
              onChange={(e) => handleChange({ limit: Number(e.target.value) })}
              className="w-full accent-yellow-600 disabled:opacity-60"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>1</span>
              <span>200</span>
            </div>
          </div>

          {/* Search Method */}
          <div>
            <label htmlFor="search-method-select" className="block text-sm text-gray-700 mb-1">Search Method</label>
            <select
              id="search-method-select"
              value={value.searchMethod}
              disabled={disabled}
              onChange={(e) => handleChange({ searchMethod: e.target.value as SearchControlsValue['searchMethod'] })}
              className="w-full px-3 py-2 border border-yellow-300 rounded-lg bg-white text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-yellow-500 disabled:opacity-60"
            >
              {SEARCH_METHODS.map((m) => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
          </div>

          {/* Search Type only governs the dense method (similarity/mmr/threshold); BM25
              ignores it entirely, so it's hidden rather than shown as a dead control. */}
          {(value.searchMethod === 'dense' || value.searchMethod === 'hybrid') && (
            <div>
              <label htmlFor="search-type-select" className="block text-sm text-gray-700 mb-1">Search Type</label>
              <select
                id="search-type-select"
                value={value.searchType}
                disabled={disabled}
                onChange={(e) => handleSearchTypeChange(e.target.value as SearchControlsValue['searchType'])}
                className="w-full px-3 py-2 border border-yellow-300 rounded-lg bg-white text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-yellow-500 disabled:opacity-60"
              >
                <option value="similarity">Cosine Similarity</option>
                <option value="similarity_score_threshold">Score Threshold</option>
                <option value="mmr">Max Marginal Relevance (MMR)</option>
              </select>
            </div>
          )}

          {/* Strategy */}
          <div>
            <label htmlFor="strategy-select" className="block text-sm text-gray-700 mb-1">Strategy</label>
            <select
              id="strategy-select"
              value={value.strategy}
              disabled={disabled}
              onChange={(e) => handleChange({ strategy: e.target.value as SearchControlsValue['strategy'] })}
              className="w-full px-3 py-2 border border-yellow-300 rounded-lg bg-white text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-yellow-500 disabled:opacity-60"
            >
              {STRATEGIES.map((s) => (
                <option key={s.value} value={s.value}>{s.label}</option>
              ))}
            </select>
          </div>

          {/* Rerank options — only when strategy is rerank */}
          {value.strategy === 'rerank' && (
            <>
              <div>
                <label htmlFor="rerank-top-n-input" className="block text-sm text-gray-700 mb-1">
                  Rerank top N
                </label>
                <input
                  id="rerank-top-n-input"
                  type="number"
                  min={1}
                  max={50}
                  value={value.topN}
                  disabled={disabled}
                  onChange={(e) => {
                    const parsed = Number(e.target.value);
                    if (!Number.isNaN(parsed)) {
                      handleChange({ topN: Math.min(50, Math.max(1, parsed)) });
                    }
                  }}
                  className="w-full px-3 py-2 border border-yellow-300 rounded-lg bg-white text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-yellow-500 disabled:opacity-60"
                />
                <p className="text-xs text-gray-500 mt-1">Documents kept after reranking (1-50).</p>
              </div>

              <div>
                <label htmlFor="rerank-similarity-threshold-input" className="block text-sm text-gray-700 mb-1">
                  Rerank similarity threshold
                </label>
                <input
                  id="rerank-similarity-threshold-input"
                  type="number"
                  step={0.01}
                  min={0}
                  max={1}
                  value={value.similarityThreshold ?? ''}
                  placeholder="Optional"
                  disabled={disabled}
                  onChange={(e) => {
                    const raw = e.target.value;
                    handleChange({ similarityThreshold: raw === '' ? null : Number(raw) });
                  }}
                  className="w-full px-3 py-2 border border-yellow-300 rounded-lg bg-white text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-yellow-500 disabled:opacity-60"
                />
                <p className="text-xs text-gray-500 mt-1">Optional minimum similarity score, 0-1, applied after reranking.</p>
              </div>
            </>
          )}

          {/* Score threshold — only when similarity_score_threshold */}
          {(value.searchMethod === 'dense' || value.searchMethod === 'hybrid') && value.searchType === 'similarity_score_threshold' && (
            <div>
              <label className="block text-sm text-gray-700 mb-1">
                Score threshold: <span className="font-semibold">{value.scoreThreshold.toFixed(2)}</span>
              </label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={value.scoreThreshold}
                disabled={disabled}
                onChange={(e) => handleChange({ scoreThreshold: Number(e.target.value) })}
                className="w-full accent-yellow-600 disabled:opacity-60"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>0.0</span>
                <span>1.0</span>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Only documents with score ≥ threshold are returned.
              </p>
            </div>
          )}

          {/* Fetch K + Lambda — only when mmr */}
          {(value.searchMethod === 'dense' || value.searchMethod === 'hybrid') && value.searchType === 'mmr' && (
            <>
              <div>
                <label htmlFor="fetch-k-input" className="block text-sm text-gray-700 mb-1">
                  Candidate pool (fetch_k)
                </label>
                <input
                  id="fetch-k-input"
                  type="number"
                  min={10}
                  max={500}
                  value={value.fetchK}
                  disabled={disabled}
                  onChange={(e) => {
                    const parsed = Number(e.target.value);
                    if (!Number.isNaN(parsed)) {
                      handleChange({ fetchK: Math.min(500, Math.max(10, parsed)) });
                    }
                  }}
                  className="w-full px-3 py-2 border border-yellow-300 rounded-lg bg-white text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-yellow-500 disabled:opacity-60"
                />
              </div>

              <div>
                <label className="block text-sm text-gray-700 mb-1">
                  Diversity (λ): <span className="font-semibold">{value.lambdaMult.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={value.lambdaMult}
                  disabled={disabled}
                  onChange={(e) => handleChange({ lambdaMult: Number(e.target.value) })}
                  className="w-full accent-yellow-600 disabled:opacity-60"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>0.0</span>
                  <span>1.0</span>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  0 = max diversity · 1 = max relevance
                </p>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
