import { useEffect, useState } from 'react';
import { Target } from 'lucide-react';

export type RagSearchType = 'similarity' | 'mmr' | 'similarity_score_threshold';
// Atomic search methods only. Combining dense + bm25 (what used to be the single
// value "hybrid") is now expressed by selecting both — the pipeline fuses them via
// Reciprocal Rank Fusion.
export type RagSearchMethod = 'dense' | 'bm25';

export interface RagConfigValue {
  rag_k: number;
  rag_search_type: RagSearchType;
  rag_score_threshold: number | null;
  rag_max_retrieval_calls: number | null;
  rag_search_method: RagSearchMethod[];
  rag_strategy: string[] | null;
  rag_rerank_top_n: number | null;
  rag_rerank_similarity_threshold: number | null;
}

interface RagConfigSectionProps {
  value: RagConfigValue;
  onChange: (patch: Partial<RagConfigValue>) => void;
}

const SEARCH_TYPES: Array<{ value: RagSearchType; label: string }> = [
  { value: 'similarity', label: 'Similarity (default)' },
  { value: 'mmr', label: 'MMR — diverse results' },
  { value: 'similarity_score_threshold', label: 'Similarity + score threshold' },
];

const SEARCH_METHODS: Array<{ value: RagSearchMethod; label: string }> = [
  { value: 'dense', label: 'Dense (embeddings similarity)' },
  { value: 'bm25', label: 'BM25 (lexical / keyword)' },
];

const STRATEGIES: Array<{ value: 'rerank' | 'cross_encoder_rerank'; label: string }> = [
  { value: 'rerank', label: 'Rerank (embeddings-based)' },
  { value: 'cross_encoder_rerank', label: 'Rerank (cross-encoder)' },
];

// Mutually exclusive — see toggleStrategy below.
const RERANK_STRATEGIES = new Set<string>(['rerank', 'cross_encoder_rerank']);

// Shared between the inline field error and the form-level submit guard.
export const SCORE_THRESHOLD_REQUIRED_MSG =
  'A score threshold (0–1) is required for the "Similarity + score threshold" strategy.';

function clampInt(raw: string, min: number, max: number, fallback: number): number {
  const n = Number.parseInt(raw, 10);
  if (Number.isNaN(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

/**
 * Per-agent RAG retrieval configuration. Always rendered — the tuning knobs (k, search
 * method/type, strategy) can be pre-configured before a silo is assigned. Values are
 * persisted on the Agent and resolved at execution time (caller > agent > system).
 */
function RagConfigSection({ value, onChange }: Readonly<RagConfigSectionProps>) {
  // Search Type only governs the dense component of retrieval (similarity/mmr/threshold);
  // BM25 ignores it entirely, so it's hidden rather than shown as a dead control.
  const searchTypeApplies = value.rag_search_method.includes('dense');

  const toggleSearchMethod = (method: RagSearchMethod, checked: boolean) => {
    const next = checked
      ? [...value.rag_search_method, method]
      : value.rag_search_method.filter((m) => m !== method);
    // At least one method must remain selected.
    if (next.length > 0) onChange({ rag_search_method: next });
  };

  const toggleStrategy = (strategy: string, checked: boolean) => {
    const current = value.rag_strategy ?? [];
    // 'rerank' and 'cross_encoder_rerank' both re-score/truncate to top_n — chaining them
    // would silently double-rerank with no way to control the order, so only one may be
    // active at a time (mirrors the backend rejection in validate_rag_strategy).
    const withoutOtherRerankers = checked
      ? current.filter((s) => !RERANK_STRATEGIES.has(s) || s === strategy)
      : current;
    const next = checked
      ? [...withoutOtherRerankers, strategy]
      : withoutOtherRerankers.filter((s) => s !== strategy);
    onChange({ rag_strategy: next.length > 0 ? next : null });
  };

  const thresholdMissing =
    searchTypeApplies && value.rag_search_type === 'similarity_score_threshold' && value.rag_score_threshold == null;

  // Local text buffer for the decimal threshold: lets the user type "0,6"/"0." freely
  // (comma normalized to dot) without the controlled-number-input swallowing keystrokes.
  const [thresholdText, setThresholdText] = useState(
    value.rag_score_threshold == null ? '' : String(value.rag_score_threshold),
  );

  // Re-sync the buffer when the value changes from outside (agent load, strategy switch).
  useEffect(() => {
    const parsed = thresholdText.trim() === '' ? null : Number.parseFloat(thresholdText.replace(',', '.'));
    const normalized = Number.isFinite(parsed as number) ? parsed : null;
    if ((value.rag_score_threshold ?? null) !== normalized) {
      setThresholdText(value.rag_score_threshold == null ? '' : String(value.rag_score_threshold));
    }
  }, [value.rag_score_threshold]);

  const handleThresholdChange = (raw: string) => {
    setThresholdText(raw);
    const n = Number.parseFloat(raw.replace(',', '.'));
    onChange({ rag_score_threshold: Number.isFinite(n) ? n : null });
  };

  const handleThresholdBlur = () => {
    if (value.rag_score_threshold == null) return;
    const clamped = Math.min(1, Math.max(0, value.rag_score_threshold));
    if (clamped !== value.rag_score_threshold) onChange({ rag_score_threshold: clamped });
    setThresholdText(String(clamped));
  };

  // Local text buffer for the decimal rerank similarity threshold, mirroring rag_score_threshold's
  // comma-decimal handling above.
  const [rerankThresholdText, setRerankThresholdText] = useState(
    value.rag_rerank_similarity_threshold == null ? '' : String(value.rag_rerank_similarity_threshold),
  );

  useEffect(() => {
    const parsed =
      rerankThresholdText.trim() === '' ? null : Number.parseFloat(rerankThresholdText.replace(',', '.'));
    const normalized = Number.isFinite(parsed as number) ? parsed : null;
    if ((value.rag_rerank_similarity_threshold ?? null) !== normalized) {
      setRerankThresholdText(
        value.rag_rerank_similarity_threshold == null ? '' : String(value.rag_rerank_similarity_threshold),
      );
    }
  }, [value.rag_rerank_similarity_threshold]);

  const handleRerankThresholdChange = (raw: string) => {
    setRerankThresholdText(raw);
    const n = Number.parseFloat(raw.replace(',', '.'));
    onChange({ rag_rerank_similarity_threshold: Number.isFinite(n) ? n : null });
  };

  const handleRerankThresholdBlur = () => {
    if (value.rag_rerank_similarity_threshold == null) return;
    const clamped = Math.min(1, Math.max(0, value.rag_rerank_similarity_threshold));
    if (clamped !== value.rag_rerank_similarity_threshold) onChange({ rag_rerank_similarity_threshold: clamped });
    setRerankThresholdText(String(clamped));
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-8">
      <div className="flex items-center mb-6">
        <div className="w-10 h-10 bg-indigo-100 rounded-xl flex items-center justify-center mr-4">
          <Target className="w-5 h-5 text-indigo-600" aria-hidden="true" />
        </div>
        <div>
          <h3 className="text-xl font-semibold text-gray-900">Retrieval (RAG)</h3>
          <p className="text-sm text-gray-500">How this agent searches its knowledge base</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label htmlFor="rag_k" className="block text-sm font-medium text-gray-700 mb-2">
            Documents to retrieve (k)
          </label>
          <input
            id="rag_k"
            type="number"
            min={1}
            max={100}
            value={value.rag_k}
            onChange={(e) => onChange({ rag_k: clampInt(e.target.value, 1, 100, value.rag_k) })}
            className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
          />
          <p className="text-xs text-gray-500 mt-1">Number of chunks fetched per search (1–100).</p>
        </div>

        <div>
          <label htmlFor="rag_max_retrieval_calls" className="block text-sm font-medium text-gray-700 mb-2">
            Max retrieval calls per turn
          </label>
          <input
            id="rag_max_retrieval_calls"
            type="number"
            min={1}
            max={20}
            value={value.rag_max_retrieval_calls ?? ''}
            placeholder="Unlimited"
            onChange={(e) =>
              onChange({
                rag_max_retrieval_calls: e.target.value ? clampInt(e.target.value, 1, 20, 4) : null,
              })
            }
            className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
          />
          <p className="text-xs text-gray-500 mt-1">Caps how often the agent can search in one turn (1–20). Empty = unlimited.</p>
        </div>

        <div>
          <span className="block text-sm font-medium text-gray-700 mb-2">Search Method</span>
          <div className="space-y-2">
            {SEARCH_METHODS.map((m) => {
              const checked = value.rag_search_method.includes(m.value);
              const isLastChecked = checked && value.rag_search_method.length === 1;
              return (
                <div key={m.value}>
                  <label htmlFor={`rag_search_method_${m.value}`} className="flex items-center gap-2">
                    <input
                      id={`rag_search_method_${m.value}`}
                      type="checkbox"
                      checked={checked}
                      disabled={isLastChecked}
                      onChange={(e) => toggleSearchMethod(m.value, e.target.checked)}
                      className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700">{m.label}</span>
                  </label>

                  {/* Search Type only governs the dense component of retrieval, so it's nested
                      under the Dense checkbox instead of shown as an unrelated top-level field. */}
                  {m.value === 'dense' && checked && (
                    <div className="ml-6 mt-2 pl-3 border-l-2 border-gray-100 space-y-2">
                      <div className="flex items-center gap-2">
                        <label htmlFor="rag_search_type" className="text-xs text-gray-500 whitespace-nowrap">
                          Search type
                        </label>
                        <select
                          id="rag_search_type"
                          value={value.rag_search_type}
                          onChange={(e) => onChange({ rag_search_type: e.target.value as RagSearchType })}
                          className="flex-1 px-2 py-1.5 text-xs border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                        >
                          {SEARCH_TYPES.map((t) => (
                            <option key={t.value} value={t.value}>
                              {t.label}
                            </option>
                          ))}
                        </select>
                      </div>

                      {value.rag_search_type === 'similarity_score_threshold' && (
                        <div>
                          <label htmlFor="rag_score_threshold" className="flex items-center text-xs text-gray-600 mb-1">
                            Score threshold
                            <span className="text-red-500 ml-1" aria-hidden="true">*</span>
                            <span className="sr-only"> (required)</span>
                          </label>
                          <input
                            id="rag_score_threshold"
                            type="text"
                            inputMode="decimal"
                            required
                            value={thresholdText}
                            placeholder="0.75"
                            aria-invalid={thresholdMissing}
                            aria-describedby="rag_score_threshold_help"
                            onChange={(e) => handleThresholdChange(e.target.value)}
                            onBlur={handleThresholdBlur}
                            className={`w-full px-2 py-1.5 text-xs border rounded-lg focus:ring-2 transition-all duration-200 ${
                              thresholdMissing
                                ? 'border-red-300 focus:ring-red-500 focus:border-red-500'
                                : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500'
                            }`}
                          />
                          {thresholdMissing ? (
                            <p id="rag_score_threshold_help" role="alert" className="text-xs text-red-600 mt-1">
                              Required for the threshold strategy (0–1).
                            </p>
                          ) : (
                            <p id="rag_score_threshold_help" className="text-xs text-gray-500 mt-1">
                              Minimum relevance score, 0–1. Lower returns more, higher is stricter.
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
          <p className="text-xs text-gray-500 mt-1">
            How the knowledge base is searched. Select both for hybrid retrieval (dense + lexical, fused via RRF).
          </p>
        </div>

        <div>
          <span className="block text-sm font-medium text-gray-700 mb-2">Strategy</span>
          <div className="space-y-2">
            {STRATEGIES.map((s) => (
              <label key={s.value} htmlFor={`rag_strategy_${s.value}`} className="flex items-center gap-2">
                <input
                  id={`rag_strategy_${s.value}`}
                  type="checkbox"
                  checked={value.rag_strategy?.includes(s.value) ?? false}
                  onChange={(e) => toggleStrategy(s.value, e.target.checked)}
                  className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-sm text-gray-700">{s.label}</span>
              </label>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-1">Optional post-retrieval reordering applied to the search results.</p>
        </div>

        {(value.rag_strategy?.includes('rerank') || value.rag_strategy?.includes('cross_encoder_rerank')) && (
          <div>
            <label htmlFor="rag_rerank_top_n" className="block text-sm font-medium text-gray-700 mb-2">
              Rerank top N
            </label>
            <input
              id="rag_rerank_top_n"
              type="number"
              min={1}
              max={50}
              value={value.rag_rerank_top_n ?? ''}
              placeholder="5"
              onChange={(e) =>
                onChange({
                  rag_rerank_top_n: e.target.value ? clampInt(e.target.value, 1, 50, 5) : null,
                })
              }
              className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
            />
            <p className="text-xs text-gray-500 mt-1">Documents kept after reranking (1–50). Empty uses the server default.</p>
          </div>
        )}

        {value.rag_strategy?.includes('rerank') && (
          <div>
            <label htmlFor="rag_rerank_similarity_threshold" className="block text-sm font-medium text-gray-700 mb-2">
              Rerank similarity threshold
            </label>
            <input
              id="rag_rerank_similarity_threshold"
              type="text"
              inputMode="decimal"
              value={rerankThresholdText}
              placeholder="0.75"
              aria-describedby="rag_rerank_similarity_threshold_help"
              onChange={(e) => handleRerankThresholdChange(e.target.value)}
              onBlur={handleRerankThresholdBlur}
              className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
            />
            <p id="rag_rerank_similarity_threshold_help" className="text-xs text-gray-500 mt-1">
              Optional minimum similarity score, 0–1, applied after reranking.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default RagConfigSection;
