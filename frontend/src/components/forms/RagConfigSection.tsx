import { useEffect, useState } from 'react';
import { Info, Target, Trash2 } from 'lucide-react';
import type { MetadataOperator, SearchFilterMetadataField } from '../playground/SearchFilters';

export type RagSearchType = 'similarity' | 'mmr' | 'similarity_score_threshold';
export type RagKMode = 'fixed' | 'per_100_chunks';

export interface RagFixedFilter {
  field: string;
  op: MetadataOperator;
  value: unknown;
  _key?: string;
}

export interface RagConfigValue {
  rag_k: number;
  rag_k_mode: RagKMode;
  rag_search_type: RagSearchType;
  rag_score_threshold: number | null;
  rag_max_retrieval_calls: number | null;
  // LightRAG only: text chunks per search (LightRAG's chunk_top_k), distinct
  // from rag_k which is entities/relations (LightRAG's top_k) for that store.
  // null = defer to the deployment's CHUNK_TOP_K env default.
  rag_chunk_top_k: number | null;
  rag_fixed_filters: RagFixedFilter[];
}

interface RagConfigSectionProps {
  value: RagConfigValue;
  onChange: (patch: Partial<RagConfigValue>) => void;
  metadataFields: SearchFilterMetadataField[];
  loadingMetadata?: boolean;
  // LightRAG silos don't support search_type/threshold/filters — see backend/tools/vector_stores/lightrag_store.py.
  // Instead they get their own query-mode knob, integrated into this same card.
  isLightRAG?: boolean;
  lightragQueryMode?: string | null;
  lightragQueryModes?: string[];
  onLightragQueryModeChange?: (mode: string) => void;
}

const RAG_K_MODES: Array<{ value: RagKMode; label: string }> = [
  { value: 'fixed', label: 'Fixed K' },
  { value: 'per_100_chunks', label: 'K per 100 chunks' },
];

// Not a hard limit — just a heads-up that very high k increases latency/cost. Mirrors the
// backend's soft warning in RagConfigFieldsMixin.validate_rag_k.
const RAG_K_SOFT_MAX = 100;

// Mirrors backend SYSTEM_METADATA_FIELDS — always filterable regardless of metadata_definition.
const SYSTEM_FIELDS = ['name', 'file_type', 'url', 'page'];

const SEARCH_TYPES: Array<{ value: RagSearchType; label: string }> = [
  { value: 'similarity', label: 'Similarity (default)' },
  { value: 'mmr', label: 'MMR — diverse results' },
  { value: 'similarity_score_threshold', label: 'Similarity + score threshold' },
];

const OPERATORS: Array<{ value: MetadataOperator; label: string }> = [
  { value: '$eq', label: 'equals' },
  { value: '$ne', label: 'not equals' },
  { value: '$gt', label: 'greater than' },
  { value: '$gte', label: 'greater or equal' },
  { value: '$lt', label: 'less than' },
  { value: '$lte', label: 'less or equal' },
  { value: '$in', label: 'in (comma-separated)' },
];

const MAX_FIXED_FILTERS = 10;

// Shared between the inline field error and the form-level submit guard.
export const SCORE_THRESHOLD_REQUIRED_MSG =
  'A score threshold (0–1) is required for the "Similarity + score threshold" strategy.';

function clampInt(raw: string, min: number, max: number, fallback: number): number {
  const n = Number.parseInt(raw, 10);
  if (Number.isNaN(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

/** Render a stored filter value for editing: $in arrays become a comma-separated string. */
function filterValueToInput(value: unknown): string {
  if (Array.isArray(value)) return value.join(', ');
  if (value === null || value === undefined) return '';
  return String(value);
}

/** Convert the edited string back to the stored shape ($in → list, others → scalar string). */
function inputToFilterValue(op: MetadataOperator, raw: string): unknown {
  if (op === '$in') {
    return raw.split(',').map((v) => v.trim()).filter(Boolean);
  }
  return raw;
}

/**
 * Per-agent RAG retrieval configuration. Only meaningful when the agent has a silo;
 * the caller renders this conditionally. Values are persisted on the Agent and resolved
 * at execution time (caller > agent > system).
 */
function RagConfigSection({
  value,
  onChange,
  metadataFields,
  loadingMetadata = false,
  isLightRAG = false,
  lightragQueryMode,
  lightragQueryModes,
  onLightragQueryModeChange,
}: Readonly<RagConfigSectionProps>) {
  const fieldOptions = [...SYSTEM_FIELDS, ...metadataFields.map((f) => f.name)];
  const filters = value.rag_fixed_filters;
  const thresholdMissing =
    value.rag_search_type === 'similarity_score_threshold' && value.rag_score_threshold == null;

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

  // Same local-text-buffer pattern as the threshold above: a plain clamp-on-change
  // (as this used to do) can never reach an empty string, because clamping NaN
  // (from "") falls back to the *previous* value, which re-renders the input with
  // that value still in it — so backspacing the last digit was a no-op and typing
  // a new number from scratch was impossible.
  const [ragKText, setRagKText] = useState(String(value.rag_k));
  useEffect(() => {
    const parsed = ragKText.trim() === '' ? NaN : Number.parseInt(ragKText, 10);
    if (parsed !== value.rag_k) setRagKText(String(value.rag_k));
  }, [value.rag_k]);

  const handleRagKChange = (raw: string) => {
    setRagKText(raw);
    const n = Number.parseInt(raw, 10);
    if (Number.isFinite(n)) onChange({ rag_k: n });
  };

  const handleRagKBlur = () => {
    const clamped = Math.max(1, value.rag_k);
    onChange({ rag_k: clamped });
    setRagKText(String(clamped));
  };

  // rag_chunk_top_k is nullable (empty = defer to the deployment's CHUNK_TOP_K
  // env default), so unlike rag_k the empty string is itself a valid committed
  // state, not just a mid-edit one.
  const [chunkTopKText, setChunkTopKText] = useState(
    value.rag_chunk_top_k == null ? '' : String(value.rag_chunk_top_k),
  );
  useEffect(() => {
    const parsed = chunkTopKText.trim() === '' ? null : Number.parseInt(chunkTopKText, 10);
    const normalized = Number.isFinite(parsed as number) ? parsed : null;
    if ((value.rag_chunk_top_k ?? null) !== normalized) {
      setChunkTopKText(value.rag_chunk_top_k == null ? '' : String(value.rag_chunk_top_k));
    }
  }, [value.rag_chunk_top_k]);

  const handleChunkTopKChange = (raw: string) => {
    setChunkTopKText(raw);
    if (raw.trim() === '') {
      onChange({ rag_chunk_top_k: null });
      return;
    }
    const n = Number.parseInt(raw, 10);
    if (Number.isFinite(n)) onChange({ rag_chunk_top_k: n });
  };

  const handleChunkTopKBlur = () => {
    if (value.rag_chunk_top_k == null) return;
    const clamped = Math.max(1, value.rag_chunk_top_k);
    if (clamped !== value.rag_chunk_top_k) onChange({ rag_chunk_top_k: clamped });
    setChunkTopKText(String(clamped));
  };

  const updateFilter = (index: number, patch: Partial<RagFixedFilter>) => {
    onChange({
      rag_fixed_filters: filters.map((f, i) => (i === index ? { ...f, ...patch } : f)),
    });
  };

  const addFilter = () => {
    if (filters.length >= MAX_FIXED_FILTERS) return;
    onChange({
      rag_fixed_filters: [
        ...filters,
        { _key: Math.random().toString(36).slice(2), field: fieldOptions[0] ?? '', op: '$eq', value: '' },
      ],
    });
  };

  const removeFilter = (index: number) => {
    onChange({ rag_fixed_filters: filters.filter((_, i) => i !== index) });
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
          <label htmlFor="rag_k_mode" className="block text-sm font-medium text-gray-700 mb-2">
            {isLightRAG ? 'Entities to retrieve' : 'Documents to retrieve'}
          </label>
          <select
            id="rag_k_mode"
            value={value.rag_k_mode}
            onChange={(e) => onChange({ rag_k_mode: e.target.value as RagKMode })}
            className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 mb-3"
          >
            {RAG_K_MODES.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
          <input
            id="rag_k"
            type="number"
            min={1}
            value={ragKText}
            onChange={(e) => handleRagKChange(e.target.value)}
            onBlur={handleRagKBlur}
            className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
          />
          <p className="text-xs text-gray-500 mt-1">
            {isLightRAG
              ? (value.rag_k_mode === 'per_100_chunks'
                ? 'Entities/relations fetched per 100 chunks indexed in the silo (LightRAG\'s top_k) — scales with knowledge-base size.'
                : 'Number of entities/relations from the knowledge graph considered per search (LightRAG\'s top_k) — not text chunks.')
              : (value.rag_k_mode === 'per_100_chunks'
                ? 'Chunks fetched per 100 chunks indexed in the silo — scales with knowledge-base size.'
                : 'Number of chunks fetched per search.')}
          </p>
          {value.rag_k > RAG_K_SOFT_MAX && (
            <p className="mt-1 text-sm text-amber-600 flex items-center gap-1">
              <Info className="w-3.5 h-3.5 shrink-0" />
              High values increase latency and cost.
            </p>
          )}
        </div>

        {isLightRAG && (
          <div>
            <label htmlFor="rag_chunk_top_k" className="block text-sm font-medium text-gray-700 mb-2">
              Documents to retrieve
            </label>
            <input
              id="rag_chunk_top_k"
              type="number"
              min={1}
              value={chunkTopKText}
              placeholder="Deployment default"
              onChange={(e) => handleChunkTopKChange(e.target.value)}
              onBlur={handleChunkTopKBlur}
              className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
            />
            <p className="text-xs text-gray-500 mt-1">
              Number of text chunks fetched per search (LightRAG's chunk_top_k). Empty uses the
              deployment's default (CHUNK_TOP_K env setting).
            </p>
          </div>
        )}

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

        {/* search_type/score_threshold/fixed_filters are ignored by LightRAGStore — it only reads
            k and lightrag_query_mode. Show the LightRAG query mode picker instead. */}
        {isLightRAG ? (
          <div className="md:col-span-2">
            <label htmlFor="lightrag_query_mode" className="block text-sm font-medium text-gray-700 mb-2">
              LightRAG Query Mode
            </label>
            <select
              id="lightrag_query_mode"
              value={lightragQueryMode ?? 'skill-routed'}
              onChange={(e) => onLightragQueryModeChange?.(e.target.value)}
              className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-green-500 transition-all duration-200"
            >
              {(lightragQueryModes ?? ['skill-routed', 'local', 'global', 'hybrid', 'mix', 'naive', 'bypass']).map((mode) => (
                <option key={mode} value={mode}>
                  {mode === 'skill-routed' ? 'Skill-Routed (auto)' : mode === 'hybrid' ? `${mode} (default fallback)` : mode}
                </option>
              ))}
            </select>
            <p className="text-xs text-gray-500 mt-1">
              skill-routed = agent picks mode per question · local = entity neighbors · global = community summaries ·
              hybrid = local + global · mix = all strategies · naive = vector-only · bypass = skip retrieval
            </p>
            {(lightragQueryMode ?? 'skill-routed') === 'skill-routed' && (
              <div className="mt-3 p-3 bg-purple-50 border border-purple-200 rounded-lg flex items-start gap-2">
                <span className="text-purple-600 mt-0.5 text-sm">⚡</span>
                <p className="text-xs text-purple-800">
                  <span className="font-semibold">LightRAG Query Router</span> — the agent will automatically
                  select the best retrieval strategy (local / global / hybrid / mix / naive) per question.
                  A routing skill will be added to this agent on save.
                </p>
              </div>
            )}
          </div>
        ) : (
          <>
            <div>
              <label htmlFor="rag_search_type" className="block text-sm font-medium text-gray-700 mb-2">
                Search strategy
              </label>
              <select
                id="rag_search_type"
                value={value.rag_search_type}
                onChange={(e) => onChange({ rag_search_type: e.target.value as RagSearchType })}
                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
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
                <label htmlFor="rag_score_threshold" className="block text-sm font-medium text-gray-700 mb-2">
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
                  className={`w-full px-4 py-3 border rounded-xl focus:ring-2 transition-all duration-200 ${
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
          </>
        )}
      </div>

      {/* Fixed filters — always-applied scoping the caller cannot loosen. Not supported by LightRAG. */}
      {!isLightRAG && (
      <div className="mt-8">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h4 className="text-sm font-semibold text-gray-900">Fixed metadata filters</h4>
            <p className="text-xs text-gray-500">Applied to every search. Leave empty to search the whole knowledge base.</p>
          </div>
          <button
            type="button"
            onClick={addFilter}
            disabled={filters.length >= MAX_FIXED_FILTERS}
            className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white text-sm rounded-lg transition-colors"
          >
            Add filter
          </button>
        </div>

        {loadingMetadata && <p className="text-sm text-gray-500">Loading metadata fields…</p>}

        {filters.length > 0 && (
          <div className="space-y-2">
            {filters.map((filter, index) => (
              <div key={filter._key ?? `${filter.field}-${index}`} className="grid grid-cols-12 gap-2 items-start">
                <select
                  aria-label={`Filter field, row ${index + 1}`}
                  value={filter.field}
                  onChange={(e) => updateFilter(index, { field: e.target.value })}
                  className="col-span-4 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {!fieldOptions.includes(filter.field) && filter.field && (
                    <option value={filter.field}>{filter.field}</option>
                  )}
                  {fieldOptions.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>

                <select
                  aria-label={`Filter operator, row ${index + 1}`}
                  value={filter.op}
                  onChange={(e) => {
                    const op = e.target.value as MetadataOperator;
                    updateFilter(index, { op, value: inputToFilterValue(op, filterValueToInput(filter.value)) });
                  }}
                  className="col-span-3 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {OPERATORS.map((op) => (
                    <option key={op.value} value={op.value}>
                      {op.label}
                    </option>
                  ))}
                </select>

                <input
                  type="text"
                  aria-label={`Filter value, row ${index + 1}`}
                  value={filterValueToInput(filter.value)}
                  placeholder={filter.op === '$in' ? 'a, b, c' : 'value'}
                  onChange={(e) => updateFilter(index, { value: inputToFilterValue(filter.op, e.target.value) })}
                  className="col-span-4 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                />

                <button
                  type="button"
                  onClick={() => removeFilter(index)}
                  title="Remove filter"
                  aria-label={`Remove filter, row ${index + 1}`}
                  className="col-span-1 flex justify-center text-red-600 hover:text-red-800 transition-colors p-2 rounded focus:outline-none focus:ring-2 focus:ring-red-500"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
      )}
    </div>
  );
}

export default RagConfigSection;
