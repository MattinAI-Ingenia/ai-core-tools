import { useEffect, useState } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { Info, Target, Trash2 } from 'lucide-react';
import type { MetadataOperator, SearchFilterMetadataField } from '../playground/SearchFilters';
import { randomId } from '../../utils/randomId';
import CardApp from '../ui/CardApp';
import FormField, { FormSelect } from '../ui/FormField';
import ButtonApp from '../ui/ButtonApp';

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
  /** Effective CHUNK_TOP_K when rag_chunk_top_k is left unset — shown so "deployment default" is a real number, not a guess. */
  chunkTopKDeploymentDefault?: number;
}

const RAG_K_MODES: Array<{ value: RagKMode; labelKey: string }> = [
  { value: 'fixed', labelKey: 'lightrag.ragConfigSection.kModeFixed' },
  { value: 'per_100_chunks', labelKey: 'lightrag.ragConfigSection.kModePer100' },
];

// Not a hard limit — just a heads-up that very high k increases latency/cost. Mirrors the
// backend's soft warning in RagConfigFieldsMixin.validate_rag_k.
const RAG_K_SOFT_MAX = 100;

// Mirrors backend SYSTEM_METADATA_FIELDS — always filterable regardless of metadata_definition.
const SYSTEM_FIELDS = ['name', 'file_type', 'url', 'page'];

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
  chunkTopKDeploymentDefault,
}: Readonly<RagConfigSectionProps>) {
  const { t } = useTranslation('common');
  const fieldOptions = [...SYSTEM_FIELDS, ...metadataFields.map((f) => f.name)];
  const filters = value.rag_fixed_filters;
  const searchTypes: Array<{ value: RagSearchType; label: string }> = [
    { value: 'similarity', label: t('agentFormPage.ragConfig.searchTypes.similarity') },
    { value: 'mmr', label: t('agentFormPage.ragConfig.searchTypes.mmr') },
    { value: 'similarity_score_threshold', label: t('agentFormPage.ragConfig.searchTypes.similarityScoreThreshold') },
  ];
  const operators: Array<{ value: MetadataOperator; label: string }> = [
    { value: '$eq', label: t('agentFormPage.ragConfig.operators.equals') },
    { value: '$ne', label: t('agentFormPage.ragConfig.operators.notEquals') },
    { value: '$gt', label: t('agentFormPage.ragConfig.operators.greaterThan') },
    { value: '$gte', label: t('agentFormPage.ragConfig.operators.greaterThanOrEqual') },
    { value: '$lt', label: t('agentFormPage.ragConfig.operators.lessThan') },
    { value: '$lte', label: t('agentFormPage.ragConfig.operators.lessThanOrEqual') },
    { value: '$in', label: t('agentFormPage.ragConfig.operators.in') },
  ];
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
        { _key: randomId(), field: fieldOptions[0] ?? '', op: '$eq', value: '' },
      ],
    });
  };

  const removeFilter = (index: number) => {
    onChange({ rag_fixed_filters: filters.filter((_, i) => i !== index) });
  };

  return (
    <CardApp
      icon={Target}
      titulo={t('agentFormPage.ragConfig.title')}
      subtitulo={t('agentFormPage.ragConfig.subtitle')}
    >

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <FormSelect
            label={isLightRAG ? t('lightrag.ragConfigSection.entitiesToRetrieve') : t('agentFormPage.ragConfig.documentsToRetrieve')}
            id="rag_k_mode"
            value={value.rag_k_mode}
            options={RAG_K_MODES.map((m) => ({ value: m.value, label: t(m.labelKey) }))}
            onChange={(e) => onChange({ rag_k_mode: e.target.value as RagKMode })}
            className="mb-3"
          />
          <FormField
            id="rag_k"
            type="number"
            min={1}
            value={ragKText}
            helpText={isLightRAG
              ? (value.rag_k_mode === 'per_100_chunks'
                ? t('lightrag.ragConfigSection.kHelpPer100')
                : t('lightrag.ragConfigSection.kHelpFixed'))
              : t('agentFormPage.ragConfig.documentsToRetrieveHelp')}
            onChange={(e) => handleRagKChange(e.target.value)}
            onBlur={handleRagKBlur}
          />
          {value.rag_k > RAG_K_SOFT_MAX && (
            <p className="mt-1 text-sm text-amber-600 flex items-center gap-1">
              <Info className="w-3.5 h-3.5 shrink-0" />
              {t('lightrag.ragConfigSection.highValues')}
            </p>
          )}
        </div>

        {isLightRAG && (
          <div>
            <FormField
              id="rag_chunk_top_k"
              type="number"
              min={1}
              value={chunkTopKText}
              placeholder={
                chunkTopKDeploymentDefault != null ? t('lightrag.ragConfigSection.chunkTopKDefault', { value: chunkTopKDeploymentDefault }) : t('lightrag.ragConfigSection.chunkTopKDeploymentDefault')
              }
              helpText={t('lightrag.ragConfigSection.chunkTopKHelp', { current: chunkTopKDeploymentDefault != null ? t('lightrag.ragConfigSection.chunkTopKCurrently', { value: chunkTopKDeploymentDefault }) : t('lightrag.ragConfigSection.chunkTopKEnv') })}
              onChange={(e) => handleChunkTopKChange(e.target.value)}
              onBlur={handleChunkTopKBlur}
            />
          </div>
        )}

        <div>
          <FormField
            label={t('agentFormPage.ragConfig.maxRetrievalCalls')}
            id="rag_max_retrieval_calls"
            type="number"
            min={1}
            max={20}
            value={value.rag_max_retrieval_calls ?? ''}
            placeholder={t('agentFormPage.ragConfig.unlimited')}
            helpText={t('agentFormPage.ragConfig.maxRetrievalCallsHelp')}
            onChange={(e) =>
              onChange({
                rag_max_retrieval_calls: e.target.value ? clampInt(e.target.value, 1, 20, 4) : null,
              })
            }
          />
        </div>

        {/* search_type/score_threshold/fixed_filters are ignored by LightRAGStore — it only reads
            k and lightrag_query_mode. Show the LightRAG query mode picker instead. */}
        {isLightRAG ? (
          <div className="md:col-span-2">
            <FormSelect
              label={t('lightrag.ragConfigSection.queryMode')}
              id="lightrag_query_mode"
              value={lightragQueryMode ?? 'skill-routed'}
              options={(lightragQueryModes ?? ['skill-routed', 'local', 'global', 'hybrid', 'mix', 'naive', 'bypass']).map((mode) => ({
                value: mode,
                label: mode === 'skill-routed' ? t('lightrag.ragConfigSection.skillRouted') : mode === 'hybrid' ? t('lightrag.ragConfigSection.hybridDefault', { mode }) : mode,
              }))}
              onChange={(e) => onLightragQueryModeChange?.(e.target.value)}
            />
            <p className="text-xs text-fg-tertiary dark:text-fg-tertiary-dark mt-1">
              {t('lightrag.ragConfigSection.modeLegend')}
            </p>
            {(lightragQueryMode ?? 'skill-routed') === 'skill-routed' && (
              <div className="mt-3 p-3 bg-purple-50 border border-purple-200 rounded-lg flex items-start gap-2 dark:bg-purple-900/20 dark:border-purple-800">
                <span className="text-purple-600 mt-0.5 text-sm">⚡</span>
                <p className="text-xs text-purple-800 dark:text-purple-200">
                  <Trans
                    i18nKey="lightrag.ragConfigSection.routerBanner"
                    t={t}
                    components={{ strong: <span className="font-semibold" /> }}
                  />
                </p>
              </div>
            )}
          </div>
        ) : (
          <>
            <FormSelect
              label={t('agentFormPage.ragConfig.searchStrategy')}
              id="rag_search_type"
              value={value.rag_search_type}
              options={searchTypes}
              onChange={(e) => onChange({ rag_search_type: e.target.value as RagSearchType })}
            />

            {value.rag_search_type === 'similarity_score_threshold' && (
              <div>
                <FormField
                  label={t('agentFormPage.ragConfig.scoreThreshold')}
                  id="rag_score_threshold"
                  type="text"
                  inputMode="decimal"
                  required
                  value={thresholdText}
                  placeholder="0.75"
                  error={thresholdMissing ? t('agentFormPage.ragConfig.scoreThresholdRequired') : undefined}
                  helpText={thresholdMissing ? undefined : t('agentFormPage.ragConfig.scoreThresholdHelp')}
                  onChange={(e) => handleThresholdChange(e.target.value)}
                  onBlur={handleThresholdBlur}
                />
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
            <h4 className="text-sm font-semibold text-fg dark:text-fg-dark">{t('agentFormPage.ragConfig.fixedMetadataFilters')}</h4>
            <p className="text-xs text-fg-secondary dark:text-fg-secondary-dark">{t('agentFormPage.ragConfig.fixedMetadataFiltersHelp')}</p>
          </div>
          <ButtonApp
            label={t('agentFormPage.ragConfig.addFilter')}
            type="button"
            onClick={addFilter}
            disabled={filters.length >= MAX_FIXED_FILTERS}
            variant="primary"
            size="medium"
          />
        </div>

        {loadingMetadata && <p className="text-sm text-fg-secondary dark:text-fg-secondary-dark">{t('agentFormPage.ragConfig.loadingMetadataFields')}</p>}

        {filters.length > 0 && (
          <div className="space-y-2">
            {filters.map((filter, index) => (
              <div key={filter._key ?? `${filter.field}-${index}`} className="grid grid-cols-12 gap-2 items-start">
                <FormSelect
                  id={`rag-filter-field-${index}`}
                  label={t('agentFormPage.ragConfig.filterField', { row: index + 1 })}
                  labelClassName="sr-only"
                  value={filter.field}
                  className="col-span-4"
                  options={[
                    ...(!fieldOptions.includes(filter.field) && filter.field
                      ? [{ value: filter.field, label: filter.field }]
                      : []),
                    ...fieldOptions.map((name) => ({ value: name, label: name })),
                  ]}
                  onChange={(e) => updateFilter(index, { field: e.target.value })}
                />

                <FormSelect
                  id={`rag-filter-operator-${index}`}
                  label={t('agentFormPage.ragConfig.filterOperator', { row: index + 1 })}
                  labelClassName="sr-only"
                  value={filter.op}
                  className="col-span-3"
                  options={operators}
                  onChange={(e) => {
                    const op = e.target.value as MetadataOperator;
                    updateFilter(index, { op, value: inputToFilterValue(op, filterValueToInput(filter.value)) });
                  }}
                />

                <FormField
                  id={`rag-filter-value-${index}`}
                  type="text"
                  ariaLabel={t('agentFormPage.ragConfig.filterValue', { row: index + 1 })}
                  value={filterValueToInput(filter.value)}
                  placeholder={filter.op === '$in'
                    ? t('agentFormPage.ragConfig.valuesAnyPlaceholder')
                    : t('agentFormPage.ragConfig.valuePlaceholder')}
                  className="col-span-4"
                  inputClassName="text-sm"
                  onChange={(e) => updateFilter(index, { value: inputToFilterValue(filter.op, e.target.value) })}
                />

                <button
                  type="button"
                  onClick={() => removeFilter(index)}
                  title={t('agentFormPage.ragConfig.removeFilter')}
                  aria-label={t('agentFormPage.ragConfig.removeFilterRow', { row: index + 1 })}
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
    </CardApp>
  );
}

export default RagConfigSection;
