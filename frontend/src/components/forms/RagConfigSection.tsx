import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Target, Trash2 } from 'lucide-react';
import type { MetadataOperator, SearchFilterMetadataField } from '../playground/SearchFilters';
import { randomId } from '../../utils/randomId';
import CardApp from '../ui/CardApp';
import FormField, { FormSelect } from '../ui/FormField';
import ButtonApp from '../ui/ButtonApp';

export type RagSearchType = 'similarity' | 'mmr' | 'similarity_score_threshold';
// Atomic search methods. Selecting both fuses them (hybrid) via Reciprocal Rank Fusion.
export type RagSearchMethod = 'dense' | 'bm25';

export interface RagFixedFilter {
  field: string;
  op: MetadataOperator;
  value: unknown;
  _key?: string;
}

export interface RagConfigValue {
  rag_k: number;
  rag_search_type: RagSearchType;
  rag_score_threshold: number | null;
  rag_max_retrieval_calls: number | null;
  rag_fixed_filters: RagFixedFilter[];
  rag_search_method: RagSearchMethod[];
  rag_strategy: string[] | null;
  rag_rerank_top_n: number | null;
  rag_rerank_similarity_threshold: number | null;
}

interface RagConfigSectionProps {
  value: RagConfigValue;
  onChange: (patch: Partial<RagConfigValue>) => void;
  metadataFields: SearchFilterMetadataField[];
  loadingMetadata?: boolean;
}

// Mirrors backend SYSTEM_METADATA_FIELDS — always filterable regardless of metadata_definition.
const SYSTEM_FIELDS = ['name', 'file_type', 'url', 'page'];

const MAX_FIXED_FILTERS = 10;

const SEARCH_METHODS: Array<{ value: RagSearchMethod; labelKey: string }> = [
  { value: 'dense', labelKey: 'searchControls.searchMethods.dense' },
  { value: 'bm25', labelKey: 'searchControls.searchMethods.bm25' },
];

const STRATEGIES: Array<{ value: 'rerank' | 'cross_encoder_rerank'; labelKey: string }> = [
  { value: 'rerank', labelKey: 'searchControls.strategies.rerank' },
  { value: 'cross_encoder_rerank', labelKey: 'searchControls.strategies.crossEncoder' },
];

// Mutually exclusive: both rerankers truncate to top_n, so chaining them would double-rerank.
const RERANK_STRATEGIES = new Set<string>(['rerank', 'cross_encoder_rerank']);

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
function RagConfigSection({ value, onChange, metadataFields, loadingMetadata = false }: Readonly<RagConfigSectionProps>) {
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

  const updateFilter = (index: number, patch: Partial<RagFixedFilter>) => {
    onChange({
      rag_fixed_filters: filters.map((f, i) => (i === index ? { ...f, ...patch } : f)),
    });
  };

  // Search Type only governs the dense component of retrieval; BM25 ignores it.
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
    const withoutOtherRerankers = checked
      ? current.filter((s) => !RERANK_STRATEGIES.has(s) || s === strategy)
      : current;
    const next = checked
      ? [...withoutOtherRerankers, strategy]
      : withoutOtherRerankers.filter((s) => s !== strategy);
    onChange({ rag_strategy: next.length > 0 ? next : null });
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
          <FormField
            label={t('agentFormPage.ragConfig.documentsToRetrieve')}
            id="rag_k"
            type="number"
            min={1}
            max={100}
            value={value.rag_k}
            helpText={t('agentFormPage.ragConfig.documentsToRetrieveHelp')}
            onChange={(e) => onChange({ rag_k: clampInt(e.target.value, 1, 100, value.rag_k) })}
          />
        </div>

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

        {searchTypeApplies && (
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

      {/* Search method and rerank strategy: methods can combine (hybrid), rerankers are exclusive */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
        <fieldset>
          <legend className="block text-sm font-medium text-fg dark:text-fg-dark mb-2">
            {t('agentFormPage.ragConfig.searchMethod')}
          </legend>
          <div className="space-y-2">
            {SEARCH_METHODS.map((m) => {
              const checked = value.rag_search_method.includes(m.value);
              const isLastChecked = checked && value.rag_search_method.length === 1;
              return (
                <label key={m.value} htmlFor={`rag_search_method_${m.value}`} className="flex items-center gap-2">
                  <input
                    id={`rag_search_method_${m.value}`}
                    type="checkbox"
                    checked={checked}
                    disabled={isLastChecked}
                    onChange={(e) => toggleSearchMethod(m.value, e.target.checked)}
                    className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm text-fg dark:text-fg-dark">{t(m.labelKey)}</span>
                </label>
              );
            })}
          </div>
          <p className="text-xs text-fg-secondary dark:text-fg-secondary-dark mt-2">
            {t('agentFormPage.ragConfig.searchMethodHelp')}
          </p>
        </fieldset>

        <fieldset>
          <legend className="block text-sm font-medium text-fg dark:text-fg-dark mb-2">
            {t('agentFormPage.ragConfig.rerankStrategy')}
          </legend>
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
                <span className="text-sm text-fg dark:text-fg-dark">{t(s.labelKey)}</span>
              </label>
            ))}
          </div>
          <p className="text-xs text-fg-secondary dark:text-fg-secondary-dark mt-2">
            {t('agentFormPage.ragConfig.rerankStrategyHelp')}
          </p>
        </fieldset>

        {value.rag_strategy?.includes('rerank') && (
          <FormField
            label={t('agentFormPage.ragConfig.rerankTopN')}
            id="rag_rerank_top_n"
            type="number"
            min={1}
            max={50}
            value={value.rag_rerank_top_n ?? ''}
            placeholder="5"
            helpText={t('agentFormPage.ragConfig.rerankTopNHelp')}
            onChange={(e) =>
              onChange({
                rag_rerank_top_n: e.target.value ? clampInt(e.target.value, 1, 50, 5) : null,
              })
            }
          />
        )}

        {value.rag_strategy?.includes('rerank') && (
          <FormField
            label={t('agentFormPage.ragConfig.rerankThreshold')}
            id="rag_rerank_similarity_threshold"
            type="number"
            step={0.01}
            min={0}
            max={1}
            value={value.rag_rerank_similarity_threshold ?? ''}
            placeholder={t('searchControls.optional')}
            helpText={t('agentFormPage.ragConfig.rerankThresholdHelp')}
            onChange={(e) =>
              onChange({
                rag_rerank_similarity_threshold: e.target.value === '' ? null : Math.min(1, Math.max(0, Number(e.target.value))),
              })
            }
          />
        )}
      </div>

      {/* Fixed filters — always-applied scoping the caller cannot loosen */}
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
    </CardApp>
  );
}

export default RagConfigSection;
