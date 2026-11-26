import { useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { FormSelect } from '../ui/FormField';

export interface SearchControlsValue {
  limit: number;
  searchType: 'similarity' | 'similarity_score_threshold' | 'mmr';
  scoreThreshold: number;
  fetchK: number;
  lambdaMult: number;
  searchMethod: 'dense' | 'bm25' | 'hybrid';
  strategy: '' | 'rerank' | 'cross_encoder_rerank';
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

interface SearchControlsProps {
  siloId: string | number;
  value: SearchControlsValue;
  onChange: (v: SearchControlsValue) => void;
  disabled?: boolean;
}

function storageKey(siloId: string | number): string {
  return `silo-search-controls-${siloId}`;
}

export default function SearchControls({ siloId, value, onChange, disabled }: Readonly<SearchControlsProps>) {
  const { t } = useTranslation('common');

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

  const labelClass = 'mb-2 flex justify-between gap-3 text-[13px] font-medium text-fg dark:text-fg-dark';
  const hintClass = 'font-normal text-fg-tertiary dark:text-fg-tertiary-dark';
  const rangeClass = 'w-full accent-ink disabled:opacity-60 dark:accent-ink-dark';
  const scaleClass = 'flex justify-between text-[11.5px] text-fg-tertiary dark:text-fg-tertiary-dark';
  const helpClass = 'mt-1.5 text-[12.5px] text-fg-tertiary dark:text-fg-tertiary-dark';
  const fieldClass = 'rounded-lg border border-line-strong bg-surface px-3 py-2.5 text-sm text-fg outline-none focus:border-accent disabled:opacity-60 dark:border-line-strong-dark dark:bg-surface-dark dark:text-fg-dark dark:focus:border-accent-dark';

  return (
    <div className="flex flex-col gap-5">
      {/* Top-K slider */}
      <div>
        <label htmlFor="top-k-range" className={labelClass}>
          <span>{t('searchControls.topKResults')}</span>
          <span>{value.limit}</span>
        </label>
        <input
          id="top-k-range"
          type="range"
          min={1}
          max={200}
          step={1}
          value={value.limit}
          disabled={disabled}
          onChange={(e) => handleChange({ limit: Number(e.target.value) })}
          className={rangeClass}
        />
        <div className={scaleClass}>
          <span>1</span>
          <span>200</span>
        </div>
      </div>

      {/* Search method */}
      <FormSelect
        label={t('searchControls.searchMethod')}
        id="search-method-select"
        value={value.searchMethod}
        disabled={disabled}
        onChange={(e) => handleChange({ searchMethod: e.target.value as SearchControlsValue['searchMethod'] })}
        options={[
          { value: 'dense', label: t('searchControls.searchMethods.dense') },
          { value: 'bm25', label: t('searchControls.searchMethods.bm25') },
          { value: 'hybrid', label: t('searchControls.searchMethods.hybrid') },
        ]}
      />

      {/* Search type only governs the dense method; BM25 ignores it, so it is hidden. */}
      {value.searchMethod !== 'bm25' && (
        <FormSelect
          label={t('searchControls.searchStrategy')}
          id="search-type-select"
          value={value.searchType}
          disabled={disabled}
          onChange={(e) => handleSearchTypeChange(e.target.value as SearchControlsValue['searchType'])}
          options={[
            { value: 'similarity', label: t('searchControls.searchTypes.similarity') },
            { value: 'similarity_score_threshold', label: t('searchControls.searchTypes.scoreThreshold') },
            { value: 'mmr', label: t('searchControls.searchTypes.mmr') },
          ]}
        />
      )}

      {/* Rerank strategy and its tuning */}
      <FormSelect
        label={t('searchControls.rerankStrategy')}
        id="strategy-select"
        value={value.strategy}
        disabled={disabled}
        onChange={(e) => handleChange({ strategy: e.target.value as SearchControlsValue['strategy'] })}
        options={[
          { value: '', label: t('searchControls.strategies.none') },
          { value: 'rerank', label: t('searchControls.strategies.rerank') },
          { value: 'cross_encoder_rerank', label: t('searchControls.strategies.crossEncoder') },
        ]}
      />

      {(value.strategy === 'rerank' || value.strategy === 'cross_encoder_rerank') && (
        <div>
          <label htmlFor="rerank-top-n-input" className={labelClass}>
            <span>{t('searchControls.rerankTopN')}</span>
            <span>{value.topN}</span>
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
              if (!Number.isNaN(parsed)) handleChange({ topN: Math.min(50, Math.max(1, parsed)) });
            }}
            className={fieldClass}
          />
          <p className={helpClass}>{t('searchControls.rerankTopNHelp')}</p>
        </div>
      )}

      {value.strategy === 'rerank' && (
        <div>
          <label htmlFor="rerank-similarity-threshold-input" className={labelClass}>
            <span>{t('searchControls.rerankThreshold')}</span>
            <span className={hintClass}>{t('searchControls.optional')}</span>
          </label>
          <input
            id="rerank-similarity-threshold-input"
            type="number"
            step={0.01}
            min={0}
            max={1}
            value={value.similarityThreshold ?? ''}
            placeholder={t('searchControls.optional')}
            disabled={disabled}
            onChange={(e) => {
              const raw = e.target.value;
              handleChange({ similarityThreshold: raw === '' ? null : Number(raw) });
            }}
            className={fieldClass}
          />
          <p className={helpClass}>{t('searchControls.rerankThresholdHelp')}</p>
        </div>
      )}


      {/* Score threshold — only when similarity_score_threshold */}
      {value.searchMethod !== 'bm25' && value.searchType === 'similarity_score_threshold' && (
        <div>
          <label htmlFor="score-threshold-range" className={labelClass}>
            <span>{t('searchControls.scoreThreshold')}</span>
            <span>{value.scoreThreshold.toFixed(2)}</span>
          </label>
          <input
            id="score-threshold-range"
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={value.scoreThreshold}
            disabled={disabled}
            onChange={(e) => handleChange({ scoreThreshold: Number(e.target.value) })}
            className={rangeClass}
          />
          <div className={scaleClass}>
            <span>0.0</span>
            <span>1.0</span>
          </div>
          <p className={helpClass}>
            {t('searchControls.scoreThresholdHelp')}
          </p>
        </div>
      )}

      {/* Fetch K + Lambda — only when mmr */}
      {value.searchMethod !== 'bm25' && value.searchType === 'mmr' && (
        <>
          <div>
            <label htmlFor="fetch-k-input" className={labelClass}>
              <span>{t('searchControls.candidatePool')} <span className={hintClass}>· fetch_k</span></span>
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
              className={`w-40 ${fieldClass}`}
            />
          </div>

          <div>
            <label htmlFor="lambda-mult-range" className={labelClass}>
              <span>{t('searchControls.diversity')} <span className={hintClass}>· λ</span></span>
              <span>{value.lambdaMult.toFixed(2)}</span>
            </label>
            <input
              id="lambda-mult-range"
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={value.lambdaMult}
              disabled={disabled}
              onChange={(e) => handleChange({ lambdaMult: Number(e.target.value) })}
              className={rangeClass}
            />
            <div className={scaleClass}>
              <span>0.0</span>
              <span>1.0</span>
            </div>
            <p className={helpClass}>
              0 = {t('searchControls.maxDiversity')} · 1 = {t('searchControls.maxRelevance')}
            </p>
          </div>
        </>
      )}
    </div>
  );
}
