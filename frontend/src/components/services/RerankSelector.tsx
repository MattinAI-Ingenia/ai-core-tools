import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { RerankServiceOptions } from '../../types/services';
import { FormField, FormSelect } from '../ui/FormField';
import { serviceOptionLabel } from '../../utils/serviceLifecycle';
import {
  RERANK_TOP_N_DEFAULT,
  RERANK_TOP_N_MAX,
  RERANK_TOP_N_MIN,
  clampRerankTopN,
  isRerankUnavailable,
  rerankResolves,
  rerankSelectableOptions,
} from '../../utils/rerankSelection';
import RerankEgressNotice from './RerankEgressNotice';

export interface RerankSelection {
  readonly enabled: boolean;
  /** null = inherit the app default, then the platform default. */
  readonly serviceId: number | null;
  readonly topN: number;
}

export interface RerankSelectorProps {
  /** Unique prefix for element ids. */
  readonly idPrefix: string;
  readonly value: RerankSelection;
  readonly onChange: (patch: Partial<RerankSelection>) => void;
  /** Loaded options; null while loading or when loading failed. */
  readonly options: RerankServiceOptions | null;
  /** Saved service of the entity being edited: a retired one stays visible (marked) for that entity only. */
  readonly savedServiceId?: number | null;
  readonly disabled?: boolean;
  /** form: labelled fields; compact: small inline selects for the compare panels. */
  readonly variant?: 'form' | 'compact';
  /** Accessible-label prefix of the compact variant (for example "A"). */
  readonly labelPrefix?: string;
  readonly helpText?: string;
  /** Shows the unavailable message as an error (set by the caller when a stored selection cannot resolve). */
  readonly forceError?: boolean;
}

const COMPACT_SELECT =
  'min-w-0 flex-1 rounded-md border border-line bg-surface px-2 py-1.5 text-xs text-fg outline-none focus:border-accent disabled:opacity-60 dark:border-line-dark dark:bg-surface-dark dark:text-fg-dark dark:focus:border-accent-dark';

/**
 * None / Rerank service selector with the inherited-default option and Top N, shared by the agent
 * RAG section and the silo playground (single search and compare panels).
 */
function RerankSelector({
  idPrefix,
  value,
  onChange,
  options,
  savedServiceId,
  disabled = false,
  variant = 'form',
  labelPrefix = '',
  helpText,
  forceError = false,
}: RerankSelectorProps) {
  const { t } = useTranslation('common');
  const unavailable = isRerankUnavailable(options);
  const unresolved = value.enabled && !rerankResolves(value.serviceId, options);
  const showError = forceError || (value.enabled && unavailable) || unresolved;
  const unavailableMessage = t('rerankControls.unavailable');

  const modeOptions = [
    { value: 'none', label: t('rerankControls.none') },
    { value: 'service', label: t('rerankControls.useService'), disabled: unavailable && !value.enabled },
  ];

  const defaultName = options?.effective_default?.name;
  const serviceOptions = [
    {
      value: '',
      label: defaultName
        ? t('rerankControls.defaultOption', { name: defaultName })
        : t('rerankControls.defaultOptionNone'),
    },
    ...rerankSelectableOptions(options, savedServiceId ?? value.serviceId).map((service) => ({
      value: String(service.service_id),
      label: serviceOptionLabel(
        service,
        `${service.is_system ? t('appSettingsPage.systemServiceOption', { name: service.name }) : service.name} (${service.provider ?? service.model_name})`,
      ),
    })),
  ];

  const handleMode = (mode: string) => {
    if (mode === 'service') onChange({ enabled: true, topN: value.topN || RERANK_TOP_N_DEFAULT });
    else onChange({ enabled: false });
  };
  const handleService = (raw: string) => onChange({ serviceId: raw === '' ? null : Number.parseInt(raw, 10) });
  // Draft text of the Top N input: lets the user clear it and retype; a typed number is clamped at once.
  const [topNDraft, setTopNDraft] = useState<string | null>(null);
  const handleTopN = (raw: string) => {
    if (raw.trim() === '') {
      setTopNDraft('');
      return;
    }
    const clamped = clampRerankTopN(raw, value.topN);
    setTopNDraft(String(clamped));
    onChange({ topN: clamped });
  };
  const topNValue = topNDraft ?? value.topN;
  const resetTopNDraft = () => setTopNDraft(null);

  const errorId = `${idPrefix}-unavailable`;

  if (variant === 'compact') {
    const prefix = labelPrefix ? `${labelPrefix} ` : '';
    return (
      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap gap-2">
          <select
            aria-label={`${prefix}${t('rerankControls.label')}`}
            aria-describedby={showError || unavailable ? errorId : undefined}
            value={value.enabled ? 'service' : 'none'}
            onChange={(e) => handleMode(e.target.value)}
            disabled={disabled}
            className={COMPACT_SELECT}
          >
            {modeOptions.map((o) => (
              <option key={o.value} value={o.value} disabled={o.disabled}>{o.label}</option>
            ))}
          </select>
          {value.enabled && (
            <>
              <select
                aria-label={`${prefix}${t('rerankControls.service')}`}
                value={value.serviceId == null ? '' : String(value.serviceId)}
                onChange={(e) => handleService(e.target.value)}
                disabled={disabled}
                className={COMPACT_SELECT}
              >
                {serviceOptions.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
              <input
                type="number"
                min={RERANK_TOP_N_MIN}
                max={RERANK_TOP_N_MAX}
                aria-label={`${prefix}${t('rerankControls.topN')}`}
                value={topNValue}
                onChange={(e) => handleTopN(e.target.value)}
                onBlur={resetTopNDraft}
                disabled={disabled}
                className={`w-20 ${COMPACT_SELECT}`}
              />
            </>
          )}
        </div>
        {(showError || unavailable) && (
          <p
            id={errorId}
            role={showError ? 'alert' : undefined}
            className={showError ? 'text-xs text-error dark:text-error-dark' : 'text-xs text-fg-tertiary dark:text-fg-tertiary-dark'}
          >
            {unavailableMessage}
          </p>
        )}
        {value.enabled && <RerankEgressNotice />}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <FormSelect
        label={t('rerankControls.label')}
        id={`${idPrefix}-mode`}
        value={value.enabled ? 'service' : 'none'}
        options={modeOptions}
        disabled={disabled}
        error={showError ? unavailableMessage : undefined}
        helpText={showError ? undefined : (unavailable ? unavailableMessage : helpText)}
        onChange={(e) => handleMode(e.target.value)}
      />
      {value.enabled && (
        <>
          <FormSelect
            label={t('rerankControls.service')}
            id={`${idPrefix}-service`}
            value={value.serviceId == null ? '' : String(value.serviceId)}
            options={serviceOptions}
            disabled={disabled}
            onChange={(e) => handleService(e.target.value)}
          />
          <FormField
            label={t('rerankControls.topN')}
            id={`${idPrefix}-top-n`}
            type="number"
            min={RERANK_TOP_N_MIN}
            max={RERANK_TOP_N_MAX}
            value={topNValue}
            disabled={disabled}
            helpText={t('rerankControls.topNHelp')}
            onChange={(e) => handleTopN(e.target.value)}
            onBlur={resetTopNDraft}
          />
          <RerankEgressNotice />
        </>
      )}
    </div>
  );
}

export default RerankSelector;
