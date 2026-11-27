import { useTranslation } from 'react-i18next';
import type { RerankStatus } from '../../services/api';

export interface RerankStatusLineProps {
  readonly status: RerankStatus | null | undefined;
  readonly className?: string;
}

/**
 * One line with the outcome of the rerank stage of a search: who reranked and how long it took, or why it
 * was skipped. The live region is always rendered so a screen reader announces the change when a search ends.
 */
function RerankStatusLine({ status, className = '' }: RerankStatusLineProps) {
  const { t } = useTranslation('common');
  let text: string | null = null;
  if (status?.applied) {
    const source = status.source
      ? t(`rerankControls.status.sources.${status.source}`, { defaultValue: status.source })
      : '';
    text = t('rerankControls.status.applied', {
      name: status.service_name ?? '',
      source,
      ms: status.latency_ms ?? 0,
    });
  } else if (status) {
    const reason = status.skipped_reason ?? 'not_configured';
    text = t('rerankControls.status.skipped', {
      reason: t(`rerankControls.status.reasons.${reason}`, { defaultValue: reason }),
    });
  }
  return (
    <div aria-live="polite" className={className}>
      {text && <p className="text-xs text-fg-tertiary dark:text-fg-tertiary-dark">{text}</p>}
    </div>
  );
}

export default RerankStatusLine;
