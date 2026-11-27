import { Info } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export interface RerankEgressNoticeProps {
  readonly className?: string;
}

/**
 * Persistent (non-dismissible) notice that a reranker receives the document
 * chunks and the user's query. Rendered with role="note" so screen readers
 * can reach it without it being announced as a live alert.
 */
function RerankEgressNotice({ className = '' }: RerankEgressNoticeProps) {
  const { t } = useTranslation('common');
  return (
    <div
      role="note"
      className={`flex gap-3 rounded-lg border border-info-border bg-info-bg p-4 dark:border-info-border-dark dark:bg-info-bg-dark ${className}`}
    >
      <Info className="h-5 w-5 shrink-0 text-info dark:text-info-dark" aria-hidden="true" />
      <p className="text-sm text-fg-secondary dark:text-fg-secondary-dark">
        {t('rerankEgressNotice.message')}
      </p>
    </div>
  );
}

export default RerankEgressNotice;
