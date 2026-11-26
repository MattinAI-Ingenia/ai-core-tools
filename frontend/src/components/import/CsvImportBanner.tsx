import { useTranslation } from 'react-i18next';
import Alert from '../ui/Alert';
import type { ImportJob } from '../../types/csvImport';

interface CsvImportBannerProps {
  job: ImportJob;
  onReview: () => void;
}

export function CsvImportBanner({ job, onReview }: Readonly<CsvImportBannerProps>) {
  const { t } = useTranslation('common');
  const { total, downloaded, failed } = job.counts;

  if (job.status === 'DOWNLOADING') {
    const done = downloaded + failed;
    return <Alert type="info" message={t('lightrag.csvImportBanner.inProgress', { done, total })} className="mb-4" />;
  }

  return (
    <Alert
      type="warning"
      className="mb-4"
      message={
        <span>
          {t('lightrag.csvImportBanner.ready', { downloaded, failed })}{' '}
          <button type="button" className="underline" onClick={onReview}>{t('lightrag.csvImportBanner.review')}</button>
        </span>
      }
    />
  );
}
