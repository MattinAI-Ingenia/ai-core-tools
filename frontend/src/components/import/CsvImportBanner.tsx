import Alert from '../ui/Alert';
import type { ImportJob } from '../../types/csvImport';

interface CsvImportBannerProps {
  job: ImportJob;
  onReview: () => void;
}

export function CsvImportBanner({ job, onReview }: Readonly<CsvImportBannerProps>) {
  const { total, downloaded, failed } = job.counts;

  if (job.status === 'DOWNLOADING') {
    const done = downloaded + failed;
    return <Alert type="info" message={`Import in progress: ${done}/${total} completed`} className="mb-4" />;
  }

  return (
    <Alert
      type="warning"
      className="mb-4"
      message={
        <span>
          {`Import ready: ${downloaded} OK, ${failed} failed — `}
          <button type="button" className="underline" onClick={onReview}>review</button>
        </span>
      }
    />
  );
}
