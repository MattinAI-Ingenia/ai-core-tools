import { useEffect, useRef, useState } from 'react';
import { apiService } from '../services/api';
import type { ImportJob } from '../types/csvImport';

const POLL_INTERVAL_MS = 2000;

export function useCsvImportPolling(appId: number, repositoryId: number) {
  const [job, setJob] = useState<ImportJob | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let cancelled = false;
    apiService.getActiveCsvImport(appId, repositoryId).then((active) => {
      if (!cancelled) setJob(active);
    });
    return () => { cancelled = true; };
  }, [appId, repositoryId]);

  useEffect(() => {
    if (!job || job.status !== 'DOWNLOADING') {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }
    const poll = async () => {
      try {
        const updated = await apiService.getCsvImport(appId, repositoryId, job.id);
        setJob(updated);
      } catch {
        // Silently ignore poll errors; will retry next interval.
      }
    };
    intervalRef.current = setInterval(() => { void poll(); }, POLL_INTERVAL_MS);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [appId, repositoryId, job?.id, job?.status]);

  return { job, setJob };
}
