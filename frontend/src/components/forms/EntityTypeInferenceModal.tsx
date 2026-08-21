import { useCallback, useEffect, useRef, useState } from 'react';
import { AlertTriangle, Loader2, X } from 'lucide-react';
import { apiService } from '../../services/api';

export interface ProposedEntityType {
  name: string;
  why: string;
  examples: string[];
}

interface InferenceJob {
  status: 'pending' | 'reading' | 'analysing' | 'consolidating' | 'done' | 'failed';
  done: number;
  total: number;
  sampled?: number;
  types?: ProposedEntityType[];
  error?: string;
}

interface Props {
  appId: number;
  siloId: number;
  /** Optional AI service override; unset uses the silo's extraction service. */
  aiServiceId?: number;
  /** Called with the confirmed comma-separated list. */
  onConfirm: (entityTypes: string) => void;
  onCancel: () => void;
}

const POLL_MS = 1500;

const STAGE_LABEL: Record<InferenceJob['status'], string> = {
  pending: 'Preparing…',
  reading: 'Reading documents',
  analysing: 'Proposing categories',
  consolidating: 'Merging the ones that overlap',
  done: 'Done',
  failed: 'Failed',
};

/**
 * Runs entity-type inference and lets the user pick which categories to keep.
 *
 * Nothing is applied automatically: the field is immutable once the silo has
 * been indexed, so a wrong list means re-indexing the whole corpus. The model
 * proposes, a person decides.
 */
export function EntityTypeInferenceModal({
  appId, siloId, aiServiceId, onConfirm, onCancel,
}: Props) {
  const [job, setJob] = useState<InferenceJob | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [extra, setExtra] = useState('');
  const [startError, setStartError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  const stopPolling = useCallback(() => {
    if (timer.current !== null) {
      window.clearInterval(timer.current);
      timer.current = null;
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const { job_id } = await apiService.inferEntityTypes(appId, siloId, aiServiceId) as { job_id: string };
        if (cancelled) return;

        timer.current = window.setInterval(async () => {
          try {
            const next = await apiService.getEntityTypeInferenceStatus(appId, siloId, job_id) as InferenceJob;
            if (cancelled) return;
            setJob(next);
            if (next.status === 'done' || next.status === 'failed') {
              stopPolling();
              // Everything on by default: the model already dropped the classes
              // it could not find instances for, so the common case is accept.
              if (next.types) setSelected(new Set(next.types.map((t) => t.name)));
            }
          } catch (err) {
            if (cancelled) return;
            stopPolling();
            setStartError(err instanceof Error ? err.message : 'Could not read the job progress.');
          }
        }, POLL_MS);
      } catch (err) {
        if (!cancelled) setStartError(err instanceof Error ? err.message : 'Could not start the analysis.');
      }
    })();

    return () => { cancelled = true; stopPolling(); };
  }, [appId, siloId, aiServiceId, stopPolling]);

  const toggle = (name: string) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(name)) next.delete(name); else next.add(name);
      return next;
    });
  };

  const chosen = [
    ...(job?.types ?? []).filter((t) => selected.has(t.name)).map((t) => t.name),
    ...extra.split(',').map((s) => s.trim()).filter(Boolean),
  ];

  const running = !!job && job.status !== 'done' && job.status !== 'failed';
  const percent = job && job.total > 0 && job.status === 'reading'
    ? Math.round((job.done / job.total) * 100)
    : null;
  const error = startError ?? (job?.status === 'failed' ? job.error : null);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-2xl max-h-[85vh] overflow-y-auto rounded-lg bg-white p-6 shadow-xl">
        <div className="mb-4 flex items-start justify-between">
          <h2 className="text-lg font-semibold text-gray-900">Entity types</h2>
          <button type="button" onClick={onCancel} aria-label="Close" className="text-gray-400 hover:text-gray-600">
            <X className="h-5 w-5" />
          </button>
        </div>

        <p className="mb-4 flex items-start gap-2 rounded-md bg-amber-50 p-3 text-sm text-amber-800">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>
            These categories shape the whole graph and <strong>cannot be changed
            once anything is indexed</strong>. Review them before continuing.
          </span>
        </p>

        {error && (
          <p className="mb-4 rounded-md bg-red-50 p-3 text-sm text-red-700">{error}</p>
        )}

        {(running || !job) && !error && (
          <div className="py-8 text-center">
            <Loader2 className="mx-auto mb-3 h-6 w-6 animate-spin text-gray-400" />
            <p className="text-sm text-gray-700">{STAGE_LABEL[job?.status ?? 'pending']}</p>
            {percent !== null && (
              <>
                <div className="mx-auto mt-3 h-2 w-64 overflow-hidden rounded-full bg-gray-200">
                  <div className="h-full bg-yellow-500 transition-all" style={{ width: `${percent}%` }} />
                </div>
                <p className="mt-2 text-xs text-gray-500">{job!.done} of {job!.total} documents</p>
              </>
            )}
          </div>
        )}

        {job?.status === 'done' && job.types && (
          <>
            <p className="mb-3 text-sm text-gray-600">
              {job.types.length} categories proposed
              {job.sampled ? ` from ${job.sampled} documents` : ''}. Untick the ones
              you do not want.
            </p>
            <ul className="mb-4 space-y-2">
              {job.types.map((type) => (
                <li key={type.name} className="rounded-md border border-gray-200 p-3">
                  <label className="flex cursor-pointer items-start gap-3">
                    <input
                      type="checkbox"
                      className="mt-1"
                      checked={selected.has(type.name)}
                      onChange={() => toggle(type.name)}
                    />
                    <span className="min-w-0">
                      <span className="font-medium text-gray-900">{type.name}</span>
                      <span className="block text-sm text-gray-600">{type.why}</span>
                      {type.examples.length > 0 && (
                        <span className="mt-1 block text-xs text-gray-500">
                          Examples: {type.examples.join(', ')}
                        </span>
                      )}
                    </span>
                  </label>
                </li>
              ))}
            </ul>

            <label htmlFor="extra-entity-types" className="mb-1 block text-sm font-medium text-gray-700">
              Add your own (comma-separated)
            </label>
            <input
              id="extra-entity-types"
              type="text"
              value={extra}
              onChange={(e) => setExtra(e.target.value)}
              placeholder="Brand, Standard"
              className="mb-4 w-full rounded-lg border border-gray-300 px-3 py-2 focus:border-transparent focus:outline-none focus:ring-2 focus:ring-yellow-500"
            />
          </>
        )}

        <div className="flex justify-end gap-2 border-t border-gray-200 pt-4">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-lg border border-gray-300 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => onConfirm(chosen.join(', '))}
            disabled={chosen.length === 0}
            className="rounded-lg bg-yellow-500 px-4 py-2 text-sm font-medium text-white hover:bg-yellow-600 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Confirm {chosen.length > 0 ? `(${chosen.length})` : ''}
          </button>
        </div>
      </div>
    </div>
  );
}
