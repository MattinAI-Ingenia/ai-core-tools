/**
 * Ingestion progress bar component with time estimates
 *
 * Usage:
 * <IngestionProgressBar
 *   appId={appId}
 *   siloId={siloId}
 *   sessionId={sessionId}
 * />
 */

import React from 'react';
import { AlertCircle, CheckCircle, Loader, PauseCircle, StopCircle } from 'lucide-react';
import { useIngestionProgress } from '../../hooks/useIngestionProgress';
import ConfirmationModal from './ConfirmationModal';
import { apiService } from '../../services/api';

interface IngestionProgressBarProps {
  appId: number;
  repositoryId: number;
  sessionId: string | null;
  onComplete?: () => void;
  /** Called after the backend accepted a pause/cancel, so the parent can notify. */
  onStopped?: (mode: 'pause' | 'cancel', stoppedCount: number) => void;
}

export const IngestionProgressBar: React.FC<IngestionProgressBarProps> = ({
  appId,
  repositoryId,
  sessionId,
  onComplete,
  onStopped,
}) => {
  const { progress, isConnected, isComplete, error } = useIngestionProgress(
    appId,
    repositoryId,
    sessionId,
  );

  // Keep onComplete stable: always call the latest version but only once, when
  // isComplete first becomes true.  Without this, every parent re-render (e.g.
  // the loadRepository refresh that onComplete itself triggers) creates a new
  // onComplete reference, causing the effect to fire again and call onComplete
  // a second time — leading to a rapid double-refresh flicker.
  const onCompleteRef = React.useRef(onComplete);
  React.useEffect(() => {
    onCompleteRef.current = onComplete;
  });

  // Stop controls. Pause is reversible (a single click); Cancel removes the
  // files it never started, so it goes through a disclaimer first. Both stay in
  // the "…ing" state after the request because the backend lets the files
  // already in flight finish before the run ends — stopping is never instant.
  const [stopState, setStopState] = React.useState<
    { mode: 'pause' | 'cancel'; phase: 'stopping' | 'failed' } | null
  >(null);

  // Cancel removes files: it needs the consequence spelled out before it runs.
  // Pause is reversible, so it stays a single click.
  const [confirmCancel, setConfirmCancel] = React.useState(false);

  const requestStop = React.useCallback(
    async (mode: 'pause' | 'cancel') => {
      setStopState({ mode, phase: 'stopping' });
      try {
        const result = await apiService.stopIngestion(appId, repositoryId, mode);
        onStopped?.(mode, result.stopped);
      } catch {
        // Leave the run alone and let the user retry: a failed request must not
        // look like a successful stop.
        setStopState({ mode, phase: 'failed' });
      }
    },
    [appId, repositoryId, onStopped],
  );

  const stopBtn = "inline-flex items-center gap-1.5 px-2.5 py-1 text-sm font-medium rounded border transition-colors disabled:opacity-60 disabled:cursor-not-allowed";
  const busy = (mode: 'pause' | 'cancel') =>
    stopState?.mode === mode && stopState.phase === 'stopping';
  const failed = (mode: 'pause' | 'cancel') =>
    stopState?.mode === mode && stopState.phase === 'failed';

  const cancelDisclaimer = (
    <ConfirmationModal
      isOpen={confirmCancel}
      variant="warning"
      title="Cancel indexing?"
      confirmLabel="Cancel indexing"
      cancelLabel="Keep indexing"
      isLoading={stopState?.mode === 'cancel' && stopState.phase === 'stopping'}
      onCancel={() => setConfirmCancel(false)}
      onConfirm={() => {
        setConfirmCancel(false);
        requestStop('cancel');
      }}
      message={
        <>
          Files already being indexed will finish. Files not started yet are
          removed from the repository — you would need to upload them again.
          Nothing already indexed is deleted.
        </>
      }
    />
  );

  const stopControls = (
    <div className="flex items-center gap-2">
      {cancelDisclaimer}
      <button
        type="button"
        onClick={() => requestStop('pause')}
        disabled={busy('pause')}
        title="Pause indexing. Keeps what is already indexed and stays resumable."
        className={`${stopBtn} ${failed('pause')
          ? 'text-red-700 border-red-400 bg-red-50 hover:bg-red-100'
          : 'text-amber-700 border-amber-300 bg-white hover:bg-amber-50'}`}
      >
        <PauseCircle className="w-4 h-4" />
        {failed('pause') ? 'Retry pause' : busy('pause') ? 'Pausing…' : 'Pause'}
      </button>
      <button
        type="button"
        onClick={() => setConfirmCancel(true)}
        disabled={busy('cancel')}
        title="Cancel indexing: finishes the files already started and removes the ones not started yet."
        className={`${stopBtn} ${failed('cancel')
          ? 'text-red-700 border-red-400 bg-red-50 hover:bg-red-100'
          : 'text-red-700 border-red-300 bg-white hover:bg-red-50'}`}
      >
        <StopCircle className="w-4 h-4" />
        {failed('cancel') ? 'Retry cancel' : busy('cancel') ? 'Cancelling…' : 'Cancel'}
      </button>
    </div>
  );

  React.useEffect(() => {
    if (!isComplete) return;
    // Let the user see the "Ingestion Complete" state briefly before the bar
    // is removed.
    const timer = setTimeout(() => {
      onCompleteRef.current?.();
    }, 1500);
    return () => clearTimeout(timer);
  }, [isComplete]);

  // ETA and the projected total only refresh every 5s — every-tick updates on
  // an estimate that itself only settles over tens of seconds just flickers
  // without adding information. Elapsed time is exact, so it updates on every
  // progress event untouched.
  //
  // Rules of Hooks: these must sit above every early `return` below (a hook
  // skipped on some renders and not others is exactly what crashed this
  // component in production — React throws "Rendered fewer hooks than
  // expected" the moment `progress` flips from null to non-null and unmounts
  // the whole tree, since there's no error boundary anywhere in the app).
  const [throttledEstimates, setThrottledEstimates] = React.useState<{
    remaining: number | null;
    total: number | null;
  }>({ remaining: null, total: null });
  const lastEstimateUpdateRef = React.useRef(0);

  React.useEffect(() => {
    if (!progress) return;
    const now = Date.now();
    if (lastEstimateUpdateRef.current === 0 || now - lastEstimateUpdateRef.current >= 5000) {
      lastEstimateUpdateRef.current = now;
      setThrottledEstimates({
        remaining: progress.estimated_remaining_seconds,
        total: progress.estimated_total_time_seconds,
      });
    }
  }, [progress]);

  if (!sessionId) {
    return null;
  }

  // Error with no data yet — show a dedicated error state
  if (error && error !== 'Connection lost' && !isConnected && !progress && !isComplete) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
        <div className="flex-1">
          <p className="text-red-800 font-medium">Ingestion Error</p>
          <p className="text-red-700 text-sm">{error}</p>
        </div>
      </div>
    );
  }

  // Waiting for first progress event
  if (!progress) {
    // Fast completion: indexing finished before we received any progress data
    // (e.g. proxy buffered the SSE stream and delivered all events at once).
    if (isComplete) {
      return (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-center gap-3">
          <CheckCircle className="w-5 h-5 text-green-600" />
          <p className="text-green-800 font-medium">Indexing complete</p>
        </div>
      );
    }
    // HTTP connection established but no progress data yet — SSE events may be
    // buffered by a proxy.  Show a different message so the user knows we are
    // connected and indexing is happening, not still waiting to connect.
    if (isConnected) {
      return (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-center gap-3">
          <Loader className="w-5 h-5 text-blue-600 animate-spin" />
          <p className="text-blue-800 font-medium flex-1">Indexing in progress...</p>
          {stopControls}
        </div>
      );
    }
    return (
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-center gap-3">
        <Loader className="w-5 h-5 text-blue-600 animate-spin" />
        <p className="text-blue-800 font-medium flex-1">Connecting to indexing session...</p>
        {stopControls}
      </div>
    );
  }

  const formatTime = (seconds: number | null) => {
    if (seconds === null || seconds === undefined) return '--:--:--';
    const total = Math.floor(seconds);
    const hrs = Math.floor(total / 3600);
    const mins = Math.floor((total % 3600) / 60);
    const secs = total % 60;
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const formatPercent = (percent: number) => percent.toFixed(1);

  return (
    <div className="space-y-3 bg-white border border-gray-200 rounded-lg p-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isComplete ? (
            <CheckCircle className="w-5 h-5 text-green-600" />
          ) : (
            <Loader className="w-5 h-5 text-blue-600 animate-spin" />
          )}
          <span className="font-medium text-gray-900">
            {isComplete ? 'Ingestion Complete' : 'Ingesting Documents'}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-sm font-mono text-gray-600">
            {formatPercent(progress.progress_percent)}%
          </span>
          {!isComplete && stopControls}
        </div>
      </div>

      {/* Progress bar */}
      <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
        <div
          className={`h-full transition-all duration-300 ${isComplete ? 'bg-green-600' : 'bg-blue-600'
            }`}
          style={{ width: `${progress.progress_percent}%` }}
        />
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-4 text-sm">
        {/* Chunks */}
        <div className="bg-gray-50 rounded p-2">
          <p className="text-gray-600 text-xs">Chunks</p>
          <p className="font-mono font-medium">
            {progress.processed_chunks}/{progress.total_chunks}
          </p>
          {progress.failed_chunks > 0 && (
            <p className="text-red-600 text-xs mt-1">
              {progress.failed_chunks} failed
            </p>
          )}
        </div>

        {/* Time */}
        <div className="bg-gray-50 rounded p-2">
          <p className="text-gray-600 text-xs">Time</p>
          <p className="font-mono font-medium">
            {formatTime(progress.elapsed_seconds)}
            {throttledEstimates.remaining !== null &&
              throttledEstimates.remaining > 0 && (
                <>
                  {' '}
                  / {formatTime(throttledEstimates.total)}
                </>
              )}
          </p>
          {throttledEstimates.remaining !== null &&
            throttledEstimates.remaining > 0 && (
              <p className="text-blue-600 text-xs mt-1">
                ETA: {formatTime(throttledEstimates.remaining)}
              </p>
            )}
        </div>
      </div>

      {/* Current chunk */}
      {progress.current_chunk_name && (
        <div className="bg-blue-50 rounded p-2 text-xs text-gray-700 truncate">
          <span className="text-gray-600">Current: </span>
          <span className="font-mono">{progress.current_chunk_name}</span>
        </div>
      )}

      {/* Connection status */}
      {!isConnected && !isComplete && (
        <div className="text-xs text-amber-600 flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-amber-600 animate-pulse" />
          Reconnecting...
        </div>
      )}

      {error && (
        <div className="text-xs text-red-600 flex items-start gap-1">
          <AlertCircle className="w-3 h-3 mt-0.5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
};

export default IngestionProgressBar;
