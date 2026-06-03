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
import { AlertCircle, CheckCircle, Loader } from 'lucide-react';
import { useIngestionProgress } from '../../hooks/useIngestionProgress';

interface IngestionProgressBarProps {
  appId: number;
  repositoryId: number;
  sessionId: string | null;
  onComplete?: () => void;
}

export const IngestionProgressBar: React.FC<IngestionProgressBarProps> = ({
  appId,
  repositoryId,
  sessionId,
  onComplete,
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

  React.useEffect(() => {
    if (!isComplete) return;
    // Let the user see the "Ingestion Complete" state briefly before the bar
    // is removed.
    const timer = setTimeout(() => {
      onCompleteRef.current?.();
    }, 1500);
    return () => clearTimeout(timer);
  }, [isComplete]);

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
    return (
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-center gap-3">
        <Loader className="w-5 h-5 text-blue-600 animate-spin" />
        <p className="text-blue-800 font-medium">Connecting to indexing session...</p>
      </div>
    );
  }

  const formatTime = (seconds: number | null) => {
    if (seconds === null || seconds === undefined) return '--:--';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
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
        <span className="text-sm font-mono text-gray-600">
          {formatPercent(progress.progress_percent)}%
        </span>
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
            {progress.estimated_remaining_seconds !== null &&
              progress.estimated_remaining_seconds > 0 && (
                <>
                  {' '}
                  / {formatTime(progress.estimated_total_time_seconds)}
                </>
              )}
          </p>
          {progress.estimated_remaining_seconds !== null &&
            progress.estimated_remaining_seconds > 0 && (
              <p className="text-blue-600 text-xs mt-1">
                ETA: {formatTime(progress.estimated_remaining_seconds)}
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
