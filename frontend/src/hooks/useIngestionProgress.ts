/**
 * Hook for monitoring ingestion progress via Server-Sent Events (SSE)
 *
 * Usage:
 * const { progress, isConnected, error } = useIngestionProgress(appId, siloId, sessionId);
 *
 * if (progress) {
 *   console.log(`${progress.progress_percent.toFixed(1)}% complete`);
 *   console.log(`ETA: ${Math.round(progress.estimated_remaining_seconds)}s`);
 * }
 */

import { useState, useEffect, useRef } from 'react';
import { configService } from '../core/ConfigService';

export interface IngestionProgressData {
  session_id: string;
  silo_id: number;
  total_chunks: number;
  processed_chunks: number;
  failed_chunks: number;
  progress_percent: number;
  current_chunk_name: string;
  elapsed_seconds: number;
  estimated_remaining_seconds: number | null;
  estimated_total_time_seconds: number | null;
}

export interface UseIngestionProgressResult {
  progress: IngestionProgressData | null;
  isConnected: boolean;
  isComplete: boolean;
  error: string | null;
  startTime: Date | null;
}

export function useIngestionProgress(
  appId: number,
  repositoryId: number,
  sessionId: string | null,
): UseIngestionProgressResult {
  const [progress, setProgress] = useState<IngestionProgressData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [startTime] = useState(new Date());
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    const baseUrl = configService.getApiBaseUrl();
    const url = `${baseUrl}/internal/apps/${appId}/repositories/${repositoryId}/ingestion-progress/${sessionId}`;

    try {
      const eventSource = new EventSource(url, { withCredentials: true });
      eventSourceRef.current = eventSource;
      setError(null);

      // Mark as connected once the HTTP handshake succeeds
      eventSource.addEventListener('open', () => {
        setIsConnected(true);
        setError(null);
      });

      // Progress updates (named event emitted by some backends)
      eventSource.addEventListener('progress', (event) => {
        try {
          const data = JSON.parse(event.data) as IngestionProgressData;
          setProgress(data);
        } catch (e) {
          console.error('Failed to parse progress event:', e);
        }
      });

      // Default (unnamed) data events — the silo SSE endpoint uses these
      eventSource.addEventListener('message', (event) => {
        try {
          const data = JSON.parse(event.data) as IngestionProgressData;
          setProgress(data);
        } catch (e) {
          // Ignore parse errors for heartbeat / empty data lines
        }
      });

      // Completion event sent by the server
      eventSource.addEventListener('complete', () => {
        setIsComplete(true);
        setIsConnected(false);
        eventSource.close();
        eventSourceRef.current = null;
      });

      // Single consolidated error handler.
      // onerror and addEventListener('error') both fire for the same events, so
      // we use only addEventListener to avoid double-handling.
      eventSource.addEventListener('error', (event) => {
        if (event instanceof MessageEvent) {
          // Server-sent `event: error` with a data payload
          setError('Ingestion error from server');
          setIsConnected(false);
          eventSource.close();
          eventSourceRef.current = null;
        } else if (eventSource.readyState === EventSource.CONNECTING) {
          // Connection dropped; browser is auto-retrying — let it.
          setIsConnected(false);
        } else {
          // Permanently closed (readyState CLOSED, no auto-retry)
          setError('Connection lost');
          setIsConnected(false);
          eventSourceRef.current = null;
        }
      });

    } catch (e) {
      setError(`Failed to connect: ${String(e)}`);
      setIsConnected(false);
    }

    // Cleanup
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, [appId, repositoryId, sessionId]);

  return { progress, isConnected, isComplete, error, startTime };
}
