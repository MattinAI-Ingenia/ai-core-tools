import { useCallback, useEffect, useId, useState } from 'react';
import { Trash2 } from 'lucide-react';
import Modal from '../ui/Modal';
import { apiService, ApiError, type GlobalMemoryFact } from '../../services/api';

interface MemoryPreferencesModalProps {
  readonly isOpen: boolean;
  readonly onClose: () => void;
  readonly appId: number;
  readonly agentId: number;
  readonly agentName: string;
}

type Status =
  | { readonly kind: 'idle' }
  | { readonly kind: 'success'; readonly message: string }
  | { readonly kind: 'error'; readonly message: string };

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof ApiError || error instanceof Error ? error.message : fallback;
}

function formatDate(iso: string | null): string | null {
  if (!iso) return null;
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? null : date.toLocaleString();
}

/**
 * Cross-session memory "Preferences" panel (D-11/FR-8(b)): lists the durable facts
 * remembered about the *current user* in this app, and erases them one at a time or
 * all at once. Identity is always the session's — no call here sends a user id.
 */
function MemoryPreferencesModal({ isOpen, onClose, appId, agentId, agentName }: MemoryPreferencesModalProps) {
  const descriptionId = useId();
  const [facts, setFacts] = useState<GlobalMemoryFact[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [deletingKey, setDeletingKey] = useState<string | null>(null);
  const [isConfirmingDeleteAll, setIsConfirmingDeleteAll] = useState(false);
  const [isDeletingAll, setIsDeletingAll] = useState(false);
  const [status, setStatus] = useState<Status>({ kind: 'idle' });

  const loadFacts = useCallback(async () => {
    setIsLoading(true);
    setLoadError(null);
    try {
      setFacts(await apiService.listMyGlobalMemory(appId));
    } catch (error) {
      setLoadError(errorMessage(error, 'Failed to load your preferences.'));
    } finally {
      setIsLoading(false);
    }
  }, [appId]);

  useEffect(() => {
    if (!isOpen) return;
    setStatus({ kind: 'idle' });
    setIsConfirmingDeleteAll(false);
    void loadFacts();
  }, [isOpen, loadFacts]);

  const handleDeleteOne = async (fact: GlobalMemoryFact) => {
    setDeletingKey(fact.key);
    try {
      await apiService.eraseMyGlobalMemoryFact(appId, fact.key);
      setFacts((current) => current.filter((f) => f.key !== fact.key));
      setStatus({ kind: 'success', message: `Deleted "${fact.text}".` });
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        // Already gone (e.g. erased from another tab) — the list was simply stale.
        setFacts((current) => current.filter((f) => f.key !== fact.key));
      } else {
        setStatus({ kind: 'error', message: errorMessage(error, 'Failed to delete the preference.') });
      }
    } finally {
      setDeletingKey(null);
    }
  };

  const handleDeleteAll = async () => {
    setIsDeletingAll(true);
    try {
      const result = await apiService.eraseMyGlobalMemory(appId);
      setFacts([]);
      setStatus({
        kind: 'success',
        message:
          result.purged > 0
            ? `Deleted ${result.purged} remembered preference${result.purged === 1 ? '' : 's'}.`
            : 'Nothing to delete — no preferences were stored about you.',
      });
    } catch (error) {
      setStatus({ kind: 'error', message: errorMessage(error, 'Failed to delete your preferences.') });
    } finally {
      setIsDeletingAll(false);
      setIsConfirmingDeleteAll(false);
    }
  };

  const isBusy = isDeletingAll || deletingKey !== null;

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Preferences" size="medium" descriptionId={descriptionId}>
      <div className="space-y-4">
        <p id={descriptionId} className="text-sm text-gray-600 dark:text-slate-300">
          What agents in this app remember about you from past conversations. It is shared by every
          agent in this app with cross-session memory, not just {agentName}. Deleting a preference
          does not delete your conversation history.
        </p>

        {status.kind !== 'idle' && (
          <div
            role={status.kind === 'error' ? 'alert' : 'status'}
            aria-live={status.kind === 'error' ? undefined : 'polite'}
            className={`text-sm rounded-lg px-3 py-2 border ${
              status.kind === 'success'
                ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 border-green-200 dark:border-green-800/40'
                : 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 border-red-200 dark:border-red-800/40'
            }`}
          >
            {status.message}
          </div>
        )}

        {isLoading && (
          <p role="status" className="text-sm text-gray-500 dark:text-slate-400">
            Loading your preferences…
          </p>
        )}

        {!isLoading && loadError && (
          <div role="alert" className="text-sm text-red-700 dark:text-red-300 space-y-2">
            <p>{loadError}</p>
            <button
              type="button"
              onClick={() => void loadFacts()}
              className="text-indigo-600 dark:text-indigo-400 underline focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 rounded"
            >
              Retry
            </button>
          </div>
        )}

        {!isLoading && !loadError && facts.length === 0 && (
          <p className="text-sm text-gray-500 dark:text-slate-400">No preferences are remembered about you yet.</p>
        )}

        {!isLoading && !loadError && facts.length > 0 && (
          <ul className="divide-y divide-gray-200 dark:divide-slate-600 border border-gray-200 dark:border-slate-600 rounded-lg">
            {facts.map((fact) => {
              const learnedAt = formatDate(fact.created_at);
              return (
                <li key={fact.key} className="flex items-start justify-between gap-3 px-3 py-2">
                  <div className="min-w-0">
                    <p className="text-sm text-gray-900 dark:text-slate-100 break-words">{fact.text}</p>
                    <p className="text-xs text-gray-500 dark:text-slate-400">
                      {fact.agent_id === agentId ? `Learned by ${agentName}` : 'Learned by another agent'}
                      {learnedAt ? ` · ${learnedAt}` : ''}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => void handleDeleteOne(fact)}
                    disabled={isBusy}
                    aria-label={`Delete preference: ${fact.text}`}
                    className="shrink-0 p-1.5 rounded-lg text-gray-500 dark:text-slate-400
                               hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20
                               disabled:opacity-50 disabled:cursor-not-allowed
                               focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
                  >
                    <Trash2 className="w-4 h-4" aria-hidden="true" />
                  </button>
                </li>
              );
            })}
          </ul>
        )}

        {!isLoading && !loadError && facts.length > 0 && (
          <div className="flex justify-end items-center gap-2 pt-2">
            {isConfirmingDeleteAll ? (
              <>
                <span className="text-sm text-gray-700 dark:text-slate-200">
                  Delete all {facts.length} preference{facts.length === 1 ? '' : 's'}?
                </span>
                <button
                  type="button"
                  onClick={() => setIsConfirmingDeleteAll(false)}
                  disabled={isDeletingAll}
                  className="px-3 py-1.5 text-sm rounded-lg border border-gray-300 dark:border-slate-500
                             text-gray-700 dark:text-slate-200 hover:bg-gray-50 dark:hover:bg-slate-700
                             disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={() => void handleDeleteAll()}
                  disabled={isDeletingAll}
                  className="px-3 py-1.5 text-sm rounded-lg bg-red-600 text-white hover:bg-red-700
                             disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500"
                >
                  {isDeletingAll ? 'Deleting…' : 'Delete all'}
                </button>
              </>
            ) : (
              <button
                type="button"
                onClick={() => setIsConfirmingDeleteAll(true)}
                disabled={isBusy}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg
                           text-red-600 dark:text-red-400 border border-red-200 dark:border-red-800/40
                           hover:bg-red-50 dark:hover:bg-red-900/20 disabled:opacity-50
                           focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500"
              >
                <Trash2 className="w-4 h-4" aria-hidden="true" />
                Delete all
              </button>
            )}
          </div>
        )}
      </div>
    </Modal>
  );
}

export default MemoryPreferencesModal;
