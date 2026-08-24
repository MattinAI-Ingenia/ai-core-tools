import { Loader2 } from 'lucide-react';
import Modal from './Modal';
import type { CostEstimationResult } from '../../pages/RepositoryDetailPage';

export function formatEstimateValue(value: number | null | undefined) {
  if (value == null) return 'Unavailable';
  return new Intl.NumberFormat().format(value);
}

// Round a predicted estimate up to 2 significant figures so it reads as a clean
// upper bound (e.g. 22,448 → 23,000) instead of false precision. Rounds up so
// the displayed value never undershoots the computed estimate.
function roundEstimateUp(value: number | null | undefined): number | null | undefined {
  if (value == null || !Number.isFinite(value) || value <= 0) return value;
  const magnitude = 10 ** (Math.floor(Math.log10(value)) - 1);
  return Math.ceil(value / magnitude) * magnitude;
}

function formatSeconds(seconds: number | null | undefined): string {
  if (seconds == null) return '?';
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  if (mins > 0) return `${mins}m ${secs}s`;
  return `${secs}s`;
}

export interface CostEstimateItem {
  key: string;
  label: string;
  meta?: string;
}

interface CostEstimateModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  confirming: boolean;
  confirmingLabel?: string;
  estimate: CostEstimationResult | null;
  description: string;
  itemsHeading: string;
  items: CostEstimateItem[];
  /**
   * Optional step that must happen before ingestion can be confirmed.
   *
   * Used for LightRAG entity types: they shape the whole graph and become
   * immutable once anything is indexed, so on a silo set to infer them the
   * ingestion is gated until a human has seen and confirmed the proposal.
   * Omit it and the modal behaves exactly as before.
   */
  gate?: {
    /** Blocks Confirm until `done`. */
    required: boolean;
    done: boolean;
    label: string;
    doneLabel: string;
    onOpen: () => void;
  };
}

export default function CostEstimateModal({
  isOpen, onClose, onConfirm, confirming, confirmingLabel = 'Confirming…', estimate, description, itemsHeading, items, gate,
}: Readonly<CostEstimateModalProps>) {
  const blocked = !!gate?.required && !gate.done;
  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Confirm LightRAG Ingestion" size="large">
      <div className="space-y-5">
        <div>
          <p className="text-gray-700">
            This will immediately start LightRAG indexing. Review the estimated ingestion cost before continuing.
          </p>
          <p className="text-sm text-gray-500 mt-2">{description}</p>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-lg border border-gray-200 bg-gray-50 p-3">
            <p className="text-xs uppercase tracking-wide text-gray-500">Chunks</p>
            <p className="mt-1 text-lg font-semibold text-gray-900">{formatEstimateValue(roundEstimateUp(estimate?.total_chunks))}</p>
          </div>
          <div className="rounded-lg border border-gray-200 bg-gray-50 p-3">
            <p className="text-xs uppercase tracking-wide text-gray-500">Chunk Size</p>
            <p className="mt-1 text-lg font-semibold text-gray-900">{formatEstimateValue(estimate?.chunk_token_size)} tokens</p>
          </div>
          <div className="rounded-lg border border-gray-200 bg-gray-50 p-3">
            <p className="text-xs uppercase tracking-wide text-gray-500">LLM Tokens</p>
            <p className="mt-1 text-lg font-semibold text-gray-900">
              {formatEstimateValue(
                roundEstimateUp(
                  (estimate?.estimated_input_tokens ?? 0) + (estimate?.estimated_output_tokens ?? 0)
                )
              )}
            </p>
          </div>
          <div className="rounded-lg border border-gray-200 bg-gray-50 p-3">
            <p className="text-xs uppercase tracking-wide text-gray-500">Embedding Tokens</p>
            <p className="mt-1 text-lg font-semibold text-gray-900">
              {formatEstimateValue(roundEstimateUp(estimate?.estimated_embedding_tokens))}
            </p>
          </div>
        </div>

        <p className="text-xs text-gray-500">
          These are estimates. Actual cost depends on the number of entities extracted per chunk and may vary.
        </p>

        <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
          <p>
            Indexing model: <span className="font-medium">{estimate?.model_name || 'Unavailable'}</span>
          </p>
          <p className="mt-1">
            Embedding model: <span className="font-medium">{estimate?.embedding_model_name || 'Unavailable'}</span>
          </p>
          {(estimate?.estimated_indexing_time_min != null || estimate?.estimated_indexing_time_avg != null) && (
            <p className="mt-1">
              Estimated indexing time:{' '}
              <span className="font-medium">
                {estimate?.estimated_indexing_time_min != null
                  ? `${formatSeconds(estimate.estimated_indexing_time_min)} – ${formatSeconds(estimate.estimated_indexing_time_max)}`
                  : formatSeconds(estimate?.estimated_indexing_time_avg)}
              </span>
            </p>
          )}
          {(estimate?.estimated_cost_min != null || estimate?.estimated_cost_max != null) ? (
            <p className="mt-1">
              Estimated cost (LLM + embeddings):{' '}
              <span className="font-medium">
                {estimate?.estimated_cost_min != null && estimate?.estimated_cost_max != null
                  ? estimate.estimated_cost_min === estimate.estimated_cost_max
                    ? `${estimate.estimated_cost_min} ${estimate?.currency ?? 'USD'}`
                    : `${estimate.estimated_cost_min} – ${estimate.estimated_cost_max} ${estimate?.currency ?? 'USD'}`
                  : `${estimate?.estimated_cost_min ?? estimate?.estimated_cost_max} ${estimate?.currency ?? 'USD'}`}
              </span>
            </p>
          ) : (
            <p className="mt-1">
              Estimated cost: <span className="font-medium">Unavailable</span>
            </p>
          )}
        </div>

        <div className="sticky top-0 z-10 -mx-6 bg-white/95 px-6 py-3 backdrop-blur-sm border-y border-gray-200">
          <div className="flex justify-end gap-3">
            <button
              onClick={onClose}
              disabled={confirming}
              className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
            {gate?.required && (
              <button
                onClick={gate.onOpen}
                disabled={confirming}
                className={`px-4 py-2 rounded-lg transition-colors disabled:opacity-50 ${
                  gate.done
                    ? 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    : 'bg-yellow-500 text-white hover:bg-yellow-600'
                }`}
              >
                {gate.done ? gate.doneLabel : gate.label}
              </button>
            )}
            <button
              onClick={onConfirm}
              disabled={confirming || blocked}
              title={blocked ? gate?.label : undefined}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors disabled:opacity-50"
            >
              {confirming && <Loader2 className="w-4 h-4 animate-spin" />}
              {confirming ? confirmingLabel : 'Confirm ingestion'}
            </button>
          </div>
          {blocked && (
            <p className="mt-2 text-right text-sm text-amber-700">
              Define the entity types before indexing — they cannot be changed afterwards.
            </p>
          )}
        </div>

        {estimate?.warnings?.length ? (
          <div className="rounded-lg border border-yellow-200 bg-yellow-50 p-3">
            <h4 className="text-sm font-medium text-yellow-900">Warnings</h4>
            <ul className="mt-2 space-y-1 text-sm text-yellow-800 list-disc list-inside">
              {estimate.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </div>
        ) : null}

        <div>
          <h4 className="text-sm font-medium text-gray-900">{itemsHeading}</h4>
          <div className="mt-2 max-h-32 overflow-y-auto rounded-lg border border-gray-200">
            {items.map((item) => (
              <div key={item.key} className="flex items-center justify-between border-b border-gray-100 px-3 py-2 text-sm last:border-b-0">
                <span className="truncate pr-4 text-gray-700">{item.label}</span>
                {item.meta && <span className="shrink-0 text-gray-500">{item.meta}</span>}
              </div>
            ))}
          </div>
        </div>
      </div>
    </Modal>
  );
}
