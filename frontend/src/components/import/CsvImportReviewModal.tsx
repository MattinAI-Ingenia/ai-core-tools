import { useEffect, useMemo, useState } from 'react';
import { Trash2, UploadCloud } from 'lucide-react';
import Modal from '../ui/Modal';
import DataTable from '../ui/DataTable';
import type { TableColumn } from '../ui/Table';
import StatusBadge from '../StatusBadge';
import { apiService } from '../../services/api';
import type { ImportJob, ImportJobRow } from '../../types/csvImport';
import { useIngestionProgress } from '../../hooks/useIngestionProgress';
import CostEstimateModal from '../ui/CostEstimateModal';
import type { CostEstimationResult } from '../../pages/RepositoryDetailPage';

interface CsvImportReviewModalProps {
  appId: number;
  repositoryId: number;
  isLightRagRepository: boolean;
  job: ImportJob;
  isOpen: boolean;
  onClose: () => void;
  onJobUpdated: (job: ImportJob | null) => void;
  onIngestionStarted: (sessionId: string) => void;
}

export function CsvImportReviewModal({
  appId, repositoryId, isLightRagRepository, job, isOpen, onClose, onJobUpdated, onIngestionStarted,
}: Readonly<CsvImportReviewModalProps>) {
  const [selected, setSelected] = useState<Set<number>>(
    () => new Set(job.rows.filter((r) => r.status === 'DOWNLOADED').map((r) => r.id)),
  );

  // Reset the selection to "everything downloaded without errors" every time
  // the modal is (re)opened, not just on the component's first mount — the
  // parent keeps this component mounted across opens/closes while the import
  // job is active, so without this the checkboxes would go stale after rows
  // finish downloading in the background.
  useEffect(() => {
    if (isOpen) {
      setSelected(new Set(job.rows.filter((r) => r.status === 'DOWNLOADED').map((r) => r.id)));
    }
  }, [isOpen]);

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [costEstimate, setCostEstimate] = useState<CostEstimationResult | null>(null);
  const ingestion = useIngestionProgress(appId, repositoryId, sessionId);

  const toggle = (rowId: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(rowId)) {
        next.delete(rowId);
      } else {
        next.add(rowId);
      }
      return next;
    });
  };

  const [retryingRowId, setRetryingRowId] = useState<number | null>(null);

  const handleRetry = async (rowId: number) => {
    setRetryingRowId(rowId);
    const minDelay = new Promise((resolve) => setTimeout(resolve, 500));
    try {
      const [updated] = await Promise.all([
        apiService.retryCsvImportRow(appId, repositoryId, job.id, rowId),
        minDelay,
      ]);
      onJobUpdated(updated);
    } finally {
      setRetryingRowId(null);
    }
  };

  const [confirming, setConfirming] = useState(false);

  const doConfirm = async () => {
    setConfirming(true);
    try {
      const result = await apiService.confirmCsvImportRows(appId, repositoryId, job.id, Array.from(selected));
      if (result.session_id) {
        setSessionId(result.session_id);
        onIngestionStarted(result.session_id);
      }
      setCostEstimate(null);
      const remainingRows = job.rows.filter((r) => !selected.has(r.id));
      if (remainingRows.length === 0) {
        onJobUpdated(null);
        onClose();
      } else {
        const updated = await apiService.getCsvImport(appId, repositoryId, job.id).catch(() => null);
        onJobUpdated(updated);
      }
    } finally {
      setConfirming(false);
    }
  };

  // Same gate as manual upload's LightRAG flow (RepositoryDetailPage.tsx
  // handleFileUpload): show the cost/time estimate first and require a
  // second confirmation before any Resource is actually created. Reuses the
  // same CostEstimateModal screen as manual upload — see estimateCsvImportRows.
  const handleConfirmClick = async () => {
    if (!isLightRagRepository) {
      await doConfirm();
      return;
    }
    const estimate = await apiService.estimateCsvImportRows(appId, repositoryId, job.id, Array.from(selected));
    setCostEstimate(estimate);
  };

  const handleDiscardUnselected = async () => {
    const unselected = job.rows.filter((r) => !selected.has(r.id)).map((r) => r.id);
    if (unselected.length === 0) return;
    await apiService.discardCsvImportRows(appId, repositoryId, job.id, unselected);
    const updated = await apiService.getCsvImport(appId, repositoryId, job.id).catch(() => null);
    onJobUpdated(updated);
  };

  const visibleRows = useMemo(
    () => job.rows.filter((r) => r.status !== 'CONFIRMED' && r.status !== 'DISCARDED'),
    [job.rows],
  );

  const columns: TableColumn<ImportJobRow>[] = [
    {
      header: '',
      render: (row) => (
        <input
          type="checkbox"
          checked={selected.has(row.id)}
          onChange={() => toggle(row.id)}
          disabled={row.status !== 'DOWNLOADED'}
        />
      ),
    },
    {
      header: 'Link',
      render: (row) => (
        <span className="block max-w-[220px] truncate" title={row.url}>{row.url}</span>
      ),
    },
    {
      header: 'Status',
      render: (row) => (
        <div className="flex items-center gap-2">
          {row.status === 'FAILED'
            ? <StatusBadge status={`Failed: ${row.failure_reason}`} />
            : <StatusBadge status={row.status.toLowerCase()} />}
          {row.status === 'FAILED' && (
            <button
              type="button"
              onClick={() => handleRetry(row.id)}
              disabled={retryingRowId === row.id}
              className="text-xs font-medium text-blue-600 hover:text-blue-800 underline disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {retryingRowId === row.id ? 'Retrying…' : 'Retry'}
            </button>
          )}
        </div>
      ),
    },
    {
      header: 'Metadata',
      render: (row) => Object.entries(row.row_metadata).map(([k, v]) => `${k}: ${v}`).join(', '),
    },
  ];

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Review CSV import" size="xlarge">
      <DataTable
        data={visibleRows}
        columns={columns}
        keyExtractor={(row) => row.id}
      />
      <CostEstimateModal
        isOpen={!!costEstimate}
        onClose={() => setCostEstimate(null)}
        onConfirm={doConfirm}
        confirming={confirming}
        estimate={costEstimate}
        description={`${selected.size} document(s) selected for ingestion.`}
        itemsHeading="Documents to ingest"
        items={Array.from(selected).map((rowId) => {
          const row = job.rows.find((r) => r.id === rowId);
          return { key: String(rowId), label: row?.url ?? String(rowId) };
        })}
      />
      <div className="flex justify-end gap-3 mt-4">
        <button
          type="button"
          onClick={handleDiscardUnselected}
          className="flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium bg-red-50 text-red-700 hover:bg-red-100 border border-red-200 transition-colors"
        >
          <Trash2 className="w-4 h-4" /> Discard unselected
        </button>
        <button
          type="button"
          onClick={handleConfirmClick}
          disabled={selected.size === 0 || !!(ingestion.progress && !ingestion.isComplete)}
          className="flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium bg-green-600 text-white hover:bg-green-700 disabled:opacity-50 disabled:hover:bg-green-600 transition-colors"
        >
          <UploadCloud className="w-4 h-4" /> Ingest selected
        </button>
      </div>
    </Modal>
  );
}
