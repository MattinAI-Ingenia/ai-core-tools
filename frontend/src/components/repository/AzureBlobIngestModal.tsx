import React, { useEffect, useRef, useState } from 'react';
import { Cloud } from 'lucide-react';
import Modal from '../ui/Modal';
import { apiService } from '../../services/api';
import type {
  AzureBlobSource,
  IngestAzureBlobsPayload,
  IngestAzureBlobsResult,
  PreviewAzureBlobsResult,
} from '../../services/api';

export const AZURE_BLOB_SAMPLE_SIZE = 20;

interface AzureBlobIngestModalProps {
  isOpen: boolean;
  onClose: () => void;
  appId: number;
  repositoryId: number;
  savedSource?: AzureBlobSource | null;
  isUpdate: boolean;
  onIngested: (result: IngestAzureBlobsResult) => void;
}

const inputClassName =
  'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500';

const parsePrefixes = (raw: string): string[] =>
  raw.split(',').map((p) => p.trim()).filter(Boolean);

export default function AzureBlobIngestModal({
  isOpen,
  onClose,
  appId,
  repositoryId,
  savedSource,
  isUpdate,
  onIngested,
}: AzureBlobIngestModalProps) {
  const [accountUrl, setAccountUrl] = useState('https://');
  const [container, setContainer] = useState('');
  const [prefixes, setPrefixes] = useState('');
  const [nameExcludes, setNameExcludes] = useState('');
  const [blobName, setBlobName] = useState('');
  const [authMode, setAuthMode] = useState<'ANONYMOUS' | 'SAS_TOKEN'>('ANONYMOUS');
  const [sasToken, setSasToken] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [counting, setCounting] = useState(false);
  const [countResult, setCountResult] = useState<PreviewAzureBlobsResult | null>(null);
  // Bumped on every field change: an in-flight count response resolving after
  // an edit must never repopulate the cleared banner with the OLD filters'
  // numbers — only the latest request's response is ever displayed.
  const countRequestRef = useRef(0);
  const [formError, setFormError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setAccountUrl(savedSource?.account_url ?? 'https://');
      setContainer(savedSource?.container ?? '');
      setPrefixes(savedSource?.prefixes?.length ? savedSource.prefixes.join(', ') : '');
      setNameExcludes(savedSource?.name_excludes?.length ? savedSource.name_excludes.join(', ') : '');
      setBlobName('');
      setAuthMode(savedSource?.auth_mode ?? 'ANONYMOUS');
      setSasToken('');
      // Invalidate any count request still in flight from a previous open of
      // this modal, not just the displayed result.
      countRequestRef.current += 1;
      setCountResult(null);
      setFormError(null);
    }
    // savedSource is deliberately read only on the open transition: it is a
    // fresh object after every background repository refresh, and re-running
    // this effect on a new identity would wipe a half-typed SAS token.
  }, [isOpen]);

  const invalidateCount = () => {
    countRequestRef.current += 1;
    setCountResult(null);
  };

  const validate = (): string | null => {
    if (!accountUrl.trim() || !container.trim()) {
      return 'Account URL and container are required.';
    }
    if (parsePrefixes(prefixes).length && blobName.trim()) {
      return 'Prefixes and single file are mutually exclusive.';
    }
    if (authMode === 'SAS_TOKEN' && !sasToken.trim()) {
      return 'A SAS token is required for SAS token auth.';
    }
    return null;
  };

  const buildPayload = (): IngestAzureBlobsPayload => ({
    account_url: accountUrl.trim(),
    container: container.trim(),
    prefixes: parsePrefixes(prefixes),
    name_excludes: parsePrefixes(nameExcludes),
    auth_mode: authMode,
    sas_token: authMode === 'SAS_TOKEN' ? sasToken.trim() : null,
    sample_size: AZURE_BLOB_SAMPLE_SIZE,
    blob_name: blobName.trim() || null,
  });

  const handleCount = async () => {
    const error = validate();
    if (error) {
      setFormError(error);
      return;
    }

    const requestId = ++countRequestRef.current;
    setCounting(true);
    setFormError(null);

    try {
      const result = await apiService.previewAzureBlobs(appId, repositoryId, buildPayload());
      if (countRequestRef.current === requestId) {
        setCountResult(result);
      }
    } catch (err: any) {
      if (countRequestRef.current === requestId) {
        setCountResult(null);
        setFormError(err.message || 'Azure Blob count failed');
      }
    } finally {
      // Unconditional: a superseded response must still release the button,
      // and the generation counter is what guards display correctness — with
      // the button disabled there is at most one request in flight anyway.
      setCounting(false);
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    const error = validate();
    if (error) {
      setFormError(error);
      return;
    }

    setSubmitting(true);
    setFormError(null);

    try {
      const result = await apiService.ingestAzureBlobs(appId, repositoryId, buildPayload());
      onClose();
      onIngested(result);
    } catch (err: any) {
      setFormError(err.message || 'Azure Blob ingestion failed');
    } finally {
      setSubmitting(false);
    }
  };

  const canCheckCount = !counting && !submitting && !!accountUrl.trim() && !!container.trim();

  return (
    <Modal
      isOpen={isOpen}
      onClose={submitting ? () => undefined : onClose}
      title={isUpdate ? 'Update from Azure Blob' : 'Load from Azure Blob'}
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="bg-blue-50 border border-blue-200 text-blue-800 text-sm rounded-md px-3 py-2">
          Validation mode: at most {AZURE_BLOB_SAMPLE_SIZE} files are ingested per run, chosen at
          random from everything not yet in this repository. Use Update to pick up the rest.
        </div>

        <div>
          <label htmlFor="azure-account-url" className="block text-sm font-medium text-gray-700 mb-2">
            Storage Account URL
          </label>
          <input
            id="azure-account-url"
            type="text"
            value={accountUrl}
            onChange={(e) => { setAccountUrl(e.target.value); invalidateCount(); }}
            className={inputClassName}
            placeholder="https://myaccount.blob.core.windows.net"
            required
          />
        </div>

        <div>
          <label htmlFor="azure-container" className="block text-sm font-medium text-gray-700 mb-2">
            Container
          </label>
          <input
            id="azure-container"
            type="text"
            value={container}
            onChange={(e) => { setContainer(e.target.value); invalidateCount(); }}
            className={inputClassName}
            placeholder="container-name"
            required
          />
        </div>

        <div>
          <label htmlFor="azure-prefixes" className="block text-sm font-medium text-gray-700 mb-2">
            Name prefixes <span className="text-gray-400 font-normal">(optional — comma-separated)</span>
          </label>
          <input
            id="azure-prefixes"
            type="text"
            value={prefixes}
            onChange={(e) => { setPrefixes(e.target.value); invalidateCount(); }}
            className={inputClassName}
            placeholder="CDOC, DSAT"
          />
          <p className="text-xs text-gray-500 mt-1">
            Only blobs whose names start with any of these are loaded (e.g. <code>CDOC, DSAT</code>
            {' '}loads <code>CDOC000933.pdf</code> and <code>DSAT001234.pdf</code>, skipping the
            rest). Values cannot contain commas. Leave empty to load the whole container.
          </p>
        </div>

        <div>
          <label htmlFor="azure-name-excludes" className="block text-sm font-medium text-gray-700 mb-2">
            Excluded characters <span className="text-gray-400 font-normal">(optional — comma-separated)</span>
          </label>
          <input
            id="azure-name-excludes"
            type="text"
            value={nameExcludes}
            onChange={(e) => { setNameExcludes(e.target.value); invalidateCount(); }}
            className={inputClassName}
            placeholder="_"
          />
          <p className="text-xs text-gray-500 mt-1">
            Files whose names contain any of these are never loaded (e.g. <code>_</code> skips
            generated duplicates like <code>CDOC002817_2a67fdb9.pdf</code>). Values cannot contain
            commas.
          </p>
        </div>

        <div>
          <label htmlFor="azure-blob-name" className="block text-sm font-medium text-gray-700 mb-2">
            Single file <span className="text-gray-400 font-normal">(optional — ingests only this one)</span>
          </label>
          <input
            id="azure-blob-name"
            type="text"
            value={blobName}
            onChange={(e) => { setBlobName(e.target.value); invalidateCount(); }}
            className={inputClassName}
            placeholder="CDOC004211.pdf"
          />
          <p className="text-xs text-gray-500 mt-1">
            Exact blob name as stored in the container, including its folder path if it has one
            (e.g. <code>2026/CDOC004211.pdf</code>). Ignoring the random-20 cap.
          </p>
        </div>

        <div>
          <label htmlFor="azure-auth-mode" className="block text-sm font-medium text-gray-700 mb-2">
            Authentication
          </label>
          <select
            id="azure-auth-mode"
            value={authMode}
            onChange={(e) => { setAuthMode(e.target.value as 'ANONYMOUS' | 'SAS_TOKEN'); invalidateCount(); }}
            className={inputClassName}
          >
            <option value="ANONYMOUS">Public access (anonymous)</option>
            <option value="SAS_TOKEN">SAS token</option>
          </select>
        </div>

        {authMode === 'SAS_TOKEN' && (
          <div>
            <label htmlFor="azure-sas-token" className="block text-sm font-medium text-gray-700 mb-2">
              SAS Token {savedSource?.auth_mode === 'SAS_TOKEN' && <span className="text-gray-400 font-normal">(re-enter to update)</span>}
            </label>
            <input
              id="azure-sas-token"
              type="password"
              value={sasToken}
              onChange={(e) => { setSasToken(e.target.value); invalidateCount(); }}
              className={inputClassName}
              placeholder="sv=…&sig=…"
            />
          </div>
        )}

        {/* Always rendered so the count is reliably announced to assistive
            tech: a live region mounted together with its first content is not. */}
        <div
          role="status"
          className={
            countResult
              ? 'bg-green-50 border border-green-200 text-green-800 text-sm rounded-md px-3 py-2'
              : 'sr-only'
          }
        >
          {countResult &&
            (countResult.pending_blobs === 0
              ? `Found ${countResult.total_blobs} document(s) — everything is already up to date.`
              : `Found ${countResult.total_blobs} document(s): ${countResult.pending_blobs} new or changed. A load ingests up to ${AZURE_BLOB_SAMPLE_SIZE} of them at random.`)}
        </div>

        {formError && (
          <div role="alert" className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-md px-3 py-2">
            {formError}
          </div>
        )}

        <div className="flex justify-end space-x-3 pt-2">
          <button
            type="button"
            onClick={handleCount}
            disabled={!canCheckCount}
            className="px-4 py-2 bg-green-700 text-white rounded-md hover:bg-green-800 disabled:opacity-50"
          >
            {counting ? 'Counting…' : 'Check count'}
          </button>
          <button
            type="button"
            onClick={onClose}
            disabled={submitting}
            className="px-4 py-2 text-gray-600 hover:text-gray-800 disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={submitting || !accountUrl.trim() || !container.trim()}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center gap-2 disabled:opacity-50"
          >
            <Cloud className="w-4 h-4" aria-hidden="true" />
            {submitting
              ? 'Checking container…'
              : isUpdate
                ? blobName.trim()
                  ? 'Update file'
                  : `Update (${AZURE_BLOB_SAMPLE_SIZE} at random)`
                : blobName.trim()
                  ? 'Load file'
                  : `Load (${AZURE_BLOB_SAMPLE_SIZE} at random)`}
          </button>
        </div>
      </form>
    </Modal>
  );
}
