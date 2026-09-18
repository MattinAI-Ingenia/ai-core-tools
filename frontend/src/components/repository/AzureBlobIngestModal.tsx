import React, { useEffect, useState } from 'react';
import { Cloud } from 'lucide-react';
import Modal from '../ui/Modal';
import { apiService } from '../../services/api';
import type {
  AzureBlobSource,
  IngestAzureBlobsPayload,
  IngestAzureBlobsResult,
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
  const [prefix, setPrefix] = useState('');
  const [blobName, setBlobName] = useState('');
  const [authMode, setAuthMode] = useState<'ANONYMOUS' | 'SAS_TOKEN'>('ANONYMOUS');
  const [sasToken, setSasToken] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setAccountUrl(savedSource?.account_url ?? 'https://');
      setContainer(savedSource?.container ?? '');
      setPrefix(savedSource?.prefix ?? '');
      setBlobName('');
      setAuthMode(savedSource?.auth_mode ?? 'ANONYMOUS');
      setSasToken('');
      setFormError(null);
    }
    // savedSource is deliberately read only on the open transition: it is a
    // fresh object after every background repository refresh, and re-running
    // this effect on a new identity would wipe a half-typed SAS token.
  }, [isOpen]);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    if (!accountUrl.trim() || !container.trim()) {
      setFormError('Account URL and container are required.');
      return;
    }
    if (prefix.trim() && blobName.trim()) {
      setFormError('Prefix and single file are mutually exclusive.');
      return;
    }
    if (authMode === 'SAS_TOKEN' && !sasToken.trim()) {
      setFormError('A SAS token is required for SAS token auth.');
      return;
    }

    setSubmitting(true);
    setFormError(null);

    const payload: IngestAzureBlobsPayload = {
      account_url: accountUrl.trim(),
      container: container.trim(),
      prefix: prefix.trim() || null,
      auth_mode: authMode,
      sas_token: authMode === 'SAS_TOKEN' ? sasToken.trim() : null,
      sample_size: AZURE_BLOB_SAMPLE_SIZE,
      blob_name: blobName.trim() || null,
    };

    try {
      const result = await apiService.ingestAzureBlobs(appId, repositoryId, payload);
      onClose();
      onIngested(result);
    } catch (err: any) {
      setFormError(err.message || 'Azure Blob ingestion failed');
    } finally {
      setSubmitting(false);
    }
  };

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
            onChange={(e) => setAccountUrl(e.target.value)}
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
            onChange={(e) => setContainer(e.target.value)}
            className={inputClassName}
            placeholder="container-name"
            required
          />
        </div>

        <div>
          <label htmlFor="azure-prefix" className="block text-sm font-medium text-gray-700 mb-2">
            Prefix <span className="text-gray-400 font-normal">(optional)</span>
          </label>
          <input
            id="azure-prefix"
            type="text"
            value={prefix}
            onChange={(e) => setPrefix(e.target.value)}
            className={inputClassName}
            placeholder="2026/"
          />
        </div>

        <div>
          <label htmlFor="azure-blob-name" className="block text-sm font-medium text-gray-700 mb-2">
            Single file <span className="text-gray-400 font-normal">(optional — ingests only this one)</span>
          </label>
          <input
            id="azure-blob-name"
            type="text"
            value={blobName}
            onChange={(e) => setBlobName(e.target.value)}
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
            onChange={(e) => setAuthMode(e.target.value as 'ANONYMOUS' | 'SAS_TOKEN')}
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
              onChange={(e) => setSasToken(e.target.value)}
              className={inputClassName}
              placeholder="sv=…&sig=…"
            />
          </div>
        )}

        {formError && (
          <div role="alert" className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-md px-3 py-2">
            {formError}
          </div>
        )}

        <div className="flex justify-end space-x-3 pt-2">
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
            <Cloud className="w-4 h-4" />
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
