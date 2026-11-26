import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import { useTranslation } from 'react-i18next';
import { Clapperboard, Mic, Folder, FolderOpen, Home, ChevronRight, Video, FileText, ArrowDownToLine, ArrowLeftRight, Trash2, Tv, Search, RefreshCw, CheckCircle, Cloud, PauseCircle, Loader } from 'lucide-react';
import { apiService } from '../services/api';
import type { AzureBlobSource, IngestAzureBlobsResult } from '../services/api';
import Modal from '../components/ui/Modal';
import FolderTree from '../components/FolderTree';
import { StatusBadge, type StatusType } from '../components/ui/Badge';
import Alert from '../components/ui/Alert';
import { useAppRole } from '../hooks/useAppRole';
import { AppRole } from '../types/roles';
import ReadOnlyBanner from '../components/ui/ReadOnlyBanner';
import IngestionProgressBar from '../components/ui/IngestionProgressBar';
import CostEstimateModal, { formatEstimateValue } from '../components/ui/CostEstimateModal';
import { EntityTypeInferenceModal } from '../components/forms/EntityTypeInferenceModal';
import ResourceMetrics, { formatDuration } from '../components/repository/ResourceMetrics';
import { CsvImportStepper } from '../components/import/CsvImportStepper';
import { CsvImportBanner } from '../components/import/CsvImportBanner';
import { CsvImportReviewModal } from '../components/import/CsvImportReviewModal';
import { useCsvImportPolling } from '../hooks/useCsvImportPolling';
import AzureBlobIngestModal, { AZURE_BLOB_SAMPLE_SIZE } from '../components/repository/AzureBlobIngestModal';
import { Title } from '../components/ui/Title';
import { ButtonApp } from '../components/ui/ButtonApp';
import BackgroundImage from '../components/ui/BackgroundImage';
import { FormField, FormSelect } from '../components/ui/FormField';
import { LoadingState } from '../components/ui/LoadingState';
import { Spinner } from '../components/ui/Spinner';
import type { ServiceLifecycleWarning } from '../types/services';
import ServiceLifecycleAlert from '../components/services/ServiceLifecycleAlert';

interface Resource {
  resource_id: number;
  name: string;
  uri: string;
  file_type: string;
  created_at: string;
  folder_id?: number;
  folder_path?: string;
  status?: string | null;
  progress_done?: number | null;
  progress_total?: number | null;
  error_message?: string | null;
}

interface RepositoryDetail {
  repository_id: number;
  name: string;
  created_at: string;
  silo_id?: number | null;
  vector_db_type?: string | null;
  total_indexing_duration_seconds?: number | null;
  azure_blob_source?: AzureBlobSource | null;
  resources: Resource[];
  folders: Array<{
    folder_id: number;
    name: string;
    parent_folder_id?: number;
  }>;
  embedding_services: Array<{
    service_id: number;
    name: string;
  }>;
  ai_services: Array<{
    service_id: number;
    name: string;
    supports_video?: boolean;
  }>;
  media: Media[];
  transcription_service_id?: number | null;
  video_ai_service_id?: number | null;
  service_warnings?: string[];
  lifecycle_warnings?: ServiceLifecycleWarning[];
}

export interface CostEstimationResult {
  total_chunks: number;
  chunk_token_size: number;
  estimated_llm_calls: number;
  estimated_embedding_calls: number;
  estimated_input_tokens: number;
  estimated_output_tokens: number;
  estimated_embedding_tokens?: number | null;
  estimated_cost_min?: number | null;
  estimated_cost_max?: number | null;
  currency?: string | null;
  estimated_indexing_time_min?: number | null;
  estimated_indexing_time_max?: number | null;
  estimated_indexing_time_avg?: number | null;
  model_name?: string | null;
  embedding_model_name?: string | null;
  warnings: string[];
}

interface Media {
  media_id: number;
  name: string;
  source_type: string;
  source_url: string | null;
  duration: number | null;
  language: string | null;
  status: string;
  processing_mode: string | null;
  error_message: string | null;
  create_date: string;
  folder_id: number | null;
}

function isTerminalMediaStatus(status: string): boolean {
  return status === 'ready' || status === 'error';
}

const IN_PROGRESS_MEDIA_STATUSES = new Set(['downloading', 'processing', 'transcribing', 'analyzing_video', 'indexing']);

function getMediaStatusBadgeType(status: string): StatusType {
  if (status === 'ready') return 'success';
  if (status === 'error') return 'error';
  if (status === 'pending') return 'pending';
  if (IN_PROGRESS_MEDIA_STATUSES.has(status)) return 'processing';
  return 'inactive';
}

function filterByFolder<T extends { folder_id?: number | null }>(
  items: T[] | undefined,
  selectedFolderId: number | null
): T[] {
  if (!items) return [];
  return items.filter(item =>
    selectedFolderId === null ? !item.folder_id : item.folder_id === selectedFolderId
  );
}

function updateMediaInRepo(prev: RepositoryDetail | null, mediaId: number, media: Media): RepositoryDetail | null {
  if (!prev) return null;
  return {
    ...prev,
    media: prev.media.map(m => (m.media_id === mediaId ? media : m)),
  };
}

// Ancestors of the selected folder (root excluded), each with its full path, for the breadcrumb
function getFolderTrail(
  folders: RepositoryDetail['folders'] | undefined,
  folderId: number | null
): Array<{ folder_id: number; name: string; path: string }> {
  if (!folders || folderId === null) return [];
  const byId = new Map(folders.map(f => [f.folder_id, f]));
  const chain: RepositoryDetail['folders'] = [];
  let current = byId.get(folderId);
  while (current && chain.length < folders.length) {
    chain.unshift(current);
    current = current.parent_folder_id ? byId.get(current.parent_folder_id) : undefined;
  }
  let path = '';
  return chain.map(f => {
    path = path ? `${path}/${f.name}` : f.name;
    return { folder_id: f.folder_id, name: f.name, path };
  });
}

function getAllFolderPaths(
  folders: Array<{ folder_id: number; name: string; subfolders?: Array<{ folder_id: number; name: string; subfolders?: unknown[] }> }>,
  parentPath: string = ''
): Array<{ folder_id: number; name: string; full_path: string }> {
  const result: Array<{ folder_id: number; name: string; full_path: string }> = [];
  for (const folder of folders) {
    const fullPath = parentPath ? `${parentPath}/${folder.name}` : folder.name;
    result.push({ folder_id: folder.folder_id, name: folder.name, full_path: fullPath });
    if (folder.subfolders && folder.subfolders.length > 0) {
      result.push(...getAllFolderPaths(folder.subfolders as Parameters<typeof getAllFolderPaths>[0], fullPath));
    }
  }
  return result;
}

const RepositoryDetailPage: React.FC = () => {
  const { appId, repositoryId } = useParams<{ appId: string; repositoryId: string }>();
  const { t, i18n } = useTranslation('common');
  const { hasMinRole, userRole } = useAppRole(appId);
  const canEdit = hasMinRole(AppRole.EDITOR);

  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [repository, setRepository] = useState<RepositoryDetail | null>(null);

  const isLightRAG = repository?.vector_db_type?.toUpperCase() === 'LIGHTRAG';
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [estimatingUpload, setEstimatingUpload] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showEstimateModal, setShowEstimateModal] = useState(false);
  // LightRAG entity types gate the ingestion when the silo is set to infer them
  // and none have been settled yet — they are immutable after the first index.
  const [siloEntityTypes, setSiloEntityTypes] = useState<{ mode: string; types: string | null; locked: boolean } | null>(null);
  const [showEntityTypesModal, setShowEntityTypesModal] = useState(false);
  const [pendingUploadFiles, setPendingUploadFiles] = useState<File[]>([]);
  const [uploadEstimate, setUploadEstimate] = useState<CostEstimationResult | null>(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [resourceToDelete, setResourceToDelete] = useState<Resource | null>(null);
  const [showDeleteAllModal, setShowDeleteAllModal] = useState(false);
  const [deletingAll, setDeletingAll] = useState(false);
  const [deleteAllError, setDeleteAllError] = useState<string | null>(null);
  const [deletingResource, setDeletingResource] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  // Failed delete attempts on the open modal; capped so a non-deletable (e.g. legacy LightRAG) file can't be spammed.
  const [deleteFailures, setDeleteFailures] = useState(0);
  const MAX_DELETE_FAILURES = 3;
  const [showMoveModal, setShowMoveModal] = useState(false);
  const [resourceToMove, setResourceToMove] = useState<Resource | null>(null);
  const [moveToFolderId, setMoveToFolderId] = useState<number | null>(null);
  const [showCsvImportModal, setShowCsvImportModal] = useState(false);
  const [showImportReviewModal, setShowImportReviewModal] = useState(false);
  const [showAzureBlobModal, setShowAzureBlobModal] = useState(false);
  const [azureBlobUpdateMode, setAzureBlobUpdateMode] = useState(false);
  const [azureBlobRunning, setAzureBlobRunning] = useState(false);
  const { job: activeImportJob, setJob: setActiveImportJob } = useCsvImportPolling(
    Number.parseInt(appId!), Number.parseInt(repositoryId!),
  );

  // Ingestion progress tracking
  const [ingestionSessionId, setIngestionSessionId] = useState<string | null>(null);
  const [isIndexing, setIsIndexing] = useState(false);
  const [resumable, setResumable] = useState(0);
  // Kept past the 3s poll's sessionId reset (see IngestionProgressBar), so the
  // idle "Resume" strip can still say why the run stopped.
  const [lastFailureMessage, setLastFailureMessage] = useState<string | null>(null);
  const [resuming, setResuming] = useState(false);
  const hasCheckedInitialIngestion = useRef(false);

  const [reindexingId, setReindexingId] = useState<number | null>(null);
  const [reindexNotice, setReindexNotice] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null);


  // Folder-related state
  const [selectedFolderId, setSelectedFolderId] = useState<number | null>(null);
  const [selectedFolderPath, setSelectedFolderPath] = useState<string>('');
  const [showCreateFolderModal, setShowCreateFolderModal] = useState(false);
  const [showRenameFolderModal, setShowRenameFolderModal] = useState(false);
  const [showDeleteFolderModal, setShowDeleteFolderModal] = useState(false);
  const [showMoveFolderModal, setShowMoveFolderModal] = useState(false);
  const [folderActionData, setFolderActionData] = useState<{
    folderId?: number;
    folderName?: string;
    parentFolderId?: number;
  }>({});
  const [folderTreeVersion, setFolderTreeVersion] = useState(0);
  const [newFolderName, setNewFolderName] = useState('');
  const [newParentFolderId, setNewParentFolderId] = useState<number | null>(null);

  // Media 
  const [showMediaUploadModal, setShowMediaUploadModal] = useState(false);

  // Only one blocking ingest dialog at a time. Upload Media, the CSV import
  // stepper and the CSV review each cover the whole page; letting two be open
  // buries one under the other (the review's confirm modal "disappears"
  // behind Upload Media) and turns clicks into pass-throughs.
  const closeIngestDialogs = () => {
    setShowMediaUploadModal(false);
    setShowCsvImportModal(false);
    setShowImportReviewModal(false);
  };
  const [mediaFiles, setMediaFiles] = useState<File[]>([]);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [uploadType, setUploadType] = useState<'file' | 'youtube'>('file');
  const pollingIntervalsRef = useRef<Map<number, NodeJS.Timeout>>(new Map());
  const [mediaConfig, setMediaConfig] = useState({
    forced_language: '',
    chunk_min_duration: 30,
    chunk_max_duration: 120,
    chunk_overlap: 5
  });

  // Media delete / move state
  const [showDeleteMediaModal, setShowDeleteMediaModal] = useState(false);
  const [mediaToDelete, setMediaToDelete] = useState<Media | null>(null);

  const [showMoveMediaModal, setShowMoveMediaModal] = useState(false);
  const [mediaToMove, setMediaToMove] = useState<Media | null>(null);
  const [moveMediaToFolderId, setMoveMediaToFolderId] = useState<number | null>(null);

  useEffect(() => {
    if (appId && repositoryId) {
      void loadRepository();
    }
  }, [appId, repositoryId]);

  // On page load only, query the backend to check whether an indexing session
  // is already running (e.g. the user navigating to the page while a
  // background index started elsewhere is in progress). Actions started from
  // this page (manual upload, CSV import confirm) already know their own
  // session_id and call setIngestionSessionId directly — this effect must NOT
  // re-run when that local state changes, or it races the background thread
  // that registers the session (IngestionProgressManager.create_session runs
  // after the thread starts, not before this component receives the id) and
  // can wrongly null out a session we just started.
  useEffect(() => {
    if (!repository || !appId || !repositoryId || hasCheckedInitialIngestion.current) return;
    hasCheckedInitialIngestion.current = true;

    let mounted = true;
    (async () => {
      try {
        const status = await apiService.getIngestionStatus(
          Number.parseInt(appId!),
          Number.parseInt(repositoryId!),
        );
        if (!mounted) return;
        setResumable(status.resumable ?? 0);
        if (status.is_indexing && status.active_session_id) {
          setIsIndexing(true);
          setIngestionSessionId(status.active_session_id);
        }
      } catch (err) {
        console.error('Error fetching ingestion status:', err);
      }
    })();

    return () => {
      mounted = false;
    };
  }, [repository, appId, repositoryId]);

  // The SSE stream reports a run's end before the 3s poll notices, so onComplete
  // would clear isIndexing while resumable was still 0 — and with both false the
  // poll below stops, leaving nothing to ever fetch the new count. The Resume
  // strip then stayed invisible until an F5. Ask the backend instead of assuming.
  const refreshResumable = async () => {
    try {
      const status = await apiService.getIngestionStatus(
        Number.parseInt(appId!),
        Number.parseInt(repositoryId!),
      );
      setResumable(status.resumable ?? 0);
    } catch {
      // Nothing to recover: a run still alive keeps the poll below going.
    }
  };

  // Poll while a run is alive OR while something is resumable, so the badges and
  // the resume button stay current without an F5.  Liveness is the backend's
  // answer, not our reading of the row statuses.
  useEffect(() => {
    if (!isIndexing && resumable === 0) return;

    const intervalId = setInterval(async () => {
      try {
        const [data, status] = await Promise.all([
          apiService.getRepository(Number.parseInt(appId!), Number.parseInt(repositoryId!)),
          apiService.getIngestionStatus(Number.parseInt(appId!), Number.parseInt(repositoryId!)),
        ]);
        setRepository(data);
        setResumable(status.resumable ?? 0);
        setIsIndexing(status.is_indexing);
        if (!status.is_indexing) {
          setIngestionSessionId(null);
        } else if (status.active_session_id && status.active_session_id !== ingestionSessionId) {
          // Covers a stale local session id that would leave the bar on
          // "Connecting…" while indexing progresses under another one.
          setIngestionSessionId(status.active_session_id);
        }
      } catch {
        // transient errors should not kill polling
      }
    }, 3000);

    return () => clearInterval(intervalId);
  }, [isIndexing, resumable, ingestionSessionId, appId, repositoryId]);

  useEffect(() => {
    return () => {
      pollingIntervalsRef.current.forEach(interval => clearInterval(interval));
    };
  }, []);

  const startPolling = (mediaId: number) => {
    if (pollingIntervalsRef.current.has(mediaId)) return;

    const interval = setInterval(async () => {
      try {
        const media = await apiService.getMediaStatus(Number.parseInt(appId!), Number.parseInt(repositoryId!), mediaId);
        setRepository(prev => updateMediaInRepo(prev, mediaId, media));
        if (isTerminalMediaStatus(media.status)) {
          stopPolling(mediaId);
        }
      } catch (err) {
        console.error('Polling error:', err);
        stopPolling(mediaId);
      }
    }, 3000);

    pollingIntervalsRef.current.set(mediaId, interval);
  };

  const stopPolling = (mediaId: number) => {
    const interval = pollingIntervalsRef.current.get(mediaId);
    if (interval) {
      clearInterval(interval);
      pollingIntervalsRef.current.delete(mediaId);
    }
  };

  const loadRepository = async (clearError: boolean = true) => {
    // Only show the full-page spinner on the very first load (no data yet).
    // Background refreshes (after upload, after indexing completes) update
    // silently so the page never flickers/blinks.
    const isFirstLoad = repository === null;
    try {
      if (isFirstLoad) setLoading(true);
      const data = await apiService.getRepository(Number.parseInt(appId!), Number.parseInt(repositoryId!));
      setRepository(data);

      // Start polling for processing media
      data.media?.forEach((m: Media) => {
        if (m.status !== 'ready' && m.status !== 'error' && !pollingIntervalsRef.current.has(m.media_id)) {
          startPolling(m.media_id);
        }
      });

      if (clearError) setError(null);
    } catch (err) {
      console.error('Error loading repository:', err);
      setError(t('repositoryDetailPage.errors.loadRepository'));
    } finally {
      if (isFirstLoad) setLoading(false);
    }
  };

  // The silo's entity-type config is only needed when the estimate modal is up,
  // so it is fetched then rather than on every page load.
  useEffect(() => {
    if (!showEstimateModal || !isLightRAG || !repository?.silo_id) return;
    let cancelled = false;
    (async () => {
      try {
        const silo = await apiService.getSilo(Number.parseInt(appId!), repository.silo_id!) as {
          lightrag_entity_types_mode?: string; lightrag_entity_types?: string | null; lightrag_config_locked?: boolean;
        };
        if (!cancelled) {
          setSiloEntityTypes({
            mode: silo.lightrag_entity_types_mode ?? 'manual',
            types: silo.lightrag_entity_types ?? null,
            locked: !!silo.lightrag_config_locked,
          });
        }
      } catch {
        // Not fatal: without the config the gate simply does not appear.
        if (!cancelled) setSiloEntityTypes(null);
      }
    })();
    return () => { cancelled = true; };
  }, [showEstimateModal, isLightRAG, repository?.silo_id, appId]);

  const entityTypesSettled = !!siloEntityTypes?.types?.trim();
  // Once a silo has been indexed, entity types are immutable — the gate has
  // nothing left to do and must stop appearing on every later ingest.
  const entityTypesGateRequired = !!siloEntityTypes && siloEntityTypes.mode === 'infer' && !siloEntityTypes.locked;

  const applyInferredEntityTypes = async (entityTypes: string) => {
    if (!repository?.silo_id) return;
    try {
      await apiService.updateSilo(Number.parseInt(appId!), repository.silo_id, {
        name: repository.name,
        lightrag_entity_types: entityTypes,
      });
      setSiloEntityTypes({ mode: 'infer', types: entityTypes, locked: false });
      setShowEntityTypesModal(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : t('lightrag.repositoryDetailPage.entityTypesSaveError'));
    }
  };

  const resetPendingUploadEstimate = () => {
    setShowEstimateModal(false);
    setShowEntityTypesModal(false);
    setPendingUploadFiles([]);
    setUploadEstimate(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const performFileUpload = async (filesToUpload: File[]) => {
    try {
      setUploading(true);
      setError(null);

      const result = await apiService.uploadResources(
        Number.parseInt(appId!),
        Number.parseInt(repositoryId!),
        filesToUpload,
        selectedFolderId || undefined,
      );

      const problems: string[] = [];
      // Files rejected before upload
      if (result.failed_files && result.failed_files.length > 0) {
        const failedMessages = result.failed_files.map((failed: { filename: string; error: string }) =>
          `${failed.filename}: ${failed.error}`
        ).join('\n');
        const allFailed = !result.created_resources || result.created_resources.length === 0;
        problems.push(t(allFailed ? 'repositoryDetailPage.errors.uploadFailed' : 'repositoryDetailPage.errors.someFilesFailed', { messages: failedMessages }));
      }
      // Uploaded but not indexed (e.g. the embedding service rejected its API key)
      if (result.indexing_failures && result.indexing_failures.length > 0) {
        const indexingMessages = result.indexing_failures.map((failed: { filename: string; error: string }) =>
          `${failed.filename}: ${failed.error}`
        ).join('\n');
        problems.push(t('repositoryDetailPage.errors.indexingFailed', { messages: indexingMessages }));
      }
      if (problems.length > 0) {
        setError(problems.join('\n\n'));
      }

      // Start monitoring ingestion progress if the backend returned a session_id
      if (result.session_id) {
        setIngestionSessionId(result.session_id);
        setIsIndexing(true);
      }

      await loadRepository(false);
      resetPendingUploadEstimate();
    } catch (err: any) {
      console.error('Error uploading files:', err);
      setError(err.message || t('repositoryDetailPage.errors.uploadFiles'));
    } finally {
      setUploading(false);
    }
  };

  const handleAzureBlobIngested = (result: IngestAzureBlobsResult) => {
    const found = result.total_blobs;
    if (result.pending_blobs === 0) {
      toast.success(t('lightrag.repositoryDetailPage.azure.upToDate', { count: found }));
    } else if (result.failed > 0) {
      const capNote = result.pending_blobs > AZURE_BLOB_SAMPLE_SIZE
        ? t('lightrag.repositoryDetailPage.azure.capNote', { sample: AZURE_BLOB_SAMPLE_SIZE })
        : '';
      toast.warning(
        t('lightrag.repositoryDetailPage.azure.partial', { count: result.queued, failed: result.failed, capNote }),
        { duration: 8000 },
      );
    } else if (result.pending_blobs > AZURE_BLOB_SAMPLE_SIZE) {
      toast.success(
        t('lightrag.repositoryDetailPage.azure.capped', { found, sample: AZURE_BLOB_SAMPLE_SIZE }),
        { duration: 8000 },
      );
    } else {
      toast.success(t('lightrag.repositoryDetailPage.azure.loading', { count: found, queued: result.queued }));
    }

    if (result.session_id) {
      setIngestionSessionId(result.session_id);
      setIsIndexing(true);
    }

    loadRepository(false);
  };

  const handleAzureBlobUpdate = async () => {
    const source = repository?.azure_blob_source;
    if (!source) return;

    if (source.auth_mode === 'SAS_TOKEN') {
      // The token is never stored, so a SAS-token source always re-asks for
      // it in the modal before an update can run.
      setAzureBlobUpdateMode(true);
      setShowAzureBlobModal(true);
      return;
    }

    try {
      setError(null);
      setAzureBlobRunning(true);
      const result = await apiService.ingestAzureBlobs(
        Number.parseInt(appId!),
        Number.parseInt(repositoryId!),
        {
          account_url: source.account_url,
          container: source.container,
          prefixes: source.prefixes ?? null,
          name_excludes: source.name_excludes ?? null,
          auth_mode: source.auth_mode,
          sample_size: AZURE_BLOB_SAMPLE_SIZE,
        },
      );
      handleAzureBlobIngested(result);
    } catch (err: any) {
      console.error('Error updating from Azure Blob:', err);
      setError(err.message || t('lightrag.repositoryDetailPage.azure.updateFailed'));
    } finally {
      setAzureBlobRunning(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const selectedFiles = Array.from(files);
    const isLightRagRepository = repository?.vector_db_type?.toUpperCase() === 'LIGHTRAG';

    try {
      setError(null);

      if (!isLightRagRepository) {
        await performFileUpload(selectedFiles);
        return;
      }

      setEstimatingUpload(true);
      const estimate = await apiService.estimateUploadResources(
        Number.parseInt(appId!),
        Number.parseInt(repositoryId!),
        selectedFiles,
        selectedFolderId || undefined,
      );

      setPendingUploadFiles(selectedFiles);
      setUploadEstimate(estimate);
      closeIngestDialogs();
      setShowEstimateModal(true);
    } catch (err: any) {
      console.error('Error estimating upload:', err);
      setError(err.message || t('lightrag.repositoryDetailPage.estimateFailed'));
      resetPendingUploadEstimate();
    } finally {
      setEstimatingUpload(false);
    }
  };

  const confirmEstimatedUpload = async () => {
    if (pendingUploadFiles.length === 0) return;
    await performFileUpload(pendingUploadFiles);
  };

  const handleMediaUpload = async () => {
    try {
      setUploading(true);
      setError(null);

      if (uploadType === 'file') {
        const result = await apiService.uploadMedia(
          Number.parseInt(appId!),
          Number.parseInt(repositoryId!),
          mediaFiles,
          selectedFolderId || undefined,
          mediaConfig,
        );
        
        if (result.failed_files && result.failed_files.length > 0) {
          const failedMessages = result.failed_files.map((f: any) => 
            `${f.filename}: ${f.error}`
          ).join('\n');
          setError(t('repositoryDetailPage.errors.someMediaFilesFailed', { messages: failedMessages }));
        }
      } else {
        await apiService.addYouTube(
          Number.parseInt(appId!),
          Number.parseInt(repositoryId!),
          youtubeUrl,
          selectedFolderId || undefined,
          mediaConfig,
        );
      }

      setShowMediaUploadModal(false);
      setMediaFiles([]);
      setYoutubeUrl('');
      await loadRepository(false);
    } catch (err: any) {
      setError(err.message || t('repositoryDetailPage.errors.uploadMedia'));
    } finally {
      setUploading(false);
    }
  };

  // Folder management functions
  const handleFolderSelect = (folderId: number | null, folderPath: string) => {
    setSelectedFolderId(folderId);
    setSelectedFolderPath(folderPath);
  };

  const handleCreateFolder = (parentFolderId?: number) => {
    setFolderActionData({ parentFolderId });
    setNewFolderName('');
    setShowCreateFolderModal(true);
  };

  const handleRenameFolder = (folderId: number, currentName: string) => {
    setFolderActionData({ folderId, folderName: currentName });
    setNewFolderName(currentName);
    setShowRenameFolderModal(true);
  };

  const handleDeleteFolder = (folderId: number, folderName: string) => {
    setFolderActionData({ folderId, folderName });
    setShowDeleteFolderModal(true);
  };

  const handleMoveFolder = (folderId: number, currentParentId: number | undefined, folderName: string) => {
    setFolderActionData({ folderId, parentFolderId: currentParentId, folderName });
    setNewParentFolderId(null); // Start with no selection so user can choose
    setShowMoveFolderModal(true);
  };

  const createFolder = async () => {
    if (!newFolderName.trim()) return;

    try {
      await apiService.createFolder(
        Number.parseInt(appId!),
        Number.parseInt(repositoryId!),
        newFolderName.trim(),
        folderActionData.parentFolderId
      );
      setShowCreateFolderModal(false);
      setNewFolderName('');
      setFolderTreeVersion(v => v + 1);
      await loadRepository();
    } catch (err: any) {
      console.error('Error creating folder:', err);
      const errorMessage = err?.response?.data?.detail || err?.message || t('repositoryDetailPage.errors.createFolder');
      setError(errorMessage);
    }
  };

  const renameFolder = async () => {
    if (!newFolderName.trim() || !folderActionData.folderId) return;

    try {
      await apiService.updateFolder(
        Number.parseInt(appId!),
        Number.parseInt(repositoryId!),
        folderActionData.folderId,
        newFolderName.trim()
      );
      setShowRenameFolderModal(false);
      setNewFolderName('');
      setFolderTreeVersion(v => v + 1);
      await loadRepository();
    } catch (err: any) {
      console.error('Error renaming folder:', err);
      const errorMessage = err?.response?.data?.detail || err?.message || t('repositoryDetailPage.errors.renameFolder');
      setError(errorMessage);
    }
  };

  const deleteFolder = async () => {
    if (!folderActionData.folderId) return;

    try {
      await apiService.deleteFolder(
        Number.parseInt(appId!),
        Number.parseInt(repositoryId!),
        folderActionData.folderId
      );
      setShowDeleteFolderModal(false);
      // Reset selection if deleted folder was selected
      if (selectedFolderId === folderActionData.folderId) {
        setSelectedFolderId(null);
        setSelectedFolderPath('');
      }
      setFolderTreeVersion(v => v + 1);
      await loadRepository();
    } catch (err: any) {
      console.error('Error deleting folder:', err);
      const errorMessage = err?.response?.data?.detail || err?.message || t('repositoryDetailPage.errors.deleteFolder');
      setError(errorMessage);
    }
  };

  const moveFolder = async () => {
    if (!folderActionData.folderId) return;

    try {
      await apiService.moveFolder(
        Number.parseInt(appId!),
        Number.parseInt(repositoryId!),
        folderActionData.folderId,
        newParentFolderId || undefined
      );
      setShowMoveFolderModal(false);
      setNewParentFolderId(null);
      setFolderTreeVersion(v => v + 1);
      await loadRepository();
    } catch (err: any) {
      console.error('Error moving folder:', err);
      const errorMessage = err?.response?.data?.detail || err?.message || t('repositoryDetailPage.errors.moveFolder');
      setError(errorMessage);
    }
  };

  // Get valid destination folders (exclude the folder being moved and its descendants)
  const getValidDestinationFolders = () => {
    if (!repository?.folders || !folderActionData.folderId) return [];
    const allFolders = getAllFolderPaths(repository.folders);
    // Backend validates circular references; frontend just excludes the moved folder itself
    return allFolders.filter(folder => folder.folder_id !== folderActionData.folderId);
  };

  // Filter resources by selected folder
  const filteredResources = filterByFolder(repository?.resources, selectedFolderId);

  // Filter media by selected folder
  const filteredMedia = filterByFolder(repository?.media, selectedFolderId);

  // Liveness comes from the backend (the silo's advisory lock), never from row
  // statuses: an interrupted run leaves resources in 'pending'/'indexing'
  // forever, and deriving "busy" from that would block uploads for good and
  // hide the resume button.
  const indexingInProgress = isIndexing;
  const uploadDisabled = uploading || estimatingUpload || indexingInProgress;

  // Direct subfolders of the selected folder, listed as rows in the files table
  const subfolders = (repository?.folders ?? []).filter(
    f => (f.parent_folder_id ?? null) === selectedFolderId
  );
  const folderTrail = getFolderTrail(repository?.folders, selectedFolderId);
  const countFolderItems = (folderId: number) =>
    (repository?.resources ?? []).filter(r => r.folder_id === folderId).length +
    (repository?.media ?? []).filter(m => m.folder_id === folderId).length;

  const isEmpty =
  subfolders.length === 0 && filteredResources.length === 0 && filteredMedia.length === 0;

  const handleResumeIngestion = async () => {
    if (resuming) return;
    setResuming(true);
    try {
      const { session_id } = await apiService.resumeIngestion(
        Number.parseInt(appId!),
        Number.parseInt(repositoryId!),
      );
      if (session_id) {
        setIngestionSessionId(session_id);
        setIsIndexing(true);
        setResumable(0);
        setLastFailureMessage(null);
      }
      await loadRepository(false);
    } catch (err) {
      setReindexNotice({
        type: 'error',
        text: err instanceof Error ? err.message : t('lightrag.repositoryDetailPage.resumeError'),
      });
    } finally {
      setResuming(false);
    }
  };

  const handleDownloadResource = async (resource: Resource) => {
    try {
      const response = await apiService.downloadResource(Number.parseInt(appId!), Number.parseInt(repositoryId!), resource.resource_id);

      // Create a blob from the response and download it
      const blob = new Blob([response], { type: 'application/octet-stream' });
      const url = globalThis.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = resource.uri; // Use uri which contains the filename with extension
      document.body.appendChild(a);
      a.click();
      globalThis.URL.revokeObjectURL(url);
      a.remove();
    } catch (err) {
      console.error('Error downloading resource:', err);
      setError(t('repositoryDetailPage.errors.downloadFile'));
    }
  };

  const handleDeleteResource = (resource: Resource) => {
    setResourceToDelete(resource);
    setDeleteError(null);
    setDeleteFailures(0);
    setShowDeleteModal(true);
  };

  const handleMoveResource = (resource: Resource) => {
    setResourceToMove(resource);
    setMoveToFolderId(null);
    setShowMoveModal(true);
  };

  const handleReindexResource = async (resource: Resource) => {
    if (reindexingId !== null) return;
    const siloId = repository?.silo_id;
    if (!siloId) {
      setReindexNotice({ type: 'error', text: t('repositoryDetailPage.errors.repositoryNotLinked') });
      return;
    }
    setReindexingId(resource.resource_id);
    setReindexNotice(null);
    try {
      await apiService.reindexSiloResource(Number.parseInt(appId!), siloId, resource.resource_id);
      setReindexNotice({ type: 'success', text: t('repositoryDetailPage.reindex.success', { name: resource.name }) });
    } catch (err) {
      setReindexNotice({ type: 'error', text: err instanceof Error ? err.message : t('repositoryDetailPage.errors.reindex') });
    } finally {
      await loadRepository(false);
      setReindexingId(null);
      setTimeout(() => setReindexNotice(null), 4000);
    }
  };

  const confirmDeleteResource = async () => {
    if (!resourceToDelete || deletingResource) return;

    setDeletingResource(true);
    setDeleteError(null);
    try {
      await apiService.deleteResource(Number.parseInt(appId!), Number.parseInt(repositoryId!), resourceToDelete.resource_id);
      await loadRepository(); // Reload to update the list
      setShowDeleteModal(false);
      setResourceToDelete(null);
    } catch (err) {
      console.error('Error deleting resource:', err);
      setDeleteError(err instanceof Error && err.message ? err.message : t('repositoryDetailPage.errors.deleteFile'));
      setDeleteFailures((n) => n + 1);
    } finally {
      setDeletingResource(false);
    }
  };

  const confirmDeleteAllResources = async () => {
    setDeletingAll(true);
    setDeleteAllError(null);
    try {
      const result = await apiService.deleteAllResources(Number.parseInt(appId!), Number.parseInt(repositoryId!));
      await loadRepository();
      if (result.failed_count > 0) {
        setDeleteAllError(t('lightrag.repositoryDetailPage.deleteAll.partialFailure', { count: result.failed_count, deleted: result.deleted_count }));
      } else {
        setShowDeleteAllModal(false);
      }
    } catch (err) {
      console.error('Error deleting all resources:', err);
      setDeleteAllError(err instanceof Error && err.message ? err.message : t('lightrag.repositoryDetailPage.deleteAll.failed'));
      await loadRepository(); // a partial failure may have removed some files
    } finally {
      setDeletingAll(false);
    }
  };

  const confirmMoveResource = async () => {
    if (!resourceToMove) return;

    try {
      await apiService.moveResource(
        Number.parseInt(appId!),
        Number.parseInt(repositoryId!),
        resourceToMove.resource_id,
        moveToFolderId || undefined
      );

      // Reload repository to get updated file list
      await loadRepository();

      setShowMoveModal(false);
      setResourceToMove(null);
      setMoveToFolderId(null);
    } catch (err) {
      console.error('Error moving resource:', err);
      setError(t('repositoryDetailPage.errors.moveFile'));
    }
  };

  const handleDeleteMedia = (media: Media) => {
    setMediaToDelete(media);
    setShowDeleteMediaModal(true);
  };

  const confirmDeleteMedia = async () => {
    if (!mediaToDelete) return;

    try {
      stopPolling(mediaToDelete.media_id);  // Add this - stop polling before delete

      await apiService.deleteMedia(
        Number.parseInt(appId!),
        Number.parseInt(repositoryId!),
        mediaToDelete.media_id
      );
      await loadRepository();
      setShowDeleteMediaModal(false);
      setMediaToDelete(null);
    } catch (err) {
      console.error(err);
      setError(t('repositoryDetailPage.errors.deleteMedia'));
    }
  };

  const handleMoveMedia = (media: Media) => {
    setMediaToMove(media);
    setMoveMediaToFolderId(null);
    setShowMoveMediaModal(true);
  };

  const confirmMoveMedia = async () => {
    if (!mediaToMove) return;

    try {
      await apiService.moveMedia(
        Number.parseInt(appId!),
        Number.parseInt(repositoryId!),
        mediaToMove.media_id,
        moveMediaToFolderId || undefined
      );
      await loadRepository();
      setShowMoveMediaModal(false);
      setMediaToMove(null);
      setMoveMediaToFolderId(null);
    } catch (err) {
      console.error(err);
      setError(t('repositoryDetailPage.errors.moveMedia'));
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString(i18n.resolvedLanguage);
  };

  if (loading) {
    return (
      <LoadingState variant="page" />
    );
  }

  if (!repository) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-fg dark:text-fg-dark mb-4">{t('repositoryDetailPage.notFound')}</h1>
          <ButtonApp
            label={t('repositoryDetailPage.backToRepositories')}
            variant="secondary"
            size="medium"
            onClick={() => navigate(`/apps/${appId}/repositories`)}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <Title
          titulo={repository.name}
          subtitulo={t('repositoryDetailPage.createdSummary', { date: formatDate(repository.created_at), count: repository.resources.length })}
          subtitleMarginBottom={0}
        />
        {!!repository.total_indexing_duration_seconds && (
          <p className="text-sm text-fg-tertiary dark:text-fg-tertiary-dark mt-1">
            {t('lightrag.repositoryDetailPage.totalIndexingTime', { duration: formatDuration(repository.total_indexing_duration_seconds, 0) })}
          </p>
        )}
        {(repository.transcription_service_id || repository.video_ai_service_id) && (
          <p className="text-sm text-fg-tertiary dark:text-fg-tertiary-dark mt-1">
            {repository.transcription_service_id && (
              <span className="inline-flex items-center gap-1 mr-3">
                <Mic className="w-4 h-4" aria-hidden="true" /> <span>{t('repositoryDetailPage.services.transcription')}: {repository.ai_services.find(s => s.service_id === repository.transcription_service_id)?.name ?? t('repositoryDetailPage.services.serviceNumber', { id: repository.transcription_service_id })}</span>
              </span>
            )}
            {repository.video_ai_service_id && (
              <span className="inline-flex items-center gap-1">
                <Clapperboard className="w-4 h-4" aria-hidden="true" /> <span>{t('repositoryDetailPage.services.videoAi')}: {repository.ai_services.find(s => s.service_id === repository.video_ai_service_id)?.name ?? t('repositoryDetailPage.services.serviceNumber', { id: repository.video_ai_service_id })}</span>
              </span>
            )}
          </p>
        )}
      </div>

      {!canEdit && <ReadOnlyBanner userRole={userRole} minRole={AppRole.EDITOR} />}

      <CsvImportStepper
        appId={Number.parseInt(appId!)}
        repositoryId={Number.parseInt(repositoryId!)}
        isOpen={showCsvImportModal}
        onClose={() => setShowCsvImportModal(false)}
        onImportStarted={(job) => setActiveImportJob(job)}
      />
      {activeImportJob && (
        <CsvImportBanner job={activeImportJob} onReview={() => { closeIngestDialogs(); setShowImportReviewModal(true); }} />
      )}
      {activeImportJob && (
        <CsvImportReviewModal
          appId={Number.parseInt(appId!)}
          repositoryId={Number.parseInt(repositoryId!)}
          isLightRagRepository={isLightRAG}
          siloId={repository?.silo_id ?? null}
          job={activeImportJob}
          isOpen={showImportReviewModal}
          onClose={() => setShowImportReviewModal(false)}
          onJobUpdated={(updated) => { setActiveImportJob(updated); if (!updated) setShowImportReviewModal(false); }}
          onIngestionStarted={(sessionId) => { setIngestionSessionId(sessionId); setIsIndexing(true); setLastFailureMessage(null); }}
        />
      )}

      <AzureBlobIngestModal
        isOpen={showAzureBlobModal}
        onClose={() => setShowAzureBlobModal(false)}
        appId={Number.parseInt(appId!)}
        repositoryId={Number.parseInt(repositoryId!)}
        savedSource={repository?.azure_blob_source ?? null}
        isUpdate={azureBlobUpdateMode}
        onIngested={handleAzureBlobIngested}
      />

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        onChange={handleFileUpload}
        className="hidden"
        accept=".pdf,.docx,.txt"
      />

      {/* Error Message */}
      {error && <Alert type="error" message={error} onDismiss={() => setError(null)} className="mb-6" />}
      {estimatingUpload && (
        <Alert
          type="info"
          message={t('lightrag.repositoryDetailPage.estimatingCost')}
          className="mb-6"
        />
      )}

      {/* Ingestion strip — the running progress bar, or the Resume control when
          a stopped run left files behind. Both live here so the ingestion
          controls are never split across the page. */}
      {(ingestionSessionId || resumable > 0) && (
        <div className="mb-6">
          <IngestionProgressBar
            appId={Number.parseInt(appId!)}
            repositoryId={Number.parseInt(repositoryId!)}
            sessionId={ingestionSessionId}
            resumable={resumable}
            resuming={resuming}
            onResume={handleResumeIngestion}
            onFailed={setLastFailureMessage}
            lastFailureMessage={lastFailureMessage}
            onComplete={() => {
              setIngestionSessionId(null);
              setIsIndexing(false);
              loadRepository(false);
              // Files a stop left behind only surface through this count.
              refreshResumable();
              // The "Pausing…"/"Cancelling…" notice from onStopped below
              // otherwise outlives the run it was reporting on.
              setReindexNotice(null);
            }}
            onStopped={(mode, stopped) => {
              // Neither mode ends the run instantly, and they differ in what
              // survives: pause drops the files in flight (LightRAG aborts them
              // at their next checkpoint) and keeps the untouched files, while
              // cancel does the opposite. Only cancel states a count — removal
              // is the surprise; for pause the strip and the Resume button
              // already say how many files are left.
              // Future tense on purpose: removal happens once the run truly
              // ends (the discard step, or the liveness sweep if the run dies
              // first), not at the instant this click lands.
              const removed = stopped > 0
                ? ` ${t('lightrag.repositoryDetailPage.notices.notStartedRemoved', { count: stopped })}` : '';
              setReindexNotice({
                type: 'info',
                text: mode === 'pause'
                  ? t('lightrag.repositoryDetailPage.notices.pausing')
                  : `${t('lightrag.repositoryDetailPage.notices.cancelling')}${removed} ${t('lightrag.repositoryDetailPage.notices.nothingDeleted')}`,
              });
            }}
          />
        </div>
      )}
      {reindexNotice && <Alert type={reindexNotice.type} message={reindexNotice.text} onDismiss={() => setReindexNotice(null)} className="mb-6" />}
      <ServiceLifecycleAlert warnings={repository?.lifecycle_warnings} variant="full" className="mb-6" />
      {repository?.service_warnings?.map((warning) => (
        <Alert
          key={warning}
          type="warning"
          title={t('repositoryDetailPage.serviceWarningTitle')}
          message={t('repositoryDetailPage.serviceWarningMessage', { warning })}
          className="mb-6"
        />
      ))}

      {/* Explorer: folder tree + files */}
      <div className="grid grid-cols-1 lg:grid-cols-[240px_minmax(0,1fr)] bg-surface border border-line rounded-lg dark:bg-surface-dark dark:border-line-dark">
        {/* Folder Tree Sidebar */}
        <div className="bg-surface-tint border-b border-line rounded-t-lg lg:border-b-0 lg:border-r lg:rounded-tr-none lg:rounded-l-lg dark:bg-surface-tint-dark dark:border-line-dark">
          <FolderTree
            appId={Number.parseInt(appId!)}
            repositoryId={Number.parseInt(repositoryId!)}
            selectedFolderId={selectedFolderId ?? undefined}
            onFolderSelect={handleFolderSelect}
            onFolderCreate={handleCreateFolder}
            onFolderRename={handleRenameFolder}
            onFolderDelete={handleDeleteFolder}
            onFolderMove={handleMoveFolder}
            canEdit={canEdit}
            refreshToken={folderTreeVersion}
          />
        </div>

        {/* Files + Media Section */}
        <div className="min-w-0">
          {/* Toolbar: breadcrumb + actions */}
          <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3 border-b border-line dark:border-line-dark">
            <nav aria-label={t('repositoryDetailPage.folderPath')} className="flex flex-wrap items-center gap-1 min-w-0 text-sm">
              <button
                onClick={() => handleFolderSelect(null, '')}
                aria-current={selectedFolderId === null ? 'page' : undefined}
                className={`inline-flex items-center gap-1.5 transition-colors ${
                  selectedFolderId === null
                    ? 'font-medium text-fg dark:text-fg-dark'
                    : 'text-fg-tertiary hover:text-accent dark:text-fg-tertiary-dark dark:hover:text-accent-dark'
                }`}
              >
                <Home className="w-4 h-4" /> {t('repositoryDetailPage.root')}
              </button>
              {folderTrail.map((crumb, index) => {
                const isCurrent = index === folderTrail.length - 1;
                return (
                  <React.Fragment key={crumb.folder_id}>
                    <ChevronRight className="w-3.5 h-3.5 text-fg-faint2 dark:text-fg-faint2-dark flex-shrink-0" />
                    <button
                      onClick={() => handleFolderSelect(crumb.folder_id, crumb.path)}
                      aria-current={isCurrent ? 'page' : undefined}
                      className={`truncate transition-colors ${
                        isCurrent
                          ? 'font-medium text-fg dark:text-fg-dark'
                          : 'text-fg-tertiary hover:text-accent dark:text-fg-tertiary-dark dark:hover:text-accent-dark'
                      }`}
                    >
                      {crumb.name}
                    </button>
                  </React.Fragment>
                );
              })}
              <span className="ml-1 text-fg-faint dark:text-fg-faint-dark">
                · {t('repositoryDetailPage.files', { count: filteredResources.length + filteredMedia.length })}
              </span>
            </nav>

            <div className="flex flex-wrap items-center gap-2">
              <ButtonApp
                label={t('repositoryDetailPage.actions.search')}
                variant="muted"
                size="control"
                icon={<Search className="w-4 h-4" />}
                onClick={() => navigate(`/apps/${appId}/repositories/${repositoryId}/playground`)}
              />
              {canEdit && (
                <>
                  <ButtonApp
                    label={t('repositoryDetailPage.actions.uploadMedia')}
                    variant="secondary"
                    size="control"
                    icon={<Video className="w-4 h-4" />}
                    onClick={() => {
                      closeIngestDialogs();
                      setShowMediaUploadModal(true);
                    }}
                    disabled={uploading}
                  />
                  <ButtonApp
                    label={t('repositoryDetailPage.actions.uploadFiles')}
                    variant="primary"
                    size="control"
                    icon={<FolderOpen className="w-4 h-4" />}
                    onClick={() => fileInputRef.current?.click()}
                    loading={uploading}
                    disabled={uploadDisabled}
                  />
                  <ButtonApp
                    label={t('lightrag.repositoryDetailPage.actions.importCsv')}
                    variant="secondary"
                    size="control"
                    onClick={() => {
                      closeIngestDialogs();
                      setShowCsvImportModal(true);
                    }}
                    disabled={uploading || indexingInProgress}
                  />
                  <ButtonApp
                    label={t('lightrag.repositoryDetailPage.actions.loadAzureBlob')}
                    variant="secondary"
                    size="control"
                    icon={<Cloud className="w-4 h-4" />}
                    onClick={() => {
                      setAzureBlobUpdateMode(false);
                      setShowAzureBlobModal(true);
                    }}
                    disabled={uploading || indexingInProgress || azureBlobRunning}
                  />
                  {repository?.azure_blob_source && (
                    <ButtonApp
                      label={t('lightrag.repositoryDetailPage.actions.updateAzureBlob')}
                      variant="secondary"
                      size="control"
                      icon={<RefreshCw className="w-4 h-4" />}
                      onClick={handleAzureBlobUpdate}
                      disabled={uploading || indexingInProgress || azureBlobRunning}
                      loading={azureBlobRunning}
                    />
                  )}
                  <ButtonApp
                    label={t('lightrag.repositoryDetailPage.actions.deleteAll')}
                    variant="secondary"
                    size="control"
                    icon={<Trash2 className="w-4 h-4" />}
                    onClick={() => {
                      setDeleteAllError(null);
                      setShowDeleteAllModal(true);
                    }}
                    disabled={uploading || (!isLightRAG && repository.resources.length === 0)}
                  />
                </>
              )}
            </div>
          </div>

          {/* SHARED EMPTY STATE */}
          {isEmpty ? (
            <div className="p-10 text-center">
              <div className="flex justify-center gap-6 mb-4">
                <FileText className="w-10 h-10 text-fg-faint2 dark:text-fg-faint2-dark" />
                <Video className="w-10 h-10 text-fg-faint2 dark:text-fg-faint2-dark" />
              </div>
              <h3 className="font-display text-lg font-normal tracking-[-0.02em] text-fg dark:text-fg-dark mb-1">
                {selectedFolderId === null
                  ? t('repositoryDetailPage.empty.rootTitle')
                  : t('repositoryDetailPage.empty.folderTitle')}
              </h3>

              <p className="text-sm text-fg-tertiary dark:text-fg-tertiary-dark">
                {selectedFolderId === null
                  ? t('repositoryDetailPage.empty.rootDescription')
                  : t('repositoryDetailPage.empty.folderDescription')}
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-line dark:divide-line-dark">
                <thead className="bg-tableCab dark:bg-tableCab-dark">
                  <tr>
                    <th className="px-5 py-3 text-left text-[11px] font-medium uppercase tracking-[0.05em] text-fg-tertiary dark:text-fg-tertiary-dark">{t('repositoryDetailPage.columns.name')}</th>
                    <th className="px-5 py-3 text-left text-[11px] font-medium uppercase tracking-[0.05em] text-fg-tertiary dark:text-fg-tertiary-dark">{t('repositoryDetailPage.columns.type')}</th>
                    <th className="px-5 py-3 text-left text-[11px] font-medium uppercase tracking-[0.05em] text-fg-tertiary dark:text-fg-tertiary-dark">{t('repositoryDetailPage.columns.uploaded')}</th>
                    <th className="px-5 py-3 text-right text-[11px] font-medium uppercase tracking-[0.05em] text-fg-tertiary dark:text-fg-tertiary-dark">
                      <span className="sr-only">{t('repositoryDetailPage.columns.actions')}</span>
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line text-sm text-fg dark:divide-line-dark dark:text-fg-dark">
                  {/* SUBFOLDERS (double-click to open) */}
                  {subfolders.map((folder) => {
                    const folderPath = selectedFolderPath ? `${selectedFolderPath}/${folder.name}` : folder.name;
                    const itemCount = countFolderItems(folder.folder_id);
                    return (
                      <tr
                        key={`folder-${folder.folder_id}`}
                        onDoubleClick={() => handleFolderSelect(folder.folder_id, folderPath)}
                        className="cursor-pointer select-none hover:bg-surface-tint dark:hover:bg-surface-tint-dark"
                      >
                        <td className="px-5 py-3">
                          <button
                            type="button"
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') handleFolderSelect(folder.folder_id, folderPath);
                            }}
                            title={t('repositoryDetailPage.doubleClickToOpen')}
                            className="flex items-center gap-3 text-left"
                          >
                            <BackgroundImage icon={Folder} size={32} iconSize={16} variant="marron" />
                            <span className="font-medium">{folder.name}</span>
                            {itemCount > 0 && (
                              <span className="text-xs text-fg-tertiary dark:text-fg-tertiary-dark">({itemCount})</span>
                            )}
                          </button>
                        </td>
                        <td className="px-5 py-3 whitespace-nowrap text-fg-tertiary dark:text-fg-tertiary-dark">{t('repositoryDetailPage.folderType')}</td>
                        <td className="px-5 py-3 whitespace-nowrap text-fg-tertiary dark:text-fg-tertiary-dark">—</td>
                        <td className="px-5 py-3" />
                      </tr>
                    );
                  })}

                  {/* FILES LIST */}
                  {filteredResources.map((resource) => (
                    <tr key={resource.resource_id} className="hover:bg-surface-tint dark:hover:bg-surface-tint-dark">
                      <td className="px-5 py-3">
                        <div className="flex items-center gap-3 min-w-0">
                          <BackgroundImage icon={FileText} size={32} iconSize={16} variant="marron" />
                          <div className="min-w-0">
                            <div className="flex items-center gap-2 min-w-0 flex-wrap">
                              <span className="font-medium truncate min-w-0" title={resource.name}>
                                {resource.name}
                              </span>
                              {resource.status === 'indexing' && indexingInProgress && (
                                <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-700">
                                  <Loader className="w-3 h-3 animate-spin shrink-0" />
                                  {resource.progress_total ? (
                                    <>
                                      <span className="w-16 h-1.5 rounded-full bg-blue-200 overflow-hidden">
                                        <span
                                          className="block h-full bg-blue-600 transition-[width] duration-500"
                                          style={{ width: `${Math.round((resource.progress_done ?? 0) * 100 / resource.progress_total)}%` }}
                                        />
                                      </span>
                                      <span className="tabular-nums">
                                        {resource.progress_done ?? 0}/{resource.progress_total}
                                      </span>
                                    </>
                                  ) : (
                                    'Indexing'
                                  )}
                                </span>
                              )}
                              {resource.status === 'ready' && (
                                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-700">
                                  <CheckCircle className="w-3 h-3" /> Indexed
                                </span>
                              )}
                              {/* A 'cancelled' row with zero progress is not resumable —
                                  it is exactly what _discard_unindexed_after_cancel removes
                                  (once the run ends, or via the liveness sweep if it never
                                  gets there). Between the click and that cleanup it is still
                                  visible, so it needs its own label rather than "Will
                                  resume": that promise is untrue for it. Read off
                                  progress_done itself rather than inferring from
                                  indexingInProgress — the same zero-vs-real-progress split
                                  also happens for a split second inside the finalization
                                  loop, while the run is technically still "alive" by the
                                  lock. The data says which one it is either way. */}
                              {resource.status === 'cancelled' && !resource.progress_done && (
                                <span
                                  title={t('lightrag.repositoryDetailPage.badges.cancelledTitle')}
                                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-gray-200 text-gray-600"
                                >
                                  <PauseCircle className="w-3 h-3" /> {t('lightrag.repositoryDetailPage.badges.cancelled')}
                                </span>
                              )}
                              {/* One badge for the remaining unfinished states. Outside a
                                  run, all of them mean the same thing: Resume picks them up
                                  and the backend counts them together. 'indexing' with no
                                  run is a run that died mid-file; 'cancelled' with real
                                  progress is resumable too — cancel promises to finish what
                                  started, and a file a race or a restart left behind still
                                  needs to. */}
                              {(resource.status === 'pending' || resource.status === 'paused'
                                || (resource.status === 'cancelled' && !!resource.progress_done)
                                || (resource.status === 'indexing' && !indexingInProgress)) && (
                                indexingInProgress && resource.status === 'pending' ? (
                                  <span
                                    title={t('lightrag.repositoryDetailPage.badges.queuedTitle')}
                                    className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-yellow-100 text-yellow-700"
                                  >
                                    {t('lightrag.repositoryDetailPage.badges.queued')}
                                  </span>
                                ) : (
                                  <span
                                    title={t('lightrag.repositoryDetailPage.badges.willResumeTitle')}
                                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-amber-100 text-amber-700"
                                  >
                                    <PauseCircle className="w-3 h-3" /> {t('lightrag.repositoryDetailPage.badges.willResume')}
                                  </span>
                                )
                              )}
                              {resource.status === 'error' && (
                                <span
                                  title={resource.error_message || undefined}
                                  className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-700"
                                >
                                  {t('lightrag.repositoryDetailPage.badges.indexFailed')}
                                </span>
                              )}
                              {repository.vector_db_type === 'LIGHTRAG' &&
                                repository.silo_id != null && (
                                  <ResourceMetrics
                                    appId={Number(appId)}
                                    siloId={repository.silo_id}
                                    resourceId={resource.resource_id}
                                    resourceStatus={resource.status ?? undefined}
                                    variant="inline"
                                  />
                                )}
                            </div>
                            {resource.status === 'error' && (
                              <p className="text-xs text-error dark:text-error-dark mt-1">
                                {resource.error_message || t('lightrag.repositoryDetailPage.badges.indexingFailed')}{canEdit ? ` ${t('lightrag.repositoryDetailPage.badges.fixAndReindex')}` : ''}
                              </p>
                            )}
                          </div>
                        </div>
                      </td>
                      <td className="px-5 py-3 whitespace-nowrap text-fg-tertiary dark:text-fg-tertiary-dark">
                        {(resource.file_type || t('repositoryDetailPage.unknownType')).toUpperCase()}
                      </td>
                      <td className="px-5 py-3 whitespace-nowrap text-fg-tertiary dark:text-fg-tertiary-dark">
                        {formatDate(resource.created_at)}
                      </td>
                      <td className="px-5 py-3">
                        <div className="flex items-center justify-end gap-1">
                          {canEdit && (
                            <button
                              onClick={() => handleMoveResource(resource)}
                              className="p-2 rounded text-fg-tertiary hover:text-fg hover:bg-surface-hover dark:text-fg-tertiary-dark dark:hover:text-fg-dark dark:hover:bg-surface-hover-dark"
                            >
                              <ArrowLeftRight className="w-4 h-4" aria-hidden="true" />
                              <span className="sr-only">{t('repositoryDetailPage.actions.moveFile', { name: resource.name })}</span>
                            </button>
                          )}
                          <button
                            onClick={() => handleDownloadResource(resource)}
                            className="p-2 rounded text-fg-tertiary hover:text-fg hover:bg-surface-hover dark:text-fg-tertiary-dark dark:hover:text-fg-dark dark:hover:bg-surface-hover-dark"
                          >
                            <ArrowDownToLine className="w-4 h-4" aria-hidden="true" />
                            <span className="sr-only">{t('repositoryDetailPage.actions.downloadFile', { name: resource.name })}</span>
                          </button>
                          {canEdit && !isLightRAG && (
                            <button
                              onClick={() => handleReindexResource(resource)}
                              disabled={reindexingId !== null}
                              title={t('repositoryDetailPage.reindex.title')}
                              aria-label={t('repositoryDetailPage.reindex.ariaLabel', { name: resource.name })}
                              className="p-2 rounded text-fg-tertiary hover:text-fg hover:bg-surface-hover disabled:opacity-50 disabled:cursor-not-allowed dark:text-fg-tertiary-dark dark:hover:text-fg-dark dark:hover:bg-surface-hover-dark"
                            >
                              {reindexingId === resource.resource_id
                                ? <Spinner size="sm" tone="current" />
                                : <RefreshCw className="w-4 h-4" />}
                            </button>
                          )}
                          {canEdit && (
                            <button
                              onClick={() => handleDeleteResource(resource)}
                              className="p-2 rounded text-fg-tertiary hover:text-error hover:bg-error-bg dark:text-fg-tertiary-dark dark:hover:text-error-dark dark:hover:bg-error-bg-dark"
                            >
                              <Trash2 className="w-4 h-4" aria-hidden="true" />
                              <span className="sr-only">{t('repositoryDetailPage.actions.deleteFile', { name: resource.name })}</span>
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}

                  {/* MEDIA LIST */}
                  {filteredMedia.map((media) => (
                    <tr key={media.media_id} className="hover:bg-surface-tint dark:hover:bg-surface-tint-dark">
                      <td className="px-5 py-3">
                        <div className="flex items-center gap-3">
                          <span className="bg-surface-ash p-2 rounded-lg dark:bg-surface-hover-dark"><Video className="w-4 h-4" /></span>
                          <div className="min-w-0">
                            <h4 className="font-medium">{media.name}</h4>
                            <p className="text-xs text-fg-tertiary dark:text-fg-tertiary-dark">
                              {media.language || t('repositoryDetailPage.media.detecting')} •{' '}
                              {media.duration
                                ? `${media.duration.toFixed(1)}s`
                                : t('repositoryDetailPage.media.processing')}
                              {media.processing_mode === 'multimodal' && (
                                <span className="ml-2 inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-bronze-bg text-bronze dark:bg-bronze-bg-dark dark:text-bronze-dark" title={t('repositoryDetailPage.media.multimodalTitle')}>
                                  {t('repositoryDetailPage.media.multimodal')}
                                </span>
                              )}
                            </p>
                            {media.error_message && (
                              <p className="text-xs text-error dark:text-error-dark mt-1">
                                {media.error_message}
                              </p>
                            )}
                          </div>
                        </div>
                      </td>
                      <td className="px-5 py-3 whitespace-nowrap text-fg-tertiary dark:text-fg-tertiary-dark">
                        {media.source_type.toUpperCase()}
                      </td>
                      <td className="px-5 py-3 whitespace-nowrap text-fg-tertiary dark:text-fg-tertiary-dark">
                        {formatDate(media.create_date)}
                      </td>
                      <td className="px-5 py-3">
                        <div className="flex items-center justify-end gap-1">
                          <StatusBadge
                            status={getMediaStatusBadgeType(media.status)}
                            customLabel={t(`repositoryDetailPage.media.statuses.${media.status}`, { defaultValue: media.status.replace('_', ' ') })}
                          />

                          {canEdit && (
                            <>
                              <button
                                onClick={() => handleMoveMedia(media)}
                                className="p-2 rounded text-fg-tertiary hover:text-fg hover:bg-surface-hover dark:text-fg-tertiary-dark dark:hover:text-fg-dark dark:hover:bg-surface-hover-dark"
                              >
                                <ArrowLeftRight className="w-4 h-4" />
                              </button>
                              {/* Legacy LightRAG silos do not support media deletion */}
                              {!isLightRAG && (
                                <button
                                  onClick={() => handleDeleteMedia(media)}
                                  className="p-2 rounded text-fg-tertiary hover:text-error hover:bg-error-bg dark:text-fg-tertiary-dark dark:hover:text-error-dark dark:hover:bg-error-bg-dark"
                                >
                                  <Trash2 className="w-4 h-4" />
                                </button>
                              )}
                            </>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      <CostEstimateModal
        isOpen={showEstimateModal}
        onClose={resetPendingUploadEstimate}
        onConfirm={confirmEstimatedUpload}
        confirming={uploading}
        confirmingLabel={t('lightrag.repositoryDetailPage.estimate.uploading')}
        estimate={uploadEstimate}
        description={selectedFolderPath
          ? t('lightrag.repositoryDetailPage.estimate.selectedForFolder', { count: pendingUploadFiles.length, path: selectedFolderPath })
          : t('lightrag.repositoryDetailPage.estimate.selectedForRoot', { count: pendingUploadFiles.length })}
        itemsHeading={t('lightrag.repositoryDetailPage.estimate.itemsHeading')}
        items={pendingUploadFiles.map((file) => ({
          key: `${file.name}-${file.size}-${file.lastModified}`,
          label: file.name,
          meta: t('lightrag.repositoryDetailPage.estimate.bytes', { size: formatEstimateValue(file.size) }),
        }))}
        gate={entityTypesGateRequired ? {
          required: true,
          done: entityTypesSettled,
          label: t('lightrag.repositoryDetailPage.estimate.defineEntityTypes'),
          doneLabel: t('lightrag.repositoryDetailPage.estimate.reviewEntityTypes'),
          onOpen: () => setShowEntityTypesModal(true),
        } : undefined}
      />

      {showEntityTypesModal && repository?.silo_id != null && (
        <EntityTypeInferenceModal
          appId={Number.parseInt(appId!)}
          siloId={repository.silo_id}
          files={pendingUploadFiles}
          onConfirm={applyInferredEntityTypes}
          onCancel={() => setShowEntityTypesModal(false)}
        />
      )}

      {/* Delete Confirmation Modal */}
      <Modal
        isOpen={showDeleteModal}
        onClose={() => {
          if (deletingResource) return;
          setShowDeleteModal(false);
          setResourceToDelete(null);
          setDeleteError(null);
        }}
        title={t('repositoryDetailPage.modals.deleteFile.title')}
      >
        <div className="p-6">
          <p className="text-fg dark:text-fg-dark mb-6">
            {t('repositoryDetailPage.modals.deleteFile.message', { name: resourceToDelete?.name })}
          </p>
          <p role="status" className="sr-only">{deletingResource ? t('lightrag.repositoryDetailPage.deleteAll.deletingFile') : ''}</p>
          {deleteError && (
            <div
              role="alert"
              className="mb-6 text-sm text-red-600 bg-red-50 border border-red-200 rounded p-2 dark:text-red-300 dark:bg-red-950/40 dark:border-red-800"
            >
              {deleteError}
            </div>
          )}
          <div className="flex justify-end gap-3">
            <ButtonApp
              label={t('repositoryDetailPage.actions.cancel')}
              onClick={() => {
                setShowDeleteModal(false);
                setResourceToDelete(null);
                setDeleteError(null);
              }}
              disabled={deletingResource}
              variant="secondary"
              size="medium"
            />
            <ButtonApp
              label={t('repositoryDetailPage.actions.delete')}
              onClick={confirmDeleteResource}
              variant="dangerSolid"
              size="medium"
              disabled={deletingResource || deleteFailures >= MAX_DELETE_FAILURES}
            />
          </div>
        </div>
      </Modal>

      {/* Delete All Files Confirmation Modal */}
      <Modal
        isOpen={showDeleteAllModal}
        onClose={() => setShowDeleteAllModal(false)}
        title={t('lightrag.repositoryDetailPage.deleteAll.title')}
      >
        <div className="p-6">
          <p className="text-fg dark:text-fg-dark mb-6">
            {isLightRAG
              ? t('lightrag.repositoryDetailPage.deleteAll.confirmLightrag', { files: repository.resources.length, media: repository.media.length })
              : t('lightrag.repositoryDetailPage.deleteAll.confirm', { count: repository.resources.length })}
          </p>
          {deleteAllError && (
            <div role="alert" className="mb-6 text-sm text-red-600 bg-red-50 border border-red-200 rounded p-2 dark:text-red-300 dark:bg-red-950/40 dark:border-red-800">
              {deleteAllError}
            </div>
          )}
          <div className="flex justify-end gap-3">
            <ButtonApp
              label={t('repositoryDetailPage.actions.cancel')}
              onClick={() => setShowDeleteAllModal(false)}
              disabled={deletingAll}
              variant="secondary"
              size="medium"
            />
            <ButtonApp
              label={deletingAll ? t('lightrag.repositoryDetailPage.deleteAll.deleting') : t('lightrag.repositoryDetailPage.deleteAll.button')}
              onClick={confirmDeleteAllResources}
              disabled={deletingAll}
              variant="dangerSolid"
              size="medium"
            />
          </div>
        </div>
      </Modal>

      {/* Create Folder Modal */}
      <Modal
        isOpen={showCreateFolderModal}
        onClose={() => setShowCreateFolderModal(false)}
        title={t('repositoryDetailPage.modals.createFolder.title')}
      >
        <div className="space-y-4">
          <FormField
            label={t('repositoryDetailPage.modals.folderName')}
            id="create-folder-name"
            type="text"
            value={newFolderName}
            onChange={(e) => setNewFolderName(e.target.value)}
            placeholder={t('repositoryDetailPage.modals.folderNamePlaceholder')}
          />
          <div className="flex justify-end space-x-3">
            <ButtonApp
              label={t('repositoryDetailPage.actions.cancel')}
              variant="secondary"
              size="medium"
              onClick={() => setShowCreateFolderModal(false)}
            />
            <ButtonApp
              label={t('repositoryDetailPage.actions.create')}
              variant="primary"
              size="medium"
              onClick={createFolder}
              disabled={!newFolderName.trim()}
            />
          </div>
        </div>
      </Modal>

      {/* Rename Folder Modal */}
      <Modal
        isOpen={showRenameFolderModal}
        onClose={() => setShowRenameFolderModal(false)}
        title={t('repositoryDetailPage.modals.renameFolder.title')}
      >
        <div className="space-y-4">
          <FormField
            label={t('repositoryDetailPage.modals.folderName')}
            id="rename-folder-name"
            type="text"
            value={newFolderName}
            onChange={(e) => setNewFolderName(e.target.value)}
            placeholder={t('repositoryDetailPage.modals.folderNamePlaceholder')}
          />
          <div className="flex justify-end space-x-3">
            <ButtonApp
              label={t('repositoryDetailPage.actions.cancel')}
              variant="secondary"
              size="medium"
              onClick={() => setShowRenameFolderModal(false)}
            />
            <ButtonApp
              label={t('repositoryDetailPage.actions.rename')}
              variant="primary"
              size="medium"
              onClick={renameFolder}
              disabled={!newFolderName.trim()}
            />
          </div>
        </div>
      </Modal>

      {/* Delete Folder Modal */}
      <Modal
        isOpen={showDeleteFolderModal}
        onClose={() => setShowDeleteFolderModal(false)}
        title={t('repositoryDetailPage.modals.deleteFolder.title')}
      >
        <div className="space-y-4">
          <p className="text-fg dark:text-fg-dark">
            {t('repositoryDetailPage.modals.deleteFolder.message', { name: folderActionData.folderName })}
          </p>
          <div className="flex justify-end space-x-3">
            <ButtonApp
              label={t('repositoryDetailPage.actions.cancel')}
              onClick={() => setShowDeleteFolderModal(false)}
              variant="secondary"
              size="medium"
            />
            <ButtonApp
              label={t('repositoryDetailPage.actions.delete')}
              onClick={deleteFolder}
              variant="dangerSolid"
              size="medium"
            />
          </div>
        </div>
      </Modal>

      {/* Delete Media Modal */}
      <Modal
        isOpen={showDeleteMediaModal}
        onClose={() => {
          setShowDeleteMediaModal(false);
          setMediaToDelete(null);
        }}
        title={t('repositoryDetailPage.modals.deleteMedia.title')}
      >
        <div className="p-6">
          <p className="text-fg dark:text-fg-dark mb-6">
            {t('repositoryDetailPage.modals.deleteMedia.message', { name: mediaToDelete?.name })}
          </p>
          <div className="flex justify-end gap-3">
            <ButtonApp
              label={t('repositoryDetailPage.actions.cancel')}
              onClick={() => setShowDeleteMediaModal(false)}
              variant="secondary"
              size="medium"
            />
            <ButtonApp
              label={t('repositoryDetailPage.actions.delete')}
              onClick={confirmDeleteMedia}
              variant="dangerSolid"
              size="medium"
            />
          </div>
        </div>
      </Modal>

      {/* Move Folder Modal */}
      <Modal
        isOpen={showMoveFolderModal}
        onClose={() => setShowMoveFolderModal(false)}
        title={t('repositoryDetailPage.modals.moveFolder.title')}
      >
        <div className="space-y-4">
          <FormSelect
            id="move-folder-destination"
            label={t('repositoryDetailPage.modals.moveTo', { name: folderActionData.folderName })}
            value={newParentFolderId || ''}
            onChange={(e) => setNewParentFolderId(e.target.value ? Number.parseInt(e.target.value) : null)}
            options={[
              { value: '', label: t('repositoryDetailPage.root') },
              ...getValidDestinationFolders().map((folder) => ({
                value: folder.folder_id,
                label: folder.full_path
              }))
            ]}
          />
          <div className="flex justify-end space-x-3">
            <ButtonApp
              label={t('repositoryDetailPage.actions.cancel')}
              onClick={() => setShowMoveFolderModal(false)}
              variant="secondary"
              size="medium"
            />
            <ButtonApp
              label={t('repositoryDetailPage.actions.move')}
              onClick={moveFolder}
              variant="primary"
              size="medium"
            />
          </div>
        </div>
      </Modal>

      {/* Move Media Modal */}
      <Modal
        isOpen={showMoveMediaModal}
        onClose={() => setShowMoveMediaModal(false)}
        title={t('repositoryDetailPage.modals.moveMedia.title')}
      >
        <div className="space-y-4">
          <FormSelect
            id="move-media-destination"
            label={t('repositoryDetailPage.modals.moveTo', { name: mediaToMove?.name })}
            value={moveMediaToFolderId || ''}
            onChange={(e) => setMoveMediaToFolderId(e.target.value ? Number.parseInt(e.target.value) : null)}
            options={[
              { value: '', label: t('repositoryDetailPage.root') },
              ...(repository?.folders
                ? getAllFolderPaths(repository.folders).map((folder) => ({
                    value: folder.folder_id,
                    label: folder.full_path
                  }))
                : [])
            ]}
          />

          <div className="flex justify-end gap-3">
            <ButtonApp
              label={t('repositoryDetailPage.actions.cancel')}
              onClick={() => setShowMoveMediaModal(false)}
              variant="secondary"
              size="medium"
            />
            <ButtonApp
              label={t('repositoryDetailPage.actions.move')}
              onClick={confirmMoveMedia}
              variant="primary"
              size="medium"
            />
          </div>
        </div>
      </Modal>

      {/* Move File Modal */}
      <Modal
        isOpen={showMoveModal}
        onClose={() => setShowMoveModal(false)}
        title={t('repositoryDetailPage.modals.moveFile.title')}
      >
        <div className="space-y-4">
          <FormSelect
            id="move-file-destination"
            label={t('repositoryDetailPage.modals.moveTo', { name: resourceToMove?.name })}
            value={moveToFolderId || ''}
            onChange={(e) => setMoveToFolderId(e.target.value ? Number.parseInt(e.target.value) : null)}
            options={[
              { value: '', label: t('repositoryDetailPage.root') },
              ...(repository?.folders
                ? getAllFolderPaths(repository.folders).map((folder) => ({
                    value: folder.folder_id,
                    label: folder.full_path
                  }))
                : [])
            ]}
          />
          <div className="flex justify-end space-x-3">
            <ButtonApp
              label={t('repositoryDetailPage.actions.cancel')}
              onClick={() => setShowMoveModal(false)}
              variant="secondary"
              size="medium"
            />
            <ButtonApp
              label={t('repositoryDetailPage.actions.move')}
              onClick={confirmMoveResource}
              variant="primary"
              size="medium"
            />
          </div>
        </div>
      </Modal>

      {/* Media Upload Modal */}
      <Modal
        isOpen={showMediaUploadModal}
        onClose={() => {
          setShowMediaUploadModal(false);
          setMediaFiles([]);
          setYoutubeUrl('');
        }}
        title={t('repositoryDetailPage.modals.uploadMedia.title')}
      >
        <div className="space-y-5">
          {/* Upload Type */}
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => setUploadType('file')}
              aria-pressed={uploadType === 'file'}
              className={`flex items-center gap-3 p-3 rounded-lg border text-left text-sm transition-colors ${
                uploadType === 'file'
                  ? 'border-ink bg-surface text-fg dark:border-ink-dark dark:bg-surface-dark dark:text-fg-dark'
                  : 'border-line bg-surface text-fg-secondary hover:bg-surface-hover dark:border-line-dark dark:bg-surface-dark dark:text-fg-secondary-dark dark:hover:bg-surface-hover-dark'
              }`}
            >
              <BackgroundImage icon={FolderOpen} variant={uploadType === 'file' ? 'marron' : 'gris'} />
              <span className="font-display tracking-[-0.02em]">{t('repositoryDetailPage.upload.file')}</span>
            </button>
            <button
              onClick={() => setUploadType('youtube')}
              aria-pressed={uploadType === 'youtube'}
              className={`flex items-center gap-3 p-3 rounded-lg border text-left text-sm transition-colors ${
                uploadType === 'youtube'
                  ? 'border-ink bg-surface text-fg dark:border-ink-dark dark:bg-surface-dark dark:text-fg-dark'
                  : 'border-line bg-surface text-fg-secondary hover:bg-surface-hover dark:border-line-dark dark:bg-surface-dark dark:text-fg-secondary-dark dark:hover:bg-surface-hover-dark'
              }`}
            >
              <BackgroundImage icon={Tv} variant={uploadType === 'youtube' ? 'marron' : 'gris'} />
              <span className="font-display tracking-[-0.02em]">{t('repositoryDetailPage.upload.youtubeUrl')}</span>
            </button>
          </div>

          {/* File Upload */}
          {uploadType === 'file' && (
            <div>
              <label htmlFor="media-files-input" className="block text-sm font-medium text-fg dark:text-fg-dark mb-1">
                {t('repositoryDetailPage.upload.selectMediaFiles')}
              </label>
              <input
                id="media-files-input"
                type="file"
                multiple
                accept="video/*,audio/*"
                onChange={(e) => setMediaFiles(Array.from(e.target.files || []))}
                className="w-full p-3 text-sm text-fg-secondary border border-dashed border-line-strong rounded-lg bg-surface-tint cursor-pointer dark:text-fg-secondary-dark dark:border-line-strong-dark dark:bg-surface-tint-dark file:mr-3 file:px-3 file:py-1.5 file:rounded-md file:border file:border-ink file:bg-surface file:text-fg file:font-display file:text-sm file:cursor-pointer hover:file:bg-surface-hover dark:file:border-ink-dark dark:file:bg-surface-dark dark:file:text-fg-dark"
              />
              {mediaFiles.length > 0 && (
                <p className="text-xs text-fg-tertiary dark:text-fg-tertiary-dark mt-1">{t('repositoryDetailPage.upload.selectedFiles', { count: mediaFiles.length })}</p>
              )}
            </div>
          )}

          {/* YouTube URL */}
          {uploadType === 'youtube' && (
            <FormField
              label={t('repositoryDetailPage.upload.youtubeUrl')}
              id="youtube-url-input"
              value={youtubeUrl}
              onChange={(e) => setYoutubeUrl(e.target.value)}
              placeholder="https://youtube.com/watch?v=..."
            />
          )}

          {/* Configuration */}
          <div className="border-t border-line pt-5 dark:border-line-dark">
            <h3 className="font-display text-base font-normal tracking-[-0.02em] text-fg dark:text-fg-dark mb-3">{t('repositoryDetailPage.upload.processingOptions')}</h3>

            {/* Repository AI Services (read-only info) */}
            <div className="mb-4 space-y-2">
              {repository?.transcription_service_id ? (
                <Alert
                  type="success"
                  title={t('repositoryDetailPage.services.transcription')}
                  message={<strong>{repository.ai_services?.find(s => s.service_id === repository.transcription_service_id)?.name ?? `Service #${repository.transcription_service_id}`}</strong>}
                />
              ) : (
                <Alert
                  type="error"
                  title={t('repositoryDetailPage.upload.transcriptionMissingTitle')}
                  message={t('repositoryDetailPage.upload.transcriptionMissingMessage')}
                />
              )}
              {repository?.video_ai_service_id && (
                <Alert
                  type="info"
                  title={t('repositoryDetailPage.upload.videoAnalysis')}
                  message={<><strong>{repository.ai_services?.find(s => s.service_id === repository.video_ai_service_id)?.name ?? t('repositoryDetailPage.services.serviceNumber', { id: repository.video_ai_service_id })}</strong> ({t('repositoryDetailPage.upload.multimodalEnabled')})</>}
                />
              )}
            </div>

            <div className="space-y-3">
              <FormSelect
                label={t('repositoryDetailPage.upload.languageOptional')}
                id="media-language-select"
                value={mediaConfig.forced_language}
                onChange={(e) => setMediaConfig({...mediaConfig, forced_language: e.target.value})}
                options={[
                  { value: '', label: t('repositoryDetailPage.upload.languages.autoDetect') },
                  { value: 'es', label: t('repositoryDetailPage.upload.languages.spanish') },
                  { value: 'en', label: t('repositoryDetailPage.upload.languages.english') },
                  { value: 'eu', label: t('repositoryDetailPage.upload.languages.basque') },
                  { value: 'fr', label: t('repositoryDetailPage.upload.languages.french') },
                ]}
              />

              <div className="grid grid-cols-3 gap-3">
                <FormField
                  label={t('repositoryDetailPage.upload.minChunk')}
                  id="chunk-min-duration"
                  type="number"
                  value={mediaConfig.chunk_min_duration}
                  onChange={(e) => setMediaConfig({...mediaConfig, chunk_min_duration: Number.parseInt(e.target.value)})}
                  min={10}
                  max={60}
                  labelClassName="block text-xs font-medium text-fg-secondary dark:text-fg-secondary-dark mb-1"
                />
                <FormField
                  label={t('repositoryDetailPage.upload.maxChunk')}
                  id="chunk-max-duration"
                  type="number"
                  value={mediaConfig.chunk_max_duration}
                  onChange={(e) => setMediaConfig({...mediaConfig, chunk_max_duration: Number.parseInt(e.target.value)})}
                  min={60}
                  max={300}
                  labelClassName="block text-xs font-medium text-fg-secondary dark:text-fg-secondary-dark mb-1"
                />
                <FormField
                  label={t('repositoryDetailPage.upload.overlap')}
                  id="chunk-overlap"
                  type="number"
                  value={mediaConfig.chunk_overlap}
                  onChange={(e) => setMediaConfig({...mediaConfig, chunk_overlap: Number.parseInt(e.target.value)})}
                  min={0}
                  max={20}
                  labelClassName="block text-xs font-medium text-fg-secondary dark:text-fg-secondary-dark mb-1"
                />
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-5 border-t border-line dark:border-line-dark">
            <ButtonApp
              label={t('repositoryDetailPage.actions.cancel')}
              variant="secondary"
              onClick={() => setShowMediaUploadModal(false)}
            />
            <ButtonApp
              label={uploading ? t('repositoryDetailPage.upload.uploading') : t('repositoryDetailPage.upload.upload')}
              variant="primary"
              onClick={handleMediaUpload}
              disabled={
                !repository?.transcription_service_id ||
                (uploadType === 'file' ? mediaFiles.length === 0 : !youtubeUrl)
              }
            />
          </div>
        </div>
      </Modal>

    </div>
  );
};

export default RepositoryDetailPage;
