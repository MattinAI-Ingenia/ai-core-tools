import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { AlertTriangle } from 'lucide-react';
import { apiService } from '../services/api';
import { useApiMutation } from '../hooks/useApiMutation';
import { MESSAGES, errorMessage } from '../constants/messages';
import {
  getRoleWarning,
  vlmBlockingError,
  type LightRAGRole,
} from '../utils/lightragModelSpecs';

interface RepositoryFormData {
  name: string;
  embedding_service_id?: number;
  vector_db_type: string;
  lightrag_vector_db_type: string;
  transcription_service_id?: number;
  video_ai_service_id?: number;
  indexing_service_id?: number; // legacy alias for extract_service_id
  query_service_id?: number;
  extract_service_id?: number;
  keywords_service_id?: number;
  vlm_service_id?: number;
}

type RoleServiceField = 'query_service_id' | 'extract_service_id' | 'keywords_service_id' | 'vlm_service_id';

interface EmbeddingService {
  service_id: number;
  name: string;
  provider?: string;
  model_name?: string;
  is_system?: boolean;
}

interface VectorDbOption {
  code: string;
  label: string;
}

interface AIService {
  service_id: number;
  name: string;
  supports_video?: boolean;
  description?: string;
  model_name?: string;
  provider?: string;
  is_system?: boolean;
}

const RepositoryFormPage: React.FC = () => {
  const { appId, repositoryId } = useParams<{ appId: string; repositoryId: string }>();
  const navigate = useNavigate();
  const mutate = useApiMutation();
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [embeddingServices, setEmbeddingServices] = useState<EmbeddingService[]>([]);
  const [aiServices, setAiServices] = useState<AIService[]>([]);
  const [vectorDbOptions, setVectorDbOptions] = useState<VectorDbOption[]>([]);
  const [formData, setFormData] = useState<RepositoryFormData>({
    name: '',
    embedding_service_id: undefined,
    vector_db_type: 'PGVECTOR',
    lightrag_vector_db_type: 'QDRANT',
    transcription_service_id: undefined,
    video_ai_service_id: undefined,
    indexing_service_id: undefined,
    query_service_id: undefined,
    extract_service_id: undefined,
    keywords_service_id: undefined,
    vlm_service_id: undefined,
  });

  const isNewRepository = repositoryId === '0';

  useEffect(() => {
    if (!appId || repositoryId === undefined) {
      return;
    }

    const appIdNumber = Number.parseInt(appId, 10);
    const repositoryIdNumber = Number.parseInt(repositoryId, 10);

    if (Number.isNaN(appIdNumber) || Number.isNaN(repositoryIdNumber)) {
      return;
    }

    const fetchRepository = async () => {
      try {
        setLoading(true);
        setError(null);

        const repository = await apiService.getRepository(appIdNumber, repositoryIdNumber);
        const embeddingServiceList: EmbeddingService[] = repository.embedding_services ?? [];
        const availableVectorDbOptions: VectorDbOption[] = repository.vector_db_options ?? [];

        let servicesToUse = embeddingServiceList;
        if (servicesToUse.length === 0) {
          try {
            servicesToUse = await apiService.getEmbeddingServices(appIdNumber);
          } catch (serviceErr) {
            console.error('Error loading embedding services:', serviceErr);
          }
        }

        setEmbeddingServices(servicesToUse);
        setVectorDbOptions(availableVectorDbOptions);
        setAiServices(repository.ai_services ?? []);

        const normalizedVectorDbType = (repository.vector_db_type || 'PGVECTOR').toUpperCase();
        const resolvedVectorDbType = availableVectorDbOptions.some((option) => option.code === normalizedVectorDbType)
          ? normalizedVectorDbType
          : (availableVectorDbOptions[0]?.code || 'PGVECTOR');

        const nextEmbeddingServiceId = repository.embedding_service_id
          ?? (isNewRepository && servicesToUse.length === 1 ? servicesToUse[0].service_id : undefined);

        setFormData({
          name: repository.name ?? '',
          embedding_service_id: nextEmbeddingServiceId,
          vector_db_type: resolvedVectorDbType,
          lightrag_vector_db_type: (repository.lightrag_vector_db_type || 'QDRANT').toUpperCase(),
          transcription_service_id: repository.transcription_service_id ?? undefined,
          video_ai_service_id: repository.video_ai_service_id ?? undefined,
          indexing_service_id: repository.indexing_service_id ?? undefined,
          query_service_id: repository.query_service_id ?? repository.indexing_service_id ?? undefined,
          extract_service_id: repository.extract_service_id ?? repository.indexing_service_id ?? undefined,
          keywords_service_id: repository.keywords_service_id ?? repository.indexing_service_id ?? undefined,
          vlm_service_id: repository.vlm_service_id ?? undefined,
        });
      } catch (err) {
        console.error('Error loading repository:', err);
        setError('Failed to load repository');
      } finally {
        setLoading(false);
      }
    };

    void fetchRepository();
  }, [appId, repositoryId]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!appId) {
      setError('Invalid application context');
      return;
    }

    const appIdNumber = Number.parseInt(appId, 10);
    if (Number.isNaN(appIdNumber)) {
      setError('Invalid application context');
      return;
    }

    const trimmedName = formData.name.trim();
    if (!trimmedName) {
      setError('Repository name is required');
      return;
    }

    const normalizedVectorDbType = formData.vector_db_type.toUpperCase();
    const normalizedLightRAGVectorDbType = (formData.lightrag_vector_db_type || 'QDRANT').toUpperCase();

    if (isNewRepository && !formData.vector_db_type) {
      setError('Vector database selection is required');
      return;
    }

    if (isNewRepository && !formData.embedding_service_id) {
      setError('Embedding service is required for new repositories');
      return;
    }

    if (isNewRepository && normalizedVectorDbType === 'LIGHTRAG') {
      if (!['PGVECTOR', 'QDRANT'].includes(normalizedLightRAGVectorDbType)) {
        setError('LightRAG Vector DB must be either PGVECTOR or QDRANT');
        return;
      }

      const hasExtractLlm = formData.extract_service_id || formData.query_service_id || formData.indexing_service_id;
      if (!hasExtractLlm) {
        setError('LightRAG repositories require at least a Query or Extract AI service');
        return;
      }
      // Block VLM if not multimodal
      const vlmModel = formData.vlm_service_id
        ? (aiServices.find(s => s.service_id === formData.vlm_service_id)?.description
           || aiServices.find(s => s.service_id === formData.vlm_service_id)?.model_name
           || aiServices.find(s => s.service_id === formData.vlm_service_id)?.name)
        : undefined;
      const vlmErr = vlmBlockingError(vlmModel);
      if (vlmErr) {
        setError(vlmErr);
        return;
      }
    }
    const repositoryIdNumber = repositoryId ? Number.parseInt(repositoryId, 10) : 0;
    if (!isNewRepository && Number.isNaN(repositoryIdNumber)) {
      setError('Invalid repository context');
      return;
    }

    const payload = {
      name: trimmedName,
      embedding_service_id: formData.embedding_service_id,
      vector_db_type: normalizedVectorDbType,
      lightrag_vector_db_type: normalizedVectorDbType === 'LIGHTRAG' ? normalizedLightRAGVectorDbType : undefined,
      transcription_service_id: formData.transcription_service_id,
      video_ai_service_id: formData.video_ai_service_id,
      // Mirror extract → legacy indexing_service_id for backward compat.
      indexing_service_id: formData.extract_service_id || formData.indexing_service_id,
      query_service_id: formData.query_service_id,
      extract_service_id: formData.extract_service_id,
      keywords_service_id: formData.keywords_service_id,
      vlm_service_id: formData.vlm_service_id,
    };

    setError(null);
    setSaving(true);
    const result = await mutate(
      () =>
        isNewRepository
          ? apiService.createRepository(appIdNumber, payload)
          : apiService.updateRepository(appIdNumber, repositoryIdNumber, payload),
      {
        loading: isNewRepository
          ? MESSAGES.CREATING('repository')
          : MESSAGES.UPDATING('repository'),
        success: isNewRepository
          ? MESSAGES.CREATED('repository')
          : MESSAGES.UPDATED('repository'),
        error: (err) => errorMessage(err, MESSAGES.SAVE_FAILED('repository')),
      },
    );
    setSaving(false);

    if (result !== undefined) {
      navigate(`/apps/${appId}/repositories`);
    }
  };

  const handleCancel = () => {
    navigate(`/apps/${appId}/repositories`);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">
          {isNewRepository ? 'Create Repository' : 'Edit Repository'}
        </h1>
        <p className="text-gray-600 mt-2">
          {isNewRepository 
            ? 'Create a new repository to organize your documents. A silo will be automatically created for vector search.' 
            : 'Update repository settings and configuration'
          }
        </p>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
          {error}
        </div>
      )}

      {/* Form */}
      <div className="bg-white border border-gray-200 rounded-lg p-6 max-w-2xl">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Repository Name */}
          <div>
            <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-2">
              Repository Name
            </label>
            <input
              type="text"
              id="name"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Enter repository name"
              required
            />
            <p className="text-sm text-gray-500 mt-1">
              Choose a descriptive name for your repository
            </p>
          </div>

          {/* Vector Database */}
          <div>
            <label htmlFor="vector_db_type" className="block text-sm font-medium text-gray-700 mb-2">
              Vector Database <span className="text-red-500">*</span>
            </label>
            <select
              id="vector_db_type"
              value={formData.vector_db_type}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  vector_db_type: e.target.value.toUpperCase(),
                })
              }
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
              required={isNewRepository}
              disabled={!isNewRepository}
            >
              {vectorDbOptions.length === 0
                ? (
                  <option value={formData.vector_db_type}>{formData.vector_db_type}</option>
                ) : (
                  vectorDbOptions.map((option) => (
                    <option key={option.code} value={option.code}>
                      {option.label}
                    </option>
                  ))
                )}
            </select>
            {isNewRepository ? (
              <p className="text-sm text-gray-500 mt-1">
                Select where embeddings for this repository will be stored.
              </p>
            ) : (
              <p className="text-sm text-amber-600 mt-1">
                The vector database cannot be changed after a repository is created.
              </p>
            )}
          </div>

          {isNewRepository && formData.vector_db_type?.toUpperCase() === 'LIGHTRAG' && (
            <div>
              <label htmlFor="lightrag_vector_db_type" className="block text-sm font-medium text-gray-700 mb-2">
                LightRAG Vector DB <span className="text-red-500">*</span>
              </label>
              <select
                id="lightrag_vector_db_type"
                value={formData.lightrag_vector_db_type}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    lightrag_vector_db_type: e.target.value.toUpperCase(),
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              >
                <option value="QDRANT">Qdrant</option>
                <option value="PGVECTOR">PGVector</option>
              </select>
              <p className="text-sm text-gray-500 mt-1">
                Select the internal vector backend used by LightRAG for this repository.
              </p>
            </div>
          )}

          {/* Embedding Service (only for new repositories) */}
          {isNewRepository && (
            <div>
              <label htmlFor="embedding_service_id" className="block text-sm font-medium text-gray-700 mb-2">
                Embedding Service <span className="text-red-500">*</span>
              </label>
              <select
                id="embedding_service_id"
                value={formData.embedding_service_id || ''}
                onChange={(e) => setFormData({
                  ...formData,
                  embedding_service_id: e.target.value ? Number.parseInt(e.target.value, 10) : undefined
                })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              >
                <option value="">Select an embedding service</option>
                {embeddingServices.map((service) => (
                  <option key={service.service_id} value={service.service_id}>
                    {service.is_system ? '[System] ' : ''}{service.provider && service.model_name
                      ? `${service.name} (${service.provider} - ${service.model_name})`
                      : service.name}
                  </option>
                ))}
              </select>
              <p className="text-sm text-gray-500 mt-1">
                This embedding service will be used for the silo that's automatically created with this repository
              </p>
            </div>
          )}

          {/* LightRAG role-specific LLM configuration (only for new repositories) */}
          {isNewRepository && formData.vector_db_type?.toUpperCase() === 'LIGHTRAG' && (
            <>
              {([
                {
                  field: 'query_service_id' as RoleServiceField,
                  role: 'query' as LightRAGRole,
                  label: 'Query AI Service',
                  required: true,
                  helper: 'LLM that generates the final answer at query time. Recommended: large model (32B+). Selecting this auto-fills Extract & Keywords.',
                  placeholder: 'Select an AI service for query generation',
                },
                {
                  field: 'extract_service_id' as RoleServiceField,
                  role: 'extract' as LightRAGRole,
                  label: 'Extract AI Service',
                  required: true,
                  helper: 'LLM used to extract entities and relationships during indexing. Recommended: mid-tier (12B+, non-reasoning).',
                  placeholder: 'Select an AI service for entity extraction',
                },
                {
                  field: 'keywords_service_id' as RoleServiceField,
                  role: 'keywords' as LightRAGRole,
                  label: 'Keywords AI Service',
                  required: false,
                  helper: 'LLM that extracts keywords from user queries. Latency-critical — pick a small fast model.',
                  placeholder: 'Select an AI service for keyword extraction',
                },
                {
                  field: 'vlm_service_id' as RoleServiceField,
                  role: 'vlm' as LightRAGRole,
                  label: 'VLM AI Service (optional)',
                  required: false,
                  helper: 'Vision-language model for images/tables inside documents. MUST be multimodal — leave empty if docs are text-only.',
                  placeholder: '(none — text-only documents)',
                },
              ]).map(({ field, role, label, required, helper, placeholder }) => {
                const value = formData[field];
                const svc = value ? aiServices.find(s => s.service_id === value) : undefined;
                const modelName = svc?.description || svc?.model_name || svc?.name;
                const warning = role !== 'vlm' ? getRoleWarning(role, modelName) : null;
                const blockingErr = role === 'vlm' ? vlmBlockingError(modelName) : null;
                return (
                  <div key={field}>
                    <label htmlFor={field} className="block text-sm font-medium text-gray-700 mb-2">
                      {label}{required && <span className="text-red-500"> *</span>}
                    </label>
                    <select
                      id={field}
                      value={value || ''}
                      onChange={(e) => {
                        const parsed = e.target.value ? Number.parseInt(e.target.value, 10) : undefined;
                        setFormData(prev => {
                          const next = { ...prev, [field]: parsed };
                          // Auto-fill Extract and Keywords when Query changes
                          if (field === 'query_service_id') {
                            const prevQuery = prev.query_service_id;
                            if (!prev.extract_service_id || prev.extract_service_id === prevQuery) {
                              next.extract_service_id = parsed;
                            }
                            if (!prev.keywords_service_id || prev.keywords_service_id === prevQuery) {
                              next.keywords_service_id = parsed;
                            }
                          }
                          return next;
                        });
                      }}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      required={required}
                    >
                      <option value="">{placeholder}</option>
                      {aiServices.map((service) => (
                        <option key={service.service_id} value={service.service_id}>
                          {service.is_system ? '[System] ' : ''}{service.name}{service.provider ? ` (${service.provider})` : ''}
                        </option>
                      ))}
                    </select>
                    <p className="text-sm text-gray-500 mt-1">{helper}</p>
                    {warning && (
                      <div className="mt-2 flex items-start gap-2 text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded p-2">
                        <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                        <span>{warning}</span>
                      </div>
                    )}
                    {blockingErr && (
                      <div className="mt-2 flex items-start gap-2 text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2">
                        <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                        <span>{blockingErr}</span>
                      </div>
                    )}
                  </div>
                );
              })}
            </>
          )}

          {/* Transcription Service */}
          {aiServices.length > 0 && (
            <div>
              <label htmlFor="transcription_service_id" className="block text-sm font-medium text-gray-700 mb-2">
                Transcription Service (Whisper)
              </label>
              <select
                id="transcription_service_id"
                value={formData.transcription_service_id || ''}
                onChange={(e) => setFormData({
                  ...formData,
                  transcription_service_id: e.target.value ? parseInt(e.target.value, 10) : undefined,
                })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">None (no transcription analysis) </option>
                {aiServices.map((service) => (
                  <option key={service.service_id} value={service.service_id}>
                    {service.name}
                  </option>
                ))}
              </select>
              <p className="text-sm text-gray-500 mt-1">
                Transcription service used for all media uploads in this repository.
              </p>
            </div>
          )}

          {/* Video AI Service */}
          {aiServices.filter(s => s.supports_video).length > 0 && (
            <div>
              <label htmlFor="video_ai_service_id" className="block text-sm font-medium text-gray-700 mb-2">
                Video Analysis Service (Gemini)
              </label>
              <select
                id="video_ai_service_id"
                value={formData.video_ai_service_id || ''}
                onChange={(e) => setFormData({
                  ...formData,
                  video_ai_service_id: e.target.value ? parseInt(e.target.value, 10) : undefined,
                })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">None (no visual analysis)</option>
                {aiServices.filter(s => s.supports_video).map((service) => (
                  <option key={service.service_id} value={service.service_id}>
                    {service.name}
                  </option>
                ))}
              </select>
              <p className="text-sm text-gray-500 mt-1">
                When set, media uploaded in multimodal mode will use this service for visual frame analysis.
              </p>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex justify-end gap-3 pt-6 border-t border-gray-200">
            <button
              type="button"
              onClick={handleCancel}
              className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={saving}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {saving && (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              )}
              {isNewRepository ? 'Create Repository' : 'Save Changes'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default RepositoryFormPage;