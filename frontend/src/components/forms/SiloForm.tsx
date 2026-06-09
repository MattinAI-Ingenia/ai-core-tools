import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { AlertTriangle, Info } from 'lucide-react';
import { apiService } from '../../services/api';
import {
  getRoleWarning,
  vlmBlockingError,
  type LightRAGRole,
} from '../../utils/lightragModelSpecs';

interface VectorDbOption {
  code: string;
  label: string;
}

// Define the Silo type for form data
interface Silo {
  silo_id: number;
  name: string;
  description?: string;
  type?: string;
  created_at?: string;
  docs_count: number;
  vector_db_type?: string;
  metadata_definition_id?: number;
  embedding_service_id?: number;
  output_parsers?: { parser_id: number; name: string }[];
  embedding_services?: { service_id: number; name: string; provider?: string; is_system?: boolean }[];
  vector_db_options?: VectorDbOption[];
  indexing_service_id?: number; // legacy alias for extract_service_id
  query_service_id?: number;
  extract_service_id?: number;
  keywords_service_id?: number;
  vlm_service_id?: number;
  lightrag_chunk_strategy?: string;
  lightrag_chunk_token_size?: number;
  lightrag_chunk_overlap_token_size?: number;
  lightrag_graph_context_enabled?: boolean;
  ai_services?: AIServiceOption[];
}

interface AIServiceOption {
  service_id: number;
  name: string;
  provider?: string;
  is_system?: boolean;
  description?: string;
  model_name?: string;
}

// Define the form data type
interface SiloFormData {
  name: string;
  description?: string;
  type?: string;
  output_parser_id?: number;
  embedding_service_id?: number;
  vector_db_type?: string;
  indexing_service_id?: number; // legacy alias for extract_service_id
  query_service_id?: number;
  extract_service_id?: number;
  keywords_service_id?: number;
  vlm_service_id?: number;
  lightrag_chunk_strategy?: string;
  lightrag_chunk_token_size?: number;
  lightrag_chunk_overlap_token_size?: number;
  lightrag_graph_context_enabled?: boolean;
}

const ROLE_SERVICE_FIELDS = [
  'query_service_id',
  'extract_service_id',
  'keywords_service_id',
  'vlm_service_id',
] as const;
type RoleServiceField = typeof ROLE_SERVICE_FIELDS[number];

// Define the props for the component
interface SiloFormProps {
  silo?: Silo;
  onSubmit: (data: SiloFormData) => Promise<void>;
  onCancel: () => void;
}

function SiloForm({ silo, onSubmit, onCancel }: Readonly<SiloFormProps>) {
  const { appId } = useParams();
  const [formData, setFormData] = useState<SiloFormData>({
    name: '',
    description: '',
    type: 'CUSTOM', // Always CUSTOM for this interface
    output_parser_id: undefined,
    embedding_service_id: undefined,
    vector_db_type: 'PGVECTOR',
    indexing_service_id: undefined,
    query_service_id: undefined,
    extract_service_id: undefined,
    keywords_service_id: undefined,
    vlm_service_id: undefined,
    lightrag_chunk_strategy: 'token_window',
    lightrag_chunk_token_size: 1200,
    lightrag_chunk_overlap_token_size: 100,
    lightrag_graph_context_enabled: false
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outputParsers, setOutputParsers] = useState<any[]>([]);
  const [embeddingServices, setEmbeddingServices] = useState<any[]>([]);
  const [vectorDbOptions, setVectorDbOptions] = useState<VectorDbOption[]>([]);
  const [aiServices, setAiServices] = useState<any[]>([]);
  const [loadingFormData, setLoadingFormData] = useState(true);

  const isEditing = !!silo && silo.silo_id !== 0;
  let submitButtonLabel = isEditing ? 'Update Silo' : 'Create Silo';
  if (isSubmitting) {
    submitButtonLabel = 'Saving...';
  }

  // Load form data (output parsers and embedding services)
  useEffect(() => {
    loadFormData();
  }, [appId]);

  // Initialize form with existing silo data
  useEffect(() => {
    const defaultVectorType = (silo?.vector_db_type || 'PGVECTOR').toUpperCase();
    const availableVectorDbOptions = silo?.vector_db_options ?? [];
    const matchedVectorDbOption = availableVectorDbOptions.find(
      (option) => option.code.toUpperCase() === defaultVectorType
    );
    const vectorDbTypeValue = matchedVectorDbOption?.code
      || availableVectorDbOptions[0]?.code
      || defaultVectorType
      || 'PGVECTOR';

    setVectorDbOptions(availableVectorDbOptions);

    if (silo?.ai_services) {
      setAiServices(silo.ai_services);
    }

    setFormData(prev => ({
      ...prev,
      name: silo?.name || '',
      description: silo?.description || '',
      type: 'CUSTOM',
      output_parser_id: silo?.metadata_definition_id || undefined,
      embedding_service_id: silo?.embedding_service_id || undefined,
      vector_db_type: vectorDbTypeValue,
      indexing_service_id: silo?.indexing_service_id || undefined,
      // LightRAG 2026.05 roles. Fall back to the legacy ``indexing_service_id``
      // for old silos that still only have the single-LLM column populated.
      query_service_id: silo?.query_service_id || silo?.indexing_service_id || undefined,
      extract_service_id: silo?.extract_service_id || silo?.indexing_service_id || undefined,
      keywords_service_id: silo?.keywords_service_id || silo?.indexing_service_id || undefined,
      vlm_service_id: silo?.vlm_service_id || undefined,
      lightrag_chunk_strategy: silo?.lightrag_chunk_strategy || 'token_window',
      lightrag_chunk_token_size: silo?.lightrag_chunk_token_size || 1200,
      lightrag_chunk_overlap_token_size: silo?.lightrag_chunk_overlap_token_size || 100,
      lightrag_graph_context_enabled: silo?.lightrag_graph_context_enabled || false
    }));
  }, [silo]);

  async function loadFormData() {
    if (!appId) return;

    try {
      setLoadingFormData(true);
      setError(null);

      const appIdNumber = Number.parseInt(appId, 10);

      // Load output parsers and embedding services in parallel
      const [parsersResponse, servicesResponse] = await Promise.all([
        apiService.getOutputParsers(appIdNumber),
        apiService.getEmbeddingServices(appIdNumber)
      ]);

      setOutputParsers(parsersResponse);
      setEmbeddingServices(servicesResponse);

      // Fetch database type options only when they have not been set
      if (vectorDbOptions.length === 0) {
        const siloOptions = await apiService.getSiloOptions(appIdNumber);
        const availableVectorDbOptions: VectorDbOption[] = siloOptions.vector_db_options ?? [];
        setVectorDbOptions(availableVectorDbOptions);
        // Do NOT set vector_db_type here — let the silo initialization effect handle it
      }

      // Default embedding service for new silo when only one available
      if (!isEditing && servicesResponse.length === 1) {
        setFormData(prev => ({
          ...prev,
          embedding_service_id: servicesResponse[0].service_id
        }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load form data');
      console.error('Error loading form data:', err);
    } finally {
      setLoadingFormData(false);
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;

    let parsedValue: string | number | undefined = value === '' ? undefined : value;

    const intFields = [
      'embedding_service_id', 'output_parser_id', 'indexing_service_id',
      'query_service_id', 'extract_service_id', 'keywords_service_id', 'vlm_service_id',
      'lightrag_chunk_token_size', 'lightrag_chunk_overlap_token_size',
    ];
    if (intFields.includes(name)) {
      parsedValue = value === '' ? undefined : Number.parseInt(value, 10);
    } else if (name === 'vector_db_type' && typeof value === 'string') {
      parsedValue = value.toUpperCase();
    }

    setFormData(prev => {
      const next = { ...prev, [name]: parsedValue };
      // Query is the primary role — when it changes, propagate the value to
      // Extract and Keywords if those slots are empty OR were matching the
      // previous Query value (treat them as auto-filled). VLM is left alone
      // because it requires a multimodal model.
      if (name === 'query_service_id') {
        const prevQuery = prev.query_service_id;
        const autofillRoles: RoleServiceField[] = ['extract_service_id', 'keywords_service_id'];
        for (const role of autofillRoles) {
          if (!prev[role] || prev[role] === prevQuery) {
            next[role] = parsedValue as number | undefined;
          }
        }
      }
      return next;
    });
  };

  // Helper: get the model identifier for an AIService row used in the dropdown.
  const getServiceModelName = (serviceId: number | undefined): string | undefined => {
    if (!serviceId) return undefined;
    const svc = aiServices.find((s) => s.service_id === serviceId);
    return svc?.description || svc?.model_name || svc?.name;
  };

  const isLightRAG = formData.vector_db_type?.toUpperCase() === 'LIGHTRAG';
  const vlmError = isLightRAG ? vlmBlockingError(getServiceModelName(formData.vlm_service_id)) : null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Basic validation
    if (!formData.name.trim()) {
      setError('Silo name is required');
      return;
    }

    if (!isEditing && !formData.vector_db_type) {
      setError('Database type selection is required');
      return;
    }

    // Block: VLM role must be multimodal when set (LightRAG only).
    if (isLightRAG && vlmError) {
      setError(vlmError);
      return;
    }

    try {
      setIsSubmitting(true);
      setError(null);
      // Always mirror Extract into the legacy indexing_service_id so older
      // backend code paths keep working until they migrate to the new column.
      const normalized: SiloFormData = {
        ...formData,
        indexing_service_id: formData.extract_service_id || formData.indexing_service_id,
      };
      const payload = isEditing
        ? (({ vector_db_type: _vdb, embedding_service_id: _esi, indexing_service_id: _isi,
              query_service_id: _qsi, extract_service_id: _esi2, keywords_service_id: _ksi,
              vlm_service_id: _vsi, ...rest }) => rest)(normalized)
        : { ...normalized, vector_db_type: normalized.vector_db_type!.toUpperCase() };
      await onSubmit(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save silo');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (loadingFormData) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-yellow-600"></div>
        <span className="ml-2">Loading form data...</span>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-gray-900">
          {isEditing ? `Edit Silo: ${silo?.name}` : 'Create New Silo'}
        </h2>
        <p className="text-gray-600">
          {isEditing
            ? 'Update your silo configuration and settings.'
            : 'Create a new silo for vector storage and semantic search.'
          }
        </p>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="bg-white shadow rounded-lg p-6">
          <div className="grid grid-cols-1 gap-6">
            {/* Silo Name */}
            <div>
              <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-2">
                Silo Name <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                id="name"
                name="name"
                value={formData.name}
                onChange={handleChange}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:border-transparent"
                placeholder="Enter silo name"
                disabled={isSubmitting}
              />
            </div>
            <div>
              <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-2">
                Silo Description <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                id="description"
                name="description"
                value={formData.description || ''}
                onChange={handleChange}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:border-transparent"
                placeholder="Enter silo description"
                disabled={isSubmitting}
              />
            </div>
            {/* Metadata Definition (Output Parser) */}
            <div>
              <label htmlFor="output_parser_id" className="block text-sm font-medium text-gray-700 mb-2">
                Metadata Definition
              </label>
              <select
                id="output_parser_id"
                name="output_parser_id"
                value={formData.output_parser_id?.toString() || ''}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:border-transparent"
                disabled={isSubmitting}
              >
                <option value="">No metadata definition</option>
                {outputParsers.map((parser) => (
                  <option key={parser.parser_id} value={parser.parser_id}>
                    {parser.name}
                  </option>
                ))}
              </select>
              <p className="mt-1 text-sm text-gray-500">
                Optional: Define structured metadata for documents in this silo.
              </p>
            </div>

            {/* Embedding Service */}
            <div>
              <label htmlFor="embedding_service_id" className="block text-sm font-medium text-gray-700 mb-2">
                Embedding Service <span className="text-red-500">*</span>
              </label>
              <select
                id="embedding_service_id"
                name="embedding_service_id"
                value={formData.embedding_service_id?.toString() || ''}
                onChange={handleChange}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:border-transparent disabled:bg-gray-100 disabled:text-gray-500 disabled:cursor-not-allowed"
                disabled={isSubmitting || isEditing}
              >
                <option value="">Select an embedding service</option>
                {embeddingServices.map((service) => (
                  <option key={service.service_id} value={service.service_id}>
                    {service.is_system ? `[System] ${service.name}` : service.name}{service.provider ? ` (${service.provider})` : ''}
                  </option>
                ))}
              </select>
              {isEditing ? (
                <p className="mt-1 text-sm text-amber-600">
                  The embedding service cannot be changed after a silo is created.
                </p>
              ) : (
                <p className="mt-1 text-sm text-gray-500">
                  Required: Choose the embedding service for vector generation.
                </p>
              )}
            </div>

            {/* Vector Database */}
            <div>
              <label htmlFor="vector_db_type" className="block text-sm font-medium text-gray-700 mb-2">
                Database type <span className="text-red-500">*</span>
              </label>
              <select
                id="vector_db_type"
                name="vector_db_type"
                value={vectorDbOptions.length === 0 ? '' : formData.vector_db_type ?? ''}
                onChange={handleChange}
                required={!isEditing}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:border-transparent disabled:bg-gray-100 disabled:text-gray-500 disabled:cursor-not-allowed"
                disabled={isSubmitting || vectorDbOptions.length === 0 || isEditing}
              >
                {vectorDbOptions.length === 0 && <option value="">No database types available</option>}
                {vectorDbOptions.map((option) => (
                  <option key={option.code} value={option.code}>
                    {option.label}
                  </option>
                ))}
              </select>
              {isEditing ? (
                <p className="mt-1 text-sm text-gray-500 flex items-center gap-1">
                  <Info className="w-3.5 h-3.5 shrink-0" />
                  The database type cannot be changed after a silo is created.
                </p>
              ) : vectorDbOptions.length === 0 ? (
                <p className="mt-1 text-sm text-red-600">
                  No database types available. Configure a silo backend before proceeding.
                </p>
              ) : (
                <p className="mt-1 text-sm text-gray-500">
                  Required: Select the storage engine that will persist vectors for this silo.
                </p>
              )}
            </div>
          </div>

          {/* LightRAG Configuration — shown only when LIGHTRAG is selected */}
          {formData.vector_db_type?.toUpperCase() === 'LIGHTRAG' && (
            <div className="border-t border-gray-200 pt-6 mt-6 space-y-6">
              <h3 className="text-sm font-semibold text-gray-800 uppercase tracking-wider">LightRAG Configuration</h3>

              {/* Role-specific LLM configuration (LightRAG 2026.05).
                  QUERY is the primary role — selecting it auto-fills EXTRACT
                  and KEYWORDS. VLM is optional and must be multimodal. */}
              {([
                {
                  field: 'query_service_id' as RoleServiceField,
                  role: 'query' as LightRAGRole,
                  label: 'Query AI Service',
                  required: true,
                  helper: 'LLM that generates the final answer at query time. Recommended: large model (32B+, e.g. GPT-4o, Claude 3.5 Sonnet). Selecting this auto-fills Extract & Keywords.',
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
                  helper: 'Vision-language model for images/tables inside documents. MUST be multimodal — leave empty if your docs are text-only.',
                  placeholder: '(none — text-only documents)',
                },
              ]).map(({ field, role, label, required, helper, placeholder }) => {
                const value = formData[field];
                const modelName = getServiceModelName(value);
                const warning = role !== 'vlm' ? getRoleWarning(role, modelName) : null;
                const blockingError = role === 'vlm' ? vlmBlockingError(modelName) : null;
                return (
                  <div key={field}>
                    <label htmlFor={field} className="block text-sm font-medium text-gray-700 mb-2">
                      {label}{required && <span className="text-red-500"> *</span>}
                    </label>
                    <select
                      id={field}
                      name={field}
                      value={value?.toString() || ''}
                      onChange={handleChange}
                      required={required}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:border-transparent disabled:bg-gray-100"
                      disabled={isSubmitting || isEditing}
                    >
                      <option value="">{placeholder}</option>
                      {aiServices.map((service: AIServiceOption) => (
                        <option key={service.service_id} value={service.service_id}>
                          {service.is_system ? `[System] ${service.name}` : service.name}{service.provider ? ` (${service.provider})` : ''}
                        </option>
                      ))}
                    </select>
                    <p className="mt-1 text-sm text-gray-500">{helper}</p>
                    {warning && (
                      <div className="mt-2 flex items-start gap-2 text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded p-2">
                        <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                        <span>{warning}</span>
                      </div>
                    )}
                    {blockingError && (
                      <div className="mt-2 flex items-start gap-2 text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2">
                        <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                        <span>{blockingError}</span>
                      </div>
                    )}
                  </div>
                );
              })}

              {/* Chunking Strategy */}
              <div>
                <label htmlFor="lightrag_chunk_strategy" className="block text-sm font-medium text-gray-700 mb-2">
                  Chunking strategy
                </label>
                <select
                  id="lightrag_chunk_strategy"
                  name="lightrag_chunk_strategy"
                  value={formData.lightrag_chunk_strategy || 'token_window'}
                  onChange={handleChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:border-transparent"
                  disabled={isSubmitting}
                >
                  <option value="token_window">Token window (default)</option>
                  <option value="split_by_char">Split by character</option>
                  <option value="split_by_char_only">Split by character only</option>
                </select>
              </div>

              {/* Token size fields — shown only for token_window strategy */}
              {(formData.lightrag_chunk_strategy || 'token_window') === 'token_window' && (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label htmlFor="lightrag_chunk_token_size" className="block text-sm font-medium text-gray-700 mb-2">
                      Chunk token size
                    </label>
                    <input
                      type="number"
                      id="lightrag_chunk_token_size"
                      name="lightrag_chunk_token_size"
                      value={formData.lightrag_chunk_token_size ?? 1200}
                      onChange={handleChange}
                      min={100}
                      max={8000}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:border-transparent"
                      disabled={isSubmitting}
                    />
                  </div>
                  <div>
                    <label htmlFor="lightrag_chunk_overlap_token_size" className="block text-sm font-medium text-gray-700 mb-2">
                      Overlap token size
                    </label>
                    <input
                      type="number"
                      id="lightrag_chunk_overlap_token_size"
                      name="lightrag_chunk_overlap_token_size"
                      value={formData.lightrag_chunk_overlap_token_size ?? 100}
                      onChange={handleChange}
                      min={0}
                      max={2000}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:border-transparent"
                      disabled={isSubmitting}
                    />
                  </div>
                </div>
              )}

              {/* Graph context toggle */}
              <div className="flex items-center justify-between">
                <div>
                  <label htmlFor="lightrag_graph_context_enabled" className="text-sm font-medium text-gray-700">
                    Graph context in retrieval metadata
                  </label>
                  <p className="text-xs text-gray-500 mt-0.5">
                    When enabled, retrieved documents may include compact graph hints (linked entities and relations).
                  </p>
                </div>
                <input
                  type="checkbox"
                  id="lightrag_graph_context_enabled"
                  name="lightrag_graph_context_enabled"
                  checked={formData.lightrag_graph_context_enabled || false}
                  onChange={(e) => setFormData(prev => ({ ...prev, lightrag_graph_context_enabled: e.target.checked }))}
                  className="h-4 w-4 text-yellow-600 focus:ring-yellow-500 border-gray-300 rounded"
                  disabled={isSubmitting}
                />
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex">
                <AlertTriangle className="w-4 h-4 text-yellow-500 mr-3 shrink-0" />
                <div>
                  <h3 className="text-sm font-medium text-red-800">Error</h3>
                  <p className="text-sm text-red-600 mt-1">{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="mt-6 flex items-center justify-between">
            <button
              type="button"
              onClick={onCancel}
              disabled={isSubmitting}
              className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting || !!vlmError}
              className="px-6 py-2 bg-yellow-600 hover:bg-yellow-700 disabled:bg-yellow-400 text-white rounded-lg flex items-center"
              title={vlmError ?? undefined}
            >
              {isSubmitting && (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              )}
              {submitButtonLabel}
            </button>
          </div>
        </div>
      </form>

      {/* Info Section */}
      <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex">
          <div className="flex-shrink-0">
            <Info className="w-4 h-4 text-blue-400" />
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-blue-800">
              About Custom Silos
            </h3>
            <div className="mt-2 text-sm text-blue-700">
              <p>
                Custom silos are vector storage containers that enable semantic search and retrieval.
                They store document embeddings and allow AI agents to find relevant information quickly.
              </p>
              <p className="mt-2">
                <strong>Embedding Service:</strong> Required for converting text to vectors.
                <br />
                <strong>Metadata Definition:</strong> Optional structured data for filtering and organization.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SiloForm; 