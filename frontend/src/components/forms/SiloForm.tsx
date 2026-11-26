import Alert from '../ui/Alert';
import { FormActions } from '../ui/FormActions';
import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { AlertTriangle, Info } from 'lucide-react';
import { apiService } from '../../services/api';
import {
  getRoleWarning,
  vlmBlockingError,
  type LightRAGRole,
} from '../../utils/lightragModelSpecs';
import { LightRAGModelHints } from './LightRAGModelHints';
import { LightRAGAdvancedSettings, sanitizeEntityTypesList } from './LightRAGAdvancedSettings';
import { ServiceOptionStatus, serviceStatusId } from '../ui/ServiceOptionStatus';
import { selectableServiceOptions, serviceOptionLabel, type ServiceOptionLike } from '../../utils/serviceLifecycle';
import { MediaServicesFields } from './MediaServicesFields';
import { FormField, FormSelect } from '../ui/FormField';
import CardApp from '../ui/CardApp';

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
  embedding_services?: (ServiceOptionLike & { provider?: string; is_system?: boolean })[];
  vector_db_options?: VectorDbOption[];
  indexing_service_id?: number; // legacy alias for extract_service_id
  extract_service_id?: number;
  keywords_service_id?: number;
  vlm_service_id?: number;
  lightrag_chunk_strategy?: string;
  lightrag_chunk_token_size?: number;
  lightrag_chunk_overlap_token_size?: number;
  lightrag_language?: string;
  lightrag_entity_extract_max_gleaning?: number;
  lightrag_max_source_ids_per_entity?: number;
  lightrag_max_source_ids_per_relation?: number;
  lightrag_entity_types?: string;
  lightrag_entity_types_mode?: 'infer' | 'manual';
  source_label_fields?: string;
  transcription_service_id?: number | null;
  video_ai_service_id?: number | null;
  ai_services?: AIServiceOption[];
}

interface AIServiceOption {
  service_id: number;
  name: string;
  provider?: string;
  is_system?: boolean;
  description?: string;
  model_name?: string;
  supports_video?: boolean;
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
  extract_service_id?: number;
  keywords_service_id?: number;
  vlm_service_id?: number;
  lightrag_chunk_strategy?: string;
  lightrag_chunk_token_size?: number;
  lightrag_chunk_overlap_token_size?: number;
  lightrag_language?: string;
  lightrag_entity_extract_max_gleaning?: number;
  lightrag_max_source_ids_per_entity?: number;
  lightrag_max_source_ids_per_relation?: number;
  lightrag_entity_types?: string;
  lightrag_entity_types_mode?: 'infer' | 'manual';
  source_label_fields?: string;
  // null clears the service; the backend keeps a field only when it is omitted
  transcription_service_id?: number | null;
  video_ai_service_id?: number | null;
}

type RoleServiceField = 'extract_service_id' | 'keywords_service_id' | 'vlm_service_id';

// Define the props for the component
interface SiloFormProps {
  silo?: Silo;
  onSubmit: (data: SiloFormData) => Promise<void>;
  onCancel: () => void;
}

function SiloForm({ silo, onSubmit, onCancel}: Readonly<SiloFormProps>) {
  const { t } = useTranslation('common');
  const { appId } = useParams();
  const [formData, setFormData] = useState<SiloFormData>({
    name: '',
    description: '',
    type: 'CUSTOM', // Always CUSTOM for this interface
    output_parser_id: undefined,
    embedding_service_id: undefined,
    vector_db_type: 'PGVECTOR',
    indexing_service_id: undefined,
    extract_service_id: undefined,
    keywords_service_id: undefined,
    vlm_service_id: undefined,
    lightrag_chunk_strategy: 'fixed_token',
    lightrag_chunk_token_size: 1200,
    lightrag_chunk_overlap_token_size: 100,
    lightrag_language: 'English',
    lightrag_entity_extract_max_gleaning: 0,
    lightrag_max_source_ids_per_entity: 1000,
    lightrag_max_source_ids_per_relation: 1000,
    lightrag_entity_types: '',
    lightrag_entity_types_mode: 'infer',
    source_label_fields: '',
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outputParsers, setOutputParsers] = useState<any[]>([]);
  const [embeddingServices, setEmbeddingServices] = useState<(ServiceOptionLike & { provider?: string; is_system?: boolean })[]>([]);
  const [vectorDbOptions, setVectorDbOptions] = useState<VectorDbOption[]>([]);
  const [aiServices, setAiServices] = useState<any[]>([]);

  const isEditing = !!silo && silo.silo_id !== 0;
  let submitButtonLabel = isEditing ? t('siloFormPage.actions.update') : t('siloFormPage.actions.create');
  if (isSubmitting) {
    submitButtonLabel = t('siloFormPage.actions.saving');
  }

  // Load form data (output parsers and embedding services)
  useEffect(() => {
    void loadFormData();
  }, [appId]);

  // Initialize form with existing silo data
  useEffect(() => {
    // A silo being created (no silo, or the silo_id===0 template) defaults to
    // LightRAG; an existing silo always keeps its own configured type.
    const isNewSilo = !silo || silo.silo_id === 0;
    const defaultVectorType = (isNewSilo ? 'LIGHTRAG' : (silo?.vector_db_type || 'PGVECTOR')).toUpperCase();
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
      extract_service_id: silo?.extract_service_id || silo?.indexing_service_id || undefined,
      keywords_service_id: silo?.keywords_service_id || silo?.indexing_service_id || undefined,
      vlm_service_id: silo?.vlm_service_id || undefined,
      lightrag_chunk_strategy: silo?.lightrag_chunk_strategy || 'fixed_token',
      lightrag_chunk_token_size: silo?.lightrag_chunk_token_size || 1200,
      lightrag_chunk_overlap_token_size: silo?.lightrag_chunk_overlap_token_size || 100,
      lightrag_language: silo?.lightrag_language || 'English',
      lightrag_entity_extract_max_gleaning: silo?.lightrag_entity_extract_max_gleaning ?? 0,
      lightrag_max_source_ids_per_entity: silo?.lightrag_max_source_ids_per_entity || 1000,
      lightrag_max_source_ids_per_relation: silo?.lightrag_max_source_ids_per_relation || 1000,
      lightrag_entity_types: silo?.lightrag_entity_types || '',
      lightrag_entity_types_mode: silo?.lightrag_entity_types_mode || 'infer',
      source_label_fields: silo?.source_label_fields || '',
      transcription_service_id: silo?.transcription_service_id ?? null,
      video_ai_service_id: silo?.video_ai_service_id ?? null,
    }));
  }, [silo]);

  async function loadFormData() {
    if (!appId) return;

    try {
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
      setError(err instanceof Error ? err.message : t('siloFormPage.loadFormFailed'));
      console.error('Error loading form data:', err);
    } 
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;

    let parsedValue: string | number | undefined = value === '' ? undefined : value;

    const intFields = [
      'embedding_service_id', 'output_parser_id', 'indexing_service_id',
      'extract_service_id', 'keywords_service_id', 'vlm_service_id',
    ];
    if (intFields.includes(name)) {
      parsedValue = value === '' ? undefined : Number.parseInt(value, 10);
    } else if (name === 'vector_db_type' && typeof value === 'string') {
      parsedValue = value.toUpperCase();
    }

    setFormData(prev => ({ ...prev, [name]: parsedValue }));
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
      setError(t('siloFormPage.validation.nameRequired'));
      return;
    }

    if (!isEditing && !formData.vector_db_type) {
      setError(t('siloFormPage.validation.vectorDbRequired'));
      return;
    }

    if (isLightRAG && !formData.keywords_service_id) {
      setError(t('lightrag.siloForm.keywordsRequired'));
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
        lightrag_entity_types: sanitizeEntityTypesList(formData.lightrag_entity_types),
      };
      const payload = isEditing
        ? (({ vector_db_type: _vdb, embedding_service_id: _esi, lightrag_language: _lang,
              lightrag_entity_extract_max_gleaning: _gleaning,
              lightrag_max_source_ids_per_entity: _maxEnt,
              lightrag_max_source_ids_per_relation: _maxRel,
              lightrag_entity_types: _entityTypes,
              vlm_service_id: _vsi, ...rest }) => rest)(normalized)
        : { ...normalized, vector_db_type: normalized.vector_db_type!.toUpperCase() };
      await onSubmit(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : t('siloFormPage.saveFailed'));
    } finally {
      setIsSubmitting(false);
    }
  };

  
  return (
    <div>
      {/* Form */}
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="">
          <div className="grid grid-cols-1 gap-6">
            {/* Silo Name */}
            <FormField
              label={t('siloFormPage.fields.name')}
              id="name"
              value={formData.name}
              onChange={handleChange}
              required
              placeholder={t('siloFormPage.fields.namePlaceholder')}
              disabled={isSubmitting}
            />
            <FormField
              label={t('siloFormPage.fields.description')}
              id="description"
              value={formData.description || ''}
              onChange={handleChange}
              required
              placeholder={t('siloFormPage.fields.descriptionPlaceholder')}
              disabled={isSubmitting}
            />
            {/* Metadata Definition (Output Parser) */}
            <div>
              <FormSelect
                label={t('siloFormPage.fields.metadataDefinition')}
                id="output_parser_id"
                value={formData.output_parser_id?.toString() || ''}
                onChange={handleChange}
                options={[
                  { value: '', label: t('siloFormPage.fields.noMetadataDefinition') },
                  ...outputParsers.map((parser) => ({
                    value: parser.parser_id,
                    label: parser.name,
                  })),
                ]}
                disabled={isSubmitting}
                helpText={t('siloFormPage.help.metadataDefinition')}
              />
            </div>

            {/* Embedding Service */}
            <div>
              <FormSelect
                label={t('siloFormPage.fields.embeddingService')}
                id="embedding_service_id"
                value={formData.embedding_service_id?.toString() || ''}
                onChange={handleChange}
                required
                options={[
                  { value: '', label: t('siloFormPage.fields.selectEmbeddingService') },
                  ...selectableServiceOptions(embeddingServices, silo?.embedding_service_id).map((service) => ({
                    value: service.service_id,
                    label: serviceOptionLabel(service, [
                      service.is_system ? `[${t('siloFormPage.system')}]` : '',
                      service.name,
                      service.provider ? `(${service.provider})` : '',
                    ].filter(Boolean).join(' ')),
                  })),
                ]}
                disabled={isSubmitting || isEditing}
                ariaDescribedBy={serviceStatusId(embeddingServices, formData.embedding_service_id, 'embedding_service_id')}
              />
              <ServiceOptionStatus selectId="embedding_service_id" options={embeddingServices} selectedId={formData.embedding_service_id} />
              {isEditing ? (
                <p className="mt-1 text-xs text-accent dark:text-accent-dark">
                  {t('siloFormPage.help.embeddingServiceImmutable')}
                </p>
              ) : (
                <p className="mt-1 text-xs text-fg-tertiary dark:text-fg-tertiary-dark">
                  {t('siloFormPage.help.embeddingServiceRequired')}
                </p>
              )}
            </div>
            {/* Vector Database */}
            <div>
              <FormSelect
                label={t('siloFormPage.fields.vectorDatabase')}
                id="vector_db_type"
                value={vectorDbOptions.length === 0 ? '' : formData.vector_db_type ?? ''}
                onChange={handleChange}
                required={!isEditing}
                options={[
                  ...(vectorDbOptions.length === 0
                    ? [{ value: '', label: t('siloFormPage.fields.noVectorDatabases') }]
                    : []),
                  ...vectorDbOptions.map((option) => ({
                    value: option.code,
                    label: option.label,
                  })),
                ]}
                disabled={isSubmitting || vectorDbOptions.length === 0 || isEditing}
                helpText={isEditing ? (
                  <span className="flex items-center gap-1 ">
                    <Info className="w-3.5 h-3.5 shrink-0" />
                    {t('siloFormPage.help.vectorDatabaseImmutable')}
                  </span>
                ) : vectorDbOptions.length === 0 ? (
                  <span className="text-error dark:text-error-dark">
                    {t('siloFormPage.help.configureVectorDatabase')}
                  </span>
                ) : (
                  t('siloFormPage.help.vectorDatabaseRequired')
                )}
              />
            </div>

            {/* LightRAG Configuration — shown only when LIGHTRAG is selected */}
            {isLightRAG && (
              <div className="border-t border-line pt-6 mt-6 space-y-6 dark:border-line-dark">
                <h3 className="text-sm font-semibold text-fg dark:text-fg-dark uppercase tracking-wider">{t('lightrag.siloForm.configTitle')}</h3>

                {/* Role-specific LLM configuration (LightRAG 2026.05).
                    QUERY is the primary role — selecting it auto-fills EXTRACT
                    and KEYWORDS. VLM is optional and must be multimodal. */}
                {([
                  {
                    field: 'extract_service_id' as RoleServiceField,
                    role: 'extract' as LightRAGRole,
                    label: t('lightrag.serviceRoles.extract.label'),
                    required: true,
                    helper: t('lightrag.serviceRoles.extract.helper'),
                    placeholder: t('lightrag.serviceRoles.extract.placeholder'),
                  },
                  {
                    field: 'keywords_service_id' as RoleServiceField,
                    role: 'keywords' as LightRAGRole,
                    label: t('lightrag.serviceRoles.keywords.label'),
                    required: true,
                    helper: t('lightrag.serviceRoles.keywords.helper'),
                    placeholder: t('lightrag.serviceRoles.keywords.placeholder'),
                  },
                  {
                    field: 'vlm_service_id' as RoleServiceField,
                    role: 'vlm' as LightRAGRole,
                    label: t('lightrag.serviceRoles.vlm.label'),
                    required: false,
                    helper: t('lightrag.serviceRoles.vlm.helper'),
                    placeholder: t('lightrag.serviceRoles.vlm.placeholder'),
                  },
                ]).map(({ field, role, label, required, helper, placeholder }) => {
                  const value = formData[field];
                  const modelName = getServiceModelName(value);
                  const warning = role !== 'vlm' ? getRoleWarning(role, modelName) : null;
                  const blockingError = role === 'vlm' ? vlmBlockingError(modelName) : null;
                  return (
                    <div key={field}>
                      <label htmlFor={field} className="block text-sm font-medium text-fg dark:text-fg-dark mb-2">
                        {label}{required && <span className="text-error">*</span>}
                      </label>
                      <select
                        id={field}
                        name={field}
                        value={value?.toString() || ''}
                        onChange={handleChange}
                        required={required}
                        className="w-full px-3 py-2 border border-line rounded-lg bg-surface text-fg focus:outline-none focus:ring-2 focus:ring-focus disabled:bg-surface-hover disabled:text-fg-tertiary dark:border-line-dark dark:bg-surface-dark dark:text-fg-dark"
                        disabled={isSubmitting || (isEditing && role === 'vlm')}
                      >
                        <option value="">{placeholder}</option>
                        {aiServices.map((service: AIServiceOption) => (
                          <option key={service.service_id} value={service.service_id}>
                            {service.is_system ? t('lightrag.serviceRoles.systemService', { name: service.name }) : service.name}{service.provider ? ` (${service.provider})` : ''}
                          </option>
                        ))}
                      </select>
                      <p className="mt-1 text-sm text-fg-tertiary dark:text-fg-tertiary-dark">{helper}</p>
                      {isEditing && role === 'vlm' && (
                        <p className="mt-1 text-sm text-amber-600 flex items-center gap-1">
                          <Info className="w-3.5 h-3.5 shrink-0" />
                          {t('lightrag.siloForm.vlmImmutable')}
                        </p>
                      )}
                      <LightRAGModelHints role={role} />
                      {warning && (
                        <div className="mt-2 flex items-start gap-2 text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded p-2 dark:text-amber-200 dark:bg-amber-950/40 dark:border-amber-800">
                          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                          <span>{warning}</span>
                        </div>
                      )}
                      {blockingError && (
                        <div className="mt-2 flex items-start gap-2 text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2 dark:text-red-300 dark:bg-red-950/40 dark:border-red-800">
                          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                          <span>{blockingError}</span>
                        </div>
                      )}
                    </div>
                  );
                })}

                <LightRAGAdvancedSettings
                  formData={formData}
                  onFieldChange={(field, value) => setFormData(prev => ({ ...prev, [field]: value }))}
                  disabled={isSubmitting}
                  locked={isEditing}
                />

                <div>
                  <label htmlFor="source_label_fields" className="block text-sm font-medium text-fg dark:text-fg-dark mb-2">
                    {t('lightrag.siloForm.sourceLabelFields')}
                  </label>
                  <input
                    id="source_label_fields"
                    type="text"
                    value={formData.source_label_fields || ''}
                    onChange={(e) => setFormData(prev => ({ ...prev, source_label_fields: e.target.value }))}
                    placeholder={t('lightrag.siloForm.sourceLabelPlaceholder')}
                    className="w-full px-3 py-2 border border-line rounded-lg bg-surface text-fg focus:outline-none focus:ring-2 focus:ring-focus disabled:bg-surface-hover dark:border-line-dark dark:bg-surface-dark dark:text-fg-dark"
                    disabled={isSubmitting}
                    aria-describedby="source_label_fields_help"
                  />
                  <p id="source_label_fields_help" className="mt-1 text-sm text-fg-tertiary dark:text-fg-tertiary-dark">
                    {t('lightrag.siloForm.sourceLabelHelp')}
                  </p>
                </div>

              </div>
            )}
          </div>
          {/* Media (video/audio) indexing */}
          {(silo?.ai_services?.length ?? 0) > 0 && (
            <CardApp className="space-y-4 pt-6 mt-5" titulo={t('siloFormPage.media.title')}>
              <MediaServicesFields
                scope="silo"
                aiServices={silo?.ai_services ?? []}
                transcriptionServiceId={formData.transcription_service_id}
                videoServiceId={formData.video_ai_service_id}
                savedTranscriptionServiceId={silo?.transcription_service_id}
                savedVideoServiceId={silo?.video_ai_service_id}
                disabled={isSubmitting}
                onChange={(field, value) => setFormData(prev => ({ ...prev, [field]: value }))}
              />
            </CardApp>
          )}

          {/* Error Message */}
          {error && (
            <Alert type="error" message={error} />
          )}

          {/* Action Buttons */}
          <FormActions
            onCancel={onCancel}
            isSubmitting={isSubmitting}
            submitLabel={submitButtonLabel}
            submittingLabel={t('siloFormPage.actions.saving')}
            cancelLabel={t('siloFormPage.actions.cancel')}
            submitDisabled={!!vlmError}
            containerClassName="mt-6"
          />
        </div>
      </form>

      {/* Info Section */}
      <Alert
        type="info"
        title={t('siloFormPage.about.title')}
        className="mt-8"
        message={(
          <>
            <p>
              {t('siloFormPage.about.description')}
            </p>
            <p className="mt-2">
              <span className="font-medium">{t('siloFormPage.fields.embeddingService')}:</span> {t('siloFormPage.about.embeddingService')}
              <br />
              <span className="font-medium">{t('siloFormPage.fields.metadataDefinition')}:</span> {t('siloFormPage.about.metadataDefinition')}
            </p>
          </>
        )}
      />
    </div>
  );
}

export default SiloForm; 