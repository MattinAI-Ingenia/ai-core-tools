import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Check, AlertTriangle, Tag, BarChart2, Info, Zap, CheckCircle2, XCircle, WifiOff, HelpCircle, Plug, Shield } from 'lucide-react';
import { apiService, ApiError } from '../../services/api';
import Alert from '../../components/ui/Alert';
import { useAppRole } from '../../hooks/useAppRole';
import { AppRole } from '../../types/roles';
import ReadOnlyBanner from '../../components/ui/ReadOnlyBanner';

// Languages remembered facts can be written in (App.memory_language). Must match
// MEMORY_LANGUAGES in backend/tools/memory/settings.py; the backend rejects others.
const MEMORY_LANGUAGE_OPTIONS = [
  { code: 'en', label: 'English' },
  { code: 'es', label: 'Español' },
  { code: 'eu', label: 'Euskara' },
  { code: 'ca', label: 'Català' },
  { code: 'gl', label: 'Galego' },
  { code: 'fr', label: 'Français' },
  { code: 'de', label: 'Deutsch' },
  { code: 'it', label: 'Italiano' },
  { code: 'pt', label: 'Português' },
] as const;

type LangsmithTestStatus = 'ok' | 'unauthorized' | 'network' | 'unknown';

interface LangsmithTestResult {
  readonly status: LangsmithTestStatus;
  readonly message: string;
  readonly projectName?: string | null;
}

interface SandboxServiceOption {
  readonly service_id: number;
  readonly name: string;
  readonly provider: string;
  readonly is_system?: boolean;
}

interface MemoryServiceOption {
  readonly service_id: number;
  readonly name: string;
}

function AppSettingsPage() {
  const { appId } = useParams();
  const { hasMinRole, userRole } = useAppRole(appId);
  const canEdit = hasMinRole(AppRole.ADMINISTRATOR);

  const [formData, setFormData] = useState({
    name: '',
    langsmith_api_key: '',
    agent_rate_limit: 0,
    max_file_size_mb: 0,
    agent_cors_origins: '',
    enable_openai_api: false,
    default_sandbox_service_id: null as number | null,  // null = inherit system default
    // Cross-session user memory (D-16, FR-2b): both null = memory inactive for this
    // App, not "use a default" — there is no system-level fallback (D-21).
    memory_extraction_service_id: null as number | null,
    memory_embedding_service_id: null as number | null,
    // Fact-extraction cadence; null = deployment default.
    memory_write_every_n_turns: null as number | null,
    // Language memory facts are written in; null = English.
    memory_language: null as string | null,
  });
  const [memoryWriteEveryDefault, setMemoryWriteEveryDefault] = useState(5);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [memoryEmbeddingError, setMemoryEmbeddingError] = useState<string | null>(null);
  const [memoryExtractionError, setMemoryExtractionError] = useState<string | null>(null);
  const [memoryExtractionServices, setMemoryExtractionServices] = useState<MemoryServiceOption[]>([]);
  const [memoryEmbeddingServices, setMemoryEmbeddingServices] = useState<MemoryServiceOption[]>([]);
  const [langsmithKeyChanged, setLangsmithKeyChanged] = useState(false);
  const [originalLangsmithKey, setOriginalLangsmithKey] = useState('');
  const [testingLangsmith, setTestingLangsmith] = useState(false);
  const [langsmithTestResult, setLangsmithTestResult] = useState<LangsmithTestResult | null>(null);

  const [slugInput, setSlugInput] = useState('');
  const [savingSlug, setSavingSlug] = useState(false);
  const [slugSaved, setSlugSaved] = useState(false);
  const [slugError, setSlugError] = useState<string | null>(null);
  const [sandboxServices, setSandboxServices] = useState<SandboxServiceOption[]>([]);

  // Load app data on mount
  useEffect(() => {
    if (appId) {
      loadAppData();
      loadSlugData();
      loadSandboxServices();
    }
  }, [appId]);

  async function loadSandboxServices() {
    if (!appId) return;
    try {
      const services = await apiService.getSandboxServices(Number.parseInt(appId)) as SandboxServiceOption[];
      setSandboxServices(services ?? []);
    } catch (err) {
      console.error('Error loading sandbox services:', err);
    }
  }

  async function loadAppData() {
    if (!appId) return;

    try {
      setLoading(true);
      setError(null);
      const app = await apiService.getApp(Number.parseInt(appId));
      const langsmithKey = app.langsmith_api_key || '';
      setFormData({
        name: app.name || '',
        langsmith_api_key: langsmithKey,
        agent_rate_limit: app.agent_rate_limit || 0,
        max_file_size_mb: app.max_file_size_mb || 0,
        agent_cors_origins: app.agent_cors_origins || '',
        enable_openai_api: app.enable_openai_api || false,
        default_sandbox_service_id: app.default_sandbox_service_id ?? null,
        memory_extraction_service_id: app.memory_extraction_service_id ?? null,
        memory_embedding_service_id: app.memory_embedding_service_id ?? null,
        memory_write_every_n_turns: app.memory_write_every_n_turns ?? null,
        memory_language: app.memory_language ?? null,
      });
      setMemoryWriteEveryDefault(app.memory_write_every_n_turns_default ?? 5);
      setMemoryExtractionServices(app.ai_services ?? []);
      setMemoryEmbeddingServices(app.embedding_services ?? []);
      setOriginalLangsmithKey(langsmithKey);
      setLangsmithKeyChanged(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load app data');
      console.error('Error loading app data:', err);
    } finally {
      setLoading(false);
    }
  }

  async function loadSlugData() {
    if (!appId) return;

    try {
      const data = await apiService.getAppSlugInfo(Number.parseInt(appId));
      setSlugInput(data.slug || '');
    } catch (err) {
      console.error('Error loading slug data:', err);
    }
  }

  async function handleSlugSubmit(e?: React.FormEvent | React.MouseEvent) {
    e?.preventDefault();
    if (!appId || !slugInput.trim()) return;

    setSavingSlug(true);
    setSlugError(null);

    try {
      const data = await apiService.updateAppSlug(Number.parseInt(appId), slugInput.trim());
      setSlugInput(data.slug || '');
      setSlugSaved(true);
      setTimeout(() => setSlugSaved(false), 3000);
    } catch (err) {
      setSlugError(err instanceof Error ? err.message : 'Failed to update slug');
      console.error('Error updating slug:', err);
    } finally {
      setSavingSlug(false);
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!appId) return;
    
    setSaving(true);
    setError(null);
    setMemoryEmbeddingError(null);
    setMemoryExtractionError(null);

    try {
      await apiService.updateApp(Number.parseInt(appId), {
        name: formData.name,
        langsmith_api_key: langsmithKeyChanged ? formData.langsmith_api_key : originalLangsmithKey,
        agent_rate_limit: formData.agent_rate_limit,
        max_file_size_mb: formData.max_file_size_mb,
        agent_cors_origins: formData.agent_cors_origins,
        enable_openai_api: formData.enable_openai_api,
        default_sandbox_service_id: formData.default_sandbox_service_id,
        memory_extraction_service_id: formData.memory_extraction_service_id,
        memory_embedding_service_id: formData.memory_embedding_service_id,
        memory_write_every_n_turns: formData.memory_write_every_n_turns,
        memory_language: formData.memory_language,
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (err) {
      // AC-38: a bad memory-service config (an extraction/embedding service id that
      // doesn't exist or doesn't belong to this app) is rejected
      // with a 400 whose detail names the offending field — surface it as a
      // field-level error next to that selector instead of a generic toast.
      // `MemoryServiceConfigError` (backend/services/app_service.py) always includes
      // the field name in its message, so route on that instead of assuming it's
      // always the embeddings field.
      if (err instanceof ApiError && err.status === 400) {
        const message = err.message;
        if (/memory_embedding_service_id/i.test(message)) {
          setMemoryEmbeddingError(message);
        } else if (/memory_extraction_service_id/i.test(message)) {
          setMemoryExtractionError(message);
        } else {
          // Can't tell which memory field it refers to — fall back to the generic
          // banner instead of guessing wrong and hiding the real cause.
          setError(message);
        }
      } else {
        setError(err instanceof Error ? err.message : 'Failed to save settings');
      }
      console.error('Error saving app settings:', err);
    } finally {
      setSaving(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = (e.target.name === 'agent_rate_limit' || e.target.name === 'max_file_size_mb')
      ? Number.parseInt(e.target.value) || 0
      : e.target.type === 'checkbox'
      ? e.target.checked
      : e.target.value;

    if (e.target.name === 'langsmith_api_key') {
      setLangsmithKeyChanged(true);
      setLangsmithTestResult(null);
    }

    setFormData(prev => ({
      ...prev,
      [e.target.name]: newValue
    }));
  };

  async function handleTestLangsmith() {
    if (!appId) return;

    setTestingLangsmith(true);
    setLangsmithTestResult(null);

    try {
      const keyToTest = langsmithKeyChanged ? formData.langsmith_api_key : undefined;
      const result = await apiService.testAppLangsmith(Number.parseInt(appId), keyToTest);
      setLangsmithTestResult({
        status: result.status,
        message: result.message,
        projectName: result.project_name ?? null,
      });
    } catch (err) {
      setLangsmithTestResult({
        status: 'unknown',
        message: err instanceof Error ? err.message : 'Failed to test LangSmith connection',
      });
    } finally {
      setTestingLangsmith(false);
    }
  }

  const canTestLangsmith = (langsmithKeyChanged
    ? Boolean(formData.langsmith_api_key.trim())
    : Boolean(originalLangsmithKey));

  if (loading) {
    return (
      
        <div className="p-6 text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-gray-600">Loading app settings...</p>
        </div>
      
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <Alert 
          type="error" 
          title="Error Loading Settings" 
          message={error}
          onDismiss={loadAppData}
        />
      </div>
    );
  }

  return (
    
      <div className="p-6">
        <div className="max-w-2xl">
          {/* Header */}
          <div className="mb-6">
            <h2 className="text-xl font-semibold text-gray-900">General Settings</h2>
            <p className="text-gray-600">Configure basic app settings and integrations</p>
            {!canEdit && <ReadOnlyBanner userRole={userRole} minRole={AppRole.ADMINISTRATOR} />}
          </div>

          {/* Settings Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="bg-white shadow rounded-lg p-6">
              <div className="grid grid-cols-1 gap-6">
                {/* App Name */}
                <div>
                  <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-2">
                    App Name
                  </label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    required
                    disabled={!canEdit}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
                    placeholder="Enter app name"
                  />
                </div>

                {/* App Slug */}
                <div>
                  <label htmlFor="slug" className="block text-sm font-medium text-gray-700 mb-2">
                    App Slug
                  </label>
                  <div className="flex items-center space-x-2">
                    <input
                      type="text"
                      id="slug"
                      value={slugInput}
                      onChange={(e) => setSlugInput(e.target.value.toLowerCase().replaceAll(/[^a-z0-9-]/g, ''))}
                      disabled={!canEdit}
                      className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
                      placeholder="my-app"
                      pattern="[a-z0-9-]+"
                    />
                    {canEdit && (
                      <button
                        type="button"
                        onClick={handleSlugSubmit}
                        disabled={savingSlug || !slugInput.trim()}
                        className="bg-gray-600 hover:bg-gray-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg flex items-center text-sm whitespace-nowrap"
                      >
                        {savingSlug && (
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                        )}
                        {savingSlug ? 'Saving...' : 'Update Slug'}
                      </button>
                    )}
                  </div>
                  <p className="mt-1 text-sm text-gray-500">
                    URL-friendly identifier for your app. Use lowercase letters, numbers, and hyphens only.
                  </p>
                  {slugSaved && (
                    <p className="mt-1 text-sm text-green-600 flex items-center">
                      <Check className="w-4 h-4 mr-1" /> Slug saved successfully
                    </p>
                  )}
                  {slugError && (
                    <p className="mt-1 text-sm text-red-600">{slugError}</p>
                  )}
                </div>

                {/* Langsmith API Key */}
                <div>
                  <label htmlFor="langsmith_api_key" className="block text-sm font-medium text-gray-700 mb-2">
                    Langsmith API Key
                  </label>
                  <div className="flex items-stretch space-x-2">
                    <input
                      type="password"
                      id="langsmith_api_key"
                      name="langsmith_api_key"
                      value={formData.langsmith_api_key}
                      onChange={handleChange}
                      onKeyDown={(e) => {
                        if (!langsmithKeyChanged && formData.langsmith_api_key.startsWith('****') && e.key.length === 1) {
                          setFormData(prev => ({ ...prev, langsmith_api_key: '' }));
                          setLangsmithKeyChanged(true);
                          setLangsmithTestResult(null);
                        }
                      }}
                      autoComplete="off"
                      data-lpignore="true"
                      data-form-type="other"
                      disabled={!canEdit}
                      className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
                      placeholder={!langsmithKeyChanged && originalLangsmithKey ? 'Leave empty to keep current key' : 'Enter Langsmith API key'}
                    />
                    {canEdit && (
                      <button
                        type="button"
                        onClick={handleTestLangsmith}
                        disabled={testingLangsmith || !canTestLangsmith}
                        className="inline-flex items-center px-3 py-2 bg-gray-100 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed text-gray-700 rounded-lg text-sm whitespace-nowrap border border-gray-300"
                        title="Verify the API key against LangSmith"
                      >
                        {testingLangsmith ? (
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600 mr-2"></div>
                        ) : (
                          <Plug className="w-4 h-4 mr-2" />
                        )}
                        {testingLangsmith ? 'Testing…' : 'Test connection'}
                      </button>
                    )}
                  </div>
                  <p className="mt-1 text-sm text-gray-500">
                    Your Langsmith API key for monitoring and tracing. Leave empty to use the global
                    <code className="mx-1 bg-gray-100 px-1 rounded text-xs">LANGSMITH_API_KEY</code>
                    environment variable (when <code className="mx-1 bg-gray-100 px-1 rounded text-xs">LANGSMITH_TRACING=true</code>).
                  </p>
                  {langsmithTestResult && (
                    <div
                      role="status"
                      className={`mt-2 inline-flex items-center text-sm rounded-md px-3 py-1.5 border ${
                        langsmithTestResult.status === 'ok'
                          ? 'bg-green-50 text-green-700 border-green-200'
                          : langsmithTestResult.status === 'unauthorized'
                          ? 'bg-red-50 text-red-700 border-red-200'
                          : langsmithTestResult.status === 'network'
                          ? 'bg-amber-50 text-amber-700 border-amber-200'
                          : 'bg-gray-50 text-gray-700 border-gray-200'
                      }`}
                    >
                      {langsmithTestResult.status === 'ok' && <CheckCircle2 className="w-4 h-4 mr-2" />}
                      {langsmithTestResult.status === 'unauthorized' && <XCircle className="w-4 h-4 mr-2" />}
                      {langsmithTestResult.status === 'network' && <WifiOff className="w-4 h-4 mr-2" />}
                      {langsmithTestResult.status === 'unknown' && <HelpCircle className="w-4 h-4 mr-2" />}
                      <span>
                        {langsmithTestResult.status === 'ok' && langsmithTestResult.projectName
                          ? `Connected to LangSmith — project "${langsmithTestResult.projectName}"`
                          : langsmithTestResult.message}
                      </span>
                    </div>
                  )}
                </div>

                {/* Agent rate limit */}
                <div>
                    <label htmlFor="agent_rate_limit" className="block text-sm font-medium text-gray-700 mb-2">
                    Agent Rate Limit
                    </label>
                    <input
                    type="number"
                    id="agent_rate_limit"
                    name="agent_rate_limit"
                    value={formData.agent_rate_limit}
                    onChange={handleChange}
                    min="0"
                    disabled={!canEdit}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
                    placeholder="Enter agent rate limit"
                    />
                  <p className="mt-1 text-sm text-gray-500">
                    This will limit the number of calls your agents can make per minute in your application.
                  </p>
                </div>

                {/* Maximum File Size */}
                <div>
                    <label htmlFor="max_file_size_mb" className="block text-sm font-medium text-gray-700 mb-2">
                    Maximum File Size (MB)
                    </label>
                    <input
                    type="number"
                    id="max_file_size_mb"
                    name="max_file_size_mb"
                    value={formData.max_file_size_mb}
                    onChange={handleChange}
                    min="0"
                    disabled={!canEdit}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
                    placeholder="Enter maximum file size in MB (0 = no limit)"
                    />
                  <p className="mt-1 text-sm text-gray-500">
                    Maximum size in megabytes for files uploaded to repositories. Set to 0 for no limit.
                  </p>
                </div>

                {/* Agent CORS Origins */}
                <div>
                    <label htmlFor="agent_cors_origins" className="block text-sm font-medium text-gray-700 mb-2">
                    Agent CORS Origins
                    </label>
                    <input
                    type="text"
                    id="agent_cors_origins"
                    name="agent_cors_origins"
                    value={formData.agent_cors_origins}
                    onChange={handleChange}
                    disabled={!canEdit}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
                    placeholder="Enter allowed CORS origins (e.g., https://example.com, https://app.example.com)"
                    />
                  <p className="mt-1 text-sm text-gray-500">
                    Comma-separated list of allowed origins for CORS requests to your agents.
                  </p>
                </div>

                {/* Enable OpenAI-compatible API */}
                <div>
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="enable_openai_api"
                      name="enable_openai_api"
                      checked={formData.enable_openai_api}
                      onChange={handleChange}
                      disabled={!canEdit}
                      className="form-checkbox h-5 w-5 text-blue-600 rounded disabled:cursor-not-allowed"
                    />
                    <span className="text-sm font-medium text-gray-700">Enable OpenAI-compatible API</span>
                  </label>
                  <p className="mt-1 text-sm text-gray-500">
                    Allow external applications to interact with your agents using the OpenAI API format. 
                    When enabled, your agents become compatible with tools and clients that expect OpenAI-compatible API endpoints.
                  </p>
                </div>

                {/* Sandbox Service */}
                <div>
                  <label htmlFor="default_sandbox_service_id" className="block text-sm font-medium text-gray-700 mb-2">
                    Code Interpreter Sandbox Service
                  </label>
                  <select
                    id="default_sandbox_service_id"
                    name="default_sandbox_service_id"
                    value={formData.default_sandbox_service_id ?? ''}
                    onChange={(e) => setFormData(prev => ({
                      ...prev,
                      default_sandbox_service_id: e.target.value ? Number.parseInt(e.target.value) : null,
                    }))}
                    disabled={!canEdit}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
                  >
                    <option value="">Inherit system default</option>
                    {sandboxServices.map((service) => (
                      <option key={service.service_id} value={service.service_id}>
                        {service.is_system ? `[System] ${service.name}` : service.name} ({service.provider})
                      </option>
                    ))}
                  </select>
                  <p className="mt-1 text-sm text-gray-500">
                    Default Sandbox Service used when agents in this app run the Code Interpreter capability.
                    Leave at "Inherit" to use the system default. Manage available services under{' '}
                    <span className="font-medium">Settings → Sandbox Services</span>.
                  </p>
                </div>

                {/* Cross-session memory: extraction model */}
                <div>
                  <label htmlFor="memory_extraction_service_id" className="block text-sm font-medium text-gray-700 mb-2">
                    Memory extraction model
                  </label>
                  <select
                    id="memory_extraction_service_id"
                    name="memory_extraction_service_id"
                    value={formData.memory_extraction_service_id ?? ''}
                    onChange={(e) => {
                      setMemoryExtractionError(null);
                      setFormData(prev => ({
                        ...prev,
                        memory_extraction_service_id: e.target.value ? Number.parseInt(e.target.value) : null,
                      }));
                    }}
                    disabled={!canEdit}
                    aria-describedby="memory_extraction_service_id_help memory_extraction_service_id_error"
                    aria-invalid={memoryExtractionError ? true : undefined}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
                  >
                    <option value="">None — cross-session memory disabled</option>
                    {memoryExtractionServices.map((service) => (
                      <option key={service.service_id} value={service.service_id}>
                        {service.name}
                      </option>
                    ))}
                  </select>
                  <p id="memory_extraction_service_id_help" className="mt-1 text-sm text-gray-500">
                    Cheap/lightweight LLM used to extract durable facts about a user from a conversation, for
                    cross-session memory. Leaving this empty means memory stays <strong>inactive</strong> for
                    every agent in this app — there is no automatic fallback. Both this and the embeddings
                    selector below must be set for the per-agent "Cross-session user memory" toggle to do
                    anything.
                  </p>
                  {memoryExtractionError && (
                    <p id="memory_extraction_service_id_error" role="alert" className="mt-1 text-sm text-red-600">
                      {memoryExtractionError}
                    </p>
                  )}
                </div>

                {/* Cross-session memory: embeddings */}
                <div>
                  <label htmlFor="memory_embedding_service_id" className="block text-sm font-medium text-gray-700 mb-2">
                    Memory embeddings
                  </label>
                  <select
                    id="memory_embedding_service_id"
                    name="memory_embedding_service_id"
                    value={formData.memory_embedding_service_id ?? ''}
                    onChange={(e) => {
                      setMemoryEmbeddingError(null);
                      setFormData(prev => ({
                        ...prev,
                        memory_embedding_service_id: e.target.value ? Number.parseInt(e.target.value) : null,
                      }));
                    }}
                    disabled={!canEdit}
                    aria-describedby="memory_embedding_service_id_help memory_embedding_service_id_error"
                    aria-invalid={memoryEmbeddingError ? true : undefined}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
                  >
                    <option value="">None — cross-session memory disabled</option>
                    {memoryEmbeddingServices.map((service) => (
                      <option key={service.service_id} value={service.service_id}>
                        {service.name}
                      </option>
                    ))}
                  </select>
                  <p id="memory_embedding_service_id_help" className="mt-1 text-sm text-gray-500">
                    Embedding model for cross-session memory vectors. Leaving this empty means memory stays
                    <strong> inactive</strong> for every agent in this app. This applies to the whole app at
                    once — every memory-enabled agent here shares it.
                  </p>
                  <p className="mt-1 text-sm text-amber-700 flex items-start gap-1">
                    <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" aria-hidden="true" />
                    <span>
                      Changing this after facts have been stored may make previously stored memory
                      unretrievable — stored memory is not re-embedded.
                    </span>
                  </p>
                  {memoryEmbeddingError && (
                    <p id="memory_embedding_service_id_error" role="alert" className="mt-1 text-sm text-red-600">
                      {memoryEmbeddingError}
                    </p>
                  )}
                </div>

                {/* Cross-session memory: language facts are written in */}
                <div>
                  <label htmlFor="memory_language" className="block text-sm font-medium text-gray-700 mb-2">
                    Memory language
                  </label>
                  <select
                    id="memory_language"
                    name="memory_language"
                    value={formData.memory_language ?? 'en'}
                    onChange={(e) => {
                      const value = e.target.value;
                      setFormData(prev => ({ ...prev, memory_language: value === 'en' ? null : value }));
                    }}
                    disabled={!canEdit}
                    aria-describedby="memory_language_help"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
                  >
                    {MEMORY_LANGUAGE_OPTIONS.map((option) => (
                      <option key={option.code} value={option.code}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <p id="memory_language_help" className="mt-1 text-sm text-gray-500">
                    Language remembered facts are written in, whatever language users write in. Users see them
                    in this language under Preferences. Changing it does not translate facts already stored.
                  </p>
                </div>

                {/* Cross-session memory: how often facts are extracted */}
                <div>
                  <label htmlFor="memory_write_every_n_turns" className="block text-sm font-medium text-gray-700 mb-2">
                    Learn from the conversation every
                  </label>
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      id="memory_write_every_n_turns"
                      name="memory_write_every_n_turns"
                      min={1}
                      max={100}
                      step={1}
                      inputMode="numeric"
                      value={formData.memory_write_every_n_turns ?? ''}
                      placeholder={String(memoryWriteEveryDefault)}
                      onChange={(e) => {
                        const parsed = Number.parseInt(e.target.value, 10);
                        setFormData(prev => ({
                          ...prev,
                          memory_write_every_n_turns: Number.isNaN(parsed) ? null : Math.min(Math.max(parsed, 1), 100),
                        }));
                      }}
                      disabled={!canEdit}
                      aria-describedby="memory_write_every_n_turns_help"
                      className="w-24 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
                    />
                    <span className="text-sm text-gray-700">messages</span>
                  </div>
                  <p id="memory_write_every_n_turns_help" className="mt-1 text-sm text-gray-500">
                    Durable facts are extracted on the first message of a conversation and then every this
                    many messages, from all the user&apos;s messages since the last time. Lower values learn
                    sooner but cost more extraction calls. Leave empty to use the default
                    ({memoryWriteEveryDefault}).
                  </p>
                </div>
              </div>

              {/* Error Display */}
              {error && (
                <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <AlertTriangle className="w-5 h-5 text-red-400" />
                    </div>
                    <div className="ml-3">
                      <h3 className="text-sm font-medium text-red-800">
                        Error Saving Settings
                      </h3>
                      <div className="mt-2 text-sm text-red-700">
                        <p>{error}</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Submit Button - Only show for admins */}
              {canEdit && (
                <div className="mt-6 flex items-center justify-between">
                  <div className="flex items-center">
                    {saved && (
                      <div className="flex items-center text-green-600">
                        <Check className="w-4 h-4 mr-1" /> Settings saved successfully
                      </div>
                    )}
                  </div>
                  <button
                    type="submit"
                    disabled={saving}
                    className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white px-6 py-2 rounded-lg flex items-center"
                  >
                    {saving && (
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    )}
                    {saving ? 'Saving...' : 'Save Changes'}
                  </button>
                </div>
              )}
            </div>
          </form>

          {/* Info Sections */}
          <div className="mt-8 space-y-6">
            {/* App Slug Info */}
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <Tag className="w-5 h-5 text-gray-400" />
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-gray-800">
                    About App Slug
                  </h3>
                  <div className="mt-2 text-sm text-gray-700">
                    <p>
                      The app slug is a unique, URL-friendly identifier for your app.
                      It's used in various endpoints including MCP server URLs to make them more readable
                      (e.g., <code className="bg-gray-200 px-1 rounded text-xs">/mcp/v1/my-app/my-server</code> instead of using numeric IDs).
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Langsmith Info */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <BarChart2 className="w-5 h-5 text-blue-400" />
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-blue-800">
                    About Langsmith Integration
                  </h3>
                  <div className="mt-2 text-sm text-blue-700">
                    <p>
                      Langsmith provides monitoring, tracing, and debugging capabilities for your AI agents.
                      Connect your Langsmith account to track agent performance and troubleshoot issues.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* OpenAI-compatible API Info */}
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <Zap className="w-5 h-5 text-purple-400" />
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-purple-800">
                    About OpenAI-compatible API
                  </h3>
                  <div className="mt-2 text-sm text-purple-700">
                    <p className="mb-2">
                      When enabled, your agents can be accessed through OpenAI-compatible API endpoints. 
                      This allows integration with third-party tools and clients that expect OpenAI API format.
                    </p>
                    <p className="mb-2">
                      <strong>Impact of this setting:</strong>
                    </p>
                    <ul className="list-disc list-inside space-y-1">
                      <li>Enables compatibility with OpenAI-compatible clients and SDKs</li>
                      <li>Provides standard chat completion endpoints</li>
                      <li>Allows seamless integration with tools expecting OpenAI API interface</li>
                      <li>Does not affect your existing internal APIs or agent functionality</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {/* App Settings Info */}
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <Info className="w-5 h-5 text-gray-400" />
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-gray-800">
                    App Configuration
                  </h3>
                  <div className="mt-2 text-sm text-gray-700">
                    <p>
                      These settings affect the overall behavior of your app. 
                      The app name is used in navigation and API responses. 
                      Changes to these settings are applied immediately.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Sandbox Service Info */}
            <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <Shield className="w-5 h-5 text-orange-400" />
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-orange-800">
                    About Sandbox Services
                  </h3>
                  <div className="mt-2 text-sm text-orange-700">
                    <p className="mb-2">
                      A Sandbox Service controls the isolated environment where agents execute Python code
                      when the Code Interpreter capability is enabled. Configure OpenSandbox (self-hosted,
                      container-isolated), Daytona, or E2B (managed cloud sandboxes) services under{' '}
                      <strong>Settings → Sandbox Services</strong>.
                    </p>
                    <ul className="list-disc list-inside space-y-1">
                      <li><strong>Select a service</strong> — pins this app's agents to a specific sandbox configuration.</li>
                      <li><strong>Inherit</strong> — uses the platform default configured by your system administrator.</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    
  );
}

export default AppSettingsPage; 
