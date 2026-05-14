import { useState, useEffect } from 'react';
import FormActions from './FormActions';
import { apiService } from '../../services/api';
import type { MCPConfig } from '../../core/types';

type HookType = 'before_model' | 'after_model' | 'wrap_model' | 'before_tool' | 'after_tool' | 'wrap_tool' | 'callback';

interface MiddlewareTypeInfo {
    value: string;
    label: string;
    description: string;
    hooks: HookType[];
    hasLimit?: boolean;
    limitLabel?: string;
    limitDefault?: number;
}

interface MiddlewareFormData {
    name: string;
    description: string;
    middleware_type: string;
    config?: Record<string, any> | null;
}

interface MiddlewareItem {
    middleware_id: number;
    name: string;
    description: string;
    middleware_type: string;
    config?: Record<string, any> | null;
    created_at: string;
}

interface HitlToolEntry {
    name: string;
    decisions: ('approve' | 'edit' | 'reject')[];
}

interface HitlToolOption {
    name: string;
    label: string;
    description?: string;
}

interface HitlMcpSource {
    configId: number;
    name: string;
    description: string;
    tools: HitlToolOption[];
    error?: string;
}

interface MiddlewareFormProps {
    middleware?: MiddlewareItem | null;
    appId?: number | string;
    onSubmit: (data: MiddlewareFormData) => Promise<void>;
    onCancel: () => void;
}

const HOOK_STYLES: Record<HookType, { label: string; bg: string; text: string }> = {
    before_model: { label: 'Before Model', bg: 'bg-blue-100', text: 'text-blue-700' },
    after_model: { label: 'After Model', bg: 'bg-blue-100', text: 'text-blue-700' },
    wrap_model: { label: 'Wrap Model', bg: 'bg-purple-100', text: 'text-purple-700' },
    before_tool: { label: 'Before Tool', bg: 'bg-amber-100', text: 'text-amber-700' },
    after_tool: { label: 'After Tool', bg: 'bg-amber-100', text: 'text-amber-700' },
    wrap_tool: { label: 'Wrap Tool', bg: 'bg-orange-100', text: 'text-orange-700' },
    callback: { label: 'Callback', bg: 'bg-gray-100', text: 'text-gray-700' },
};

const MIDDLEWARE_TYPES: MiddlewareTypeInfo[] = [
    {
        value: 'monitoring',
        label: 'Monitoring',
        description: 'Tracks token usage (input/output tokens) and number of LLM calls per conversation turn.',
        hooks: ['callback'],
    },
    {
        value: 'summarization',
        label: 'Summarization',
        description: 'Automatically summarizes conversation history when it exceeds token/message limits to keep context manageable.',
        hooks: ['before_model'],
    },
    {
        value: 'model_call_limit',
        label: 'Model Call Limit',
        description: 'Limits the number of LLM calls per agent run to prevent infinite loops and control costs.',
        hooks: ['before_model', 'after_model'],
        hasLimit: true,
        limitLabel: 'Max LLM calls per run',
        limitDefault: 50,
    },
    {
        value: 'tool_call_limit',
        label: 'Tool Call Limit',
        description: 'Limits the total number of tool invocations per agent run to prevent runaway execution.',
        hooks: ['after_model'],
        hasLimit: true,
        limitLabel: 'Max tool calls per run',
        limitDefault: 100,
    },
    {
        value: 'pii',
        label: 'PII Detection',
        description: 'Detects and redacts personally identifiable information before sending to the LLM, and restores it in responses.',
        hooks: ['before_model', 'after_model'],
    },
    {
        value: 'human_in_the_loop',
        label: 'Human in the Loop',
        description: 'Pauses agent execution before selected tools run and waits for human approval, edit, or rejection.',
        hooks: ['after_model'],
    },
];

const BUILTIN_TOOLS: { name: string; label: string }[] = [
    { name: 'get_current_date', label: 'Get Current Date' },
    { name: 'python_repl', label: 'Python REPL (code interpreter)' },
    { name: 'download_url_to_workspace', label: 'Download URL to Workspace' },
];

const ALL_HITL_DECISIONS = ['approve', 'edit', 'reject'] as const;
type HitlDecision = typeof ALL_HITL_DECISIONS[number];

type SummarizationModelOption = { value: string; label: string; provider: string | null; description: string };

const SUMMARIZATION_MODELS: SummarizationModelOption[] = [
    { value: 'agent_llm', label: "Agent's LLM (default)", provider: null, description: "Uses the same LLM configured on the agent — no extra cost" },
    // OpenAI
    { value: 'openai:gpt-5.4-mini', label: 'GPT-5.4 mini', provider: 'OpenAI', description: '$0.75 / $4.50 per 1M tokens — strong mini model' },
    { value: 'openai:gpt-5.4-nano', label: 'GPT-5.4 nano', provider: 'OpenAI', description: '$0.20 / $1.25 per 1M tokens — cheapest OpenAI option' },
    // Anthropic
    { value: 'anthropic:claude-haiku-4-5', label: 'Claude Haiku 4.5', provider: 'Anthropic', description: '$1 / $5 per 1M tokens — fastest Claude' },
    { value: 'anthropic:claude-haiku-3-5', label: 'Claude Haiku 3.5', provider: 'Anthropic', description: '$0.80 / $4 per 1M tokens — fast and cheap' },
    { value: 'anthropic:claude-haiku-3', label: 'Claude Haiku 3', provider: 'Anthropic', description: '$0.25 / $1.25 per 1M tokens — lowest Anthropic cost' },
    // Mistral
    { value: 'mistral:ministral-3b-2512', label: 'Ministral 3B', provider: 'Mistral', description: '$0.10 / $0.10 per 1M tokens — tiny and ultra-efficient' },
    { value: 'mistral:mistral-small-2603', label: 'Mistral Small 4', provider: 'Mistral', description: '$0.15 / $0.60 per 1M tokens — powerful hybrid model' },
];

const SUMMARIZATION_MODEL_PROVIDERS = [
    { label: 'OpenAI', values: SUMMARIZATION_MODELS.filter(m => m.provider === 'OpenAI') },
    { label: 'Anthropic', values: SUMMARIZATION_MODELS.filter(m => m.provider === 'Anthropic') },
    { label: 'Mistral', values: SUMMARIZATION_MODELS.filter(m => m.provider === 'Mistral') },
];

function MiddlewareForm({ middleware, appId, onSubmit, onCancel }: Readonly<MiddlewareFormProps>) {
    const [formData, setFormData] = useState<MiddlewareFormData>({
        name: '',
        description: '',
        middleware_type: 'monitoring',
        config: null
    });
    const [limitValue, setLimitValue] = useState<number | ''>('');
    const [summarizationModel, setSummarizationModel] = useState('agent_llm');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [hitlTools, setHitlTools] = useState<HitlToolEntry[]>([]);
    const [customToolInput, setCustomToolInput] = useState('');
    const [appAgentTools, setAppAgentTools] = useState<HitlToolOption[]>([]);
    const [appMcpSources, setAppMcpSources] = useState<HitlMcpSource[]>([]);
    const [loadingHitlSources, setLoadingHitlSources] = useState(false);

    const isEditing = !!middleware && middleware.middleware_id !== 0;

    useEffect(() => {
        if (middleware) {
            setFormData({
                name: middleware.name || '',
                description: middleware.description || '',
                middleware_type: middleware.middleware_type || 'monitoring',
                config: middleware.config || null
            });
            if (middleware.config?.max_calls) {
                setLimitValue(middleware.config.max_calls);
            }
            if (middleware.config?.summarization_model) {
                setSummarizationModel(middleware.config.summarization_model);
            }
            if (middleware.middleware_type === 'human_in_the_loop' && middleware.config?.interrupt_on) {
                const entries: HitlToolEntry[] = Object.entries(middleware.config.interrupt_on).map(
                    ([name, cfg]: [string, any]) => ({
                        name,
                        decisions: (cfg?.allowed_decisions ?? ['approve', 'edit', 'reject']) as HitlDecision[],
                    })
                );
                setHitlTools(entries);
            }
        }
    }, [middleware]);

    useEffect(() => {
        if (formData.middleware_type !== 'human_in_the_loop' || !appId) return;
        let cancelled = false;

        const loadHitlSources = async () => {
            setLoadingHitlSources(true);

            try {
                const appIdNumber = Number(appId);
                const [agentsResponse, mcpConfigsResponse] = await Promise.all([
                    apiService.getAgents(appIdNumber),
                    apiService.getMCPConfigs(appIdNumber),
                ]);

                const toolAgents = (agentsResponse as any[])
                    .filter((agent) => agent.is_tool)
                    .map((agent) => ({
                        name: (agent.name as string).replace(/ /g, '_'),
                        label: agent.name,
                        description: agent.description || 'Agent exposed as a tool',
                    }));

                const mcpSources = await Promise.all(
                    (mcpConfigsResponse as MCPConfig[]).map(async (config) => {
                        try {
                            const testResult = await apiService.testMCPConnection(appIdNumber, config.config_id);
                            const tools = Array.isArray(testResult?.tools)
                                ? testResult.tools.map((tool: any) => ({
                                    name: tool.name,
                                    label: tool.name,
                                    description: tool.description || '',
                                }))
                                : [];

                            return {
                                configId: config.config_id,
                                name: config.name,
                                description: config.description || 'MCP configurado en esta app',
                                tools,
                            } as HitlMcpSource;
                        } catch (loadError) {
                            return {
                                configId: config.config_id,
                                name: config.name,
                                description: config.description || 'MCP configurado en esta app',
                                tools: [],
                                error: loadError instanceof Error ? loadError.message : 'No se pudieron cargar sus tools',
                            } as HitlMcpSource;
                        }
                    })
                );

                if (!cancelled) {
                    setAppAgentTools(toolAgents);
                    setAppMcpSources(mcpSources);
                }
            } catch (loadError) {
                if (!cancelled) {
                    setAppAgentTools([]);
                    setAppMcpSources([]);
                    setError(loadError instanceof Error ? loadError.message : 'Failed to load HITL sources');
                }
            } finally {
                if (!cancelled) {
                    setLoadingHitlSources(false);
                }
            }
        };

        void loadHitlSources();

        return () => {
            cancelled = true;
        };
    }, [formData.middleware_type, appId]);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleTypeSelect = (typeValue: string) => {
        const typeInfo = MIDDLEWARE_TYPES.find(t => t.value === typeValue);
        if (!typeInfo) return;
        let newConfig: Record<string, any> | null = null;
        if (typeInfo.hasLimit) {
            newConfig = { max_calls: typeInfo.limitDefault };
        }
        if (typeValue === 'summarization') {
            newConfig = { summarization_model: 'agent_llm' };
            setSummarizationModel('agent_llm');
        }
        if (typeValue === 'human_in_the_loop') {
            newConfig = { interrupt_on: {} };
            setHitlTools([]);
            setCustomToolInput('');
            setAppAgentTools([]);
            setAppMcpSources([]);
        }
        setLimitValue(typeInfo.hasLimit ? (typeInfo.limitDefault ?? '') : '');
        setFormData(prev => ({
            ...prev,
            middleware_type: typeValue,
            name: typeInfo.label,
            description: typeInfo.description,
            config: newConfig
        }));
    };

    // HITL helpers
    const toggleTool = (toolName: string) => {
        setHitlTools(prev => {
            if (prev.find(t => t.name === toolName)) {
                return prev.filter(t => t.name !== toolName);
            }
            return [...prev, { name: toolName, decisions: ['approve', 'edit', 'reject'] }];
        });
    };

    const toggleDecision = (toolName: string, decision: HitlDecision) => {
        setHitlTools(prev => prev.map(t => {
            if (t.name !== toolName) return t;
            const has = t.decisions.includes(decision);
            const next = has ? t.decisions.filter(d => d !== decision) : [...t.decisions, decision];
            return { ...t, decisions: next };
        }));
    };

    const addCustomTool = () => {
        const name = customToolInput.trim();
        if (!name || hitlTools.find(t => t.name === name)) return;
        setHitlTools(prev => [...prev, { name, decisions: ['approve', 'edit', 'reject'] }]);
        setCustomToolInput('');
    };

    const removeCustomTool = (toolName: string) => {
        setHitlTools(prev => prev.filter(t => t.name !== toolName));
    };

    const handleModelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const model = e.target.value;
        setSummarizationModel(model);
        setFormData(prev => ({
            ...prev,
            config: { ...prev.config, summarization_model: model }
        }));
    };

    const handleLimitChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const val = e.target.value === '' ? '' : parseInt(e.target.value, 10);
        setLimitValue(val);
        setFormData(prev => ({
            ...prev,
            config: val === '' ? null : { max_calls: val }
        }));
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        if (!formData.name.trim()) {
            setError('Middleware name is required');
            return;
        }

        if (formData.middleware_type === 'human_in_the_loop' && hitlTools.length === 0) {
            setError('Select at least one tool that requires human approval.');
            return;
        }

        if (formData.middleware_type === 'human_in_the_loop') {
            const invalid = hitlTools.filter(t => t.decisions.length === 0);
            if (invalid.length > 0) {
                setError(`Tool "${invalid[0].name}" must have at least one allowed decision.`);
                return;
            }
        }

        setIsSubmitting(true);
        setError(null);

        try {
            let submitData = formData;
            if (formData.middleware_type === 'human_in_the_loop') {
                const interrupt_on: Record<string, { allowed_decisions: string[] }> = {};
                hitlTools.forEach(t => { interrupt_on[t.name] = { allowed_decisions: t.decisions }; });
                submitData = { ...formData, config: { interrupt_on } };
            }
            await onSubmit(submitData);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to save middleware');
        } finally {
            setIsSubmitting(false);
        }
    };

    const selectedType = MIDDLEWARE_TYPES.find(t => t.value === formData.middleware_type);

    return (
        <form onSubmit={handleSubmit} className="space-y-6">
            {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded relative">
                    {error}
                </div>
            )}

            {/* Middleware Type Selection */}
            <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                    Type <span className="text-red-500">*</span>
                </label>
                <div className="space-y-2 max-h-[400px] overflow-y-auto pr-1">
                    {MIDDLEWARE_TYPES.map((type) => (
                        <button
                            key={type.value}
                            type="button"
                            className={`w-full p-4 rounded-xl border-2 text-left transition-all duration-200 ${formData.middleware_type === type.value
                                ? 'border-indigo-500 bg-indigo-50'
                                : 'border-gray-200 bg-gray-50 hover:border-gray-300'
                                }`}
                            onClick={() => handleTypeSelect(type.value)}
                        >
                            <div className="flex items-center justify-between">
                                <span className="text-sm font-medium text-gray-900">{type.label}</span>
                                <div className={`w-3 h-3 rounded-full ${formData.middleware_type === type.value ? 'bg-indigo-500' : 'bg-gray-300'
                                    }`} />
                            </div>
                            <p className="mt-1 text-xs text-gray-500">{type.description}</p>
                            <div className="mt-2 flex flex-wrap gap-1">
                                {type.hooks.map((hook) => {
                                    const style = HOOK_STYLES[hook];
                                    return (
                                        <span
                                            key={hook}
                                            className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium ${style.bg} ${style.text}`}
                                        >
                                            {style.label}
                                        </span>
                                    );
                                })}
                            </div>
                        </button>
                    ))}
                </div>

                {/* Coming soon notice */}
                <div className="mt-3 p-3 bg-gray-50 border border-gray-200 rounded-lg">
                    <p className="text-xs text-gray-500 italic">
                        Custom middleware (upload .py files with LangChain middleware classes) will be available in a future release.
                    </p>
                </div>
            </div>

            {/* Limit Configuration - shown for types that need it */}
            {selectedType?.hasLimit && (
                <div>
                    <label htmlFor="limit" className="block text-sm font-medium text-gray-700 mb-1">
                        {selectedType.limitLabel} <span className="text-red-500">*</span>
                    </label>
                    <input
                        type="number"
                        id="limit"
                        min={1}
                        max={10000}
                        value={limitValue}
                        onChange={handleLimitChange}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
                        placeholder={`Default: ${selectedType.limitDefault}`}
                        disabled={isSubmitting}
                    />
                    <p className="mt-1 text-xs text-gray-500">
                        Agent execution will stop after reaching this limit. Default: {selectedType.limitDefault}
                    </p>
                </div>
            )}

            {/* Summarization Model Selector */}
            {formData.middleware_type === 'summarization' && (
                <div>
                    <label htmlFor="summarization_model" className="block text-sm font-medium text-gray-700 mb-1">
                        Summarization Model
                    </label>
                    <select
                        id="summarization_model"
                        value={summarizationModel}
                        onChange={handleModelChange}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
                        disabled={isSubmitting}
                    >
                        <option value="agent_llm">Agent's LLM (default)</option>
                        {SUMMARIZATION_MODEL_PROVIDERS.map(({ label, values }) => (
                            <optgroup key={label} label={label}>
                                {values.map((model) => (
                                    <option key={model.value} value={model.value}>
                                        {model.label}
                                    </option>
                                ))}
                            </optgroup>
                        ))}
                    </select>
                    <p className="mt-1 text-xs text-gray-500">
                        {SUMMARIZATION_MODELS.find(m => m.value === summarizationModel)?.description ||
                            'Select a model for conversation summarization'}
                    </p>
                    {summarizationModel !== 'agent_llm' && (
                        <p className="mt-1 text-xs text-blue-600">
                            Uses the API key of the matching provider AIService configured in this app.
                        </p>
                    )}
                </div>
            )}

            {/* Human in the Loop — tool selector */}
            {formData.middleware_type === 'human_in_the_loop' && (
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Tools requiring approval <span className="text-red-500">*</span>
                    </label>

                    {loadingHitlSources ? (
                        <p className="text-sm text-gray-500">Cargando tools y MCPs de la app…</p>
                    ) : (
                        <div className="space-y-4">
                            <div className="rounded-lg border border-indigo-100 bg-indigo-50/50 p-4">
                                <h4 className="text-sm font-semibold text-gray-900">Tools de la app</h4>
                                <p className="mt-1 text-xs text-gray-500">Aquí aparecen los agentes marcados como tool en esta app.</p>
                                <div className="mt-3 space-y-2">
                                    {appAgentTools.length > 0 ? appAgentTools.map((tool) => {
                                        const entry = hitlTools.find(t => t.name === tool.name);
                                        const isSelected = !!entry;
                                        return (
                                            <div key={tool.name} className={`rounded-md border px-4 py-3 ${isSelected ? 'border-indigo-300 bg-white' : 'border-indigo-100 bg-white/70'}`}>
                                                <div className="flex items-center justify-between gap-4">
                                                    <label className="flex items-center gap-2 cursor-pointer min-w-0">
                                                        <input
                                                            type="checkbox"
                                                            checked={isSelected}
                                                            onChange={() => toggleTool(tool.name)}
                                                            disabled={isSubmitting}
                                                            className="h-4 w-4 rounded border-gray-300 text-indigo-600"
                                                        />
                                                        <span className="text-sm font-medium text-gray-900 truncate">{tool.label}</span>
                                                        <code className="text-xs text-gray-400 shrink-0">{tool.name}</code>
                                                    </label>
                                                    {isSelected && (
                                                        <div className="flex flex-wrap gap-3 shrink-0">
                                                            {ALL_HITL_DECISIONS.map((decision) => (
                                                                <label key={decision} className="flex items-center gap-1 cursor-pointer">
                                                                    <input
                                                                        type="checkbox"
                                                                        checked={entry.decisions.includes(decision)}
                                                                        onChange={() => toggleDecision(tool.name, decision)}
                                                                        disabled={isSubmitting}
                                                                        className="h-3.5 w-3.5 rounded border-gray-300 text-indigo-600"
                                                                    />
                                                                    <span className="text-xs capitalize text-gray-600">{decision}</span>
                                                                </label>
                                                            ))}
                                                        </div>
                                                    )}
                                                </div>
                                                {tool.description && <p className="mt-1 text-xs text-gray-500">{tool.description}</p>}
                                            </div>
                                        );
                                    }) : (
                                        <p className="text-sm text-gray-500">No hay agentes configurados como tool en esta app.</p>
                                    )}
                                </div>
                            </div>

                            <div className="rounded-lg border border-indigo-100 bg-indigo-50/50 p-4">
                                <h4 className="text-sm font-semibold text-gray-900">MCPs de la app</h4>
                                <p className="mt-1 text-xs text-gray-500">Para cada MCP se prueban sus conexiones y se listan las tools que devuelve el servidor.</p>
                                <div className="mt-3 space-y-3">
                                    {appMcpSources.length > 0 ? appMcpSources.map((mcp) => (
                                        <div key={mcp.configId} className="rounded-md border border-indigo-100 bg-white p-4">
                                            <h5 className="text-sm font-medium text-gray-900">{mcp.name}</h5>
                                            <p className="mt-1 text-xs text-gray-500">{mcp.description}</p>

                                            {mcp.error ? (
                                                <p className="mt-3 text-xs text-amber-600">No se pudieron cargar sus tools: {mcp.error}</p>
                                            ) : mcp.tools.length > 0 ? (
                                                <div className="mt-3 space-y-2">
                                                    {mcp.tools.map((tool) => {
                                                        const entry = hitlTools.find(t => t.name === tool.name);
                                                        const isSelected = !!entry;
                                                        return (
                                                            <div key={tool.name} className={`rounded-md border px-4 py-3 ${isSelected ? 'border-indigo-300 bg-indigo-50' : 'border-gray-200 bg-gray-50'}`}>
                                                                <div className="flex items-center justify-between gap-4">
                                                                    <label className="flex items-center gap-2 cursor-pointer min-w-0">
                                                                        <input
                                                                            type="checkbox"
                                                                            checked={isSelected}
                                                                            onChange={() => toggleTool(tool.name)}
                                                                            disabled={isSubmitting}
                                                                            className="h-4 w-4 rounded border-gray-300 text-indigo-600"
                                                                        />
                                                                        <span className="text-sm text-gray-900 truncate">{tool.label}</span>
                                                                        <code className="text-xs text-gray-400 shrink-0">{tool.name}</code>
                                                                    </label>
                                                                    {isSelected && (
                                                                        <div className="flex flex-wrap gap-3 shrink-0">
                                                                            {ALL_HITL_DECISIONS.map((decision) => (
                                                                                <label key={decision} className="flex items-center gap-1 cursor-pointer">
                                                                                    <input
                                                                                        type="checkbox"
                                                                                        checked={entry.decisions.includes(decision)}
                                                                                        onChange={() => toggleDecision(tool.name, decision)}
                                                                                        disabled={isSubmitting}
                                                                                        className="h-3.5 w-3.5 rounded border-gray-300 text-indigo-600"
                                                                                    />
                                                                                    <span className="text-xs capitalize text-gray-600">{decision}</span>
                                                                                </label>
                                                                            ))}
                                                                        </div>
                                                                    )}
                                                                </div>
                                                                {tool.description && <p className="mt-1 text-xs text-gray-500">{tool.description}</p>}
                                                            </div>
                                                        );
                                                    })}
                                                </div>
                                            ) : (
                                                <p className="mt-3 text-sm text-gray-500">Este MCP no devolvió tools visibles.</p>
                                            )}
                                        </div>
                                    )) : (
                                        <p className="text-sm text-gray-500">No hay MCPs configurados en esta app.</p>
                                    )}
                                </div>
                            </div>

                            <div className="rounded-lg border border-gray-200 bg-white p-4">
                                <h4 className="text-sm font-semibold text-gray-900">Tools manuales</h4>
                                <p className="mt-1 text-xs text-gray-500">Úsalo solo si la tool no aparece arriba. Aquí puedes meter MCP tools o tools personalizadas por nombre.</p>
                                <div className="mt-3 flex gap-2">
                                    <input
                                        type="text"
                                        value={customToolInput}
                                        onChange={e => setCustomToolInput(e.target.value)}
                                        onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); addCustomTool(); } }}
                                        placeholder="e.g., web_search, send_email..."
                                        disabled={isSubmitting}
                                        className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-indigo-500 focus:border-indigo-500"
                                    />
                                    <button
                                        type="button"
                                        onClick={addCustomTool}
                                        disabled={isSubmitting || !customToolInput.trim()}
                                        className="px-3 py-2 bg-gray-100 hover:bg-gray-200 disabled:opacity-40 rounded-md text-sm font-medium"
                                    >
                                        Add
                                    </button>
                                </div>

                                {hitlTools
                                    .filter(tool =>
                                        !appAgentTools.find(a => a.name === tool.name) &&
                                        !BUILTIN_TOOLS.find(b => b.name === tool.name) &&
                                        !appMcpSources.some(mcp => mcp.tools.some(mcpTool => mcpTool.name === tool.name))
                                    )
                                    .map(entry => (
                                        <div key={entry.name} className="mt-3 rounded-md border border-gray-200 bg-gray-50 px-4 py-3">
                                            <div className="flex items-center justify-between gap-4">
                                                <code className="text-sm text-gray-800">{entry.name}</code>
                                                <div className="flex items-center gap-3 shrink-0">
                                                    {ALL_HITL_DECISIONS.map((decision) => (
                                                        <label key={decision} className="flex items-center gap-1 cursor-pointer">
                                                            <input
                                                                type="checkbox"
                                                                checked={entry.decisions.includes(decision)}
                                                                onChange={() => toggleDecision(entry.name, decision)}
                                                                disabled={isSubmitting}
                                                                className="h-3.5 w-3.5 rounded border-gray-300 text-indigo-600"
                                                            />
                                                            <span className="text-xs capitalize text-gray-600">{decision}</span>
                                                        </label>
                                                    ))}
                                                    <button
                                                        type="button"
                                                        onClick={() => removeCustomTool(entry.name)}
                                                        disabled={isSubmitting}
                                                        className="ml-1 text-red-400 hover:text-red-600 text-lg leading-none"
                                                        aria-label={`Remove ${entry.name}`}
                                                    >
                                                        ×
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                            </div>

                            <div className="rounded-lg border border-dashed border-gray-200 bg-gray-50 p-4">
                                <h4 className="text-sm font-semibold text-gray-900">Tools internas</h4>
                                <p className="mt-1 text-xs text-gray-500">Son herramientas globales de la app. Si las quieres bloquear, márcalas también aquí.</p>
                                <div className="mt-3 space-y-2">
                                    {BUILTIN_TOOLS.map((tool) => {
                                        const entry = hitlTools.find(t => t.name === tool.name);
                                        const isSelected = !!entry;
                                        return (
                                            <div key={tool.name} className={`rounded-md border px-4 py-3 ${isSelected ? 'border-indigo-300 bg-white' : 'border-gray-200 bg-white'}`}>
                                                <div className="flex items-center justify-between gap-4">
                                                    <label className="flex items-center gap-2 cursor-pointer min-w-0">
                                                        <input
                                                            type="checkbox"
                                                            checked={isSelected}
                                                            onChange={() => toggleTool(tool.name)}
                                                            disabled={isSubmitting}
                                                            className="h-4 w-4 rounded border-gray-300 text-indigo-600"
                                                        />
                                                        <span className="text-sm text-gray-900 truncate">{tool.label}</span>
                                                        <code className="text-xs text-gray-400 shrink-0">{tool.name}</code>
                                                    </label>
                                                    {isSelected && (
                                                        <div className="flex flex-wrap gap-3 shrink-0">
                                                            {ALL_HITL_DECISIONS.map((decision) => (
                                                                <label key={decision} className="flex items-center gap-1 cursor-pointer">
                                                                    <input
                                                                        type="checkbox"
                                                                        checked={entry.decisions.includes(decision)}
                                                                        onChange={() => toggleDecision(tool.name, decision)}
                                                                        disabled={isSubmitting}
                                                                        className="h-3.5 w-3.5 rounded border-gray-300 text-indigo-600"
                                                                    />
                                                                    <span className="text-xs capitalize text-gray-600">{decision}</span>
                                                                </label>
                                                            ))}
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        </div>
                    )}
                    <p className="mt-2 text-xs text-gray-500">
                        Cuando una tool seleccionada se vaya a ejecutar, el sistema se para antes y pide la decisión humana.
                        <strong> Approve</strong> la ejecuta, <strong>Edit</strong> permite cambiar sus argumentos y <strong>Reject</strong> la bloquea.
                    </p>
                </div>
            )}

            {/* Name Field */}
            <div>
                <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
                    Name <span className="text-red-500">*</span>
                </label>
                <input
                    type="text"
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
                    placeholder="e.g., Token Usage Monitor"
                    disabled={isSubmitting}
                    required
                />
            </div>

            {/* Description Field */}
            <div>
                <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-1">
                    Description
                </label>
                <textarea
                    id="description"
                    name="description"
                    rows={3}
                    value={formData.description}
                    onChange={handleChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
                    placeholder="Describe what this middleware does..."
                    disabled={isSubmitting}
                />
            </div>

            <FormActions
                isEditing={isEditing}
                isSubmitting={isSubmitting}
                onCancel={onCancel}
            />
        </form>
    );
}

export default MiddlewareForm;
