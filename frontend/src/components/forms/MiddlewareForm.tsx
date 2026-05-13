import { useState, useEffect } from 'react';
import FormActions from './FormActions';

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

interface MiddlewareFormProps {
    middleware?: MiddlewareItem | null;
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
];

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

function MiddlewareForm({ middleware, onSubmit, onCancel }: Readonly<MiddlewareFormProps>) {
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
        }
    }, [middleware]);

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
        setLimitValue(typeInfo.hasLimit ? (typeInfo.limitDefault ?? '') : '');
        setFormData(prev => ({
            ...prev,
            middleware_type: typeValue,
            name: typeInfo.label,
            description: typeInfo.description,
            config: newConfig
        }));
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

        setIsSubmitting(true);
        setError(null);

        try {
            await onSubmit(formData);
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
