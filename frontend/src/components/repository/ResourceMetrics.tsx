import React, { useEffect, useState } from 'react';
import { Clock, Cpu, AlertCircle, Loader2, DollarSign } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { apiService } from '../../services/api';

interface IndexingMetric {
    metric_id: number;
    status: string;
    total_tokens: number;
    prompt_tokens: number;
    completion_tokens: number;
    embedding_tokens: number | null;
    tokens_source: string | null;
    llm_calls: number;
    duration_seconds: number | null;
    cost: number | null;
    currency: string | null;
    model_name: string | null;
    embedding_model_name: string | null;
    created_at: string | null;
}

/** >1h → "1h 02m 03s", >1min → "5m 23s", otherwise plain seconds with `decimals` precision. */
export function formatDuration(seconds: number, decimals: number): string {
    if (seconds > 3600) {
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        return `${hrs}h ${String(mins).padStart(2, '0')}m ${String(secs).padStart(2, '0')}s`;
    }
    if (seconds > 60) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}m ${String(secs).padStart(2, '0')}s`;
    }
    return `${seconds.toFixed(decimals)}s`;
}

/** A "successful" reindex that made zero LLM/embedding calls: LightRAG found
 * this exact content already indexed under a different internal doc id (its
 * own dedup rejecting the resubmission) and did no new work. Not a failure —
 * but showing a bare "0.0s" with nothing else looks like one. */
function isNoOpReindex(metric: IndexingMetric): boolean {
    return (
        metric.status === 'success' &&
        metric.llm_calls === 0 &&
        metric.total_tokens === 0 &&
        !(metric.embedding_tokens ?? 0)
    );
}

interface ResourceMetricsProps {
    appId: number;
    siloId: number;
    resourceId: number;
    resourceStatus?: string;
    /** Display variant: 'inline' shows a condensed row; 'panel' shows full detail */
    variant?: 'inline' | 'panel';
}

/**
 * Fetches and displays the latest indexing metric for a single resource.
 * Renders nothing when the silo is not LightRAG (caller responsibility) or
 * when no metric has been recorded yet (204).
 */
const ResourceMetrics: React.FC<ResourceMetricsProps> = ({
    appId,
    siloId,
    resourceId,
    resourceStatus,
    variant = 'inline',
}) => {
    const { t } = useTranslation('common');
    const [metric, setMetric] = useState<IndexingMetric | null>(null);
    const [loading, setLoading] = useState(true);
    const [noData, setNoData] = useState(false);

    useEffect(() => {
        let cancelled = false;

        // Metrics are only expected once indexing is completed.
        if (resourceStatus !== 'ready') {
            setMetric(null);
            setNoData(true);
            setLoading(false);
            return () => {
                cancelled = true;
            };
        }

        const fetchMetric = async () => {
            setLoading(true);
            setNoData(false);
            try {
                // Retry a few times because resource status may become ready
                // slightly before the metric row is persisted.
                let foundMetric: IndexingMetric | null = null;
                for (let attempt = 0; attempt < 4; attempt += 1) {
                    const response = await apiService.getResourceIndexingMetrics(appId, siloId, resourceId);
                    if (response != null) {
                        foundMetric = response as IndexingMetric;
                        break;
                    }
                    if (attempt < 3) {
                        await new Promise((resolve) => setTimeout(resolve, 1500));
                    }
                }

                if (!cancelled) {
                    if (foundMetric == null) {
                        setNoData(true);
                    } else {
                        setMetric(foundMetric);
                    }
                }
            } catch {
                if (!cancelled) setNoData(true);
            } finally {
                if (!cancelled) setLoading(false);
            }
        };

        fetchMetric();
        return () => {
            cancelled = true;
        };
    }, [appId, siloId, resourceId, resourceStatus]);

    if (loading) {
        return (
            <span className="inline-flex items-center gap-1 text-xs text-gray-400">
                <Loader2 className="w-3 h-3 animate-spin" />
            </span>
        );
    }

    if (noData || !metric) {
        return null; // No metric recorded yet — silent
    }

    if (variant === 'inline') {
        return (
            <span className="inline-flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 ml-2">
                {isNoOpReindex(metric) && (
                    <span title={t('lightrag.resourceMetrics.noOpTooltip')}>
                        {t('lightrag.resourceMetrics.noOpInline')}
                    </span>
                )}
                {metric.total_tokens > 0 && (
                    <span title={t(metric.tokens_source === 'estimated' ? 'lightrag.resourceMetrics.llmTokensTooltipEstimated' : 'lightrag.resourceMetrics.llmTokensTooltip', { input: metric.prompt_tokens, output: metric.completion_tokens })}>
                        <Cpu className="w-3 h-3 inline mr-0.5" />
                        {t('lightrag.resourceMetrics.llmShort', { value: metric.total_tokens.toLocaleString() })}
                    </span>
                )}
                {(metric.embedding_tokens ?? 0) > 0 && (
                    <span title={t('lightrag.resourceMetrics.embeddingTokensTooltip', { value: metric.embedding_tokens!.toLocaleString() })}>
                        {t('lightrag.resourceMetrics.embShort', { value: metric.embedding_tokens!.toLocaleString() })}
                    </span>
                )}
                {metric.duration_seconds != null && (
                    <span title={t('lightrag.resourceMetrics.durationTooltip')}>
                        <Clock className="w-3 h-3 inline mr-0.5" />
                        {formatDuration(metric.duration_seconds, 1)}
                    </span>
                )}
                {metric.cost != null && (
                    <span title={t('lightrag.resourceMetrics.costTooltip', { currency: metric.currency ?? 'USD' })}>
                        <DollarSign className="w-3 h-3 inline mr-0.5" />
                        {metric.cost < 0.001 ? '<$0.001' : `$${metric.cost.toFixed(4)}`}
                    </span>
                )}
                {metric.status === 'failed' && (
                    <span title={t('lightrag.resourceMetrics.failedTooltip')}>
                        <AlertCircle className="w-3 h-3 text-red-400" />
                    </span>
                )}
            </span>
        );
    }

    // Panel variant — full detail card
    return (
        <div className="text-xs bg-gray-50 dark:bg-gray-800 rounded p-3 space-y-1.5 border border-gray-200 dark:border-gray-700">
            <div className="font-semibold text-gray-600 dark:text-gray-300 mb-1">{t('lightrag.resourceMetrics.panelTitle')}</div>
            {isNoOpReindex(metric) && (
                <div className="text-gray-400 italic">
                    {t('lightrag.resourceMetrics.noOpPanel')}
                </div>
            )}
            <div className="flex justify-between">
                <span className="text-gray-500">{t('lightrag.resourceMetrics.status')}</span>
                <span className={metric.status === 'success' ? 'text-green-600' : 'text-red-500'}>
                    {metric.status}
                </span>
            </div>
            <div className="flex justify-between">
                <span className="text-gray-500">{t('lightrag.resourceMetrics.llmTokens')}</span>
                <span>
                    {metric.total_tokens.toLocaleString()}
                    {metric.tokens_source === 'estimated' && (
                        <span className="ml-1 text-gray-400">{t('lightrag.resourceMetrics.estShort')}</span>
                    )}
                </span>
            </div>
            <div className="flex justify-between">
                <span className="text-gray-500">{t('lightrag.resourceMetrics.promptCompletion')}</span>
                <span>
                    {metric.prompt_tokens.toLocaleString()} / {metric.completion_tokens.toLocaleString()}
                </span>
            </div>
            {(metric.embedding_tokens ?? 0) > 0 && (
                <div className="flex justify-between">
                    <span className="text-gray-500">{t('lightrag.resourceMetrics.embeddingTokens')}</span>
                    <span>{metric.embedding_tokens!.toLocaleString()} <span className="text-gray-400">{t('lightrag.resourceMetrics.estShort')}</span></span>
                </div>
            )}
            <div className="flex justify-between">
                <span className="text-gray-500">{t('lightrag.resourceMetrics.llmCalls')}</span>
                <span>{metric.llm_calls}</span>
            </div>
            {metric.duration_seconds != null && (
                <div className="flex justify-between">
                    <span className="text-gray-500">{t('lightrag.resourceMetrics.duration')}</span>
                    <span>{formatDuration(metric.duration_seconds, 2)}</span>
                </div>
            )}
            <div className="flex justify-between">
                <span className="text-gray-500">{t('lightrag.resourceMetrics.estimatedCost')}</span>
                <span>
                    {metric.cost != null
                        ? `${metric.cost < 0.0001 ? '<$0.0001' : `$${metric.cost.toFixed(6)}`} ${metric.currency ?? ''}`
                        : <span className="text-gray-400">{t('lightrag.resourceMetrics.pricingUnavailable')}</span>
                    }
                </span>
            </div>
            {metric.model_name && (
                <div className="flex justify-between">
                    <span className="text-gray-500">{t('lightrag.resourceMetrics.llmModel')}</span>
                    <span className="truncate max-w-[60%]" title={metric.model_name}>
                        {metric.model_name}
                    </span>
                </div>
            )}
            {metric.embedding_model_name && (
                <div className="flex justify-between">
                    <span className="text-gray-500">{t('lightrag.resourceMetrics.embeddingModel')}</span>
                    <span className="truncate max-w-[60%]" title={metric.embedding_model_name}>
                        {metric.embedding_model_name}
                    </span>
                </div>
            )}
        </div>
    );
};

export default ResourceMetrics;
