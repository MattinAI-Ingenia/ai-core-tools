import { Lightbulb } from 'lucide-react';
import type { LightRAGRole } from '../../utils/lightragModelSpecs';

// Latest recommended models per provider, per role. Reviewed 2026-09-14.
// ponytail: update inline as providers ship.
// Open-source picks are backed by a real extraction/keyword benchmark on
// this repo's own LightRAG prompts — see
// docs/testing/lightrag_extraction_benchmark_corpus.md and
// docs/dependencies/lightrag.md#911-recomendación-de-modelos-por-rol.
// The cloud-provider rows are NOT benchmarked the same way — no measured
// quality data exists for them except gpt-5.4-mini (see its own comment
// below). Don't swap any of these for a cheaper sibling on price alone
// (checked against this app's live PricingCatalog on 2026-09-14) without
// first checking whether it has any real quality data backing it — a price
// cut that also drops model generation/capability isn't a safe trade for
// `extract`, which needs reliable structured JSON output.
const RECS: Partial<Record<LightRAGRole, { provider: string; model: string }[]>> = {
  extract: [
    // GPT-5.4 mini is kept over cheaper same-generation siblings (e.g.
    // GPT-5 mini) on purpose: it's the only cloud model here with REAL
    // measured extraction quality on this repo's own corpus (see
    // docs/testing/lightrag_extraction_benchmark_corpus.md — 48.1% non-hub
    // relationships, on par with the winning Qwen3-30B-A3B pick). A cheaper
    // untested sibling is a real quality risk, not just a price change.
    { provider: 'OpenAI', model: 'GPT-5.4 mini' },
    { provider: 'Anthropic', model: 'Claude Haiku 4.5' },
    { provider: 'Mistral', model: 'Mistral Small 4' },
    { provider: 'Google', model: 'Gemini 3.1 Flash-Lite' },
    { provider: 'Open-source', model: 'Qwen3-30B-A3B-Instruct' },
  ],
  keywords: [
    { provider: 'OpenAI', model: 'GPT-5.4 nano' },
    { provider: 'Anthropic', model: 'Claude Haiku 4.5' },
    { provider: 'Mistral', model: 'Ministral 3-3B' },
    { provider: 'Google', model: 'Gemini 3.1 Flash-Lite' },
    { provider: 'Open-source', model: 'Qwen3-4B-Instruct' },
  ],
};

export function LightRAGModelHints({ role }: Readonly<{ role: LightRAGRole }>) {
  const recs = RECS[role];
  if (!recs) return null;
  return (
    <details className="mt-1 text-sm">
      <summary className="cursor-pointer select-none flex items-center gap-1 text-gray-500 hover:text-gray-700">
        <Lightbulb className="w-3.5 h-3.5" /> Suggested models per provider
      </summary>
      <ul className="mt-2 space-y-1 bg-gray-50 border border-gray-200 rounded p-3">
        {recs.map(({ provider, model }) => (
          <li key={provider} className="text-gray-700">
            <span className="font-medium">{provider}:</span> {model}
          </li>
        ))}
      </ul>
    </details>
  );
}
