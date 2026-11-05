import { Lightbulb } from 'lucide-react';
import type { LightRAGRole } from '../../utils/lightragModelSpecs';

// Latest recommended models per provider, per role. Reviewed 2026-09-14;
// extract Open-source pick re-reviewed 2026-10-30 after the full-corpus A/B
// on silo 54 — see docs/dependencies/lightrag.md#911-recomendación-de-modelos-por-rol.
// ponytail: update inline as providers ship.
// Open-source picks are backed by a real extraction/keyword benchmark on
// this repo's own LightRAG prompts — see
// docs/testing/lightrag_extraction_benchmark_corpus.md and
// docs/dependencies/lightrag.md#911-recomendación-de-modelos-por-rol.
// The cloud-provider rows are NOT benchmarked the same way — no measured
// quality data exists for them except the OpenAI ones (see their comments
// below). Don't swap any of these for a cheaper sibling on price alone
// (checked against this app's live PricingCatalog on 2026-09-14) without
// first checking whether it has any real quality data backing it — a price
// cut that also drops model generation/capability isn't a safe trade for
// `extract`, which needs reliable structured JSON output.
const RECS: Partial<Record<LightRAGRole, { provider: string; model: string }[]>> = {
  extract: [
    // GPT-6 Luna replaces GPT-5.4 mini (2026-11-05): side by side on the same
    // chunks with the production JSON schema it extracted more (45/42 vs 37/34
    // entities/relations, coherent, nothing truncated) at ~1/7 of the cost per
    // chunk; slower (~2x), which doesn't matter for indexing. Small sample — see
    // lightrag.md §9.1.1. Reasoning model: its reasoning tokens are billed as
    // output (~3,500 out/call measured, used by the indexing cost estimate).
    // The production open-source pick below is now Qwen3.6-35B-A3B: it
    // supersedes Qwen3-30B-A3B after the silo-54 corpus A/B (better recall
    // and 2.2x indexing throughput — see lightrag.md §9.1.1). A cheaper
    // untested sibling is a real quality risk, not just a price change.
    { provider: 'OpenAI', model: 'GPT-6 Luna' },
    { provider: 'Anthropic', model: 'Claude Haiku 4.5' },
    { provider: 'Mistral', model: 'Mistral Small 4' },
    { provider: 'Google', model: 'Gemini 3.1 Flash-Lite' },
    { provider: 'Open-source', model: 'Qwen3.6-35B-A3B' },
  ],
  keywords: [
    // GPT-6 Luna over GPT-5.4 nano (verbose/redundant keywords in the benchmark)
    // and GPT-5.4 mini (left low-level keywords empty on 1 of 3 test questions).
    { provider: 'OpenAI', model: 'GPT-6 Luna' },
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
