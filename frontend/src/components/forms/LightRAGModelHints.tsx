import { Lightbulb } from 'lucide-react';
import type { LightRAGRole } from '../../utils/lightragModelSpecs';

// Latest recommended models per provider, per role. Updated June 2026.
// ponytail: update inline as providers ship.
// Open-source picks are backed by a real extraction/keyword benchmark on
// this repo's own LightRAG prompts — see
// docs/testing/lightrag_extraction_benchmark_corpus.md and
// docs/dependencies/lightrag.md#911-recomendación-de-modelos-por-rol.
const RECS: Partial<Record<LightRAGRole, { provider: string; model: string }[]>> = {
  extract: [
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
