import { useState } from 'react';
import { screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import RagConfigSection, { type RagConfigValue } from '../../src/components/forms/RagConfigSection';
import type { RerankServiceOptions } from '../../src/types/services';
import { renderWithProviders, tr } from './testUtils';

const OPTIONS = [
  { service_id: 1, name: 'Cohere', provider: 'Cohere', model_name: 'rerank-v3.5', is_system: false, is_platform_default: false, lifecycle_status: 'active' },
  { service_id: 2, name: 'Infinity', provider: 'Infinity', model_name: 'bge', is_system: true, is_platform_default: true, lifecycle_status: 'active' },
  { service_id: 3, name: 'Gone', provider: 'Jina', model_name: 'old', is_system: false, is_platform_default: false, lifecycle_status: 'retired' },
] as const;

const WITH_DEFAULT = { options: OPTIONS, app_default: null, platform_default: OPTIONS[1], effective_default: OPTIONS[1] } as never;
const NONE = { options: [], app_default: null, platform_default: null, effective_default: null } as never;

const BASE: RagConfigValue = {
  rag_k: 10,
  rag_search_type: 'similarity',
  rag_score_threshold: null,
  rag_max_retrieval_calls: 4,
  rag_fixed_filters: [],
  rag_search_method: ['dense'],
  rag_rerank_enabled: false,
  rag_rerank_service_id: null,
  rag_rerank_top_n: null,
};

let latest: RagConfigValue = BASE;

function Harness({ initial, options, saved = null }: Readonly<{ initial?: Partial<RagConfigValue>; options: RerankServiceOptions | null; saved?: number | null }>) {
  const [value, setValue] = useState<RagConfigValue>({ ...BASE, ...initial });
  latest = value;
  return (
    <RagConfigSection
      value={value}
      onChange={(patch) => setValue((prev) => ({ ...prev, ...patch }))}
      metadataFields={[]}
      rerankOptions={options}
      savedRerankServiceId={saved}
    />
  );
}

const mode = () => screen.getByRole('combobox', { name: tr('rerankControls.label') });
const service = () => screen.getByRole('combobox', { name: tr('rerankControls.service') });

describe('RagConfigSection: rerank', () => {
  it('lists the inherited default first, then non-retired app and system services', () => {
    renderWithProviders(<Harness initial={{ rag_rerank_enabled: true }} options={WITH_DEFAULT} />);
    const labels = Array.from(service().querySelectorAll('option')).map((o) => o.textContent ?? '');
    expect(labels[0]).toBe(tr('rerankControls.defaultOption', { name: 'Infinity' }));
    expect(service()).toHaveValue('');
    expect(labels.some((l) => l.includes('Cohere'))).toBe(true);
    expect(labels.some((l) => l.includes('Infinity') && l.includes('System') && !l.startsWith('Default'))).toBe(true);
    expect(labels.some((l) => l.includes('Gone'))).toBe(false);
  });

  it('keeps a retired saved service visible but marked', () => {
    renderWithProviders(<Harness initial={{ rag_rerank_enabled: true, rag_rerank_service_id: 3 }} saved={3} options={WITH_DEFAULT} />);
    expect(service()).toHaveValue('3');
    expect(screen.getByRole('option', { name: /Gone.*retired/i })).toBeInTheDocument();
  });

  it('clamps Top N to 1..50 and defaults to 30', async () => {
    const { user } = renderWithProviders(<Harness initial={{ rag_rerank_enabled: true }} options={WITH_DEFAULT} />);
    const topN = screen.getByLabelText(tr('rerankControls.topN'));
    expect(topN).toHaveValue(30);
    await user.clear(topN);
    await user.type(topN, '99');
    expect(latest.rag_rerank_top_n).toBe(50);
    await user.clear(topN);
    await user.type(topN, '0');
    expect(latest.rag_rerank_top_n).toBe(1);
  });

  it('has no similarity threshold control and no strategy checkboxes', () => {
    renderWithProviders(<Harness initial={{ rag_rerank_enabled: true }} options={WITH_DEFAULT} />);
    expect(screen.queryByLabelText(/similarity threshold/i)).not.toBeInTheDocument();
    expect(screen.queryByRole('checkbox', { name: /cross-encoder/i })).not.toBeInTheDocument();
  });

  it('shows the egress notice only while reranking is on', async () => {
    const { user } = renderWithProviders(<Harness options={WITH_DEFAULT} />);
    expect(screen.queryByRole('note')).not.toBeInTheDocument();
    await user.selectOptions(mode(), 'service');
    expect(screen.getByRole('note')).toHaveTextContent(tr('rerankEgressNotice.message'));
    expect(latest.rag_rerank_enabled).toBe(true);
    expect(latest.rag_rerank_service_id).toBeNull();
    await user.selectOptions(mode(), 'none');
    expect(screen.queryByRole('note')).not.toBeInTheDocument();
  });

  it('disables the Rerank service option when nothing is configured', () => {
    renderWithProviders(<Harness options={NONE} />);
    expect(screen.getByRole('option', { name: tr('rerankControls.useService') })).toBeDisabled();
    expect(screen.getByText(tr('rerankControls.unavailable'))).toBeInTheDocument();
  });

  it('shows the same message as an error when a stored agent has reranking on but nothing resolves', () => {
    renderWithProviders(<Harness initial={{ rag_rerank_enabled: true }} options={NONE} />);
    expect(screen.getByText(tr('rerankControls.unavailable'))).toBeInTheDocument();
    expect(mode()).toHaveAttribute('aria-invalid', 'true');
  });
});
