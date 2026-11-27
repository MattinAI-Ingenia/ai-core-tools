import { useState } from 'react';
import { screen } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';

import SearchControls, { DEFAULT_SEARCH_CONTROLS, type SearchControlsValue } from '../../src/components/playground/SearchControls';
import type { RerankServiceOptions } from '../../src/types/services';
import { renderWithProviders, tr } from './testUtils';

const OPTIONS = [
  { service_id: 1, name: 'Cohere', provider: 'Cohere', model_name: 'rerank-v3.5', is_system: false, is_platform_default: false, lifecycle_status: 'active' },
  { service_id: 2, name: 'Infinity', provider: 'Infinity', model_name: 'bge', is_system: true, is_platform_default: true, lifecycle_status: 'active' },
  { service_id: 3, name: 'Gone', provider: 'Jina', model_name: 'old', is_system: false, is_platform_default: false, lifecycle_status: 'retired' },
] as const;

const WITH_DEFAULT = { options: OPTIONS, app_default: null, platform_default: OPTIONS[1], effective_default: OPTIONS[1] } as never;
const NONE = { options: [], app_default: null, platform_default: null, effective_default: null } as never;

let latest: SearchControlsValue = DEFAULT_SEARCH_CONTROLS;

function Harness({ options, initial }: Readonly<{ options: RerankServiceOptions | null; initial?: Partial<SearchControlsValue> }>) {
  const [value, setValue] = useState<SearchControlsValue>({ ...DEFAULT_SEARCH_CONTROLS, ...initial });
  latest = value;
  return <SearchControls siloId={1} value={value} onChange={setValue} rerankOptions={options} />;
}

const mode = () => screen.getByRole('combobox', { name: tr('rerankControls.label') });

describe('SearchControls: rerank', () => {
  beforeEach(() => localStorage.clear());

  it('defaults to no rerank with Top N 30', () => {
    expect(DEFAULT_SEARCH_CONTROLS).toMatchObject({ rerank: false, rerankServiceId: null, rerankTopN: 30 });
  });

  it('offers None / Rerank service and the inherited default first', async () => {
    const { user } = renderWithProviders(<Harness options={WITH_DEFAULT} />);
    expect(screen.getByRole('option', { name: tr('rerankControls.none') })).toBeInTheDocument();
    await user.selectOptions(mode(), 'service');
    const select = screen.getByRole('combobox', { name: tr('rerankControls.service') });
    expect(select.querySelectorAll('option')[0]).toHaveTextContent(tr('rerankControls.defaultOption', { name: 'Infinity' }));
    expect(latest.rerank).toBe(true);
    expect(latest.rerankServiceId).toBeNull();
    await user.selectOptions(select, '1');
    expect(latest.rerankServiceId).toBe(1);
  });

  it('clamps Top N and shows the egress notice while on', async () => {
    const { user } = renderWithProviders(<Harness options={WITH_DEFAULT} initial={{ rerank: true }} />);
    expect(screen.getByRole('note')).toBeInTheDocument();
    const topN = screen.getByLabelText(tr('rerankControls.topN'));
    await user.clear(topN);
    await user.type(topN, '75');
    expect(latest.rerankTopN).toBe(50);
  });

  it('has no similarity threshold or strategy controls', () => {
    renderWithProviders(<Harness options={WITH_DEFAULT} initial={{ rerank: true }} />);
    expect(screen.queryByLabelText(/similarity threshold/i)).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: /cross-encoder/i })).not.toBeInTheDocument();
  });

  it('disables the Rerank service option when no reranker is configured', () => {
    renderWithProviders(<Harness options={NONE} />);
    expect(screen.getByRole('option', { name: tr('rerankControls.useService') })).toBeDisabled();
    expect(screen.getByText(tr('rerankControls.unavailable'))).toBeInTheDocument();
  });

  it('drops the keys saved by older versions', () => {
    localStorage.setItem('silo-search-controls-1', JSON.stringify({ strategy: 'rerank', topN: 9, limit: 7 }));
    renderWithProviders(<Harness options={WITH_DEFAULT} />);
    expect(latest.limit).toBe(7);
    expect(latest).not.toHaveProperty('strategy');
    expect(latest).not.toHaveProperty('topN');
  });
});
