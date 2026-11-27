import { screen, waitFor } from '@testing-library/react';
import { Route, Routes } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';

import AppSettingsPage from '../../src/pages/settings/AppSettingsPage';
import { apiMock } from './apiMock';
import { renderWithProviders, tr } from './testUtils';

const OPTIONS = [
  { service_id: 1, name: 'Cohere', provider: 'Cohere', model_name: 'rerank-v3.5', is_system: false, is_platform_default: false, lifecycle_status: 'active' },
  { service_id: 2, name: 'Infinity', provider: 'Infinity', model_name: 'bge', is_system: true, is_platform_default: true, lifecycle_status: 'active' },
  { service_id: 3, name: 'Gone', provider: 'Jina', model_name: 'old', is_system: false, is_platform_default: false, lifecycle_status: 'retired' },
  { service_id: 4, name: 'Sunset', provider: 'Voyage', model_name: 'v', is_system: false, is_platform_default: false, lifecycle_status: 'deprecated', end_of_service_at: '2099-01-01T00:00:00Z' },
];

function setup(rerankId: number | null, { role = 'owner', platform = true, fail = false } = {}) {
  apiMock.getApp.mockResolvedValue({
    app_id: 7,
    name: 'Demo',
    user_role: role,
    agent_rate_limit: 0,
    max_file_size_mb: 0,
    default_sandbox_service_id: null,
    default_embedding_service_id: null,
    default_rerank_service_id: rerankId,
    langsmith_api_key: '****abcd',
  } as never);
  apiMock.getAppSlugInfo.mockResolvedValue({ slug: 'demo' } as never);
  apiMock.getSandboxServices.mockResolvedValue([] as never);
  apiMock.getEmbeddingServices.mockResolvedValue([] as never);
  if (fail) apiMock.getRerankServiceOptions.mockRejectedValue(new Error('down'));
  else {
    apiMock.getRerankServiceOptions.mockResolvedValue({
      options: OPTIONS,
      app_default: OPTIONS.find((o) => o.service_id === rerankId) ?? null,
      platform_default: platform ? OPTIONS[1] : null,
      effective_default: null,
    } as never);
  }
  apiMock.updateApp.mockResolvedValue({} as never);
  return renderWithProviders(
    <Routes>
      <Route path="/apps/:appId/settings" element={<AppSettingsPage />} />
    </Routes>,
    { path: '/apps/7/settings' },
  );
}

const selector = () => screen.getByRole('combobox', { name: tr('appSettingsPage.rerankService') });

describe('App settings: default rerank service', () => {
  it('lists own and system services, hides retired ones and flags deprecated ones', async () => {
    setup(null);
    await waitFor(() => expect(screen.getByRole('option', { name: /Cohere/ })).toBeInTheDocument());
    expect(apiMock.getRerankServiceOptions).toHaveBeenCalledWith(7);
    expect(screen.getByRole('option', { name: /\[System\] Infinity/ })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: /Gone/ })).not.toBeInTheDocument();
    expect(screen.getByRole('option', { name: /Sunset.*2099|Sunset.*ends/i })).toBeInTheDocument();
  });

  it('names the platform default in the empty option', async () => {
    setup(null);
    await waitFor(() => expect(selector()).toHaveValue(''));
    expect(await screen.findByRole('option', { name: tr('appSettingsPage.noDefaultRerankServiceWithPlatform', { name: 'Infinity' }) })).toBeInTheDocument();
  });

  it('says there is no platform default when none is set', async () => {
    setup(null, { platform: false });
    expect(await screen.findByRole('option', { name: tr('appSettingsPage.noDefaultRerankServiceNoPlatform') })).toBeInTheDocument();
  });

  it('keeps a retired saved service visible, flagged as retired', async () => {
    setup(3);
    await waitFor(() => expect(selector()).toHaveValue('3'));
    expect(screen.getByRole('option', { name: /Gone.*retired/i })).toBeInTheDocument();
  });

  it('shows the egress notice', async () => {
    setup(null);
    await waitFor(() => expect(selector()).toBeInTheDocument());
    expect(screen.getByText(tr('rerankEgressNotice.message'))).toBeInTheDocument();
  });

  it('saves the chosen id with the update', async () => {
    const { user } = setup(null);
    await waitFor(() => expect(screen.getByRole('option', { name: /Cohere/ })).toBeInTheDocument());
    await user.selectOptions(selector(), '1');
    await user.click(screen.getByRole('button', { name: tr('appSettingsPage.saveChanges') }));
    await waitFor(() => expect(apiMock.updateApp).toHaveBeenCalledTimes(1));
    expect(apiMock.updateApp.mock.calls[0][1]).toMatchObject({ default_rerank_service_id: 1 });
  });

  it('clears with an explicit null', async () => {
    const { user } = setup(1);
    await waitFor(() => expect(selector()).toHaveValue('1'));
    await user.selectOptions(selector(), '');
    await user.click(screen.getByRole('button', { name: tr('appSettingsPage.saveChanges') }));
    await waitFor(() => expect(apiMock.updateApp).toHaveBeenCalledTimes(1));
    expect(apiMock.updateApp.mock.calls[0][1]).toHaveProperty('default_rerank_service_id', null);
  });

  it('shows an inline warning when the options cannot be loaded', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    setup(null, { fail: true });
    expect(await screen.findByText(tr('appSettingsPage.rerankServicesLoadFailed'))).toBeInTheDocument();
  });

  it('is read-only for a member who cannot edit the App settings', async () => {
    setup(1, { role: 'editor' });
    await waitFor(() => expect(selector()).toHaveValue('1'));
    expect(selector()).toBeDisabled();
  });
});
