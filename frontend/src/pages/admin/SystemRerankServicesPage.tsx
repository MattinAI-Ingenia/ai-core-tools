import React, { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Users, Pencil, Trash2, ListOrdered, Star, StarOff } from 'lucide-react';
import { apiService } from '../../services/api';
import ServiceWizard from '../../components/services/wizard/ServiceWizard';
import CompactServiceEditor from '../../components/services/CompactServiceEditor';
import type { RerankServiceDetail, RerankServiceFormData } from '../../types/services';
import { LoadingState } from '../../components/ui/LoadingState';
import { ErrorState } from '../../components/ui/ErrorState';
import ActionDropdown from '../../components/ui/ActionDropdown';
import ServiceStatusBadge from '../../components/ui/ServiceStatusBadge';
import ServiceUsageModal from '../../components/services/ServiceUsageModal';
import { getServiceApiClient } from '../../components/services/serviceApi';
import { useServiceConnectionTest } from '../../components/services/useServiceConnectionTest';
import { SERVICE_USAGE_STRINGS } from '../../utils/serviceLifecycle';
import { useSettingsCache } from '../../contexts/SettingsCacheContext';
import { useApiMutation } from '../../hooks/useApiMutation';
import { errorMessage } from '../../constants/messages';
import Title from '../../components/ui/Title';
import ButtonApp from '../../components/ui/ButtonApp';
import Table, { type TableColumn } from '../../components/ui/Table';
import BackgroundImage from '../../components/ui/BackgroundImage';
import SidePanel from '../../components/ui/SidePanel';
import { Badge } from '../../components/ui/Badge';
import RerankEgressNotice from '../../components/services/RerankEgressNotice';


const SystemRerankServicesPage: React.FC = () => {
  const { t } = useTranslation('common');
  const settingsCache = useSettingsCache();
  const mutate = useApiMutation();

  const [services, setServices] = useState<RerankServiceDetail[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [usageService, setUsageService] = useState<{ id: number; name: string; deleteRequested?: boolean } | null>(null);
  const [settingDefaultId, setSettingDefaultId] = useState<number | null>(null);
  const [editingService, setEditingService] = useState<RerankServiceDetail | null>(null);
  const { testAction, modal: connectionTestModal } = useServiceConnectionTest('rerank');

  const fetchServices = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await apiService.getSystemRerankServices();
      setServices(data);
    } catch (err) {
      setError(errorMessage(err, t('systemRerankServicesPage.loadFailed')));
    } finally {
      setIsLoading(false);
    }
  }, [t]);

  useEffect(() => {
    void fetchServices();
  }, [fetchServices]);

  const handleOpenCreate = () => {
    setEditingService(null);
    setShowForm(true);
  };

  const handleOpenEdit = (svc: RerankServiceDetail) => {
    setEditingService(svc);
    setShowForm(true);
  };

  const handleCancel = () => {
    setShowForm(false);
    setEditingService(null);
  };

  const handleSave = async (data: RerankServiceFormData) => {
    const result = await mutate(
      () =>
        editingService
          ? apiService.updateSystemRerankService(editingService.service_id, data)
          : apiService.createSystemRerankService(data),
      {
        loading: editingService
          ? t('systemRerankServicesPage.updating')
          : t('systemRerankServicesPage.creating'),
        success: editingService
          ? t('systemRerankServicesPage.updated')
          : t('systemRerankServicesPage.created'),
        error: (err) => errorMessage(
          err,
          editingService
            ? t('systemRerankServicesPage.updateFailed')
            : t('systemRerankServicesPage.createFailed'),
        ),
      },
    );
    if (result === undefined) return;

    settingsCache.invalidateSystemServiceCaches();
    handleCancel();
    await fetchServices();
  };

  const testConnection = (serviceId: number) => {
    const service = services.find((item) => item.service_id === serviceId);
    if (!service) throw new Error(t('systemRerankServicesPage.serviceNotFound'));
    return getServiceApiClient('rerank', 'system', undefined).testConnection({
      name: service.name,
      provider: service.provider ?? '',
      model_name: service.model_name,
      api_key: service.api_key,
      base_url: service.base_url,
    }, serviceId);
  };

  const runDelete = async (serviceId: number): Promise<boolean> => {
    const result = await mutate(
      () => apiService.deleteSystemRerankService(serviceId),
      {
        loading: t('systemRerankServicesPage.deleting'),
        success: t('systemRerankServicesPage.deleted'),
        error: (err) => errorMessage(err, t('systemRerankServicesPage.deleteFailed')),
      },
    );
    if (result === undefined) return false;
    settingsCache.invalidateSystemServiceCaches();
    await fetchServices();
    return true;
  };

  // Only one service is the platform default; marking one clears the previous mark (server side).
  const handleSetPlatformDefault = async (svc: RerankServiceDetail, makeDefault: boolean) => {
    setSettingDefaultId(svc.service_id);
    try {
      const result = await mutate(
        () => apiService.setPlatformDefaultRerankService(makeDefault ? svc.service_id : null),
        {
          loading: t('systemRerankServicesPage.settingDefault'),
          success: makeDefault
            ? t('systemRerankServicesPage.defaultSet', { name: svc.name })
            : t('systemRerankServicesPage.defaultCleared'),
          error: (err) => errorMessage(err, t('systemRerankServicesPage.defaultFailed')),
        },
      );
      if (result === undefined) return;
      settingsCache.invalidateSystemServiceCaches();
      setServices(result);
    } finally {
      setSettingDefaultId(null);
    }
  };

  const columns: TableColumn<RerankServiceDetail>[] = [
    {
      header: t('rerankServicesPage.columns.name'),
      className: 'px-6 py-4 whitespace-nowrap',
      render: (svc) => (
        <div className="flex items-center gap-2">
          <BackgroundImage icon={ListOrdered} variant="marron" size={34} iconSize={17} />
          <button
            type="button"
            onClick={() => handleOpenEdit(svc)}
            className="text-left text-sm font-medium text-fg dark:text-fg-dark"
          >
            {svc.name}
          </button>
          {svc.is_platform_default && (
            <Badge label={SERVICE_USAGE_STRINGS.platformDefaultBadge} variant="info" />
          )}
        </div>
      ),
    },
    {
      header: t('rerankServicesPage.columns.provider'),
      render: (svc) => svc.provider ?? t('rerankServicesPage.notAvailable'),
      className: 'px-6 py-4 whitespace-nowrap text-xs text-fg-secondary dark:text-fg-secondary-dark',
    },
    {
      header: t('rerankServicesPage.columns.model'),
      accessor: 'model_name',
      className: 'px-6 py-4 whitespace-nowrap text-xs text-fg-secondary dark:text-fg-secondary-dark',
    },
    {
      header: t('rerankServicesPage.columns.endpoint'),
      className: 'px-6 py-4 max-w-[16rem] truncate text-xs text-fg-secondary dark:text-fg-secondary-dark',
      render: (svc) => <span title={svc.base_url}>{svc.base_url}</span>,
    },
    {
      header: t('rerankServicesPage.columns.status'),
      className: 'px-6 py-4 text-sm',
      render: (svc) => (
        <ServiceStatusBadge
          lastError={svc.last_error}
          lastErrorAt={svc.last_error_at}
          lifecycleStatus={svc.lifecycle_status}
          endOfServiceAt={svc.end_of_service_at}
          deprecationNotice={svc.deprecation_notice}
        />
      ),
    },
    {
      header: t('rerankServicesPage.columns.actions'),
      sticky: 'right' as const,
      headerClassName: 'text-right',
      className: 'px-6 py-4 whitespace-nowrap text-right',
      render: (svc) => (
        <div className="inline-flex justify-end">
          <ActionDropdown
            size="sm"
            triggerAriaLabel={t('rerankServicesPage.actions.menuForService', { name: svc.name })}
            actions={[
              {
                label: SERVICE_USAGE_STRINGS.action,
                icon: <Users className="w-4 h-4" />,
                onClick: () => setUsageService({ id: svc.service_id, name: svc.name }),
              },
              testAction(svc.service_id, svc.name, () => testConnection(svc.service_id)),
              svc.is_platform_default
                ? {
                    label: t('systemRerankServicesPage.clearDefault'),
                    icon: <StarOff className="w-4 h-4" />,
                    disabled: settingDefaultId !== null,
                    onClick: () => void handleSetPlatformDefault(svc, false),
                  }
                : {
                    label: t('systemRerankServicesPage.setDefault'),
                    icon: <Star className="w-4 h-4" />,
                    disabled: settingDefaultId !== null,
                    onClick: () => void handleSetPlatformDefault(svc, true),
                  },
              {
                label: t('rerankServicesPage.actions.edit'),
                icon: <Pencil className="w-4 h-4" />,
                onClick: () => handleOpenEdit(svc),
              },
              {
                label: t('rerankServicesPage.actions.delete'),
                icon: <Trash2 className="w-4 h-4" />,
                variant: 'danger',
                onClick: () => setUsageService({ id: svc.service_id, name: svc.name, deleteRequested: true }),
              },
            ]}
          />
        </div>
      ),
    },
  ];

  if (isLoading) return (
    <div className="space-y-6">
            {/* Page Header */}
            <div className="flex items-center justify-between">
              <Title variant='titPrincipal' titulo={t('systemRerankServicesPage.title')} subtitulo={t('systemRerankServicesPage.subtitle')}></Title>
            </div>
    
            <LoadingState variant="page" message={t('systemRerankServicesPage.loading')} />
          </div>
  ) ;
  if (error) return <ErrorState error={error} onRetry={fetchServices} />;

  return (
    <div className="mx-auto space-y-6">
      <div className="flex items-start justify-between gap-4">
        <Title variant='titPrincipal' titulo={t('systemRerankServicesPage.title')} subtitulo={t('systemRerankServicesPage.subtitle')}></Title>
        <ButtonApp
              label={t('rerankServicesPage.addService')}
              onClick={handleOpenCreate}
              variant="primary"
              size="medium"
            />
      </div>

      <Table
        columns={columns}
        stickyEdgeColumns
        data={services}
        keyExtractor={(svc) => svc.service_id}
        emptyIcon={<BackgroundImage icon={ListOrdered} size={64} iconSize={32} />}
        emptyMessage={t('systemRerankServicesPage.empty')}
      />

      {services.length === 0 && (
        <div className="text-center py-6">
          <ButtonApp
            label={t('rerankServicesPage.addFirstService')}
            onClick={handleOpenCreate}
            variant="primary"
            size="medium"
          />
        </div>
      )}

      <RerankEgressNotice />

      {showForm && !editingService && (
        <ServiceWizard
          isOpen
          kind="rerank"
          scope="system"
          existingNames={services.map((s) => s.name)}
          onClose={handleCancel}
          onSave={handleSave}
        />
      )}

      {showForm && editingService && (
        <CompactServiceEditor
          isOpen
          variant="panel"
          kind="rerank"
          scope="system"
          service={{ ...editingService, provider: editingService.provider ?? '' }}
          existingNames={services.map((s) => s.name)}
          onClose={handleCancel}
          onSave={handleSave}
        />
      )}

      {/* Service usage side panel — fixed wrapper that starts below the 56px (h-14) app header */}
      {usageService && (
        <div className="fixed inset-x-0 bottom-0 top-14 z-30 !mt-0">
          <SidePanel
            open
            side="right"
            onClose={() => setUsageService(null)}
            title={t('systemRerankServicesPage.title')}
            headerActions={(
              <span className="rounded-full bg-surface-hover px-2 py-0.5 text-[11px] normal-case tracking-normal text-fg-secondary dark:bg-surface-hover-dark dark:text-fg-secondary-dark">
                {SERVICE_USAGE_STRINGS.action}
              </span>
            )}
            widthClassName="w-[640px] max-w-full"
          >
            <ServiceUsageModal
              key={usageService.id}
              variant="panel"
              kind="rerank"
              scope="system"
              serviceId={usageService.id}
              serviceName={usageService.name}
              onDelete={usageService.deleteRequested ? () => runDelete(usageService.id) : undefined}
              onClose={() => setUsageService(null)}
            />
          </SidePanel>
        </div>
      )}
      {connectionTestModal}
    </div>
  );
};

export default SystemRerankServicesPage;
