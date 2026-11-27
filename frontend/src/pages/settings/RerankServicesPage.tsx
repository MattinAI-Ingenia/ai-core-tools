import { useParams } from 'react-router-dom';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Users, Upload, ArrowDownToLine, Pencil, Trash2, ListOrdered } from 'lucide-react';
import ServiceWizard from '../../components/services/wizard/ServiceWizard';
import CompactServiceEditor from '../../components/services/CompactServiceEditor';
import ActionDropdown from '../../components/ui/ActionDropdown';
import ImportModal from '../../components/ui/ImportModal';
import { useAppRole } from '../../hooks/useAppRole';
import ReadOnlyBanner from '../../components/ui/ReadOnlyBanner';
import Alert from '../../components/ui/Alert';
import Table from '../../components/ui/Table';
import { useServicesManager } from '../../hooks/useServicesManager';
import { Badge, ProviderBadge } from '../../components/ui/Badge';
import { AppRole } from '../../types/roles';
import type { RerankServiceFormData, RerankServiceListItem } from '../../types/services';
import ServiceStatusBadge from '../../components/ui/ServiceStatusBadge';
import UsedByButton from '../../components/services/UsedByButton';
import SystemServiceBadge from '../../components/services/SystemServiceBadge';
import ServiceUsageModal from '../../components/services/ServiceUsageModal';
import { SERVICE_USAGE_STRINGS } from '../../utils/serviceLifecycle';
import type { ConflictMode, ImportResponse } from '../../components/ui/ImportModal';
import { getServiceApiClient } from '../../components/services/serviceApi';
import { useServiceConnectionTest } from '../../components/services/useServiceConnectionTest';
import Title from '../../components/ui/Title';
import LoadingState from '../../components/ui/LoadingState';
import ButtonApp from '../../components/ui/ButtonApp';
import BackgroundImage from '../../components/ui/BackgroundImage';
import SidePanel from '../../components/ui/SidePanel';
import Spinner from '../../components/ui/Spinner';
import RerankEgressNotice from '../../components/services/RerankEgressNotice';


function RerankServicesPage() {
  const { appId } = useParams();
  const { t, i18n } = useTranslation('common');
  const { hasMinRole, userRole } = useAppRole(appId);
  const canEdit = hasMinRole(AppRole.ADMINISTRATOR);

  const api = {
    getAll: (id: number) => import('../../services/api').then(m => m.apiService.getRerankServices(id)),
    getOne: (id: number, sid: number) => import('../../services/api').then(m => m.apiService.getRerankService(id, sid)),
    create: (id: number, data: RerankServiceFormData) => import('../../services/api').then(m => m.apiService.createRerankService(id, data)),
    update: (id: number, sid: number, data: RerankServiceFormData) => import('../../services/api').then(m => m.apiService.updateRerankService(id, sid, data)),
    delete: (id: number, sid: number) => import('../../services/api').then(m => m.apiService.deleteRerankService(id, sid)),
  };

  // Rerank services are not part of the shared settings cache: the list is always fetched fresh.
  const cache = {
    get: () => null,
    set: () => undefined,
    invalidate: () => undefined,
  };

  const {
    services,
    loading,
    error,
    isModalOpen,
    editingService,
    handleClose,
    handleCreate,
    handleDelete,
    handleEdit,
    handleSave,
    setIsModalOpen,
    setEditingService,
    forceReload,
  } = useServicesManager<RerankServiceListItem>(appId, api, cache, { entity: t('rerankServicesPage.entity') });

  const [showImportModal, setShowImportModal] = useState(false);
  const [usageService, setUsageService] = useState<{ id: number; name: string; deleteRequested?: boolean } | null>(null);
  const [exportingServiceId, setExportingServiceId] = useState<number | null>(null);
  const [notification, setNotification] = useState<{message: string; type: 'success' | 'error'} | null>(null);
  const { testAction, modal: connectionTestModal } = useServiceConnectionTest('rerank');

  async function testConnection(serviceId: number) {
    if (!appId) throw new Error(t('rerankServicesPage.appIdNotFound'));
    const numericAppId = Number.parseInt(appId, 10);
    const apiService = (await import('../../services/api')).apiService;
    const detail = await apiService.getRerankService(numericAppId, serviceId);
    return getServiceApiClient('rerank', 'app', numericAppId).testConnection({
      name: detail.name,
      provider: detail.provider ?? '',
      model_name: detail.model_name,
      api_key: detail.api_key,
      base_url: detail.base_url,
    }, serviceId);
  }

  async function handleExport(serviceId: number) {
    if (!appId) return;
    setExportingServiceId(serviceId);
    
    try {
      const apiService = (await import('../../services/api')).apiService;
      const blob = await apiService.exportRerankService(Number.parseInt(appId), serviceId);
      
      // Find service name for filename
      const service = services.find(s => s.service_id === serviceId);
      const serviceName = service?.name || 'rerank-service';
      const sanitizedName = serviceName.replaceAll(/[^a-z0-9]/gi, '-').toLowerCase();
      const timestamp = new Date().toISOString().split('T')[0];
      const filename = `rerank-service-${sanitizedName}-${timestamp}.json`;
      
      // Create download link
      const url = globalThis.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      globalThis.URL.revokeObjectURL(url);
      a.remove();
    } catch (err) {
      alert(err instanceof Error ? err.message : t('rerankServicesPage.exportFailed'));
    } finally {
      setExportingServiceId(null);
    }
  }

  async function handleImport(
    file: File,
    conflictMode: ConflictMode,
    newName?: string
  ): Promise<ImportResponse> {
    if (!appId) throw new Error(t('rerankServicesPage.appIdNotFound'));
    
    try {
      const apiService = (await import('../../services/api')).apiService;
      const result = await apiService.importRerankService(
        Number.parseInt(appId),
        file,
        conflictMode,
        newName
      );
      
      if (result.success) {
        // Close modal immediately
        setShowImportModal(false);
        // Show success notification
        setNotification({
          message: result.message || t('rerankServicesPage.importSucceeded'),
          type: 'success'
        });
        // Refresh the list
        void forceReload();
        // Auto-dismiss notification after 5 seconds
        setTimeout(() => setNotification(null), 5000);
      }
      
      return result;
    } catch (err) {
      // Show error notification
      setNotification({
        message: err instanceof Error ? err.message : t('rerankServicesPage.importFailed'),
        type: 'error'
      });
      setTimeout(() => setNotification(null), 5000);
      // Re-throw the error so ImportModal can catch it
      throw err;
    }
  }

  if (loading) return (
    <div className="space-y-6">
        {/* Page Header */}
        <div className="flex items-center justify-between">
          <Title variant='titSecundario' titulo={t('rerankServicesPage.title')} subtitulo={t('rerankServicesPage.subtitle')} />
        </div>
        <LoadingState variant="page" message={t('rerankServicesPage.loading')} />
      </div>
  );

  if (error) return (
    <div className="p-6">
      <div className="flex items-center justify-between">
          <Title variant='titSecundario' titulo={t('rerankServicesPage.title')} subtitulo={t('rerankServicesPage.subtitle')} />
        </div>

      <Alert
        type="error"
        message={(
          <>
            {t('rerankServicesPage.loadError', { error })}
            <button type="button" onClick={() => { setIsModalOpen(false); setEditingService(null); }} className="mt-2 block underline">{t('rerankServicesPage.tryAgain')}</button>
          </>
        )}
      />
    </div>
  );

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <Title variant='titSecundario' titulo={t('rerankServicesPage.title')} subtitulo={t('rerankServicesPage.subtitle')} />
        {canEdit && (
          <div className="flex gap-2">
            <ButtonApp
              label={t('rerankServicesPage.import')}
              onClick={() => setShowImportModal(true)}
              icon={<Upload className="w-4 h-4" aria-hidden="true" />}
              variant="secondary"
              size="medium"
            />
            <ButtonApp
              label={t('rerankServicesPage.addService')}
              onClick={handleCreate}
              icon={<span aria-hidden="true">+</span>}
              variant="primary"
              size="medium"
            />
          </div>
        )}
      </div>

      {!canEdit && <ReadOnlyBanner userRole={userRole} minRole={AppRole.ADMINISTRATOR} />}

      {notification && (
        <Alert
          type={notification.type}
          message={notification.message}
          onDismiss={() => setNotification(null)}
          className="mb-4"
        />
      )}

      <Table
        data={services}
        stickyEdgeColumns
        keyExtractor={(service) => service.service_id.toString()}
        columns={[
          { header: t('rerankServicesPage.columns.name'), render: (service: RerankServiceListItem) => (
            <div className="flex items-center gap-1.5">
              <BackgroundImage icon={ListOrdered} variant='marron' size={34} iconSize={17} />
              {canEdit && !service.is_system ? (
                <button type="button" className="text-sm font-medium text-fg dark:text-fg-dark text-left" onClick={() => handleEdit(service.service_id)}>{service.name}</button>
              ) : (
                <span className="text-sm font-medium text-fg dark:text-fg-dark">{service.name}</span>
              )}
              {service.is_system && (
                <SystemServiceBadge />
              )}
              {service.is_platform_default && (
                <Badge label={SERVICE_USAGE_STRINGS.platformDefaultBadge} variant="info" />
              )}
            </div>
          ) },
          { header: t('rerankServicesPage.columns.model'), accessor: 'model_name', className: 'text-xs font-medium text-fg-secondary dark:text-fg-secondary-dark' },
          { header: t('rerankServicesPage.columns.provider'), render: (service: RerankServiceListItem) => (
            service.provider ? <ProviderBadge provider={service.provider} /> : <span className="text-xs text-fg-secondary dark:text-fg-secondary-dark">{t('rerankServicesPage.notAvailable')}</span>
          ) },
          { header: t('rerankServicesPage.columns.endpoint'), className: 'max-w-[16rem] truncate text-xs font-medium text-fg-secondary dark:text-fg-secondary-dark', render: (service: RerankServiceListItem) => (
            <span title={service.base_url}>{service.base_url}</span>
          ) },
          { header: t('rerankServicesPage.columns.status'), render: (service: RerankServiceListItem) => (
            <ServiceStatusBadge needsApiKey={service.needs_api_key} lastError={service.last_error} lastErrorAt={service.last_error_at} lifecycleStatus={service.lifecycle_status} endOfServiceAt={service.end_of_service_at} deprecationNotice={service.deprecation_notice} />
          ) },
          { header: t('rerankServicesPage.columns.created'), className: 'text-xs font-medium text-fg-secondary dark:text-fg-secondary-dark', render: (service: RerankServiceListItem) => (service.created_at ? new Date(service.created_at).toLocaleDateString(i18n.resolvedLanguage) : t('rerankServicesPage.notAvailable')) },
          { header: t('rerankServicesPage.columns.actions'), sticky: 'right', className: 'relative', render: (service: RerankServiceListItem) => {
            if (service.is_system) {
              return (
                <span className="flex flex-col items-center gap-1">
                  <span className="text-fg-secondary dark:text-fg-secondary-dark text-xs">{t('rerankServicesPage.system')}</span>
                  <UsedByButton serviceName={service.name} onClick={() => setUsageService({ id: service.service_id, name: service.name })} />
                </span>
              );
            }

            if (!canEdit) {
              return <UsedByButton serviceName={service.name} onClick={() => setUsageService({ id: service.service_id, name: service.name })} />;
            }

            return (
              <ActionDropdown triggerAriaLabel={t('rerankServicesPage.actions.menuForService', { name: service.name })} actions={[
                { label: SERVICE_USAGE_STRINGS.action, onClick: () => setUsageService({ id: service.service_id, name: service.name }), icon: <Users className="w-4 h-4" /> },
                testAction(service.service_id, service.name, () => testConnection(service.service_id)),
                {
                  label: exportingServiceId === service.service_id ? t('rerankServicesPage.actions.exporting') : t('rerankServicesPage.actions.export'),
                  onClick: () => void handleExport(service.service_id),
                  icon: exportingServiceId === service.service_id ? <Spinner size="sm" tone="current" /> : <ArrowDownToLine className="w-4 h-4" />,
                  variant: 'primary' as const,
                  disabled: exportingServiceId === service.service_id
                },
                { label: t('rerankServicesPage.actions.edit'), onClick: () => void handleEdit(service.service_id), icon: <Pencil className="w-4 h-4" />, variant: 'primary' },
                { label: t('rerankServicesPage.actions.delete'), onClick: () => setUsageService({ id: service.service_id, name: service.name, deleteRequested: true }), icon: <Trash2 className="w-4 h-4" />, variant: 'danger' }
              ]} size="sm" />
            );
          } }
        ]}
        emptyIcon={<BackgroundImage icon={ListOrdered} size={64} iconSize={32} />}
        emptyMessage={t('rerankServicesPage.empty.title')}
        emptySubMessage={t('rerankServicesPage.empty.subtitle')}
        loading={loading}
      />

      {!loading && services.length === 0 && canEdit && (
        <div className="text-center py-6">
          <ButtonApp
            label={t('rerankServicesPage.addFirstService')}
            onClick={handleCreate}
            icon={<span aria-hidden="true">+</span>}
            variant="primary"
            size="medium"
          />
        </div>
      )}

      <Alert type="info" title={t('rerankServicesPage.about.title')} message={t('rerankServicesPage.about.message')} className="mt-6" />
      <RerankEgressNotice className="mt-4" />

      {isModalOpen && !editingService && (
        <ServiceWizard
          isOpen
          kind="rerank"
          scope="app"
          appId={appId ? Number.parseInt(appId) : undefined}
          existingNames={services.map((s) => s.name)}
          onClose={handleClose}
          onSave={handleSave}
        />
      )}

      {isModalOpen && editingService && (
        <CompactServiceEditor
          isOpen
          variant="panel"
          kind="rerank"
          scope="app"
          appId={appId ? Number.parseInt(appId) : undefined}
          service={editingService}
          existingNames={services.map((s) => s.name)}
          onClose={handleClose}
          onSave={handleSave}
        />
      )}

      {/* Service usage side panel — fixed wrapper that starts below the 56px (h-14) app header */}
      {usageService && appId && (
        <div className="fixed inset-x-0 bottom-0 top-14 z-30 !mt-0">
          <SidePanel
            open
            side="right"
            onClose={() => setUsageService(null)}
            title={t('rerankServicesPage.title')}
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
              scope="app"
              canNavigate={hasMinRole(AppRole.EDITOR)}
              appId={Number.parseInt(appId, 10)}
              serviceId={usageService.id}
              serviceName={usageService.name}
              onDelete={usageService.deleteRequested ? () => handleDelete(usageService.id) : undefined}
              onClose={() => setUsageService(null)}
            />
          </SidePanel>
        </div>
      )}

      {connectionTestModal}

      {/* Import Modal */}
      <ImportModal
        isOpen={showImportModal}
        onClose={() => setShowImportModal(false)}
        onImport={handleImport}
        componentType="rerank_service"
            componentLabel={t('rerankServicesPage.componentLabel')}
      />
    </div>
  );

}

export default RerankServicesPage;