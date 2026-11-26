import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import Modal from '../ui/Modal';
import { StepperContainer, type StepDefinition } from '../ui/Stepper';
import { apiService } from '../../services/api';
import type { ImportJob } from '../../types/csvImport';

interface CsvImportStepperProps {
  appId: number;
  repositoryId: number;
  isOpen: boolean;
  onClose: () => void;
  onImportStarted: (job: ImportJob) => void;
}

export function CsvImportStepper({ appId, repositoryId, isOpen, onClose, onImportStarted }: Readonly<CsvImportStepperProps>) {
  const { t } = useTranslation('common');
  const steps: StepDefinition[] = [
    { id: 'map', label: t('lightrag.csvImportStepper.stepMap') },
    { id: 'starting', label: t('lightrag.csvImportStepper.stepStarting') },
  ];
  const [step, setStep] = useState(0);
  const [file, setFile] = useState<File | null>(null);
  const [headers, setHeaders] = useState<string[]>([]);
  const [linkColumn, setLinkColumn] = useState<string>('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reset = () => {
    setStep(0);
    setFile(null);
    setHeaders([]);
    setLinkColumn('');
    setError(null);
  };

  const handleFileSelected = async (selected: File) => {
    setFile(selected);
    setError(null);
    try {
      const { headers: parsedHeaders } = await apiService.previewCsvImport(appId, repositoryId, selected);
      setHeaders(parsedHeaders);
      setLinkColumn(parsedHeaders[0] ?? '');
    } catch {
      setError(t('lightrag.csvImportStepper.readError'));
    }
  };

  const handleStart = async () => {
    if (!file || !linkColumn) return;
    setIsSubmitting(true);
    setStep(1);
    try {
      const job = await apiService.createCsvImport(appId, repositoryId, file, linkColumn);
      onImportStarted(job);
      reset();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : t('lightrag.csvImportStepper.startError'));
      setStep(0);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={() => { reset(); onClose(); }} title={t('lightrag.csvImportStepper.title')} size="medium">
      <StepperContainer
        steps={steps}
        currentStep={step}
        onNext={handleStart}
        onBack={() => setStep(0)}
        onCancel={() => { reset(); onClose(); }}
        nextLabel={t('lightrag.csvImportStepper.startImport')}
        nextDisabled={!file || !linkColumn || isSubmitting}
        showBack={false}
        isSubmitting={isSubmitting}
      >
        {step === 0 && (
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <input
                type="file"
                id="csv_import_file"
                accept=".csv"
                onChange={(e) => e.target.files?.[0] && handleFileSelected(e.target.files[0])}
                className="hidden"
              />
              <label
                htmlFor="csv_import_file"
                className="inline-block cursor-pointer px-3 py-1.5 text-sm border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                {t('lightrag.csvImportStepper.chooseFile')}
              </label>
              <span className="text-sm text-gray-500 truncate">
                {file ? file.name : t('lightrag.csvImportStepper.noFile')}
              </span>
            </div>
            {headers.length > 0 && (
              <div>
                <label className="block text-sm font-medium mb-1">{t('lightrag.csvImportStepper.linkColumn')}</label>
                <select
                  value={linkColumn}
                  onChange={(e) => setLinkColumn(e.target.value)}
                  className="border rounded px-2 py-1 w-full"
                >
                  {headers.map((h) => (
                    <option key={h} value={h}>{h}</option>
                  ))}
                </select>
                <p className="text-xs text-gray-500 mt-2">
                  {t('lightrag.csvImportStepper.metadataHint')}
                </p>
              </div>
            )}
            {error && <p className="text-sm text-red-600">{error}</p>}
          </div>
        )}
        {step === 1 && <p>{t('lightrag.csvImportStepper.starting')}</p>}
      </StepperContainer>
    </Modal>
  );
}
