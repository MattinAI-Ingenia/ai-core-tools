import { useTranslation } from 'react-i18next';
import { Info } from 'lucide-react';

export interface LightRAGAdvancedSettingsData {
  lightrag_chunk_strategy?: string;
  lightrag_chunk_token_size?: number;
  lightrag_chunk_overlap_token_size?: number;
  lightrag_language?: string;
  lightrag_entity_extract_max_gleaning?: number;
  lightrag_max_source_ids_per_entity?: number;
  lightrag_max_source_ids_per_relation?: number;
  lightrag_entity_types?: string;
  lightrag_entity_types_mode?: 'infer' | 'manual';
}

interface LightRAGAdvancedSettingsProps {
  formData: LightRAGAdvancedSettingsData;
  onFieldChange: (field: keyof LightRAGAdvancedSettingsData, value: string | number | undefined) => void;
  disabled?: boolean;
  /** True once the resource exists — disables language/gleaning/source-ids caps and shows the immutability note. */
  locked?: boolean;
  /** Fields are still editable (resource doesn't exist yet) but won't be after creation — show the note without disabling. */
  forewarnImmutable?: boolean;
}

// ponytail: closed list instead of free text — LightRAG interpolates this
// literal into the prompt, a typo degrades silently with no error.
export const LIGHTRAG_LANGUAGES = ['English', 'Spanish', 'French', 'Italian'];

// LightRAG's own built-in default entity types (English) and this project's
// Spanish translation of them — shown as the placeholder for lightrag_entity_types
// depending on the sibling lightrag_language field. LightRAG does not localize
// this list itself. Keep the Spanish list in sync with the backend copy in
// backend/tools/vector_stores/lightrag_store.py (_SPANISH_DEFAULT_ENTITY_TYPES) —
// that is the list actually used when a Spanish silo is left blank.
const LIGHTRAG_DEFAULT_ENTITY_TYPES_EN = [
  'Person', 'Creature', 'Organization', 'Location', 'Event',
  'Concept', 'Method', 'Content', 'Data', 'Artifact', 'NaturalObject',
];
const LIGHTRAG_DEFAULT_ENTITY_TYPES_ES = [
  'Persona', 'Criatura', 'Organización', 'Lugar', 'Evento',
  'Concepto', 'Método', 'Contenido', 'Datos', 'Artefacto', 'ObjetoNatural',
];

function defaultEntityTypesPlaceholder(language: string | undefined): string {
  const list = language === 'Spanish' ? LIGHTRAG_DEFAULT_ENTITY_TYPES_ES : LIGHTRAG_DEFAULT_ENTITY_TYPES_EN;
  return list.join(', ');
}

/** Trim, drop empty items, and drop case-insensitive duplicates (keep first occurrence). */
export function sanitizeEntityTypesList(value: string | undefined): string | undefined {
  if (!value) return undefined;
  const seen = new Set<string>();
  const result: string[] = [];
  for (const item of value.split(',')) {
    const cleaned = item.trim();
    if (!cleaned) continue;
    const key = cleaned.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(cleaned);
  }
  return result.length > 0 ? result.join(', ') : undefined;
}

function ImmutableNote() {
  const { t } = useTranslation('common');
  return (
    <p className="mt-1 text-sm text-amber-600 flex items-center gap-1">
      <Info className="w-3.5 h-3.5 shrink-0" />
      {t('lightrag.advancedSettings.immutable')}
    </p>
  );
}

const inputClass = 'w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:border-transparent disabled:bg-gray-100';

export function LightRAGAdvancedSettings({ formData, onFieldChange, disabled, locked, forewarnImmutable }: Readonly<LightRAGAdvancedSettingsProps>) {
  const { t } = useTranslation('common');
  const toInt = (value: string) => (value ? Number.parseInt(value, 10) : undefined);
  const showNote = locked || forewarnImmutable;
  // Unset means 'infer' for a silo being created; a silo saved before this
  // field existed carries 'manual' from the column's server_default, so its
  // behaviour never changes underneath it.
  const mode = formData.lightrag_entity_types_mode ?? 'infer';

  return (
    <details className="border-t border-gray-200 pt-6">
      <summary className="text-sm font-semibold text-gray-800 uppercase tracking-wider cursor-pointer select-none">
        {t('lightrag.advancedSettings.title')}
      </summary>
      <div className="mt-4 space-y-6">

        {/* Chunking Strategy */}
        <div>
          <label htmlFor="lightrag_chunk_strategy" className="block text-sm font-medium text-gray-700 mb-2">
            {t('lightrag.advancedSettings.chunkStrategy')}
          </label>
          <select
            id="lightrag_chunk_strategy"
            value={formData.lightrag_chunk_strategy || 'fixed_token'}
            onChange={(e) => onFieldChange('lightrag_chunk_strategy', e.target.value)}
            className={inputClass}
            disabled={disabled}
          >
            <option value="fixed_token">{t('lightrag.advancedSettings.strategyFixedToken')}</option>
            <option value="recursive_character">{t('lightrag.advancedSettings.strategyRecursive')}</option>
            <option value="semantic_vector">{t('lightrag.advancedSettings.strategySemantic')}</option>
            {/* Needs a .blocks.jsonl sidecar that only LightRAG's own document
                parser produces (native: DOCX only today; PDF needs the mineru
                or docling external engine, neither wired up here) — our
                extraction feeds LightRAG plain text directly, so this sidecar
                never exists. Shown, not hidden, so it reads as "not yet" and
                not "doesn't exist"; a native <select> cannot host an icon per
                option, so the explanation lives in the title tooltip instead. */}
            <option value="paragraph_semantic" disabled title={t('lightrag.advancedSettings.strategyParagraphTooltip')}>
              {t('lightrag.advancedSettings.strategyParagraph')}
            </option>
          </select>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label htmlFor="lightrag_chunk_token_size" className="block text-sm font-medium text-gray-700 mb-2">
              {t('lightrag.advancedSettings.chunkTokenSize')}
            </label>
            <input
              type="number"
              id="lightrag_chunk_token_size"
              value={formData.lightrag_chunk_token_size ?? 1200}
              onChange={(e) => onFieldChange('lightrag_chunk_token_size', toInt(e.target.value))}
              min={100}
              max={8000}
              className={inputClass}
              disabled={disabled}
            />
          </div>
          <div>
            <label htmlFor="lightrag_chunk_overlap_token_size" className="block text-sm font-medium text-gray-700 mb-2">
              {t('lightrag.advancedSettings.overlapTokenSize')}
            </label>
            <input
              type="number"
              id="lightrag_chunk_overlap_token_size"
              value={formData.lightrag_chunk_overlap_token_size ?? 100}
              onChange={(e) => onFieldChange('lightrag_chunk_overlap_token_size', toInt(e.target.value))}
              min={0}
              max={2000}
              className={inputClass}
              disabled={disabled}
            />
          </div>
        </div>

        {/* Output language — shared by entity extraction (indexing) and
            query keyword extraction, LightRAG has no per-role override. */}
        <div>
          <label htmlFor="lightrag_language" className="block text-sm font-medium text-gray-700 mb-2">
            {t('lightrag.advancedSettings.language')}
          </label>
          <select
            id="lightrag_language"
            value={formData.lightrag_language || 'English'}
            onChange={(e) => onFieldChange('lightrag_language', e.target.value)}
            className={inputClass}
            disabled={disabled || locked}
          >
            {LIGHTRAG_LANGUAGES.map((lang) => (
              <option key={lang} value={lang}>{t(`lightrag.advancedSettings.languages.${lang}`)}</option>
            ))}
          </select>
          <p className="mt-1 text-sm text-gray-500">
            {t('lightrag.advancedSettings.languageHelp')}
          </p>
          {showNote && <ImmutableNote />}
        </div>

        <div>
          <label htmlFor="lightrag_entity_extract_max_gleaning" className="block text-sm font-medium text-gray-700 mb-2">
            {t('lightrag.advancedSettings.gleaning')}
          </label>
          <input
            type="number"
            id="lightrag_entity_extract_max_gleaning"
            value={formData.lightrag_entity_extract_max_gleaning ?? 0}
            onChange={(e) => onFieldChange('lightrag_entity_extract_max_gleaning', toInt(e.target.value))}
            min={0}
            max={5}
            className={inputClass}
            disabled={disabled || locked}
          />
          <p className="mt-1 text-sm text-gray-500">
            {t('lightrag.advancedSettings.gleaningHelp')}
          </p>
          {showNote && <ImmutableNote />}
        </div>

        <div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="lightrag_max_source_ids_per_entity" className="block text-sm font-medium text-gray-700 mb-2">
                {t('lightrag.advancedSettings.maxSourceEntity')}
              </label>
              <input
                type="number"
                id="lightrag_max_source_ids_per_entity"
                value={formData.lightrag_max_source_ids_per_entity ?? 1000}
                onChange={(e) => onFieldChange('lightrag_max_source_ids_per_entity', toInt(e.target.value))}
                min={1}
                className={inputClass}
                disabled={disabled || locked}
              />
            </div>
            <div>
              <label htmlFor="lightrag_max_source_ids_per_relation" className="block text-sm font-medium text-gray-700 mb-2">
                {t('lightrag.advancedSettings.maxSourceRelation')}
              </label>
              <input
                type="number"
                id="lightrag_max_source_ids_per_relation"
                value={formData.lightrag_max_source_ids_per_relation ?? 1000}
                onChange={(e) => onFieldChange('lightrag_max_source_ids_per_relation', toInt(e.target.value))}
                min={1}
                className={inputClass}
                disabled={disabled || locked}
              />
            </div>
          </div>
          <p className="mt-1 text-sm text-gray-500">
            {t('lightrag.advancedSettings.maxSourceHelp')}
          </p>
          {showNote && <ImmutableNote />}
        </div>

        <div>
          <span className="block text-sm font-medium text-gray-700 mb-2">{t('lightrag.advancedSettings.entityTypes')}</span>
          <div className="space-y-2 mb-3">
            <label className="flex items-start gap-2 text-sm text-gray-700">
              <input
                type="radio"
                name="lightrag_entity_types_mode"
                className="mt-1"
                checked={mode === 'infer'}
                onChange={() => onFieldChange('lightrag_entity_types_mode', 'infer')}
                disabled={disabled || locked}
              />
              <span>
                {t('lightrag.advancedSettings.modeInfer')}
                <span className="block text-gray-500">
                  {t('lightrag.advancedSettings.modeInferHelp')}
                </span>
              </span>
            </label>
            <label className="flex items-start gap-2 text-sm text-gray-700">
              <input
                type="radio"
                name="lightrag_entity_types_mode"
                className="mt-1"
                checked={mode === 'manual'}
                onChange={() => onFieldChange('lightrag_entity_types_mode', 'manual')}
                disabled={disabled || locked}
              />
              <span>
                {t('lightrag.advancedSettings.modeManual')}
                <span className="block text-gray-500">
                  {t('lightrag.advancedSettings.modeManualHelp')}
                </span>
              </span>
            </label>
          </div>
          {mode === 'manual' && (
            <>
              <textarea
                id="lightrag_entity_types"
                rows={3}
                value={formData.lightrag_entity_types || ''}
                onChange={(e) => onFieldChange('lightrag_entity_types', e.target.value)}
                placeholder={defaultEntityTypesPlaceholder(formData.lightrag_language)}
                className={`${inputClass} placeholder:text-gray-400`}
                disabled={disabled || locked}
              />
              {/* Native file inputs render their button label in the browser's
                  locale (e.g. "Elegir archivo" on a Spanish browser) with no HTML
                  override — hide it and drive the picker from an English label
                  instead, so it matches the rest of this English-only form. */}
              <input
                type="file"
                id="lightrag_entity_types_file"
                accept=".txt"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (!file) return;
                  const reader = new FileReader();
                  reader.onload = () => {
                    if (typeof reader.result === 'string') {
                      onFieldChange('lightrag_entity_types', reader.result);
                    }
                  };
                  reader.readAsText(file);
                  e.target.value = '';
                }}
                className="hidden"
                disabled={disabled || locked}
              />
              <label
                htmlFor="lightrag_entity_types_file"
                className={`mt-2 inline-block px-3 py-1.5 text-sm border border-gray-300 rounded-lg ${
                  disabled || locked ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:bg-gray-50'
                }`}
              >
                {t('lightrag.advancedSettings.chooseFile')}
              </label>
              <p className="mt-1 text-sm text-gray-500">
                {t('lightrag.advancedSettings.entityTypesHelp')}
              </p>
            </>
          )}
          {showNote && <ImmutableNote />}
        </div>
      </div>
    </details>
  );
}
