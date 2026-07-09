import { Info } from 'lucide-react';

export interface LightRAGAdvancedSettingsData {
  lightrag_chunk_strategy?: string;
  lightrag_chunk_token_size?: number;
  lightrag_chunk_overlap_token_size?: number;
  lightrag_language?: string;
  lightrag_entity_extract_max_gleaning?: number;
  lightrag_max_source_ids_per_entity?: number;
  lightrag_max_source_ids_per_relation?: number;
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
export const LIGHTRAG_LANGUAGES = ['English', 'Spanish'];

function ImmutableNote() {
  return (
    <p className="mt-1 text-sm text-amber-600 flex items-center gap-1">
      <Info className="w-3.5 h-3.5 shrink-0" />
      Cannot be changed after creation.
    </p>
  );
}

const inputClass = 'w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:border-transparent disabled:bg-gray-100';

export function LightRAGAdvancedSettings({ formData, onFieldChange, disabled, locked, forewarnImmutable }: Readonly<LightRAGAdvancedSettingsProps>) {
  const toInt = (value: string) => (value ? Number.parseInt(value, 10) : undefined);
  const showNote = locked || forewarnImmutable;

  return (
    <details className="border-t border-gray-200 pt-6">
      <summary className="text-sm font-semibold text-gray-800 uppercase tracking-wider cursor-pointer select-none">
        Advanced settings
      </summary>
      <div className="mt-4 space-y-6">

        {/* Chunking Strategy */}
        <div>
          <label htmlFor="lightrag_chunk_strategy" className="block text-sm font-medium text-gray-700 mb-2">
            Chunking strategy
          </label>
          <select
            id="lightrag_chunk_strategy"
            value={formData.lightrag_chunk_strategy || 'fixed_token'}
            onChange={(e) => onFieldChange('lightrag_chunk_strategy', e.target.value)}
            className={inputClass}
            disabled={disabled}
          >
            <option value="fixed_token">Fixed token (default)</option>
            <option value="recursive_character">Recursive character</option>
            <option value="semantic_vector">Semantic vector</option>
            <option value="paragraph_semantic">Paragraph semantic</option>
          </select>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label htmlFor="lightrag_chunk_token_size" className="block text-sm font-medium text-gray-700 mb-2">
              Chunk token size
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
              Overlap token size
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
            Language
          </label>
          <select
            id="lightrag_language"
            value={formData.lightrag_language || 'English'}
            onChange={(e) => onFieldChange('lightrag_language', e.target.value)}
            className={inputClass}
            disabled={disabled || locked}
          >
            {LIGHTRAG_LANGUAGES.map((lang) => (
              <option key={lang} value={lang}>{lang}</option>
            ))}
          </select>
          <p className="mt-1 text-sm text-gray-500">
            Output language for extracted entities and query keywords. If your
            corpus isn't English, set this — otherwise keyword extraction may
            translate query keywords away from your documents' language and
            miss matches.
          </p>
          {showNote && <ImmutableNote />}
        </div>

        <div>
          <label htmlFor="lightrag_entity_extract_max_gleaning" className="block text-sm font-medium text-gray-700 mb-2">
            Entity extraction gleaning iterations
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
            Extra re-processing passes the entity-extraction pipeline runs during
            indexing. 0 disables it (recommended default to start with).
          </p>
          {showNote && <ImmutableNote />}
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label htmlFor="lightrag_max_source_ids_per_entity" className="block text-sm font-medium text-gray-700 mb-2">
              Max source-ids per entity
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
              Max source-ids per relation
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
        <p className="text-sm text-gray-500">
          Max chunk-ids kept per entity/relation node in the graph (LightRAG default: 300).
        </p>
        {showNote && <ImmutableNote />}
      </div>
    </details>
  );
}
