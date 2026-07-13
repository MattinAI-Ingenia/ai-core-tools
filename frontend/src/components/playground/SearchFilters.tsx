import { useEffect, useMemo, useState } from 'react';
import { Search } from 'lucide-react';

export type MetadataOperator = '$eq' | '$ne' | '$gt' | '$gte' | '$lt' | '$lte' | '$in';
export type SupportedDbType = 'PGVECTOR' | 'QDRANT';

export interface SearchFilterMetadataField {
  name: string;
  type: string;
  description?: string;
}

interface PreparedFilter {
  fieldName: string;
  operator: MetadataOperator;
  nativeOperator: string;
  value: unknown;
}

interface SearchFiltersProps {
  metadataFields?: SearchFilterMetadataField[];
  dbType?: string;
  disabled?: boolean;
  onFilterMetadataChange: (filterMetadata: Record<string, unknown> | undefined) => void;
}

const DEFAULT_DB_TYPE: SupportedDbType = 'PGVECTOR';
const QDRANT_METADATA_PREFIX = 'metadata.';

const FILTER_OPERATOR_MAPPINGS: Record<SupportedDbType, Record<MetadataOperator, string>> = {
  PGVECTOR: {
    $eq: '$eq',
    $ne: '$ne',
    $gt: '$gt',
    $gte: '$gte',
    $lt: '$lt',
    $lte: '$lte',
    $in: '$in',
  },
  QDRANT: {
    $eq: 'match',
    $ne: 'must_not_match',
    $gt: 'gt',
    $gte: 'gte',
    $lt: 'lt',
    $lte: 'lte',
    $in: 'match_any',
  },
};

const OPERATOR_LABELS: Record<MetadataOperator, string> = {
  $eq: 'equals',
  $ne: 'not equals',
  $gt: 'greater than',
  $gte: 'greater than or equal',
  $lt: 'less than',
  $lte: 'less than or equal',
  $in: 'in (any of)',
};

function normalizeDbType(dbType?: string): SupportedDbType {
  if (dbType?.toUpperCase() === 'QDRANT') {
    return 'QDRANT';
  }
  return DEFAULT_DB_TYPE;
}

function isStringType(fieldType: string): boolean {
  return ['string', 'str', 'keyword', 'text'].includes(fieldType.toLowerCase());
}

function isNumericType(fieldType: string): boolean {
  return ['int', 'float', 'number'].includes(fieldType.toLowerCase());
}

function isBoolType(fieldType: string): boolean {
  return fieldType.toLowerCase() === 'bool';
}

function getOperatorsForType(fieldType: string): MetadataOperator[] {
  if (isNumericType(fieldType)) {
    return ['$eq', '$ne', '$gt', '$gte', '$lt', '$lte'];
  }
  if (isBoolType(fieldType)) {
    return ['$eq', '$ne'];
  }
  // string, str, keyword, text, or unknown
  return ['$eq', '$ne', '$in'];
}

function buildPgvectorFilter(filters: PreparedFilter[], logicalOperator: '$and' | '$or') {
  if (filters.length === 0) {
    return undefined;
  }

  if (filters.length === 1) {
    const { fieldName, nativeOperator, value } = filters[0];
    return { [fieldName]: { [nativeOperator]: value } };
  }

  const conditions = filters.map(({ fieldName, nativeOperator, value }) => ({
    [fieldName]: { [nativeOperator]: value },
  }));

  return { [logicalOperator]: conditions };
}

function buildQdrantFilter(filters: PreparedFilter[], logicalOperator: '$and' | '$or') {
  if (filters.length === 0) {
    return undefined;
  }

  const must: Array<Record<string, unknown>> = [];
  const should: Array<Record<string, unknown>> = [];
  const mustNot: Array<Record<string, unknown>> = [];
  const targetList = logicalOperator === '$and' ? must : should;

  filters.forEach(({ fieldName, nativeOperator, value }) => {
    const key = `${QDRANT_METADATA_PREFIX}${fieldName}`;

    switch (nativeOperator) {
      case 'must_not_match':
        mustNot.push({ key, match: { value } });
        break;
      case 'match':
        targetList.push({ key, match: { value } });
        break;
      case 'match_any':
        targetList.push({ key, match: { any: value } });
        break;
      case 'gt':
      case 'gte':
      case 'lt':
      case 'lte':
        targetList.push({ key, range: { [nativeOperator]: value } });
        break;
      default:
        targetList.push({ key, match: { value } });
    }
  });

  const qdrantFilter: Record<string, unknown> = {};
  if (must.length > 0) qdrantFilter.must = must;
  if (should.length > 0) qdrantFilter.should = should;
  if (mustNot.length > 0) qdrantFilter.must_not = mustNot;

  return Object.keys(qdrantFilter).length > 0 ? qdrantFilter : undefined;
}

function convertMetadataValue(fieldType: string, rawValue: string): unknown {
  switch (fieldType) {
    case 'int': {
      const parsed = Number.parseInt(rawValue, 10);
      return Number.isNaN(parsed) ? rawValue : parsed;
    }
    case 'float': {
      const parsed = Number.parseFloat(rawValue);
      return Number.isNaN(parsed) ? rawValue : parsed;
    }
    case 'bool':
      return rawValue.trim().toLowerCase() === 'true';
    default:
      return rawValue;
  }
}

function prepareFilters(
  metadataFilters: Record<string, string>,
  filterOperators: Record<string, MetadataOperator>,
  metadataFields: SearchFilterMetadataField[] | undefined,
  operatorMapping: Record<MetadataOperator, string>,
): PreparedFilter[] {
  const prepared: PreparedFilter[] = [];

  Object.entries(metadataFilters).forEach(([fieldName, rawValue]) => {
    const trimmedValue = rawValue.trim();
    if (!trimmedValue) return;

    const selectedOperator = filterOperators[fieldName] || '$eq';
    const nativeOperator = operatorMapping[selectedOperator];
    if (!nativeOperator) return;

    const fieldDefinition = metadataFields?.find((field) => field.name === fieldName);
    let convertedValue: unknown;

    if (selectedOperator === '$in') {
      convertedValue = trimmedValue.split(',').map((v) => v.trim()).filter(Boolean);
    } else {
      convertedValue = fieldDefinition
        ? convertMetadataValue(fieldDefinition.type, trimmedValue)
        : trimmedValue;
    }

    prepared.push({ fieldName, operator: selectedOperator, nativeOperator, value: convertedValue });
  });

  return prepared;
}

function loadSavedFilters(key: string): SavedFilter[] {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as SavedFilter[]) : [];
  } catch {
    return [];
  }
}

export function SearchFilters({
  metadataFields,
  dbType,
  disabled = false,
  onFilterMetadataChange,
}: Readonly<SearchFiltersProps>) {
  const [metadataFilters, setMetadataFilters] = useState<Record<string, string>>({});
  const [filterOperators, setFilterOperators] = useState<Record<string, MetadataOperator>>({});
  const [logicalOperator, setLogicalOperator] = useState<'$and' | '$or'>('$and');

  // Autocomplete
  const [suggestions, setSuggestions] = useState<Record<string, string[]>>({});
  const [openSuggestionField, setOpenSuggestionField] = useState<string | null>(null);
  const debounceTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({});

  // Saved filters
  const storageKey = useMemo(
    () => `silo-saved-filters-${siloStorageKey ?? 'global'}`,
    [siloStorageKey],
  );
  const [savedFilters, setSavedFilters] = useState<SavedFilter[]>(() => loadSavedFilters(storageKey));
  const [showSaveForm, setShowSaveForm] = useState(false);
  const [saveFilterName, setSaveFilterName] = useState('');
  const [loadSelectValue, setLoadSelectValue] = useState('');
  const [loadedFilterName, setLoadedFilterName] = useState<string | null>(null);

  // Reload saved filters when storage key changes (different silo)
  useEffect(() => {
    setSavedFilters(loadSavedFilters(storageKey));
    setLoadedFilterName(null);
    setLoadSelectValue('');
  }, [storageKey]);

  useEffect(() => {
    setMetadataFilters({});
    setFilterOperators({});
    setLogicalOperator('$and');
    setLoadedFilterName(null);
  }, [metadataFields]);

  useEffect(() => {
    if (!metadataFields || metadataFields.length === 0) {
      onFilterMetadataChange(undefined);
    }
  }, [metadataFields, onFilterMetadataChange]);

  const normalizedDbType = useMemo(() => normalizeDbType(dbType), [dbType]);
  const operatorMapping = FILTER_OPERATOR_MAPPINGS[normalizedDbType];

  useEffect(() => {
    const prepared = prepareFilters(metadataFilters, filterOperators, metadataFields, operatorMapping);
    const filterMetadata = normalizedDbType === 'QDRANT'
      ? buildQdrantFilter(prepared, logicalOperator)
      : buildPgvectorFilter(prepared, logicalOperator);

    onFilterMetadataChange(filterMetadata);
  }, [metadataFilters, filterOperators, logicalOperator, metadataFields, normalizedDbType, operatorMapping, onFilterMetadataChange]);

  const handleMetadataFilterChange = (fieldName: string, value: string, operator: MetadataOperator) => {
    setMetadataFilters((prev) => ({
      ...prev,
      [fieldName]: value,
    }));
    setFilterOperators((prev) => ({
      ...prev,
      [fieldName]: operator,
    }));
  };

  if (!metadataFields || metadataFields.length === 0) {
    return null;
  }

  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-gray-50">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-700">
          <Search className="w-4 h-4 mr-2 inline-block" aria-hidden="true" />
          Filter by Metadata
        </h3>
        <div className="flex items-center gap-2">
          <label htmlFor="logicalOperator" className="text-sm text-gray-600">
            Match:
          </label>
          <select
            id="logicalOperator"
            value={logicalOperator}
            onChange={(e) => setLogicalOperator(e.target.value as '$and' | '$or')}
            className="px-3 py-1 border border-gray-300 rounded-lg text-sm font-medium bg-white"
            disabled={disabled}
          >
            <option value="$and">ALL filters (AND)</option>
            <option value="$or">ANY filter (OR)</option>
          </select>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {metadataFields.map((field) => {
          const operator = filterOperators[field.name] || '$eq';
          return (
            <div key={field.name}>
              <label htmlFor={`filter_${field.name}`} className="block text-sm font-medium text-gray-700 mb-1">
                {field.name}
                <span className="text-xs text-gray-500 ml-1">({field.type})</span>
              </label>
              <div className="flex items-center gap-2">
                <select
                  value={operator}
                  onChange={(e) => handleMetadataFilterChange(field.name, metadataFilters[field.name] || '', e.target.value as MetadataOperator)}
                  className="px-2 py-1 border border-gray-300 rounded-lg text-sm"
                  disabled={disabled}
                >
                  <option value="$eq">equals</option>
                  <option value="$ne">not equals</option>
                  <option value="$gt">greater than</option>
                  <option value="$gte">greater than or equal</option>
                  <option value="$lt">less than</option>
                  <option value="$lte">less than or equal</option>
                </select>
                <input
                  type="text"
                  id={`filter_${field.name}`}
                  value={metadataFilters[field.name] || ''}
                  onChange={(e) => handleMetadataFilterChange(field.name, e.target.value, operator)}
                  placeholder={`Filter by ${field.name}`}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:border-transparent text-sm"
                  disabled={disabled}
                />
              </div>
              {field.description && (
                <p className="text-xs text-gray-500 mt-1">{field.description}</p>
              )}
            </div>
          );
        })}
      </div>

      {/* FR-3.5 — Live JSON preview */}
      {filterMetadata && (
        <details className="mt-3">
          <summary className="text-xs text-gray-500 cursor-pointer select-none hover:text-gray-700">
            Filter JSON preview
          </summary>
          <pre className="mt-1 text-xs bg-white border border-gray-200 rounded p-2 overflow-x-auto text-gray-700 leading-relaxed">
            {JSON.stringify(filterMetadata, null, 2)}
          </pre>
        </details>
      )}

      {/* FR-3.6 — Saved filters */}
      <div className="mt-3 flex flex-wrap items-center gap-2">
        {filterMetadata && !showSaveForm && (
          <button
            type="button"
            onClick={() => setShowSaveForm(true)}
            className="text-xs px-2 py-1 border border-amber-300 text-amber-700 rounded hover:bg-amber-50"
          >
            Save filter
          </button>
        )}
        {showSaveForm && (
          <div className="flex items-center gap-1">
            <input
              type="text"
              value={saveFilterName}
              onChange={(e) => setSaveFilterName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSaveFilter();
                if (e.key === 'Escape') { setShowSaveForm(false); setSaveFilterName(''); }
              }}
              placeholder="Filter name..."
              className="px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-amber-400"
              autoFocus
            />
            <button
              type="button"
              onClick={handleSaveFilter}
              disabled={!saveFilterName.trim()}
              className="text-xs px-2 py-1 bg-amber-500 text-white rounded hover:bg-amber-600 disabled:opacity-50"
            >
              Save
            </button>
            <button
              type="button"
              onClick={() => { setShowSaveForm(false); setSaveFilterName(''); }}
              className="text-xs text-gray-500 hover:text-gray-700 underline"
            >
              Cancel
            </button>
          </div>
        )}
        {savedFilters.length > 0 && (
          <select
            value={loadSelectValue}
            onChange={handleLoadFilter}
            className="text-xs px-2 py-1 border border-gray-300 rounded bg-white"
          >
            <option value="">Load saved filter…</option>
            {savedFilters.map((f) => (
              <option key={f.id} value={f.id}>
                {f.name}
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Load banner */}
      {loadedFilterName && (
        <div className="mt-2 flex items-center gap-2 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-2 py-1">
          <span>
            Loaded: <strong>{loadedFilterName}</strong>
          </span>
          <button
            type="button"
            onClick={() => { onFilterMetadataChange(undefined); setLoadedFilterName(null); }}
            className="underline hover:no-underline"
          >
            Clear
          </button>
          <button
            type="button"
            onClick={handleRemoveLoadedFilter}
            className="underline hover:no-underline text-red-500 hover:text-red-700"
          >
            Delete
          </button>
        </div>
      )}
    </div>
  );
}

export default SearchFilters;
