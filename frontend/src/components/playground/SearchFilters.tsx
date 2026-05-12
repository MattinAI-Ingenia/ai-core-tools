import { useEffect, useMemo, useState } from 'react';
import { ChevronDown, ChevronRight, Plus, Search, X } from 'lucide-react';

export type MetadataOperator = '$eq' | '$ne' | '$gt' | '$gte' | '$lt' | '$lte';
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

interface CustomFilterRow {
  id: string;
  key: string;
  operator: MetadataOperator;
  value: string;
}

interface SystemFieldGroup {
  label: string;
  fields: Array<{ name: string; type: string; description: string }>;
}

const SYSTEM_FIELD_GROUPS: SystemFieldGroup[] = [
  {
    label: 'Media chunks',
    fields: [
      { name: 'chunk_type', type: 'str', description: 'visual · audio · text' },
      { name: 'content_type', type: 'str', description: 'media_chunk' },
      { name: 'language', type: 'str', description: 'e.g. english' },
      { name: 'start_time', type: 'float', description: 'chunk start (seconds)' },
      { name: 'end_time', type: 'float', description: 'chunk end (seconds)' },
      { name: 'media_id', type: 'int', description: 'media record ID' },
      { name: 'processing_mode', type: 'str', description: 'multimodal · transcription' },
    ],
  },
  {
    label: 'Documents',
    fields: [
      { name: 'resource_id', type: 'int', description: 'resource record ID' },
      { name: 'page', type: 'int', description: 'page number' },
      { name: 'total_pages', type: 'int', description: 'total pages in document' },
      { name: 'file_type', type: 'str', description: 'e.g. .pdf · .mp4' },
    ],
  },
  {
    label: 'Common',
    fields: [
      { name: 'repository_id', type: 'int', description: 'repository record ID' },
      { name: 'folder_id', type: 'int', description: 'folder record ID' },
      { name: 'silo_id', type: 'int', description: 'silo record ID' },
      { name: 'source_type', type: 'str', description: 'upload · url · domain' },
    ],
  },
];

interface SearchFiltersProps {
  metadataFields?: SearchFilterMetadataField[];
  dbType?: string;
  disabled?: boolean;
  onFilterMetadataChange: (filterMetadata: Record<string, unknown> | undefined) => void;
  pendingCustomFilter?: { key: string; value: string } | null;
  onPendingCustomFilterConsumed?: () => void;
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
  },
  QDRANT: {
    $eq: 'match',
    $ne: 'must_not_match',
    $gt: 'gt',
    $gte: 'gte',
    $lt: 'lt',
    $lte: 'lte',
  },
};

function normalizeDbType(dbType?: string): SupportedDbType {
  if (dbType?.toUpperCase() === 'QDRANT') {
    return 'QDRANT';
  }
  return DEFAULT_DB_TYPE;
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
        mustNot.push({
          key,
          match: { value },
        });
        break;
      case 'match':
        targetList.push({
          key,
          match: { value },
        });
        break;
      case 'gt':
      case 'gte':
      case 'lt':
      case 'lte':
        targetList.push({
          key,
          range: { [nativeOperator]: value },
        });
        break;
      default:
        // Unknown operator: fall back to match filter
        targetList.push({ key, match: { value } });
    }
  });

  const qdrantFilter: Record<string, unknown> = {};
  if (must.length > 0) {
    qdrantFilter.must = must;
  }
  if (should.length > 0) {
    qdrantFilter.should = should;
  }
  if (mustNot.length > 0) {
    qdrantFilter.must_not = mustNot;
  }

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
    if (!trimmedValue) {
      return;
    }

    const selectedOperator = filterOperators[fieldName] || '$eq';
    const nativeOperator = operatorMapping[selectedOperator];
    if (!nativeOperator) {
      return;
    }

    const fieldDefinition = metadataFields?.find((field) => field.name === fieldName);
    const convertedValue = fieldDefinition ? convertMetadataValue(fieldDefinition.type, trimmedValue) : trimmedValue;

    prepared.push({
      fieldName,
      operator: selectedOperator,
      nativeOperator,
      value: convertedValue,
    });
  });

  return prepared;
}

export function SearchFilters({
  metadataFields,
  dbType,
  disabled = false,
  onFilterMetadataChange,
  pendingCustomFilter,
  onPendingCustomFilterConsumed,
}: Readonly<SearchFiltersProps>) {
  const [metadataFilters, setMetadataFilters] = useState<Record<string, string>>({});
  const [filterOperators, setFilterOperators] = useState<Record<string, MetadataOperator>>({});
  const [logicalOperator, setLogicalOperator] = useState<'$and' | '$or'>('$and');
  const [customRows, setCustomRows] = useState<CustomFilterRow[]>([]);
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});

  useEffect(() => {
    setMetadataFilters({});
    setFilterOperators({});
    setLogicalOperator('$and');
  }, [metadataFields]);

  // Consume injected filter (from click-to-filter on result metadata badges)
  useEffect(() => {
    if (!pendingCustomFilter) return;
    const { key, value } = pendingCustomFilter;
    setCustomRows(prev => {
      const existing = prev.find(r => r.key === key);
      if (existing) {
        return prev.map(r => r.id === existing.id ? { ...r, value: String(value) } : r);
      }
      return [...prev, { id: crypto.randomUUID(), key, operator: '$eq', value: String(value) }];
    });
    onPendingCustomFilterConsumed?.();
  }, [pendingCustomFilter, onPendingCustomFilterConsumed]);

  const normalizedDbType = useMemo(() => normalizeDbType(dbType), [dbType]);
  const operatorMapping = FILTER_OPERATOR_MAPPINGS[normalizedDbType];

  useEffect(() => {
    const schemaPrepared = prepareFilters(metadataFilters, filterOperators, metadataFields, operatorMapping);
    const customPrepared: PreparedFilter[] = customRows
      .filter(r => r.key.trim() && r.value.trim())
      .map(r => ({
        fieldName: r.key.trim(),
        operator: r.operator,
        nativeOperator: operatorMapping[r.operator],
        value: r.value.trim(),
      }));
    const allPrepared = [...schemaPrepared, ...customPrepared];
    const filterMetadata = normalizedDbType === 'QDRANT'
      ? buildQdrantFilter(allPrepared, logicalOperator)
      : buildPgvectorFilter(allPrepared, logicalOperator);
    onFilterMetadataChange(allPrepared.length === 0 ? undefined : filterMetadata);
  }, [metadataFilters, filterOperators, logicalOperator, metadataFields, normalizedDbType, operatorMapping, onFilterMetadataChange, customRows]);

  const handleMetadataFilterChange = (fieldName: string, value: string, operator: MetadataOperator) => {
    setMetadataFilters((prev) => ({ ...prev, [fieldName]: value }));
    setFilterOperators((prev) => ({ ...prev, [fieldName]: operator }));
  };

  const addCustomRow = (key = '', value = '') => {
    setCustomRows(prev => [...prev, { id: crypto.randomUUID(), key, operator: '$eq', value }]);
  };

  const activeCustomKeys = new Set(customRows.map(r => r.key));
  const hasSchemaFields = metadataFields && metadataFields.length > 0;
  const hasCustomRows = customRows.length > 0;
  const hasActiveFilters = hasCustomRows || Object.values(metadataFilters).some(v => v.trim());

  return (
    <div className="border border-gray-200 rounded-lg bg-gray-50 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3">
        <h3 className="text-sm font-medium text-gray-700 flex items-center gap-1.5">
          <Search className="w-4 h-4" aria-hidden="true" />
          Filter by Metadata
          {hasActiveFilters && (
            <span className="ml-1 px-1.5 py-0.5 bg-yellow-100 text-yellow-800 text-xs rounded-full">
              {customRows.filter(r => r.key.trim() && r.value.trim()).length +
                Object.values(metadataFilters).filter(v => v.trim()).length} active
            </span>
          )}
        </h3>
        <div className="flex items-center gap-2">
          <label htmlFor="logicalOperator" className="text-xs text-gray-500">Match:</label>
          <select
            id="logicalOperator"
            value={logicalOperator}
            onChange={(e) => setLogicalOperator(e.target.value as '$and' | '$or')}
            className="px-2 py-1 border border-gray-300 rounded-lg text-xs font-medium bg-white"
            disabled={disabled}
          >
            <option value="$and">ALL (AND)</option>
            <option value="$or">ANY (OR)</option>
          </select>
        </div>
      </div>

      <div className="px-4 pb-4 space-y-4">
        {/* Schema-defined fields (from OutputParser) */}
        {hasSchemaFields && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {metadataFields.map((field) => {
              const operator = filterOperators[field.name] || '$eq';
              return (
                <div key={field.name}>
                  <label htmlFor={`filter_${field.name}`} className="block text-xs font-medium text-gray-700 mb-1">
                    {field.name}
                    <span className="text-gray-400 ml-1">({field.type})</span>
                  </label>
                  <div className="flex items-center gap-1">
                    <select
                      value={operator}
                      onChange={(e) => handleMetadataFilterChange(field.name, metadataFilters[field.name] || '', e.target.value as MetadataOperator)}
                      className="px-2 py-1.5 border border-gray-300 rounded-lg text-xs bg-white"
                      disabled={disabled}
                    >
                      <option value="$eq">equals</option>
                      <option value="$ne">≠</option>
                      <option value="$gt">&gt;</option>
                      <option value="$gte">≥</option>
                      <option value="$lt">&lt;</option>
                      <option value="$lte">≤</option>
                    </select>
                    <input
                      type="text"
                      id={`filter_${field.name}`}
                      value={metadataFilters[field.name] || ''}
                      onChange={(e) => handleMetadataFilterChange(field.name, e.target.value, operator)}
                      placeholder={field.description || `Filter by ${field.name}`}
                      className="w-full px-2 py-1.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:border-transparent text-xs bg-white"
                      disabled={disabled}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* System field groups */}
        <div className="space-y-1">
          {SYSTEM_FIELD_GROUPS.map(group => {
            const isOpen = expandedGroups[group.label] ?? false;
            return (
              <div key={group.label} className="border border-gray-200 rounded-lg overflow-hidden bg-white">
                <button
                  type="button"
                  onClick={() => setExpandedGroups(prev => ({ ...prev, [group.label]: !isOpen }))}
                  className="w-full flex items-center justify-between px-3 py-2 text-xs font-medium text-gray-600 hover:bg-gray-50"
                >
                  <span className="flex items-center gap-1.5">
                    {isOpen ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                    {group.label} fields
                  </span>
                  <span className="text-gray-400">{group.fields.length} fields</span>
                </button>
                {isOpen && (
                  <div className="px-3 pb-3 flex flex-wrap gap-1.5 border-t border-gray-100 pt-2">
                    {group.fields.map(field => {
                      const isActive = activeCustomKeys.has(field.name);
                      return (
                        <button
                          key={field.name}
                          type="button"
                          onClick={() => {
                            if (!isActive) addCustomRow(field.name);
                          }}
                          disabled={disabled || isActive}
                          title={field.description}
                          className={`inline-flex items-center gap-1 px-2 py-1 border rounded text-xs transition-colors ${
                            isActive
                              ? 'bg-yellow-50 border-yellow-300 text-yellow-700 cursor-default'
                              : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-yellow-50 hover:border-yellow-400 hover:text-yellow-700 cursor-pointer'
                          }`}
                        >
                          <Plus className={`w-3 h-3 ${isActive ? 'opacity-0' : ''}`} />
                          {field.name}
                          <span className="text-gray-400">({field.type})</span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Active custom filter rows */}
        {hasCustomRows && (
          <div className="space-y-2">
            {customRows.map(row => (
              <div key={row.id} className="flex items-center gap-2">
                <input
                  type="text"
                  value={row.key}
                  onChange={e => setCustomRows(prev => prev.map(r => r.id === row.id ? { ...r, key: e.target.value } : r))}
                  placeholder="field name"
                  className="w-36 px-2 py-1.5 border border-gray-300 rounded-lg text-xs focus:ring-2 focus:ring-yellow-500 focus:outline-none bg-white"
                  disabled={disabled}
                />
                <select
                  value={row.operator}
                  onChange={e => setCustomRows(prev => prev.map(r => r.id === row.id ? { ...r, operator: e.target.value as MetadataOperator } : r))}
                  className="px-2 py-1.5 border border-gray-300 rounded-lg text-xs bg-white"
                  disabled={disabled}
                >
                  <option value="$eq">equals</option>
                  <option value="$ne">≠</option>
                  <option value="$gt">&gt;</option>
                  <option value="$gte">≥</option>
                  <option value="$lt">&lt;</option>
                  <option value="$lte">≤</option>
                </select>
                <input
                  type="text"
                  value={row.value}
                  onChange={e => setCustomRows(prev => prev.map(r => r.id === row.id ? { ...r, value: e.target.value } : r))}
                  placeholder="value"
                  className="flex-1 px-2 py-1.5 border border-gray-300 rounded-lg text-xs focus:ring-2 focus:ring-yellow-500 focus:outline-none bg-white"
                  disabled={disabled}
                />
                <button
                  type="button"
                  onClick={() => setCustomRows(prev => prev.filter(r => r.id !== row.id))}
                  className="p-1.5 text-gray-400 hover:text-red-500 rounded"
                  title="Remove filter"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Add custom filter */}
        <button
          type="button"
          onClick={() => addCustomRow()}
          className="text-xs text-yellow-700 hover:text-yellow-900 flex items-center gap-1"
          disabled={disabled}
        >
          <Plus className="w-3.5 h-3.5" /> Add custom filter
        </button>
      </div>
    </div>
  );
}

export default SearchFilters;
