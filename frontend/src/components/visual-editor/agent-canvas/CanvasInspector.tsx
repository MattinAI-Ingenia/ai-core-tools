import { useId, useMemo, useRef, useState, type FocusEvent } from 'react';
import { Link } from 'react-router-dom';
import { toast } from 'sonner';
import RagConfigSection, { type RagConfigValue, type RagFixedFilter } from '../../forms/RagConfigSection';
import type { SearchFilterMetadataField } from '../../playground/SearchFilters';
import type { Agent } from '../../../services/api';
import { PIECE_VISUALS } from '../nodeKindConfig';
import { portsFor, type AgentType } from './agentPorts';
import type { CanvasSelection } from './AgentCanvas';
import type { CanvasPiece } from './useAgentCanvasData';

/** Saves `patch` on the edited agent through the canvas save queue; resolves true when it saved. */
export type SaveAgentFields = (patch: Record<string, unknown>) => Promise<boolean>;

export interface CanvasInspectorProps {
  readonly appId: number;
  readonly agent: Agent;
  readonly agentType: AgentType;
  /** Selection already resolved against the current agent and pieces; null shows the empty hint. */
  readonly selection: CanvasSelection | null;
  /** False for roles below EDITOR: every input is disabled. */
  readonly canEdit: boolean;
  readonly onSaveFields: SaveAgentFields;
}

interface AgentFieldValues {
  name: string;
  description: string;
  system_prompt: string;
  vision_system_prompt: string;
  text_system_prompt: string;
  temperature: number;
  is_tool: boolean;
  has_memory: boolean;
  enable_code_interpreter: boolean;
  skill_router_enabled: boolean;
}

type TextField = 'name' | 'description' | 'system_prompt' | 'vision_system_prompt' | 'text_system_prompt';

/** Same range as the agent form's temperature slider. */
const TEMPERATURE_MIN = 0;
const TEMPERATURE_MAX = 2;

/** Keys of the RAG config that the threshold strategy depends on. */
const THRESHOLD_KEYS: ReadonlyArray<keyof RagConfigValue> = ['rag_search_type', 'rag_score_threshold'];

const INPUT_CLASS =
  'w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:cursor-not-allowed disabled:opacity-60 dark:border-gray-600 dark:bg-gray-900 dark:text-gray-100 dark:placeholder:text-gray-500';
const LABEL_CLASS = 'mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300';
const HELP_CLASS = 'mt-1 text-xs text-gray-500 dark:text-gray-400';
const LINK_CLASS =
  'rounded text-sm font-medium text-indigo-600 underline-offset-2 hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 dark:text-indigo-300';
const CHECKBOX_CLASS =
  'mt-0.5 h-4 w-4 shrink-0 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 disabled:cursor-not-allowed dark:border-gray-600 dark:bg-gray-900';

function savedAgentFields(agent: Agent): AgentFieldValues {
  return {
    name: agent.name,
    description: agent.description ?? '',
    system_prompt: agent.system_prompt ?? '',
    vision_system_prompt: agent.vision_system_prompt ?? '',
    text_system_prompt: agent.text_system_prompt ?? '',
    temperature: agent.temperature,
    is_tool: agent.is_tool,
    has_memory: agent.has_memory,
    enable_code_interpreter: agent.enable_code_interpreter,
    skill_router_enabled: agent.skill_router_enabled === true,
  };
}

/** The agent's saved RAG values, with the same fallbacks the agent form uses for missing fields. */
function savedRagValue(agent: Agent): RagConfigValue {
  return {
    rag_k: agent.rag_k ?? 10,
    rag_search_type: agent.rag_search_type ?? 'similarity',
    rag_score_threshold: agent.rag_score_threshold ?? null,
    // Null means unlimited retrieval calls, so it is kept as is (not coerced to a number).
    rag_max_retrieval_calls: agent.rag_max_retrieval_calls ?? null,
    rag_fixed_filters: agent.rag_fixed_filters ?? [],
  };
}

/** Drops incomplete rows and the editor-only `_key`, the same cleaning the agent form applies before saving. */
function cleanFilters(filters: readonly RagFixedFilter[]): Array<Omit<RagFixedFilter, '_key'>> {
  return filters
    .filter((filter) => filter.field && (Array.isArray(filter.value) ? filter.value.length > 0 : String(filter.value ?? '').trim() !== ''))
    .map(({ _key: _ignored, ...rest }) => rest);
}

/** Route of a piece's own editor page (an agent opens its canvas view). */
function pieceEditPath(appId: number, piece: CanvasPiece): string {
  const base = `/apps/${appId}`;
  switch (piece.kind) {
    case 'agent':
      return `${base}/agents/${piece.id}?view=visual`;
    case 'silo':
      return `${base}/silos/${piece.id}`;
    case 'ai_service':
      return `${base}/settings/ai-services`;
    case 'embedding':
      return `${base}/settings/embedding-services`;
    case 'skill':
      return `${base}/skills`;
    case 'mcp':
      return `${base}/settings/mcp-configs`;
    case 'parser':
      return `${base}/settings/data-structures`;
    case 'sandbox':
      return `${base}/settings/sandbox-services`;
    case 'middleware':
      return `${base}/middlewares`;
  }
}

interface Drafts<T> {
  /** Unsaved edits, layered over the saved values. */
  readonly values: Partial<T>;
  /** Latest edits, readable synchronously (a blur right after a change must not see a stale render). */
  readonly read: () => Partial<T>;
  readonly set: <K extends keyof T>(key: K, value: T[K]) => void;
  readonly setAll: (patch: Partial<T>) => void;
  readonly clear: (key: keyof T) => void;
  /** Drops the draft only while it still holds `sent`: an edit made during the save keeps its draft. */
  readonly settle: (key: keyof T, sent: unknown) => void;
  /** Records a save as dispatched; until its refetch lands it is the value edits compare against. */
  readonly dispatch: (patch: Record<string, unknown>) => void;
  /** Forgets a dispatched value once its save resolved (the refetch has landed by then), if no newer save replaced it. */
  readonly landed: (key: keyof T, sent: unknown) => void;
  /** Last dispatched value for `key`, else `saved`. */
  readonly baseline: <K extends keyof T>(key: K, saved: T[K]) => T[K];
  /** True while a save for `key` is in flight or not yet refetched. */
  readonly isPending: (key: keyof T) => boolean;
}

/**
 * Unsaved field edits. Drafts show at once and are dropped when a save settles,
 * so a failed save reverts the field. Dispatched values bridge the gap until the
 * refetch lands, so an edit compares against what was last sent, not the stale fetch.
 */
function useDrafts<T extends object>(): Drafts<T> {
  const [values, setValues] = useState<Partial<T>>({});
  const latest = useRef<Partial<T>>({});
  const dispatched = useRef<Map<string, unknown>>(new Map());

  const replace = (next: Partial<T>) => {
    latest.current = next;
    setValues(next);
  };
  const clear = (key: keyof T) => {
    const next = { ...latest.current };
    delete next[key];
    replace(next);
  };

  return {
    values,
    read: () => latest.current,
    set: (key, value) => replace({ ...latest.current, [key]: value }),
    setAll: (patch) => replace({ ...latest.current, ...patch }),
    clear,
    settle: (key, sent) => {
      if (latest.current[key] === sent) clear(key);
    },
    dispatch: (patch) => {
      for (const [key, value] of Object.entries(patch)) dispatched.current.set(key, value);
    },
    landed: (key, sent) => {
      const name = String(key);
      if (dispatched.current.has(name) && dispatched.current.get(name) === sent) dispatched.current.delete(name);
    },
    baseline: (key, saved) =>
      dispatched.current.has(String(key)) ? (dispatched.current.get(String(key)) as typeof saved) : saved,
    isPending: (key) => dispatched.current.has(String(key)),
  };
}

interface ToggleFieldProps {
  readonly id: string;
  readonly label: string;
  readonly help: string;
  readonly checked: boolean;
  readonly disabled: boolean;
  readonly onToggle: (checked: boolean) => void;
}

function ToggleField({ id, label, help, checked, disabled, onToggle }: ToggleFieldProps) {
  const helpId = `${id}-help`;
  return (
    <div className="flex items-start gap-3 rounded-lg bg-gray-50 p-3 dark:bg-gray-900/40">
      <input
        id={id}
        type="checkbox"
        checked={checked}
        disabled={disabled}
        aria-describedby={helpId}
        onChange={(event) => onToggle(event.currentTarget.checked)}
        className={CHECKBOX_CLASS}
      />
      <div className="min-w-0">
        <label htmlFor={id} className="text-sm font-medium text-gray-900 dark:text-gray-100">
          {label}
        </label>
        <p id={helpId} className={HELP_CLASS}>
          {help}
        </p>
      </div>
    </div>
  );
}

interface AgentFieldsProps {
  readonly agent: Agent;
  readonly agentType: AgentType;
  readonly canEdit: boolean;
  readonly onSaveFields: SaveAgentFields;
}

/** Editable fields of the selected agent. Text saves on blur (when changed); toggles and the number save on change. */
function AgentFields({ agent, agentType, canEdit, onSaveFields }: AgentFieldsProps) {
  const baseId = useId();
  const saved = useMemo(() => savedAgentFields(agent), [agent]);
  const drafts = useDrafts<AgentFieldValues>();
  const values = useMemo<AgentFieldValues>(() => ({ ...saved, ...drafts.values }), [saved, drafts.values]);
  const isOcr = agentType === 'ocr_agent';
  // Temperature text while the user types; null shows the value. Parsed and clamped on blur only.
  const [temperatureText, setTemperatureText] = useState<string | null>(null);

  /** Saves one field unless it matches the last dispatched (or saved) value; a failed save reverts through settle. */
  async function persist<K extends keyof AgentFieldValues>(field: K, value: AgentFieldValues[K]): Promise<void> {
    if (value === drafts.baseline(field, saved[field])) {
      if (!drafts.isPending(field)) drafts.clear(field);
      return;
    }
    drafts.set(field, value);
    drafts.dispatch({ [field]: value });
    await onSaveFields({ [field]: value });
    drafts.landed(field, value);
    drafts.settle(field, value);
  }

  function commitText(field: TextField, text: string): void {
    if (field === 'name' && text.trim() === '') {
      // A blank name is never saved: drop the draft so the field shows the saved name again.
      drafts.clear('name');
      toast.error('Name is required');
      return;
    }
    void persist(field, text);
  }

  function commitTemperature(text: string): void {
    setTemperatureText(null);
    const parsed = Number.parseFloat(text);
    if (!Number.isFinite(parsed)) return;
    void persist('temperature', Math.min(TEMPERATURE_MAX, Math.max(TEMPERATURE_MIN, parsed)));
  }

  const textInput = (field: TextField, label: string) => (
    <div>
      <label htmlFor={`${baseId}-${field}`} className={LABEL_CLASS}>
        {label}
      </label>
      <input
        id={`${baseId}-${field}`}
        type="text"
        value={values[field]}
        disabled={!canEdit}
        onChange={(event) => drafts.set(field, event.currentTarget.value)}
        onBlur={(event) => commitText(field, event.currentTarget.value)}
        className={INPUT_CLASS}
      />
    </div>
  );

  const promptInput = (field: TextField, label: string) => (
    <div>
      <label htmlFor={`${baseId}-${field}`} className={LABEL_CLASS}>
        {label}
      </label>
      <textarea
        id={`${baseId}-${field}`}
        rows={6}
        value={values[field]}
        disabled={!canEdit}
        onChange={(event) => drafts.set(field, event.currentTarget.value)}
        onBlur={(event) => commitText(field, event.currentTarget.value)}
        className={`${INPUT_CLASS} resize-y`}
      />
    </div>
  );

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-base font-semibold text-gray-900 dark:text-gray-100">{isOcr ? 'OCR agent' : 'Agent'}</h3>
        <Link to="?view=form" className={LINK_CLASS}>
          Open full editor
        </Link>
      </div>

      {textInput('name', 'Name')}
      {textInput('description', 'Description')}
      {isOcr ? (
        <>
          {promptInput('vision_system_prompt', 'Vision prompt')}
          {promptInput('text_system_prompt', 'Text prompt')}
        </>
      ) : (
        promptInput('system_prompt', 'System prompt')
      )}

      <div>
        <label htmlFor={`${baseId}-temperature`} className={LABEL_CLASS}>
          Temperature
        </label>
        <input
          id={`${baseId}-temperature`}
          type="number"
          min={TEMPERATURE_MIN}
          max={TEMPERATURE_MAX}
          step={0.1}
          value={temperatureText ?? String(values.temperature)}
          disabled={!canEdit}
          aria-describedby={`${baseId}-temperature-help`}
          onChange={(event) => setTemperatureText(event.currentTarget.value)}
          onBlur={(event) => commitTemperature(event.currentTarget.value)}
          className={INPUT_CLASS}
        />
        <p id={`${baseId}-temperature-help`} className={HELP_CLASS}>
          0 = deterministic, 2 = very creative.
        </p>
      </div>

      <div className="space-y-2">
        <ToggleField
          id={`${baseId}-is_tool`}
          label="Tool agent"
          help="Can be used by other agents"
          checked={values.is_tool}
          disabled={!canEdit}
          onToggle={(checked) => void persist('is_tool', checked)}
        />
        <ToggleField
          id={`${baseId}-has_memory`}
          label="Conversational"
          help="Maintains conversation memory"
          checked={values.has_memory}
          disabled={!canEdit}
          onToggle={(checked) => {
            // Backend rule: conversational agents index chat uploads in a per-conversation
            // silo, which needs a media embedding service. Refuse without one (non-OCR only).
            if (checked && !isOcr && !agent.media_embedding_service_id) {
              toast.error('Conversational agents need a media embedding service to index files uploaded in chat — connect one to the Media embedding port first');
              return;
            }
            void persist('has_memory', checked);
          }}
        />
        <ToggleField
          id={`${baseId}-enable_code_interpreter`}
          label="Code interpreter"
          help="Allows the agent to execute Python code"
          checked={values.enable_code_interpreter}
          disabled={!canEdit}
          onToggle={(checked) => void persist('enable_code_interpreter', checked)}
        />
        <ToggleField
          id={`${baseId}-skill_router_enabled`}
          label="Skill router"
          help="Let the model pre-select at most 2 skills per turn"
          checked={values.skill_router_enabled}
          disabled={!canEdit}
          onToggle={(checked) => void persist('skill_router_enabled', checked)}
        />
      </div>
    </div>
  );
}

interface KnowledgeSettingsProps {
  readonly agent: Agent;
  readonly canEdit: boolean;
  readonly onSaveFields: SaveAgentFields;
}

/** RAG retrieval settings for the agent's knowledge edge. Saves on leaving the section, or at once when a filter row is removed. */
function KnowledgeSettings({ agent, canEdit, onSaveFields }: KnowledgeSettingsProps) {
  const saved = useMemo(() => savedRagValue(agent), [agent]);
  const drafts = useDrafts<RagConfigValue>();
  const value = useMemo<RagConfigValue>(() => ({ ...saved, ...drafts.values }), [saved, drafts.values]);
  const metadataFields = useMemo<SearchFilterMetadataField[]>(
    () =>
      (agent.silo?.metadata_definition?.fields ?? []).map((field) => ({
        name: field.name,
        type: field.type,
        description: field.description,
      })),
    [agent],
  );

  /**
   * Saves the drafts that differ from the saved agent. A threshold strategy
   * without a threshold is held back (the section asks for one); everything
   * else still saves.
   */
  function commit(): void {
    const pending = drafts.read();
    const effective = { ...saved, ...pending };
    const thresholdMissing =
      effective.rag_search_type === 'similarity_score_threshold' && effective.rag_score_threshold == null;

    const patch: Record<string, unknown> = {};
    const sent: Record<string, unknown> = {};
    for (const [key, draft] of Object.entries(pending) as Array<[keyof RagConfigValue, unknown]>) {
      if (draft === undefined) continue;
      if (thresholdMissing && THRESHOLD_KEYS.includes(key)) continue;
      const next = Array.isArray(draft) ? cleanFilters(draft) : draft;
      if (JSON.stringify(next) === JSON.stringify(drafts.baseline(key, saved[key]))) {
        if (!drafts.isPending(key)) drafts.clear(key);
        continue;
      }
      patch[key] = next;
      sent[key] = draft;
    }
    // Leaving the threshold strategy drops the saved threshold too (as the agent form does).
    if (
      patch.rag_search_type !== undefined &&
      effective.rag_search_type !== 'similarity_score_threshold' &&
      drafts.baseline('rag_score_threshold', saved.rag_score_threshold) !== null
    ) {
      patch.rag_score_threshold = null;
    }
    if (thresholdMissing && (pending.rag_search_type !== undefined || pending.rag_score_threshold !== undefined)) {
      toast.info('Score threshold (0–1) is required before the threshold strategy can be saved');
    }
    if (Object.keys(patch).length === 0) return;

    drafts.dispatch(patch);
    void onSaveFields(patch).then(() => {
      for (const key of Object.keys(patch) as Array<keyof RagConfigValue>) drafts.landed(key, patch[key]);
      for (const key of Object.keys(sent) as Array<keyof RagConfigValue>) drafts.settle(key, sent[key]);
    });
  }

  function handleChange(patch: Partial<RagConfigValue>): void {
    const removesFilter =
      patch.rag_fixed_filters !== undefined && patch.rag_fixed_filters.length < value.rag_fixed_filters.length;
    drafts.setAll(patch);
    // A removed row's button leaves the DOM, so the section never blurs: save now.
    if (removesFilter) commit();
  }

  function handleBlur(event: FocusEvent<HTMLFieldSetElement>): void {
    // Focus moving between controls of the section is not a commit.
    if (event.relatedTarget instanceof Node && event.currentTarget.contains(event.relatedTarget)) return;
    commit();
  }

  // RagConfigSection has no disabled prop, so the fieldset disables its controls.
  return (
    <fieldset
      disabled={!canEdit}
      onBlur={handleBlur}
      className="m-0 min-w-0 border-0 p-0 [&_:disabled]:cursor-not-allowed [&_:disabled]:opacity-60"
    >
      <legend className="sr-only">Retrieval settings</legend>
      <RagConfigSection value={value} onChange={handleChange} metadataFields={metadataFields} loadingMetadata={false} />
    </fieldset>
  );
}

interface PieceDetailsProps {
  readonly appId: number;
  readonly piece: CanvasPiece;
  /** Sentence describing the connection when an edge is selected. */
  readonly connection?: string;
}

function PieceDetails({ appId, piece, connection }: PieceDetailsProps) {
  const visual = PIECE_VISUALS[piece.kind];
  const Icon = visual.icon;
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <div className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${visual.chipClass}`}>
          <Icon className={`h-4 w-4 ${visual.iconClass}`} aria-hidden="true" />
        </div>
        <div className="min-w-0">
          <p className={`text-xs font-medium ${visual.captionClass}`}>{visual.typeCaption}</p>
          <h3 className="truncate text-base font-semibold text-gray-900 dark:text-gray-100" title={piece.name}>
            {piece.name}
          </h3>
        </div>
      </div>
      {connection && <p className="text-sm text-gray-600 dark:text-gray-400">{connection}</p>}
      <Link to={pieceEditPath(appId, piece)} className={LINK_CLASS}>
        {piece.kind === 'agent' ? 'Open its canvas' : 'Edit'}
      </Link>
    </div>
  );
}

/**
 * Right-hand inspector of the per-agent canvas. Shows the editor for the
 * current selection: the agent's fields, a piece's name and edit link, or the
 * RAG settings / connection of an edge. Every save goes through `onSaveFields`
 * (the canvas save queue), so field saves never race a connection.
 */
export function CanvasInspector({ appId, agent, agentType, selection, canEdit, onSaveFields }: CanvasInspectorProps) {
  return (
    <aside
      aria-label="Inspector"
      className="flex max-h-96 w-full shrink-0 flex-col overflow-hidden rounded-xl border border-gray-200 bg-white md:max-h-none md:w-96 dark:border-gray-700 dark:bg-gray-800"
    >
      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {selection === null && (
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Select the agent, a piece or a connection on the canvas to edit it here.
          </p>
        )}

        {selection?.type === 'agent' && (
          <AgentFields
            key={agent.agent_id}
            agent={agent}
            agentType={agentType}
            canEdit={canEdit}
            onSaveFields={onSaveFields}
          />
        )}

        {selection?.type === 'piece' && <PieceDetails appId={appId} piece={selection.piece} />}

        {selection?.type === 'edge' && selection.port === 'knowledge' && (
          <div className="space-y-4">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Knowledge: <span className="font-medium text-gray-900 dark:text-gray-100">{selection.piece.name}</span>
            </p>
            <KnowledgeSettings key={agent.agent_id} agent={agent} canEdit={canEdit} onSaveFields={onSaveFields} />
          </div>
        )}

        {selection?.type === 'edge' && selection.port !== 'knowledge' && (
          <PieceDetails
            appId={appId}
            piece={selection.piece}
            connection={`Connected to ${portsFor(agentType).find((port) => port.id === selection.port)?.label ?? 'this port'}.${canEdit ? ' Press Delete or Backspace on the canvas to disconnect it.' : ''}`}
          />
        )}
      </div>
    </aside>
  );
}
