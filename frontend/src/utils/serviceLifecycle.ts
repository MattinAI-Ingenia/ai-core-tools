/**
 * End-of-service helpers and centralised strings (NFR-6: no hardcoded copy in components).
 * Lifecycle status is computed by the backend (D-9) — only dates are handled here.
 * Dates are calendar days stored as 00:00 UTC (D-1), so every conversion uses UTC.
 */

/** Date only, user locale, UTC so the picked calendar day never shifts. */
export function formatEndOfServiceDate(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleDateString(undefined, { timeZone: 'UTC' });
}

/** ISO datetime -> `YYYY-MM-DD` (UTC) for `<input type="date">`. Empty string when unset. */
export function dateInputValueFromIso(iso: string | null | undefined): string {
  if (!iso) return '';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  return date.toISOString().slice(0, 10);
}

/** `YYYY-MM-DD` -> `YYYY-MM-DDT00:00:00Z`, or null when empty. */
export function isoFromDateInput(value: string): string | null {
  return value ? `${value}T00:00:00Z` : null;
}

/** True when the picked day is today or earlier (UTC), i.e. saving retires the service now. */
export function isDateInputRetiredNow(value: string): boolean {
  if (!value) return false;
  const todayUtc = new Date().toISOString().slice(0, 10);
  return value <= todayUtc;
}

export const SERVICE_LIFECYCLE_STRINGS = {
  retired: 'Retired',
  retiredOn: (date: string) => `Retired on ${date}`,
  deprecatedNoDate: 'Deprecated',
  deprecated: (date: string) => `Deprecated – ends ${date}`,
  endOfServiceLabel: 'End of service',
  endOfServiceHelp:
    'From this date the service stops working. Deprecated warnings start before it.',
  clearDate: 'Clear',
  clearDateAria: 'Clear end of service date',
  deprecationNoticeLabel: 'Deprecation notice',
  deprecationNoticeHelp: 'Shown next to the service to explain the change. Optional.',
  noticeCounter: (used: number, max: number) => `${used}/${max}`,
  retireImmediately: 'Saving with this date retires the service immediately.',
  embeddingWarning: (date: string) =>
    `Retiring this embedding service makes the silo unusable. Re-index it into another embedding service before ${date}.`,
  aiWarning: (date: string) =>
    `Agents using this AI service stop working on ${date}. Move them to another AI service before then.`,
} as const;

export const SERVICE_WARNING_STRINGS = {
  titleRetired: 'Retired services in use',
  titleDeprecated: 'Deprecated services in use',
  retiredStatus: (date: string) => `Retired on ${date}`,
  deprecatedStatus: (date: string) => `Deprecated, ends ${date}`,
  roles: {
    llm: 'LLM',
    ocr_vision: 'OCR vision',
    transcription: 'Transcription',
    video: 'Video',
    media_embedding: 'Media embedding',
    embedding: 'Embedding',
    sandbox: 'Code interpreter sandbox',
    sandbox_app_default: 'Code interpreter sandbox (app default)',
  },
  silo_embedding: (via: string) => `Embedding (via silo ${via})`,
  middleware: (via: string) => `Middleware ${via}`,
  sandboxWarning: (date: string) =>
    `The code interpreter stops working on ${date}. Select another sandbox service before then.`,
  compactRetired: (n: number) => `Retired ${n === 1 ? 'service' : 'services'} (${n})`,
  compactDeprecated: (n: number) => `Deprecated ${n === 1 ? 'service' : 'services'} (${n})`,
  compactAria: (items: string) => `Service lifecycle: ${items}`,
} as const;

export const SERVICE_USAGE_STRINGS = {
  action: 'Used by',
  title: (name: string) => `Used by: ${name}`,
  loading: 'Loading usage...',
  loadError: 'Could not load what uses this service.',
  retry: 'Retry',
  close: 'Close',
  nothing: 'Nothing uses this service.',
  empty: 'None',
  sections: {
    agents: 'Agents',
    silos: 'Silos',
    repositories: 'Repositories',
    middlewares: 'Middlewares',
    scheduledTasks: 'Scheduled tasks',
  },
  appLabel: (name: string) => `App: ${name}`,
  agentRoles: {
    llm: 'LLM',
    ocr_vision: 'OCR vision',
    transcription: 'Transcription',
    video: 'Video',
    media_embedding: 'Media embedding',
    silo_embedding: 'Via silo',
    sandbox: 'Sandbox',
    sandbox_app_default: 'Sandbox (app default)',
  },
  siloRoles: {
    embedding: 'Embedding',
    transcription: 'Transcription',
    video: 'Video',
  },
  reindexWarning:
    'Retiring this embedding service makes the silos listed here unusable. Re-index each silo into another embedding service before the end-of-service date.',
  moveTitle: 'Move agents to another AI service',
  moveCount: (n: number) => `${n} ${n === 1 ? 'agent uses' : 'agents use'} this service as their main LLM`,
  moveNone: 'No agents use this service as their main LLM.',
  moveTargetLabel: 'Target AI service',
  moveTargetPlaceholder: 'Select a service...',
  moveTargetEnds: (name: string, date: string) => `${name} (ends ${date})`,
  moveNoTargets: 'There is no other available AI service to move the agents to.',
  moveButton: 'Move agents...',
  moveConfirm: (n: number, from: string, to: string) =>
    `Move ${n} ${n === 1 ? 'agent' : 'agents'} from ${from} to ${to}?`,
  moveConfirmButton: 'Confirm move',
  moveConfirming: 'Moving...',
  moveCancel: 'Cancel',
  moveSuccess: (n: number) => `Moved ${n} ${n === 1 ? 'agent' : 'agents'}`,
  moveCountChanged: (n: number) =>
    `The number of agents changed (now ${n}). Review the list and confirm again.`,
  moveNote:
    'Transcription, video, OCR vision and middleware references are not moved. They are listed above and must be changed manually.',
  moveError: 'Could not move the agents.',
} as const;

export const MAX_DEPRECATION_NOTICE_LENGTH = 1000;
