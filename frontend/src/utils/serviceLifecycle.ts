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

export const MAX_DEPRECATION_NOTICE_LENGTH = 1000;
