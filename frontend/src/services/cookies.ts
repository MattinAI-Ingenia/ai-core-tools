/**
 * Utilities for reading browser cookies.
 * Used to extract the non-httpOnly csrf_token set by the LOCAL auth backend.
 */

/**
 * Parses document.cookie and returns the value of the named cookie,
 * or null if not present.
 */
export function getCookieValue(name: string): string | null {
  if (typeof document === 'undefined') return null;

  const encoded = encodeURIComponent(name);
  const prefix = `${encoded}=`;
  const parts = document.cookie.split(';');

  for (const part of parts) {
    const trimmed = part.trim();
    if (trimmed.startsWith(prefix)) {
      return decodeURIComponent(trimmed.slice(prefix.length));
    }
  }

  return null;
}

/**
 * Returns the current CSRF token from the csrf_token cookie set by the
 * LOCAL auth backend (readable, non-httpOnly).  Returns null when the
 * cookie is absent (OIDC mode or unauthenticated).
 */
export function getCsrfToken(): string | null {
  return getCookieValue('csrf_token');
}
