import { configService } from '../core/ConfigService';
import type { User } from 'oidc-client-ts';
import { getCsrfToken } from './cookies';

class AuthService {
  private get baseURL(): string {
    return configService.getApiBaseUrl();
  }

  /**
   * Extracts a human-readable message from a failed FastAPI response.
   *
   * Handles both shapes the backend can emit:
   * - ``{ detail: "string" }`` — HTTPException raised in the route handler.
   * - ``{ detail: [{ msg, loc, ... }] }`` — Pydantic 422 schema-validation error.
   *
   * Without this, a 422 (e.g. password fails the schema-level policy check)
   * surfaces to the user as a bare ``HTTP 422`` because ``detail`` is an array,
   * not a string.
   */
  private async extractErrorMessage(response: Response, fallback: string): Promise<string> {
    const data = await response.json().catch(() => null);
    const detail = (data as { detail?: unknown } | null)?.detail;

    if (typeof detail === 'string') {
      return detail;
    }
    if (Array.isArray(detail) && detail.length > 0) {
      const first = detail[0] as { msg?: unknown };
      if (typeof first.msg === 'string') {
        // Pydantic v2 prefixes ValueError messages with "Value error, ".
        return first.msg.replace(/^Value error,\s*/i, '');
      }
    }
    return fallback;
  }

  // ==================== OIDC BRIDGE ====================
  // The OIDC library manages its own tokens in localStorage under oidc-client-ts
  // keys.  We only store the OIDC ID token in this module-level variable so
  // api.ts can read it synchronously for the Authorization: Bearer header.
  private oidcAccessToken: string | null = null;

  /**
   * Called by OIDCProvider after a successful login or silent-renew.
   * Stores the OIDC ID token in memory — NOT in localStorage under our own key.
   *
   * We send the ID token (aud = client_id), NOT the access token: Azure AD issues
   * the access token for the `<audience>/.default` scope with aud = api://<client_id>,
   * which fails the backend's ID-token audience validation
   * (ENTRA_TOKEN_TYPE=id_token, validate_audience=true) → 401 on every endpoint.
   */
  setOIDCToken(user: User) {
    this.oidcAccessToken = user.id_token ?? null;
  }

  /**
   * Returns the in-memory OIDC access token, or null when not in OIDC mode.
   */
  getOIDCToken(): string | null {
    return this.oidcAccessToken;
  }

  /**
   * Clears the in-memory OIDC token.  Called by OIDCProvider on logout / expiry.
   */
  clearOIDCToken() {
    this.oidcAccessToken = null;
  }

  // ==================== LEGACY STUB (no-op) ====================
  // These no-ops keep callers compiled while the references are removed.
  clearAuth() {
    this.clearOIDCToken();
  }

  /**
   * Auth state for LOCAL mode is derived from the cookie session (checked via
   * /internal/me), not from localStorage.  For OIDC mode the OIDC library is
   * authoritative.  This method is therefore no longer a reliable check and
   * callers should use UserContext.user instead.
   */
  isAuthenticated(): boolean {
    return this.oidcAccessToken !== null;
  }

  // ==================== LOCAL AUTH ENDPOINTS ====================

  /**
   * POST /internal/auth/login — sets httpOnly access_token + refresh_token
   * cookies and a readable csrf_token cookie.  Returns the user object.
   * Never writes to localStorage.
   */
  async localLogin(email: string, password: string): Promise<{ user: { user_id: number; email: string; name?: string; is_admin?: boolean; is_omniadmin?: boolean } }> {
    const url = `${this.baseURL}/internal/auth/login`;
    const response = await fetch(url, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      throw new Error(await this.extractErrorMessage(response, 'Login failed'));
    }

    return response.json();
  }

  /**
   * POST /internal/auth/logout — clears session cookies.
   */
  async logout(): Promise<void> {
    const url = `${this.baseURL}/internal/auth/logout`;
    const csrf = getCsrfToken();
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (csrf) {
      headers['X-CSRF-Token'] = csrf;
    }

    await fetch(url, {
      method: 'POST',
      credentials: 'include',
      headers,
    }).catch(() => {
      // Best-effort — even if this fails we clear local state
    });
  }

  /**
   * POST /internal/auth/refresh — rotates the cookie pair.
   * Used internally by api.ts for silent-refresh on 401.
   * Returns true on success, false on failure.
   */
  async refresh(): Promise<boolean> {
    const url = `${this.baseURL}/internal/auth/refresh`;
    const csrf = getCsrfToken();
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (csrf) {
      headers['X-CSRF-Token'] = csrf;
    }

    try {
      const response = await fetch(url, {
        method: 'POST',
        credentials: 'include',
        headers,
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * POST /internal/auth/change-password — authenticated; rotates all sessions.
   */
  async changePassword(currentPassword: string, newPassword: string): Promise<void> {
    const url = `${this.baseURL}/internal/auth/change-password`;
    const csrf = getCsrfToken();
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (csrf) {
      headers['X-CSRF-Token'] = csrf;
    }

    const response = await fetch(url, {
      method: 'POST',
      credentials: 'include',
      headers,
      body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
    });

    if (!response.ok) {
      throw new Error(await this.extractErrorMessage(response, 'Password change failed'));
    }
  }

  /**
   * POST /internal/auth/set-password — unauthenticated one-time token flow.
   * No session is created; the user must log in after this.
   */
  async setPassword(token: string, password: string): Promise<void> {
    const url = `${this.baseURL}/internal/auth/set-password`;
    const response = await fetch(url, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token, new_password: password }),
    });

    if (!response.ok) {
      throw new Error(await this.extractErrorMessage(response, 'Set password failed. The link may have expired.'));
    }
  }

  // ==================== SHARED ====================

  /**
   * GET /internal/me — returns the current user from the backend.
   * Works for both cookie sessions (LOCAL) and OIDC (bearer header added
   * by api.ts's normal request path).  Call via authService directly only
   * for simple probing; prefer apiService for standard requests.
   */
  async getCurrentUser(): Promise<{ user_id: number; email: string; name?: string; is_admin?: boolean; is_omniadmin?: boolean }> {
    const url = `${this.baseURL}/internal/me`;
    const headers: Record<string, string> = {};

    // For OIDC we attach the bearer; for LOCAL the cookie handles it.
    const oidcToken = this.oidcAccessToken;
    if (oidcToken) {
      headers['Authorization'] = `Bearer ${oidcToken}`;
    }

    const response = await fetch(url, {
      credentials: 'include',
      headers,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    return response.json();
  }
}

export const authService = new AuthService();
