import React, { createContext, useContext, useState, useEffect, useMemo, useCallback } from 'react';
import type { ReactNode } from 'react';
import { authService } from '../services/auth';
import { OIDCContext } from '../auth/OIDCProvider';
import { configService } from '../core/ConfigService';

export interface User {
  user_id: number;
  email: string;
  name?: string;
  is_authenticated: boolean;
  is_admin?: boolean;
  is_omniadmin?: boolean;
  platform_role?: 'viewer' | 'editor' | 'admin';
  /** True for editors and admins; false for viewers */
  is_editor?: boolean;
}

interface UserContextType {
  user: User | null;
  loading: boolean;
  setUser: (user: User | null) => void;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

export const useUser = () => {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
};

interface UserProviderProps {
  children: ReactNode;
}

/**
 * Fetches /internal/me with the given Authorization header (OIDC) or relying
 * on cookies (LOCAL).  Returns a User or null.
 */
async function fetchUserFromBackend(
  bearerToken: string | null,
): Promise<User | null> {
  const baseUrl = configService.getApiBaseUrl();
  const headers: Record<string, string> = {};

  if (bearerToken) {
    headers['Authorization'] = `Bearer ${bearerToken}`;
  }

  try {
    const response = await fetch(`${baseUrl}/internal/me`, {
      credentials: 'include',
      headers,
    });

    if (!response.ok) return null;

    const userData: {
      user_id: number;
      email: string;
      name?: string;
      is_admin?: boolean;
      is_omniadmin?: boolean;
      platform_role?: 'viewer' | 'editor' | 'admin';
    } = await response.json();

    return {
      user_id: userData.user_id,
      email: userData.email,
      name: userData.name,
      is_authenticated: true,
      is_admin: userData.is_admin ?? userData.is_omniadmin ?? false,
      is_omniadmin: userData.is_omniadmin ?? false,
      platform_role: userData.platform_role,
      is_editor: (userData.is_admin ?? false) || userData.platform_role !== 'viewer',
    };
  } catch {
    return null;
  }
}

export const UserProvider: React.FC<UserProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // Access OIDC context directly to avoid circular dependency with useAuth
  const oidcContext = useContext(OIDCContext);

  const refreshUser = useCallback(async () => {
    try {
      if (oidcContext?.user) {
        // OIDC mode: attach the OIDC ID token as bearer (aud = client_id).
        const token = oidcContext.user.id_token ?? null;
        const resolved = await fetchUserFromBackend(token);

        if (resolved) {
          setUser(resolved);
          return;
        }

        // Fallback: use OIDC profile fields when the backend call fails.
        const oidcUser = oidcContext.user;
        setUser({
          user_id: 0,
          email: (oidcUser.profile as Record<string, string>)?.email ?? '',
          name: (oidcUser.profile as Record<string, string>)?.name
            ?? (oidcUser.profile as Record<string, string>)?.preferred_username,
          is_authenticated: true,
          is_admin: false,
          is_omniadmin: false,
          platform_role: 'viewer',
          is_editor: false,
        });
      } else {
        // LOCAL cookie mode (or unauthenticated): probe /internal/me with cookies.
        // No bearer token; the httpOnly access_token cookie is sent automatically.
        const resolved = await fetchUserFromBackend(null);
        setUser(resolved);
      }
    } catch {
      setUser(null);
    }
  }, [oidcContext?.user]);

  const logout = useCallback(async () => {
    if (oidcContext?.user) {
      // OIDC logout
      await oidcContext.logout();
    } else {
      // LOCAL cookie logout — clears server-side session and cookies
      await authService.logout();
    }
    setUser(null);
  }, [oidcContext?.user, oidcContext?.logout]);

  useEffect(() => {
    const initializeUser = async () => {
      try {
        if (oidcContext?.user) {
          // OIDC mode: attach the OIDC ID token as bearer (aud = client_id).
          const token = oidcContext.user.id_token ?? null;
          const resolved = await fetchUserFromBackend(token);

          if (resolved) {
            setUser(resolved);
            setLoading(false);
            return;
          }

          // Fallback to OIDC profile when the backend call fails.
          const oidcUser = oidcContext.user;
          setUser({
            user_id: 0,
            email: (oidcUser.profile as Record<string, string>)?.email ?? '',
            name: (oidcUser.profile as Record<string, string>)?.name
              ?? (oidcUser.profile as Record<string, string>)?.preferred_username,
            is_authenticated: true,
            is_admin: false,
            is_omniadmin: false,
            platform_role: 'viewer',
            is_editor: false,
          });
        } else {
          // LOCAL mode: probe with cookies
          const resolved = await fetchUserFromBackend(null);
          setUser(resolved);
        }
      } catch {
        setUser(null);
      } finally {
        setLoading(false);
      }
    };

    // Wait for OIDC library to finish its own initialization
    if (!oidcContext?.loading) {
      initializeUser();
    }
  }, [oidcContext?.user, oidcContext?.loading]);

  // Mirror OIDC loading state so the app doesn't flash an unauthenticated
  // state while the OIDC library is processing the callback or silent-renew.
  useEffect(() => {
    if (oidcContext) {
      setLoading(oidcContext.loading);
    }
  }, [oidcContext?.loading]);

  const value: UserContextType = useMemo(
    () => ({
      user,
      loading,
      setUser,
      logout,
      refreshUser,
    }),
    [user, loading, logout, refreshUser],
  );

  return <UserContext.Provider value={value}>{children}</UserContext.Provider>;
};
