import React, { createContext, useContext, useState, useEffect } from 'react';
import type { ReactNode } from 'react';
import { configService } from '../core/ConfigService';
import { setApiAuthMode } from '../services/api';

export interface TierLimits {
  apps: number;
  agents: number;
  silos: number;
  llm_calls: number;
  collaborators: number;
  mcp_servers: number;
}

export interface Tiers {
  free: TierLimits;
  starter: TierLimits;
  pro: TierLimits;
}

export type AuthMode = 'oidc' | 'local';

/**
 * Resolve the auth mode from the build-time / runtime OIDC env flag.
 * Used as the fallback when GET /internal/config does not report auth_mode
 * (older backend) or is unreachable — so an OIDC deployment with a transient
 * config failure is NOT silently degraded to LOCAL.
 */
function resolveEnvAuthMode(): AuthMode {
  const runtimeConfig = (globalThis as unknown as Record<string, Record<string, string>>).__RUNTIME_CONFIG__;
  const oidcEnabled = runtimeConfig?.VITE_OIDC_ENABLED === undefined
    ? import.meta.env.VITE_OIDC_ENABLED === 'true'
    : runtimeConfig.VITE_OIDC_ENABLED === 'true';
  return oidcEnabled ? 'oidc' : 'local';
}

interface DeploymentModeContextType {
  readonly isSaasMode: boolean;
  readonly isLoading: boolean;
  readonly tiers: Tiers | null;
  /**
   * The auth mode reported by the backend's GET /internal/config endpoint.
   * 'local'  — admin-provisioned email+password with cookie sessions.
   * 'oidc'   — enterprise OIDC (Microsoft Entra, etc.).
   * null     — not yet resolved (while isLoading is true).
   */
  readonly authMode: AuthMode | null;
}

const DeploymentModeContext = createContext<DeploymentModeContextType>({
  isSaasMode: false,
  isLoading: true,
  tiers: null,
  authMode: null,
});

export const useDeploymentMode = () => useContext(DeploymentModeContext);

interface DeploymentModeProviderProps {
  children: ReactNode;
}

export const DeploymentModeProvider: React.FC<DeploymentModeProviderProps> = ({ children }) => {
  const [isSaasMode, setIsSaasMode] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [tiers, setTiers] = useState<Tiers | null>(null);
  const [authMode, setAuthMode] = useState<AuthMode | null>(null);

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const baseUrl = configService.getApiBaseUrl();
        const response = await fetch(`${baseUrl}/internal/config`);
        if (response.ok) {
          const data: {
            deployment_mode?: string;
            tiers?: Tiers;
            auth_mode?: string;
          } = await response.json();
          setIsSaasMode(data.deployment_mode === 'saas');
          setTiers(data.tiers ?? null);

          // Prefer the backend-reported auth_mode; fall back to the env flag
          // when absent (older backend) so existing deployments are not broken.
          const resolvedMode: AuthMode =
            data.auth_mode === 'oidc' || data.auth_mode === 'local'
              ? data.auth_mode
              : resolveEnvAuthMode();
          setAuthMode(resolvedMode);
          setApiAuthMode(resolvedMode);
        }
      } catch {
        // Endpoint unreachable: preserve the env-intended mode rather than
        // hardcoding 'local', which would break OIDC logins on a transient
        // config failure.
        const fallbackMode = resolveEnvAuthMode();
        setIsSaasMode(false);
        setTiers(null);
        setAuthMode(fallbackMode);
        setApiAuthMode(fallbackMode);
      } finally {
        setIsLoading(false);
      }
    };

    fetchConfig();
  }, []);

  return (
    <DeploymentModeContext.Provider value={{ isSaasMode, isLoading, tiers, authMode }}>
      {children}
    </DeploymentModeContext.Provider>
  );
};
