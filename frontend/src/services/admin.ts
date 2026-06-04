import { apiService } from './api';

export interface LocalUserCreated {
  readonly user_id: number;
  readonly email: string;
  readonly name: string;
  readonly set_password_token: string;
  readonly expires_at: string;
}

export interface ResetLinkResult {
  readonly set_password_token: string;
  readonly expires_at: string;
}

export interface User {
  readonly user_id: number;
  readonly email: string;
  readonly name?: string;
  readonly created_at: string;
  readonly owned_apps_count: number;
  readonly api_keys_count: number;
  readonly is_active: boolean;
  readonly is_omniadmin?: boolean;
}

export interface UserListResponse {
  readonly users: User[];
  readonly total: number;
  readonly page: number;
  readonly per_page: number;
  readonly total_pages: number;
}

export interface SystemStats {
  total_users: number;
  active_users: number;
  inactive_users: number;
  total_apps: number;
  total_agents: number;
  total_api_keys: number;
  active_api_keys: number;
  inactive_api_keys: number;
  recent_users: Array<{
    user_id: number;
    email: string;
    name?: string;
    created_at: string;
  }>;
  users_with_apps: number;
}

class AdminService {
  private readonly baseUrl = '/internal/admin';

  async getUsers(page: number = 1, perPage: number = 10, search?: string): Promise<UserListResponse> {
    const params = new URLSearchParams({
      page: page.toString(),
      per_page: perPage.toString(),
    });
    
    if (search) {
      params.append('search', search);
    }

    return await apiService.request(`${this.baseUrl}/users?${params}`);
  }

  async getUser(userId: number): Promise<User> {
    return await apiService.request(`${this.baseUrl}/users/${userId}`);
  }

  async deleteUser(userId: number): Promise<{ message: string }> {
    return await apiService.request(`${this.baseUrl}/users/${userId}`, {
      method: 'DELETE',
    });
  }

  async getSystemStats(): Promise<SystemStats> {
    return await apiService.request(`${this.baseUrl}/stats`);
  }

  async activateUser(userId: number): Promise<{ message: string; user_id: number; is_active: boolean }> {
    return await apiService.request(`${this.baseUrl}/users/${userId}/activate`, {
      method: 'POST',
    });
  }

  async deactivateUser(userId: number): Promise<{ message: string; user_id: number; is_active: boolean }> {
    return await apiService.request(`${this.baseUrl}/users/${userId}/deactivate`, {
      method: 'POST',
    });
  }

  async resetUserMarketplaceQuota(userId: number): Promise<{ message: string; user_id: number; previous_count: number; new_count: number; reset_by: string; timestamp: string }> {
    return await apiService.request(`${this.baseUrl}/users/${userId}/reset-marketplace-quota`, {
      method: 'POST',
    });
  }

  async createLocalUser(email: string, name: string): Promise<LocalUserCreated> {
    return await apiService.request(`${this.baseUrl}/users/local`, {
      method: 'POST',
      body: JSON.stringify({ email, name }),
    }) as LocalUserCreated;
  }

  async issueResetLink(userId: number): Promise<ResetLinkResult> {
    return await apiService.request(`${this.baseUrl}/users/${userId}/reset-link`, {
      method: 'POST',
    }) as ResetLinkResult;
  }

}

export const adminService = new AdminService(); 