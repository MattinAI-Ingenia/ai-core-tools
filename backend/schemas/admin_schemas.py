from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict


class UserListResponse(BaseModel):
    users: List[dict]
    total: int
    page: int
    per_page: int
    total_pages: int


class UserDetailResponse(BaseModel):
    user_id: int
    email: str
    name: Optional[str] = None
    created_at: str
    owned_apps_count: int
    api_keys_count: int
    is_active: bool
    platform_role: str = 'editor'


class SetPlatformRoleRequest(BaseModel):
    role: str


class SystemStatsResponse(BaseModel):
    total_users: int
    active_users: int
    inactive_users: int
    total_apps: int
    total_agents: int
    total_api_keys: int
    active_api_keys: int
    inactive_api_keys: int
    recent_users: List[dict]
    users_with_apps: int


class MarketplaceQuotaResetResponse(BaseModel):
    message: str
    user_id: int
    user_email: str
    previous_count: int
    new_count: int
    reset_by: str
    timestamp: str


# ==================== SAAS ADMIN SCHEMAS ====================

class UserAdminRead(BaseModel):
    """Extended user info for OMNIADMIN SaaS dashboard."""
    user_id: int
    email: str
    name: Optional[str]
    is_active: bool
    auth_method: Optional[str] = 'oidc'
    email_verified: bool = True
    tier: Optional[str] = None
    billing_status: Optional[str] = None
    stripe_customer_id: Optional[str] = None
    call_count: int = 0
    call_limit: int = 0
    owned_apps_count: int = 0

    model_config = ConfigDict(from_attributes=True)


class TierOverrideRequest(BaseModel):
    """Request body for OMNIADMIN manual tier override."""
    tier: str  # 'free', 'starter', or 'pro'


# ==================== APP OWNERSHIP TRANSFER SCHEMAS ====================

class TransferOwnerRequest(BaseModel):
    """Request body for OMNIADMIN administrative-direct ownership transfer.

    Args:
        new_owner_id: Primary key of the user who will become the new app owner.
            Must be an active user and not the current owner.
    """

    new_owner_id: int


class AppTransferSummary(BaseModel):
    """Response returned after a successful ownership transfer.

    Returns enough information for the admin UI to update its local state
    without requiring a full app-list refresh.
    """

    app_id: int
    name: Optional[str] = None  # App.name is practically always set; Optional guards null-name rows
    previous_owner_id: int
    new_owner_id: int

    model_config = ConfigDict(from_attributes=True)


# ==================== USER DELETION SCHEMAS ====================


class DeleteUserRequest(BaseModel):
    """Optional request body for ``DELETE /admin/users/{user_id}``.

    All fields are optional so that omitting the body entirely is valid and
    preserves backward compatibility (defaults to ``mode='block'``).

    Args:
        mode: Strategy for handling apps owned by the target user.
            ``'block'`` (default) — reject with HTTP 409 if the user owns any
            apps; no mutation occurs.  The 409 body includes the list of owned
            apps so the client can present a transfer/cascade dialog.
            ``'cascade_apps'`` — call ``AppService.delete_app`` for every owned
            app (ordered, vector-store-cleaning), then delete the user.
            ``'transfer_apps'`` — reassign every owned app to
            ``transfer_to_user_id`` (administrative-direct, no handshake), then
            delete the user.
        transfer_to_user_id: Required when ``mode='transfer_apps'``.
            Primary key of the user who will receive all transferred apps.
            Ignored for other modes.
    """

    mode: Literal["block", "cascade_apps", "transfer_apps"] = "block"
    transfer_to_user_id: Optional[int] = None


class OwnedAppConflictItem(BaseModel):
    """One entry in the ``owned_apps`` list of a 409 owned-apps conflict response.

    Args:
        app_id: Primary key of the owned app.
        name: Display name of the app.
    """

    app_id: int
    name: str


class OwnedAppsConflictResponse(BaseModel):
    """HTTP 409 response body when deleting a user who owns apps with ``mode='block'``.

    The ``owned_apps`` list gives the client enough information to render a
    transfer/cascade dialog without a separate preflight request.

    Args:
        detail: Human-readable explanation of the conflict.
        owned_apps: Summaries of every app owned by the target user.
    """

    detail: str
    owned_apps: List[OwnedAppConflictItem]
