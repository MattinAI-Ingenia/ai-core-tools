from pydantic import BaseModel, ConfigDict, field_validator
from typing import Literal, Optional, Dict, Any
from datetime import datetime

# ==================== APP SCHEMAS ====================

class AppUsageStatsSchema(BaseModel):
    """Schema for app usage statistics"""
    usage_percentage: float
    stress_level: str  # "low", "moderate", "high", "critical", "unlimited"
    current_usage: int
    limit: int
    remaining: int
    reset_in_seconds: int
    is_over_limit: bool
    
    model_config = ConfigDict(from_attributes=True)


class AppListItemSchema(BaseModel):
    """Schema for app list items"""
    app_id: int
    name: str
    role: str  # "owner", "admin", "member"
    created_at: Optional[datetime] = None
    langsmith_configured: bool
    owner_id: int
    owner_name: Optional[str] = None
    owner_email: Optional[str] = None
    agent_rate_limit: int
    max_file_size_mb: Optional[int] = 0
    agent_cors_origins: Optional[str] = None
    enable_openai_api: bool = False
    # Entity counts for table display
    agent_count: int = 0
    repository_count: int = 0
    domain_count: int = 0
    silo_count: int = 0
    collaborator_count: int = 0
    # Usage statistics for speedometer
    usage_stats: Optional[AppUsageStatsSchema] = None
    is_frozen: bool = False

    model_config = ConfigDict(from_attributes=True)


class AppDetailSchema(BaseModel):
    """Schema for detailed app information"""
    app_id: int
    name: str
    langsmith_api_key: str
    user_role: str
    created_at: Optional[datetime] = None
    owner_id: int
    owner_email: Optional[str] = None
    owner_name: Optional[str] = None
    agent_rate_limit: int
    max_file_size_mb: Optional[int] = 0
    agent_cors_origins: Optional[str] = None
    enable_openai_api: bool = False
    # Entity counts for dashboard display
     
    agent_count: int = 0
    repository_count: int = 0
    domain_count: int = 0
    silo_count: int = 0
    collaborator_count: int = 0
    onboarding_dismissed: bool = False
    is_frozen: bool = False

    model_config = ConfigDict(from_attributes=True)


def _strip_if_string(v):
    # Common normalizer for credentials pasted from emails / .env files /
    # password managers — trailing whitespace breaks HTTP header construction.
    return v.strip() if isinstance(v, str) else v


class CreateAppSchema(BaseModel):
    """Schema for creating a new app"""
    name: str
    langsmith_api_key: Optional[str] = ""
    agent_rate_limit: Optional[int] = 0
    max_file_size_mb: Optional[int] = 0
    agent_cors_origins: Optional[str] = None

    @field_validator("langsmith_api_key", mode="before")
    @classmethod
    def _strip_credentials(cls, v):
        return _strip_if_string(v)


class UpdateAppSchema(BaseModel):
    """Schema for updating an app"""
    name: str
    langsmith_api_key: Optional[str] = ""
    agent_rate_limit: Optional[int] = 0
    max_file_size_mb: Optional[int] = 0
    agent_cors_origins: Optional[str] = None
    enable_openai_api: bool = False

    @field_validator("langsmith_api_key", mode="before")
    @classmethod
    def _strip_credentials(cls, v):
        return _strip_if_string(v)


class LangSmithTestRequestSchema(BaseModel):
    """Schema for testing a LangSmith API key.

    If ``api_key`` is omitted or matches the masked placeholder, the test runs
    against the key already persisted for the app.
    """
    api_key: Optional[str] = None

    @field_validator("api_key", mode="before")
    @classmethod
    def _strip_credentials(cls, v):
        return _strip_if_string(v)


class LangSmithTestResponseSchema(BaseModel):
    """Schema for the LangSmith API key test result."""
    valid: bool
    status: Literal["ok", "unauthorized", "network", "unknown"]
    message: str
    project_name: Optional[str] = None
    source: Optional[Literal["app", "env", "request"]] = None


# ==================== COLLABORATION SCHEMAS ====================

class CollaboratorListItemSchema(BaseModel):
    """Schema for collaborator list items"""
    id: int
    user_id: int
    user_name: str
    user_email: str
    role: str
    status: str
    invited_at: Optional[datetime] = None
    accepted_at: Optional[datetime] = None
    invited_by_name: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class CollaboratorDetailSchema(BaseModel):
    """Schema for detailed collaborator information"""
    id: int
    app_id: int
    user_id: int
    role: str
    status: str
    invited_by: int
    invited_at: Optional[datetime] = None
    accepted_at: Optional[datetime] = None
    user: Optional[Dict[str, Any]] = None
    inviter: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


class InviteCollaboratorSchema(BaseModel):
    """Schema for inviting a collaborator"""
    email: str
    role: str = "editor"  # "admin", "editor", "viewer"


class UpdateCollaboratorRoleSchema(BaseModel):
    """Schema for updating collaborator role"""
    role: str  # "admin", "editor", "viewer"


class InvitationResponseSchema(BaseModel):
    """Schema for responding to collaboration invitations"""
    action: str  # "accept" or "decline"


class CollaborationResponseSchema(BaseModel):
    """Schema for collaboration response"""
    success: bool
    message: str
    collaborator: Optional[CollaboratorDetailSchema] = None

    model_config = ConfigDict(from_attributes=True)


# ==================== OWNERSHIP TRANSFER SCHEMAS ====================

class OwnershipOfferRequest(BaseModel):
    """Request body for ``POST /internal/apps/{app_id}/ownership/offer``.

    Args:
        new_owner_id: Primary key of the user being offered ownership.
            Must be an active user and not the current owner.
    """

    new_owner_id: int


class OwnershipOfferResponse(BaseModel):
    """Response returned after a pending ownership offer is created or refreshed.

    Args:
        collaboration_id: PK of the ``AppCollaborator`` offer row.
        app_id: PK of the app for which ownership is being offered.
        new_owner_id: PK of the intended recipient.
        actor_user_id: PK of the user who created the offer.
    """

    collaboration_id: int
    app_id: int
    new_owner_id: int
    actor_user_id: int

    model_config = ConfigDict(from_attributes=True)


class OwnershipAcceptResponse(BaseModel):
    """Response returned after the recipient accepts an ownership offer.

    Returns a minimal app summary so the frontend can update its local state.

    Args:
        app_id: PK of the transferred app.
        name: Display name of the app.
        new_owner_id: PK of the user who is now the owner (the actor).
        previous_owner_id: PK of the former owner (now an ADMINISTRATOR collaborator).
    """

    app_id: int
    name: Optional[str] = None
    new_owner_id: int
    previous_owner_id: int

    model_config = ConfigDict(from_attributes=True)



