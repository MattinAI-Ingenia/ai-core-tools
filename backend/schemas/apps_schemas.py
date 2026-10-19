from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime

class AppUsageStatsSchema(BaseModel):
    usage_percentage: float
    stress_level: str  # "low", "moderate", "high", "critical", "unlimited"
    current_usage: int
    limit: int
    remaining: int
    reset_in_seconds: int
    is_over_limit: bool
    
    model_config = ConfigDict(from_attributes=True)


class AppListItemSchema(BaseModel):
    app_id: int
    name: str
    role: str
    created_at: Optional[datetime] = None
    langsmith_configured: bool
    owner_id: int
    owner_name: Optional[str] = None
    owner_email: Optional[str] = None
    agent_rate_limit: int
    max_file_size_mb: Optional[int] = 0
    agent_cors_origins: Optional[str] = None
    enable_openai_api: bool = False
    agent_count: int = 0
    repository_count: int = 0
    domain_count: int = 0
    silo_count: int = 0
    collaborator_count: int = 0
    usage_stats: Optional[AppUsageStatsSchema] = None
    is_frozen: bool = False

    model_config = ConfigDict(from_attributes=True)


class AppDetailSchema(BaseModel):
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
    agent_count: int = 0
    repository_count: int = 0
    domain_count: int = 0
    silo_count: int = 0
    collaborator_count: int = 0
    onboarding_dismissed: bool = False
    is_frozen: bool = False
    default_sandbox_service_id: Optional[int] = None  # None = inherit system default
    # Cross-session user memory (D-16, FR-2b): both NULL = memory inactive for this App
    memory_extraction_service_id: Optional[int] = None
    memory_embedding_service_id: Optional[int] = None
    # Fact-extraction cadence: turn 1 and every Nth turn after. None = the
    # deployment default, reported alongside so the UI can show it.
    memory_write_every_n_turns: Optional[int] = None
    memory_write_every_n_turns_default: int = 5
    # Language memory facts are written in (code); None = English.
    memory_language: Optional[str] = None
    # Option lists for the App-settings memory service selectors (FR-3(b)); App-visible
    # plus system-level services, labelled like the agent form's `ai_services` list.
    ai_services: List[Dict[str, Any]] = []
    embedding_services: List[Dict[str, Any]] = []

    model_config = ConfigDict(from_attributes=True)


def _strip_if_string(v):
    # Common normalizer for credentials pasted from emails / .env files /
    # password managers — trailing whitespace breaks HTTP header construction.
    return v.strip() if isinstance(v, str) else v


class CreateAppSchema(BaseModel):
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
    name: str
    langsmith_api_key: Optional[str] = ""
    agent_rate_limit: Optional[int] = 0
    max_file_size_mb: Optional[int] = 0
    agent_cors_origins: Optional[str] = None
    enable_openai_api: bool = False
    default_sandbox_service_id: Optional[int] = None  # None = inherit system default
    # Cross-session user memory (D-16, FR-2b): both None = memory inactive for this App,
    # not "use a default" — there is no system-level fallback (D-21).
    # NOTE: the `PUT /internal/apps/{id}` handler reads `model_fields_set` for these two
    # fields specifically, so omitting them in the request body leaves the persisted
    # value untouched instead of nulling it out (security-audit LOW, step_009
    # follow-up) — an explicit `null` still clears the configuration.
    memory_extraction_service_id: Optional[int] = None
    memory_embedding_service_id: Optional[int] = None
    # Fact-extraction cadence (turn 1 and every Nth turn after); null = deployment
    # default. Same `model_fields_set` handling as the two fields above.
    memory_write_every_n_turns: Optional[int] = Field(default=None, ge=1, le=100)
    # Language memory facts are written in; null = English. A closed set of
    # codes (tools/memory/settings.py::MEMORY_LANGUAGES), never free text.
    memory_language: Optional[Literal["en", "es", "eu", "ca", "gl", "fr", "de", "it", "pt"]] = None

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
    id: int
    user_id: int
    user_name: str
    user_email: str
    role: str
    status: str
    invited_at: Optional[datetime] = None
    accepted_at: Optional[datetime] = None
    invited_by_name: Optional[str] = None
    platform_role: str = 'editor'

    model_config = ConfigDict(from_attributes=True)


class CollaboratorDetailSchema(BaseModel):
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
    email: str
    role: str = "editor"


class UpdateCollaboratorRoleSchema(BaseModel):
    role: str


class InvitationResponseSchema(BaseModel):
    action: str  # "accept" or "decline"


class CollaborationResponseSchema(BaseModel):
    success: bool
    message: str
    collaborator: Optional[CollaboratorDetailSchema] = None

    model_config = ConfigDict(from_attributes=True)


class OwnershipOfferRequest(BaseModel):
    new_owner_id: int


class OwnershipOfferResponse(BaseModel):
    collaboration_id: int
    app_id: int
    new_owner_id: int
    actor_user_id: int

    model_config = ConfigDict(from_attributes=True)


class OwnershipAcceptResponse(BaseModel):
    """Returned after the recipient accepts; previous_owner_id is now an ADMINISTRATOR collaborator."""

    app_id: int
    name: Optional[str] = None
    new_owner_id: int
    previous_owner_id: int

    model_config = ConfigDict(from_attributes=True)



