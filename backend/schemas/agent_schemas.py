from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime
from models.agent import DEFAULT_AGENT_TEMPERATURE, DEFAULT_MEMORY_SUMMARIZE_THRESHOLD


# ==================== RETRIEVAL CONFIG ====================

class RetrievalConfig(BaseModel):
    """Configures how an agent retrieves documents from its linked Silo.

    Priority at runtime: runtime search_params > retrieval_config > system defaults.
    """
    search_type: Optional[Literal["similarity", "mmr", "similarity_score_threshold"]] = "similarity"
    k: Optional[int] = 30
    # MMR-specific
    fetch_k: Optional[int] = 100
    lambda_mult: Optional[float] = 0.5
    # Similarity-score-threshold-specific
    score_threshold: Optional[float] = None

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (1 <= v <= 200):
            raise ValueError("k must be between 1 and 200")
        return v

    @field_validator("fetch_k")
    @classmethod
    def validate_fetch_k(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("fetch_k must be at least 1")
        return v

    @field_validator("lambda_mult")
    @classmethod
    def validate_lambda_mult(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("lambda_mult must be between 0.0 and 1.0")
        return v

    @field_validator("score_threshold")
    @classmethod
    def validate_score_threshold(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("score_threshold must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def validate_consistency(self) -> "RetrievalConfig":
        if self.search_type == "similarity_score_threshold" and self.score_threshold is None:
            raise ValueError("score_threshold is required when search_type is 'similarity_score_threshold'")
        return self

# ==================== AGENT SCHEMAS ====================

class AgentListItemSchema(BaseModel):
    """Schema for agent list items"""
    agent_id: int
    name: str
    description: Optional[str] = None
    type: str  # "agent", "ocr_agent", etc.
    is_tool: bool
    created_at: Optional[datetime] = None
    request_count: int
    service_id: Optional[int] = None
    ai_service: Optional[Dict[str, Any]] = None  # AI service details
    marketplace_visibility: Optional[str] = None
    is_frozen: bool = False

    model_config = ConfigDict(from_attributes=True)


class AgentDetailSchema(BaseModel):
    """Schema for detailed agent information"""
    agent_id: int
    name: str
    description: str
    system_prompt: str
    prompt_template: str
    type: str
    is_tool: bool
    has_memory: bool
    enable_code_interpreter: bool = False
    server_tools: List[str] = []
    memory_max_messages: int = 20
    memory_max_tokens: Optional[int] = 4000
    memory_summarize_threshold: int = DEFAULT_MEMORY_SUMMARIZE_THRESHOLD
    service_id: Optional[int] = None
    silo_id: Optional[int] = None
    output_parser_id: Optional[int] = None
    temperature: float = DEFAULT_AGENT_TEMPERATURE
    tool_ids: List[int] = []
    mcp_config_ids: List[int] = []
    skill_ids: List[int] = []
    retrieval_config: Optional[Dict[str, Any]] = None
    middleware_ids: List[int] = []
    created_at: Optional[datetime] = None
    request_count: int
    # OCR-specific fields
    vision_service_id: Optional[int] = None
    vision_system_prompt: Optional[str] = None
    text_system_prompt: Optional[str] = None
    # Silo information for playground
    silo: Optional[Dict[str, Any]] = None
    # Output parser information for playground
    output_parser: Optional[Dict[str, Any]] = None
    # Form data for editing
    ai_services: List[Dict[str, Any]]
    silos: List[Dict[str, Any]]
    output_parsers: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]
    mcp_configs: List[Dict[str, Any]]
    skills: List[Dict[str, Any]]
    middlewares: List[Dict[str, Any]] = []
    marketplace_visibility: Optional[str] = None
    marketplace_profile: Optional[Dict[str, Any]] = None
    is_frozen: bool = False

    model_config = ConfigDict(from_attributes=True)


class CreateUpdateAgentSchema(BaseModel):
    """Schema for creating or updating an agent"""
    name: str
    description: Optional[str] = ""
    system_prompt: Optional[str] = ""
    prompt_template: Optional[str] = ""
    type: str = "agent"  # "agent", "ocr_agent"
    is_tool: bool = False
    has_memory: bool = False
    enable_code_interpreter: bool = False
    server_tools: Optional[List[str]] = []
    memory_max_messages: Optional[int] = 20
    memory_max_tokens: Optional[int] = 4000
    memory_summarize_threshold: Optional[int] = DEFAULT_MEMORY_SUMMARIZE_THRESHOLD
    service_id: Optional[int] = None
    silo_id: Optional[int] = None
    output_parser_id: Optional[int] = None
    temperature: Optional[float] = DEFAULT_AGENT_TEMPERATURE
    tool_ids: Optional[List[int]] = []
    mcp_config_ids: Optional[List[int]] = []
    skill_ids: Optional[List[int]] = []
    retrieval_config: Optional[RetrievalConfig] = None
    middleware_ids: Optional[List[int]] = []
    # OCR-specific fields
    vision_service_id: Optional[int] = None
    vision_system_prompt: Optional[str] = None
    text_system_prompt: Optional[str] = None


class UpdatePromptSchema(BaseModel):
    """Schema for updating agent prompts"""
    type: str  # "system" or "template"
    prompt: str


# ==================== PUBLIC API SCHEMAS ====================

class PublicAgentSchema(BaseModel):
    """Public agent schema for API responses"""
    model_config = ConfigDict(from_attributes=True)
    
    agent_id: int
    name: str
    description: Optional[str] = None
    type: str
    status: Optional[str] = None
    is_tool: bool
    has_memory: Optional[bool] = None
    create_date: Optional[datetime] = None
    request_count: int


class PublicAgentDetailSchema(BaseModel):
    """Detailed public agent schema for API responses"""
    model_config = ConfigDict(from_attributes=True)
    
    agent_id: int
    name: str
    description: Optional[str] = None
    type: str
    status: Optional[str] = None
    is_tool: bool
    has_memory: Optional[bool] = None
    memory_max_messages: Optional[int] = 20
    memory_max_tokens: Optional[int] = 4000
    memory_summarize_threshold: Optional[int] = DEFAULT_MEMORY_SUMMARIZE_THRESHOLD
    system_prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    create_date: Optional[datetime] = None
    request_count: int
    service_id: Optional[int] = None
    silo_id: Optional[int] = None
    output_parser_id: Optional[int] = None
    temperature: Optional[float] = DEFAULT_AGENT_TEMPERATURE
    retrieval_config: Optional[Dict[str, Any]] = None
    # OCR-specific fields
    vision_service_id: Optional[int] = None
    vision_system_prompt: Optional[str] = None
    text_system_prompt: Optional[str] = None


class CreateAgentRequestSchema(BaseModel):
    """Schema for creating a new agent via public API"""
    name: str
    description: Optional[str] = ""
    type: Literal["agent"] = "agent"
    is_tool: bool = False
    has_memory: bool = False
    memory_max_messages: Optional[int] = 20
    memory_max_tokens: Optional[int] = 4000
    memory_summarize_threshold: Optional[int] = DEFAULT_MEMORY_SUMMARIZE_THRESHOLD
    system_prompt: Optional[str] = ""
    prompt_template: Optional[str] = ""
    service_id: Optional[int] = None
    silo_id: Optional[int] = None
    output_parser_id: Optional[int] = None
    temperature: Optional[float] = DEFAULT_AGENT_TEMPERATURE
    tool_ids: Optional[List[int]] = []
    mcp_config_ids: Optional[List[int]] = []
    skill_ids: Optional[List[int]] = []
    retrieval_config: Optional[RetrievalConfig] = None


class CreateOCRAgentRequestSchema(BaseModel):
    """Schema for creating a new OCR agent via public API"""
    name: str
    description: Optional[str] = ""
    is_tool: bool = False
    has_memory: bool = False
    memory_max_messages: Optional[int] = 20
    memory_max_tokens: Optional[int] = 4000
    memory_summarize_threshold: Optional[int] = DEFAULT_MEMORY_SUMMARIZE_THRESHOLD
    service_id: Optional[int] = None
    vision_service_id: Optional[int] = None
    vision_system_prompt: Optional[str] = ""
    text_system_prompt: Optional[str] = ""
    output_parser_id: Optional[int] = None
    temperature: Optional[float] = DEFAULT_AGENT_TEMPERATURE
    tool_ids: Optional[List[int]] = []
    mcp_config_ids: Optional[List[int]] = []
    skill_ids: Optional[List[int]] = []


class UpdateAgentRequestSchema(BaseModel):
    """Schema for updating an existing agent via public API"""
    name: Optional[str] = None
    description: Optional[str] = None
    is_tool: Optional[bool] = None
    has_memory: Optional[bool] = None
    memory_max_messages: Optional[int] = None
    memory_max_tokens: Optional[int] = None
    memory_summarize_threshold: Optional[int] = None
    system_prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    service_id: Optional[int] = None
    silo_id: Optional[int] = None
    output_parser_id: Optional[int] = None
    temperature: Optional[float] = None
    tool_ids: Optional[List[int]] = None
    mcp_config_ids: Optional[List[int]] = None
    skill_ids: Optional[List[int]] = None
    retrieval_config: Optional[RetrievalConfig] = None


class UpdateOCRAgentRequestSchema(BaseModel):
    """Schema for updating an existing OCR agent via public API"""
    name: Optional[str] = None
    description: Optional[str] = None
    is_tool: Optional[bool] = None
    has_memory: Optional[bool] = None
    memory_max_messages: Optional[int] = None
    memory_max_tokens: Optional[int] = None
    memory_summarize_threshold: Optional[int] = None
    service_id: Optional[int] = None
    vision_service_id: Optional[int] = None
    vision_system_prompt: Optional[str] = None
    text_system_prompt: Optional[str] = None
    output_parser_id: Optional[int] = None
    temperature: Optional[float] = None
    tool_ids: Optional[List[int]] = None
    mcp_config_ids: Optional[List[int]] = None
    skill_ids: Optional[List[int]] = None


class PublicAgentsResponseSchema(BaseModel):
    """Multiple agents response for public API"""
    agents: List[PublicAgentSchema]


class PublicAgentResponseSchema(BaseModel):
    """Single agent response for public API"""
    agent: PublicAgentDetailSchema
