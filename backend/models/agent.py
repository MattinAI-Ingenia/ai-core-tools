import enum

from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, Table, DateTime, Float, Enum, JSON
from sqlalchemy.orm import relationship
from db.database import Base
from datetime import datetime


class MarketplaceVisibility(enum.Enum):
    UNPUBLISHED = "unpublished"
    PRIVATE = "private"
    PUBLIC = "public"

AGENT_ID = 'Agent.agent_id'

# Default temperature for agents
DEFAULT_AGENT_TEMPERATURE = 0.7

# Default memory summarize threshold (number of messages)
DEFAULT_MEMORY_SUMMARIZE_THRESHOLD = 20

# Default retrieval configuration (per-agent RAG behaviour)
DEFAULT_RETRIEVAL_SEARCH_TYPE = "similarity"  # similarity | mmr | similarity_score_threshold
DEFAULT_RETRIEVAL_K = 30                      # candidate documents fetched from the vector store
DEFAULT_RETRIEVAL_STRATEGY = "passthrough"    # passthrough | rerank


class AgentSkill(Base):
    """Association table for Agent-Skill many-to-many relationship"""
    __tablename__ = 'agent_skills'
    agent_id = Column(Integer, ForeignKey(AGENT_ID), primary_key=True)
    skill_id = Column(Integer, ForeignKey('Skill.skill_id'), primary_key=True)
    description = Column(Text, nullable=True)  # Description of how this skill is used

    agent = relationship('Agent', foreign_keys=[agent_id], back_populates='skill_associations')
    skill = relationship('Skill', foreign_keys=[skill_id])


class AgentMCP(Base):
    __tablename__ = 'agent_mcps'
    agent_id = Column(Integer, ForeignKey(AGENT_ID), primary_key=True)
    config_id = Column(Integer, ForeignKey('MCPConfig.config_id'), primary_key=True)
    description = Column(Text, nullable=True)  # Description of what this MCP is used for
    
    agent = relationship('Agent', foreign_keys=[agent_id], back_populates='mcp_associations')
    mcp = relationship('MCPConfig', foreign_keys=[config_id])

class AgentTool(Base):
    __tablename__ = 'agent_tools'
    agent_id = Column(Integer, ForeignKey(AGENT_ID), primary_key=True)
    tool_id = Column(Integer, ForeignKey(AGENT_ID), primary_key=True)
    description = Column(Text, nullable=True)  # Description of what this tool is used for
    
    # Add relationships to both the agent and the tool
    agent = relationship('Agent', foreign_keys=[agent_id], back_populates='tool_associations')
    tool = relationship('Agent', foreign_keys=[tool_id])

class Agent(Base):
    __tablename__ = 'Agent'
    agent_id = Column(Integer, primary_key=True)
    name = Column(String(255))
    description = Column(String(1000))
    create_date = Column(DateTime, default=datetime.now)
    system_prompt = Column(Text)
    prompt_template = Column(Text)
    type = Column(String(45), nullable=False, default='agent')
    status = Column(String(45))
    request_count = Column(Integer, default=0)
    is_tool = Column(Boolean, default=False)
    service_id = Column(Integer,
                        ForeignKey('AIService.service_id', ondelete='SET NULL'),
                        nullable=True)
    silo_id = Column(Integer,
                        ForeignKey('Silo.silo_id'),
                        nullable=True)
    app_id = Column(Integer,
                        ForeignKey('App.app_id'),
                        nullable=True)

    has_memory = Column(Boolean)
    enable_code_interpreter = Column(Boolean, default=False, nullable=False, server_default='false')
    server_tools = Column(JSON, default=list, nullable=False, server_default='[]')

    # Memory management via LangChain SummarizationMiddleware (when has_memory=True)
    memory_max_messages = Column(Integer, default=20, nullable=False)  # SummarizationMiddleware.keep=("messages", N) — messages to preserve after summarization
    memory_max_tokens = Column(Integer, default=4000, nullable=True)  # SummarizationMiddleware.trigger=("tokens", N) — token count that triggers summarization
    memory_summarize_threshold = Column(Integer, default=DEFAULT_MEMORY_SUMMARIZE_THRESHOLD, nullable=False)  # SummarizationMiddleware.trim_tokens_to_summarize — max tokens sent to summarizer LLM
    
    output_parser_id = Column(Integer,
                        ForeignKey('OutputParser.parser_id'),
                        nullable=True)
    temperature = Column(Float, default=DEFAULT_AGENT_TEMPERATURE, nullable=False)
    is_frozen = Column(Boolean, default=False, nullable=False)

    # Retrieval configuration (per-agent RAG behaviour).
    #
    # Two stable *selectors* are stored as typed columns:
    #   - retrieval_search_type: how the vector store is queried (Phase 1)
    #   - retrieval_strategy:    post-retrieval strategy applied (Phase 2)
    #
    # Every component-specific knob (k, top_n, similarity_threshold, and any
    # future search-method / strategy parameter such as a hybrid `alpha`) lives
    # in the single ``retrieval_params`` JSON column. Storing them there means
    # adding a new retrieval parameter never requires a schema migration. The
    # ``retrieval_k`` / ``retrieval_top_n`` properties below keep the historical
    # flat attribute access working transparently on top of the JSON payload.
    retrieval_search_type = Column(
        String(45),
        default=DEFAULT_RETRIEVAL_SEARCH_TYPE,
        nullable=False,
        server_default=DEFAULT_RETRIEVAL_SEARCH_TYPE,
    )  # how the vector store is queried (Phase 1)
    retrieval_strategy = Column(
        String(45),
        default=DEFAULT_RETRIEVAL_STRATEGY,
        nullable=False,
        server_default=DEFAULT_RETRIEVAL_STRATEGY,
    )  # post-retrieval strategy applied to candidates (Phase 2)
    retrieval_params = Column(
        JSON,
        default=dict,
        nullable=False,
        server_default='{}',
    )  # component-specific knobs: k, top_n, similarity_threshold, …

    marketplace_visibility = Column(
        Enum(MarketplaceVisibility),
        nullable=False,
        default=MarketplaceVisibility.UNPUBLISHED
    )

    ai_service = relationship('AIService',
                           foreign_keys=[service_id])

    silo = relationship('Silo',
                           back_populates='agents',
                           foreign_keys=[silo_id])

    app = relationship('App',
                           back_populates='agents',
                           foreign_keys=[app_id])
    
    output_parser = relationship('OutputParser',
                           foreign_keys=[output_parser_id])
    
    # Modified relationship to use back_populates instead of backref
    tool_associations = relationship('AgentTool',
                                   primaryjoin=(agent_id == AgentTool.agent_id),
                                   back_populates='agent')
    
    # Add new MCP relationship
    mcp_associations = relationship('AgentMCP',
                                  primaryjoin=(agent_id == AgentMCP.agent_id),
                                  back_populates='agent')

    # Add Skill relationship
    skill_associations = relationship('AgentSkill',
                                     primaryjoin=(agent_id == AgentSkill.agent_id),
                                     back_populates='agent')

    # Marketplace profile (1:1)
    marketplace_profile = relationship(
        'AgentMarketplaceProfile',
        back_populates='agent',
        uselist=False,
        cascade='all, delete-orphan'
    )

    # ---- Retrieval knob accessors (backed by the retrieval_params JSON) -------
    # These expose the most common knobs as flat attributes so callers (schemas,
    # services, retrieval tools) keep using ``agent.retrieval_k`` /
    # ``agent.retrieval_top_n`` unchanged while the values live inside the JSON
    # payload. Setters always reassign the dict so SQLAlchemy tracks the change.
    def _set_retrieval_param(self, key, value):
        params = dict(self.retrieval_params or {})
        if value is None:
            params.pop(key, None)
        else:
            params[key] = value
        self.retrieval_params = params

    @property
    def retrieval_k(self):
        return (self.retrieval_params or {}).get('k', DEFAULT_RETRIEVAL_K)

    @retrieval_k.setter
    def retrieval_k(self, value):
        self._set_retrieval_param('k', value)

    @property
    def retrieval_top_n(self):
        return (self.retrieval_params or {}).get('top_n')

    @retrieval_top_n.setter
    def retrieval_top_n(self, value):
        self._set_retrieval_param('top_n', value)

    __mapper_args__ = {
        'polymorphic_identity': 'agent',
        'polymorphic_on': type
    } 