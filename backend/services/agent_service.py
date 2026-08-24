from typing import Union, List, Dict, Any, Optional
from sqlalchemy.orm import Session
import config
from models.agent import Agent, AgentSkill, DEFAULT_AGENT_TEMPERATURE, DEFAULT_MEMORY_SUMMARIZE_THRESHOLD
from models.ocr_agent import OCRAgent
from models.skill import Skill
from schemas.agent_schemas import AgentListItemSchema, AgentDetailSchema
from repositories.agent_repository import AgentRepository
from repositories.skill_repository import SkillRepository
from repositories.middleware_repository import MiddlewareRepository
from models.middleware import AgentMiddleware
from utils.logger import get_logger

logger = get_logger(__name__)

LIGHTRAG_ROUTER_SKILL_NAME = "LightRAG Query Router"

ROUTER_SKILL_CONTENT = """# LightRAG Query Router

You have access to `retrieve_from_knowledge_base(query, mode)`.
Before every retrieval call, select the `mode` that best matches the CURRENT question.

**Re-derive the mode fresh every single time.** Never reuse the mode from a
previous tool call in this conversation just because it worked before — a
run of nine `hybrid` calls in a row does not make the tenth question
`hybrid` too. Judge each new question independently against the guide below.

---

## Mode Guide

### global
**Strategy:** Queries community-level summaries built from clusters of related
entities — captures high-level themes and patterns across the whole graph.

**Use when the question asks for synthesis, enumeration, or overview across the document:**
- Conclusions, recommendations, or takeaways: "What is the conclusion?", "What does the paper recommend?"
- Enumerations of a whole category: "What categories/types/kinds of X are there?", "List all the risks/causes/consequences"
- Thematic summaries: "What are the main topics?", "Summarize the overall approach to X"
- Trends or big-picture patterns: "What are the key trends?", "What is the central argument?"
- Questions with "overall", "in general", "across", "in total", "as a whole"

**Key rule:** Generic terms like "risk", "agency", "data source", "consequence" are NOT specific
named entities. A question built only around generic terms belongs to global, not local.

**But single-concept questions are NOT global:** "What is X and why does it matter?"
targets ONE concept (X) — that is local, even though "why it matters" sounds thematic.
Global requires the question to span the WHOLE document or MANY entities (conclusions,
enumerations of a full category, overall themes) — not the causes/effects of a single concept.

**Avoid when:** The question targets a single named person, organization, or specific technical term by name.

---

### local
**Strategy:** Traverses entity neighborhoods in the knowledge graph — direct
relationships, attributes, and co-occurring concepts around a specific node.

**Use when the question anchors to a specific proper noun or well-defined named concept:**
- "Who is [named person]?", "What does [named organization] do?"
- "What is [specific technical term] and how does it work?"
- "What is the role of [specific named entity] in X?"
- Questions that name a single entity and ask only about that entity

**Definitional / explanatory questions about ONE named concept are local — this is the most common case:**
- "What is X?", "Define X", "Explain X" where X is a domain concept or term
  (e.g. "What is concept drift?", "What is operationalization bias?", "What does 'model staleness' mean?")
- This stays local EVEN when the question also asks why that one concept matters,
  how it works, or what it causes/affects — as long as it centers on a single
  concept (e.g. "What is concept drift and why does it affect ML models?" → local).
  Asking about the causes/effects of one concept is NOT document-wide synthesis.

**Local beats naive for these:** a "what is / define / explain" question about a
concept needs the graph neighborhood, not a verbatim span — send it to local, not naive.

**Avoid when:** The question enumerates a whole category, compares two things, or
asks for patterns across the WHOLE document spanning many entities.

---

### mix
**Strategy:** Runs local + global + naive vector search — maximum coverage at
the cost of speed and token usage.

**Use when the question contains an explicit comparison or an explicit demand
for exhaustive/complete coverage — check this BEFORE hybrid, even if the
question also names a specific entity:**
- Explicit comparison between two or more things: "Compare X and Y", "X versus Y", "how does X differ from Y"
- Explicit exhaustive-coverage language: "List ALL...", "every...", "all the ways...", "a complete list of..."
- The user explicitly asks for a comprehensive or exhaustive answer

**Avoid when:** Response time matters, or the question is a single non-comparative ask.

---

### hybrid *(default)*
**Strategy:** Runs local + global in parallel and merges results — balances
entity precision with thematic breadth.

**Use when the question does NOT match mix above, and:**
- The question mixes a specific entity with broader context: "How does [X] relate to the broader strategy?"
- The question asks about a named concept AND its implications or relationships
- The question asks what **changed** about something, or what **consequences / impact / effects** a named entity or event had on something else — even if only one entity is named: "What changes occurred to [X]?", "What were the consequences of [X] for [Y]?", "How did [X] affect [Y]?"
- The question has a **causal or temporal chain**: cause → effect, before → after, event → downstream impact
- You are unsure which of local or global is more appropriate

**Default to hybrid when the question type is ambiguous and doesn't match any mode above.**

---

### naive
**Strategy:** Pure vector similarity — no graph traversal, no community summaries.
Fast and precise for direct lexical or semantic matches.

**Use when:**
- Verbatim lookups: "Who are the authors?", "What year was this published?", "Find the section about Y"
- The question is looking for a specific quoted phrase, number, or name

**naive is ONLY for surface facts** — a name, date, number, or quoted span you could
find by Ctrl+F. If the question asks to *define, explain, or understand* a concept
("What is X?", "What does X mean?"), that is **local**, not naive.

**Avoid when:** The answer depends on relationships between entities, on understanding a concept, or on thematic context.

---

## When to call the tool

**Default: always call `retrieve_from_knowledge_base` first.**
The knowledge base contains domain-specific information you do not have in your
training data. A question about a person, organization, concept, or event that
could plausibly be in the knowledge base MUST go through the tool — do not
answer from memory alone.

Only skip the tool when the question is unambiguously about general world
knowledge with zero domain specificity (e.g. "What is the capital of France?").
When in doubt, call the tool. An empty result is more honest than a hallucinated answer.

## Decision flowchart

Evaluate in order; take the FIRST match.

**Step 1 — Explicit comparison ("X vs Y", "compare A and B") or explicit demand for exhaustive/complete coverage ("list ALL", "every", "comprehensive")?**
  → Yes → mix (even if it also names a specific entity)

**Step 2 — Does the question center on ONE named concept, term, person, or entity — including "What is X?", "Define X", "Explain X", or "What is X and why does it matter / how does it work / what does it affect"?**
  → Yes, but it asks what CHANGED about X, what CONSEQUENCES/IMPACT/EFFECTS X had, or how X affected something else → hybrid (causal/temporal chains always need both local + global)
  → Yes, and it stays focused on that one thing with no causal/temporal dimension → local
  → Yes, but it also pulls in document-wide themes or several other entities → hybrid

**Step 3 — Is the answer a surface-level verbatim fact (a name, date, number, or quoted phrase you could Ctrl+F)?**
  → Yes → naive

**Step 4 — Does the question ask for document-wide synthesis, enumeration of a whole category, or overall themes spanning many entities (no single anchor concept)?**
  → Yes → global

**Otherwise → hybrid**
"""


def _serialize_marketplace_profile(profile) -> Optional[Dict[str, Any]]:
    """Serialize an AgentMarketplaceProfile to a dict for schema response."""
    if not profile:
        return None
    published_at = profile.published_at.isoformat() if profile.published_at else None
    updated_at = profile.updated_at.isoformat() if profile.updated_at else None
    return {
        "id": profile.id,
        "agent_id": profile.agent_id,
        "display_name": profile.display_name,
        "short_description": profile.short_description,
        "long_description": profile.long_description,
        "category": profile.category,
        "tags": profile.tags,
        "icon_url": profile.icon_url,
        "cover_image_url": profile.cover_image_url,
        "published_at": published_at,
        "updated_at": updated_at,
    }

class AgentService:

    def get_agents_list(self, db: Session, app_id: int) -> List[AgentListItemSchema]:
        """Get list of agents with AI service details for display"""
        agents = AgentRepository.get_by_app_id(db, app_id)
        
        # Get AI services for this app
        ai_services_dict = AgentRepository.get_ai_services_dict_by_app_id(db, app_id)
        
        result = []
        for agent in agents:
            # Get AI service details if agent has one
            ai_service_info = None
            if hasattr(agent, 'service_id') and agent.service_id and agent.service_id in ai_services_dict:
                ai_service_info = ai_services_dict[agent.service_id]
            
            result.append(AgentListItemSchema(
                agent_id=agent.agent_id,
                name=agent.name,
                description=getattr(agent, 'description', None),
                type=agent.type or "agent",
                is_tool=agent.is_tool or False,
                created_at=agent.create_date,
                request_count=agent.request_count or 0,
                service_id=getattr(agent, 'service_id', None),
                ai_service=ai_service_info,
                marketplace_visibility=(
                    agent.marketplace_visibility.value
                    if hasattr(agent, 'marketplace_visibility') and agent.marketplace_visibility
                    else None
                ),
            ))
        
        return result

    def get_agents(self, db: Session, app_id: int) -> List[Agent]:
        """Get raw agent objects"""
        return AgentRepository.get_by_app_id(db, app_id)

    def get_tool_agents(self, db: Session, app_id: int, exclude_agent_id: int = None) -> List[Agent]:
        """Get agents that are marked as tools"""
        return AgentRepository.get_tool_agents_by_app_id(db, app_id, exclude_agent_id)

    def get_agent_detail(self, db: Session, app_id: int, agent_id: int) -> Optional[AgentDetailSchema]:
        """Get detailed agent information with form data for editing"""
        
        # Get agent details
        agent = self._get_agent_for_detail(db, agent_id)
        if agent_id != 0 and not agent:
            return None
        
        # Get form data for dropdowns
        form_data = self._get_form_data(db, app_id, agent_id)
        
        # Get agent associations
        associations = self._get_agent_associations(db, agent_id)
        
        # Get related information
        silo_info = self._get_silo_info(db, agent) if agent_id != 0 else None
        output_parser_info = self._get_output_parser_info(db, agent) if agent_id != 0 else None
        lightrag_query_mode = getattr(agent, 'lightrag_query_mode', None)

        return AgentDetailSchema(
            agent_id=agent.agent_id,
            name=agent.name or "",
            description=getattr(agent, 'description', '') or "",
            system_prompt=getattr(agent, 'system_prompt', '') or "",
            prompt_template=getattr(agent, 'prompt_template', '') or "",
            type=agent.type or "agent",
            is_tool=agent.is_tool or False,
            has_memory=getattr(agent, 'has_memory', False) or False,
            enable_code_interpreter=getattr(agent, 'enable_code_interpreter', False) or False,
            server_tools=getattr(agent, 'server_tools', None) or [],
            memory_max_messages=getattr(agent, 'memory_max_messages', 20) or 20,
            memory_max_tokens=getattr(agent, 'memory_max_tokens', 4000),
            memory_summarize_threshold=getattr(agent, 'memory_summarize_threshold', DEFAULT_MEMORY_SUMMARIZE_THRESHOLD) or DEFAULT_MEMORY_SUMMARIZE_THRESHOLD,
            service_id=getattr(agent, 'service_id', None),
            silo_id=getattr(agent, 'silo_id', None),
            output_parser_id=getattr(agent, 'output_parser_id', None),
            temperature=agent.temperature if agent.temperature is not None else DEFAULT_AGENT_TEMPERATURE,
            tool_ids=associations.get('tool_ids', []),
            mcp_config_ids=associations.get('mcp_ids', []),
            skill_ids=associations.get('skill_ids', []),
            lightrag_query_mode=lightrag_query_mode,
            middleware_ids=associations.get('middleware_ids', []),
            created_at=agent.create_date,
            request_count=getattr(agent, 'request_count', 0) or 0,
            # OCR-specific fields
            vision_service_id=getattr(agent, 'vision_service_id', None),
            vision_system_prompt=getattr(agent, 'vision_system_prompt', None),
            text_system_prompt=getattr(agent, 'text_system_prompt', None),
            # Related information
            silo=silo_info,
            output_parser=output_parser_info,
            # Form data
            ai_services=form_data.get('ai_services', []),
            silos=form_data.get('silos', []),
            output_parsers=form_data.get('output_parsers', []),
            tools=form_data.get('tools', []),
            mcp_configs=form_data.get('mcp_configs', []),
            skills=form_data.get('skills', []),
            middlewares=form_data.get('middlewares', []),
            # Marketplace
            marketplace_visibility=(
                agent.marketplace_visibility.value
                if hasattr(agent, 'marketplace_visibility') and agent.marketplace_visibility
                else None
            ),
            marketplace_profile=_serialize_marketplace_profile(
                getattr(agent, 'marketplace_profile', None)
            ),
            # RAG retrieval config (step_008)
            rag_k=getattr(agent, 'rag_k', None) if isinstance(getattr(agent, 'rag_k', None), (int, type(None))) else None,
            rag_k_mode=getattr(agent, 'rag_k_mode', None) if isinstance(getattr(agent, 'rag_k_mode', None), (str, type(None))) else None,
            rag_search_type=getattr(agent, 'rag_search_type', None) if isinstance(getattr(agent, 'rag_search_type', None), (str, type(None))) else None,
            rag_score_threshold=getattr(agent, 'rag_score_threshold', None) if isinstance(getattr(agent, 'rag_score_threshold', None), (float, int, type(None))) else None,
            rag_max_retrieval_calls=getattr(agent, 'rag_max_retrieval_calls', None) if isinstance(getattr(agent, 'rag_max_retrieval_calls', None), (int, type(None))) else None,
            rag_fixed_filters=getattr(agent, 'rag_fixed_filters', None) if isinstance(getattr(agent, 'rag_fixed_filters', None), (list, type(None))) else None,
            rag_chunk_top_k=getattr(agent, 'rag_chunk_top_k', None) if isinstance(getattr(agent, 'rag_chunk_top_k', None), (int, type(None))) else None,
            rag_chunk_top_k_deployment_default=config.LIGHTRAG_CHUNK_TOP_K_DEFAULT,
        )

    def _get_agent_for_detail(self, db: Session, agent_id: int):
        """Get agent for detail view"""
        if agent_id == 0:
            # New agent
            return type('Agent', (), {
                'agent_id': 0, 'name': '', 'system_prompt': '', 'prompt_template': '',
                'type': 'agent', 'is_tool': False, 'create_date': None, 'request_count': 0,
                'temperature': DEFAULT_AGENT_TEMPERATURE,
                # Mirror _update_normal_agent's new-agent defaults so the create
                # form shows the actual configured values, not a guess.
                'rag_k': config.AGENT_DEFAULT_RAG_K, 'rag_k_mode': 'fixed',
                'rag_max_retrieval_calls': 4, 'rag_chunk_top_k': None,
                'rag_chunk_top_k_deployment_default': config.LIGHTRAG_CHUNK_TOP_K_DEFAULT,
            })()
        else:
            # Existing agent - determine if it's OCR agent or regular agent
            agent = self.get_agent(db, agent_id)
            if not agent:
                return None
            
            # If it's an OCR agent, get the OCR-specific data
            if agent.type == 'ocr_agent':
                agent = self.get_agent(db, agent_id, 'ocr')
            return agent

    def _get_form_data(self, db: Session, app_id: int, agent_id: int) -> Dict[str, List]:
        """Get form data for dropdowns"""
        return AgentRepository.get_form_data_for_agent(db, app_id, agent_id)

    def _get_agent_associations(self, db: Session, agent_id: int) -> Dict[str, List]:
        """Get agent's current associations"""
        return AgentRepository.get_agent_associations_dict(db, agent_id)

    def _get_silo_info(self, db: Session, agent) -> Optional[Dict[str, Any]]:
        """Get silo information if agent has one"""
        if not hasattr(agent, 'silo_id') or not agent.silo_id:
            return None
        
        return AgentRepository.get_silo_with_metadata_definition(db, agent.silo_id)

    def _get_output_parser_info(self, db: Session, agent) -> Optional[Dict[str, Any]]:
        """Get output parser information if agent has one"""
        if not hasattr(agent, 'output_parser_id') or not agent.output_parser_id:
            return None
        
        return AgentRepository.get_output_parser_info(db, agent.output_parser_id)

    def get_agent(self, db: Session, agent_id: int, agent_type: str = 'basic') -> Union[Agent, OCRAgent]:
        """Get agent by ID and type"""
        return AgentRepository.get_agent_by_id_and_type(db, agent_id, agent_type)
    
    def create_or_update_agent(self, db: Session, agent_data: dict, agent_type: str, user_id: int = None) -> int:
        """Create or update agent"""
        agent_id = agent_data.get('agent_id')

        # If agent_id is 0, treat it as a new agent
        if agent_id == 0:
            agent_id = None

        agent = AgentRepository.get_agent_by_id_and_type(db, agent_id, agent_type) if agent_id else None

        if not agent:
            # Enforce per-app agent limit before creation (SaaS mode only)
            app_id = agent_data.get('app_id')
            if app_id:
                from services.tier_enforcement_service import TierEnforcementService
                TierEnforcementService.check_resource_limit(db, app_id, 'agents')

            # Create the appropriate agent instance based on type
            if agent_type == 'ocr_agent':
                agent = OCRAgent()
            else:
                agent = Agent()

        # Validate rag_fixed_filters fields against the silo's metadata_definition.
        # Fixed filters are only meaningful with a silo; reject them otherwise so a
        # caller never persists dead, never-applied scoping config.
        raw_fixed_filters = agent_data.get('rag_fixed_filters')
        if raw_fixed_filters:
            silo_id = agent_data.get('silo_id') or getattr(agent, 'silo_id', None)
            if not silo_id:
                raise ValueError(
                    "rag_fixed_filters can only be set on an agent that has a silo"
                )
            from tools.vector_stores.metadata_filters import SYSTEM_METADATA_FIELDS
            silo_info = AgentRepository.get_silo_with_metadata_definition(db, silo_id)
            metadata_def = silo_info.get('metadata_definition') if silo_info else None
            # With no metadata_definition only the system fields are filterable.
            declared_fields = frozenset(
                f['name'] for f in ((metadata_def or {}).get('fields') or [])
                if isinstance(f, dict) and f.get('name')
            ) | SYSTEM_METADATA_FIELDS
            unknown = [
                c['field'] for c in raw_fixed_filters
                if isinstance(c, dict) and c.get('field') not in declared_fields
            ]
            if unknown:
                raise ValueError(
                    f"rag_fixed_filters references unknown metadata fields: {unknown}. "
                    f"Allowed fields: {sorted(declared_fields)}"
                )

        update_method = self._update_normal_agent
        update_method(agent, agent_data)

        # Threshold search needs a threshold value, else it degrades to plain similarity at
        # retrieval. Checked on the merged state (the schema can't see the stored value on a
        # partial update).
        if agent.rag_search_type == 'similarity_score_threshold' and agent.rag_score_threshold is None:
            raise ValueError(
                "rag_score_threshold is required when rag_search_type is "
                "'similarity_score_threshold'"
            )

        # Set type only if it's not already set (OCRAgent sets it in __init__)
        if not hasattr(agent, 'type') or agent.type is None:
            agent.type = agent_type
        
        # Use repository to save the agent
        if agent.agent_id:
            agent = AgentRepository.update(db, agent)
        else:
            agent = AgentRepository.create(db, agent)
        
        # Return the agent ID
        return agent.agent_id


    
    def _update_normal_agent(self, agent: Agent, data: dict):
        """Update agent fields"""
        agent.name = data['name']
        agent.description = data.get('description', '')  # Ensure it's not None
        agent.system_prompt = data.get('system_prompt')
        agent.prompt_template = data.get('prompt_template')
        agent.status = data.get('status')
        agent.service_id = data.get('service_id') or None
        agent.app_id = data['app_id']
        agent.silo_id = data.get('silo_id') or None
        # Handle has_memory field - can be boolean from API or 'on' from form
        has_memory_value = data.get('has_memory')
        if isinstance(has_memory_value, bool):
            agent.has_memory = has_memory_value
        else:
            agent.has_memory = has_memory_value == 'on'

        enable_ci_value = data.get('enable_code_interpreter', False)
        agent.enable_code_interpreter = bool(enable_ci_value)

        agent.server_tools = data.get('server_tools') or []

        # Memory management fields
        if data.get('memory_max_messages') is not None:
            agent.memory_max_messages = data['memory_max_messages']
        if data.get('memory_max_tokens') is not None:
            agent.memory_max_tokens = data['memory_max_tokens']
        if data.get('memory_summarize_threshold') is not None:
            agent.memory_summarize_threshold = data['memory_summarize_threshold']

        agent.output_parser_id = data.get('output_parser_id') or None

        # Handle temperature field - default to DEFAULT_AGENT_TEMPERATURE if not provided
        agent.temperature = data.get('temperature', DEFAULT_AGENT_TEMPERATURE)

        # OCR-specific fields (only set if the agent is an OCRAgent instance)
        if isinstance(agent, OCRAgent):
            agent.vision_service_id = data.get('vision_service_id')
            agent.vision_system_prompt = data.get('vision_system_prompt')
            agent.text_system_prompt = data.get('text_system_prompt')

        agent.lightrag_query_mode = data.get('lightrag_query_mode')

        # Handle is_tool field - can be boolean from API or 'on' from form
        is_tool_value = data.get('is_tool')
        if isinstance(is_tool_value, bool):
            agent.is_tool = is_tool_value
        else:
            agent.is_tool = is_tool_value == 'on'

        # RAG retrieval config (step_008).
        # New agents: apply opinionated defaults (k=AGENT_DEFAULT_RAG_K env, max_calls=4)
        # when caller omits the field. Updates: only touch the column when the
        # caller explicitly supplies a value.
        is_new = not getattr(agent, 'agent_id', None)

        if data.get('rag_k') is not None:
            agent.rag_k = data['rag_k']
        elif is_new:
            agent.rag_k = config.AGENT_DEFAULT_RAG_K

        if data.get('rag_max_retrieval_calls') is not None:
            agent.rag_max_retrieval_calls = data['rag_max_retrieval_calls']
        elif is_new:
            agent.rag_max_retrieval_calls = 4

        if data.get('rag_search_type') is not None:
            agent.rag_search_type = data['rag_search_type']

        if data.get('rag_k_mode') is not None:
            agent.rag_k_mode = data['rag_k_mode']
        elif is_new:
            agent.rag_k_mode = 'fixed'

        # score_threshold, fixed_filters and chunk_top_k: clear-able via explicit
        # None/empty. rag_chunk_top_k has no forced default even for new
        # agents — None deliberately means "use the global CHUNK_TOP_K env var".
        if 'rag_score_threshold' in data:
            agent.rag_score_threshold = data['rag_score_threshold']
        if 'rag_fixed_filters' in data:
            agent.rag_fixed_filters = data['rag_fixed_filters']
        if 'rag_chunk_top_k' in data:
            agent.rag_chunk_top_k = data['rag_chunk_top_k']

    def update_agent_tools(self, db: Session, agent_id: int, tool_ids: list, form_data: dict = None):
        """Update agent tools associations"""
        # Get the agent
        agent = AgentRepository.get_by_id(db, agent_id)
        if not agent:
            return
        
        # Get existing tool associations
        existing_tools = {assoc.tool_id: assoc for assoc in AgentRepository.get_agent_tool_associations(db, agent_id)}
        
        # Convert tool_ids to set of integers and filter out non-tool agents
        valid_tool_ids = set(AgentRepository.get_valid_tool_ids(db, [int(id) for id in tool_ids if id]))
        
        # Remove associations that are no longer needed
        for tool_id in existing_tools.keys():
            if tool_id not in valid_tool_ids:
                AgentRepository.delete_agent_tool_association(db, existing_tools[tool_id])
        
        # Update or create associations
        for tool_id in valid_tool_ids:
            description = form_data.get(f'tool_description_{tool_id}') if form_data else None
            
            if tool_id in existing_tools:
                # Update existing association
                existing_tools[tool_id].description = description
                db.add(existing_tools[tool_id])
            else:
                # Create new association
                AgentRepository.create_agent_tool_association(db, agent_id, tool_id, description)
        
        db.commit()
    
    def update_agent_mcps(self, db: Session, agent_id: int, mcp_ids: list, form_data: dict = None):
        """Update agent MCP associations"""
        # Get the agent
        agent = AgentRepository.get_by_id(db, agent_id)
        if not agent:
            return
        
        # Convert mcp_ids to list if it's not already
        if isinstance(mcp_ids, str):
            mcp_ids = [mcp_ids]
        elif not isinstance(mcp_ids, list):
            mcp_ids = []

        # Get existing MCP associations
        existing_mcps = {assoc.config_id: assoc for assoc in AgentRepository.get_agent_mcp_associations(db, agent_id)}
        
        # Convert mcp_ids to set of integers
        valid_mcp_ids = {int(id) for id in mcp_ids if id}
        
        # Remove associations that are no longer needed
        for mcp_id in existing_mcps.keys():
            if mcp_id not in valid_mcp_ids:
                AgentRepository.delete_agent_mcp_association(db, existing_mcps[mcp_id])
        
        # Update or create associations
        for mcp_id in valid_mcp_ids:
            description = form_data.get(f'mcp_description_{mcp_id}') if form_data else None
            
            if mcp_id in existing_mcps:
                # Update existing association
                existing_mcps[mcp_id].description = description
                db.add(existing_mcps[mcp_id])
            else:
                # Create new association
                AgentRepository.create_agent_mcp_association(db, agent_id, mcp_id, description)
        
        db.commit()

    def update_agent_skills(self, db: Session, agent_id: int, skill_ids: list, form_data: dict = None):
        """Update agent skill associations"""
        # Get the agent
        agent = AgentRepository.get_by_id(db, agent_id)
        if not agent:
            return

        # Convert skill_ids to list if it's not already
        if isinstance(skill_ids, str):
            skill_ids = [skill_ids]
        elif not isinstance(skill_ids, list):
            skill_ids = []

        # Get existing skill associations
        existing_skills = {assoc.skill_id: assoc for assoc in AgentRepository.get_agent_skill_associations(db, agent_id)}

        # Convert skill_ids to set of integers
        requested_skill_ids = {int(id) for id in skill_ids if id}
        
        # Validate that skills exist and belong to the same app as the agent
        # This prevents cross-app associations and FK errors
        valid_skill_ids = SkillRepository.get_valid_skill_ids_for_app(db, requested_skill_ids, agent.app_id)

        # Remove associations that are no longer needed
        for skill_id in existing_skills.keys():
            if skill_id not in valid_skill_ids:
                AgentRepository.delete_agent_skill_association(db, existing_skills[skill_id])

        # Update or create associations
        for skill_id in valid_skill_ids:
            description = form_data.get(f'skill_description_{skill_id}') if form_data else None

            if skill_id in existing_skills:
                # Update existing association
                existing_skills[skill_id].description = description
                db.add(existing_skills[skill_id])
            else:
                # Create new association
                AgentRepository.create_agent_skill_association(db, agent_id, skill_id, description)

        db.commit()

    def ensure_lightrag_router_skill(self, db: Session, app_id: int) -> Skill:
        """Find or create the LightRAG Query Router skill for this app."""
        existing = db.query(Skill).filter(
            Skill.app_id == app_id,
            Skill.name == LIGHTRAG_ROUTER_SKILL_NAME,
        ).first()
        if existing:
            if existing.content != ROUTER_SKILL_CONTENT:
                existing.content = ROUTER_SKILL_CONTENT
                db.commit()
            return existing
        skill = Skill(
            name=LIGHTRAG_ROUTER_SKILL_NAME,
            description="Automatically selects the optimal LightRAG query mode (local/global/hybrid/mix/naive) per question.",
            content=ROUTER_SKILL_CONTENT,
            app_id=app_id,
        )
        db.add(skill)
        db.commit()
        db.refresh(skill)
        return skill

    def attach_skill_to_agent(self, db: Session, agent_id: int, skill_id: int) -> None:
        """Attach a skill to an agent if not already attached."""
        existing = db.query(AgentSkill).filter(
            AgentSkill.agent_id == agent_id,
            AgentSkill.skill_id == skill_id,
        ).first()
        if not existing:
            db.add(AgentSkill(agent_id=agent_id, skill_id=skill_id))
            db.commit()

    def cleanup_lightrag_router_skill(self, db: Session, app_id: int, agent_id: int) -> None:
        """Detach the routing skill from this agent; delete from app if orphaned."""
        skill = db.query(Skill).filter(
            Skill.app_id == app_id,
            Skill.name == LIGHTRAG_ROUTER_SKILL_NAME,
        ).first()
        if not skill:
            return
        db.query(AgentSkill).filter(
            AgentSkill.agent_id == agent_id,
            AgentSkill.skill_id == skill.skill_id,
        ).delete(synchronize_session=False)
        remaining = db.query(AgentSkill).filter(AgentSkill.skill_id == skill.skill_id).count()
        if remaining == 0:
            db.delete(skill)
        db.commit()

    def update_agent_middlewares(self, db: Session, agent_id: int, middleware_ids: list):
        """Update agent middleware associations"""
        agent = AgentRepository.get_by_id(db, agent_id)
        if not agent:
            return

        if not isinstance(middleware_ids, list):
            middleware_ids = []

        # Get existing middleware associations
        existing = {assoc.middleware_id: assoc for assoc in db.query(AgentMiddleware).filter(AgentMiddleware.agent_id == agent_id).all()}

        # Validate middleware IDs
        requested = {int(mid) for mid in middleware_ids if mid}
        valid_ids = MiddlewareRepository.get_valid_middleware_ids_for_app(db, requested, agent.app_id)

        # Remove stale
        for mid in existing:
            if mid not in valid_ids:
                db.delete(existing[mid])

        # Add new
        for mid in valid_ids:
            if mid not in existing:
                assoc = AgentMiddleware(agent_id=agent_id, middleware_id=mid)
                db.add(assoc)

        # Auto-enable memory when a human_in_the_loop middleware is associated,
        # because HumanInTheLoopMiddleware requires a LangGraph checkpointer.
        if valid_ids:
            from models.middleware import Middleware, MiddlewareType
            hitl_exists = db.query(Middleware).filter(
                Middleware.middleware_id.in_(valid_ids),
                Middleware.middleware_type == MiddlewareType.HUMAN_IN_THE_LOOP,
            ).first()
            if hitl_exists and not agent.has_memory:
                agent.has_memory = True
                db.add(agent)

        db.commit()

    def delete_agent(self, db: Session, agent_id: int) -> bool:
        """Delete agent"""
        return AgentRepository.delete_by_id(db, agent_id)

    def _remove_tool_references(self, db: Session, tool_id: int):
        """Remove all tool associations where this agent is used as a tool"""
        AgentRepository.remove_tool_references(db, tool_id)

    def update_agent_prompt(self, db: Session, agent_id: int, prompt_type: str, prompt: str) -> bool:
        """Update agent prompt (system or template)"""
        agent = AgentRepository.get_agent_by_id_and_type(db, agent_id)
        if not agent:
            return False
        
        # Update the appropriate prompt field directly
        if prompt_type == 'system':
            agent.system_prompt = prompt
        elif prompt_type == 'template':
            agent.prompt_template = prompt
        else:
            return False
        
        # Save the changes
        db.commit()
        return True

    def get_agent_playground_data(self, db: Session, agent_id: int) -> Optional[Dict[str, Any]]:
        """Get agent playground data"""
        agent = AgentRepository.get_agent_by_id_and_type(db, agent_id)
        if not agent:
            return None
        
        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "type": agent.type,
            "playground_url": f"/playground/{agent_id}"
        }

    def get_agent_analytics(self, db: Session, agent_id: int) -> Optional[Dict[str, Any]]:
        """Get agent analytics data"""
        agent = AgentRepository.get_agent_by_id_and_type(db, agent_id)
        if not agent:
            return None
        
        # Return analytics data with actual implementation placeholder
        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "request_count": agent.request_count or 0,
            "analytics_data": "Analytics feature coming soon"
        } 