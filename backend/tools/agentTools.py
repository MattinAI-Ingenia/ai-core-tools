from langchain.messages import HumanMessage, SystemMessage, AnyMessage
from langchain.agents import create_agent as create_langchain_agent, AgentState
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitMiddleware
from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from langchain.agents.middleware.pii import PIIMiddleware
from langchain.agents.middleware import HumanInTheLoopMiddleware, AgentMiddleware
from utils.schema_utils import sanitize_identifier, ensure_json_schema_types
from models.agent import Agent, DEFAULT_AGENT_TEMPERATURE, DEFAULT_MEMORY_SUMMARIZE_THRESHOLD
from models.silo import Silo
from langchain.tools import BaseTool, tool
from tools.outputParserTools import get_parser_model_by_id
from tools.aiServiceTools import get_llm, get_output_parser
from tools.ai.fileTools import fetch_file_in_base64
from tools.ai.workspaceTools import create_download_url_tool
from typing import Any, Optional, Dict, List, Tuple, Type
from pydantic import BaseModel, Field
import types as _types
from services.silo_service import SiloService
from db.database import SessionLocal
from langchain_mcp_adapters.client import MultiServerMCPClient
from services.agent_cache_service import CheckpointerCacheService
from langchain_core.documents import Document
from langchain_core.callbacks import UsageMetadataCallbackHandler
import langsmith as ls
from langchain_core.tools import StructuredTool
import asyncio
import json
import os
import base64
import mimetypes
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlparse
from utils.logger import get_logger
from utils.config import get_app_config
from utils.mcp_auth_utils import prepare_mcp_headers, get_user_token_from_context
from utils.mcp_ssl_utils import inject_ssl_config
from tools.skill_tools import create_skill_loader_tool, generate_skills_system_prompt_section
from services.agent_service import LIGHTRAG_ROUTER_SKILL_NAME, ROUTER_SKILL_CONTENT
from tools.sandbox import (
    create_sandbox_builtin_tools,
    create_sandbox_repl_tools,
    resolve_provider,
)
from tools.sandbox.factory import SandboxProviderUnavailableError

logger = get_logger(__name__)

MCP_TOOLS_TIMEOUT = 10  # seconds to wait for MCP servers to respond


def _create_dynamic_lightrag_tool(
    silo: Silo, search_params=None, offset: Optional[List[int]] = None, lock: Optional[asyncio.Lock] = None,
):
    """Return a retrieve_from_knowledge_base(query, mode) LangChain tool for skill-routed agents."""
    silo_id = silo.silo_id
    VALID_MODES = {"local", "global", "hybrid", "mix", "naive"}
    # Shared across calls within the same turn — see _append_lightrag_citation_sources.
    # `offset` lets the caller pass the SAME counter used by a sibling tool
    # (e.g. _create_coverage_tool) so citations stay globally numbered when
    # both tools fire in one turn.
    _citation_offset: List[int] = offset if offset is not None else [0]
    # `lock` serializes this tool against a sibling LightRAG tool sharing the
    # same offset — see the call site in _resolve_and_build_retriever_tool for why.
    _lock = lock if lock is not None else asyncio.Lock()

    @tool(response_format="content_and_artifact")
    async def retrieve_from_knowledge_base(query: str, mode: str) -> tuple:
        """Search the domain knowledge base. Call this tool for any question about
        specific people, organizations, concepts, or events that may be in the
        knowledge base — do not answer from memory alone.

        Args:
            query: The search query to run against the knowledge base.
            mode: Retrieval strategy — one of: local, global, hybrid, mix, naive.
                  See the LightRAG Query Router skill for selection guidance.
        """
        resolved_mode = mode if mode in VALID_MODES else "hybrid"
        logger.info(
            "[skill-routed] route=%r query=%r mode_requested=%r mode_used=%r",
            resolved_mode, query, mode, resolved_mode,
        )
        # Serialized against the sibling coverage tool — see call site comment.
        async with _lock:
            retriever = SiloService.get_silo_retriever(
                silo_id,
                {**(search_params or {}), "lightrag_query_mode": resolved_mode},
            )
            docs = await retriever.ainvoke(query)
            if not docs:
                return "No relevant documents found.", []
            _INTERNAL_META = {"lightrag_raw_data", "lightrag_keywords"}
            parts = []
            for doc in docs:
                meta = {k: v for k, v in (doc.metadata or {}).items() if k not in _INTERNAL_META}
                metadata_str = json.dumps(meta, ensure_ascii=False) if meta else "{}"
                parts.append(f"Content: {doc.page_content}\nMetadata: {metadata_str}")
            # naive has no graph but still returns chunks — surface them so chunk chips
            # + inline citations work. The frontend hides the "Ver subgrafo" button when
            # there are no entities/relationships (LightRAGGraphBubble), so naive shows
            # chunks-only. ponytail: header still reads "Subgrafo · 0 entidades…"; relabel
            # only if it bothers anyone.
            content = _append_lightrag_citation_sources("\n\n---\n\n".join(parts), docs, _citation_offset)
            return content, docs

    return retrieve_from_knowledge_base


def _create_coverage_tool(
    silo: Silo, app_id: int, offset: Optional[List[int]] = None, lock: Optional[asyncio.Lock] = None,
):
    """Return a list_documents_mentioning(term, doc) tool for skill-routed agents.

    Bypasses chunk_top_k entirely (see find_chunks_mentioning) — for
    membership/enumeration questions ("in which documents does X appear",
    "list every X in manual Y"), not for explaining or reasoning about X.
    """
    silo_id = silo.silo_id
    # `offset` lets the caller pass the SAME counter used by a sibling tool
    # (e.g. _create_dynamic_lightrag_tool) so citations stay globally numbered
    # when both tools fire in one turn.
    _citation_offset: List[int] = offset if offset is not None else [0]
    # `lock` serializes this tool against that sibling tool — see the call
    # site in _resolve_and_build_retriever_tool for why.
    _lock = lock if lock is not None else asyncio.Lock()

    @tool(response_format="content_and_artifact")
    async def list_documents_mentioning(term: Optional[str] = None, doc: Optional[str] = None) -> tuple:
        """Search EVERY document for literal mentions of a term — use this
        instead of retrieve_from_knowledge_base when the question asks to
        enumerate all documents where something appears, or list everything
        of some kind within one document. Do not use it to explain, compare,
        or reason about X — only to find where it is present.

        Args:
            term: the word or phrase to search for, verbatim. Omit it when the
                question has no single literal term to search for (e.g. "list
                every parameter in manual Z") — pass `doc` alone and every
                chunk of that document is returned, unfiltered.
            doc: a document code, or a product's commercial name (e.g.
                 "TERMAT"), to scope the search to just that document.
                 Required when `term` is omitted.
        """
        logger.info("[coverage] route='cobertura' term=%r doc=%r", term, doc)
        async with _lock:
            try:
                from services.silo_service import SiloService
                from db.database import SessionLocal

                resolved_doc_filter = None
                if doc:
                    db = SessionLocal()
                    try:
                        resource_ids = await asyncio.to_thread(
                            SiloService.resolve_document_by_name, app_id, doc, silo_id, db,
                        )
                    finally:
                        db.close()
                    if not resource_ids:
                        return f"No se pudo identificar el documento o producto '{doc}'.", []
                    # Every matching resource, not just the first: a name can
                    # legitimately resolve to several real documents (a product
                    # family sharing a name prefix), and picking one arbitrarily
                    # hides that the others were never searched.
                    resolved_doc_filter = resource_ids

                if not term and resolved_doc_filter is None:
                    return "Falta indicar un término a buscar o un documento al que acotar la búsqueda.", []

                search_term = term
                if term:
                    # The knowledge graph's entity names are the real on-page
                    # spelling ("cenicero" -> "Cenicero Compresor Automatico") —
                    # search every variant too, not just the literal term, since
                    # ILIKE alone misses phrasing/wording differences.
                    variants = await asyncio.to_thread(
                        SiloService.resolve_term_variants, silo_id, term,
                    )
                    if variants:
                        search_term = [term] + [v for v in variants if v.lower() != term.lower()]

                grouped, cap_hit = await asyncio.to_thread(
                    SiloService.find_chunks_mentioning, silo_id, search_term, resolved_doc_filter,
                )

                if not grouped:
                    abstain_term = term or f"el documento '{doc}'"
                    return f"No se encontró ningún documento que mencione {abstain_term}.", []
                chunks = []
                if term:
                    # Multi-document mode: one line per DOCUMENT is the useful
                    # enumeration (cobertura/G09-style — "in which documents does
                    # X appear"). Built deterministically, not left to the LLM: a
                    # synthesized answer over a long SOURCES block reliably drops
                    # entries (seen live — 32 documents found, 4 cited in prose).
                    # One line per document with its own citation link cannot be
                    # summarized away.
                    enumerated_lines = []
                    for resource_id, snippets in grouped.items():
                        file_path, content, page = snippets[0]
                        n = _citation_offset[0] + len(chunks) + 1
                        enumerated_lines.append(f"- {file_path} [{n}](cite://{n})")
                        chunks.append({"file_path": file_path, "content": content, "resource_id": int(resource_id), "page": page})
                    for resource_id, snippets in grouped.items():
                        for file_path, content, page in snippets[1:]:
                            chunks.append({"file_path": file_path, "content": content, "resource_id": int(resource_id), "page": page})
                    summary = f"{len(grouped)} documento(s) mencionan '{term}'."
                    instruction = (
                        "IMPORTANT: reproduce the list below in your answer EXACTLY as given, "
                        "one line per document, all of them — do not summarize, select a "
                        "subset, or omit any entry, even if there are many."
                    )
                    body = "\n".join(enumerated_lines)
                else:
                    # Whole-document mode: usually one document, but `doc` can
                    # resolve to several (an ambiguous/family name) — grouped may
                    # have more than one key. Either way a per-document list is
                    # useless here: what needs enumerating is inside the raw
                    # content itself (e.g. every parameter code in a service
                    # manual), spread across many chunks. That's genuine
                    # extraction, not something a pre-built list can shortcut; the
                    # only lever here is telling the model not to stop early.
                    for resource_id, snippets in grouped.items():
                        for file_path, content, page in snippets:
                            chunks.append({"file_path": file_path, "content": content, "resource_id": int(resource_id), "page": page})
                    doc_word = "documento" if len(grouped) == 1 else f"{len(grouped)} documentos"
                    summary = f"Contenido completo del/de los {doc_word} solicitado(s) ({doc}), en {len(chunks)} fragmento(s)."
                    instruction = (
                        "IMPORTANT: the sources below are the ENTIRE content of the requested "
                        "document(s), split across many fragments — read through ALL of them before "
                        "answering. Extract and list EVERY distinct instance of what was asked "
                        "(e.g. every parameter code and its value), not just the first few you see."
                    )
                    body = None
                page_content = f"{summary}\n\n{instruction}" + (f"\n\n{body}" if body else "")
                wrapper_doc = Document(
                    page_content=page_content,
                    metadata={"lightrag_raw_data": {"data": {"chunks": chunks}}},
                )
                content = _append_lightrag_citation_sources(
                    wrapper_doc.page_content, [wrapper_doc], _citation_offset,
                )
                if cap_hit:
                    content += (
                        "\n\nNota: la búsqueda encontró más resultados de los que se "
                        "pudieron procesar; puede haber documentos adicionales no listados."
                    )
                return content, [wrapper_doc]
            except Exception as exc:
                logger.error("[coverage] list_documents_mentioning failed", exc_info=True)
                return _SEARCH_ERROR_MSG, []

    return list_documents_mentioning


def _extract_mcp_root_causes(exc: BaseException) -> str:
    """Extract concise root-cause messages from (possibly nested) ExceptionGroups."""
    causes: list[str] = []
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            causes.append(_extract_mcp_root_causes(sub))
    else:
        causes.append(f"{type(exc).__name__}: {exc}")
    return "; ".join(causes)


_AUTH_QUERY_PARAMS = {
    "access_token",
    "api_key",
    "apikey",
    "api-key",
    "key",
    "tavilyapikey",
    "token",
}


def _url_contains_auth_credentials(url: str) -> bool:
    try:
        query_params = parse_qsl(urlparse(url).query, keep_blank_values=False)
    except ValueError:
        return False
    return any(name.lower() in _AUTH_QUERY_PARAMS for name, _value in query_params)


def _merge_mcp_auth_headers(
    server_name: str,
    server_config: Dict[str, Any],
    auth_headers: Dict[str, str],
) -> None:
    """Add Mattin auth headers without replacing MCP-specific credentials."""
    if 'url' not in server_config:
        return

    existing_headers = server_config.setdefault('headers', {})
    existing_header_names = {str(key).lower() for key in existing_headers}
    applied_headers = {}
    has_url_credentials = _url_contains_auth_credentials(str(server_config.get("url", "")))

    for header_name, header_value in auth_headers.items():
        normalized_header_name = header_name.lower()
        if normalized_header_name in existing_header_names:
            logger.info(
                "Preserving configured MCP header '%s' for server: %s",
                header_name,
                server_name,
            )
            continue
        if normalized_header_name == "authorization" and has_url_credentials:
            logger.info(
                "Skipping Mattin Authorization header for MCP server with URL credentials: %s",
                server_name,
            )
            continue
        applied_headers[header_name] = header_value

    if applied_headers:
        existing_headers.update(applied_headers)
        logger.info(f"Added auth headers to MCP server: {server_name}")


def _redact_mcp_connections(connections: Dict[str, Any]) -> Dict[str, Any]:
    redacted = {}
    sensitive_headers = {"authorization", "x-api-key", "api-key"}

    for server_name, server_config in connections.items():
        if not isinstance(server_config, dict):
            redacted[server_name] = server_config
            continue

        safe_config = dict(server_config)
        headers = safe_config.get("headers")
        if isinstance(headers, dict):
            safe_config["headers"] = {
                key: "<redacted>" if str(key).lower() in sensitive_headers else value
                for key, value in headers.items()
            }
        redacted[server_name] = safe_config

    return redacted


def _tool_name_for_language(language: str) -> str:
    return "python_repl" if language == "python" else f"{language}_repl"


def _sandbox_id_for_log(sandbox_handle: Any) -> str:
    sandbox_id = getattr(sandbox_handle, "sandbox_id_if_created", None)
    if sandbox_id:
        return sandbox_id
    is_materialized = getattr(sandbox_handle, "is_materialized", None)
    if callable(is_materialized) and not is_materialized():
        return "<lazy>"
    return getattr(sandbox_handle, "sandbox_id", "<unknown>")


class MCPClientManager:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MCPClientManager, cls).__new__(cls)
        return cls._instance

    async def get_client(self, agent: Agent = None, user_context: Optional[Dict] = None):
        """Get or create an MCP client for the given agent with authentication support.
        
        Args:
            agent: The agent to create the client for
            user_context: Optional user context containing authentication tokens
            
        Returns:
            MultiServerMCPClient or None
        """
        # Always create a new client for each agent execution to avoid ClosedResourceError
        # Don't use singleton pattern as the client lifecycle is tied to the agent execution
        if agent is not None:
            connections = {}
            for mcp_assoc in agent.mcp_associations:
                mcp_config = mcp_assoc.mcp
                try:
                    # Get the config from the database
                    connection_config = mcp_config.to_connection_dict()
                    if connection_config:
                        # Add authentication headers if user context is provided
                        if user_context:
                            auth_token = get_user_token_from_context(user_context)
                            if auth_token:
                                # Prepare headers for MCP server authentication
                                headers = prepare_mcp_headers(auth_token)

                                # Add headers to each connection in the config,
                                # preserving provider credentials already configured.
                                for server_name, server_config in connection_config.items():
                                    if isinstance(server_config, dict):
                                        _merge_mcp_auth_headers(server_name, server_config, headers)

                        connections.update(connection_config)
                except ValueError as e:
                    logger.error(f"Error configuring MCP {mcp_config.name}: {e}")
                    continue
                
            if connections:
                # Inject SSL configuration for connections that need it
                # Check each MCP config's ssl_verify setting
                for mcp_assoc in agent.mcp_associations:
                    mcp_cfg = mcp_assoc.mcp
                    ssl_verify = mcp_cfg.ssl_verify if mcp_cfg.ssl_verify is not None else True
                    if not ssl_verify:
                        cfg_dict = mcp_cfg.to_connection_dict()
                        for server_name in cfg_dict:
                            if server_name in connections:
                                inject_ssl_config({server_name: connections[server_name]}, ssl_verify=False)
                
                logger.info(
                    "Creating new MCP client with connections: %s",
                    _redact_mcp_connections(connections),
                )
                # Create a new client each time - don't reuse the singleton
                # As of langchain-mcp-adapters 0.1.0, MultiServerMCPClient cannot be used as a context manager
                client = MultiServerMCPClient(connections=connections)
                return client
            else:
                logger.warning("No valid MCP configurations found for agent")
                return None
                
        return None

    async def close(self):
        # As of langchain-mcp-adapters 0.1.0, MultiServerMCPClient doesn't need manual cleanup
        # The client is managed internally by the library
        if self._client is not None:
            self._client = None

def _build_summarization_llm_from_service(agent, ai_service_id: int):
    """Build a summarization LLM from a specific AIService ID.

    Looks up the AIService by ID within the agent's app and instantiates the
    appropriate LangChain chat model via :func:`tools.aiServiceTools.create_llm_from_service`.
    Supports all configured providers (OpenAI, Anthropic, MistralAI, Azure, Google, Custom).
    """
    from tools.aiServiceTools import create_llm_from_service

    if hasattr(agent, 'app') and agent.app and hasattr(agent.app, 'ai_services'):
        for svc in agent.app.ai_services:
            if svc.service_id == ai_service_id:
                try:
                    llm = create_llm_from_service(svc, temperature=0)
                    logger.info(f"Summarization using AIService id={ai_service_id} ({svc.name})")
                    return llm
                except Exception as e:
                    logger.warning(f"Failed to build summarization LLM from service {ai_service_id}: {e}")
                    return None

    logger.warning(f"AIService id={ai_service_id} not found in agent's app — using agent LLM")
    return None


async def create_agent(
    agent: Agent,
    search_params=None,
    session_id=None,
    user_context: Optional[Dict] = None,
    working_dir: Optional[str] = None,
    sandbox_handle: Optional[Any] = None,
    sandbox_provider: Optional[Any] = None,
    sandbox_session_key: Optional[str] = None,
    sandbox_session_service: Optional[Any] = None,
    attached_files: Optional[List[Dict]] = None,
    temp_silo_ids: Optional[List[int]] = None,
):
    """Create a new agent instance with cached checkpointer if memory is enabled.

    Args:
        agent: The agent to create
        search_params: Optional per-call retrieval overrides (highest priority)
        session_id: Optional session ID for memory-enabled agents (used to cache checkpointer)
        user_context: Optional user context containing authentication tokens for MCP
        working_dir: Optional per-conversation working directory
        sandbox_handle: Optional sandbox handle created during turn preparation
        sandbox_provider: Optional provider matching sandbox_handle
        sandbox_session_key: Optional key for sandbox active-use leasing
        sandbox_session_service: Optional service used for sandbox active-use leasing
        attached_files: Optional list of attached files to pass to the agent chain
        temp_silo_ids: Optional list of temporary silo IDs (e.g. playground media) to include as extra retrievers
    """
    # The router skill's mode-selection rules must be visible from the very
    # first message — inline it directly into the system prompt (see below) and
    # drop it from the generic on-demand skill list so it isn't offered twice.
    router_skill_active = (
        getattr(agent, "lightrag_query_mode", None) == "skill-routed"
        and any(
            getattr(a.skill, "name", None) == LIGHTRAG_ROUTER_SKILL_NAME
            for a in (getattr(agent, "skill_associations", None) or [])
            if a.skill
        )
    )
    other_skill_associations = [
        a for a in (getattr(agent, "skill_associations", None) or [])
        if getattr(a.skill, "name", None) != LIGHTRAG_ROUTER_SKILL_NAME
    ]

    llm = get_llm(agent)
    if llm is None:
        raise ValueError("No LLM found for agent")

    output_parser = get_output_parser(agent)
    format_instructions = ""
    pydantic_model = None

    if agent.output_parser_id is not None:
        try:
            pydantic_model = get_parser_model_by_id(agent.output_parser_id)
            format_instructions = output_parser.get_format_instructions()
            format_instructions = format_instructions.replace('{', '{{').replace('}', '}}')
        except Exception as e:
            logger.error(f"Error getting Pydantic model: {str(e)}")
            pydantic_model = None

    # Handle checkpointer management for memory-enabled agents
    checkpointer = None
    if agent.has_memory:
        # Use the session_id if provided, otherwise use "default"
        cache_session_id = session_id if session_id else "default"
        # Create the async PostgreSQL checkpointer in the current event loop
        # This ensures the checkpointer uses the same event loop as ainvoke()
        checkpointer = await CheckpointerCacheService.get_async_checkpointer()
        logger.info(f"Using async PostgreSQL checkpointer for agent {agent.agent_id} (session: {cache_session_id})")

    ci_provider_for_prompt = None
    ci_languages: list[str] = []
    if agent.enable_code_interpreter and working_dir:
        try:
            ci_provider_for_prompt = sandbox_provider or resolve_provider(agent)
            ci_languages = ci_provider_for_prompt.get_supported_languages()
        except SandboxProviderUnavailableError as exc:
            logger.warning(
                "Code interpreter unavailable for agent %s: %s", agent.agent_id, exc
            )
            ci_provider_for_prompt = None
            ci_languages = []

    # Build system prompt with optional skills section and format instructions
    # In LangChain v1, system_prompt is a static string passed to create_agent
    system_prompt_content = agent.system_prompt
    if router_skill_active:
        system_prompt_content = system_prompt_content + "\n\n" + ROUTER_SKILL_CONTENT
    # Inject current date to avoid need for a tool call
    current_date = datetime.now().strftime("%Y-%m-%d")
    system_prompt_content += f"\n\nToday's date is {current_date}."
    if other_skill_associations:
        skills_section = generate_skills_system_prompt_section(other_skill_associations)
        if skills_section:
            system_prompt_content = system_prompt_content + "\n" + skills_section

    if working_dir:
        system_prompt_content = (
            system_prompt_content
            + "\n\n<workspace>\n"
            + f"Working directory: {working_dir}\n"
            + "Workspace layout:\n"
            + "- input/: user-provided files. Treat these as source material.\n"
            + "- work/: scratch files, scripts, dependencies, extracted content, and intermediate data.\n"
            + "- output/: final files intended for the user to download.\n"
            + "User-uploaded files are available under input/; reference them as input/<filename>.\n"
            + "Use `download_url_to_workspace` to save any URL (generated image, PDF, report…) "
            + "to output/ so the user can download it from the files panel.\n"
            + "</workspace>"
        )

    if agent.enable_code_interpreter and working_dir and ci_provider_for_prompt is not None:
        _ci_languages = ci_languages
        _tool_names = ", ".join(f"`{_tool_name_for_language(lang)}`" for lang in _ci_languages)
        system_prompt_content = (
            system_prompt_content
            + "\n\n<code_interpreter>\n"
            + f"You have access to the following code execution tools: {_tool_names}.\n"
            + "Each tool accepts source code in the corresponding language and returns stdout + stderr.\n"
            + (
                "When bash is available, you also have sandbox builtin tools: "
                "`SandboxInfo`, `PWD`, `Read`, `Write`, `Edit`, `LS`, `Glob`, `Grep`, `Stat`, `Bash`, "
                "`BashOutput`, and `KillShell`. These operate exclusively inside the Linux sandbox. "
                "`Read`/`Write`/`Edit` require an absolute path, and the absolute workspace root "
                "differs by provider — never guess a prefix like `/workspace`. Before your first "
                "call to any of `Read`/`Write`/`Edit`, call `PWD` (or `SandboxInfo`, which also "
                "reports it) once to learn the sandbox's actual working directory, then build every "
                "absolute path from that.\n"
                if "bash" in _ci_languages else ""
            )
            + "Read uploaded files from input/<filename>.\n"
            + "Use work/ for temporary files, package installs, scripts, and dependencies.\n"
            + "Save only final user-facing deliverables in output/ and print the output/<filename> path.\n"
            + "</code_interpreter>"
        )

    if format_instructions:
        system_prompt_content = (
            system_prompt_content
            + "\n<output_format_instructions>"
            + format_instructions
            + "</output_format_instructions>"
        )

    # Localize current time and timezone for the agent context
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    user_tz = None
    if user_context and (tz_name := user_context.get("timezone")):
        try:
            user_tz = ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            logger.warning(f"Invalid timezone in user_context: {tz_name}")

    local_now = datetime.now(user_tz) if user_tz else datetime.now(timezone.utc)
    local_time_str = local_now.strftime("%Y-%m-%d %H:%M:%S %Z" if user_tz else "%Y-%m-%d %H:%M:%S UTC")

    system_prompt_content = (
        system_prompt_content
        + f"\n\n<current_time>\n"
        + f"Current date and time in user's location: {local_time_str}\n"
        + f"User's timezone: {user_context.get('timezone', 'UTC') if user_context else 'UTC'}\n"
        + f"</current_time>"
    )

    middleware = []

    # If a summarization middleware entity is attached, let it override
    # memory-based defaults even when has_memory=True.
    summarization_assoc_config = None
    if hasattr(agent, 'middleware_associations') and agent.middleware_associations:
        for assoc in agent.middleware_associations:
            if assoc.middleware and assoc.middleware.middleware_type.value == 'summarization':
                summarization_assoc_config = assoc.middleware.config or {}
                break

    if agent.has_memory:
        max_tokens = agent.memory_max_tokens or 4000
        max_messages = agent.memory_max_messages or 20
        summarization_llm = llm
        from models.agent import DEFAULT_MEMORY_SUMMARIZE_THRESHOLD
        trim_tokens = agent.memory_summarize_threshold or DEFAULT_MEMORY_SUMMARIZE_THRESHOLD
        if summarization_assoc_config is not None:
            max_tokens = summarization_assoc_config.get('trigger_tokens', max_tokens)
            max_messages = summarization_assoc_config.get('keep_messages', max_messages)
            # Use 4000 as fallback, not agent.memory_summarize_threshold, to avoid inheriting unrelated agent settings
            trim_tokens = summarization_assoc_config.get('trim_tokens', 4000)

            summarization_llm = llm
            summarization_model_value = summarization_assoc_config.get('summarization_model', 'agent_llm')
            if summarization_model_value and summarization_model_value != 'agent_llm':
                try:
                    service_id = int(summarization_model_value.split(':', 1)[1])
                    summarization_llm = _build_summarization_llm_from_service(agent, service_id) or llm
                except (ValueError, IndexError):
                    logger.warning(f"Invalid ai_service format '{summarization_model_value}' — using agent LLM")

        _trigger_tokens = max_tokens
        _keep_messages = max_messages
        _trim_tokens = trim_tokens
        _s_llm = summarization_llm
        _agent_id = agent.agent_id

        class _DiagnosticSummarizationMiddleware(SummarizationMiddleware):
            """Wraps SummarizationMiddleware to add diagnostic logging."""
            async def abefore_model(self, state, runtime):
                msgs = state.get("messages", [])
                approx = self.token_counter(msgs)
                logger.info(
                    f"[Summarization] abefore_model: agent={_agent_id}, "
                    f"messages={len(msgs)}, approx_tokens={approx}, trigger={self.trigger}"
                )
                result = await super().abefore_model(state, runtime)
                if result is not None:
                    new_count = len(result.get("messages", []))
                    logger.info(
                        f"[Summarization] TRIGGERED for agent {_agent_id}: "
                        f"reduced to {new_count} messages (summary generated)"
                    )
                else:
                    logger.info(
                        f"[Summarization] NOT triggered for agent {_agent_id} "
                        f"(approx_tokens={approx} < trigger={self.trigger})"
                    )
                return result

        summarization = _DiagnosticSummarizationMiddleware(
            model=_s_llm,
            trigger=("tokens", _trigger_tokens),
            keep=("messages", _keep_messages),
            trim_tokens_to_summarize=_trim_tokens,
        )
        middleware.append(summarization)
        logger.info(
            f"SummarizationMiddleware configured for agent {agent.agent_id}: "
            f"trigger=('tokens', {max_tokens}), keep=('messages', {max_messages}), "
            f"trim_tokens_to_summarize={trim_tokens}"
        )

    monitoring_handler = None
    if hasattr(agent, 'middleware_associations') and agent.middleware_associations:
        for assoc in agent.middleware_associations:
            if not assoc.middleware:
                continue
            mw_type = assoc.middleware.middleware_type.value
            mw_config = assoc.middleware.config or {}
            if mw_type == 'monitoring':
                monitoring_handler = UsageMetadataCallbackHandler()
                logger.info(f"MonitoringMiddleware (UsageMetadataCallbackHandler) enabled for agent {agent.agent_id}")
            elif mw_type == 'summarization':
                # Only add if not already added via has_memory (avoid duplicates)
                if not agent.has_memory:
                    summarization_model_value = mw_config.get('summarization_model', 'agent_llm')
                    summarization_llm = llm
                    if summarization_model_value and summarization_model_value != 'agent_llm':
                        try:
                            service_id = int(summarization_model_value.split(':', 1)[1])
                            summarization_llm = _build_summarization_llm_from_service(agent, service_id) or llm
                        except (ValueError, IndexError):
                            logger.warning(f"Invalid ai_service format '{summarization_model_value}' — using agent LLM")
                    else:
                        logger.info(f"Summarization using agent's own LLM")
                    trigger_tokens = mw_config.get('trigger_tokens', 4000)
                    keep_messages = mw_config.get('keep_messages', 20)
                    trim_tokens = mw_config.get('trim_tokens', 4000)
                    summarization_mw = SummarizationMiddleware(
                        model=summarization_llm,
                        trigger=("tokens", trigger_tokens),
                        keep=("messages", keep_messages),
                        trim_tokens_to_summarize=trim_tokens,
                    )
                    middleware.append(summarization_mw)
                    logger.info(
                        f"SummarizationMiddleware added via middleware entity for agent {agent.agent_id}: "
                        f"trigger=('tokens', {trigger_tokens}), keep=('messages', {keep_messages}), "
                        f"trim_tokens_to_summarize={trim_tokens}"
                    )
            elif mw_type == 'model_call_limit':
                max_calls = mw_config.get('max_calls', 50)
                middleware.append(ModelCallLimitMiddleware(run_limit=max_calls))
                logger.info(f"ModelCallLimitMiddleware enabled for agent {agent.agent_id} (limit={max_calls})")
            elif mw_type == 'tool_call_limit':
                max_calls = mw_config.get('max_calls', 100)
                middleware.append(ToolCallLimitMiddleware(run_limit=max_calls))
                logger.info(f"ToolCallLimitMiddleware enabled for agent {agent.agent_id} (limit={max_calls})")
            elif mw_type == 'pii':
                pii_types = mw_config.get('pii_types', ['email', 'credit_card', 'ip', 'mac_address', 'url'])
                strategy = mw_config.get('strategy', 'redact')
                apply_to_input = mw_config.get('apply_to_input', True)
                apply_to_output = mw_config.get('apply_to_output', True)
                apply_to_tool_results = mw_config.get('apply_to_tool_results', True)
                # PIIMiddleware accepts a single pii_type — create one instance per type
                for pii_type in pii_types:
                    middleware.append(PIIMiddleware(
                        pii_type=pii_type,
                        strategy=strategy,
                        apply_to_input=apply_to_input,
                        apply_to_output=apply_to_output,
                        apply_to_tool_results=apply_to_tool_results,
                    ))
                # Add a logging middleware after PII to show redacted content
                class _PIILogMiddleware(AgentMiddleware):
                    def before_model(self, state, runtime):
                        msgs = state.get("messages", [])
                        for msg in reversed(msgs):
                            if isinstance(msg, HumanMessage):
                                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                                logger.info(f"[PII] Message after redaction: {content[:300]}")
                                break
                        return None
                middleware.append(_PIILogMiddleware())
                logger.info(f"PIIMiddleware enabled for agent {agent.agent_id} (types={pii_types}, strategy={strategy})")
            elif mw_type == 'human_in_the_loop':
                interrupt_on = mw_config.get('interrupt_on', {})
                if interrupt_on:
                    description_prefix = mw_config.get('description_prefix', 'Tool execution requires approval')
                    middleware.append(HumanInTheLoopMiddleware(
                        interrupt_on=interrupt_on,
                        description_prefix=description_prefix,
                    ))
                    logger.info(f"HumanInTheLoopMiddleware enabled for agent {agent.agent_id} (tools={list(interrupt_on.keys())})")
                else:
                    logger.warning(f"HumanInTheLoopMiddleware skipped for agent {agent.agent_id}: 'interrupt_on' config is empty")
            
            elif mw_type == 'guardrails':
                from tools.middleware.guardrails import GuardrailsMiddleware
                mw_instance = GuardrailsMiddleware(config=mw_config)
                middleware.append(mw_instance)
                logger.info(f"GuardrailsMiddleware enabled for agent {agent.agent_id}")

            else:
                logger.warning(f"Unknown middleware type '{mw_type}' for agent {agent.agent_id} — skipped")

    tools = []

    # Provider-side tools — injected from agent.server_tools using provider-specific formats
    _SERVER_TOOL_FORMATS = {
        "OpenAI":     {"web_search": {"type": "web_search"}, "image_generation": {"type": "image_generation"}, "code_interpreter": {"type": "code_interpreter"}, "file_search": {"type": "file_search"}},
        "Azure":      {"web_search": {"type": "web_search"}, "image_generation": {"type": "image_generation"}, "code_interpreter": {"type": "code_interpreter"}, "file_search": {"type": "file_search"}},
        "Anthropic":  {"web_search": {"type": "web_search_20250305"}, "code_interpreter": {"type": "codeExecution_20250825"}},
        "Google":     {"web_search": {"type": "google_search"}, "code_interpreter": {"type": "code_execution"}},
        "MistralAI":  {},
        "Custom":     {},
    }
    provider_name = agent.ai_service.provider if agent.ai_service else None
    provider_map = _SERVER_TOOL_FORMATS.get(provider_name, {})
    for tool_name in (getattr(agent, 'server_tools', None) or []):
        tool_def = provider_map.get(tool_name)
        if tool_def:
            tools.append(tool_def)
            logger.info("Server-side tool '%s' injected for provider %s", tool_name, provider_name)
        else:
            logger.warning("Server-side tool '%s' not supported by provider %s — skipped", tool_name, provider_name)

    for tool in agent.tool_associations:
        sub_agent = tool.tool
        tools.append(await discover_tool(
            sub_agent,
            user_context=user_context,
            working_dir=working_dir,
            sandbox_handle=sandbox_handle,
            sandbox_provider=sandbox_provider,
            sandbox_session_key=sandbox_session_key,
            sandbox_session_service=sandbox_session_service,
            attached_files=attached_files,
        ))

    # Base tools — always available for every agent
    if working_dir:
        tools.append(create_download_url_tool(working_dir))

    if agent.silo_id is not None:
        # Resolve precedence (caller > agent RAG config > system) AND build the tool
        # off the event loop: both precedence resolution (lazy-loads
        # silo.metadata_definition) and construction (distinct-value sampling) do
        # synchronous DB work.
        retriever_tool = await asyncio.to_thread(
            _resolve_and_build_retriever_tool, agent, search_params
        )
        if retriever_tool is not None:
            if isinstance(retriever_tool, list):
                tools.extend(retriever_tool)
            else:
                tools.append(retriever_tool)

    # Add temp silo retrievers (e.g. playground media/file uploads)
    has_temp_retrievers = False
    has_media_content = False
    has_file_content = False
    if temp_silo_ids:
        def _build_temp_retrievers():
            """Sync silo/media inspection + retriever build (own DB session).

            Runs off the event loop via ``asyncio.to_thread`` so the synchronous
            embedding/vector-store setup inside ``get_retriever_tool`` and the
            Media lookups do not block the loop during chain construction.
            """
            from repositories.silo_repository import SiloRepository
            from models.media import Media
            from db.database import SessionLocal
            built = []
            media_found = False
            temp_db = SessionLocal()
            try:
                for idx, temp_silo_id in enumerate(temp_silo_ids):
                    temp_silo = SiloRepository.get_by_id(temp_silo_id, temp_db)
                    if not temp_silo:
                        continue
                    temp_tool = get_retriever_tool(temp_silo, search_params)
                    if not temp_tool:
                        continue
                    temp_tool.name = f"playground_content_retriever_{idx}"
                    temp_tool.description = (
                        "Use this tool to search for information from content uploaded "
                        "in this conversation (documents, PDFs, text files, video/audio "
                        "transcriptions). Always use this when the user asks about "
                        "attached or uploaded content."
                    )
                    built.append(temp_tool)

                    # Check if silo contains media content (for timestamp instructions)
                    if temp_silo.repository:
                        repo_list = temp_silo.repository if isinstance(temp_silo.repository, list) else [temp_silo.repository]
                        for repo in repo_list:
                            if temp_db.query(Media).filter(Media.repository_id == repo.repository_id).first():
                                media_found = True
                                break
            finally:
                temp_db.close()
            return built, media_found

        temp_tools, has_media_content = await asyncio.to_thread(_build_temp_retrievers)
        for temp_tool in temp_tools:
            tools.append(temp_tool)
            has_temp_retrievers = True
            logger.info(f"Added temp silo retriever for agent {agent.agent_id}")

        # Check if files were vectorized by looking at silo content
        if has_temp_retrievers:
            has_file_content = True  # files are always possible when temp retrievers exist

    # Inject timestamp citation instruction when media content is in the silo
    if has_media_content:
        system_prompt_content = (
            system_prompt_content
            + "\n\n<media_timestamp_instructions>\n"
            + "The user has uploaded video/audio media. When answering questions about media content, "
            + "you MUST include timestamps from the retrieved chunks metadata.\n"
            + "Each retrieved chunk has metadata with the fields 'name' (the media filename), "
            + "'start_time' and 'end_time' (in seconds).\n"
            + "When you reference a specific moment, cite it using this EXACT format, including the "
            + "media filename so the moment is unambiguously tied to its media:\n"
            + "[<name> @ MM:SS - MM:SS]\n"
            + "where <name> is the exact 'name' value from that chunk's metadata. Convert start_time "
            + "and end_time from seconds to MM:SS (or HH:MM:SS when over an hour).\n"
            + "Example: if a chunk has name=\"lesson.mp4\", start_time=125 and end_time=180, "
            + "cite it as [lesson.mp4 @ 02:05 - 03:00].\n"
            + "Only cite timestamps for the specific media you are actually referencing. When multiple "
            + "media were uploaded, NEVER mix timestamps from one media into a citation for another, and "
            + "only cite the media that genuinely answer the user's question.\n"
            + "Always cite the relevant timestamps so the user can navigate to those moments in the media.\n"
            + "</media_timestamp_instructions>"
        )

    # Inject file retriever instruction when files have been vectorized
    if has_file_content:
        system_prompt_content = (
            system_prompt_content
            + "\n\n<uploaded_files_instructions>\n"
            + "The user has uploaded documents (PDFs, text files) in this conversation. "
            + "Their content has been indexed for semantic search.\n"
            + "When the user asks about or references attached files, you MUST use the "
            + "uploaded-file semantic search retriever tool available in this conversation "
            + "to search for relevant information.\n"
            + "Always cite the source filename from the retrieved chunk metadata.\n"
            + "</uploaded_files_instructions>"
        )

    if agent.enable_code_interpreter and working_dir:
        os.makedirs(working_dir, exist_ok=True)
        if sandbox_provider is None:
            try:
                sandbox_provider = ci_provider_for_prompt or resolve_provider(agent)
            except SandboxProviderUnavailableError as exc:
                logger.warning(
                    "Skipping code interpreter tools for agent %s: %s",
                    agent.agent_id,
                    exc,
                )
                sandbox_provider = None
        if sandbox_provider is not None and sandbox_handle is None:
            logger.warning(
                "Code interpreter tool requested without prepared sandbox handle; "
                "creating fallback sandbox during tool assembly for agent %s",
                agent.agent_id,
            )
            try:
                sandbox_handle = sandbox_provider.create_sandbox(working_dir=working_dir)
            except Exception as exc:
                # A provider that resolves fine but is actually unreachable
                # (e.g. registered but the backing service isn't running)
                # must not crash tool assembly — degrade the same way an
                # unavailable provider does just above.
                logger.warning(
                    "Fallback sandbox creation failed for agent %s: %s",
                    agent.agent_id,
                    exc,
                )
                sandbox_provider = None
                sandbox_handle = None
        if sandbox_provider is not None and sandbox_handle is not None:
            if sandbox_session_service is None and sandbox_session_key is not None:
                try:
                    from services.sandbox_session_service import sandbox_session_service as _sss
                    sandbox_session_service = _sss
                except Exception:
                    sandbox_session_service = None
            repl_tools = create_sandbox_repl_tools(
                sandbox_handle,
                sandbox_provider,
                session_key=sandbox_session_key,
                session_service=sandbox_session_service,
            )
            tools.extend(repl_tools)
            builtin_tools = create_sandbox_builtin_tools(
                sandbox_handle,
                sandbox_provider,
                session_key=sandbox_session_key,
                session_service=sandbox_session_service,
            )
            tools.extend(builtin_tools)
            logger.info(
                "Sandbox tools added for agent %s (repl=%s, builtins=%s, working_dir=%s, sandbox_id=%s, provider=%s)",
                agent.agent_id,
                [t.name for t in repl_tools],
                [t.name for t in builtin_tools],
                working_dir,
                _sandbox_id_for_log(sandbox_handle),
                sandbox_handle.provider_name,
            )

    mcp_client = None
    try:
        logger.info("Starting MCP tools loading...")
        mcp_client = await MCPClientManager().get_client(agent, user_context)
        if mcp_client:
            mcp_tools = await asyncio.wait_for(
                mcp_client.get_tools(), timeout=MCP_TOOLS_TIMEOUT
            )
            logger.info(f"MCP tools loaded successfully: {len(mcp_tools)} tools")
            for tool in mcp_tools:
                if hasattr(tool, "args_schema") and isinstance(tool.args_schema, dict):
                    ensure_json_schema_types(tool.args_schema)
            if mcp_tools:
                tools.extend(mcp_tools)
    except asyncio.TimeoutError:
        logger.warning(
            f"MCP tools loading timed out after {MCP_TOOLS_TIMEOUT}s — "
            "agent will continue without MCP tools"
        )
        mcp_client = None
    except Exception as e:
        root_cause = _extract_mcp_root_causes(e) if isinstance(e, BaseExceptionGroup) else str(e)
        logger.warning(
            f"MCP tools unavailable (agent will continue without them): {root_cause}"
        )
        logger.debug("Full MCP tools loading error:", exc_info=True)
        mcp_client = None

    # Add skill loader tool for non-router skills (the router skill is inlined
    # into the system prompt above, not offered as an on-demand load).
    if other_skill_associations:
        skill_tool = create_skill_loader_tool(other_skill_associations)
        if skill_tool:
            tools.append(skill_tool)
            logger.info(f"Skill loader tool added with {len(other_skill_associations)} skills")

    # Pre-bind tools with parallel_tool_calls=False so the LLM never fans out
    # multiple simultaneous calls to the same tool (e.g. retrieve_from_knowledge_base
    # called 6× in one turn). LangGraph's _should_bind_tools detects the pre-bound
    # model and skips its own bind, preserving this setting.
    try:
        llm = llm.bind_tools(tools, parallel_tool_calls=False)
    except TypeError:
        # Some providers (e.g. older Anthropic builds) don't accept parallel_tool_calls;
        # fall back to plain bind so the agent still works.
        llm = llm.bind_tools(tools)

    if pydantic_model:
        # In LangChain v1, response_format accepts the pydantic model directly.
        # It defaults to ProviderStrategy (native structured output) if supported,
        # falling back to ToolStrategy (artificial tool calling) otherwise.
        agent_chain = create_langchain_agent(
            model=llm,
            system_prompt=system_prompt_content,
            response_format=pydantic_model,
            tools=tools,
            checkpointer=checkpointer,
            middleware=middleware or [],
        )
    else:
        agent_chain = create_langchain_agent(
            model=llm,
            system_prompt=system_prompt_content,
            tools=tools,
            checkpointer=checkpointer,
            middleware=middleware or [],
        )

    langsmith_config = None  # TODO: build from agent.app via resolve_langsmith_settings

    # Add logging for the created agent
    logger.info(f"Created agent with {len(tools)} tools")
    logger.info(f"Memory enabled: {agent.has_memory}")
    logger.info(f"Output parser: {agent.output_parser_id is not None}")
    logger.info(f"LangSmith configured: {langsmith_config is not None}")
    logger.info(f"Monitoring enabled: {monitoring_handler is not None}")

    return agent_chain, langsmith_config, mcp_client, monitoring_handler


_DEFAULT_RECURSION_LIMIT = 50


def _load_recursion_limit() -> int:
    """Read AICT_AGENT_RECURSION_LIMIT from the environment once at module load.

    Logs a WARNING and falls back to 50 when the value is absent, non-integer, or
    below 1 (LangGraph requires recursion_limit >= 1; a value < 1 fails every turn).
    """
    raw = os.getenv("AICT_AGENT_RECURSION_LIMIT")
    if raw is None:
        return _DEFAULT_RECURSION_LIMIT
    try:
        value = int(raw)
    except (ValueError, TypeError):
        logger.warning(
            "prepare_agent_config: AICT_AGENT_RECURSION_LIMIT=%r is not a valid integer; "
            "using default %d",
            raw, _DEFAULT_RECURSION_LIMIT,
        )
        return _DEFAULT_RECURSION_LIMIT
    if value < 1:
        logger.warning(
            "prepare_agent_config: AICT_AGENT_RECURSION_LIMIT=%r is < 1 (invalid); "
            "using default %d",
            raw, _DEFAULT_RECURSION_LIMIT,
        )
        return _DEFAULT_RECURSION_LIMIT
    return value


AICT_AGENT_RECURSION_LIMIT: int = _load_recursion_limit()


def _resolve_and_build_retriever_tool(agent, caller_search_params):
    """Resolve RAG precedence then build the dynamic retriever tool for *agent*.

    Runs synchronous DB work — precedence resolution lazy-loads
    ``silo.metadata_definition`` and ``get_retriever_tool`` samples distinct values.
    MUST be invoked via ``asyncio.to_thread`` so it never blocks the event loop.

    LightRAG silos ignore metadata filters (the query mode is their only knob):
    ``agent.lightrag_query_mode`` selects it. ``skill-routed`` returns a
    list[BaseTool] (the dynamic retrieval tool plus list_documents_mentioning)
    so the LLM picks the mode per call (router skill must be active, else it
    degrades transparently to ``hybrid``, returning a single tool like every
    other mode).
    """
    from services.silo_service import resolve_search_params  # noqa: PLC0415 — avoids import cycle

    silo = agent.silo
    is_lightrag = bool(
        getattr(silo, "vector_db_type", None)
        and str(silo.vector_db_type).upper() == "LIGHTRAG"
    )
    lightrag_mode = getattr(agent, "lightrag_query_mode", None) if is_lightrag else None

    # Resolved BEFORE the skill-routed branch, not after. resolve_search_params
    # is the only place that maps Agent.rag_k -> k and Agent.rag_chunk_top_k ->
    # lightrag_chunk_top_k; returning early with the raw caller params (None on
    # the agent-execution path) silently dropped every per-agent RAG setting for
    # skill-routed agents, which then ran on the deployment-wide env defaults
    # instead. It was invisible: the UI and the eval harness both read the
    # stored DB values, so they reported a config that was never applied — an
    # A/B of rag_chunk_top_k 30 vs 60 ran at 30 both times and "proved" the
    # change did nothing.
    resolved_sp, resolved_pinned = resolve_search_params(agent, caller_search_params)

    if lightrag_mode == "skill-routed":
        skill_assocs = getattr(agent, "skill_associations", None) or []
        if any(getattr(a.skill, "name", None) == LIGHTRAG_ROUTER_SKILL_NAME
               for a in skill_assocs if a.skill):
            # The dynamic tool sets lightrag_query_mode itself, per call. Both
            # tools share ONE citation offset so numbering stays global across
            # a turn that calls both (see _append_lightrag_citation_sources).
            #
            # FIXED (2026-09-08): LangGraph's ToolNode runs both tools
            # CONCURRENTLY when the LLM calls both in one turn (the normal
            # case for a coverage-router question) — but the shared offset is
            # claimed in call order while the frontend's merged chunks[] array
            # (agent_streaming_service.py -> merge_lightrag_graph in
            # tools/streaming_utils.py) is built in event-ARRIVAL order.
            # Whichever tool finished first streamed its chunks first, so if
            # the tool that claimed offset numbers first was the slower one,
            # its [N](cite://N) markers ended up pointing at the OTHER tool's
            # chunks once merged. Confirmed live: asked "¿Qué significa P20 en
            # el manual DUAL CLIMA HT?" (fires both list_documents_mentioning
            # and retrieve_from_knowledge_base) — citation [4] in the answer
            # text (about DSAT000120 p.108) opened CDOC004352 p.19 instead.
            # Fixed by serializing the two tools with a shared asyncio.Lock so
            # only one can run at a time within a turn — offset-claim order
            # and chunk-arrival order can no longer diverge.
            # ponytail: this trades away the parallelism ToolNode normally
            # gives multi-tool turns — the two LightRAG calls now run back to
            # back instead of concurrently, adding roughly one call's latency
            # to a coverage-router turn. Revisit alongside future
            # inference-time/latency work (e.g. renumber after merge instead
            # of during execution, to get the parallelism back) if that
            # latency becomes a problem.
            offset: List[int] = [0]
            lock = asyncio.Lock()
            return [
                _create_dynamic_lightrag_tool(silo, resolved_sp, offset, lock),
                _create_coverage_tool(silo, agent.app_id, offset, lock),
            ]
        # ponytail: router skill toggled off — fall back transparently to hybrid
        lightrag_mode = "hybrid"

    if lightrag_mode:
        resolved_sp["lightrag_query_mode"] = lightrag_mode
    return get_retriever_tool(
        agent.silo,
        resolved_sp,
        getattr(agent, "rag_max_retrieval_calls", None),
        resolved_pinned,
    )


def prepare_agent_config(agent):
    """Helper function to prepare agent configuration."""
    config = {
        "configurable": {
            "thread_id": f"thread_{agent.agent_id}"
        },
        "recursion_limit": AICT_AGENT_RECURSION_LIMIT,
    }
    return config


def parse_agent_response(response_text, agent):
    """Helper function to parse agent response.
    
    In LangChain v1, structured output is returned in the 'structured_response' key
    of the agent result when response_format is used with create_agent.
    """
    if agent.output_parser_id is not None:
        # If response is already a dict (from structured output), return it directly
        if isinstance(response_text, dict):
            return response_text
        
        # If response is a Pydantic model instance, convert to dict
        if hasattr(response_text, 'model_dump'):
            return response_text.model_dump()
        
        # If response is a string, try to parse it as JSON
        content = response_text.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            return response_text
    return response_text


def build_human_message(
    agent: Agent,
    message: str,
    image_files: List[Dict],
    user_context: Optional[Dict] = None,
) -> HumanMessage:
    """Build the HumanMessage that will be fed into the agent chain.

    When ``image_files`` is non-empty the content becomes a multimodal list of
    text + image_url blocks.  Images are served via a signed URL when
    ``AICT_BASE_URL`` is set (production), or inlined as base64 data URIs in
    development mode.

    Args:
        agent: The (freshly loaded) Agent ORM instance.
        message: The already-enhanced text message (with file content appended
            if applicable).
        image_files: List of image-file dicts (``file_path`` key required).
        user_context: Caller context dict used to generate signed URLs.

    Returns:
        A ``HumanMessage`` instance ready for ``agent_chain.ainvoke()`` /
        ``agent_chain.astream()``.
    """
    from utils.config import get_app_config

    formatted_message = agent.prompt_template.format(question=message) if agent.prompt_template else message

    if not image_files:
        return HumanMessage(content=formatted_message)

    app_config = get_app_config()
    tmp_base_folder = app_config["TMP_BASE_FOLDER"]
    aict_base_url = os.getenv("AICT_BASE_URL")

    content: List[Dict] = [{"type": "text", "text": formatted_message}]

    for img in image_files:
        file_path: str = img.get("file_path", "")
        if not file_path:
            logger.warning("Image file has no file_path — skipping: %s", img)
            continue

        # Normalise to forward slashes and strip leading slash
        file_path = file_path.replace("\\", "/").lstrip("/")

        if aict_base_url:
            # Production mode — generate a signed static URL
            aict_base_url = aict_base_url.rstrip("/")
            user_email: Optional[str] = (
                user_context.get("email") if user_context else None
            )
            if user_email:
                from utils.security import generate_signature

                sig = generate_signature(file_path, user_email)
                url = (
                    f"{aict_base_url}/static/{file_path}"
                    f"?user={user_email}&sig={sig}"
                )
            else:
                url = f"{aict_base_url}/static/{file_path}"

            logger.info("Adding image to message using signed URL: %s", url)
            content.append({"type": "image_url", "image_url": {"url": url}})
        else:
            # Development mode — inline as base64 data URI
            full_path = os.path.join(tmp_base_folder, file_path)
            if os.path.exists(full_path):
                try:
                    mime_type, _ = mimetypes.guess_type(full_path)
                    if not mime_type:
                        mime_type = "image/jpeg"
                    with open(full_path, "rb") as fh:
                        encoded = base64.b64encode(fh.read()).decode("utf-8")
                    data_url = f"data:{mime_type};base64,{encoded}"
                    logger.info(
                        "Adding image as base64 (length: %d)", len(encoded)
                    )
                    content.append(
                        {"type": "image_url", "image_url": {"url": data_url}}
                    )
                except Exception as exc:
                    logger.error(
                        "Error encoding image as base64: %s — falling back to URL",
                        exc,
                    )
                    url = f"http://localhost:8000/static/{file_path}"
                    content.append(
                        {"type": "image_url", "image_url": {"url": url}}
                    )
            else:
                url = f"http://localhost:8000/static/{file_path}"
                logger.warning(
                    "Image not found at %s — falling back to URL: %s",
                    full_path,
                    url,
                )
                content.append(
                    {"type": "image_url", "image_url": {"url": url}}
                )

    return HumanMessage(content=content)


class AgentToolInput(BaseModel):
    """Input schema for an agent-as-tool.

    Pinned explicitly so the provider function schema is exactly
    ``{query: string}``. Without it, ``BaseTool`` derives the schema from the
    ``_run(self, query, *args, **kwargs)`` signature, which emits an untyped
    ``args`` array (``items`` with no ``type``) that OpenAI rejects with
    ``schema must have a 'type' key``.
    """

    query: str = Field(description="The question or instruction to send to the sub-agent.")


class IACTTool(BaseTool):
    name: str = "agent_tool"
    description: str = "Search for a repository"
    args_schema: Type[BaseModel] = AgentToolInput
    agent: Agent
    user_context: Optional[Dict] = None
    react_agent: Any = None
    mcp_client: Any = None
    llm: Any = None
    working_dir: Optional[str] = None
    sandbox_handle: Any = None
    sandbox_provider: Any = None
    sandbox_session_key: Optional[str] = None
    sandbox_session_service: Any = None
    attached_files: Optional[List[Dict]] = None

    def __init__(
        self,
        agent: Agent,
        user_context: Optional[Dict] = None,
        working_dir: Optional[str] = None,
        sandbox_handle: Optional[Any] = None,
        sandbox_provider: Optional[Any] = None,
        sandbox_session_key: Optional[str] = None,
        sandbox_session_service: Optional[Any] = None,
        attached_files: Optional[List[Dict]] = None,
    ) -> None:
        super().__init__(
            agent=agent,
            user_context=user_context,
            working_dir=working_dir,
            sandbox_handle=sandbox_handle,
            sandbox_provider=sandbox_provider,
            sandbox_session_key=sandbox_session_key,
            sandbox_session_service=sandbox_session_service,
            attached_files=attached_files,
        )

        self.agent = agent
        self.user_context = user_context
        self.working_dir = working_dir
        self.sandbox_handle = sandbox_handle
        self.sandbox_provider = sandbox_provider
        self.sandbox_session_key = sandbox_session_key
        self.sandbox_session_service = sandbox_session_service
        self.attached_files = attached_files or []
        self.name = sanitize_identifier(agent.name)
        self.description = agent.description or "Agent tool"
        self.llm = get_llm(agent)
        if self.llm is None:
            raise ValueError("No LLM found for agent")
        self.react_agent = None
        self.mcp_client = None

    @classmethod
    async def create(
        cls,
        agent: Agent,
        user_context: Optional[Dict] = None,
        working_dir: Optional[str] = None,
        sandbox_handle: Optional[Any] = None,
        sandbox_provider: Optional[Any] = None,
        sandbox_session_key: Optional[str] = None,
        sandbox_session_service: Optional[Any] = None,
        attached_files: Optional[List[Dict]] = None,
    ) -> "IACTTool":
        """Build an agent-as-tool, including the sub-agent's MCP tools.

        MCP tools are loaded with an awaited MultiServerMCPClient, which is not
        possible inside a synchronous ``__init__``; hence this async factory. It
        is the only supported way to obtain a ready-to-use ``IACTTool``.
        """
        instance = cls(
            agent,
            user_context=user_context,
            working_dir=working_dir,
            sandbox_handle=sandbox_handle,
            sandbox_provider=sandbox_provider,
            sandbox_session_key=sandbox_session_key,
            sandbox_session_service=sandbox_session_service,
            attached_files=attached_files,
        )

        tools = []
        # Add nested tool agents recursively
        for tool in agent.tool_associations:
            sub_agent = tool.tool
            tools.append(await discover_tool(sub_agent, user_context=user_context, attached_files=attached_files))

        # Add base useful tools
        tools.append(fetch_file_in_base64)

        # Add silo retriever if configured. The sub-agent uses the same dynamic
        # metadata-aware tool as the root agent, driven by its OWN RAG config
        # (rag_k / rag_search_type / rag_score_threshold / rag_fixed_filters /
        # rag_max_retrieval_calls). Caller search params are NOT propagated from the
        # root agent (caller_search_params=None) — sub-agents are self-contained (FR-12).
        if agent.silo_id is not None:
            # Caller params NOT propagated (None) — the sub-agent uses its OWN config.
            # Off the event loop: resolution + construction do synchronous DB work.
            retriever_tool = await asyncio.to_thread(
                _resolve_and_build_retriever_tool, agent, None
            )
            if retriever_tool is not None:
                if isinstance(retriever_tool, list):
                    tools.extend(retriever_tool)
                else:
                    tools.append(retriever_tool)

        if agent.enable_code_interpreter and working_dir and sandbox_handle is not None and sandbox_provider is None:
            try:
                sandbox_provider = resolve_provider(agent)
                instance.sandbox_provider = sandbox_provider
            except SandboxProviderUnavailableError as exc:
                logger.warning(
                    "Skipping code interpreter tools for sub-agent %s: %s",
                    agent.agent_id,
                    exc,
                )

        if agent.enable_code_interpreter and working_dir and sandbox_handle is not None and sandbox_provider is not None:
            repl_tools = create_sandbox_repl_tools(
                sandbox_handle,
                sandbox_provider,
                session_key=sandbox_session_key,
                session_service=sandbox_session_service,
            )
            tools.extend(repl_tools)
            logger.info(
                "Shared sandbox REPL tools added for sub-agent %s "
                "(languages=%s, working_dir=%s, sandbox_id=%s, provider=%s)",
                agent.agent_id,
                [t.name for t in repl_tools],
                working_dir,
                _sandbox_id_for_log(sandbox_handle),
                sandbox_handle.provider_name,
            )

        # Add MCP tools — mirrors create_agent. A failing MCP server degrades the
        # sub-agent but never breaks its construction.
        try:
            logger.info(f"Starting MCP tools loading for sub-agent {agent.agent_id}...")
            instance.mcp_client = await MCPClientManager().get_client(agent, user_context)
            if instance.mcp_client:
                mcp_tools = await asyncio.wait_for(
                    instance.mcp_client.get_tools(), timeout=MCP_TOOLS_TIMEOUT
                )
                logger.info(
                    f"MCP tools loaded successfully for sub-agent {agent.agent_id}: "
                    f"{len(mcp_tools)} tools"
                )
                if mcp_tools:
                    tools.extend(mcp_tools)
        except asyncio.TimeoutError:
            logger.warning(
                f"MCP tools loading timed out for sub-agent {agent.agent_id} "
                f"after {MCP_TOOLS_TIMEOUT}s — sub-agent will continue without MCP tools"
            )
            instance.mcp_client = None
        except Exception as e:
            root_cause = _extract_mcp_root_causes(e) if isinstance(e, BaseExceptionGroup) else str(e)
            logger.warning(
                f"MCP tools unavailable for sub-agent {agent.agent_id} "
                f"(will continue without them): {root_cause}"
            )
            logger.debug("Full MCP tools loading error for sub-agent:", exc_info=True)
            instance.mcp_client = None

        # Build system prompt with optional skills section (LangChain v1 pattern)
        tool_system_prompt = agent.system_prompt or ""
        # Inject current date to avoid need for a tool call
        current_date = datetime.now().strftime("%Y-%m-%d")
        tool_system_prompt += f"\n\nToday's date is {current_date}."
        if agent.system_prompt and hasattr(agent, 'skill_associations') and agent.skill_associations:
            skills_section = generate_skills_system_prompt_section(agent.skill_associations)
            if skills_section:
                tool_system_prompt = tool_system_prompt + "\n" + skills_section

        if working_dir:
            tool_system_prompt = (
                tool_system_prompt
                + "\n\n<workspace>\n"
                + f"Working directory: {working_dir}\n"
                + "Workspace layout:\n"
                + "- input/: user-provided files. Treat these as source material.\n"
                + "- work/: scratch files, scripts, dependencies, extracted content, and intermediate data.\n"
                + "- output/: final files intended for the user to download.\n"
                + "User-uploaded files are available under input/; reference them as input/<filename>.\n"
                + "</workspace>"
            )

        if agent.enable_code_interpreter and working_dir and sandbox_handle is not None and sandbox_provider is not None:
            _ci_languages = sandbox_provider.get_supported_languages()
            _tool_names = ", ".join(f"`{_tool_name_for_language(lang)}`" for lang in _ci_languages)
            tool_system_prompt = (
                tool_system_prompt
                + "\n\n<code_interpreter>\n"
                + f"You have access to the following code execution tools: {_tool_names}.\n"
                + "They run in the same sandbox used by the parent agent and sibling tools.\n"
                + "Read uploaded files from input/<filename>.\n"
                + "Use work/ for temporary files, package installs, scripts, and dependencies.\n"
                + "Save only final user-facing deliverables in output/ and print the output/<filename> path.\n"
                + "</code_interpreter>"
            )

        # Create sub-agent
        instance.react_agent = create_langchain_agent(
            model=instance.llm,
            tools=tools,
            system_prompt=tool_system_prompt if tool_system_prompt else None,
        )
        return instance

    def _run(self, query: str, *args, **kwargs) -> str:
        """Synchronous execution of the agent tool"""
        if self.react_agent is None:
            raise RuntimeError(
                "IACTTool must be built via 'await IACTTool.create(...)' before use."
            )
        try:
            formatted_prompt = self._format_tool_query(query)
            messages = [HumanMessage(content=formatted_prompt)]
            result = self.react_agent.invoke({"messages": messages})
            return self._extract_last_message_content(result)
        except Exception as e:
            logger.error(f"Error executing agent tool {self.name}: {str(e)}")
            return f"Error executing agent tool: {str(e)}"

    def _format_tool_query(self, query: str) -> str:
        """Apply the sub-agent prompt template to a tool query."""
        if self.agent.prompt_template:
            try:
                return self.agent.prompt_template.format(question=query)
            except KeyError:
                try:
                    return self.agent.prompt_template.format(query=query)
                except KeyError:
                    logger.warning(
                        f"Could not format prompt_template for agent {self.agent.name}, using query directly"
                    )
        return query

    @staticmethod
    def _extract_last_message_content(result: Any) -> str:
        """Return the last non-empty message content from a LangGraph result/state."""
        if isinstance(result, dict) and "messages" in result:
            messages_list = result["messages"]
            for msg in reversed(messages_list):
                if hasattr(msg, 'content') and msg.content:
                    return str(msg.content)
            if messages_list:
                last_msg = messages_list[-1]
                return str(last_msg.content) if hasattr(last_msg, 'content') else str(last_msg)
        return str(result)

    def _get_stream_writer_or_none(self):
        try:
            from langgraph.config import get_stream_writer

            return get_stream_writer()
        except Exception:
            return None

    def _emit_subagent_stream_event(self, writer: Any, event: dict) -> None:
        """Forward a nested sub-agent event to the parent graph custom stream."""
        if writer is None or not isinstance(event, dict):
            return

        event_type = event.get("type")
        data = event.get("data")
        if not isinstance(data, dict):
            data = {}

        if event_type in {"tool_start", "tool_end", "thinking"}:
            raw_tool_call_id = data.get("tool_call_id")
            data = {
                **data,
                "parent_tool_name": self.name,
                "subagent_name": self.agent.name,
                "subagent_id": self.agent.agent_id,
            }
            if raw_tool_call_id:
                data["raw_tool_call_id"] = raw_tool_call_id
                data["tool_call_id"] = f"{self.name}:{self.agent.agent_id}:{raw_tool_call_id}"
            try:
                writer({"type": event_type, "data": data})
            except Exception:
                pass
            return

        if event_type == "code_output":
            try:
                writer({
                    "type": "code_output",
                    "tool_name": data.get("tool_name"),
                    "stream": data.get("stream", "stdout"),
                    "line": data.get("line", ""),
                    "parent_tool_name": self.name,
                    "subagent_name": self.agent.name,
                    "subagent_id": self.agent.agent_id,
                })
            except Exception:
                pass

    async def _arun(self, query: str, *args, **kwargs) -> str:
        """Asynchronous execution of the agent tool"""
        if self.react_agent is None:
            raise RuntimeError(
                "IACTTool must be built via 'await IACTTool.create(...)' before use."
            )
        try:
            formatted_prompt = self._format_tool_query(query)
            messages = [HumanMessage(content=formatted_prompt)]
            stream_writer = self._get_stream_writer_or_none()

            if stream_writer is None:
                result = await self.react_agent.ainvoke({"messages": messages})
                return self._extract_last_message_content(result)

            if not hasattr(self.react_agent, "astream"):
                result = await self.react_agent.ainvoke({"messages": messages})
                return self._extract_last_message_content(result)

            from tools.streaming_utils import map_stream_event

            latest_state: Any = None
            async for mode, chunk in self.react_agent.astream(
                {"messages": messages},
                stream_mode=["updates", "custom"],
            ):
                if mode == "updates" and isinstance(chunk, dict):
                    for state_delta in chunk.values():
                        if isinstance(state_delta, dict) and "messages" in state_delta:
                            latest_state = state_delta

                events = map_stream_event(mode, chunk)
                if not events:
                    continue
                for event in events:
                    self._emit_subagent_stream_event(stream_writer, event)

            if latest_state is not None:
                return self._extract_last_message_content(latest_state)

            return ""

        except Exception as e:
            logger.error(f"Error executing agent tool {self.name} (async): {str(e)}")
            return f"Error executing agent tool: {str(e)}"


async def _execute_tool_agent_ocr(
    agent_id: int,
    pdf_path: str,
    user_context: Dict,
) -> Dict[str, Any]:
    """Delegate to AgentExecutionService.execute_agent_ocr synchronously."""
    from sqlalchemy.orm import Session
    from db.database import SessionLocal
    from services.agent_execution_service import AgentExecutionService
    from fastapi import UploadFile as FastAPIUploadFile

    db: Session = SessionLocal()
    try:
        # Wrap pdf_path in an in-memory UploadFile so we can reuse
        # AgentExecutionService.execute_agent_ocr without duplicating OCR logic.
        class _PathUploadFile(FastAPIUploadFile):
            """Thin UploadFile wrapper backed by a file path."""
            _file: Any = None

            def __init__(self, file_path: str) -> None:
                super().__init__(
                    filename=os.path.basename(file_path),
                    file=open(file_path, "rb"),
                )
                self._file = self.file  # keep reference alive

        upload = _PathUploadFile(pdf_path)
        exec_svc = AgentExecutionService()
        return await exec_svc.execute_agent_ocr(
            agent_id=agent_id,
            pdf_file=upload,
            user_context=user_context,
            for_api=True,
            db=db,
        )
    finally:
        db.close()

        if upload is not None:
            upload.file.close()


class IACTOCRTool(BaseTool):
    """
    Tool wrapper for OCR agents.

    PDF documents are discovered automatically from
    self.attached_files.

    When one or more PDFs are attached, OCR processing is executed.
    Otherwise the tool falls back to normal chat behaviour.
    """

    name: str = "ocr_agent_tool"
    description: str = "OCR agent tool for extracting text from PDF documents"
    args_schema: Type[BaseModel] = AgentToolInput
    agent: Agent
    user_context: Optional[Dict] = None
    react_agent: Any = None
    mcp_client: Any = None
    llm: Any = None
    has_memory: bool = False
    memory_max_messages: int = 20
    memory_max_tokens: int = 4000
    memory_summarize_threshold: int = DEFAULT_MEMORY_SUMMARIZE_THRESHOLD
    output_parser_id: Optional[int] = None
    temperature: float = DEFAULT_AGENT_TEMPERATURE
    attached_files: Optional[List[Dict]] = None

    def __init__(
        self,
        agent: Agent,
        user_context: Optional[Dict] = None,
        attached_files: Optional[List[Dict]] = None,
    ) -> None:
        super().__init__(agent=agent, user_context=user_context)
        self.agent = agent
        self.user_context = user_context
        self.attached_files = attached_files or []
        self.name = sanitize_identifier(agent.name)
        self.description = agent.description or "OCR agent tool"
        try:
            self.llm = get_llm(agent, is_vision=False)
        except Exception:
            self.llm = None
        if self.llm is None:
            logger.warning(
                "OCR agent %s has no configured LLM; OCR will not work until service_id is set",
                agent.name,
            )
        self.react_agent = None
        self.mcp_client = None
        self.has_memory = getattr(agent, "has_memory", False) or False
        self.memory_max_messages = getattr(agent, "memory_max_messages", 20) or 20
        self.memory_max_tokens = getattr(agent, "memory_max_tokens", 4000)
        self.memory_summarize_threshold = (
            getattr(agent, "memory_summarize_threshold", DEFAULT_MEMORY_SUMMARIZE_THRESHOLD)
            or DEFAULT_MEMORY_SUMMARIZE_THRESHOLD
        )
        self.output_parser_id = getattr(agent, "output_parser_id", None)
        self.temperature = getattr(agent, "temperature", DEFAULT_AGENT_TEMPERATURE) or DEFAULT_AGENT_TEMPERATURE

    @classmethod
    async def create(
        cls,
        agent: Agent,
        user_context: Optional[Dict] = None,
        attached_files: Optional[List[Dict]] = None,
    ) -> "IACTOCRTool":
        """Build an OCR agent-as-tool with MCP support.

        Similar to ``IACTTool.create`` but adds OCR-specific validation
        and tools.  A failing MCP server degrades the sub-agent but never
        breaks construction.
        """
        instance = cls(agent, user_context=user_context, attached_files=attached_files)

        tools: list = [fetch_file_in_base64]

        # Nested tool agents — only recurse into non-OCR agents, because
        for t in agent.tool_associations:
            sub_agent = t.tool
            nested = await discover_tool(sub_agent, user_context=user_context, attached_files=attached_files)
            tools.append(nested)

        # MCP tools
        try:
            logger.info(f"Starting MCP tools loading for OCR sub-agent {agent.agent_id}...")
            instance.mcp_client = await MCPClientManager().get_client(agent, user_context)
            if instance.mcp_client:
                mcp_tools = await instance.mcp_client.get_tools()
                logger.info(
                    f"MCP tools loaded successfully for OCR sub-agent {agent.agent_id}: "
                    f"{len(mcp_tools)} tools"
                )
                if mcp_tools:
                    tools.extend(mcp_tools)
        except Exception as e:
            logger.error(
                f"Error loading MCP tools for OCR sub-agent {agent.agent_id}: {e}",
                exc_info=True,
            )
            instance.mcp_client = None

        # System prompt — same pattern as IACTTool
        tool_system_prompt = agent.system_prompt or ""
        current_date = datetime.now().strftime("%Y-%m-%d")
        tool_system_prompt += f"\n\nToday's date is {current_date}."
        if agent.system_prompt and hasattr(agent, "skill_associations") and agent.skill_associations:
            skills_section = generate_skills_system_prompt_section(agent.skill_associations)
            if skills_section:
                tool_system_prompt = tool_system_prompt + "\n" + skills_section

        instance.react_agent = create_langchain_agent(
            model=instance.llm,
            tools=tools,
            system_prompt=tool_system_prompt if tool_system_prompt else None,
        )
        return instance

    def _run(self, query: str, **kwargs: Any) -> str:
        """Synchronous execution of the OCR agent tool."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._arun(query=query)
            )
        finally:
            loop.close()

    async def _arun(
        self,
        query: str,
        **kwargs: Any,
    ) -> str:
        """Asynchronous execution of the OCR agent tool."""

        if self.react_agent is None:
            raise RuntimeError(
                "IACTOCRTool must be built via 'await IACTOCRTool.create(...)' before use."
            )

        pdf_files = [
            f for f in self.attached_files
            if f.get("type") == "pdf"
        ]

        #
        # OCR PATH
        #
        if pdf_files:

            results = []

            for pdf_file in pdf_files:

                pdf_path = pdf_file.get("file_path")
                filename = pdf_file.get("filename", "unknown.pdf")

                if not pdf_path:
                    results.append({
                        "file": filename,
                        "error": "PDF file path not found"
                    })
                    continue

                if not os.path.isabs(pdf_path):
                    pdf_path = os.path.join(
                        get_app_config()["TMP_BASE_FOLDER"],
                        pdf_path
                    )

                if not os.path.exists(pdf_path):
                    results.append({
                        "file": filename,
                        "error": f"PDF file not found: {pdf_path}"
                    })
                    continue

                try:

                    ocr_result = await _execute_tool_agent_ocr(
                        agent_id=self.agent.agent_id,
                        pdf_path=pdf_path,
                        user_context=self.user_context,
                    )

                    results.append({
                        "file": filename,
                        "content": (
                            ocr_result.get("content")
                            if isinstance(ocr_result, dict)
                            else ocr_result
                        )
                    })

                except Exception as exc:

                    logger.exception(
                        "OCR failed for %s",
                        filename
                    )

                    results.append({
                        "file": filename,
                        "error": str(exc)
                    })

            return json.dumps(
                results,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        #
        # CHAT FALLBACK PATH
        #
        if self.agent.prompt_template:
            try:
                formatted_prompt = self.agent.prompt_template.format(
                    question=query
                )
            except KeyError:
                try:
                    formatted_prompt = self.agent.prompt_template.format(
                        query=query
                    )
                except KeyError:
                    logger.warning(
                        "Could not format prompt_template for OCR agent %s, using query directly",
                        self.agent.name,
                    )
                    formatted_prompt = query
        else:
            formatted_prompt = query

        messages = [
            HumanMessage(content=formatted_prompt)
        ]

        try:

            result = await self.react_agent.ainvoke(
                {"messages": messages}
            )

            if isinstance(result, dict) and "messages" in result:

                messages_list = result["messages"]

                for msg in reversed(messages_list):
                    if hasattr(msg, "content") and msg.content:
                        return str(msg.content)

                if messages_list:
                    last_msg = messages_list[-1]

                    return (
                        str(last_msg.content)
                        if hasattr(last_msg, "content")
                        else str(last_msg)
                    )

            return str(result)

        except Exception as e:

            logger.error(
                "Error executing OCR chat fallback: %s",
                e,
            )

            return f"Error executing agent tool: {str(e)}"


async def discover_tool(
    agent: Agent,
    user_context: Optional[Dict] = None,
    working_dir: Optional[str] = None,
    sandbox_handle: Optional[Any] = None,
    sandbox_provider: Optional[Any] = None,
    sandbox_session_key: Optional[str] = None,
    sandbox_session_service: Optional[Any] = None,
    attached_files: Optional[List[Dict]] = None,
) -> BaseTool:
    """Return the appropriate tool wrapper for *agent*.

    Routes to :class:`IACTOCRTool` when the agent is an OCR agent
    (``type == 'ocr_agent'``), otherwise to :class:`IACTTool`. Sandbox
    parameters are only meaningful for :class:`IACTTool` — OCR agents
    don't support code interpreter.
    """
    if agent.type == "ocr_agent":
        return await IACTOCRTool.create(agent, user_context=user_context, attached_files=attached_files)
    return await IACTTool.create(
        agent,
        user_context=user_context,
        working_dir=working_dir,
        sandbox_handle=sandbox_handle,
        sandbox_provider=sandbox_provider,
        sandbox_session_key=sandbox_session_key,
        sandbox_session_service=sandbox_session_service,
        attached_files=attached_files,
    )


_SEARCH_ERROR_MSG = "The knowledge base search failed; try rephrasing or removing filters."


# --- LightRAG inline citations ---------------------------------------------
# When a LightRAG retriever surfaces document chunks, we number them and ask the
# LLM to cite statements inline as [N](cite://N) Markdown links. N is the chunk's
# 1-based position in lightrag_raw_data.data.chunks — the same order the frontend
# renders chunk chips (LightRAGGraphBubble) and resolves cite://N against, so a
# citation opens the exact source chunk.
# Numbering is global across every retrieval call within the same turn (see the
# `offset` mutable-cell param below) and agent_streaming_service.py merges every
# _lightrag_graph event of the turn instead of keeping only the last one — so a
# turn with several retrieval calls (e.g. multi-silo) still resolves correctly.
_CITATION_EXCERPT_CHARS = 240

_CITATION_INSTRUCTION = (
    "The knowledge base returned the numbered sources below. When a sentence in "
    "your answer uses information from a source, append an inline citation "
    "immediately after that sentence using this exact Markdown syntax: "
    "[N](cite://N), where N is the source number. Cite only sources you actually "
    "used; combine several as [1](cite://1)[2](cite://2) when a sentence draws on "
    "more than one. Do not add a separate reference list at the end. "
    "In a bulleted or numbered list where every item comes from the same source, "
    "repeat the FULL [N](cite://N) link on every single item — never shorten it to "
    "just [N] after the first occurrence, even though it is the same source. A "
    "bare [N] with no (cite://N) is plain text, not a citation, and will not work. "
    "Each source's label already includes its document name and page number "
    "(e.g. 'CDOC004043.pdf p.63') — when the question asks WHERE something is "
    "documented (which page, which section), state that document name and page "
    "number explicitly in your answer text, not just as an inline citation marker. "
    "When a question asks for a superlative or single best answer across several "
    "product families or categories (e.g. 'which has the highest/most/largest...'), "
    "check every family your sources cover BEFORE writing the answer, then lead "
    "with whichever one actually wins — never present the first family you happened "
    "to check as the main answer and mention a bigger or more correct one only in a "
    "trailing note. If you are genuinely unsure a footnoted alternative should count "
    "(e.g. a different product line), say so plainly, but do not bury the correct "
    "winner below an incorrect headline answer."
)


def _append_lightrag_citation_sources(
    content: str, docs: List[Document], offset: Optional[List[int]] = None
) -> str:
    """Append a numbered SOURCES block + citation instruction for LightRAG chunks.

    No-op when *docs* carry no LightRAG chunk provenance (e.g. PGVector/Qdrant
    silos), so it is safe to call on any retriever path. Numbering mirrors
    ``lightrag_raw_data.data.chunks`` order 1:1 (no dedup) to match the chunk
    order the frontend renders and resolves ``cite://N`` against.

    offset: same mutable-cell pattern as ``_call_count`` — pass the SAME
    single-item list across multiple retrieval calls in one turn so numbering
    continues (4, 5, 6...) instead of restarting at 1 each call. Omit for the
    old single-call behavior.
    """
    counter = offset if offset is not None else [0]
    lines: List[str] = []
    for doc in docs:
        raw = (doc.metadata or {}).get("lightrag_raw_data") or {}
        data = raw.get("data") or {}
        for chunk in (data.get("chunks") or []):
            counter[0] += 1
            src = chunk.get("file_path") or "Unknown source"
            excerpt = " ".join((chunk.get("content") or "").split())[:_CITATION_EXCERPT_CHARS]
            lines.append(f"[{counter[0]}] (source: {src}) {excerpt}")
    if not lines:
        return content
    return f"{content}\n\n{_CITATION_INSTRUCTION}\n\n" + "\n".join(lines)


def _format_docs_with_metadata(docs: List[Document]) -> str:
    """Serialize retrieved documents as a text block with metadata for the LLM."""
    parts: List[str] = []
    for doc in docs:
        metadata_str = json.dumps(doc.metadata, ensure_ascii=False) if doc.metadata else "{}"
        parts.append(f"Content: {doc.page_content}\nMetadata: {metadata_str}")
    return "\n\n---\n\n".join(parts)


def _capture_silo_data(silo: Silo) -> dict:
    """Extract all silo ORM data needed by the retrieval coroutine into plain values.

    After this call the coroutine never accesses the ORM object or its lazy
    relationships, avoiding DetachedInstanceError.
    """
    metadata_definition = getattr(silo, "metadata_definition", None)
    captured_fields_list: List[dict] = []
    if metadata_definition is not None:
        for fspec in (metadata_definition.fields or []):
            if isinstance(fspec, dict) and fspec.get("name"):
                captured_fields_list.append(dict(fspec))

    return {
        "silo_id": silo.silo_id,
        "vector_db_type": getattr(silo, "vector_db_type", None) or "PGVECTOR",
        "captured_fields_list": captured_fields_list,
        "captured_metadata_def": (
            _types.SimpleNamespace(fields=captured_fields_list)
            if captured_fields_list
            else None
        ),
        "metadata_field_types": {
            f["name"]: f.get("type", "str") for f in captured_fields_list
        },
    }


def _build_pinned_filter(
    raw_caller_filter: dict,
    captured_metadata_def: Any,
    vector_db_type: str,
) -> dict:
    """Convert the caller's flat {field: value} filter to a backend filter dict.

    Skips the field whitelist (validate_clauses) so undeclared fields pass through —
    pinned filters come from trusted caller code, not the LLM. Type coercion is applied
    using declared field types; undeclared fields are treated as str.
    """
    from tools.vector_stores.metadata_filters import (
        MetadataFilterClause,
        convert_clause_types,
        to_backend_filter,
    )

    clauses: List[MetadataFilterClause] = []
    for field, value in raw_caller_filter.items():
        try:
            clauses.append(MetadataFilterClause(field=field, op="$eq", value=value))
        except Exception:
            logger.warning(
                "get_retriever_tool: could not build pinned clause for field '%s' — skipped",
                field,
            )

    if not clauses:
        return {}

    typed = convert_clause_types(clauses, captured_metadata_def)
    return to_backend_filter(typed)


def _build_llm_filter(
    metadata_kwargs: dict,
    metadata_field_types: dict,
    captured_metadata_def: Any,
    vector_db_type: str,
    tool_name: str,
) -> dict:
    """Build a backend filter dict from LLM-inferred kwargs.

    Applies the strict field whitelist (only declared fields pass) — LLM input
    is untrusted.
    """
    from tools.vector_stores.metadata_filters import MetadataFilterClause, build_filter_dict

    llm_clauses: List[MetadataFilterClause] = []
    for field, value in metadata_kwargs.items():
        if value is None:
            continue
        if field not in metadata_field_types:
            logger.warning(
                "get_retriever_tool[%s]: field '%s' not in metadata_definition — skipped (AC-5)",
                tool_name,
                field,
            )
            continue
        try:
            llm_clauses.append(MetadataFilterClause(field=field, op="$eq", value=value))
        except Exception:
            logger.warning(
                "get_retriever_tool[%s]: could not build LLM clause for field '%s' — skipped",
                tool_name,
                field,
            )

    if not llm_clauses:
        return {}
    return build_filter_dict(llm_clauses, captured_metadata_def, vector_db_type)


async def _run_retrieval(
    silo_id: int,
    call_search_params: Optional[dict],
    query: str,
) -> List[Document]:
    """Invoke SiloService.get_silo_retriever off the event loop, then run ainvoke."""
    retriever = await asyncio.to_thread(
        SiloService.get_silo_retriever, silo_id, call_search_params
    )
    return await retriever.ainvoke(query)


async def _build_fallback_notice(
    silo_id: int,
    llm_filter_fields: List[str],
    distinct_values: dict,
    tool_name: str,
) -> str:
    """Build the [notice] string for AC-8 fallback, fetching missing values via thread."""
    from tools.vector_stores.metadata_filters import sanitize_metadata_value, MAX_EXAMPLE_VALUES
    from services.metadata_values_cache_service import MetadataValuesCacheService

    def _fetch(f: str) -> List[str]:
        with SessionLocal() as s:
            return MetadataValuesCacheService.get_distinct_values(
                silo_id=silo_id, field=f, db=s
            )

    notice_parts: List[str] = []
    for field in llm_filter_fields:
        cached_vals = distinct_values.get(field, [])
        if not cached_vals:
            try:
                cached_vals = await asyncio.to_thread(_fetch, field)
            except Exception:
                logger.warning(
                    "get_retriever_tool[%s]: could not fetch distinct values for field '%s'",
                    tool_name,
                    field,
                )
                cached_vals = []

        sanitized = [
            sanitize_metadata_value(str(v))
            for v in cached_vals
            if v is not None
        ]
        sanitized = [v for v in sanitized if v][:MAX_EXAMPLE_VALUES]
        if sanitized:
            notice_parts.append(f"{field}: {', '.join(sanitized)}")

    notice = (
        f"[notice] No results with the inferred filter {llm_filter_fields}; "
        f"retried without it."
    )
    if notice_parts:
        notice += " Existing values — " + "; ".join(notice_parts) + "."
    return notice


def get_retriever_tool(
    silo: Silo,
    search_params: Optional[dict] = None,
    max_retrieval_calls: Optional[int] = None,
    pinned_filter: Optional[dict] = None,
) -> Optional[StructuredTool]:
    """Build the dynamic retrieval tool for *silo*.

    All silo ORM state is captured in plain variables at construction time.
    The coroutine never accesses the ORM object directly, avoiding DetachedInstanceError.

    Args:
        silo: Attached Silo ORM instance.
        search_params: Optional caller-level search parameters (tuning only — no
            'filter' key expected when ``pinned_filter`` is provided).
        max_retrieval_calls: Optional ceiling on tool invocations per agent turn.
        pinned_filter: Optional pre-built backend filter dict
            ``{field: {op: value}}``.  When provided it is used directly as the
            pinned filter and ``_build_pinned_filter`` is skipped.  When None the
            existing behaviour is preserved: the ``filter`` key from
            ``search_params`` is translated via ``_build_pinned_filter``.

    Returns:
        A StructuredTool whose coroutine performs metadata-aware retrieval,
        or None when silo.silo_id is falsy.
    """
    if not silo.silo_id:
        return None

    from tools.retriever_tool_builder import (
        build_retriever_args_schema,
        build_retriever_description,
        build_retriever_tool_name,
        collect_distinct_values,
    )
    from tools.vector_stores.metadata_filters import merge_filters_and

    captured = _capture_silo_data(silo)
    silo_id: int = captured["silo_id"]
    vector_db_type: str = captured["vector_db_type"]
    captured_fields_list: List[dict] = captured["captured_fields_list"]
    captured_metadata_def = captured["captured_metadata_def"]
    metadata_field_types: dict[str, str] = captured["metadata_field_types"]

    with SessionLocal() as db_session:
        distinct_values: dict[str, List[str]] = collect_distinct_values(silo, db=db_session)

    effective_search_params: Optional[dict] = search_params

    if pinned_filter is not None:
        # Pre-built filter supplied by resolve_search_params — use directly.
        _pinned_filter: dict[str, Any] = pinned_filter
        if _pinned_filter:
            effective_search_params = {**(search_params or {}), "filter": _pinned_filter}
    else:
        # Legacy path: translate search_params["filter"] flat dict.
        _pinned_filter = {}
        if search_params and search_params.get("filter"):
            _pinned_filter = _build_pinned_filter(
                search_params["filter"], captured_metadata_def, vector_db_type
            )
            if _pinned_filter:
                effective_search_params = {**search_params, "filter": _pinned_filter}

    # Alias for closure capture
    resolved_pinned = _pinned_filter

    tool_name: str = build_retriever_tool_name(silo)
    tool_description: str = build_retriever_description(silo, distinct_values)
    args_schema = build_retriever_args_schema(silo, distinct_values)

    # Not async-safe under parallel tool calls; safe with LangGraph's serialized model.
    _call_count: List[int] = [0]
    # Shared across calls within the same turn — see _append_lightrag_citation_sources.
    _citation_offset: List[int] = [0]

    async def _search(query: str, **metadata_kwargs: Any) -> Tuple[str, List[Document]]:
        if max_retrieval_calls is not None and _call_count[0] >= max_retrieval_calls:
            logger.info(
                "get_retriever_tool[%s]: retrieval ceiling reached (%d/%d)",
                tool_name, _call_count[0], max_retrieval_calls,
            )
            return (
                "Search limit reached — answer with the information you already have "
                "or state what is missing.",
                [],
            )
        _call_count[0] += 1

        llm_filter = _build_llm_filter(
            metadata_kwargs, metadata_field_types, captured_metadata_def,
            vector_db_type, tool_name,
        )
        merged_filter = merge_filters_and(resolved_pinned, llm_filter)

        if merged_filter:
            base = dict(effective_search_params) if effective_search_params else {}
            call_search_params: Optional[dict] = {**base, "filter": merged_filter}
        else:
            call_search_params = effective_search_params

        applied_fields = list(merged_filter.keys()) if merged_filter else []
        logger.info(
            "get_retriever_tool[%s]: call #%d — applied filter fields=%s",
            tool_name, _call_count[0], applied_fields or "(none)",
        )

        try:
            docs: List[Document] = await _run_retrieval(silo_id, call_search_params, query)
        except Exception as exc:
            logger.error(
                "get_retriever_tool[%s]: vector store error",
                tool_name, exc_info=True,
            )
            return (_SEARCH_ERROR_MSG, [])

        if not docs and llm_filter:
            logger.info(
                "get_retriever_tool[%s]: 0 results with LLM filter — retrying with pinned only (AC-8)",
                tool_name,
            )
            notice = await _build_fallback_notice(
                silo_id, list(llm_filter.keys()), distinct_values, tool_name
            )

            try:
                docs = await _run_retrieval(silo_id, effective_search_params, query)
            except Exception as exc:
                logger.error(
                    "get_retriever_tool[%s]: fallback vector store error",
                    tool_name, exc_info=True,
                )
                return (_SEARCH_ERROR_MSG, [])

            filter_label = f"with filter {resolved_pinned}" if resolved_pinned else "(no metadata filter)"
            content = (
                f"{notice}\n\n"
                f"{len(docs)} results {filter_label}\n\n"
                f"{_format_docs_with_metadata(docs)}"
            )
            content = _append_lightrag_citation_sources(content, docs, _citation_offset)
            logger.info(
                "get_retriever_tool[%s]: fallback retrieved %d docs for silo_id=%d",
                tool_name, len(docs), silo_id,
            )
            return (content, docs)

        filter_label = f"with filter {merged_filter}" if merged_filter else "(no metadata filter)"
        content = (
            f"{len(docs)} results {filter_label}\n\n"
            f"{_format_docs_with_metadata(docs)}"
        )
        content = _append_lightrag_citation_sources(content, docs, _citation_offset)
        logger.info(
            "get_retriever_tool[%s]: retrieved %d docs for silo_id=%d, filter_fields=%s",
            tool_name, len(docs), silo_id, applied_fields or "(none)",
        )
        return (content, docs)

    return StructuredTool.from_function(
        coroutine=_search,
        name=tool_name,
        description=tool_description,
        args_schema=args_schema,
        response_format="content_and_artifact",
    )

