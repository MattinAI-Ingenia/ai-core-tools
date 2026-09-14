"""
AgentExecutionContext — plain data container produced by _prepare_turn().

Carries every field that the execution and finalization phases need so that
_prepare_turn, _execute_agent_async, and _finalize_turn can pass state without
a growing argument list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentExecutionContext:
    """Immutable snapshot of all setup state for one agent chat turn."""

    # Core identity
    agent_id: int
    agent: Any                          # Agent ORM instance (lightweight, pre-access-check)
    fresh_agent: Any                    # Agent ORM instance with all relationships loaded

    # Message
    enhanced_message: str               # Text message after file-content injection
    image_files: List[Dict[str, Any]]   # Image file dicts extracted from processed_files

    # Session / conversation
    session: Optional[Any] = None       # SessionManagementService session object
    conversation: Optional[Any] = None  # Conversation ORM instance (None when no memory)
    effective_conv_id: Optional[int] = None  # Resolved conversation ID (auto-created or passed)
    session_id_for_cache: Optional[str] = None  # LangGraph thread ID suffix

    # Working directory
    working_dir: Optional[str] = None
    pre_existing_files: set = field(default_factory=set)

    # Sandbox (IT-4) — populated when enable_code_interpreter is True for non-subprocess providers
    sandbox_handle: Optional[Any] = None           # SandboxHandle (typed as Any to avoid circular import)
    sandbox_provider: Optional[Any] = None         # SandboxProvider instance
    sandbox_session_key: Optional[str] = None      # Key used with SandboxSessionService
    pre_existing_remote_files: set = field(default_factory=set)  # Snapshot before turn for remote pull diff

    # Original inputs (needed by finalize for metadata)
    processed_files: List[Dict[str, Any]] = field(default_factory=list)
    search_params: Optional[Dict[str, Any]] = None
    user_context: Optional[Dict[str, Any]] = None

    # Cross-session global user memory (spec `user-memory-mem0`, FR-6/FR-7,
    # AD-6). Populated by `_prepare_turn`'s memory-activation step; all
    # `None` when memory is inactive for this turn (kill switch off, flag
    # off, read not due, ineligible identity, or unresolved services — every
    # negative outcome degrades to "no memory", never an error, NFR-4).
    # Typed `Any` to avoid import cycles (matches the `sandbox_handle`
    # convention above).
    global_memory_identity: Optional[Any] = None        # tools.memory.identity.MemoryIdentity
    global_memory_services: Optional[Any] = None         # tools.memory.resolution.ResolvedMemoryServices
    global_memory_llm: Optional[Any] = None              # extraction LLM, reused by the write path (step_008)
    global_memory_embeddings: Optional[Any] = None       # embeddings client used for the read-path search
    memory_context: Optional[str] = None                 # rendered "remembered context" block, or None
    # 1-based turn index (AD-13, `tools.memory.settings.turn_index`),
    # resolved once in `_prepare_turn` from `Conversation.message_count`
    # BEFORE `_finalize_turn` increments it for this turn. The write hook
    # (step_008) must read this value rather than recomputing `turn_index()`
    # of its own accord: `_finalize_turn` increments `message_count` before
    # its write-scheduling step runs, so a recompute there would see one
    # turn ahead and silently break the `turn_index == 1` write trigger
    # (R-16). Always populated (never None) once `_prepare_turn` returns,
    # independent of whether memory itself is active this turn.
    memory_turn_index: Optional[int] = None
