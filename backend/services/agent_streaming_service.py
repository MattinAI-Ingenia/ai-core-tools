"""
Streaming agent execution service.

A thin SSE adapter over AgentExecutionService.  The setup and post-processing
phases are fully delegated to AgentExecutionService._prepare_turn() and
_finalize_turn(); this service only owns the astream loop that yields tokens
and tool events to the client.
"""

from typing import AsyncGenerator, Dict, List, Any

import psycopg.errors
from sqlalchemy.orm import Session

from tools.agentTools import create_agent, prepare_agent_config, build_human_message
from tools.langsmith_config import (
    apply_tracing_to_config,
    build_tracing_config,
    resolve_langsmith_settings,
)
from tools.streaming_utils import (
    format_sse_event,
    map_stream_event,
    SSE_TOKEN,
    SSE_HITL_INTERRUPT,
)
from services.agent_execution_service import AgentExecutionService
from utils.logger import get_logger

logger = get_logger(__name__)


def _emit_monitoring_log(
    agent_id: int,
    monitoring_handler,
    monitoring_config: dict | None,
    log_fn,
) -> None:
    """Emit the [Monitoring] log line, filtering to only selected metrics.

    Args:
        agent_id: The agent being monitored.
        monitoring_handler: _CountingUsageMetadataCallbackHandler instance
            (a UsageMetadataCallbackHandler subclass that also tracks call_count).
        monitoring_config: The ``config`` dict from the Monitoring middleware
            entity, or None.  Absent or None means all metrics enabled.
        log_fn: callable used for logging (e.g. logger.info).
    """
    try:
        usage_by_model = monitoring_handler.usage_metadata
        metrics_cfg: dict = (monitoring_config or {}).get("metrics", {})

        # Default-on: absent key => metric is enabled
        def enabled(key: str) -> bool:
            return metrics_cfg.get(key, True)

        parts: list[str] = [f"[Monitoring] agent_id={agent_id}"]

        if enabled("models"):
            parts.append(f"models={list(usage_by_model.keys())}")
        if enabled("input_tokens"):
            total = sum(u.get("input_tokens", 0) for u in usage_by_model.values())
            parts.append(f"input_tokens={total}")
        if enabled("output_tokens"):
            total = sum(u.get("output_tokens", 0) for u in usage_by_model.values())
            parts.append(f"output_tokens={total}")
        if enabled("total_tokens"):
            total = sum(u.get("total_tokens", 0) for u in usage_by_model.values())
            parts.append(f"total_tokens={total}")
        if enabled("llm_calls"):
            call_count = getattr(monitoring_handler, "call_count", len(usage_by_model))
            parts.append(f"llm_calls={call_count}")

        if len(parts) > 1:  # more than just the prefix
            log_fn(" | ".join(parts))
    except Exception as monitor_err:
        import logging
        logging.getLogger(__name__).warning(f"Error reading monitoring metrics: {monitor_err}")


class AgentStreamingService:
    """Service for streaming agent responses via Server-Sent Events."""

    def __init__(self, db: Session = None) -> None:
        self.execution_service = AgentExecutionService()
        self.db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def stream_agent_chat(
        self,
        agent_id: int,
        message: str,
        file_references: list | None = None,
        search_params: dict | None = None,
        user_context: dict | None = None,
        conversation_id: int | None = None,
        db: Session | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream an agent chat turn as SSE events.

        Yields ``format_sse_event`` strings for each event in the following
        sequence:

        1. ``metadata`` — emitted immediately after setup with conversation/agent
           metadata so the client can bind the conversation ID before tokens
           arrive.
        2. ``thinking`` / ``tool_start`` / ``tool_end`` — emitted while the agent
           reasons and calls tools.
        3. ``token`` — one per partial LLM text chunk.
        4. ``done`` — emitted once after the stream finishes, carrying the full
           parsed response, conversation ID, and any generated files.
        5. ``error`` — emitted instead of ``done`` if an unhandled exception
           occurs.

        Args:
            agent_id: Primary key of the agent to execute.
            message: The user's text message.
            file_references: Pre-resolved file-reference objects as returned by
                ``FileManagementService``.  Each object must expose
                ``filename``, ``content``, ``file_type``, ``file_id``, and
                ``file_path``.
            search_params: Optional silo search parameters forwarded to
                ``create_agent``.
            user_context: Caller context dict (``user_id``, ``app_id``,
                ``email``, …).
            conversation_id: ID of an existing conversation to continue.  When
                ``None`` and the agent has memory enabled a new conversation is
                created automatically.
            db: SQLAlchemy session.  If omitted the instance-level ``self.db``
                is used.

        Yields:
            SSE-formatted strings (``"data: {...}\\n\\n"``).
        """
        effective_db = db or self.db
        mcp_client = None

        try:
            # ----------------------------------------------------------------
            # 1. Setup phase — delegates entirely to AgentExecutionService
            # ----------------------------------------------------------------
            ctx = await self.execution_service._prepare_turn(
                agent_id=agent_id,
                message=message,
                file_references=file_references,
                search_params=search_params,
                user_context=user_context,
                conversation_id=conversation_id,
                db=effective_db,
            )

            # ----------------------------------------------------------------
            # 2. Emit early metadata event so the client has conversation_id
            # ----------------------------------------------------------------
            yield format_sse_event(
                "metadata",
                {
                    "conversation_id": ctx.effective_conv_id,
                    "agent_id": agent_id,
                    "agent_name": ctx.agent.name,
                    "has_memory": ctx.agent.has_memory,
                },
            )

            # ----------------------------------------------------------------
            # 3. Build agent chain
            # ----------------------------------------------------------------
            agent_chain, mcp_client, monitoring_handler = await create_agent(
                ctx.fresh_agent,
                ctx.search_params,
                ctx.session_id_for_cache,
                ctx.user_context,
                ctx.working_dir,
            )

            config = prepare_agent_config(ctx.fresh_agent)

            if ctx.fresh_agent.has_memory and ctx.session_id_for_cache:
                config["configurable"]["thread_id"] = (
                    f"thread_{ctx.fresh_agent.agent_id}_{ctx.session_id_for_cache}"
                )
                logger.info(
                    "Using session-aware thread_id: %s",
                    config["configurable"]["thread_id"],
                )
            else:
                config["configurable"]["thread_id"] = (
                    f"thread_{ctx.fresh_agent.agent_id}"
                )

            config["configurable"]["question"] = ctx.enhanced_message

            # Attach monitoring callback if enabled
            if monitoring_handler is not None:
                config.setdefault("callbacks", []).append(monitoring_handler)

            # ----------------------------------------------------------------
            # 4. Build the HumanMessage payload (handles multimodal images)
            # ----------------------------------------------------------------
            message_payload = build_human_message(
                ctx.fresh_agent, ctx.enhanced_message, ctx.image_files, ctx.user_context
            )

            # ----------------------------------------------------------------
            # 5. Attach LangSmith tracer + metadata when configured
            # ----------------------------------------------------------------
            ls_settings = resolve_langsmith_settings(ctx.fresh_agent.app)
            if ls_settings:
                tracer, overrides = build_tracing_config(
                    ls_settings,
                    agent=ctx.fresh_agent,
                    user_context=ctx.user_context,
                    conversation_id=ctx.effective_conv_id,
                    session_id=ctx.session_id_for_cache,
                )
                apply_tracing_to_config(config, tracer, overrides)
                logger.info(
                    "LangSmith tracing ENABLED — project='%s' source='%s'",
                    ls_settings.project_name,
                    ls_settings.source,
                )

            # ----------------------------------------------------------------
            # 6. Streaming loop — the only part that stays in this service
            # ----------------------------------------------------------------
            # Return the sync connection to the pool for the duration of the
            # stream: astream uses the async checkpointer, not this session, so
            # holding it across LLM I/O would exhaust the pool. ctx objects expire
            # but stay attached, so _finalize_turn reloads them on demand.
            if effective_db is not None:
                effective_db.commit()

            accumulated_content = ""
            structured_response = None

            async for mode, chunk in agent_chain.astream(
                {"messages": [message_payload]},
                config=config,
                stream_mode=["messages", "updates", "custom"],
            ):

                if mode == "updates":
                    if (
                        isinstance(chunk, dict)
                        and "model" in chunk
                        and isinstance(chunk["model"], dict)
                        and "structured_response" in chunk["model"]
                    ):
                        structured_response = chunk["model"]["structured_response"]

                events = map_stream_event(mode, chunk)
                if events:
                    for event in events:
                        if event["type"] == SSE_TOKEN:
                            accumulated_content += event["data"].get("content", "")
                        yield format_sse_event(event["type"], event["data"])

            raw_response = (
                structured_response
                if structured_response is not None
                else accumulated_content
            )
            logger.info("Stream completed — accumulated_content length=%d", len(accumulated_content))

            # Check for pending interrupts after stream completes
            has_pending_interrupt = False
            try:
                graph_state = await agent_chain.aget_state(config)
                if hasattr(graph_state, 'tasks'):
                    for task in graph_state.tasks:
                        if hasattr(task, 'interrupts') and task.interrupts:
                            has_pending_interrupt = True
                            logger.info("PENDING INTERRUPT found in graph state: %s", task.interrupts)
                            for intr in task.interrupts:
                                payload = intr.value if hasattr(intr, 'value') else intr
                                action_requests = []
                                review_configs = []
                                if isinstance(payload, dict):
                                    action_requests = payload.get("action_requests", [])
                                    review_configs = payload.get("review_configs", [])
                                yield format_sse_event(
                                    "hitl_interrupt",
                                    {
                                        "action_requests": action_requests,
                                        "review_configs": review_configs,
                                    },
                                )
            except Exception as state_err:
                logger.warning("Could not check graph state for interrupts: %s", state_err)

            # If HITL interrupted, emit done with interrupt message and skip normal finalization
            if has_pending_interrupt:
                yield format_sse_event(
                    "done",
                    {
                        "response": "⏸️ Execution paused — awaiting human approval.",
                        "conversation_id": ctx.effective_conv_id,
                        "files": [],
                        "hitl_paused": True,
                    },
                )
            else:
                # ----------------------------------------------------------------
                # 7. Post-processing phase — delegates to AgentExecutionService
                # ----------------------------------------------------------------

                # Log usage metrics if monitoring is enabled
                if monitoring_handler is not None:
                    monitoring_mw_config = None
                    for _assoc in (getattr(ctx.fresh_agent, 'middleware_associations', None) or []):
                        if _assoc.middleware and _assoc.middleware.middleware_type.value == 'monitoring':
                            monitoring_mw_config = _assoc.middleware.config or {}
                            break
                    _emit_monitoring_log(ctx.fresh_agent.agent_id, monitoring_handler, monitoring_mw_config, logger.info)

                result = await self.execution_service._finalize_turn(
                    ctx, raw_response, effective_db
                )

                # ----------------------------------------------------------------
                # 8. Emit done event
                # ----------------------------------------------------------------
                yield format_sse_event(
                    "done",
                    {
                        "response": result["parsed_response"],
                        "conversation_id": result["effective_conv_id"],
                        "files": result["files_data"],
                    },
                )

        except (
            psycopg.errors.AdminShutdown,
            psycopg.errors.ConnectionFailure,
            psycopg.OperationalError,
        ) as exc:
            # Stale pool connection terminated by PostgreSQL (e.g. server
            # restart or pg_terminate_backend). The pool discards the bad
            # connection automatically; a single retry will receive a fresh one.
            logger.warning(
                "Checkpointer connection lost (%s), retrying once: %s",
                type(exc).__name__,
                str(exc),
            )
            yield format_sse_event("error", {"message": "Connection error, please retry."})
        except Exception as exc:
            # Check if this is a GraphInterrupt from HumanInTheLoop middleware
            from langgraph.errors import GraphInterrupt
            if isinstance(exc, GraphInterrupt):
                interrupts = getattr(exc, "interrupts", [])
                logger.info(
                    "GraphInterrupt caught — HITL middleware paused execution. "
                    "Interrupts: %s", interrupts
                )
                # Emit HITL interrupt event to the frontend
                for intr in interrupts:
                    payload = intr.value if hasattr(intr, "value") else intr
                    action_requests = []
                    review_configs = []
                    if isinstance(payload, dict):
                        action_requests = payload.get("action_requests", [])
                        review_configs = payload.get("review_configs", [])
                    yield format_sse_event(
                        "hitl_interrupt",
                        {
                            "action_requests": action_requests,
                            "review_configs": review_configs,
                        },
                    )
                # Emit done with a placeholder so the UI shows the interrupt message
                yield format_sse_event(
                    "done",
                    {
                        "response": "⏸️ Execution paused — awaiting human approval.",
                        "conversation_id": ctx.effective_conv_id if ctx else None,
                        "files": [],
                        "hitl_paused": True,
                    },
                )
            else:
                logger.error("Error in streaming agent chat: %s", str(exc), exc_info=True)
                yield format_sse_event("error", {"message": str(exc)})

        finally:
            if mcp_client:
                logger.info("MCP client will be cleaned up automatically")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def stream_resume_agent_chat(
        self,
        agent_id: int,
        decisions: list[dict],
        user_context: dict | None = None,
        conversation_id: int | None = None,
        db: Session | None = None,
    ) -> AsyncGenerator[str, None]:
        """Resume a HITL-interrupted agent turn by sending decisions back.

        After a ``hitl_interrupt`` event paused the graph, the client calls
        this method with the user's decisions (approve / edit / reject) to
        resume execution from the saved checkpoint.

        Yields:
            SSE-formatted strings identical to ``stream_agent_chat``.
        """
        from langgraph.types import Command

        effective_db = db or self.db
        mcp_client = None
        ctx = None

        try:
            # 1. Prepare turn (reuses same session / conversation)
            ctx = await self.execution_service._prepare_turn(
                agent_id=agent_id,
                message="",  # No new user message for resume
                user_context=user_context,
                conversation_id=conversation_id,
                db=effective_db,
            )

            yield format_sse_event(
                "metadata",
                {
                    "conversation_id": ctx.effective_conv_id,
                    "session_id": ctx.conversation.session_id if ctx.conversation else None,
                    "agent_id": agent_id,
                    "agent_name": ctx.agent.name,
                    "has_memory": ctx.agent.has_memory,
                },
            )

            # 2. Rebuild agent chain (same tools, same checkpointer)
            agent_chain, mcp_client, monitoring_handler = await create_agent(
                ctx.fresh_agent,
                ctx.search_params,
                ctx.session_id_for_cache,
                ctx.user_context,
                ctx.working_dir,
            )

            config = prepare_agent_config(ctx.fresh_agent)
            if ctx.fresh_agent.has_memory and ctx.session_id_for_cache:
                config["configurable"]["thread_id"] = (
                    f"thread_{ctx.fresh_agent.agent_id}_{ctx.session_id_for_cache}"
                )

            # Attach monitoring callback if enabled
            if monitoring_handler is not None:
                config.setdefault("callbacks", []).append(monitoring_handler)

            # 3. Build the Command to resume with decisions
            resume_value = {"decisions": decisions}
            resume_input = Command(resume=resume_value)

            logger.info(
                "Resuming HITL for agent %s with %d decision(s): %s",
                agent_id, len(decisions),
                [d.get("type") for d in decisions],
            )

            # 4. Stream resumed execution
            accumulated_content = ""

            async for mode, chunk in agent_chain.astream(
                resume_input,
                config=config,
                stream_mode=["messages", "updates", "custom"],
            ):
                events = map_stream_event(mode, chunk)
                if events:
                    for event in events:
                        if event["type"] == SSE_TOKEN:
                            accumulated_content += event["data"].get("content", "")
                        yield format_sse_event(event["type"], event["data"])

            # 5. Check for further interrupts (chained HITL)
            has_pending_interrupt = False
            try:
                graph_state = await agent_chain.aget_state(config)
                for task in getattr(graph_state, 'tasks', []):
                    if hasattr(task, 'interrupts') and task.interrupts:
                        has_pending_interrupt = True
                        for intr in task.interrupts:
                            payload = intr.value if hasattr(intr, 'value') else intr
                            if isinstance(payload, dict):
                                action_requests = payload.get("action_requests", [])
                                review_configs = payload.get("review_configs", [])
                            else:
                                action_requests = []
                                review_configs = []
                            yield format_sse_event(
                                "hitl_interrupt",
                                {
                                    "action_requests": action_requests,
                                    "review_configs": review_configs,
                                },
                            )
            except Exception as state_err:
                logger.warning("Could not check graph state after resume: %s", state_err)

            if has_pending_interrupt:
                yield format_sse_event(
                    "done",
                    {
                        "response": "⏸️ Execution paused — awaiting human approval.",
                        "conversation_id": ctx.effective_conv_id,
                        "files": [],
                        "hitl_paused": True,
                    },
                )
            else:
                # Log usage metrics if monitoring is enabled
                if monitoring_handler is not None:
                    monitoring_mw_config = None
                    for _assoc in (getattr(ctx.fresh_agent, 'middleware_associations', None) or []):
                        if _assoc.middleware and _assoc.middleware.middleware_type.value == 'monitoring':
                            monitoring_mw_config = _assoc.middleware.config or {}
                            break
                    _emit_monitoring_log(ctx.fresh_agent.agent_id, monitoring_handler, monitoring_mw_config, logger.info)

                # 6. Normal finalization
                result = await self.execution_service._finalize_turn(
                    ctx, accumulated_content, effective_db
                )
                yield format_sse_event(
                    "done",
                    {
                        "response": result["parsed_response"],
                        "conversation_id": result["effective_conv_id"],
                        "files": result["files_data"],
                    },
                )

        except Exception as exc:
            logger.error("Error resuming HITL agent chat: %s", str(exc), exc_info=True)
            yield format_sse_event("error", {"message": str(exc)})

        finally:
            if mcp_client:
                logger.info("MCP client will be cleaned up automatically")
