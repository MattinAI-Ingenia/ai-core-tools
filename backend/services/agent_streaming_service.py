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
)
from services.agent_execution_service import AgentExecutionService
from utils.logger import get_logger

logger = get_logger(__name__)


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
            agent_chain, mcp_client = await create_agent(
            # Resolve temporary playground media silos
            temp_silo_ids = None
            session_id_for_media = ctx.conversation.session_id if ctx.conversation else None
            if session_id_for_media and effective_db:
                try:
                    from services.playground_media_service import PlaygroundMediaService
                    app_id = user_context.get("app_id") if user_context else None
                    if app_id:
                        temp_silo_ids = PlaygroundMediaService.get_temp_silo_ids_for_agent(
                            app_id, agent_id, session_id_for_media, effective_db
                        )
                except Exception:
                    pass

                # Vectorize attached files (PDFs, text) into a temp silo for RAG
                temp_silo_ids = self.execution_service._vectorize_and_resolve_file_silos(
                    ctx, agent_id, session_id_for_media, effective_db, temp_silo_ids
                )

            agent_chain, langsmith_config, mcp_client, monitoring_handler = await create_agent(
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
            accumulated_content = ""

            async for mode, chunk in agent_chain.astream(
                {"messages": [message_payload]},
                config=config,
                stream_mode=["messages", "updates"],
            ):
                events = map_stream_event(mode, chunk)
                if events:
                    for event in events:
                        if event["type"] == SSE_TOKEN:
                            accumulated_content += event["data"].get("content", "")
                        yield format_sse_event(event["type"], event["data"])

            # ----------------------------------------------------------------
            # 7. Post-processing phase — delegates to AgentExecutionService
            # ----------------------------------------------------------------

            # Log usage metrics if monitoring is enabled
            if monitoring_handler is not None:
                try:
                    usage = monitoring_handler.usage_metadata
                    logger.info(
                        f"[Monitoring] agent_id={ctx.fresh_agent.agent_id} | "
                        f"input_tokens={usage.get('input_tokens', 0)} | "
                        f"output_tokens={usage.get('output_tokens', 0)} | "
                        f"total_tokens={usage.get('total_tokens', 0)} | "
                        f"llm_calls={len(monitoring_handler.usage_metadata_list)}"
                    )
                except Exception as monitor_err:
                    logger.warning(f"Error reading monitoring metrics: {monitor_err}")

            result = await self.execution_service._finalize_turn(
                ctx, accumulated_content, effective_db
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
            logger.error("Error in streaming agent chat: %s", str(exc), exc_info=True)
            yield format_sse_event("error", {"message": str(exc)})

        finally:
            if mcp_client:
                logger.info("MCP client will be cleaned up automatically")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
