"""Rerank stage: re-scores first-stage candidates with an external rerank service.

Failure never fails the search: any rerank error falls back to the first-stage order cut to `top_n`.
"""
import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Protocol, runtime_checkable

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from tools.rerank import RerankTarget, rerank_async, rerank_sync
from tools.rerank.errors import RerankError, RerankTimeoutError
from tools.retrieval.retrieval_component import RetrievalTransformer
from tools.retrieval.retrieval_context import RetrievalContext

logger = logging.getLogger(__name__)

# Single source of truth for the runtime default when no top N is configured.
# Candidates sent to the reranker = max(rag_k, top_n) (computed by the caller).
DEFAULT_RERANK_TOP_N = 30

SkippedReason = Literal[
    "not_configured", "service_unavailable", "service_retired", "service_error", "timeout", "no_candidates", "busy"
]


@dataclass
class RerankOutcome:
    requested: bool = True
    applied: bool = False
    service_id: Optional[int] = None
    service_name: Optional[str] = None
    source: Optional[str] = None
    skipped_reason: Optional[SkippedReason] = None
    latency_ms: Optional[int] = None
    scores: List[float] = field(default_factory=list)


@runtime_checkable
class RerankHealthRecorder(Protocol):
    def record_success(self, target: RerankTarget) -> None: ...

    def record_failure(self, target: RerankTarget, message: str) -> None: ...


@dataclass
class RerankPlan:
    """Per-retrieval rerank plan. Single-use: build a new plan for every retrieval invocation,
    because `outcome` is reset and mutated by each call of the retriever."""

    top_n: int
    candidate_limit: Optional[int] = None
    target: Optional[RerankTarget] = None
    source: Optional[str] = None
    skip_reason: Optional[str] = None
    skip_service_id: Optional[int] = None
    app_id: Optional[int] = None
    outcome: RerankOutcome = field(default_factory=RerankOutcome)
    health_recorder: Optional[RerankHealthRecorder] = None


# Failure health writes are throttled per service (in-process) so a dead reranker cannot turn every
# search into a DB write.
FAILURE_RECORD_INTERVAL_S = 30.0
_failure_last: dict = {}
_failure_lock = threading.Lock()


def _failure_throttled(service_id: Optional[int]) -> bool:
    now = time.monotonic()
    with _failure_lock:
        last = _failure_last.get(service_id)
        if last is not None and now - last < FAILURE_RECORD_INTERVAL_S:
            return True
        _failure_last[service_id] = now
        return False


def _should_record(plan: RerankPlan, method: str) -> bool:
    if plan.health_recorder is None or plan.target is None:
        return False
    if method == "record_success":
        # Only write state changes: a healthy service with no recorded error needs no write.
        return plan.target.last_error is not None
    return not _failure_throttled(plan.target.service_id)


def _safe_record(plan: RerankPlan, method: str, *args: Any) -> None:
    if not _should_record(plan, method):
        return
    recorder = plan.health_recorder
    try:
        getattr(recorder, method)(plan.target, *args)
    except Exception as exc:  # a health write must never break retrieval
        logger.warning("Rerank health recorder failed (%s): %s", method, type(exc).__name__)


class RerankRetriever(BaseRetriever):
    """Wraps a first-stage retriever and reorders its candidates with the rerank service."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: BaseRetriever
    plan: RerankPlan

    # ---- shared helpers -------------------------------------------------
    def _truncate(self, docs: List[Document]) -> List[Document]:
        limit = self.plan.candidate_limit
        return docs[:limit] if limit else docs

    def _reset(self) -> None:
        self.plan.outcome = RerankOutcome()

    def _skip(self, docs: List[Document], reason: str) -> List[Document]:
        self.plan.outcome.applied = False
        self.plan.outcome.skipped_reason = reason  # type: ignore[assignment]
        return docs[: self.plan.top_n]

    def _pre_check(self, docs: List[Document]) -> Optional[List[Document]]:
        plan = self.plan
        if not docs:
            return self._skip(docs, "no_candidates")
        if plan.target is None:
            reason = plan.skip_reason or "not_configured"
            if reason in ("service_unavailable", "service_retired"):
                logger.warning(
                    "Rerank skipped (%s): service_id=%s app_id=%s", reason, plan.skip_service_id, plan.app_id
                )
            else:
                logger.info("Rerank skipped (%s): app_id=%s", reason, plan.app_id)
            return self._skip(docs, reason)
        return None

    def _apply(self, docs: List[Document], results: list, latency_ms: int) -> List[Document]:
        plan = self.plan
        ordered = [(docs[r.index], r.score) for r in results if 0 <= r.index < len(docs)]
        out = plan.outcome
        out.applied = True
        out.skipped_reason = None
        out.service_id = plan.target.service_id
        out.service_name = plan.target.name
        out.source = plan.source
        out.latency_ms = latency_ms
        out.scores = [float(s) for _, s in ordered]
        return [d for d, _ in ordered][: plan.top_n]

    def _fallback(self, docs: List[Document], exc: Exception, latency_ms: int) -> List[Document]:
        plan = self.plan
        reason = "timeout" if isinstance(exc, RerankTimeoutError) else "service_error"
        if isinstance(exc, RerankError) and exc.kind == "busy":
            reason = "busy"
        message = exc.message if isinstance(exc, RerankError) else type(exc).__name__
        logger.warning(
            "Rerank failed, using first-stage order: service_id=%s app_id=%s provider=%s "
            "error_class=%s latency_ms=%s candidates=%d",
            plan.target.service_id, plan.target.app_id, plan.target.provider,
            type(exc).__name__, latency_ms, len(docs),
        )
        out = plan.outcome
        out.service_id, out.service_name, out.source = plan.target.service_id, plan.target.name, plan.source
        out.latency_ms = latency_ms
        if not (isinstance(exc, RerankError) and not exc.service_failure):
            _safe_record(plan, "record_failure", message)
        return self._skip(docs, reason)

    # ---- retriever API --------------------------------------------------
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        self._reset()
        docs = self._truncate(self.base_retriever.invoke(query, config={"callbacks": run_manager.get_child()}))
        early = self._pre_check(docs)
        if early is not None:
            return early
        plan = self.plan
        start = time.perf_counter()
        try:
            results = rerank_sync(plan.target, query, [d.page_content for d in docs], plan.top_n)
            out = self._apply(docs, results, int((time.perf_counter() - start) * 1000))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._fallback(docs, exc, int((time.perf_counter() - start) * 1000))
        _safe_record(plan, "record_success")
        return out

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        self._reset()
        docs = self._truncate(
            await self.base_retriever.ainvoke(query, config={"callbacks": run_manager.get_child()})
        )
        early = self._pre_check(docs)
        if early is not None:
            return early
        plan = self.plan
        start = time.perf_counter()
        try:
            results = await rerank_async(plan.target, query, [d.page_content for d in docs], plan.top_n)
            out = self._apply(docs, results, int((time.perf_counter() - start) * 1000))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            # the fallback may write health synchronously; keep the event loop free
            return await asyncio.to_thread(self._fallback, docs, exc, latency_ms)
        if _should_record(plan, "record_success"):
            await asyncio.to_thread(_safe_record, plan, "record_success")
        return out


class ServiceRerankTransformer(RetrievalTransformer):
    def apply(self, retrievers: List[BaseRetriever], ctx: RetrievalContext) -> BaseRetriever:
        return RerankRetriever(base_retriever=retrievers[0], plan=ctx.rerank)
