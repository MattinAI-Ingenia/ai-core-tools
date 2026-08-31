"""Propose a silo's LightRAG entity types from the documents already uploaded.

Runs as a job because reading tables of contents out of a few dozen PDFs takes
tens of seconds and the two LLM passes take more, which is longer than a
request should hold open — and because the UI shows a progress bar while it
runs.

The result is never written to the silo here. ``lightrag_entity_types`` is
immutable after the first index, so the proposal is returned for a human to
edit and confirm; the save goes through the normal silo update.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from models.enums.import_row_status import ImportRowStatus
from models.entity_type_inference_job import EntityTypeInferenceJob
from models.import_job_row import ImportJobRow
from models.resource import Resource
from models.silo import Silo
from tools.vector_stores.lightrag.entity_type_inference import (
    ENTITY_TYPES_JSON_SCHEMA,
    build_consolidation_prompt,
    build_prompt,
    extract_outline,
    fit_budget,
    sample_sections,
    select_diverse,
)

logger = logging.getLogger(__name__)

MAX_DOCUMENTS = 30
# One per _SECTION_BUCKETS entry, and that count is the real ceiling: the
# sampler takes the first `limit` buckets in priority order, so anything above
# 4 is a no-op (measured: limit=4 and limit=6 produce byte-identical payloads).
#
# It was 2, which was quietly the whole quality problem. The buckets are
# ordered [especificaciones, parametros, errores, bloqueos] and the first two
# are both tables of magnitudes with units — so the model saw ONLY measurement
# tables from every manual's body and proposed one class per table row
# (PresionMaxima, Caudal, Potencia, Consumo…), which is exactly what the prompt
# forbids. The buckets that carry other kinds of instance — error codes,
# safety blocks — existed all along and were unreachable.
#
# Costs +8.9% payload tokens on this corpus (31,293 -> 34,072), not double:
# most manuals do not have all four sections, so the extra excerpts are often
# simply absent.
SAMPLES_PER_DOCUMENT = 4
# Budgets the document payload only, in REAL tokens. The arithmetic that has
# to hold, against the smallest window in play (self-hosted extraction servers
# are commonly started with max_model_len=20000):
#
#   payload (this) + instructions (~2.1k, measured) + completion  <=  window
#
# The completion is the expensive half: up to MAX_TYPES classes, each with a
# "why" sentence and examples. A truncated run generated 7.4k and was still
# cut off mid-JSON, so budget ~10k for it. 20000 - 2100 - 10000 leaves ~7.9k,
# hence 7000 with margin.
#
# History, because this constant has now broken twice for the same reason —
# it is only meaningful relative to how tokens are counted:
#   14000: sized against no real counting at all; truncated.
#   10500: sized while approx_tokens still used len/CHARS_PER_TOKEN, which
#          OVERESTIMATED Spanish text by ~1.65x (real: 3.8 chars/token, the
#          constant assumed 2.3). fit_budget therefore stopped trimming at
#          ~6.4k REAL tokens, and 10500 never actually applied. Making the
#          counter accurate removed that accidental safety margin, the payload
#          grew to the full 10500, and the completion was squeezed to 7.4k —
#          truncated again (prompt_tokens=12562, total exactly 20000).
#   7000:  sized against real tiktoken counting, so the number now means what
#          it says. Close to the ~6.4k that empirically worked.
#
# Used when the model's real window cannot be determined — see
# resolve_document_budget, which prefers the advertised one.
DOCUMENT_TOKEN_BUDGET = 7000
MAX_TYPES = 10

# Everything the request needs BESIDES the document payload, so the budget can
# be derived from a window instead of guessed.
_INSTRUCTION_OVERHEAD_TOKENS = 2_100   # measured: prompt 12562 with a 10500 payload
# Largest response the JSON schema can permit (20 classes x bounded fields).
# Bounding the schema is what made this knowable at all; before that the
# completion was unbounded and no budget could be proved safe.
_WORST_CASE_COMPLETION_TOKENS = 4_100
# Absorbs tokenizer drift and any per-provider framing we do not model.
_WINDOW_SAFETY_MARGIN_TOKENS = 2_000
# Below this the payload is too thin to infer anything useful from; better to
# let the request fail loudly against a tiny window than to ask a model to
# characterise a corpus from two pages.
_MIN_DOCUMENT_BUDGET = 3_000


def resolve_document_budget(window_tokens: Optional[int]) -> int:
    """Payload budget for a model whose context window is *window_tokens*.

    A single constant cannot serve both models in play: 7000 wastes 110k of
    gpt-oss-120b's 131k window (and covered only 6 of 30 documents on the
    DOMUSA corpus), while the 32000 that corpus actually needs overflows the
    Qwen extraction server's 20k outright. Derived per model, the same code
    path gives ~11.8k on a 20k window and ~123k on a 131k one.

    Falls back to the conservative constant when the window is unknown —
    OpenAI does not advertise one, and an endpoint can be down.
    """
    if not window_tokens or window_tokens <= 0:
        return DOCUMENT_TOKEN_BUDGET
    budget = window_tokens - (
        _INSTRUCTION_OVERHEAD_TOKENS
        + _WORST_CASE_COMPLETION_TOKENS
        + _WINDOW_SAFETY_MARGIN_TOKENS
    )
    return max(_MIN_DOCUMENT_BUDGET, budget)


def _advertised_window(llm) -> Optional[int]:
    """``max_model_len`` from an OpenAI-compatible endpoint, or None.

    vLLM and SGLang report it in /v1/models, which makes the real ceiling
    knowable instead of assumed — the alternative is a curated table that
    drifts every time a server is relaunched with different flags. Best-effort
    by design: hosted APIs do not expose it, and a self-hosted box can be
    restarting. Never raises, never blocks the job.
    """
    base_url = getattr(llm, "openai_api_base", None)
    model_name = getattr(llm, "model_name", None)
    if not base_url:
        return None
    try:
        import urllib.request  # noqa: PLC0415

        request = urllib.request.Request(f"{base_url.rstrip('/')}/models")
        for header, value in (getattr(llm, "default_headers", None) or {}).items():
            request.add_header(header, value)
        with urllib.request.urlopen(request, timeout=5) as response:
            entries = json.load(response).get("data") or []
    except Exception:  # noqa: BLE001 — an unknown window is a valid outcome
        logger.info("Could not read the context window from %s", base_url)
        return None

    for entry in entries:
        if model_name and entry.get("id") != model_name:
            continue
        window = entry.get("max_model_len")
        if isinstance(window, int) and window > 0:
            return window
    return None

# A finished job is dropped once the UI has had a generous window to read it.
_JOB_TTL_SECONDS = 3600


def _update(db: Session, job_id: str, **fields) -> None:
    """Merge *fields* into the job's payload. Never raises.

    Progress reporting must not be able to fail the job it is reporting on, and
    ``run`` calls this from its own ``except`` branch — if that update threw,
    the real error would be replaced by a database one. A rollback first, so an
    earlier failed statement in this session cannot poison the write.
    """
    try:
        db.rollback()
        row = db.query(EntityTypeInferenceJob).filter(
            EntityTypeInferenceJob.job_id == job_id,
        ).first()
        if row is None:
            return
        # Reassigned, not mutated in place: SQLAlchemy does not track mutation
        # inside a JSON column, so `row.payload[k] = v` would never be written.
        row.payload = {**row.payload, **fields}
        db.commit()
    except Exception:  # noqa: BLE001
        logger.exception("Could not update inference job %s", job_id)
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass


class EntityTypeInferenceService:
    """Stateless orchestrator — all methods are static."""

    @staticmethod
    def start(silo_id: int, ai_service_id: Optional[int], db: Session) -> str:
        """Register a job and return its id. The caller runs :meth:`run`."""
        EntityTypeInferenceService._resolve_ai_service(silo_id, ai_service_id, db)

        job_id = uuid.uuid4().hex
        _expire_old_jobs(db)
        db.add(EntityTypeInferenceJob(
            job_id=job_id,
            silo_id=silo_id,
            payload={
                "job_id": job_id,
                "silo_id": silo_id,
                "status": "pending",
                "done": 0,
                "total": 0,
                "types": None,
                "error": None,
            },
        ))
        db.commit()
        return job_id

    @staticmethod
    def status(job_id: str, db: Session) -> Optional[Dict[str, Any]]:
        """The job's payload, or None. Readable from ANY worker — that is the
        whole point of this living in the database."""
        row = db.query(EntityTypeInferenceJob).filter(
            EntityTypeInferenceJob.job_id == job_id,
        ).first()
        return dict(row.payload) if row else None

    @staticmethod
    def run(
        job_id: str, silo_id: int, ai_service_id: Optional[int], db: Session,
        import_job_id: Optional[int] = None,
    ) -> None:
        """Do the work. Exceptions land on the job, never on the caller."""
        try:
            llm = EntityTypeInferenceService._resolve_ai_service(silo_id, ai_service_id, db)
            silo = db.query(Silo).filter(Silo.silo_id == silo_id).first()
            language = getattr(silo, "lightrag_language", None)

            paths = EntityTypeInferenceService._document_paths(silo_id, db, import_job_id)
            if not paths:
                raise ValueError(
                    "This silo has no documents uploaded yet. Upload them before "
                    "inferring the entity types."
                )

            _update(db, job_id, status="reading", total=len(paths))
            outlines = []
            for index, (doc_id, path) in enumerate(paths, start=1):
                outline = extract_outline(path, doc_id)
                if outline:
                    sample_sections(path, outline, limit=SAMPLES_PER_DOCUMENT)
                    outlines.append(outline)
                _update(db, job_id, done=index)

            if not outlines:
                raise ValueError(
                    "None of the documents has a usable text layer, so there is "
                    "no vocabulary to infer types from."
                )

            chosen = select_diverse(outlines, MAX_DOCUMENTS)

            # fit_budget is applied HERE, not left to build_prompt, so the job
            # reports what the model actually saw. It used to report
            # len(chosen) while build_prompt trimmed the payload internally:
            # on this corpus that meant claiming 30 documents when the prompt
            # carried 6 (31,293 tokens of outlines against a 7,000 budget), and
            # the proposal came back describing only the spec tables that
            # survived — trimming drops second excerpts first, which is the
            # diverse half. A drastic, invisible truncation reads as a bad model.
            budget = resolve_document_budget(_advertised_window(llm))
            fitted = fit_budget(chosen, budget)
            if len(fitted) < len(chosen):
                logger.warning(
                    "Entity-type inference for silo %s: a %d-token payload budget "
                    "fits only %d of %d documents, so the proposal describes that "
                    "subset. Re-run with a larger-window model via ai_service_id "
                    "to cover the whole corpus.",
                    silo_id, budget, len(fitted), len(chosen),
                )
            else:
                logger.info(
                    "Entity-type inference for silo %s: all %d documents fit in a "
                    "%d-token budget.", silo_id, len(fitted), budget,
                )
            # budget is reported so the UI can explain a partial proposal
            # instead of it looking like the model simply did a poor job.
            _update(
                db, job_id, status="analysing",
                sampled=len(fitted), considered=len(chosen), token_budget=budget,
            )

            first = EntityTypeInferenceService._ask(
                llm,
                build_prompt(fitted, language=language, max_tokens=budget),
            )

            _update(db, job_id, status="consolidating")
            merged = EntityTypeInferenceService._ask(
                llm, build_consolidation_prompt(first, language, MAX_TYPES)
            )

            # The merge pass is the one that can misfire (it once renamed every
            # class and invented an empty one), so a result that came back
            # emptier than it went in is discarded in favour of the first pass.
            types = merged if merged else first
            _update(db, job_id, status="done", types=types, candidates=first)
            logger.info(
                "Entity types inferred for silo %s: %s",
                silo_id, ", ".join(t.get("name", "?") for t in types),
            )
        except Exception as exc:  # noqa: BLE001 — the job carries the failure
            logger.exception("Entity type inference failed for silo %s", silo_id)
            _update(db, job_id, status="failed", error=str(exc))
        finally:
            db.close()

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _resolve_ai_service(silo_id: int, ai_service_id: Optional[int], db: Session):
        """Build the LLM: the caller's choice, else the silo's extraction one.

        The override is not persisted — inference runs once, and a smarter model
        is worth paying for on a decision that cannot be undone after indexing.
        """
        from models.ai_service import AIService  # noqa: PLC0415 — avoids a cycle
        from tools.aiServiceTools import create_llm_from_service  # noqa: PLC0415

        silo = db.query(Silo).filter(Silo.silo_id == silo_id).first()
        if not silo:
            raise ValueError(f"Silo {silo_id} not found.")

        if ai_service_id:
            service = db.query(AIService).filter(
                AIService.service_id == ai_service_id,
                AIService.app_id == silo.app_id,
            ).first()
            if not service:
                raise ValueError("The selected AI service does not exist in this app.")
        else:
            service = silo.extract_service or silo.indexing_service
            if not service:
                raise ValueError(
                    "This silo has no extraction service configured; pick one to "
                    "infer the types."
                )

        return create_llm_from_service(service, 0.2, False)

    @staticmethod
    def _document_paths(
        silo_id: int, db: Session, import_job_id: Optional[int] = None,
    ) -> List[tuple]:
        """(doc_id, absolute path) for every PDF across the silo's repositories.

        A Silo can back more than one Repository (1:N), so ``Silo.repository``
        is a list, not a single object.

        On a silo's very first ingest there are no Resource rows yet — nothing
        is persisted until the batch is confirmed, which is exactly what this
        inference is meant to gate. When ``import_job_id`` is given (a CSV
        import review not yet confirmed), fall back to that job's staged
        (downloaded but not yet ingested) PDFs.
        """
        from services.resource_service import ResourceService  # noqa: PLC0415

        repositories = getattr(
            db.query(Silo).filter(Silo.silo_id == silo_id).first(), "repository", None
        )
        resources = []
        if repositories:
            resources = db.query(Resource).filter(
                Resource.repository_id.in_([r.repository_id for r in repositories])
            ).all()

        paths = []
        for resource in resources:
            if not (resource.uri or "").lower().endswith(".pdf"):
                continue
            path = ResourceService.get_resource_file_path(resource.resource_id, db)
            if path:
                paths.append(((resource.name or resource.uri).rsplit(".", 1)[0], path))

        if paths or not import_job_id:
            return paths

        from models.import_job import ImportJob  # noqa: PLC0415 — avoids a cycle

        repository_ids = {r.repository_id for r in repositories} if repositories else set()
        job = db.query(ImportJob).filter(ImportJob.id == import_job_id).first()
        if not job or job.repository_id not in repository_ids:
            return []

        rows = db.query(ImportJobRow).filter(
            ImportJobRow.import_job_id == import_job_id,
            ImportJobRow.status == ImportRowStatus.DOWNLOADED,
            ImportJobRow.staged_path.isnot(None),
        ).all()
        for row in rows:
            if row.staged_path and os.path.exists(row.staged_path):
                doc_id = (row.url.rsplit("/", 1)[-1] or f"row-{row.id}").rsplit(".", 1)[0]
                paths.append((doc_id, row.staged_path))
        return paths

    @staticmethod
    def _ask(llm, prompt: str) -> List[dict]:
        """One structured call. Tolerates a model that fences its JSON."""
        response = llm.invoke(
            prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "entity_types",
                    "schema": ENTITY_TYPES_JSON_SCHEMA,
                },
            },
        )
        text = (getattr(response, "content", None) or str(response)).strip()
        if text.startswith("```"):
            text = text.split("```")[1].removeprefix("json").strip()
        return json.loads(text).get("types", [])


def _expire_old_jobs(db: Session) -> None:
    """Delete jobs older than the TTL. Best-effort housekeeping.

    Runs on start() rather than on a timer: jobs are only created there, so
    that is the one moment the table can grow, and it keeps the service free of
    background threads that a multi-worker deployment would run N times over.
    """
    cutoff = datetime.utcnow() - timedelta(seconds=_JOB_TTL_SECONDS)
    try:
        db.query(EntityTypeInferenceJob).filter(
            EntityTypeInferenceJob.created_at < cutoff,
        ).delete(synchronize_session=False)
        db.commit()
    except Exception:  # noqa: BLE001 — never block a new job over cleanup
        logger.exception("Could not expire old inference jobs")
        db.rollback()
