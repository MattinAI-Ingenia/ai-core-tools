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
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from models.enums.import_row_status import ImportRowStatus
from models.import_job_row import ImportJobRow
from models.resource import Resource
from models.silo import Silo
from tools.vector_stores.lightrag.entity_type_inference import (
    ENTITY_TYPES_JSON_SCHEMA,
    build_consolidation_prompt,
    build_prompt,
    extract_outline,
    sample_sections,
    select_diverse,
)

logger = logging.getLogger(__name__)

MAX_DOCUMENTS = 30
SAMPLES_PER_DOCUMENT = 2
# Budgets the document payload only. A self-hosted model's window is commonly
# 20k total; the instructions add ~0.9k, so 14000 here left the completion
# (up to 12 classes, each with a "why" sentence and examples — several
# thousand tokens) too little room and the model's response got cut off
# mid-JSON (observed: prompt_tokens=15433, completion capped at the window,
# unparseable). 10500 leaves ~8.5k for completion — about double what that
# failure needed — while keeping most of the original 30-document breadth.
DOCUMENT_TOKEN_BUDGET = 10500
MAX_TYPES = 10

# Jobs live in memory: they last a couple of minutes, carry nothing worth
# surviving a restart, and a lost one just means pressing the button again.
# ponytail: single-process only — move to a table if the backend is ever run
# with more than one worker.
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()

# A finished job is dropped once the UI has had a generous window to read it.
_JOB_TTL_SECONDS = 3600


def _update(job_id: str, **fields) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job.update(fields)


class EntityTypeInferenceService:
    """Stateless orchestrator — all methods are static."""

    @staticmethod
    def start(silo_id: int, ai_service_id: Optional[int], db: Session) -> str:
        """Register a job and return its id. The caller runs :meth:`run`."""
        EntityTypeInferenceService._resolve_ai_service(silo_id, ai_service_id, db)

        job_id = uuid.uuid4().hex
        with _JOBS_LOCK:
            _expire_old_jobs()
            _JOBS[job_id] = {
                "job_id": job_id,
                "silo_id": silo_id,
                "status": "pending",
                "done": 0,
                "total": 0,
                "types": None,
                "error": None,
                "created_at": time.time(),
            }
        return job_id

    @staticmethod
    def status(job_id: str) -> Optional[Dict[str, Any]]:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            return dict(job) if job else None

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

            _update(job_id, status="reading", total=len(paths))
            outlines = []
            for index, (doc_id, path) in enumerate(paths, start=1):
                outline = extract_outline(path, doc_id)
                if outline:
                    sample_sections(path, outline, limit=SAMPLES_PER_DOCUMENT)
                    outlines.append(outline)
                _update(job_id, done=index)

            if not outlines:
                raise ValueError(
                    "None of the documents has a usable text layer, so there is "
                    "no vocabulary to infer types from."
                )

            chosen = select_diverse(outlines, MAX_DOCUMENTS)
            _update(job_id, status="analysing", sampled=len(chosen))

            first = EntityTypeInferenceService._ask(
                llm,
                build_prompt(chosen, language=language, max_tokens=DOCUMENT_TOKEN_BUDGET),
            )

            _update(job_id, status="consolidating")
            merged = EntityTypeInferenceService._ask(
                llm, build_consolidation_prompt(first, language, MAX_TYPES)
            )

            # The merge pass is the one that can misfire (it once renamed every
            # class and invented an empty one), so a result that came back
            # emptier than it went in is discarded in favour of the first pass.
            types = merged if merged else first
            _update(job_id, status="done", types=types, candidates=first)
            logger.info(
                "Entity types inferred for silo %s: %s",
                silo_id, ", ".join(t.get("name", "?") for t in types),
            )
        except Exception as exc:  # noqa: BLE001 — the job carries the failure
            logger.exception("Entity type inference failed for silo %s", silo_id)
            _update(job_id, status="failed", error=str(exc))
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


def _expire_old_jobs() -> None:
    """Caller must hold _JOBS_LOCK."""
    cutoff = time.time() - _JOB_TTL_SECONDS
    for job_id in [k for k, v in _JOBS.items() if v["created_at"] < cutoff]:
        del _JOBS[job_id]
