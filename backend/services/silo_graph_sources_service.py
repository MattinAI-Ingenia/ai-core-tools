"""Source-document resolver for the graph explorer's detail panels.

A graph node/edge carries ``properties.source_id`` — chunk ids joined by
``<SEP>``. Chunk ids shaped ``res{id}-p{page}-chunk-N`` yield (resource_id,
page) directly, with one ORM lookup for the Resource name. Ids that don't
parse (crawled Domain pages, legacy indexes) fall back to LightRAG's KV
table (``LIGHTRAG_DOC_CHUNKS``), whose ``file_path`` already carries the
"(p. N)" page label (see ``_page_label``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from tools.vector_stores.lightrag_store import CHUNK_ID_RESOURCE_PAGE_RE

logger = logging.getLogger(__name__)


class SiloGraphSourcesService:
    """Resolves chunk ids to the documents (and pages) they came from."""

    MAX_IDS = 100

    @classmethod
    def resolve_sources(
        cls,
        silo_id: int,
        chunk_ids: List[str],
        db,
    ) -> List[Dict[str, Any]]:
        """Return ``[{chunk_id, resource_id, page, file_name}]``, deduped by
        ``(file_name, page)`` — one chip per document page, order preserved."""
        entries: List[Dict[str, Any]] = []
        parsed: List[Tuple[str, int, Optional[int]]] = []
        unresolved: List[str] = []

        for chunk_id in chunk_ids:
            match = CHUNK_ID_RESOURCE_PAGE_RE.match(chunk_id.strip())
            if match:
                parsed.append((
                    chunk_id,
                    int(match.group(1)),
                    int(match.group(2)) if match.group(2) else None,
                ))
            else:
                unresolved.append(chunk_id)

        name_by_id: Dict[int, str] = {}
        resource_ids = sorted({rid for _, rid, _ in parsed})
        if resource_ids and db is not None:
            from models.resource import Resource
            rows = (
                db.query(Resource.resource_id, Resource.name)
                .filter(Resource.resource_id.in_(resource_ids))
                .all()
            )
            name_by_id = dict(rows)

        for chunk_id, resource_id, page in parsed:
            entries.append({
                "chunk_id": chunk_id,
                "resource_id": resource_id,
                "page": page,
                "file_name": name_by_id.get(resource_id),
            })

        if unresolved and db is not None:
            file_by_id = cls._lookup_kv_paths(silo_id, unresolved, db)
            for chunk_id in unresolved:
                entries.append({
                    "chunk_id": chunk_id,
                    "resource_id": None,
                    "page": None,
                    "file_name": file_by_id.get(chunk_id),
                })

        deduped: List[Dict[str, Any]] = []
        seen: set = set()
        for entry in entries:
            # Unknown sources (no file_name) are never merged — two distinct
            # documents must not collapse into one "Unknown source" chip.
            key = (entry["file_name"], entry["page"]) if entry["file_name"] else None
            if key is not None:
                if key in seen:
                    continue
                seen.add(key)
            deduped.append(entry)
        return deduped

    @staticmethod
    def _lookup_kv_paths(silo_id: int, chunk_ids: List[str], db) -> Dict[str, str]:
        """LightRAG's chunk KV (same Postgres, ORM-unmanaged table). Best-effort:
        a failure degrades to 'no file_path' like the rest of the graph access."""
        from sqlalchemy import text

        try:
            rows = db.execute(
                text(
                    "SELECT id, file_path FROM lightrag_doc_chunks "
                    "WHERE workspace = :ws AND id = ANY(:ids)"
                ),
                {"ws": f"silo_{silo_id}", "ids": chunk_ids},
            ).mappings().all()
            return {r["id"]: r["file_path"] for r in rows if r.get("file_path")}
        except Exception as exc:  # noqa: BLE001 — best-effort fallback
            logger.warning(
                "Failed to resolve chunk file paths for silo %d: %s", silo_id, exc,
            )
            return {}
