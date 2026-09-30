"""Data access for EntityNameVariant — spelling-variant counts per canonical entity name.

Written by the LightRAG wrapper at the end of each indexing run (drained
from ``EntityNameVariantCollector``, see
``tools.vector_stores.lightrag.entity_name_normalization``) and read at
graph-render time to pick the most-mentioned variant as the display name.
"""

from typing import Dict, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from models.entity_name_variant import EntityNameVariant

# Entity names observed per indexing run are few (tens of canonical keys × a
# handful of variants); the cap is a safety valve against pathological
# extraction runs, not an expected limit.
_MAX_ROWS_PER_FLUSH = 2000
_MAX_NAME_LENGTH = 500


def _truncate(value: str) -> str:
    """Length-cap AND strip control bytes (Postgres rejects NUL in varchar).

    One poisoned name must not abort the whole flush for its run — dropping
    the control characters keeps the other variants writable; the cleaned
    name itself still matches nothing meaningful.
    """
    return "".join(ch for ch in (value or "") if ch >= " ")[:_MAX_NAME_LENGTH]


class EntityNameVariantRepository:
    """Session-bound helpers over the ``entity_name_variant`` table."""

    @staticmethod
    def record_mentions(silo_id: int, counts: Dict[str, Dict[str, int]], db) -> int:
        """Bulk-increment mention counts for one silo. Returns rows written.

        ``counts`` maps canonical name → {raw variant → mentions this run}.
        A conflict on (silo_id, canonical_name, variant) adds to the stored
        count instead of replacing it, so indexing runs compose across days.
        Any failure is rolled back and re-raised — the CALLER
        (``lightrag_store._persist_variant_counts``) logs and swallows it, per
        its own "persistence failure must never fail the indexing run" contract.
        """
        if not counts:
            return 0
        rows = []
        for canonical, variants in counts.items():
            if not canonical:
                continue
            for variant, mention_count in variants.items():
                if not variant or not mention_count:
                    continue
                rows.append({
                    "silo_id": silo_id,
                    "canonical_name": _truncate(canonical),
                    "variant": _truncate(variant),
                    "mention_count": int(mention_count),
                })
        if not rows:
            return 0

        written = 0
        try:
            # Chunked so a runaway run cannot exceed parameter limits.
            for start in range(0, len(rows), _MAX_ROWS_PER_FLUSH):
                chunk = rows[start:start + _MAX_ROWS_PER_FLUSH]
                stmt = pg_insert(EntityNameVariant).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_entvar_silo_canonical_variant",
                    set_={
                        "mention_count": EntityNameVariant.mention_count + stmt.excluded.mention_count,
                        "updated_at": func.now(),
                    },
                )
                result = db.execute(stmt)
                written += result.rowcount or 0
            db.commit()
        except Exception:
            db.rollback()
            raise
        return written

    @staticmethod
    def best_variant_map(
        silo_id: int,
        canonical_names: Optional[List[str]] = None,
        db=None,
    ) -> Dict[str, str]:
        """Most-mentioned variant per canonical key (empty when unknown).

        ``canonical_names`` scopes the lookup to the entities currently in a
        graph response; ``None`` returns the whole silo's map. Ties break on
        insertion order (first-seen wins), which keeps rendering stable.
        """
        if db is None:
            return {}
        query = db.query(
            EntityNameVariant.canonical_name,
            EntityNameVariant.variant,
            EntityNameVariant.mention_count,
        )
        if canonical_names:
            query = query.filter(
                EntityNameVariant.canonical_name.in_(_truncate(n) for n in canonical_names)
            )
        query = query.filter(EntityNameVariant.silo_id == silo_id)
        # DISTINCT ON needs its leading ORDER BY columns to match.
        rows = (
            query.distinct(EntityNameVariant.canonical_name)
            .order_by(
                EntityNameVariant.canonical_name,
                EntityNameVariant.mention_count.desc(),
                EntityNameVariant.id.asc(),
            )
            .all()
        )
        return {canonical: variant for canonical, variant, _count in rows}
