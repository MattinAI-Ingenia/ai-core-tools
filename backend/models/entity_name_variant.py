"""EntityNameVariant — observed spellings per canonical LightRAG entity name.

LightRAG merges nodes only on an exact, canonicalized entity name (see
``tools.vector_stores.lightrag.entity_name_normalization``): the merge key
folds case/accents/spaces/hyphens, so "MCF-40", "mcf 40" and "Mcf 40" land on
one node whose name is the folded string — which is ugly to display. This
table keeps, per (silo, canonical key), the raw spellings that were seen and
how often, so graph rendering can show the most-mentioned variant instead
("MCF-40") while the node identity stays canonical.

Rows are pure presentation bookkeeping: losing them never corrupts the graph
(the node keeps its canonical name as fallback). ``silo_id`` is CASCADE so
deleting a silo drops its variant history with it.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, UniqueConstraint

from db.database import Base


class EntityNameVariant(Base):
    __tablename__ = "entity_name_variant"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Owner silo. LightRAG workspaces are always ``silo_{id}``, so this is the
    # only scoping key; CASCADE ties the rows' lifetime to the silo's. No
    # explicit index: the unique constraint below already serves every access
    # pattern (its (silo_id, canonical_name) prefix covers the display lookup
    # and the FK-cascade scan; an extra index would just add write amplification).
    silo_id = Column(
        Integer,
        ForeignKey("Silo.silo_id", ondelete="CASCADE"),
        nullable=False,
    )

    # Canonical (folded) merge key — the exact string LightRAG stores as the
    # node name. 500 chars matches the guard used at write time; real names
    # are far shorter.
    canonical_name = Column(String(500), nullable=False)

    # A raw spelling that LightRAG's own cleanup preserved (case/accents kept).
    variant = Column(String(500), nullable=False)

    # Mentions recorded (entity records + relation endpoints; see the module
    # docstring in entity_name_normalization.py). Incremented on conflict.
    mention_count = Column(Integer, nullable=False, server_default="0")

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # One row per observed spelling per canonical key per silo. The unique
    # constraint doubles as the only index this table needs: its leftmost
    # prefix serves the FK-cascade scan and the (silo_id, canonical_name)
    # display lookup — DISTINCT ON (canonical_name) with ORDER BY
    # canonical_name, mention_count DESC sorts a handful of rows per group.
    __table_args__ = (
        UniqueConstraint(
            "silo_id", "canonical_name", "variant",
            name="uq_entvar_silo_canonical_variant",
        ),
    )

    def __repr__(self):
        return (
            f"<EntityNameVariant silo_id={self.silo_id} canonical={self.canonical_name!r} "
            f"variant={self.variant!r} count={self.mention_count}>"
        )
