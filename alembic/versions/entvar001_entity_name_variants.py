"""entity_name_variant — spelling-variant counts per canonical LightRAG entity name.

Companion of the LightRAG merge-key canonicalization
(``tools.vector_stores/lightrag/entity_name_normalization.py``, decision
documented in ``docs/dependencies/lightrag.md`` §6.1): entity names fold into
a canonical key so LightRAG fuses "MCF-40"/"mcf 40"/"Mcf 40" into ONE node,
whose name is the folded string. This table records which raw spellings were
seen and how often, so graph rendering can display the most-mentioned one.

Down-revision note: rides on top of ingest003 (single head at the time).
"""

import sqlalchemy as sa
from alembic import op

revision = 'entvar001'
down_revision = 'ingest003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'entity_name_variant',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('silo_id', sa.Integer(), nullable=False),
        sa.Column('canonical_name', sa.String(length=500), nullable=False),
        sa.Column('variant', sa.String(length=500), nullable=False),
        sa.Column('mention_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ['silo_id'], ['Silo.silo_id'], name='fk_entity_name_variant_silo_id', ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'silo_id', 'canonical_name', 'variant',
            name='uq_entvar_silo_canonical_variant',
        ),
    )


def downgrade() -> None:
    op.drop_table('entity_name_variant')
