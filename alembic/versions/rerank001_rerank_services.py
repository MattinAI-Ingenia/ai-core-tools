"""Rerank services: RerankService table, App default, Agent rerank columns.

- New table rerank_service (BaseService columns + lifecycle + health columns + provider, model_name,
  app_id NULL = system service, is_platform_default). CHECK ck_rerank_service_platform_default_is_system
  (only a system service can be the platform default) and partial unique index
  uq_rerank_service_platform_default (at most one platform default).
- App.default_rerank_service_id: nullable FK rerank_service.service_id ON DELETE SET NULL
  (named App_default_rerank_service_id_fkey, added as a separate step because of the App <-> rerank_service
  cycle). NULL = no app default; agents inherit the platform default.
- Agent.rag_rerank_enabled (Boolean NOT NULL, server_default false) and Agent.rag_rerank_service_id
  (nullable FK, NO ACTION on delete on purpose: an explicit choice must never silently become "inherit")
  with index ix_Agent_rag_rerank_service_id.
- Agent.rag_strategy (JSON list) and Agent.rag_rerank_similarity_threshold are dropped.

Data mapping (upgrade): rag_rerank_enabled = true when rag_strategy contains 'rerank' or
'cross_encoder_rerank' (JSON array, or a JSON string scalar with that value); false otherwise (SQL NULL,
JSON null, empty array, other values). jsonb_typeof is the CASE subject so array-only operators are never
applied to scalars (same approach as ragcfg003). rag_rerank_service_id stays NULL (inherits the default);
rag_rerank_top_n is untouched. The dropped threshold values are lost.

Downgrade is LOSSY: re-adds rag_strategy (JSON NULL) and rag_rerank_similarity_threshold (NULL), sets
rag_strategy = ["rerank"] for enabled agents. It cannot recover cross-encoder vs embeddings, explicit
service choices, app/platform defaults, nor the rerank services themselves (table dropped).

Lock safety: env.py sets the session lock_timeout (ALEMBIC_LOCK_TIMEOUT) for every migration and runs them in
one transaction, so this migration deliberately does not touch lock_timeout.

Deploy: stop old replicas BEFORE upgrading (old code selects the dropped rag_strategy /
rag_rerank_similarity_threshold columns). To roll back, run `alembic downgrade lexfts001` BEFORE reverting
the image.

Behaviour change: agents that had reranking enabled with no stored top N go from the old runtime default of 5
to the new default of 30, and the first stage now fetches max(rag_k, 30) candidates.

Revision ID: rerank001
Revises: lexfts001
"""

from alembic import op
import sqlalchemy as sa

revision = "rerank001"
down_revision = "lexfts001"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "rerank_service",
        sa.Column("service_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("create_date", sa.DateTime(), nullable=True),
        sa.Column("endpoint", sa.String(length=255), nullable=True),
        sa.Column("api_key", sa.Text(), nullable=True),
        sa.Column("description", sa.String(length=1000), nullable=True),
        sa.Column("api_version", sa.String(length=50), nullable=True),
        sa.Column("extra_config", sa.Text(), nullable=True),
        sa.Column("end_of_service_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deprecation_notice", sa.String(length=1000), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("last_error_at", sa.DateTime(), nullable=True),
        sa.Column("provider", sa.String(length=45), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("app_id", sa.Integer(), nullable=True),
        sa.Column("is_platform_default", sa.Boolean(), server_default="false", nullable=False),
        sa.CheckConstraint(
            "NOT is_platform_default OR app_id IS NULL",
            name="ck_rerank_service_platform_default_is_system",
        ),
        sa.ForeignKeyConstraint(["app_id"], ["App.app_id"]),
        sa.PrimaryKeyConstraint("service_id"),
    )
    op.create_index("ix_rerank_service_service_id", "rerank_service", ["service_id"], unique=False)
    op.create_index("ix_rerank_service_app_id", "rerank_service", ["app_id"], unique=False)
    op.create_index(
        "uq_rerank_service_platform_default", "rerank_service", ["is_platform_default"],
        unique=True, postgresql_where=sa.text("is_platform_default"),
    )

    op.add_column("App", sa.Column("default_rerank_service_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "App_default_rerank_service_id_fkey",
        "App", "rerank_service", ["default_rerank_service_id"], ["service_id"],
        ondelete="SET NULL",
    )

    op.add_column(
        "Agent", sa.Column("rag_rerank_enabled", sa.Boolean(), server_default="false", nullable=False)
    )
    op.add_column("Agent", sa.Column("rag_rerank_service_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "Agent_rag_rerank_service_id_fkey",
        "Agent", "rerank_service", ["rag_rerank_service_id"], ["service_id"],
    )
    op.create_index("ix_Agent_rag_rerank_service_id", "Agent", ["rag_rerank_service_id"], unique=False)

    op.execute(
        """
        UPDATE "Agent" SET rag_rerank_enabled = true
        WHERE rag_strategy IS NOT NULL AND
          CASE jsonb_typeof(rag_strategy::jsonb)
            WHEN 'array' THEN
              (rag_strategy::jsonb ? 'rerank') OR (rag_strategy::jsonb ? 'cross_encoder_rerank')
            WHEN 'string' THEN
              (rag_strategy::jsonb #>> '{}') IN ('rerank', 'cross_encoder_rerank')
            ELSE false
          END
        """
    )

    op.drop_column("Agent", "rag_strategy")
    op.drop_column("Agent", "rag_rerank_similarity_threshold")


def downgrade():
    op.add_column("Agent", sa.Column("rag_rerank_similarity_threshold", sa.Float(), nullable=True))
    op.add_column("Agent", sa.Column("rag_strategy", sa.JSON(none_as_null=True), nullable=True))
    op.execute("""UPDATE "Agent" SET rag_strategy = '["rerank"]'::json WHERE rag_rerank_enabled""")

    op.drop_index("ix_Agent_rag_rerank_service_id", table_name="Agent")
    op.drop_constraint("Agent_rag_rerank_service_id_fkey", "Agent", type_="foreignkey")
    op.drop_column("Agent", "rag_rerank_service_id")
    op.drop_column("Agent", "rag_rerank_enabled")

    op.drop_constraint("App_default_rerank_service_id_fkey", "App", type_="foreignkey")
    op.drop_column("App", "default_rerank_service_id")

    op.drop_table("rerank_service")
