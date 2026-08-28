"""convert agent rag_search_method / rag_strategy to multiselect JSON columns

Revision ID: ragcfg003
Revises: ragcfg002
Create Date: 2026-07-28

Purpose
-------
`Agent.rag_search_method` and `Agent.rag_strategy` were single VARCHAR(45) values
(one search method / one strategy per agent). This migration converts both to
JSON columns holding a list of values, so an agent can combine multiple search
methods (e.g. dense + bm25 hybrid retrieval) and/or multiple post-retrieval
strategies in the future, without another column-type migration.

- rag_search_method: JSON list, NOT NULL, e.g. ["dense"], ["dense", "bm25"].
  Values come from SearchMethodFactory.IMPLEMENTED_SEARCH_METHODS.
- rag_strategy: JSON list, e.g. ["rerank"] or [] / SQL NULL for "no strategy".
  Values come from StrategyFactory.IMPLEMENTED_STRATEGIES.

Upgrade (backfill mapping)
---------------------------
Uses an add-backfill-constrain-drop-rename pattern so the column is populated
atomically within the same migration (no separate data-migration step required):
  1. Add temporary JSON columns (`rag_search_method_new`, `rag_strategy_new`),
     nullable, so the backfill UPDATE has somewhere to write without violating
     a NOT NULL constraint mid-backfill.
  2. Backfill from the existing string columns:
       rag_search_method: 'dense'  -> '["dense"]'
                          'bm25'   -> '["bm25"]'
                          'hybrid' -> '["dense","bm25"]'  (single canonical encoding
                                      for "both dense and bm25" going forward -- see
                                      "Encoding decision" below)
                          anything else / unexpected -> '["dense"]' (safe default)
       rag_strategy:      'rerank' -> '["rerank"]'
                          NULL     -> NULL (not coerced to [] or ["rerank"])
  3. Only once the backfill UPDATE has covered every row (the CASE/ELSE branch
     guarantees no NULLs remain in rag_search_method_new), alter that column to
     NOT NULL with server_default='["dense"]'. rag_strategy_new stays nullable.
  4. Drop the old string columns.
  5. Rename the temp columns to the original names.

Encoding decision (architecture-reviewer finding)
--------------------------------------------------
The legacy scalar value 'hybrid' meant "use both dense and bm25". On upgrade we
backfill it to '["dense","bm25"]' (NOT '["hybrid"]') so there is only ONE stored
representation of "both dense and bm25 combined" going forward -- the literal
string 'hybrid' is never written into the new JSON list by this migration.
downgrade() below still recognizes a literal `"hybrid"` element defensively (in
case such a value was ever written by a source other than this migration's own
upgrade path, e.g. manual data seeding), but that branch is not exercised by
data produced by upgrade() itself.

Downgrade (collapse rule -- must be applied consistently if this is ever rolled back)
--------------------------------------------------------------------------------------
Restores both columns as VARCHAR(45), collapsing the JSON value back to a single
string. The collapse CASE is TOTAL over every real shape a JSON column can hold
at downgrade time -- not just the list shape produced by upgrade() -- because by
the time downgrade() runs, application code from an intermediate deploy may have
already written JSON scalars or JSON `null` into these columns (see Risk notes).
Note: with the ORM model as defined in this revision (`JSON(none_as_null=True)`
on both columns, and `rag_search_method`'s Python-side default producing
`["dense"]` rather than `None`), current SQLAlchemy writes through `Agent`
cannot themselves produce a JSON `null` in either column -- the JSON-`null`
branches below are defensive, covering rows written before this model change,
or via raw SQL / other write paths (e.g. direct `UPDATE`s, a prior model
revision, or manual data seeding) that bypass the ORM's `none_as_null`
coercion.
Every branch checks `jsonb_typeof(...)` before calling any array-only function
(`jsonb_array_length`, `?`, `->> 0`), so no branch can raise
`DataError: cannot get array length of a scalar`:
  - rag_search_method: SQL NULL, JSON null, or empty JSON array -> 'dense' (the
    historical default / implicit behaviour). JSON string scalar -> that string,
    capped to 45 chars. JSON array containing 'hybrid', or containing BOTH
    'dense' AND 'bm25' -> 'hybrid'. Any other non-empty array -> its first
    element, capped to 45 chars.
  - rag_strategy: SQL NULL, JSON null, or empty JSON array -> SQL NULL (never
    coerced to 'rerank'). JSON string scalar -> that string, capped to 45 chars.
    Non-empty JSON array -> its first element, capped to 45 chars.

Risk notes
----------
This migration performs a data backfill (not just a schema change): existing
single-string values are transformed into JSON lists/objects on upgrade, and
collapsed back on downgrade per the rule above.

IMPORTANT -- this is NOT a lossless round trip in either direction:
  1. Collapsing a JSON array like ["dense","bm25"] back down to the scalar
     'hybrid' on downgrade, and then re-running upgrade(), produces
     '["dense","bm25"]' again -- so upgrade(downgrade(x)) is stable for that
     specific case, but downgrade() is generally lossy for any list with more
     than one element that isn't exactly {'dense','bm25'} (only the first
     element survives). This migration does not attempt full fidelity for
     hypothetical future multi-value combinations beyond dense+bm25; it only
     guarantees a safe, non-erroring, sensible-default collapse.
  2. Between the moment this migration's upgrade() runs and any future
     downgrade() of it, the application layer may still be the *pre-multiselect*
     version (this migration intentionally lands before agent_service.py /
     agent_schemas.py / silo_service.py / SearchMethodFactory are updated to
     read/write lists -- that is a separate, already-planned step). During that
     window, unconverted application code can write JSON *scalars* (e.g.
     `agent.rag_search_method = "dense"`) into these columns. (`rag_strategy`
     and, as of this revision, `rag_search_method` both use
     `JSON(none_as_null=True)`, so an ORM write of Python `None` persists as
     SQL NULL, not JSON `null` -- see note below on the defensive downgrade
     branches for the *other* ways JSON `null` can still show up.) downgrade()
     is written to handle all of these shapes without raising.
  3. Consequently: **Do NOT run this migration in ANY environment -- including
     local dev databases -- until the consumer step (agent_schemas.py /
     agent_service.py / silo_service.py / SearchMethodFactory / StrategyFactory
     made list-aware) lands in the same branch state.** This is not merely a
     deployment-ordering concern: the same broken window is reachable by ANY
     developer running `alembic upgrade head` locally on this branch right now,
     before deployment is even in scope. Concretely, running this migration
     alone causes:
       (a) `backend/services/silo_service.py`'s `_build_pipeline_retriever`
           nests the (still-scalar-shaped) value it builds into a list --
           `[search_method_name or DEFAULT]` becomes `[["dense"]]` once
           `search_method_name` itself is already a list from this column --
           so `SearchMethodFactory.get_search_method` raises
           `AttributeError: 'list' object has no attribute 'lower'` for every
           RAG-enabled agent.
       (b) editing any pre-existing agent through the current (unconverted)
           internal API silently resets its `rag_strategy` to NULL, because
           `agent_service.py`'s scalar-write path re-persists whatever the
           (broken) read path coerced it to.
     Coordinating that deploy is a separate, already-planned step and is not
     addressed by this migration.
"""
from alembic import op
import sqlalchemy as sa


revision = 'ragcfg003'
down_revision = 'ragcfg002'
branch_labels = None
depends_on = None


def upgrade():
    # Fail fast instead of queuing indefinitely behind live traffic on the
    # Agent table (cheap safety net; no downside for this table's size).
    op.execute("SET LOCAL lock_timeout = '3s'")

    # --- rag_search_method: String(45) NOT NULL -> JSON (list) NOT NULL ---
    op.add_column('Agent', sa.Column('rag_search_method_new', sa.JSON(), nullable=True))
    op.execute(
        """
        UPDATE "Agent" SET rag_search_method_new =
          CASE
            WHEN rag_search_method = 'dense' THEN '["dense"]'::json
            WHEN rag_search_method = 'bm25' THEN '["bm25"]'::json
            WHEN rag_search_method = 'hybrid' THEN '["dense","bm25"]'::json
            ELSE '["dense"]'::json
          END
        """
    )
    # Every row was covered by the CASE/ELSE above, so it is safe to enforce
    # NOT NULL now, before dropping the old column.
    op.alter_column(
        'Agent', 'rag_search_method_new',
        existing_type=sa.JSON(), nullable=False, server_default='["dense"]',
    )
    op.drop_column('Agent', 'rag_search_method')
    op.alter_column('Agent', 'rag_search_method_new', new_column_name='rag_search_method')

    # --- rag_strategy: String(45) NULL -> JSON (list) NULL ---
    op.add_column('Agent', sa.Column('rag_strategy_new', sa.JSON(), nullable=True))
    op.execute(
        """
        UPDATE "Agent" SET rag_strategy_new =
          CASE
            WHEN rag_strategy = 'rerank' THEN '["rerank"]'::json
            WHEN rag_strategy IS NULL THEN NULL
            ELSE NULL
          END
        """
    )
    op.drop_column('Agent', 'rag_strategy')
    op.alter_column('Agent', 'rag_strategy_new', new_column_name='rag_strategy')

    # Restore the session-local lock_timeout so it doesn't leak into whatever
    # migration runs next within this same `alembic upgrade head` transaction
    # (Alembic wraps the whole run in one transaction; SET LOCAL persists
    # until COMMIT, not just until the end of this function).
    op.execute("SET LOCAL lock_timeout = DEFAULT")


def downgrade():
    # Same lock-timeout safety net as upgrade() -- these ALTER/UPDATE
    # statements take a table-level lock on Agent.
    op.execute("SET LOCAL lock_timeout = '3s'")

    # --- rag_search_method: JSON -> String(45) NOT NULL, default 'dense' ---
    # Total over every shape the column can hold at this point: SQL NULL, JSON
    # null, JSON string scalar, or JSON array (empty or non-empty). The type
    # discriminant (jsonb_typeof) is the CASE *subject*, not an AND-chained
    # condition -- Postgres guarantees CASE evaluates WHEN branches in order
    # and only evaluates the matched branch, so no array-only function
    # (jsonb_array_length, ->> 0) can ever be reached against a non-array
    # value. Relying on AND's left-to-right operand evaluation for this same
    # purpose is not guaranteed by the SQL standard/Postgres, even though it
    # happened to work in testing.
    op.add_column('Agent', sa.Column('rag_search_method_old', sa.String(45), nullable=True))
    op.execute(
        """
        UPDATE "Agent" SET rag_search_method_old =
          CASE
            WHEN rag_search_method IS NULL THEN 'dense'
            ELSE
              CASE jsonb_typeof(rag_search_method::jsonb)
                WHEN 'null' THEN 'dense'
                WHEN 'string' THEN LEFT(rag_search_method::jsonb #>> '{}', 45)
                WHEN 'array' THEN
                  CASE
                    WHEN jsonb_array_length(rag_search_method::jsonb) = 0 THEN 'dense'
                    WHEN rag_search_method::jsonb ? 'hybrid' THEN 'hybrid'
                    WHEN (rag_search_method::jsonb ? 'dense')
                         AND (rag_search_method::jsonb ? 'bm25') THEN 'hybrid'
                    ELSE LEFT(COALESCE(rag_search_method::jsonb ->> 0, 'dense'), 45)
                  END
                ELSE 'dense'
              END
          END
        """
    )
    op.drop_column('Agent', 'rag_search_method')
    op.alter_column('Agent', 'rag_search_method_old', new_column_name='rag_search_method')
    op.alter_column(
        'Agent', 'rag_search_method',
        existing_type=sa.String(45), nullable=False, server_default='dense',
    )

    # --- rag_strategy: JSON -> String(45) NULL ---
    # Same total-CASE treatment as rag_search_method above: jsonb_typeof is the
    # CASE subject (not an AND-chained condition), so array-only functions are
    # only ever reached inside the 'array' branch. SQL NULL / JSON null / empty
    # array all collapse to SQL NULL (never coerced to 'rerank').
    op.add_column('Agent', sa.Column('rag_strategy_old', sa.String(45), nullable=True))
    op.execute(
        """
        UPDATE "Agent" SET rag_strategy_old =
          CASE
            WHEN rag_strategy IS NULL THEN NULL
            ELSE
              CASE jsonb_typeof(rag_strategy::jsonb)
                WHEN 'null' THEN NULL
                WHEN 'string' THEN LEFT(rag_strategy::jsonb #>> '{}', 45)
                WHEN 'array' THEN
                  CASE
                    WHEN jsonb_array_length(rag_strategy::jsonb) = 0 THEN NULL
                    ELSE LEFT(rag_strategy::jsonb ->> 0, 45)
                  END
                ELSE NULL
              END
          END
        """
    )
    op.drop_column('Agent', 'rag_strategy')
    op.alter_column('Agent', 'rag_strategy_old', new_column_name='rag_strategy')

    # Restore the session-local lock_timeout, matching upgrade().
    op.execute("SET LOCAL lock_timeout = DEFAULT")
