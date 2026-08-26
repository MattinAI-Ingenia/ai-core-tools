"""Repository for IndexingMetric persistence and queries."""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import asc, desc, func
from sqlalchemy.orm import Session

from models.indexing_metric import IndexingMetric
from models.repository import Repository
from models.resource import Resource


class IndexingMetricRepository:
    """Data-access layer for :class:`IndexingMetric`."""

    @staticmethod
    def create(db: Session, **kwargs) -> IndexingMetric:
        """Persist a new IndexingMetric row and return it."""
        metric = IndexingMetric(**kwargs)
        db.add(metric)
        db.commit()
        db.refresh(metric)
        return metric

    @staticmethod
    def get_latest_by_resource(
        db: Session,
        resource_id: int,
        silo_id: int,
    ) -> Optional[IndexingMetric]:
        """Return the most recent metric for a resource, or None."""
        return (
            db.query(IndexingMetric)
            .filter(
                IndexingMetric.resource_id == resource_id,
                IndexingMetric.silo_id == silo_id,
            )
            .order_by(desc(IndexingMetric.created_at))
            .first()
        )

    @staticmethod
    def _sum_rows(rows: List[IndexingMetric], *, resource_id: Optional[int], silo_id: int) -> dict:
        """Fold a resource's indexing runs into one totals dict.

        A resource can need more than one run to reach 'ready' — an
        interrupted batch resumed later, or a manual reindex after a partial
        failure. Each run is its own row (see IndexingMetric's docstring);
        showing only the latest silently drops the cost/time/tokens of every
        earlier run, understating what the resource actually cost.

        ``rows`` must be non-empty, ordered oldest-first (so ``rows[-1]`` is
        the latest run).
        """
        latest = rows[-1]
        costs = [r.cost for r in rows if r.cost is not None]
        currencies = [r.currency for r in rows if r.currency]

        return {
            "metric_id": latest.metric_id,
            "silo_id": silo_id,
            "resource_id": resource_id,
            "content_ref": latest.content_ref,
            # Sums answer "what did this resource cost in total"; status
            # answers "is the most recent attempt done" — the latest row's.
            "status": latest.status,
            "prompt_tokens": sum(r.prompt_tokens or 0 for r in rows),
            "completion_tokens": sum(r.completion_tokens or 0 for r in rows),
            "total_tokens": sum(r.total_tokens or 0 for r in rows),
            # Conservative: any run whose tokens were estimated (not billed
            # by the provider) taints the whole sum as an estimate.
            "tokens_source": "estimated" if any(r.tokens_source == "estimated" for r in rows) else "provider",
            "embedding_tokens": sum(r.embedding_tokens or 0 for r in rows) or None,
            "llm_calls": sum(r.llm_calls or 0 for r in rows),
            "duration_seconds": sum(r.duration_seconds or 0.0 for r in rows),
            # Only runs that resolved a price contribute; a run with no
            # pricing data available understates the true total rather than
            # nulling it out entirely.
            "cost": round(sum(costs), 8) if costs else None,
            "currency": currencies[0] if currencies else None,
            "model_name": latest.model_name,
            "embedding_model_name": latest.embedding_model_name,
            "created_at": latest.created_at,
        }

    @staticmethod
    def get_summed_by_resource(
        db: Session,
        resource_id: int,
        silo_id: int,
    ) -> Optional[dict]:
        """Sum every indexing run recorded for a resource into one totals dict.

        Row count per resource is small (a handful at most), so summing in
        Python rather than a SQL aggregate keeps this simple to read.

        Returns ``None`` when no metric has been recorded yet.
        """
        rows = (
            db.query(IndexingMetric)
            .filter(
                IndexingMetric.resource_id == resource_id,
                IndexingMetric.silo_id == silo_id,
            )
            .order_by(asc(IndexingMetric.created_at))
            .all()
        )
        if not rows:
            return None
        return IndexingMetricRepository._sum_rows(rows, resource_id=resource_id, silo_id=silo_id)

    @staticmethod
    def list_latest_by_silo(
        db: Session,
        silo_id: int,
        limit: int = 100,
        offset: int = 0,
    ) -> List[IndexingMetric]:
        """Return the latest metric per resource for an entire silo.

        Uses a subquery to find the max ``created_at`` per ``resource_id``,
        then joins back to fetch the full rows.  Non-resource content
        (``resource_id IS NULL``) is included as individual rows.
        """
        from sqlalchemy import func, and_

        # Subquery: max created_at per (silo_id, resource_id)
        sub = (
            db.query(
                IndexingMetric.resource_id,
                func.max(IndexingMetric.created_at).label("max_created_at"),
            )
            .filter(
                IndexingMetric.silo_id == silo_id,
                IndexingMetric.resource_id.isnot(None),
            )
            .group_by(IndexingMetric.resource_id)
            .subquery()
        )

        rows = (
            db.query(IndexingMetric)
            .join(
                sub,
                and_(
                    IndexingMetric.resource_id == sub.c.resource_id,
                    IndexingMetric.created_at == sub.c.max_created_at,
                    IndexingMetric.silo_id == silo_id,
                ),
            )
            .order_by(desc(IndexingMetric.created_at))
            .offset(offset)
            .limit(limit)
            .all()
        )
        return rows

    @staticmethod
    def list_summed_by_silo(
        db: Session,
        silo_id: int,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict]:
        """One summed row per resource for an entire silo (page ordered by
        most-recently-active resource first — same ordering ``list_latest_by_silo``
        used, just summing each resource's full history instead of only its
        latest run)."""
        latest_rows = IndexingMetricRepository.list_latest_by_silo(db, silo_id, limit=limit, offset=offset)
        resource_ids = [r.resource_id for r in latest_rows]
        if not resource_ids:
            return []

        all_rows = (
            db.query(IndexingMetric)
            .filter(
                IndexingMetric.silo_id == silo_id,
                IndexingMetric.resource_id.in_(resource_ids),
            )
            .order_by(asc(IndexingMetric.created_at))
            .all()
        )
        by_resource: dict[int, list[IndexingMetric]] = {}
        for row in all_rows:
            by_resource.setdefault(row.resource_id, []).append(row)

        # Preserve list_latest_by_silo's ordering (most-recently-active first).
        return [
            IndexingMetricRepository._sum_rows(by_resource[rid], resource_id=rid, silo_id=silo_id)
            for rid in resource_ids
        ]

    @staticmethod
    def get_silo_totals(db: Session, silo_id: int) -> dict:
        """Aggregate token/cost totals across every run recorded in a silo.

        A straight SQL sum over every row — unlike the per-resource views,
        there is no "latest run" concept to preserve here, so no need to
        group by resource first.
        """
        from sqlalchemy import func as sqlfunc

        prompt_tokens, completion_tokens, total_tokens, llm_calls, indexed_resources = (
            db.query(
                sqlfunc.coalesce(sqlfunc.sum(IndexingMetric.prompt_tokens), 0),
                sqlfunc.coalesce(sqlfunc.sum(IndexingMetric.completion_tokens), 0),
                sqlfunc.coalesce(sqlfunc.sum(IndexingMetric.total_tokens), 0),
                sqlfunc.coalesce(sqlfunc.sum(IndexingMetric.llm_calls), 0),
                sqlfunc.count(sqlfunc.distinct(IndexingMetric.resource_id)),
            )
            .filter(IndexingMetric.silo_id == silo_id)
            .one()
        )

        # Cost: only rows that resolved a price contribute — a run with no
        # pricing data understates the true total rather than nulling it out.
        cost_row = (
            db.query(sqlfunc.sum(IndexingMetric.cost))
            .filter(IndexingMetric.silo_id == silo_id, IndexingMetric.cost.isnot(None))
            .scalar()
        )
        total_cost = round(cost_row, 6) if cost_row is not None else None

        currency = (
            db.query(IndexingMetric.currency)
            .filter(IndexingMetric.silo_id == silo_id, IndexingMetric.currency.isnot(None))
            .limit(1)
            .scalar()
        )

        return {
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "currency": currency,
            "total_llm_calls": llm_calls,
            "indexed_resources": indexed_resources,
        }

    @staticmethod
    def get_repository_duration_total(db: Session, repository_id: int) -> Optional[float]:
        """Sum every recorded run's duration for every resource in a repository.

        Same "every run, not just latest" reasoning as ``get_silo_totals``: a
        resource that needed more than one run to reach 'ready' contributes
        each run's time, not just its last one.

        Two sums, because a run is recorded one of two ways. Per-resource runs
        carry a ``resource_id``; a LightRAG batch is one pipeline pass over
        several documents, so it records a single row with ``resource_id`` NULL
        (see ``SiloService.process_enqueued_batch``) whose only route back is the
        silo. Without the second sum, batch-indexed repositories reported no time
        at all while the seconds sat in the table unreachable.

        That route reads "this row is silo X's, this repository uses silo X, so
        it is this repository's" — exact only while a silo belongs to one
        repository, which is how they are built (``create_repository`` always
        creates its own ``SiloType.REPO`` silo). Two repositories sharing one
        would both claim its batch rows; that would need a ``repository_id`` on
        the row. ``resource_id IS NULL`` keeps the sums disjoint: per-resource
        rows carry a ``silo_id`` too, and would otherwise be counted twice.

        Returns ``None`` when nothing in this repository has been indexed yet.
        """
        per_resource = (
            db.query(func.sum(IndexingMetric.duration_seconds))
            .join(Resource, Resource.resource_id == IndexingMetric.resource_id)
            .filter(Resource.repository_id == repository_id)
            .scalar()
        )
        silo_id = (
            db.query(Repository.silo_id)
            .filter(Repository.repository_id == repository_id)
            .scalar()
        )
        batch = (
            db.query(func.sum(IndexingMetric.duration_seconds))
            .filter(
                IndexingMetric.resource_id.is_(None),
                IndexingMetric.silo_id == silo_id,
            )
            .scalar()
        ) if silo_id else None
        if per_resource is None and batch is None:
            return None
        return round((per_resource or 0) + (batch or 0), 1)
