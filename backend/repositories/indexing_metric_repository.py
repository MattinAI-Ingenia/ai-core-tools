"""Repository for IndexingMetric persistence and queries."""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from models.indexing_metric import IndexingMetric


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
    def get_silo_totals(db: Session, silo_id: int) -> dict:
        """Aggregate token/cost totals for all latest-run metrics in a silo."""
        from sqlalchemy import func as sqlfunc

        rows = IndexingMetricRepository.list_latest_by_silo(db, silo_id, limit=10_000)
        total_tokens = sum(r.total_tokens or 0 for r in rows)
        documents = len(rows)

        # Cost aggregation: only rows with cost != NULL
        costs = [r.cost for r in rows if r.cost is not None]
        total_cost = round(sum(costs), 6) if costs else None

        # Currency: use the first non-null value (all should agree for a silo)
        currencies = [r.currency for r in rows if r.currency]
        currency = currencies[0] if currencies else None

        return {
            "total_tokens": total_tokens,
            "cost": total_cost,
            "currency": currency,
            "documents": documents,
        }
