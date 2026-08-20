"""``get_summed_by_resource`` must add up every run, not just show the latest.

A resource can take more than one run to reach 'ready' — an interrupted batch
resumed later, or a manual reindex after a partial failure creates a new
IndexingMetric row each time (see IndexingMetric's docstring: "Re-indexing a
document inserts a *new* row; history is preserved"). The endpoint used to
show only the row with the highest created_at, silently dropping every
earlier run's cost/time/tokens from what the UI displayed.
"""

import pytest

from models.repository import Repository
from models.resource import Resource
from repositories.indexing_metric_repository import IndexingMetricRepository


@pytest.fixture
def repo(db, fake_app, fake_silo):
    repository = Repository(name="metric-sum-repo", app_id=fake_app.app_id, silo_id=fake_silo.silo_id)
    db.add(repository)
    db.flush()
    return repository


@pytest.fixture
def resource(db, repo):
    r = Resource(name="f.pdf", uri="f.pdf", type=".pdf", status="ready", repository_id=repo.repository_id)
    db.add(r)
    db.flush()
    return r


def _record(db, *, app_id, silo_id, resource_id, **kwargs):
    defaults = dict(
        status="success",
        prompt_tokens=0, completion_tokens=0, total_tokens=0, tokens_source="provider",
        llm_calls=0, duration_seconds=0.0, cost=None, currency=None,
    )
    defaults.update(kwargs)
    return IndexingMetricRepository.create(
        db, app_id=app_id, silo_id=silo_id, resource_id=resource_id, **defaults,
    )


def test_returns_none_when_no_metric_recorded(db, fake_app, fake_silo, resource):
    assert IndexingMetricRepository.get_summed_by_resource(db, resource_id=resource.resource_id, silo_id=fake_silo.silo_id) is None


def test_sums_tokens_duration_and_cost_across_runs(db, fake_app, fake_silo, resource):
    # Run 1: an interrupted batch got 40 chunks done before it died.
    _record(
        db, app_id=fake_app.app_id, silo_id=fake_silo.silo_id, resource_id=resource.resource_id,
        status="success", prompt_tokens=4000, completion_tokens=1000, total_tokens=5000,
        llm_calls=40, duration_seconds=500.0, cost=0.01, currency="USD",
    )
    # Run 2: resumed, finished the remaining 10 chunks.
    _record(
        db, app_id=fake_app.app_id, silo_id=fake_silo.silo_id, resource_id=resource.resource_id,
        status="success", prompt_tokens=1000, completion_tokens=250, total_tokens=1250,
        llm_calls=10, duration_seconds=10.0, cost=0.0025, currency="USD",
    )

    summed = IndexingMetricRepository.get_summed_by_resource(db, resource_id=resource.resource_id, silo_id=fake_silo.silo_id)

    assert summed["total_tokens"] == 6250
    assert summed["prompt_tokens"] == 5000
    assert summed["completion_tokens"] == 1250
    assert summed["llm_calls"] == 50
    assert summed["duration_seconds"] == pytest.approx(510.0)
    assert summed["cost"] == pytest.approx(0.0125)
    assert summed["currency"] == "USD"
    assert summed["status"] == "success"  # the latest run's outcome


def test_a_run_with_no_pricing_data_does_not_null_out_the_total(db, fake_app, fake_silo, resource):
    _record(
        db, app_id=fake_app.app_id, silo_id=fake_silo.silo_id, resource_id=resource.resource_id,
        total_tokens=100, cost=0.001, currency="USD",
    )
    _record(
        db, app_id=fake_app.app_id, silo_id=fake_silo.silo_id, resource_id=resource.resource_id,
        total_tokens=200, cost=None, currency=None,  # model not in the pricing catalog
    )

    summed = IndexingMetricRepository.get_summed_by_resource(db, resource_id=resource.resource_id, silo_id=fake_silo.silo_id)

    assert summed["total_tokens"] == 300
    assert summed["cost"] == pytest.approx(0.001)  # understates, but not silently zeroed


def test_any_estimated_run_taints_the_whole_sum_as_estimated(db, fake_app, fake_silo, resource):
    _record(
        db, app_id=fake_app.app_id, silo_id=fake_silo.silo_id, resource_id=resource.resource_id,
        tokens_source="provider",
    )
    _record(
        db, app_id=fake_app.app_id, silo_id=fake_silo.silo_id, resource_id=resource.resource_id,
        tokens_source="estimated",
    )

    summed = IndexingMetricRepository.get_summed_by_resource(db, resource_id=resource.resource_id, silo_id=fake_silo.silo_id)

    assert summed["tokens_source"] == "estimated"


def test_embedding_tokens_sum_ignores_null_runs(db, fake_app, fake_silo, resource):
    _record(
        db, app_id=fake_app.app_id, silo_id=fake_silo.silo_id, resource_id=resource.resource_id,
        embedding_tokens=1000,
    )
    _record(
        db, app_id=fake_app.app_id, silo_id=fake_silo.silo_id, resource_id=resource.resource_id,
        embedding_tokens=None,
    )

    summed = IndexingMetricRepository.get_summed_by_resource(db, resource_id=resource.resource_id, silo_id=fake_silo.silo_id)

    assert summed["embedding_tokens"] == 1000


# ---------------------------------------------------------------------------
# Silo-wide aggregation: the same "latest run only" bug, one level up.
# ---------------------------------------------------------------------------


@pytest.fixture
def second_resource(db, repo):
    r = Resource(name="g.pdf", uri="g.pdf", type=".pdf", status="ready", repository_id=repo.repository_id)
    db.add(r)
    db.flush()
    return r


def test_silo_totals_sum_every_run_not_just_the_latest_per_resource(db, fake_app, fake_silo, resource, second_resource):
    # resource: two runs (interrupted + resumed), same story as the per-resource test.
    _record(
        db, app_id=fake_app.app_id, silo_id=fake_silo.silo_id, resource_id=resource.resource_id,
        total_tokens=5000, llm_calls=40, cost=0.01, currency="USD",
    )
    _record(
        db, app_id=fake_app.app_id, silo_id=fake_silo.silo_id, resource_id=resource.resource_id,
        total_tokens=1250, llm_calls=10, cost=0.0025, currency="USD",
    )
    # second_resource: indexed cleanly in one run.
    _record(
        db, app_id=fake_app.app_id, silo_id=fake_silo.silo_id, resource_id=second_resource.resource_id,
        total_tokens=300, llm_calls=3, cost=0.0005, currency="USD",
    )

    totals = IndexingMetricRepository.get_silo_totals(db, silo_id=fake_silo.silo_id)

    assert totals["total_tokens"] == 6550  # 5000 + 1250 + 300, not 1250 + 300
    assert totals["total_llm_calls"] == 53
    assert totals["total_cost"] == pytest.approx(0.0130)
    assert totals["currency"] == "USD"
    assert totals["indexed_resources"] == 2  # distinct resources, not run count


def test_list_summed_by_silo_sums_each_resources_full_history(db, fake_app, fake_silo, resource, second_resource):
    _record(
        db, app_id=fake_app.app_id, silo_id=fake_silo.silo_id, resource_id=resource.resource_id,
        total_tokens=5000, cost=0.01, currency="USD",
    )
    _record(
        db, app_id=fake_app.app_id, silo_id=fake_silo.silo_id, resource_id=resource.resource_id,
        total_tokens=1250, cost=0.0025, currency="USD",
    )
    _record(
        db, app_id=fake_app.app_id, silo_id=fake_silo.silo_id, resource_id=second_resource.resource_id,
        total_tokens=300, cost=0.0005, currency="USD",
    )

    rows = IndexingMetricRepository.list_summed_by_silo(db, silo_id=fake_silo.silo_id, limit=100, offset=0)

    by_resource = {r["resource_id"]: r for r in rows}
    assert len(rows) == 2  # one row per resource, not per run
    assert by_resource[resource.resource_id]["total_tokens"] == 6250
    assert by_resource[second_resource.resource_id]["total_tokens"] == 300


# ---------------------------------------------------------------------------
# Repository-wide total indexing duration (shown on the repository detail page)
# ---------------------------------------------------------------------------


def test_repository_duration_total_sums_across_all_resources_and_runs(db, fake_app, fake_silo, repo, resource, second_resource):
    _record(
        db, app_id=fake_app.app_id, silo_id=fake_silo.silo_id, resource_id=resource.resource_id,
        duration_seconds=500.0,
    )
    _record(
        db, app_id=fake_app.app_id, silo_id=fake_silo.silo_id, resource_id=resource.resource_id,
        duration_seconds=10.0,
    )
    _record(
        db, app_id=fake_app.app_id, silo_id=fake_silo.silo_id, resource_id=second_resource.resource_id,
        duration_seconds=42.5,
    )

    total = IndexingMetricRepository.get_repository_duration_total(db, repository_id=repo.repository_id)

    assert total == pytest.approx(552.5)


def test_repository_duration_total_is_none_when_nothing_indexed(db, repo):
    assert IndexingMetricRepository.get_repository_duration_total(db, repository_id=repo.repository_id) is None
