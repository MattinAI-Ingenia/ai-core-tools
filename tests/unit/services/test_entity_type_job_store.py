"""Entity-type inference jobs must be readable from a different session.

They used to live in a module-level dict, which works only in a single process.
This deployment runs UVICORN_WORKERS=4: the POST that starts a job lands on one
worker and the status polls are load-balanced across all four, so roughly three
polls in four answered "Job not found" for a job that existed — just not in the
worker that was asked. Two independent sessions here stand in for two workers.

Also pins two details that would each fail silently:
* the payload is REASSIGNED, not mutated — SQLAlchemy does not track mutation
  inside a JSON column, so ``row.payload[k] = v`` is never written;
* ``_update`` swallows its own errors, because ``run`` calls it from its
  ``except`` branch and a raising update would replace the real failure.
"""

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.database import Base
from models.entity_type_inference_job import EntityTypeInferenceJob
from services.entity_type_inference_service import _update


@pytest.fixture()
def sessions():
    """Two sessions over one in-memory database — two 'workers'.

    SQLite is enough for the read/merge/write path this is about; the column is
    declared JSON with a JSONB variant precisely so it compiles here too.
    """
    engine = create_engine("sqlite://")
    EntityTypeInferenceJob.__table__.create(engine)
    Maker = sessionmaker(bind=engine)
    a, b = Maker(), Maker()
    yield a, b
    a.close()
    b.close()
    engine.dispose()


def _insert(session, job_id="j1", silo_id=35, **extra):
    session.add(EntityTypeInferenceJob(
        job_id=job_id,
        silo_id=silo_id,
        payload={"job_id": job_id, "silo_id": silo_id, "status": "pending",
                 "done": 0, "total": 0, **extra},
    ))
    session.commit()


def _read(session, job_id="j1"):
    session.expire_all()
    row = session.query(EntityTypeInferenceJob).filter(
        EntityTypeInferenceJob.job_id == job_id,
    ).first()
    return dict(row.payload) if row else None


def test_a_job_created_in_one_session_is_visible_in_another(sessions):
    a, b = sessions
    _insert(a)
    assert _read(b) is not None, "this is the 'Job not found' bug"


def test_progress_written_by_one_session_is_read_by_another(sessions):
    a, b = sessions
    _insert(a)
    _update(a, "j1", status="analysing", done=7, total=30)
    job = _read(b)
    assert (job["status"], job["done"], job["total"]) == ("analysing", 7, 30)


def test_update_merges_instead_of_replacing(sessions):
    """Progress arrives field by field across the run; an update must not drop
    the fields set by earlier phases."""
    a, _ = sessions
    _insert(a)
    _update(a, "j1", total=30)
    _update(a, "j1", done=5)
    job = _read(a)
    assert job["total"] == 30 and job["done"] == 5
    assert job["silo_id"] == 35, "the router checks silo_id; it must survive"


def test_update_can_add_fields_absent_from_the_initial_payload(sessions):
    """`sampled` and `candidates` only appear mid-run — the store has to accept
    keys it was not created with."""
    a, b = sessions
    _insert(a)
    _update(a, "j1", sampled=12, candidates=[{"name": "Modelo"}])
    job = _read(b)
    assert job["sampled"] == 12
    assert job["candidates"] == [{"name": "Modelo"}]


def test_updating_an_unknown_job_is_a_no_op(sessions):
    """A poll can outlive the row after expiry; that must not raise."""
    a, _ = sessions
    _update(a, "does-not-exist", status="done")


def test_update_never_raises_even_on_a_broken_session(sessions):
    """run() calls _update from its except branch. If that threw, the real
    error would be replaced by a database one and lost."""
    a, _ = sessions
    _insert(a)
    a.close()  # any statement on this session now fails
    _update(a, "j1", status="failed", error="boom")
