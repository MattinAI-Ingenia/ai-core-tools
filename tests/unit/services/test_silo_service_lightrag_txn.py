"""Unit tests for SiloService._end_read_transaction_before_lightrag.

LightRAG's first-run storage init issues ``CREATE INDEX CONCURRENTLY``, which
waits for every transaction that was already open — a request or indexing
session holding its read snapshot across the build deadlocks the run against
itself. The helper ends such read-only transactions (and disables
``expire_on_commit`` so the build's attribute reads on already-loaded objects
don't re-open a transaction mid-build), while never committing pending writes
that belong to someone else.
"""

import pytest
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from services.silo_service import _end_read_transaction_before_lightrag

Base = declarative_base()


class _Thing(Base):
    __tablename__ = "txn_helper_things"
    id = Column(Integer, primary_key=True)
    name = Column(String)


@pytest.fixture()
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    s = factory()
    yield s
    s.close()


def test_ends_an_open_read_transaction(session):
    session.add(_Thing(id=1, name="a"))
    session.commit()
    session.execute(_Thing.__table__.select())
    assert session.in_transaction()

    _end_read_transaction_before_lightrag(session)

    assert not session.in_transaction()


def test_loaded_attributes_stay_readable_after_the_commit(session):
    session.add(_Thing(id=1, name="a"))
    session.commit()
    thing = session.query(_Thing).first()
    session.execute(_Thing.__table__.select())

    _end_read_transaction_before_lightrag(session)

    # Without expire_on_commit=False this read would lazy-load and re-open a
    # transaction — exactly the deadlock the helper exists to prevent.
    assert thing.name == "a"
    assert not session.in_transaction()


def test_pending_writes_are_left_uncommitted(session):
    session.add(_Thing(id=1, name="a"))

    _end_read_transaction_before_lightrag(session)

    assert session.in_transaction()  # untouched — not ours to commit
    session.rollback()


def test_none_or_idle_session_is_a_noop():
    _end_read_transaction_before_lightrag(None)
