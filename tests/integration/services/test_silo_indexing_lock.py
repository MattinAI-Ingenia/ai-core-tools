"""The silo indexing lock must be visible across connections, not per-process.

The guard it replaces lived in a module-level dict, so with several uvicorn
workers most requests could not see a run started elsewhere and two ingestions
could overlap on the same silo — which LightRAG cannot survive (its asyncio
locks are bound to one event loop).
"""

import pytest

from services import silo_indexing_lock


@pytest.fixture
def released():
    """Release anything a test acquired, even if it fails mid-way."""
    held = []
    yield held
    for session, silo_id in held:
        silo_indexing_lock.release(session, silo_id)


def test_second_acquire_on_same_silo_is_refused(db, released):
    first = silo_indexing_lock.acquire(4242)
    released.append((first, 4242))
    assert first is not None

    # A different connection — i.e. what another uvicorn worker would do.
    assert silo_indexing_lock.acquire(4242) is None


def test_holder_does_not_sit_in_a_transaction(db, released):
    """A run can last hours; an open transaction that long blocks DDL and vacuum."""
    session = silo_indexing_lock.acquire(4247)
    released.append((session, 4247))

    assert session.in_transaction() is False  # no bloquea DDL ni vacuum
    # ...and the lock still holds after that commit (it is session-scoped).
    assert silo_indexing_lock.acquire(4247) is None


def test_release_frees_the_silo(db):
    session = silo_indexing_lock.acquire(4244)
    assert session is not None
    silo_indexing_lock.release(session, 4244)

    again = silo_indexing_lock.acquire(4244)
    assert again is not None, "the silo stayed blocked after release"
    silo_indexing_lock.release(again, 4244)


def test_different_silos_do_not_block_each_other(db, released):
    a = silo_indexing_lock.acquire(4245)
    b = silo_indexing_lock.acquire(4246)
    released += [(a, 4245), (b, 4246)]
    assert a is not None and b is not None


def test_release_of_none_is_a_noop():
    """Called on the path where no lock was taken (silo_id == 0)."""
    silo_indexing_lock.release(None, 0)
