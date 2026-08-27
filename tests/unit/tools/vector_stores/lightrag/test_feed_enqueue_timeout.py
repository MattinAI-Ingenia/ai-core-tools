"""Pins the enqueue-call timeout mechanism added to _aenqueue_with_backpressure.

The real function is a closure defined inside _aprocess_enqueued_with_progress
(itself deep inside a LightRAG RAG instance's own pipeline call), not
independently importable — unit-testing it directly would mean mocking most of
LightRAG's RAG object, disproportionate to what this change is. What IS
independently verifiable is the exact mechanism the fix relies on:
asyncio.wait_for cutting off a coroutine that never resolves.

Context for why this exists: a live run went silent for close to an hour with
no error anywhere in the logs. The feeder — a single sequential coroutine —
had wedged on a stale pooled connection (visible from outside only as a
CLOSE_WAIT socket to Qdrant/the LLM host, left behind by an earlier transient
disconnect) inside _aenqueue, which does no LLM work and had no timeout of its
own. Nothing else in the batch noticed: already-fed resources kept finishing
on their own connections while every resource still waiting simply never got
fed. FEED_ENQUEUE_CALL_TIMEOUT_S turns that into a loud, logged failure
(caught by _feed_guarded) within a bounded time instead of an unbounded,
invisible hang.
"""

import asyncio

import pytest

from tools.vector_stores.lightrag_store import FEED_ENQUEUE_CALL_TIMEOUT_S


async def _never_resolves():
    """Stands in for _aenqueue hanging on a dead pooled connection: no
    exception, no return — just never completes."""
    await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_a_hung_call_is_cut_off_instead_of_blocking_forever():
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(_never_resolves(), timeout=0.05)


@pytest.mark.asyncio
async def test_a_call_that_finishes_in_time_is_unaffected():
    async def resolves_quickly():
        await asyncio.sleep(0.01)
        return "enqueued"

    result = await asyncio.wait_for(resolves_quickly(), timeout=0.2)
    assert result == "enqueued"


def test_the_configured_timeout_is_well_below_the_llm_worker_timeouts():
    """_aenqueue does no LLM work (see its own docstring) — it must stay far
    below the 120s/240s LLM worker timeouts logged at RAG init, or a hung
    enqueue would take as long to surface as a hung generation call, which
    defeats the point of having a separate, tighter bound for it."""
    assert FEED_ENQUEUE_CALL_TIMEOUT_S <= 120
