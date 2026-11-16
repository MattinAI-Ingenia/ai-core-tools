"""Unit tests for _aprocess_enqueued_with_progress's CHUNK-level reporting
layer (``_report`` / ``_doc_chunk_progress``).

A resource is now one LightRAG document (prior steps of this spec), so the
doc-level ``(done, total)`` this function already tracked internally (see
``test_lightrag_progress.py`` for the sibling, still doc-level,
``_ainsert_with_progress`` path) would only ever show 0/1 -> 1/1 — no real
progress. This adds a parallel CHUNK-level layer on top, without touching the
doc-level bookkeeping that decides *when* a resource is actually done.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from services.resource_service import ResourceService
from tools.vector_stores import lightrag_store

pytestmark = pytest.mark.asyncio


def _rag(doc_status_sequence, insert_seconds):
    """Fake rag: doc_status.get_docs_by_ids walks a scripted sequence across
    polls, holding the last value once exhausted (same idiom as
    test_lightrag_progress.py's own ``_rag`` helper)."""
    seq = iter(doc_status_sequence)
    last: dict = {}

    async def get_docs_by_ids(_ids):
        nonlocal last
        last = next(seq, last)
        return last

    async def apipeline_process_enqueue_documents():
        await asyncio.sleep(insert_seconds)

    return SimpleNamespace(
        workspace="silo_1",
        doc_status=SimpleNamespace(get_docs_by_ids=get_docs_by_ids),
        apipeline_process_enqueue_documents=apipeline_process_enqueue_documents,
    )


async def _run_batch(rag, doc_ids_by_resource, chunk_mock, calls, extra_patches=(), **kwargs):
    """Drive _aprocess_enqueued_with_progress with *chunk_mock* swapped in for
    _doc_chunk_progress, then assert it was ONLY ever invoked via
    asyncio.to_thread (never directly on the event loop — it does a real DB
    query)."""
    real_to_thread = asyncio.to_thread
    to_thread_calls = []

    async def spying_to_thread(func, *args, **kw):
        if func is chunk_mock:
            to_thread_calls.append((args, kw))
        return await real_to_thread(func, *args, **kw)

    patches = [
        patch.object(lightrag_store, "_doc_chunk_progress", chunk_mock),
        patch("asyncio.to_thread", spying_to_thread),
        *extra_patches,
    ]
    for p in patches:
        p.start()
    try:
        result = await lightrag_store._aprocess_enqueued_with_progress(
            rag, doc_ids_by_resource,
            progress_callback=lambda *a: calls.append(a),
            **kwargs,
        )
    finally:
        for p in reversed(patches):
            p.stop()

    assert chunk_mock.call_count == len(to_thread_calls), (
        "_doc_chunk_progress must only ever be invoked via asyncio.to_thread"
    )
    return result


async def test_core_scenario_reports_chunks_clamped_until_doc_status_is_terminal():
    """done/total go from an estimate-free real count straight up to the real
    total, clamped below it while the doc is still mid-flight, and only hit
    100% once doc_status itself goes terminal — never before."""
    doc_status_seq = [
        {"res7": SimpleNamespace(status="processing")},
        {"res7": SimpleNamespace(status="processing")},
        {"res7": SimpleNamespace(status="processed")},
    ]
    chunk_seq = [{"res7": (2, 5)}, {"res7": (5, 5)}, {"res7": (5, 5)}]
    rag = _rag(doc_status_seq, insert_seconds=1.3)
    chunk_mock = Mock(side_effect=chunk_seq)
    calls: list[tuple] = []

    await _run_batch(rag, {7: ["res7"]}, chunk_mock, calls, feed=None)

    assert calls == [(7, 2, 5), (7, 4, 5), (7, 5, 5)], calls
    assert calls[-1] == (7, 5, 5), "last call must be the terminal one"

    dones = [c[1] for c in calls]
    assert dones == sorted(dones), f"done must be non-decreasing: {dones}"
    for rid, n, total in calls[:-1]:
        assert n < total, f"every call before the last must be strictly below total: {(rid, n, total)}"

    # Only the LAST (n, total) pair flips the resource to 'ready'.
    ready_flags = [
        ResourceService._progress_update_fields(n, total).get('status') == 'ready'
        for _, n, total in calls
    ]
    assert ready_flags == [False, False, True], ready_flags


async def test_estimate_is_corrected_to_the_real_total_without_stale_clamping():
    """An early report can carry a provisional/wrong total (e.g. a
    still-settling chunks_count); once it corrects, the callback must report
    the NEW total — and done must never be dragged up by a clamp computed
    against the OLD total."""
    doc_status_seq = [{"res7": SimpleNamespace(status="processing")}]  # never finishes
    chunk_seq = [
        {"res7": (8, 9)},
        {"res7": (3, 5)},
        {"res7": (3, 5)},
        {"res7": (3, 5)},
    ]
    rag = _rag(doc_status_seq, insert_seconds=1.3)
    chunk_mock = Mock(side_effect=chunk_seq)
    calls: list[tuple] = []

    await _run_batch(rag, {7: ["res7"]}, chunk_mock, calls, feed=None)

    assert calls == [(7, 8, 9), (7, 3, 5)], calls
    assert all(n <= total for _, n, total in calls), calls


async def test_a_chunk_progress_query_failure_never_aborts_the_run():
    """_doc_chunk_progress raising must degrade the bar (no update), not the
    run: indexing still completes, and the flush at the end reports
    (None, None) — finished, real chunk count unknown."""
    rag = _rag(
        [{"res7": SimpleNamespace(status="processed")}],
        insert_seconds=0.05,  # shorter than one 0.5s poll tick: zero polls fire
    )
    chunk_mock = Mock(side_effect=RuntimeError("boom"))
    calls: list[tuple] = []

    await _run_batch(rag, {7: ["res7"]}, chunk_mock, calls, feed=None)

    assert calls == [(7, None, None)], calls

    fields = ResourceService._progress_update_fields(None, None, estimate=12)
    assert fields == {'progress_done': 12, 'progress_total': 12, 'status': 'ready'}


async def test_feeder_path_reports_finished_unknown_total_on_a_content_dedup_hit():
    """Driven through ``feed`` instead of pre-enqueued work. The doc never
    progresses in doc_status, but a content-dedup hit (another doc already
    indexed this same content) finishes it anyway — chunk count stays
    unknown, so the flush reports (None, None)."""
    rag = _rag(
        [{}],  # get_docs_by_ids never returns a record for res7
        insert_seconds=0.05,  # zero polls fire; only the final flush runs
    )
    chunk_mock = Mock(return_value={})

    pending = [(7, ["t"], ["m.pdf"], ["res7"])]

    async def feed():
        return pending.pop(0) if pending else None

    async def noop_aenqueue(*_args, **_kwargs):
        return None

    fake_db = MagicMock()
    fake_db.execute.return_value.fetchall.return_value = [("m.pdf",)]  # dedup hit

    calls: list[tuple] = []
    await _run_batch(
        rag, {}, chunk_mock, calls,
        extra_patches=[
            patch.object(lightrag_store, "_aenqueue", noop_aenqueue),
            patch("db.database.SessionLocal", return_value=fake_db),
        ],
        feed=feed,
    )

    assert calls == [(7, None, None)], calls

    fields = ResourceService._progress_update_fields(None, None, estimate=12)
    assert fields['status'] == 'ready'
