"""Unit tests for ``_ainsert_with_progress``'s ``(done, total)`` reporting.

Two things are pinned here:

* both numbers are in the same unit (LightRAG documents — one per PDF page),
  since the per-file bar divides one by the other;
* ``done`` counts *finished* documents.  Counting admissions instead (what
  ``pipeline_status["cur_batch"]`` does) made the first ``max_parallel_insert``
  pages look instantaneous and biased the ETA low.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.vector_stores import lightrag_store

pytestmark = pytest.mark.asyncio


def _rag(status_counts_sequence):
    """A fake rag whose doc_status walks through the given count dicts."""
    seq = iter(status_counts_sequence)
    last = {}

    async def get_status_counts():
        nonlocal last
        last = next(seq, last)
        return last

    return SimpleNamespace(
        workspace="silo_1",
        addon_params={},
        doc_status=SimpleNamespace(get_status_counts=get_status_counts),
    )


async def _run(rag, texts, calls, insert_seconds=1.2):
    async def fake_insert(_rag, _texts, file_paths=None, ids=None, process_options="F"):
        await asyncio.sleep(insert_seconds)

    with patch.object(lightrag_store, "_ainsert", fake_insert):
        await lightrag_store._ainsert_with_progress(
            rag, texts, progress_callback=lambda *a: calls.append(a)
        )


async def test_reports_finished_documents_and_total():
    texts = ["p1", "p2", "p3", "p4"]
    calls: list[tuple[int, int]] = []
    # baseline read (0 finished), then 2 documents finish.
    rag = _rag([{}, {"processed": 2, "processing": 2}])

    await _run(rag, texts, calls)

    assert all(len(c) == 2 for c in calls), f"callback must be (done, total): {calls}"
    assert all(total == len(texts) for _, total in calls)
    assert 2 in [done for done, _ in calls], calls
    assert calls[-1] == (len(texts), len(texts)), "final call must be 100%"


async def test_documents_from_earlier_resources_are_not_counted():
    """The silo already holds 10 processed documents from a previous file."""
    texts = ["p1", "p2", "p3"]
    calls: list[tuple[int, int]] = []
    rag = _rag([{"processed": 10}, {"processed": 11, "processing": 2}])

    await _run(rag, texts, calls)

    mid = [done for done, _ in calls[:-1]]
    assert mid == [1] * len(mid), f"expected the delta (1), got {mid}"


async def test_total_follows_the_documents_this_run_will_process():
    """On a resumed file LightRAG skips pages already PROCESSED.

    Using len(texts) as the denominator would show 1/4 and snap to 100%; the real
    figure is published as pipeline_status["docs"].
    """
    texts = ["p1", "p2", "p3", "p4"]
    calls: list[tuple[int, int]] = []
    rag = _rag([{"processed": 3}, {"processed": 4, "processing": 1}])

    async def namespace(_name, workspace=None):
        return {"docs": 1}  # only one page left to process

    fake_kg = SimpleNamespace(get_namespace_data=namespace)
    with patch.dict("sys.modules", {"lightrag.kg.shared_storage": fake_kg}):
        await _run(rag, texts, calls)

    assert all(total == 1 for _, total in calls), calls
    assert calls[-1] == (1, 1)


async def test_bogus_run_total_is_ignored():
    """That value is per-workspace state: early ticks can still hold the
    previous run's number, so anything outside 1..len(texts) is discarded."""
    texts = ["p1", "p2"]
    calls: list[tuple[int, int]] = []
    rag = _rag([{}, {"processed": 1}])

    async def namespace(_name, workspace=None):
        return {"docs": 99}  # left over from a bigger run

    fake_kg = SimpleNamespace(get_namespace_data=namespace)
    with patch.dict("sys.modules", {"lightrag.kg.shared_storage": fake_kg}):
        await _run(rag, texts, calls)

    assert all(total == len(texts) for _, total in calls), calls


async def test_failed_documents_count_as_finished():
    """A failed page must advance the bar, or it stalls until the run ends."""
    texts = ["p1", "p2"]
    calls: list[tuple[int, int]] = []
    rag = _rag([{}, {"failed": 1, "processing": 1}])

    await _run(rag, texts, calls)

    assert 1 in [done for done, _ in calls[:-1]], calls
