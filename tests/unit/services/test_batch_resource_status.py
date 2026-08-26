"""_resolve_batch_resource_status: what one LightRAG-batch resource ends up as.

The bug this pins: a resource that had genuinely finished (8/8 pages) got
marked 'error' anyway because the batch call raised an exception elsewhere in
the run (a container restart mid-batch, here) — the finalization loop trusted
only the call's return value, which an exception wipes to {}, discarding the
resource's own live-updated progress row along with it.
"""

from services.resource_service import ResourceService


def _resolve(**kwargs):
    counts = kwargs.pop('counts', {})
    return ResourceService._resolve_batch_resource_status(
        counts,
        kwargs.pop('row_progress_done', None),
        kwargs.pop('row_progress_total', None),
        kwargs.pop('parked_status', None),
    )


class TestSurvivesAnExceptionInTheBatchCall:
    """counts={} is what every resource gets when the batch call raised —
    completeness must fall back to the row's own progress."""

    def test_a_resource_that_had_finished_still_lands_ready(self):
        assert _resolve(counts={}, row_progress_done=8, row_progress_total=8) == (True, 'ready')

    def test_a_resource_that_had_not_finished_lands_error(self):
        assert _resolve(counts={}, row_progress_done=2, row_progress_total=23) == (False, 'error')

    def test_a_resource_never_touched_lands_error(self):
        assert _resolve(counts={}, row_progress_done=None, row_progress_total=None) == (False, 'error')


class TestNormalReturn:
    """counts carries real numbers when the batch call succeeded — same
    completeness rule, just fed from the call's own return value."""

    def test_complete_from_counts(self):
        assert _resolve(counts={'total': 10, 'processed': 10}) == (True, 'ready')

    def test_incomplete_from_counts(self):
        assert _resolve(counts={'total': 10, 'processed': 4}) == (False, 'error')


class TestParkedStatusOnlyAppliesWhenIncomplete:
    """A stop request must not downgrade a resource that actually finished —
    LightRAG lets already-fed documents run to completion under both pause
    and cancel; a finished one is 'ready', not 'paused'/'cancelled'."""

    def test_complete_ignores_parked_status(self):
        assert _resolve(counts={'total': 5, 'processed': 5}, parked_status='cancelled') == (True, 'ready')

    def test_incomplete_pause_is_parked_paused(self):
        assert _resolve(counts={'total': 5, 'processed': 2}, parked_status='paused') == (False, 'paused')

    def test_incomplete_cancel_is_parked_cancelled(self):
        assert _resolve(counts={'total': 5, 'processed': 2}, parked_status='cancelled') == (False, 'cancelled')


class TestProgressUpdateFields:
    """A resource's own progress tick decides its own finalization — it does
    not wait for the whole batch call to return (see the docstring on
    _progress_update_fields for why that matters beyond just responsiveness).
    """

    def test_always_carries_the_raw_progress(self):
        fields = ResourceService._progress_update_fields(3, 10)
        assert fields['progress_done'] == 3
        assert fields['progress_total'] == 10

    def test_flips_to_ready_on_reaching_total(self):
        assert ResourceService._progress_update_fields(8, 8)['status'] == 'ready'

    def test_flips_to_ready_past_total(self):
        """processed + failed can exceed the original page count on a retry
        that re-chunked a document differently; still done either way."""
        assert ResourceService._progress_update_fields(9, 8)['status'] == 'ready'

    def test_does_not_flip_before_total(self):
        assert 'status' not in ResourceService._progress_update_fields(3, 8)

    def test_does_not_flip_on_a_zero_total(self):
        """total not yet known (chunk counting still running) must not read
        as "already done"."""
        assert 'status' not in ResourceService._progress_update_fields(0, 0)
