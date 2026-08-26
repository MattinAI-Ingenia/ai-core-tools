"""Unit tests for the progress bar's clock and ETA.

The two interact: `elapsed_seconds` carries the minutes banked by earlier runs
so the timer survives a pause, while the ETA must extrapolate over the current
batch alone — the banked minutes belong to pages that are not in `done`.
Feeding them to the estimate inflates it by however long the run was paused.

The DB session is a MagicMock returning one prepared chain per query, in the
order get_indexing_progress issues them.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

from services.resource_service import ETA_WARMUP_SECONDS, ResourceService


def _progress(*, batch_age_seconds, banked, done, total):
    """Run get_indexing_progress against a batch of the given age and history."""
    batch = datetime.now() - timedelta(seconds=batch_age_seconds)

    q_batch = MagicMock()
    q_batch.filter.return_value.scalar.return_value = batch
    q_sums = MagicMock()
    q_sums.filter.return_value.one.return_value = (done, total, 0)
    q_current = MagicMock()
    q_current.filter.return_value.limit.return_value.scalar.return_value = 'file.pdf'
    q_banked = MagicMock()
    q_banked.filter.return_value.scalar.return_value = banked

    db = MagicMock()
    db.query.side_effect = [q_batch, q_sums, q_current, q_banked]
    return ResourceService.get_indexing_progress(db, repository_id=1)


class TestElapsedSurvivesAPause:
    def test_banked_seconds_are_added_to_the_clock(self):
        p = _progress(batch_age_seconds=60, banked=1200, done=5, total=30)
        assert 1255 <= p['elapsed_seconds'] <= 1262  # 20 min banked + ~1 min live


class TestEtaIgnoresBankedTime:
    def test_eta_extrapolates_over_this_batch_only(self):
        """The regression this pins: with banked time in the formula the ETA
        would be ~(25 * 1260 / 5) = 6300s instead of ~300s."""
        p = _progress(batch_age_seconds=60, banked=1200, done=5, total=30)
        assert 280 <= p['estimated_remaining_seconds'] <= 330

    def test_a_paused_run_and_a_fresh_one_estimate_the_same(self):
        """Same work, same pace — the pause must not change the estimate."""
        paused = _progress(batch_age_seconds=60, banked=1200, done=5, total=30)
        fresh = _progress(batch_age_seconds=60, banked=0, done=5, total=30)
        assert paused['estimated_remaining_seconds'] == fresh['estimated_remaining_seconds']


class TestEtaWarmup:
    def test_withheld_before_the_warmup(self):
        """Early on, `done` is dominated by pages LightRAG skips by dedup, so
        the extrapolation reads absurdly low. No answer beats a wrong one."""
        p = _progress(batch_age_seconds=ETA_WARMUP_SECONDS - 5, banked=0, done=5, total=30)
        assert p['estimated_remaining_seconds'] is None
        assert p['estimated_total_time_seconds'] is None
        assert p['elapsed_seconds'] > 0, "the clock still runs, only the ETA waits"

    def test_published_after_the_warmup(self):
        p = _progress(batch_age_seconds=ETA_WARMUP_SECONDS + 5, banked=0, done=5, total=30)
        assert p['estimated_remaining_seconds'] is not None

    def test_a_resumed_batch_waits_too(self):
        """Banked time must not let a young batch skip the warmup."""
        p = _progress(batch_age_seconds=2, banked=99999, done=5, total=30)
        assert p['estimated_remaining_seconds'] is None
