"""Unit tests for pausing/cancelling a running ingestion.

The DB session is a MagicMock: these pin the *decision logic* (which status a
mode parks resources in, what counts as a stop signal, refusing a bad mode),
not SQLAlchemy itself.
"""

import pytest
from unittest.mock import MagicMock, patch

from services.resource_service import ResourceService


def _db_with_active_batch(batch="2026-08-25T09:00:00", updated=3):
    """A session whose queries report one live batch and *updated* rows changed."""
    db = MagicMock()
    db.query.return_value.filter.return_value.scalar.return_value = batch
    db.query.return_value.filter.return_value.update.return_value = updated
    return db


class TestRequestIngestionStopMode:
    @pytest.mark.parametrize(
        "mode,expected_status",
        [("pause", "paused"), ("cancel", "cancelled")],
    )
    def test_mode_parks_resources_in_matching_status(self, mode, expected_status):
        db = _db_with_active_batch()
        with patch("services.silo_indexing_lock.is_locked", return_value=True):
            result = ResourceService.request_ingestion_stop(db, repository_id=1, mode=mode)

        assert result["mode"] == mode
        assert result["stopped"] == 3
        assert result["was_running"] is True
        update_call = db.query.return_value.filter.return_value.update
        assert update_call.call_args[0][0] == {"status": expected_status}

    @pytest.mark.parametrize("bad_mode", ["stop", "PAUSE", "", None, "delete"])
    def test_unknown_mode_is_refused(self, bad_mode):
        """Refuse before touching the DB: a typo must not silently do nothing."""
        db = MagicMock()
        with pytest.raises(ValueError, match="pause"):
            ResourceService.request_ingestion_stop(db, repository_id=1, mode=bad_mode)
        db.query.assert_not_called()

    def test_no_active_batch_reports_nothing_stopped(self):
        """No batch to park, but the signal is still recorded and committed.

        (It used to skip the commit here; that was the bug — see
        TestStopSignalIsIndependentOfPendingRows.)
        """
        db = MagicMock()
        db.query.return_value.filter.return_value.scalar.return_value = None
        with patch("services.silo_indexing_lock.is_locked", return_value=False):
            result = ResourceService.request_ingestion_stop(db, repository_id=1, mode="pause")
        assert result == {"mode": "pause", "stopped": 0, "was_running": False}
        db.commit.assert_called_once()

    def test_only_not_yet_started_resources_are_parked(self):
        """The file being indexed is left alone — the loop owns its status.

        Reads the bound value out of the SQLAlchemy expression handed to
        ``filter``: the UPDATE must select ``status == 'pending'`` and never
        touch a row already in ``indexing``.
        """
        db = _db_with_active_batch()
        with patch("services.silo_indexing_lock.is_locked", return_value=True):
            ResourceService.request_ingestion_stop(db, repository_id=1, mode="cancel")

        status_values = []
        for call in db.query.return_value.filter.call_args_list:
            for expr in call.args:
                left = getattr(expr, "left", None)
                right = getattr(expr, "right", None)
                if getattr(left, "key", None) == "status" and hasattr(right, "value"):
                    status_values.append(right.value)

        assert "pending" in status_values, status_values
        assert "indexing" not in status_values, status_values


class TestStopSignalIsIndependentOfPendingRows:
    """Regression guard for the gap found while testing this live.

    The signal used to be derived from "is any resource parked?", which looked
    equivalent and was not: once the whole batch is enqueued there are no
    'pending' rows left to park, so a stop arriving during the long LightRAG
    batch — the hour-long part this button exists for — wrote nothing and was
    silently ignored.
    """

    def test_signal_is_written_even_with_nothing_left_to_park(self):
        db = MagicMock()
        # No active batch => no resource can be parked.
        db.query.return_value.filter.return_value.scalar.return_value = None
        with patch("services.silo_indexing_lock.is_locked", return_value=True):
            result = ResourceService.request_ingestion_stop(db, 1, mode="cancel")

        assert result["stopped"] == 0          # nothing to park...
        updates = [c.args[0] for c in db.query.return_value.filter.return_value.update.call_args_list]
        assert {"ingestion_stop_mode": "cancel"} in updates  # ...but the signal is set
        db.commit.assert_called_once()

    @pytest.mark.parametrize(
        "stored,expected",
        [("pause", "pause"), ("cancel", "cancel"), (None, None), ("", None), ("bogus", None)],
    )
    def test_stop_check_returns_the_mode(self, stored, expected):
        """The mode itself, not a bool: the run needs it to tell pause from cancel."""
        db = MagicMock()
        db.query.return_value.filter.return_value.scalar.return_value = stored
        with patch("db.database.SessionLocal", return_value=db):
            assert ResourceService.ingestion_stop_mode(1) == expected

    def test_stop_check_never_raises(self):
        """A failing check must not kill a running indexing job."""
        with patch("db.database.SessionLocal", side_effect=RuntimeError("db down")):
            assert ResourceService.ingestion_stop_mode(1) is None

    def test_clear_wipes_the_signal(self):
        """A stop from a previous run must not abort the next ingestion."""
        db = MagicMock()
        ResourceService.clear_ingestion_stop(db, 1)
        updates = [c.args[0] for c in db.query.return_value.filter.return_value.update.call_args_list]
        assert {"ingestion_stop_mode": None} in updates
        db.commit.assert_called_once()


class TestStopStatusesContract:
    def test_both_modes_have_a_parked_status(self):
        assert set(ResourceService.STOP_STATUSES) == {"paused", "cancelled"}


class TestCancelDiscardsUnindexed:
    """Cancel must not leave rows for files the silo never saw."""

    def test_discards_immediately_when_no_run_is_alive(self):
        """The run does the discarding on its way out; with no run, do it here.

        Otherwise a cancel that races the end of a run (or lands after it)
        leaves those resources in 'cancelled' for good.
        """
        db = _db_with_active_batch()
        with patch("services.silo_indexing_lock.is_locked", return_value=False), \
             patch.object(ResourceService, "_discard_unindexed_after_cancel") as discard:
            ResourceService.request_ingestion_stop(db, 1, mode="cancel")
        discard.assert_called_once_with(1)

    def test_does_not_discard_while_a_run_is_alive(self):
        """The run finishes its in-flight files first, then discards."""
        db = _db_with_active_batch()
        with patch("services.silo_indexing_lock.is_locked", return_value=True), \
             patch.object(ResourceService, "_discard_unindexed_after_cancel") as discard:
            ResourceService.request_ingestion_stop(db, 1, mode="cancel")
        discard.assert_not_called()

    def test_pause_never_discards(self):
        """Pause exists to keep them — discarding would defeat it."""
        db = _db_with_active_batch()
        with patch("services.silo_indexing_lock.is_locked", return_value=False), \
             patch.object(ResourceService, "_discard_unindexed_after_cancel") as discard:
            ResourceService.request_ingestion_stop(db, 1, mode="pause")
        discard.assert_not_called()
