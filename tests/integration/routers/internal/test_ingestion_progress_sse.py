"""Bug regression: the ingestion-progress SSE stream must not report success
for a run that stopped without finishing.

``GET /internal/apps/{app_id}/repositories/{repository_id}/ingestion-progress/{session_id}``
gates its ``complete`` event purely on
``ResourceService.get_ingestion_liveness(...)['is_indexing']``: as soon as the
silo's advisory lock is not held, it emits a bare
``event: complete\\ndata: {}\\n\\n`` with no check of whether the resources that
just stopped are ``'ready'`` (real success) or ``'error'``/``'pending'``
(aborted -- e.g. left behind by the batch-failure bug in
``_index_resources_background``, see ``test_indexing_batch_failure.py``).

A dead background thread (crashed, or the process restarted) leaves exactly
this signature: no lock held, but resources still not 'ready'. The client
reads ``event: complete`` as "all done" regardless.
"""

from datetime import datetime, timedelta


def _add_resource(db, repo, *, name, status, progress_started_at=None, error_message=None):
    from models.resource import Resource

    resource = Resource(
        name=name, uri=name, type=".pdf", status=status, repository_id=repo.repository_id,
        progress_started_at=progress_started_at, error_message=error_message,
    )
    db.add(resource)
    db.flush()
    return resource


def _sse_url(app_id: int, repository_id: int, session_id: str = "any-session-id") -> str:
    return f"/internal/apps/{app_id}/repositories/{repository_id}/ingestion-progress/{session_id}"


class TestIngestionProgressSSE:
    def test_sse_does_not_report_success_when_a_dead_run_left_work_behind(
        self, client, fake_app, repository, auth_headers, db
    ):
        """No active silo lock (the background thread is dead), but resources
        are left 'error' -- simulating the aftermath of the batch-failure bug.
        Today's endpoint still reports plain success.

        ``progress_started_at`` is set on both, matching what
        ``_index_resources_background`` actually stamps on every resource of a
        batch when it starts (see ``count_failed_resources``, which scopes its
        count to the latest such stamp) -- a real 'error' left behind by a
        batch always carries one.
        """
        batch_started = datetime.now()
        _add_resource(
            db, repository, name="left-behind-1.pdf", status="error",
            progress_started_at=batch_started,
            error_message="RuntimeError: 429 insufficient_quota",
        )
        _add_resource(
            db, repository, name="left-behind-2.pdf", status="error",
            progress_started_at=batch_started,
        )
        db.commit()

        resp = client.get(
            _sse_url(fake_app.app_id, repository.repository_id),
            headers=auth_headers,
        )

        assert resp.status_code == 200
        first_event = resp.text.split("\n\n")[0] + "\n\n"

        # Correct behavior: a dead run that left 'error' resources behind must
        # not be reported as unqualified success.
        assert first_event.startswith("event: failed\ndata: ")
        assert "2 file(s) could not be indexed" in first_event
        # The real exception message must be surfaced too, not just the
        # generic "could not be indexed" count.
        assert "429 insufficient_quota" in first_event

    def test_sse_reports_success_for_a_genuinely_finished_run(
        self, client, fake_app, repository, auth_headers, db
    ):
        """Companion/guard: a genuinely finished run (all resources 'ready',
        no lock held) must still read as plain success once bug 2 is fixed."""
        _add_resource(db, repository, name="done-1.pdf", status="ready")
        _add_resource(db, repository, name="done-2.pdf", status="ready")
        db.commit()

        resp = client.get(
            _sse_url(fake_app.app_id, repository.repository_id),
            headers=auth_headers,
        )

        assert resp.status_code == 200
        first_event = resp.text.split("\n\n")[0] + "\n\n"
        assert first_event == "event: complete\ndata: {}\n\n"

    def test_sse_ignores_a_stale_error_from_an_older_unrelated_batch(
        self, client, fake_app, repository, auth_headers, db
    ):
        """Regression guard: an 'error' row left behind by an OLDER batch must
        not taint a brand-new, fully successful run. ``count_failed_resources``
        has to scope its count to the latest ``progress_started_at`` batch, not
        the repository as a whole."""
        older_batch = datetime.now() - timedelta(hours=1)
        latest_batch = datetime.now()

        _add_resource(
            db, repository, name="old-failed.pdf", status="error",
            progress_started_at=older_batch,
        )
        _add_resource(
            db, repository, name="new-done-1.pdf", status="ready",
            progress_started_at=latest_batch,
        )
        _add_resource(
            db, repository, name="new-done-2.pdf", status="ready",
            progress_started_at=latest_batch,
        )
        db.commit()

        resp = client.get(
            _sse_url(fake_app.app_id, repository.repository_id),
            headers=auth_headers,
        )

        assert resp.status_code == 200
        first_event = resp.text.split("\n\n")[0] + "\n\n"
        assert first_event == "event: complete\ndata: {}\n\n", (
            "a stale 'error' resource from an older, unrelated batch must not "
            "make a brand-new successful run report failure"
        )
