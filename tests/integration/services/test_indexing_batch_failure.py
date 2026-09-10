"""Bug regression: a crash inside the LightRAG batch call must not silently
leave resources 'pending' forever.

``ResourceService._index_resources_background``'s LightRAG branch enqueues
every resource then drains the whole batch in ONE
``SiloService.process_enqueued_batch(...)`` call. If that call raises before
ever invoking the ``feed`` callback it was handed (e.g. an OpenAI 429 during
the embedding probe -- the very first thing the pipeline does), the bare
``except Exception:`` around that call swallows it and ``enqueued_resource_ids``
stays empty -- so the finalization loop (the ONLY code that calls
``_resolve_batch_resource_status`` and would mark a resource 'error') iterates
zero times. Every resource stamped 'pending' at the top of the run therefore
stays 'pending' forever: the run "completes" with nothing indexed and no
visible failure.

This exercises ``_index_resources_background`` with real, committed rows: it
opens its own ``SessionLocal()`` internally, so the shared, rolled-back ``db``
fixture session would never see its writes. Follows the ``committed_repo``
fixture pattern from ``test_resume_indexing_status_reset.py``.
"""

import threading
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from models.app import App
from models.repository import Repository
from models.resource import Resource
from models.silo import Silo
from models.user import User
from services.resource_service import ResourceService
from services.silo_service import SiloService


@pytest.fixture
def committed_lightrag_repo(test_engine):
    """A really-committed User -> App -> LIGHTRAG Silo -> Repository chain, cleaned up after."""
    session = Session(bind=test_engine)
    user = User(email="batch-failure-test@mattin-test.com", name="Batch Failure Test",
                is_active=True, platform_role="editor")
    session.add(user)
    session.flush()
    app_obj = App(name="Batch Failure Test App", slug="batch-failure-test-app",
                  owner_id=user.user_id, agent_rate_limit=0, max_file_size_mb=10)
    session.add(app_obj)
    session.flush()
    silo = Silo(name="Batch Failure Test Silo", description="", silo_type="DOMAIN",
                app_id=app_obj.app_id, vector_db_type="LIGHTRAG")
    session.add(silo)
    session.flush()
    repository = Repository(name="batch-failure-repo", app_id=app_obj.app_id, silo_id=silo.silo_id)
    session.add(repository)
    session.commit()

    yield session, repository, silo

    session.query(Resource).filter(Resource.repository_id == repository.repository_id).delete()
    session.query(Repository).filter(Repository.repository_id == repository.repository_id).delete()
    session.query(Silo).filter(Silo.silo_id == silo.silo_id).delete()
    session.query(App).filter(App.app_id == app_obj.app_id).delete()
    session.query(User).filter(User.user_id == user.user_id).delete()
    session.commit()
    session.close()


def _add(session, repo, *, name, status="pending"):
    resource = Resource(name=name, uri=name, type=".pdf", status=status, repository_id=repo.repository_id)
    session.add(resource)
    session.commit()
    return resource


class _SyncThread(threading.Thread):
    """``threading.Thread`` subclass that runs the *indexing* thread's target
    synchronously on ``.start()`` -- deterministic, no real background thread.

    Only intercepts threads named ``index-*`` (the one
    ``_index_resources_background`` creates): the LightRAG batch's own
    feeder uses ``asyncio.to_thread``, which is itself backed by a real
    ``threading.Thread``-based ``ThreadPoolExecutor`` worker under the hood --
    patching ``threading.Thread`` unconditionally deadlocks that worker
    (its `.start()` would run the pool's own blocking `work_queue.get()`
    inline instead of on a real thread).
    """

    def start(self):
        if self.name.startswith("index-"):
            self.run()
        else:
            super().start()


@pytest.fixture
def no_op_indexing_plumbing():
    """Deterministic background-thread run: no real thread, no real advisory
    lock, no real LightRAG postgres pool reset, and no real file extraction
    for the chunk-count pass (there is no file on disk for these rows)."""
    with patch("threading.Thread", _SyncThread), \
         patch("services.silo_indexing_lock.acquire", return_value=object()), \
         patch("services.silo_indexing_lock.release"), \
         patch("tools.vector_stores.lightrag_store.reset_lightrag_postgres_pool"), \
         patch.object(SiloService, "count_resource_chunks", return_value=3):
        yield


class TestBatchFailureLeavesResourcesInError:
    def test_batch_failure_marks_resources_error_not_pending(
        self, committed_lightrag_repo, no_op_indexing_plumbing
    ):
        session, repo, silo = committed_lightrag_repo
        r1 = _add(session, repo, name="one.pdf")
        r2 = _add(session, repo, name="two.pdf")

        with patch.object(
            SiloService, "process_enqueued_batch",
            side_effect=RuntimeError("429 insufficient_quota"),
        ):
            ResourceService._index_resources_background([r1, r2], silo_id=silo.silo_id)

        session.expire_all()  # the run wrote through its own session/connection

        # (1) The bug: these currently stay 'pending' forever instead of
        # surfacing the failure as 'error'.
        r1_reloaded = session.get(Resource, r1.resource_id)
        r2_reloaded = session.get(Resource, r2.resource_id)
        assert r1_reloaded.status == "error", (
            "a batch call that crashed before feeding any resource must not "
            "leave resources stuck 'pending' forever"
        )
        assert r2_reloaded.status == "error"

        # The real exception message must be preserved, not just the status,
        # so the UI can show the actual cause instead of a generic failure.
        assert "429 insufficient_quota" in r1_reloaded.error_message
        assert "429 insufficient_quota" in r2_reloaded.error_message

        # (2) No phantom active batch should be reported once the run has
        # ended (successfully or not).
        progress = ResourceService.get_indexing_progress(session, repo.repository_id)
        assert progress is None, (
            "get_indexing_progress must not report an active batch once the "
            "run has ended"
        )

        # (3) Regression guard: this already passes today (liveness is driven
        # by the advisory lock, not by row status), and must keep passing.
        liveness = ResourceService.get_ingestion_liveness(session, repo.repository_id)
        assert liveness["is_indexing"] is False

    def test_batch_failure_after_partial_feeding_marks_the_unfed_ones_error(
        self, committed_lightrag_repo, no_op_indexing_plumbing
    ):
        """Same bug, mid-batch: the batch call feeds resources one at a time
        via the ``feed`` callback it is handed. A crash after it has already
        consumed one resource must not leave the *other*, never-fed
        resource(s) stuck 'pending' either."""
        session, repo, silo = committed_lightrag_repo
        fed = _add(session, repo, name="fed.pdf")
        never_fed = _add(session, repo, name="never-fed.pdf")

        def _feed_once_then_raise(silo_id, doc_ids_by_resource, progress_callback=None,
                                   should_cancel=None, retry_failed=False, feed=None, window=3):
            import asyncio
            # ``feed`` is the async ``_feed_next`` closure; run it once on this
            # thread's event loop (already set up by ``_run()``) to simulate
            # the pipeline consuming exactly one resource before blowing up.
            asyncio.get_event_loop().run_until_complete(feed())
            raise RuntimeError("429 insufficient_quota")

        with patch.object(SiloService, "process_enqueued_batch", side_effect=_feed_once_then_raise), \
             patch.object(SiloService, "extract_resource_documents", return_value=("batch", ["fake-doc"])), \
             patch.object(SiloService, "split_documents_for_lightrag",
                          return_value=(["text"], ["path"], ["id"])):
            ResourceService._index_resources_background([fed, never_fed], silo_id=silo.silo_id)

        session.expire_all()

        never_fed_reloaded = session.get(Resource, never_fed.resource_id)
        assert never_fed_reloaded.status == "error", (
            "a resource that was never handed to the pipeline before the "
            "crash must not stay 'pending' forever either"
        )
        assert "429 insufficient_quota" in never_fed_reloaded.error_message
