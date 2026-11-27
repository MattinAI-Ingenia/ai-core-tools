from types import SimpleNamespace
from unittest.mock import patch

from services.rerank_health import DbRerankHealthRecorder
from tools.retrieval.rerank_transformer import RerankHealthRecorder


def test_db_recorder_satisfies_the_retrieval_interface():
    assert isinstance(DbRerankHealthRecorder(), RerankHealthRecorder)


def test_recorder_delegates_to_the_health_service():
    target = SimpleNamespace(service_id=5, last_error=None)
    with patch("services.rerank_health.ServiceHealthService") as health:
        recorder = DbRerankHealthRecorder()
        recorder.record_failure(target, "boom")
        recorder.record_success(target)
    assert health.record_failure_message.call_args.args[1:] == (5, "boom")
    assert health.record_success.call_args.args[1] is target
