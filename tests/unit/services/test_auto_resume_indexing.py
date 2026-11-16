"""A batch that ends with 'error' resources is resumed once, unless the user stopped it."""
from unittest.mock import MagicMock, patch

import pytest

from services.resource_service import ResourceService


def _resume_calls(stop_mode, has_errors):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = (1,) if has_errors else None
    with patch("db.database.SessionLocal", return_value=db), \
         patch.object(ResourceService, "ingestion_stop_mode", return_value=stop_mode), \
         patch.object(ResourceService, "resume_indexing") as resume:
        ResourceService._auto_resume_once(7, [1, 2])
    return resume.call_count


def test_resumes_once_when_the_batch_left_errors():
    assert _resume_calls(None, True) == 1


@pytest.mark.parametrize("stop_mode,has_errors", [("pause", True), ("cancel", True), (None, False)])
def test_does_not_resume_after_a_stop_or_without_errors(stop_mode, has_errors):
    assert _resume_calls(stop_mode, has_errors) == 0
