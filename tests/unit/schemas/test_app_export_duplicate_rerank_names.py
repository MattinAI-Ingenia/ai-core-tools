"""Unit tests: AppExportFileSchema rejects duplicate rerank service names."""
import pytest
from pydantic import ValidationError

from schemas.export_schemas import AppExportFileSchema


def _file(names: list[str]) -> dict:
    return {
        "metadata": {"export_version": "1.0", "export_date": "2026-01-01T00:00:00", "component_type": "app"},
        "app": {"name": "A"},
        "rerank_services": [
            {"name": n, "provider": "CohereCompatible", "model_name": "m", "endpoint": f"https://h{i}.example.com/v1"}
            for i, n in enumerate(names)
        ],
    }


def test_duplicate_name_rejected():
    with pytest.raises(ValidationError, match="Duplicate rerank service name"):
        AppExportFileSchema(**_file(["rr", "rr"]))


def test_case_sensitive_names_allowed():
    assert len(AppExportFileSchema(**_file(["rr", "RR"])).rerank_services) == 2
