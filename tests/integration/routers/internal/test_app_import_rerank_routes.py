"""Route-level tests: duplicate rerank names -> 422; preview exposes service endpoints."""
import json

import pytest

pytestmark = pytest.mark.integration


def _file(rerank, embeddings=()):
    return {
        "metadata": {"export_version": "1.0.0", "export_date": "2026-01-01T00:00:00", "component_type": "app"},
        "app": {"name": "Imported"},
        "rerank_services": rerank,
        "embedding_services": list(embeddings),
    }


def _rr(name, endpoint):
    return {"name": name, "provider": "CohereCompatible", "model_name": "m", "endpoint": endpoint}


def _upload(data):
    return {"file": ("app.json", json.dumps(data), "application/json")}


@pytest.mark.parametrize("path", ["/internal/apps/import", "/internal/apps/preview-import"])
def test_duplicate_rerank_names_return_422(client, auth_headers, path):
    data = _file([_rr("rr", "https://a.example.com/v1"), _rr("rr", "https://b.example.com/v1")])
    r = client.post(path, files=_upload(data), headers=auth_headers)
    assert r.status_code == 422, r.text
    assert "Duplicate rerank service name" in r.text


def test_preview_includes_endpoints(client, auth_headers):
    emb = {"name": "emb", "provider": "OpenAI", "model_name": "e", "endpoint": "https://emb.example.com/v1"}
    data = _file([_rr("rr", "https://rr.example.com:8443/v1/rerank")], [emb])
    r = client.post("/internal/apps/preview-import", files=_upload(data), headers=auth_headers)
    assert r.status_code == 200, r.text
    body = r.json()
    items = [i for v in body.values() if isinstance(v, list) for i in v if isinstance(i, dict) and "component_type" in i]
    by_name = {i["component_name"]: i for i in items}
    assert by_name["rr"]["endpoint"] == "https://rr.example.com:8443/v1/rerank"
    assert by_name["emb"]["endpoint"] == "https://emb.example.com/v1"
