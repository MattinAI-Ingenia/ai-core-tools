import json
import sys
import types
from types import SimpleNamespace

from services.embedding_service_service import EmbeddingServiceService
from tools.embeddingTools import get_embeddings_model


class FakeGoogleGenerativeAIEmbeddings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def embed_query(self, text: str):
        return [1.0, 2.0, 3.0]


def _install_fake_google_embeddings(monkeypatch):
    module = types.ModuleType("langchain_google_genai")
    module.GoogleGenerativeAIEmbeddings = FakeGoogleGenerativeAIEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_google_genai", module)


def _install_fake_google_credentials(monkeypatch):
    google_module = types.ModuleType("google")
    oauth2_module = types.ModuleType("google.oauth2")
    service_account_module = types.ModuleType("google.oauth2.service_account")

    class FakeCredentials:
        @staticmethod
        def from_service_account_info(info, scopes):
            return {"info": info, "scopes": scopes}

    service_account_module.Credentials = FakeCredentials
    oauth2_module.service_account = service_account_module
    google_module.oauth2 = oauth2_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.oauth2", oauth2_module)
    monkeypatch.setitem(sys.modules, "google.oauth2.service_account", service_account_module)


def test_google_embeddings_uses_ai_studio_api_key(monkeypatch):
    _install_fake_google_embeddings(monkeypatch)
    service = SimpleNamespace(
        provider="Google",
        description="gemini-embedding-2-preview",
        api_key="AIza-test",
        endpoint="",
        api_version=None,
    )

    embeddings = get_embeddings_model(service)

    assert embeddings.kwargs == {
        "model": "gemini-embedding-2-preview",
        "api_key": "AIza-test",
    }


def test_google_cloud_embeddings_uses_vertex_credentials(monkeypatch):
    _install_fake_google_embeddings(monkeypatch)
    _install_fake_google_credentials(monkeypatch)
    service_account_json = {"type": "service_account", "project_id": "mattin-test"}
    service = SimpleNamespace(
        provider="GoogleCloud",
        description="gemini-embedding-2-preview",
        api_key=json.dumps(service_account_json),
        endpoint="mattin-test",
        api_version="europe-west1",
    )

    embeddings = get_embeddings_model(service)

    assert embeddings.kwargs["model"] == "gemini-embedding-2-preview"
    assert embeddings.kwargs["project"] == "mattin-test"
    assert embeddings.kwargs["location"] == "europe-west1"
    assert embeddings.kwargs["vertexai"] is True
    assert embeddings.kwargs["credentials"]["info"] == service_account_json
    assert embeddings.kwargs["credentials"]["scopes"] == [
        "https://www.googleapis.com/auth/cloud-platform"
    ]


def test_google_cloud_connection_requires_project_id():
    result = EmbeddingServiceService.test_connection_with_config(
        {
            "provider": "GoogleCloud",
            "description": "gemini-embedding-2-preview",
            "api_key": '{"type":"service_account"}',
            "endpoint": "",
        }
    )

    assert result == {
        "status": "error",
        "message": "Project ID is required for Google Cloud provider",
    }
