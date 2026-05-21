"""Unit tests for ``tools.langsmith_config``.

These tests cover the three responsibilities of the module:

  * Settings resolution (per-App key vs env fallback)
  * Validation classification (auth/network/unknown failures)
  * RunnableConfig overrides (metadata, tags, run_name)

All LangSmith network calls are mocked — no live API is touched.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from tools.langsmith_config import (
    DEFAULT_LANGSMITH_ENDPOINT,
    LangSmithSettings,
    apply_tracing_to_config,
    build_tracing_config,
    clear_client_cache,
    resolve_langsmith_settings,
    validate_langsmith_key,
)


def _client_with_info_error(exc: BaseException) -> MagicMock:
    """Build a fake LangSmith client whose ``info`` property raises ``exc``."""
    client = MagicMock()
    type(client).info = PropertyMock(side_effect=exc)
    return client


def _make_app(*, app_id: int = 1, name: str = "test-app", key: str | None = None):
    """Build a lightweight stand-in for the App ORM model."""
    return SimpleNamespace(app_id=app_id, name=name, langsmith_api_key=key)


def _make_agent(*, agent_id: int = 7, name: str = "demo-agent", app=None, agent_type: str = "agent"):
    return SimpleNamespace(
        agent_id=agent_id,
        name=name,
        type=agent_type,
        has_memory=True,
        app=app or _make_app(),
    )


@pytest.fixture(autouse=True)
def _isolate_caches_and_env(monkeypatch):
    """Reset the module-level client cache and clear LangSmith env vars."""
    for var in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
    ):
        monkeypatch.delenv(var, raising=False)
    clear_client_cache()
    yield
    clear_client_cache()


class TestResolveLangsmithSettings:
    def test_returns_none_when_no_key_and_tracing_off(self):
        app = _make_app(key=None)
        assert resolve_langsmith_settings(app) is None

    def test_uses_app_key_with_app_name_as_project(self):
        app = _make_app(name="acme", key="lsv2_pt_real")
        settings = resolve_langsmith_settings(app)
        assert settings is not None
        assert settings.api_key == "lsv2_pt_real"
        assert settings.project_name == "acme"
        assert settings.source == "app"
        assert settings.endpoint == DEFAULT_LANGSMITH_ENDPOINT

    def test_app_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_API_KEY", "env_key")
        monkeypatch.setenv("LANGSMITH_PROJECT", "env_project")
        app = _make_app(name="acme", key="app_key")

        settings = resolve_langsmith_settings(app)
        assert settings is not None
        assert settings.api_key == "app_key"
        assert settings.project_name == "acme"
        assert settings.source == "app"

    def test_falls_back_to_env_when_app_empty_and_tracing_on(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_TRACING", "TRUE")
        monkeypatch.setenv("LANGSMITH_API_KEY", "global_key")
        monkeypatch.setenv("LANGSMITH_PROJECT", "global_project")

        settings = resolve_langsmith_settings(_make_app(key=""))
        assert settings is not None
        assert settings.api_key == "global_key"
        assert settings.project_name == "global_project"
        assert settings.source == "env"

    def test_ignores_env_when_tracing_disabled(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_TRACING", "false")
        monkeypatch.setenv("LANGSMITH_API_KEY", "global_key")
        assert resolve_langsmith_settings(_make_app(key="")) is None

    def test_ignores_env_when_api_key_missing(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        # No LANGSMITH_API_KEY set
        assert resolve_langsmith_settings(_make_app(key="")) is None

    def test_strips_whitespace_from_app_key(self):
        app = _make_app(key="  lsv2_pt_padded  ")
        settings = resolve_langsmith_settings(app)
        assert settings is not None
        assert settings.api_key == "lsv2_pt_padded"


class TestValidateLangsmithKey:
    """Validation calls ``Client.info`` — the broadest LangSmith endpoint
    that any authenticated key can hit regardless of read/write scopes."""

    def test_empty_key_is_unauthorized(self):
        result = validate_langsmith_key("")
        assert result.valid is False
        assert result.status == "unauthorized"

    def test_successful_call_returns_ok(self):
        fake_client = MagicMock()
        fake_client.info = MagicMock(name="LangSmithInfo", version="0.15.3")

        with patch("tools.langsmith_config.get_langsmith_client", return_value=fake_client):
            result = validate_langsmith_key("good_key")

        assert result.valid is True
        assert result.status == "ok"

    def test_empty_info_payload_is_unknown(self):
        fake_client = MagicMock()
        fake_client.info = None

        with patch("tools.langsmith_config.get_langsmith_client", return_value=fake_client):
            result = validate_langsmith_key("good_key")

        assert result.valid is False
        assert result.status == "unknown"

    def test_401_is_classified_as_unauthorized(self):
        fake_client = _client_with_info_error(PermissionError("401 Unauthorized"))

        with patch("tools.langsmith_config.get_langsmith_client", return_value=fake_client):
            result = validate_langsmith_key("bad_key")

        assert result.status == "unauthorized"

    def test_403_is_classified_as_unauthorized(self):
        # Rare on /info, but if LangSmith ever returns it we still report
        # the key as rejected.
        fake_client = _client_with_info_error(PermissionError("HTTPError 403 Forbidden"))

        with patch("tools.langsmith_config.get_langsmith_client", return_value=fake_client):
            result = validate_langsmith_key("bad_key")

        assert result.valid is False
        assert result.status == "unauthorized"

    def test_connection_error_is_classified_as_network(self):
        fake_client = _client_with_info_error(
            ConnectionError("Max retries exceeded — connection refused")
        )

        with patch("tools.langsmith_config.get_langsmith_client", return_value=fake_client):
            result = validate_langsmith_key("any_key")

        assert result.valid is False
        assert result.status == "network"

    def test_unknown_error_is_classified_as_unknown(self):
        fake_client = _client_with_info_error(RuntimeError("Something weird happened"))

        with patch("tools.langsmith_config.get_langsmith_client", return_value=fake_client):
            result = validate_langsmith_key("any_key")

        assert result.valid is False
        assert result.status == "unknown"


class TestBuildTracingConfig:
    @pytest.fixture
    def settings(self):
        return LangSmithSettings(
            api_key="key123",
            project_name="acme",
            endpoint=DEFAULT_LANGSMITH_ENDPOINT,
            source="app",
        )

    def test_overrides_include_expected_metadata_keys(self, settings):
        agent = _make_agent(agent_id=42, name="copilot", app=_make_app(app_id=9, name="acme"))
        user_context = {"user_id": 101, "email": "a@b.com"}

        with patch("tools.langsmith_config.LangChainTracer"), patch(
            "tools.langsmith_config.get_langsmith_client"
        ):
            _, overrides = build_tracing_config(
                settings,
                agent=agent,
                user_context=user_context,
                conversation_id=555,
                session_id="sess-xyz",
            )

        assert overrides["run_name"] == "agent:copilot"
        metadata = overrides["metadata"]
        assert metadata["agent_id"] == 42
        assert metadata["agent_name"] == "copilot"
        assert metadata["app_id"] == 9
        assert metadata["app_name"] == "acme"
        assert metadata["user_id"] == 101
        assert metadata["user_email"] == "a@b.com"
        assert metadata["conversation_id"] == 555
        assert metadata["session_id"] == "sess-xyz"
        assert metadata["tracing_source"] == "app"
        assert metadata["has_memory"] is True

    def test_overrides_drop_none_metadata(self, settings):
        agent = _make_agent(app=_make_app(app_id=9, name="acme"))

        with patch("tools.langsmith_config.LangChainTracer"), patch(
            "tools.langsmith_config.get_langsmith_client"
        ):
            _, overrides = build_tracing_config(
                settings,
                agent=agent,
                user_context=None,
                conversation_id=None,
                session_id=None,
            )

        # No conversation/session/user — those keys should be absent
        assert "conversation_id" not in overrides["metadata"]
        assert "session_id" not in overrides["metadata"]
        assert "user_id" not in overrides["metadata"]

    def test_tags_for_ocr_agent_include_ocr_marker(self, settings):
        agent = _make_agent(agent_id=3, agent_type="ocr_agent")

        with patch("tools.langsmith_config.LangChainTracer"), patch(
            "tools.langsmith_config.get_langsmith_client"
        ):
            _, overrides = build_tracing_config(
                settings,
                agent=agent,
                user_context={},
                conversation_id=None,
                session_id=None,
            )

        assert "agent:3" in overrides["tags"]
        assert "agent_type:ocr_agent" in overrides["tags"]
        assert "ocr" in overrides["tags"]


class TestApplyTracingToConfig:
    def test_merges_callbacks_metadata_and_tags(self):
        tracer = MagicMock(name="tracer")
        overrides = {
            "run_name": "agent:x",
            "metadata": {"agent_id": 1, "extra": "yes"},
            "tags": ["agent:1", "app:demo"],
        }
        config = {
            "callbacks": [MagicMock(name="existing-callback")],
            "metadata": {"trace_id": "abc"},
            "tags": ["pre-existing"],
            "configurable": {"thread_id": "t1"},
        }

        result = apply_tracing_to_config(config, tracer, overrides)

        assert tracer in result["callbacks"]
        assert result["metadata"]["trace_id"] == "abc"
        assert result["metadata"]["agent_id"] == 1
        assert result["metadata"]["extra"] == "yes"
        assert set(result["tags"]) == {"pre-existing", "agent:1", "app:demo"}
        assert result["run_name"] == "agent:x"
        # Existing configurable section is untouched
        assert result["configurable"]["thread_id"] == "t1"

    def test_handles_missing_keys_in_config(self):
        tracer = MagicMock(name="tracer")
        overrides = {
            "run_name": "agent:y",
            "metadata": {"k": "v"},
            "tags": ["t1"],
        }
        config: dict = {}

        result = apply_tracing_to_config(config, tracer, overrides)

        assert result["callbacks"] == [tracer]
        assert result["metadata"] == {"k": "v"}
        assert result["tags"] == ["t1"]
        assert result["run_name"] == "agent:y"
