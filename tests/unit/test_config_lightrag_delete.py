"""Unit tests for LIGHTRAG_DELETE_LLM_CACHE_ON_DOC_DELETE env parsing in config."""
import importlib

import pytest

import config

VAR = "LIGHTRAG_DELETE_LLM_CACHE_ON_DOC_DELETE"


@pytest.fixture
def reload_config(monkeypatch):
    yield lambda: importlib.reload(config)
    monkeypatch.undo()
    importlib.reload(config)


def test_unset_defaults_to_true(monkeypatch, reload_config):
    monkeypatch.delenv(VAR, raising=False)
    reload_config()
    assert config.LIGHTRAG_DELETE_LLM_CACHE_ON_DOC_DELETE is True


@pytest.mark.parametrize("raw, expected", [
    ("false", False),
    ("FALSE", False),
    ("TRUE", True),
    ("yes", False),
])
def test_env_value_parsed(monkeypatch, reload_config, raw, expected):
    monkeypatch.setenv(VAR, raw)
    reload_config()
    assert config.LIGHTRAG_DELETE_LLM_CACHE_ON_DOC_DELETE is expected
