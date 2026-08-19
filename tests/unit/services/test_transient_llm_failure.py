"""Classifier that decides whether a batch should abort or keep going.

With the LLM endpoint down, every remaining resource fails in milliseconds: a
1000-file batch would march to 'error' without extracting a single token. The
batch must stop instead, so "resume indexing" can pick it up when the endpoint
is back.
"""

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError

from services.resource_service import _is_transient_llm_failure


def _request():
    return httpx.Request("POST", "http://vllm:8095/v1/chat/completions")


@pytest.mark.parametrize(
    "exc",
    [
        APIConnectionError(request=_request()),
        APITimeoutError(request=_request()),
    ],
)
def test_endpoint_unreachable_is_transient(exc):
    assert _is_transient_llm_failure(exc) is True


def test_lightrag_timeout_is_transient():
    """LightRAG caps role calls at LLM_TIMEOUT and re-raises the builtin
    TimeoutError, not the provider's class — the likely error after a suspend."""
    assert _is_transient_llm_failure(TimeoutError("LLM call timed out after 240s")) is True
    # asyncio.TimeoutError is the same class on 3.11+, pinned here on purpose.
    import asyncio

    assert _is_transient_llm_failure(asyncio.TimeoutError()) is True


def test_wrapped_cause_is_found():
    """LangChain wraps provider errors, so the outermost exception is generic."""
    try:
        try:
            raise APIConnectionError(request=_request())
        except APIConnectionError as inner:
            raise RuntimeError("Error in LLM call") from inner
    except RuntimeError as outer:
        assert _is_transient_llm_failure(outer) is True


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("No content extracted from resource"),
        RuntimeError("lightrag-hku is not installed"),
    ],
)
def test_document_level_failures_are_not_transient(exc):
    """A bad file must stay 'error'; retrying the batch would not help it."""
    assert _is_transient_llm_failure(exc) is False


def test_cycle_in_the_cause_chain_terminates():
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert _is_transient_llm_failure(a) is False
