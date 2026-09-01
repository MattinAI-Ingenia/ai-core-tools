"""Pins the hard token clamp on retrieved LightRAG context.

MAX_TOTAL_TOKENS is passed to LightRAG's own QueryParam, but its entity/
relation expansion can overshoot it (seen live: 155k tokens against a 120k
budget, crashing the synthesis call with a 400 from the LLM). Without a
clamp on our side, that failure is silent until it hits the LLM.
"""

from langchain_core.documents import Document

from tools.vector_stores.lightrag_store import _clamp_docs_to_token_budget


def test_over_budget_content_is_truncated():
    doc = Document(page_content="hola mundo " * 1000)
    _clamp_docs_to_token_budget([doc], max_total_tokens=10)
    assert len(doc.page_content) < len("hola mundo " * 1000)


def test_under_budget_content_is_untouched():
    doc = Document(page_content="hola mundo")
    _clamp_docs_to_token_budget([doc], max_total_tokens=1000)
    assert doc.page_content == "hola mundo"


def test_no_budget_falls_back_to_the_env_default(monkeypatch):
    monkeypatch.setenv("MAX_TOTAL_TOKENS", "5")
    doc = Document(page_content="hola mundo " * 1000)
    _clamp_docs_to_token_budget([doc], max_total_tokens=None)
    assert len(doc.page_content) < len("hola mundo " * 1000)
