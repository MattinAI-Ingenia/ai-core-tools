"""Unit tests for the LightRAG indexing token accumulator (T027).

These run without a database — pure logic tests.
"""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

# Ensure backend is importable
_backend = str(Path(__file__).resolve().parent.parent.parent / "backend")
if _backend not in sys.path:
    sys.path.insert(0, _backend)

from tools.vector_stores.lightrag.token_accumulator import IndexingTokenAccumulator


class TestIndexingTokenAccumulator:
    def test_empty(self):
        acc = IndexingTokenAccumulator()
        t = acc.totals()
        assert t["prompt_tokens"] == 0
        assert t["completion_tokens"] == 0
        assert t["total_tokens"] == 0
        assert t["llm_calls"] == 0
        assert t["tokens_source"] == "provider"

    def test_single_provider_call(self):
        acc = IndexingTokenAccumulator()
        acc.add_llm_usage(prompt=500, completion=200, source="provider")
        t = acc.totals()
        assert t["prompt_tokens"] == 500
        assert t["completion_tokens"] == 200
        assert t["total_tokens"] == 700
        assert t["llm_calls"] == 1
        assert t["tokens_source"] == "provider"

    def test_multiple_provider_calls_accumulate(self):
        acc = IndexingTokenAccumulator()
        acc.add_llm_usage(100, 30, "provider")
        acc.add_llm_usage(200, 60, "provider")
        acc.add_llm_usage(150, 40, "provider")
        t = acc.totals()
        assert t["prompt_tokens"] == 450
        assert t["completion_tokens"] == 130
        assert t["total_tokens"] == 580
        assert t["llm_calls"] == 3
        assert t["tokens_source"] == "provider"

    def test_any_estimated_call_downgrades_source(self):
        acc = IndexingTokenAccumulator()
        acc.add_llm_usage(100, 30, "provider")
        acc.add_llm_usage(200, 60, "estimated")  # ← triggers downgrade
        t = acc.totals()
        assert t["tokens_source"] == "estimated"

    def test_all_estimated_calls(self):
        acc = IndexingTokenAccumulator()
        acc.add_llm_usage(100, 30, "estimated")
        acc.add_llm_usage(200, 60, "estimated")
        t = acc.totals()
        assert t["tokens_source"] == "estimated"
        assert t["total_tokens"] == 390

    def test_total_equals_prompt_plus_completion(self):
        acc = IndexingTokenAccumulator()
        acc.add_llm_usage(314, 159, "provider")
        t = acc.totals()
        assert t["total_tokens"] == t["prompt_tokens"] + t["completion_tokens"]
