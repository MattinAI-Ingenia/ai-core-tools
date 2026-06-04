"""Per-indexing-run LLM token accumulator.

This module provides :class:`IndexingTokenAccumulator`, a lightweight object
that collects ``prompt_tokens`` / ``completion_tokens`` from each LLM call
made during a single document indexing run.

Usage::

    acc = IndexingTokenAccumulator()
    # In each llm_model_func wrapper:
    acc.add_llm_usage(prompt=input_toks, completion=output_toks, source="provider")
    # After the run:
    totals = acc.totals()

``tokens_source`` is ``"provider"`` only when *all* LLM calls provided
provider-reported usage.  A single estimated call downgrades the whole run
to ``"estimated"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TokenSource = Literal["provider", "estimated"]


@dataclass
class IndexingTokenAccumulator:
    """Accumulates LLM token usage across all calls in one indexing run."""

    _prompt_tokens: int = field(default=0, init=False, repr=False)
    _completion_tokens: int = field(default=0, init=False, repr=False)
    _llm_calls: int = field(default=0, init=False, repr=False)
    _has_estimated: bool = field(default=False, init=False, repr=False)

    def add_llm_usage(
        self,
        prompt: int,
        completion: int,
        source: TokenSource = "provider",
    ) -> None:
        """Add usage from one LLM call."""
        self._prompt_tokens += prompt
        self._completion_tokens += completion
        self._llm_calls += 1
        if source == "estimated":
            self._has_estimated = True

    def totals(self) -> dict:
        """Return a dict suitable for passing directly to ``IndexingMetricRepository.create``."""
        return {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._prompt_tokens + self._completion_tokens,
            "tokens_source": "estimated" if self._has_estimated else "provider",
            "llm_calls": self._llm_calls,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"IndexingTokenAccumulator(calls={self._llm_calls}, "
            f"total={self._prompt_tokens + self._completion_tokens}, "
            f"source={'estimated' if self._has_estimated else 'provider'})"
        )
