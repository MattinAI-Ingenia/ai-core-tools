"""Pins the indexing-time estimate against this deployment's measured throughput.

The estimate was reporting 11-28 min for a corpus that took 115, and got *more*
optimistic every time the concurrency knobs were raised. Three independent
faults compounded, each harmless-looking on its own:

1. Per-call throughput multiplied by ``min(MAX_ASYNC_LLM, MAX_PARALLEL_INSERT)``
   — 48 here — assuming 3,360-8,640 tok/s where one GPU sustains 932-1,246.
2. Only prompt tokens were divided by a throughput measured over
   prompt+completion, understating the work by ~a third.
3. The empirical correction used ``min()``, so a reality slower than the
   heuristic was discarded and the estimate could never learn.

Measured anchors (indexing_metric, prompt+completion over wall-clock,
Qwen3-30B-A3B self-hosted with MAX_ASYNC_LLM=48): 932 tok/s over 115 min on one
silo, 1,246 tok/s over 74 min on another.
"""

import pytest

from services.silo_service import (
    _CLOUD_CONCURRENCY_CAP,
    _LLM_AGGREGATE_SELF_HOSTED,
    _llm_aggregate_throughput,
)

MEASURED_AGGREGATE_RANGE = (932.0, 1246.0)


class TestSelfHostedIsGpuBoundNotWorkerBound:
    """One GPU has a fixed token throughput: allowing more concurrent calls
    queues more work, it does not create more FLOPs."""

    @pytest.mark.parametrize("workers", [1, 4, 48, 512])
    def test_worker_count_does_not_change_the_aggregate(self, workers):
        assert _llm_aggregate_throughput(
            "Qwen3-30B-A3B-Instruct", True, workers,
        ) == _LLM_AGGREGATE_SELF_HOSTED

    def test_the_band_brackets_what_was_actually_measured(self):
        opt, pess = _llm_aggregate_throughput("Qwen3-30B-A3B-Instruct", True, 48)
        low, high = MEASURED_AGGREGATE_RANGE
        assert pess <= low, f"pessimistic {pess} must not exceed the slowest run {low}"
        assert opt >= high, f"optimistic {opt} must not undercut the fastest run {high}"

    def test_a_small_model_is_credited_as_faster(self):
        big = _llm_aggregate_throughput("Qwen3-30B-A3B-Instruct", True, 48)
        small = _llm_aggregate_throughput("Qwen3-4B-Instruct", True, 48)
        assert small[0] > big[0] and small[1] > big[1]

    def test_optimistic_is_never_below_pessimistic(self):
        opt, pess = _llm_aggregate_throughput("Qwen3-30B-A3B-Instruct", True, 48)
        assert opt > pess


class TestCloudStillScalesWithConcurrency:
    """A hosted API answers from a fleet, so concurrency genuinely multiplies
    throughput there — but bounded, or a large MAX_ASYNC_LLM yields an absurd
    estimate."""

    def test_more_workers_means_more_throughput(self):
        few = _llm_aggregate_throughput("gpt-5.4", False, 2)
        many = _llm_aggregate_throughput("gpt-5.4", False, 8)
        assert many[0] > few[0]

    def test_scaling_is_capped(self):
        at_cap = _llm_aggregate_throughput("gpt-5.4", False, _CLOUD_CONCURRENCY_CAP)
        way_over = _llm_aggregate_throughput("gpt-5.4", False, _CLOUD_CONCURRENCY_CAP * 100)
        assert at_cap == way_over

    def test_cloud_beats_one_local_gpu(self):
        assert (
            _llm_aggregate_throughput("gpt-5.4", False, 48)[0]
            > _llm_aggregate_throughput("Qwen3-30B-A3B-Instruct", True, 48)[0]
        )

    def test_zero_workers_does_not_divide_the_estimate_by_nothing(self):
        opt, pess = _llm_aggregate_throughput("gpt-5.4", False, 0)
        assert opt > 0 and pess > 0


class TestTheEstimateLandsNearReality:
    """End-to-end arithmetic on the corpus whose real duration is known: 1267
    chunks, 7.58M LLM tokens (prompt+completion), measured at 115.4 min."""

    MEASURED_MINUTES = 115.4
    LLM_TOKENS = 5_045_194 + 2_534_000
    OVERHEAD_FACTOR = 1.1

    def _band_minutes(self):
        opt, pess = _llm_aggregate_throughput("Qwen3-30B-A3B-Instruct", True, 48)
        return (
            self.LLM_TOKENS / opt * self.OVERHEAD_FACTOR / 60,
            self.LLM_TOKENS / pess * self.OVERHEAD_FACTOR / 60,
        )

    def test_the_real_duration_falls_inside_the_band(self):
        low, high = self._band_minutes()
        assert low <= self.MEASURED_MINUTES <= high, (
            f"measured {self.MEASURED_MINUTES:.0f} min outside {low:.0f}-{high:.0f}"
        )

    def test_the_band_is_not_uselessly_wide(self):
        low, high = self._band_minutes()
        assert high / low < 2.0, "a band wider than 2x tells the user nothing"

    def test_it_is_no_longer_wildly_optimistic(self):
        """The regression this replaces predicted 11-28 min for this corpus."""
        low, _ = self._band_minutes()
        assert low > 60, f"optimistic end {low:.0f} min is back in fantasy territory"
