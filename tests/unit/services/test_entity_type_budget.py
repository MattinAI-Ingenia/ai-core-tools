"""Pins the entity-type inference token budget against the model window.

This constant has broken twice, both times the same way: it is only meaningful
relative to how tokens are counted, and it was sized against a counter that
overestimated Spanish text by ~1.65x. When the counter was made accurate, the
payload silently grew to fill the full budget, squeezed the completion, and the
model was cut off mid-JSON — surfacing to the user as "Could not parse response
content as the length limit was reached".

The failure is invisible until someone clicks the button, and the decision it
blocks (entity types) cannot be changed after indexing. So the arithmetic gets
a test rather than a comment alone.
"""

import pytest

from services.entity_type_inference_service import (
    DOCUMENT_TOKEN_BUDGET,
    MAX_TYPES,
)
from tools.vector_stores.lightrag.entity_type_inference import _count_tokens

# Self-hosted extraction servers in this deployment run with
# max_model_len=20000; the budget must hold for the smallest window in play.
SMALLEST_MODEL_WINDOW = 20_000
# Measured: prompt_tokens 12562 with a 10500-token payload.
MEASURED_INSTRUCTION_OVERHEAD = 2_100
# A truncated run produced 7438 completion tokens and was still incomplete.
OBSERVED_INSUFFICIENT_COMPLETION = 7_438


def test_budget_leaves_more_completion_room_than_a_truncated_run_needed():
    room = SMALLEST_MODEL_WINDOW - DOCUMENT_TOKEN_BUDGET - MEASURED_INSTRUCTION_OVERHEAD
    assert room > OBSERVED_INSUFFICIENT_COMPLETION, (
        f"only {room} tokens left for the completion; a run that generated "
        f"{OBSERVED_INSUFFICIENT_COMPLETION} was already truncated"
    )


def test_budget_has_real_margin_not_just_a_hair():
    """Enough for MAX_TYPES classes with a rationale and examples each, plus
    slack — not merely more than the known-bad figure."""
    room = SMALLEST_MODEL_WINDOW - DOCUMENT_TOKEN_BUDGET - MEASURED_INSTRUCTION_OVERHEAD
    assert room >= 10_000, f"{room} tokens is too tight for {MAX_TYPES} classes"


def test_the_whole_request_fits_the_window():
    total = DOCUMENT_TOKEN_BUDGET + MEASURED_INSTRUCTION_OVERHEAD + 10_000
    assert total <= SMALLEST_MODEL_WINDOW


class TestTheCounterTheBudgetIsSizedAgainst:
    """The budget assumes real token counting. If approx_tokens ever reverts to
    a chars/token heuristic, the budget stops meaning what it says — which is
    exactly how this broke."""

    def test_spanish_text_is_far_from_the_old_2_3_chars_per_token_assumption(self):
        text = (
            "ADVERTENCIAS DE SEGURIDAD. La valvula de seguridad debe instalarse "
            "en la ida del circuito de calefaccion. Cantidad de refrigerante "
            "R290: 150 g. Caracteristicas tecnicas del modulo hidraulico. "
        ) * 20
        chars_per_token = len(text) / _count_tokens(text)
        assert chars_per_token > 3.0, (
            "real Spanish tokenization is ~3.8 chars/token; a budget sized "
            "against 2.3 silently admits ~1.65x more payload than it states"
        )

    def test_empty_text_costs_nothing(self):
        assert _count_tokens("") == 0

    def test_whitespace_is_not_free(self):
        """Not a quirk to paper over: whitespace really is a token, and the
        budget is summed over concatenated samples where it adds up."""
        assert _count_tokens("   ") >= 1
