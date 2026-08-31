"""Pins entity-type inference against the model's context window.

This request has been truncated three times, and each time the arithmetic was
wrong in a different place:

1. DOCUMENT_TOKEN_BUDGET=14000 — no real token counting at all.
2. 10500 — sized while approx_tokens used len/2.3, which overestimated Spanish
   by ~1.65x, so fit_budget silently stopped at ~6.4k REAL tokens. Making the
   counter accurate removed that accidental margin, the payload grew to the full
   10500, and the completion was squeezed into 7.4k of a 20k window.
3. The completion itself was UNBOUNDED: the JSON schema capped the number of
   classes but not the examples array inside each. Guided decoding forces valid
   JSON, not short JSON, so the model emitted 11,673 tokens of examples until it
   ran out of context. No payload budget could have been small enough.

So the invariant is not "leave more room than last time" — it is that the
schema-bounded worst-case response plus the prompt fits the window. That worst
case is derived from the schema here, so removing a bound fails this test.
"""

import pytest

from services.entity_type_inference_service import (
    DOCUMENT_TOKEN_BUDGET,
    MAX_TYPES,
)
from tools.vector_stores.lightrag.entity_type_inference import (
    ENTITY_TYPES_JSON_SCHEMA,
    _count_tokens,
)

# Self-hosted extraction servers in this deployment run with
# max_model_len=20000; the budget must hold for the smallest window in play.
SMALLEST_MODEL_WINDOW = 20_000
# Measured: prompt_tokens 12562 with a 10500-token payload.
MEASURED_INSTRUCTION_OVERHEAD = 2_100
# Real Spanish tokenization, measured on this corpus.
CHARS_PER_TOKEN = 3.8
# Braces, quotes and field names per class in the rendered JSON.
JSON_OVERHEAD_PER_CLASS = 25


def _schema_bounds():
    types = ENTITY_TYPES_JSON_SCHEMA["properties"]["types"]
    item = types["items"]["properties"]
    return types, item


def worst_case_completion_tokens() -> int:
    """Largest response the schema can possibly permit."""
    types, item = _schema_bounds()
    examples = item["examples"]
    per_class = (
        item["name"]["maxLength"]
        + item["why"]["maxLength"]
        + examples["maxItems"] * examples["items"]["maxLength"]
    ) / CHARS_PER_TOKEN + JSON_OVERHEAD_PER_CLASS
    return int(types["maxItems"] * per_class)


class TestTheSchemaBoundsTheResponse:
    """Every dimension the model can grow must have a ceiling. An unbounded one
    is not a style issue: it is how the request got truncated."""

    def test_the_number_of_classes_is_bounded(self):
        types, _ = _schema_bounds()
        assert types.get("maxItems"), "types must cap how many classes come back"

    def test_the_examples_array_is_bounded(self):
        _, item = _schema_bounds()
        assert item["examples"].get("maxItems"), (
            "an unbounded examples array is what produced 11,673 completion "
            "tokens — guided decoding forces valid JSON, not short JSON"
        )

    @pytest.mark.parametrize("field", ["name", "why"])
    def test_free_text_fields_are_bounded(self, field):
        _, item = _schema_bounds()
        assert item[field].get("maxLength"), f"{field} can grow without a maxLength"

    def test_example_strings_are_bounded(self):
        _, item = _schema_bounds()
        assert item["examples"]["items"].get("maxLength")

    def test_examples_still_require_at_least_two(self):
        """Load-bearing the other way: a class with no instances is exactly the
        signal that the class should not exist."""
        _, item = _schema_bounds()
        assert item["examples"]["minItems"] >= 2


class TestTheWholeRequestFitsTheWindow:
    def test_prompt_plus_worst_case_response_fits(self):
        total = (
            DOCUMENT_TOKEN_BUDGET
            + MEASURED_INSTRUCTION_OVERHEAD
            + worst_case_completion_tokens()
        )
        assert total <= SMALLEST_MODEL_WINDOW, (
            f"{total} tokens against a {SMALLEST_MODEL_WINDOW} window"
        )

    def test_there_is_real_margin_not_a_hair(self):
        total = (
            DOCUMENT_TOKEN_BUDGET
            + MEASURED_INSTRUCTION_OVERHEAD
            + worst_case_completion_tokens()
        )
        assert SMALLEST_MODEL_WINDOW - total >= 3_000, "too tight to absorb drift"

    def test_the_consolidated_answer_is_smaller_than_the_first_pass(self):
        """The second pass merges down to MAX_TYPES, so it can only be smaller —
        if MAX_TYPES ever exceeded the schema cap the prompt would be asking for
        more than the schema allows."""
        types, _ = _schema_bounds()
        assert MAX_TYPES <= types["maxItems"]


class TestTheCounterTheBudgetIsSizedAgainst:
    """The payload budget assumes real token counting. Reverting to a
    chars/token heuristic makes the number stop meaning what it says."""

    def test_spanish_text_is_far_from_the_old_2_3_chars_per_token_assumption(self):
        text = (
            "ADVERTENCIAS DE SEGURIDAD. La valvula de seguridad debe instalarse "
            "en la ida del circuito de calefaccion. Cantidad de refrigerante "
            "R290: 150 g. Caracteristicas tecnicas del modulo hidraulico. "
        ) * 20
        assert len(text) / _count_tokens(text) > 3.0

    def test_empty_text_costs_nothing(self):
        assert _count_tokens("") == 0

    def test_whitespace_is_not_free(self):
        assert _count_tokens("   ") >= 1
