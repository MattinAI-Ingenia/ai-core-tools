"""Per-question usage comes from agent_execution_event, matched by answer length."""

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "run_eval_rag.py"
_spec = importlib.util.spec_from_file_location("run_eval_rag", _SCRIPT)
run_eval_rag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval_rag)

# rows: input, output, total, llm_calls, duration_ms, model, response_chars (start order)
ROWS = [[1000, 100, 1100, 3, 5000, "Qwen3.8-27B", 20], [2000, 200, 2200, 4, 9000, "Qwen3.8-27B", 10]]


def _usage(results):
    out = SimpleNamespace(stdout=json.dumps(ROWS))
    with patch.object(run_eval_rag.subprocess, "run", return_value=out):
        return run_eval_rag.fetch_usage_events(
            1, datetime(2026, 11, 10, tzinfo=timezone.utc), results, 1.0, 10.0)


def test_each_result_gets_the_row_with_its_answer_length_even_out_of_order():
    # parallel runs finish in any order: answer lengths 10 then 20 reverse the row order
    u = _usage([{"respuesta_agente": "x" * 10}, {"respuesta_agente": "y" * 20}])
    assert [x["input_tokens"] for x in u] == [2000, 1000]
    assert u[0]["duration_s"] == 9.0
    assert u[0]["cost_usd_upper_bound"] == round(2000 / 1e6 * 1.0 + 200 / 1e6 * 10.0, 6)


def test_a_result_without_a_matching_row_gets_an_empty_dict():
    assert _usage([{"respuesta_agente": "z" * 7}, {"respuesta_agente": None}]) == [{}, {}]
