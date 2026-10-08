"""The multilingual eval derivation: suffix wording and which pairs are used."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "derivar_eval_multiling", Path(__file__).resolve().parents[2] / "scripts" / "derivar_eval_multiling.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_suffix_goes_before_the_closing_question_mark():
    assert mod.add_suffix("¿Qué P02 hay?", "en todos los idiomas") == "¿Qué P02 hay, en todos los idiomas?"


def test_suffix_on_a_statement_replaces_the_final_period():
    assert mod.add_suffix("Lista los manuales de biomasa.", "en los manuales en inglés") == (
        "Lista los manuales de biomasa, en los manuales en inglés."
    )


def _q(id_, text, docs, uc="UC1"):
    return {"id": id_, "caso_uso": uc, "nivel": "facil", "pregunta": text,
            "respuesta_esperada": f"resp {id_}", "docs_esperados": docs}


def test_only_identical_question_pairs_are_used_and_expectations_are_per_language():
    es = {"preguntas": [_q("UC1-1", "¿A?", ["E1"]), _q("UC1-2", "¿B?", ["E2"])]}
    gb = {"preguntas": [
        {**_q("EN-UC1-1", "¿A?", ["G1"]), "origen_ES": "UC1-1"},
        {**_q("EN-UC1-2", "¿B distinta?", ["G2"]), "origen_ES": "UC1-2"},
    ]}

    out = mod.build(es, gb, per_lang=5)

    assert out["n_pares"] == 1
    by_id = {q["id"]: q for q in out["preguntas"]}
    assert by_id["ML-ALL-UC1-1"]["docs_esperados"] == ["E1", "G1"]
    assert by_id["ML-ES-UC1-1"]["docs_esperados"] == ["E1"]
    assert by_id["ML-EN-UC1-1"]["docs_esperados"] == ["G1"]
    assert by_id["ML-EN-UC1-1"]["pregunta"] == "¿A, en los manuales en inglés?"
    # no suffix: nothing restricts the question, so both languages are expected
    assert by_id["ML-NS-UC1-1"]["pregunta"] == "¿A?"
    assert by_id["ML-NS-UC1-1"]["idioma_esperado"] == ["es", "en"]
    assert by_id["ML-NS-UC1-1"]["docs_esperados"] == ["E1", "G1"]


def _runner():
    spec = importlib.util.spec_from_file_location(
        "run_eval_multiling", Path(__file__).resolve().parents[2] / "scripts" / "run_eval_multiling.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_scoring_flags_wrong_silo_and_dangling_cites():
    run = _runner()
    q = {"idioma_esperado": ["es"], "docs_esperados": ["CDOC1", "CDOC2"]}
    graph = {"data": {"chunks": [
        {"id": "a", "file_path": "CDOC1.pdf", "silo_id": 37},
        {"id": "b", "file_path": "CDOC9.pdf", "silo_id": 39},
    ]}}

    s = run.score(q, "x [1](cite://1) y [2](cite://2) z [5](cite://5)", graph, es_silo=37, en_silo=39)

    assert s["bad_cites"] == [5]
    assert s["wrong_silo_cited"] == [39]
    assert s["routing_ok"] is False
    assert s["docs_found"] == ["CDOC1"] and s["doc_recall"] == 0.5


def test_scoring_all_block_needs_both_silos():
    run = _runner()
    q = {"idioma_esperado": ["es", "en"], "docs_esperados": []}
    both = {"data": {"chunks": [{"silo_id": 37, "file_path": "a"}, {"silo_id": 39, "file_path": "b"}]}}
    only_es = {"data": {"chunks": [{"silo_id": 37, "file_path": "a"}]}}

    assert run.score(q, "", both, 37, 39)["routing_ok"] is True
    assert run.score(q, "", only_es, 37, 39)["routing_ok"] is False
    assert run.score(q, "", None, 37, 39)["routing_ok"] is None


def test_parse_sse_done_returns_response_and_graph():
    run = _runner()
    lines = [
        b'data: {"type": "token", "data": {"content": "x"}}\n', b"\n",
        b'data: {"type": "done", "data": {"response": "answer", "lightrag_graph": {"data": {"chunks": [1]}}}}\n',
    ]
    assert run.parse_sse_done(lines) == ("answer", {"data": {"chunks": [1]}})


def test_parse_sse_done_counts_tool_calls_per_sub_agent():
    run = _runner()
    stats = {}
    lines = [
        b'data: {"type": "tool_start", "data": {"tool_name": "consultar_varios"}}\n',
        b'data: {"type": "tool_start", "data": {"tool_name": "retrieve", "subagent_name": "kr_es"}}\n',
        b'data: {"type": "tool_start", "data": {"tool_name": "retrieve", "subagent_name": "kr_es"}}\n',
        b'data: {"type": "done", "data": {"response": "r"}}\n',
    ]
    run.parse_sse_done(lines, stats)
    assert stats["tool_calls"] == {"router:consultar_varios": 1, "kr_es:retrieve": 2}


def test_parse_sse_done_raises_on_error_event_and_on_missing_done():
    import pytest as _pytest

    run = _runner()
    with _pytest.raises(RuntimeError, match="boom"):
        run.parse_sse_done([b'data: {"type": "error", "data": {"message": "boom"}}\n'])
    with _pytest.raises(RuntimeError, match="without a done"):
        run.parse_sse_done([b'data: {"type": "token", "data": {}}\n'])
