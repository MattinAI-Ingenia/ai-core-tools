"""Pins the eval's automatic document-recall scoring.

This metric exists because label grading (CORRECT/PARTIAL/WRONG) drifted between
passes by roughly as much as the effect being measured: two runs of an identical
config differed by 9 points on `cobertura`, and a reranker A/B could not be
called either way from labels alone. Counting which expected documents the
answer actually cites is deterministic, so it has to stay deterministic — hence
these tests.
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "run_eval_rag.py"
_spec = importlib.util.spec_from_file_location("run_eval_rag", _SCRIPT)
run_eval_rag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval_rag)

corpus_vocabulary = run_eval_rag.corpus_vocabulary
score_doc_recall = run_eval_rag.score_doc_recall
aggregate_doc_recall = run_eval_rag.aggregate_doc_recall

VOCAB = ["CDOC004273", "CDOC003912", "CDOC001018", "DSAT000120"]


def test_vocabulary_is_the_union_of_every_expected_doc():
    eval_set = {"preguntas": [
        {"docs_esperados": ["CDOC000078", "CDOC001009"]},
        {"docs_esperados": ["CDOC001009"]},
        {"docs_esperados": None},
        {},
    ]}
    assert sorted(corpus_vocabulary(eval_set)) == ["CDOC000078", "CDOC001009"]


def test_counts_only_the_expected_docs_the_answer_cites():
    score = score_doc_recall(
        "Aparece en CDOC004273 y en CDOC001018.",
        ["CDOC004273", "CDOC001018", "DSAT000120"],
        VOCAB,
    )
    assert score["found"] == 2
    assert score["expected"] == 3
    assert score["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert score["missing"] == ["DSAT000120"]
    assert score["extra"] == []


def test_real_corpus_docs_cited_but_not_expected_are_extra():
    score = score_doc_recall(
        "Aparece en CDOC004273 y tambien en CDOC003912.",
        ["CDOC004273"],
        VOCAB,
    )
    assert score["recall"] == 1.0
    assert score["extra"] == ["CDOC003912"], "a full-recall answer can still over-cite"


def test_an_invented_code_is_not_counted_as_extra():
    """Only real corpus documents count. A made-up code is a hallucination —
    a different failure, deliberately out of scope here rather than silently
    folded into the precision figure."""
    score = score_doc_recall("Ver CDOC004273 y CDOC999999.", ["CDOC004273"], VOCAB)
    assert score["extra"] == []


def test_questions_without_document_ground_truth_are_skipped():
    assert score_doc_recall("cualquier cosa", [], VOCAB) is None


def test_no_answer_scores_zero_rather_than_crashing():
    score = score_doc_recall(None, ["CDOC004273"], VOCAB)
    assert score["found"] == 0 and score["recall"] == 0.0


def test_aggregate_separates_macro_from_micro_and_ignores_errored_rows():
    """Macro weights every question equally; micro is dominated by the wide
    fan-out ones. On this eval set one question expects 25 docs and others
    expect 1, so the two genuinely diverge and both are reported."""
    results = [
        {"tipo": "cobertura", "doc_recall": {"found": 1, "expected": 10, "recall": 0.1, "extra": []}},
        {"tipo": "cobertura", "doc_recall": {"found": 1, "expected": 1, "recall": 1.0, "extra": ["X"]}},
        {"tipo": "metadato", "doc_recall": {"found": 2, "expected": 2, "recall": 1.0, "extra": []}},
        {"tipo": "cobertura", "error": "boom", "doc_recall": {"found": 9, "expected": 9, "recall": 1.0, "extra": []}},
        {"tipo": "valor_puntual", "doc_recall": None},
    ]
    agg = aggregate_doc_recall(results)

    assert agg["global"]["n_preguntas"] == 3, "errored and unscorable rows excluded"
    assert agg["global"]["docs_encontrados"] == 4
    assert agg["global"]["docs_esperados"] == 13
    assert agg["global"]["recall_micro"] == pytest.approx(4 / 13, abs=1e-4)
    assert agg["global"]["recall_macro"] == pytest.approx((0.1 + 1.0 + 1.0) / 3, abs=1e-4)
    assert agg["global"]["citas_extra"] == 1
    assert agg["por_tipo"]["cobertura"]["n_preguntas"] == 2
    assert "valor_puntual" not in agg["por_tipo"]


def test_empty_input_returns_an_empty_block_not_a_crash():
    assert aggregate_doc_recall([{"doc_recall": None}]) == {}
