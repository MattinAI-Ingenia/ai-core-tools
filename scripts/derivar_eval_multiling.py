#!/usr/bin/env python3
"""Derive the multilingual eval set from the ES and GB sets that already exist.

Both sets are written in Spanish and pair up through ``origen_ES``. Only the pairs
whose question text is IDENTICAL in both (same question, two corpora) are used.
Each pair yields up to three questions, with a suffix that says which documents
the question is about:

  ALL  "..., en todos los idiomas"            -> both silos, expected = union
  ES   "..., en los manuales en español"      -> ES silo only
  EN   "..., en los manuales en inglés"       -> EN silo only
  NS   the question as is (no suffix)         -> both silos: nothing restricts it

ALL covers every pair; ES, EN and NS take the first --per-lang pairs per use case
(round-robin) to keep the run short.

    python scripts/derivar_eval_multiling.py [--root <repo with benchmark/>] [--per-lang 20]
"""
from __future__ import annotations

import argparse
import json
from itertools import zip_longest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SUFFIX = {
    "ALL": "en todos los idiomas",
    "ES": "en los manuales en español",
    "EN": "en los manuales en inglés",
}
LANG_LABEL = {"ES": "español", "EN": "inglés"}


def add_suffix(question: str, suffix: str) -> str:
    """Put the suffix before the last '?' (or at the end) as a comma-separated tail."""
    q = question.rstrip()
    if q.endswith("?"):
        return f"{q[:-1].rstrip()}, {suffix}?"
    return f"{q.rstrip('.')}, {suffix}."


def pick_round_robin(pairs: list[dict], n: int) -> list[dict]:
    by_uc: dict[str, list[dict]] = {}
    for p in pairs:
        by_uc.setdefault(p["caso_uso"], []).append(p)
    ordered = [p for group in zip_longest(*by_uc.values()) for p in group if p]
    return ordered[:n]


def build(es_set: dict, gb_set: dict, per_lang: int) -> dict:
    es_by_id = {q["id"]: q for q in es_set["preguntas"]}
    pairs = []
    for g in gb_set["preguntas"]:
        e = es_by_id.get(g["origen_ES"])
        if e and e["pregunta"] == g["pregunta"]:
            pairs.append({"es": e, "en": g, "caso_uso": e["caso_uso"], "nivel": e.get("nivel")})

    def item(kind: str, p: dict) -> dict:
        e, g = p["es"], p["en"]
        base = {"caso_uso": p["caso_uso"], "nivel": p["nivel"], "origen_ES": e["id"], "origen_EN": g["id"]}
        q = e["pregunta"] if kind == "NS" else add_suffix(e["pregunta"], SUFFIX[kind])
        if kind == "ES":
            return {"id": f"ML-ES-{e['id']}", **base, "idioma_esperado": ["es"], "pregunta": q,
                    "respuesta_esperada": e["respuesta_esperada"], "docs_esperados": e["docs_esperados"]}
        if kind == "EN":
            return {"id": f"ML-EN-{e['id']}", **base, "idioma_esperado": ["en"], "pregunta": q,
                    "respuesta_esperada": g["respuesta_esperada"], "docs_esperados": g["docs_esperados"]}
        return {
            "id": f"ML-{kind}-{e['id']}", **base, "idioma_esperado": ["es", "en"], "pregunta": q,
            "respuesta_esperada": (
                f"En los manuales en español: {e['respuesta_esperada']}\n"
                f"En los manuales en inglés: {g['respuesta_esperada']}"
            ),
            "docs_esperados": sorted(set(e["docs_esperados"]) | set(g["docs_esperados"])),
            "docs_esperados_es": e["docs_esperados"], "docs_esperados_en": g["docs_esperados"],
        }

    questions = [item("ALL", p) for p in pairs]
    questions += [item("ES", p) for p in pick_round_robin(pairs, per_lang)]
    questions += [item("EN", p) for p in pick_round_robin(pairs, per_lang)]
    questions += [item("NS", p) for p in pick_round_robin(pairs, per_lang)]
    return {
        "corpus": "DOMUSA: silo ES + silo EN (router por idioma)",
        "origen": "eval_set_domusa.json + eval_set_gb_261001_v2.json (pares con la misma pregunta)",
        "idioma_preguntas": "es",
        "n_pares": len(pairs),
        "n_preguntas": len(questions),
        "sufijos": SUFFIX,
        "preguntas": questions,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO_ROOT), help="repo root that holds benchmark/")
    ap.add_argument("--per-lang", type=int, default=20)
    args = ap.parse_args()
    root = Path(args.root)
    es = json.loads((root / "benchmark/preguntas/es/eval_set_domusa.json").read_text(encoding="utf-8"))
    gb = json.loads((root / "benchmark/preguntas/gb/eval_set_gb_261001_v2.json").read_text(encoding="utf-8"))
    out = build(es, gb, args.per_lang)
    path = root / "benchmark/preguntas/multiling/eval_set_multiling_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{path}: {out['n_preguntas']} preguntas ({out['n_pares']} pares)")


if __name__ == "__main__":
    main()
