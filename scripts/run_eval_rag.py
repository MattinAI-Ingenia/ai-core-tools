#!/usr/bin/env python3
"""Run eval_set.json against a live Mattin AI agent and save the answers.

Introspects the Agent's/Silo's LightRAG config (chunk size/strategy, query mode)
straight from the backend container's DB session, so the output filename and
metadata always reflect what was actually indexed/queried — no manual bookkeeping.

Usage
-----
    python scripts/run_eval_rag.py --api-key <key> [--app-id 1] [--agent-id 1]

No key yet? Create one (needs the backend container running)::

    docker exec mattin-backend python -c "
    from db.database import SessionLocal
    from services.api_key_service import APIKeyService
    db = SessionLocal()
    res = APIKeyService().create_api_key(db, app_id=1, user_id=1, name='eval-rag', is_active=True)
    db.commit()
    print(res.key_value)"

Output: benchmark/resultados/resultado_chunk<N>_<STRATEGY>_<query_mode>_<YYYYMMDD>.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_CONTAINER = "mattin-backend"

# Verified 2026-08-24 (OpenRouter/Morph/CloudZero) — standard tier, no long-context
# surcharge, no cache discount. Applied to the FULL per-call token count logged by
# the "monitoring" middleware, which sums every model the agent chain used in that
# call (the OpenAI synthesis model + any local keyword-extraction model) — the
# local model's tokens are free, so this is a deliberate upper bound, not an
# exact figure. Matched against the agent's synthesis-model description
# (see introspect_config) by substring, longest/most-specific key first.
PRICING_PER_1M = {
    "gpt-5.4": (2.50, 15.00),
    "gpt-5": (1.25, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o": (2.50, 10.00),
    # Self-hosted on our own GPUs: no per-token billing. Listed explicitly
    # rather than left to the fallback, which would price a free run at
    # gpt-5.4 rates and report ~$8.7 for something that cost nothing.
    # Longest-key-first matching puts these ahead of the "gpt-5"/"gpt-4o"
    # substrings they do not actually contain anyway.
    "gpt-oss-120b": (0.0, 0.0),
    "gpt-oss-20b": (0.0, 0.0),
    "glm-4.5-air": (0.0, 0.0),
    "qwen3": (0.0, 0.0),
}


def resolve_pricing(model_description: str | None) -> tuple[float, float, str]:
    """(input_per_1m, output_per_1m, matched_key) for the agent's synthesis model.

    Falls back to gpt-5.4's price (this eval's original model) with a clearly
    wrong-looking key when the description doesn't match anything known, so a
    silent misprice is easy to spot in the output JSON instead of invisible.
    """
    desc = (model_description or "").lower()
    for key in sorted(PRICING_PER_1M, key=len, reverse=True):
        if key in desc:
            i, o = PRICING_PER_1M[key]
            return i, o, key
    i, o = PRICING_PER_1M["gpt-5.4"]
    return i, o, f"UNKNOWN_MODEL_FALLBACK_TO_gpt-5.4(raw='{model_description}')"


_MONITORING_LINE_RE = re.compile(
    r"\[Monitoring\] agent_id=(\d+) \| models=(\[.*?\]) \| "
    r"input_tokens=(\d+) \| output_tokens=(\d+) \| total_tokens=(\d+) \| llm_calls=(\d+)"
)
# Matches both retrieve_from_knowledge_base's "route='local'|'global'|'hybrid'|
# 'mix'|'naive'" and list_documents_mentioning's "route='cobertura'" (see
# agentTools.py) — same tag format, one regex covers both tools.
_ROUTE_LINE_RE = re.compile(r"route='([a-z]+)'")


def corpus_vocabulary(eval_set: dict) -> list[str]:
    """Every document code the eval set ever expects, longest first.

    Derived from the data instead of a hard-coded regex like ``CDOC\\d{6}``:
    the corpus's naming scheme is not this script's business, and the union of
    all ``docs_esperados`` covers the whole corpus anyway (30/30 for the DOMUSA
    set). Longest-first so a code that is a prefix of another can't shadow it.
    """
    vocab = set()
    for q in eval_set.get("preguntas", []):
        vocab.update(q.get("docs_esperados") or [])
    return sorted(vocab, key=len, reverse=True)


def score_doc_recall(answer: str | None, expected: list[str], vocab: list[str]) -> dict | None:
    """Which expected documents the answer actually cites.

    Objective counterpart to reading the answers by hand: a human (or an LLM)
    grading "did it cover the right documents?" drifts between passes, and on
    this eval set that drift was worth roughly as much as the effect being
    measured. Counting citations is deterministic and free, so a config change
    can be judged on it without a grading pass at all.

    ``extra`` only counts REAL corpus documents cited but not expected — an
    invented code is a hallucination, a different question this does not try to
    answer. Returns None when the question has no document-level ground truth.
    """
    if not expected:
        return None
    text = answer or ""
    mentioned = {code for code in vocab if code in text}
    expected_set = set(expected)
    found = sorted(mentioned & expected_set)
    return {
        "found": len(found),
        "expected": len(expected_set),
        "recall": round(len(found) / len(expected_set), 4),
        "missing": sorted(expected_set - mentioned),
        "extra": sorted(mentioned - expected_set),
    }


def aggregate_doc_recall(results: list[dict]) -> dict:
    """Corpus-level recall, overall and per ``tipo``.

    Reports macro (mean of per-question recall) and micro (total hits / total
    expected) side by side: macro weights every question equally, micro is
    dominated by the wide fan-out questions. They can diverge a lot here — one
    question expects 25 documents and several expect 1 — and which one moved
    is itself informative.
    """
    scored = [
        (r, r["doc_recall"]) for r in results
        if not r.get("error") and r.get("doc_recall")
    ]
    if not scored:
        return {}

    def _block(rows):
        hits = sum(s["found"] for _, s in rows)
        total = sum(s["expected"] for _, s in rows)
        return {
            "n_preguntas": len(rows),
            "recall_macro": round(sum(s["recall"] for _, s in rows) / len(rows), 4),
            "recall_micro": round(hits / total, 4) if total else None,
            "docs_encontrados": hits,
            "docs_esperados": total,
            "citas_extra": sum(len(s["extra"]) for _, s in rows),
        }

    by_tipo = {}
    for r, s in scored:
        by_tipo.setdefault(r.get("tipo") or "sin_tipo", []).append((r, s))

    return {
        "global": _block(scored),
        "por_tipo": {t: _block(rows) for t, rows in sorted(by_tipo.items())},
        "nota": (
            "Recall de citas de documento, calculado sin intervencion humana ni LLM: "
            "se busca cada codigo del corpus (union de docs_esperados del eval set) "
            "en respuesta_agente. 'citas_extra' son documentos reales del corpus "
            "citados pero no esperados; un codigo inventado no se cuenta aqui. "
            "Solo cubre preguntas con docs_esperados; las de valor puntual no."
        ),
    }


def introspect_config(app_id: int, agent_id: int) -> dict:
    """Pull the agent+silo LightRAG config from the running backend container."""
    snippet = f"""
import json
import os as _os
import config as _cfg
from db.database import SessionLocal
from models.agent import Agent
from models.silo import Silo
from models.ai_service import AIService
db = SessionLocal()
agent = db.query(Agent).filter(Agent.agent_id == {agent_id}, Agent.app_id == {app_id}).one()
silo = db.query(Silo).filter(Silo.silo_id == agent.silo_id).one() if agent.silo_id else None
llm_service = db.query(AIService).filter(AIService.service_id == agent.service_id).one() if agent.service_id else None
print(json.dumps({{
    "agent_name": agent.name,
    "query_mode": agent.lightrag_query_mode,
    "rag_k": agent.rag_k,
    "rag_chunk_top_k": agent.rag_chunk_top_k,
    "chunk_size": (silo.lightrag_chunk_token_size if silo else None),
    "chunk_strategy": (silo.lightrag_chunk_strategy if silo else None),
    "llm_model": (llm_service.description if llm_service else None),
    "has_system_prompt": bool((agent.system_prompt or "").strip()),
    # Read from the backend's own config, not the DB: rerank is deployment-wide
    # config, and whether it was on is the single biggest confounder when
    # comparing two runs.
    "rerank": _cfg.LIGHTRAG_RERANK_URL or None,
    "rerank_model": (_cfg.LIGHTRAG_RERANK_MODEL if _cfg.LIGHTRAG_RERANK_URL else None),
    # The EFFECTIVE chunk cap: LightRAG falls back to its env CHUNK_TOP_K when
    # the agent leaves rag_chunk_top_k unset. Reporting only the DB value once
    # made an A/B report "60" for a run that actually executed at 30.
    "effective_chunk_top_k": (
        agent.rag_chunk_top_k
        if agent.rag_chunk_top_k is not None
        else int(_os.getenv("CHUNK_TOP_K", "30"))
    ),
}}))
"""
    out = subprocess.run(
        ["docker", "exec", BACKEND_CONTAINER, "python", "-c", snippet],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def fetch_monitoring_usage(
    agent_id: int, since: datetime, n_calls: int, input_per_1m: float, output_per_1m: float,
) -> list[dict]:
    """Pull per-call token usage logged by the agent's "monitoring" middleware.

    Requires a Middleware(middleware_type='MONITORING') attached to the agent
    (see UsageMetadataCallbackHandler in tools/agentTools.py) — without it, no
    [Monitoring] lines are logged and this returns an empty list for every call.

    Correlates the Nth [Monitoring] line for this agent to the Nth question,
    which only holds because this script calls the agent strictly sequentially
    (no concurrent requests). The handler logs each call's totals TWICE
    (duplicated log statement in agent_execution_service.py) — consecutive
    duplicates are collapsed here.
    """
    since_str = since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        out = subprocess.run(
            ["docker", "logs", BACKEND_CONTAINER, "--since", since_str],
            capture_output=True, text=True, check=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  (no se pudo leer logs de coste: {exc})", file=sys.stderr)
        return [{} for _ in range(n_calls)]

    events = []
    last_key = None
    for line in out.stdout.splitlines() + out.stderr.splitlines():
        m = _MONITORING_LINE_RE.search(line)
        if not m or int(m.group(1)) != agent_id:
            continue
        key = m.groups()
        if key == last_key:
            continue  # duplicate log statement, same call
        last_key = key
        models = json.loads(m.group(2).replace("'", '"'))
        input_tokens, output_tokens, total_tokens, llm_calls = (int(x) for x in m.groups()[2:])
        events.append({
            "models": models,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "llm_calls": llm_calls,
            "cost_usd_upper_bound": round(
                input_tokens / 1_000_000 * input_per_1m
                + output_tokens / 1_000_000 * output_per_1m,
                6,
            ),
        })

    if len(events) != n_calls:
        print(
            f"  (aviso: {len(events)} eventos de coste para {n_calls} llamadas — "
            "puede haber una llamada de otra fuente en la ventana de tiempo; "
            "se toman los ultimos N eventos, que son siempre los mas recientes)",
            file=sys.stderr,
        )
    # Take the LAST n_calls — `--since` can pick up a stray earlier call to this
    # same agent from outside this run (e.g. a manual smoke test), but never a
    # LATER one, so the tail is always this run's own events in order.
    if len(events) >= n_calls:
        return events[-n_calls:]
    return events + [{}] * (n_calls - len(events))


def fetch_routes_used(agent_id: int, since: datetime, n_calls: int) -> list[list[str]]:
    """Which retrieval route(s) (cobertura/naive/local/global/hybrid/mix) fired
    per question, read from the same backend logs as fetch_monitoring_usage.

    A tool call always logs its route BEFORE the turn's final [Monitoring]
    line (see agentTools.py), so bucketing every route line by the next
    non-duplicate [Monitoring] line reuses that function's own "one event per
    call, strictly sequential" assumption instead of adding a second one.
    """
    since_str = since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        out = subprocess.run(
            ["docker", "logs", BACKEND_CONTAINER, "--since", since_str],
            capture_output=True, text=True, check=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  (no se pudo leer logs de ruta: {exc})", file=sys.stderr)
        return [[] for _ in range(n_calls)]

    events = []
    current_routes: list[str] = []
    last_key = None
    for line in out.stdout.splitlines() + out.stderr.splitlines():
        route_match = _ROUTE_LINE_RE.search(line)
        if route_match:
            current_routes.append(route_match.group(1))
            continue
        m = _MONITORING_LINE_RE.search(line)
        if not m or int(m.group(1)) != agent_id:
            continue
        key = m.groups()
        if key == last_key:
            continue  # duplicate log statement, same call — already flushed
        last_key = key
        events.append(list(dict.fromkeys(current_routes)))  # de-dup, keep order
        current_routes = []

    if len(events) >= n_calls:
        return events[-n_calls:]
    return events + [[] for _ in range(n_calls - len(events))]


def call_agent(base_url: str, app_id: int, agent_id: int, api_key: str, message: str, timeout: int = 120) -> dict:
    url = f"{base_url}/public/v1/app/{app_id}/chat/{agent_id}/call"
    boundary = "----evalragboundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="message"\r\n\r\n'
        f"{message}\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("X-API-KEY", api_key)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_questions(
    questions: list[dict], base_url: str, app_id: int, agent_id: int, api_key: str,
    timeout: int, vocab: list[str],
) -> list[dict]:
    """Call the agent for each question in order, scoring doc recall as it goes.

    Shared by a full run and a --retry-failed run: identical per-question
    behaviour, only the input list differs.
    """
    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']}: {q['pregunta'][:70]}...", file=sys.stderr)
        t0 = time.monotonic()
        try:
            answer = call_agent(base_url, app_id, agent_id, api_key, q["pregunta"], timeout=timeout)
            error = None
        except Exception as exc:  # noqa: BLE001
            answer = None
            error = str(exc)
        response_text = answer["response"] if answer else None
        results.append({
            **q,
            "respuesta_agente": response_text,
            "error": error,
            "latency_s": round(time.monotonic() - t0, 2),
            "doc_recall": score_doc_recall(response_text, q.get("docs_esperados") or [], vocab),
        })
    return results


def retry_failed(args, eval_set: dict, vocab: list[str], override_ids: Optional[list[str]] = None) -> int:
    """Re-run only the questions that errored in an existing result file.

    Overwrites those entries IN PLACE (same file) and recomputes the file's
    aggregate doc_recall/coste_estimado over the merged results — everything
    that did not error the first time is left untouched.
    """
    out_path = Path(args.retry_failed)
    existing = json.loads(out_path.read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in eval_set["preguntas"]}
    if override_ids is not None:
        failed_ids = override_ids
        for i in failed_ids:
            if i not in by_id:
                print(f"Aviso: id '{i}' no existe en el eval set — se omite.", file=sys.stderr)
    else:
        failed_ids = [
            r["id"] for r in existing["resultados"]
            if r.get("error") or not r.get("respuesta_agente")
        ]
    if not failed_ids:
        print("Nada que reintentar: ninguna pregunta tiene error/respuesta vacia.", file=sys.stderr)
        return 0

    to_retry = [by_id[i] for i in failed_ids if i in by_id]
    print(f"Reintentando {len(to_retry)} pregunta(s) con timeout={args.timeout}s: {failed_ids}", file=sys.stderr)
    new_results = run_questions(
        to_retry, args.base_url, args.app_id, args.agent_id, args.api_key,
        args.timeout, vocab,
    )

    config = existing.get("config", {})
    input_per_1m, output_per_1m, _ = resolve_pricing(config.get("llm_model"))
    usage_events = fetch_monitoring_usage(
        args.agent_id, datetime.now(timezone.utc), len(new_results), input_per_1m, output_per_1m,
    )
    route_events = fetch_routes_used(args.agent_id, datetime.now(timezone.utc), len(new_results))
    for r, usage, routes in zip(new_results, usage_events, route_events):
        r["usage"] = usage or None
        r["routes"] = routes

    merged_by_id = {r["id"]: r for r in existing["resultados"]}
    still_failing = []
    for r in new_results:
        merged_by_id[r["id"]] = r
        if r.get("error") or not r.get("respuesta_agente"):
            still_failing.append(r["id"])
    existing_ids = [r["id"] for r in existing["resultados"]]
    extra_ids = dict.fromkeys(r["id"] for r in new_results if r["id"] not in existing_ids)
    merged_results = [merged_by_id[i] for i in existing_ids] + [merged_by_id[i] for i in extra_ids]

    existing["resultados"] = merged_results
    existing["doc_recall"] = aggregate_doc_recall(merged_results)
    total_cost = sum((r.get("usage") or {}).get("cost_usd_upper_bound", 0.0) for r in merged_results)
    total_tokens = sum((r.get("usage") or {}).get("total_tokens", 0) for r in merged_results)
    existing.setdefault("coste_estimado", {})["total_tokens"] = total_tokens
    existing["coste_estimado"]["cost_usd_upper_bound"] = round(total_cost, 4)

    out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado (sobrescrito): {out_path}", file=sys.stderr)
    if still_failing:
        print(f"Siguen fallando {len(still_failing)}: {still_failing}", file=sys.stderr)
    else:
        print("Todas las reintentadas se resolvieron.", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--app-id", type=int, default=1)
    parser.add_argument("--agent-id", type=int, default=1)
    parser.add_argument("--base-url", default="http://localhost")
    parser.add_argument("--eval-set", default=str(REPO_ROOT / "benchmark" / "data" / "eval_set.json"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "benchmark" / "resultados"))
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions (smoke test)")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout per question, in seconds")
    parser.add_argument(
        "--retry-failed", default=None, metavar="RESULTADO_JSON",
        help="Re-run only the questions that errored in this existing result file, "
             "and overwrite them in place (ignores --limit)",
    )
    parser.add_argument(
        "--only-ids", default=None,
        help="Comma-separated question ids to run against an EXISTING result "
             "file, overwriting just those entries (e.g. after a code change "
             "you want to re-measure without paying for a full 97-question run)",
    )
    args = parser.parse_args()

    if args.only_ids:
        eval_set = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
        vocab = corpus_vocabulary(eval_set)
        ids = [i.strip() for i in args.only_ids.split(",") if i.strip()]
        return retry_failed(
            SimpleNamespace(**{**vars(args), "retry_failed": args.retry_failed}),
            eval_set, vocab, override_ids=ids,
        )

    if args.retry_failed:
        eval_set = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
        vocab = corpus_vocabulary(eval_set)
        return retry_failed(args, eval_set, vocab)

    config = introspect_config(args.app_id, args.agent_id)
    chunk_size = config["chunk_size"] or 1200  # LightRAG default when unset
    chunk_strategy = (config["chunk_strategy"] or "DEFAULT").upper()
    query_mode = config["query_mode"] or "default"
    llm_model = config.get("llm_model") or "unknown-model"
    has_system_prompt = bool(config.get("has_system_prompt"))
    input_per_1m, output_per_1m, pricing_key = resolve_pricing(llm_model)
    model_slug = re.sub(r"[^A-Za-z0-9.]+", "-", llm_model)
    prompt_tag = "withprompt" if has_system_prompt else "noprompt"

    eval_set = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    questions = eval_set["preguntas"]
    vocab = corpus_vocabulary(eval_set)
    if args.limit:
        questions = questions[: args.limit]

    run_started_at = datetime.now(timezone.utc)
    results = run_questions(
        questions, args.base_url, args.app_id, args.agent_id, args.api_key,
        args.timeout, vocab,
    )

    usage_events = fetch_monitoring_usage(
        args.agent_id, run_started_at, len(results), input_per_1m, output_per_1m,
    )
    route_events = fetch_routes_used(args.agent_id, run_started_at, len(results))
    total_cost = 0.0
    total_tokens = 0
    for r, usage, routes in zip(results, usage_events, route_events):
        r["usage"] = usage or None
        r["routes"] = routes
        total_cost += (usage or {}).get("cost_usd_upper_bound", 0.0)
        total_tokens += (usage or {}).get("total_tokens", 0)

    today = date.today().isoformat().replace("-", "")
    # The EFFECTIVE value, not the DB one — see introspect_config. A filename
    # that says 60 for a run that executed at 30 is worse than no filename tag.
    chunk_top_k_tag = f"chunktopk{config['effective_chunk_top_k']}"
    rerank_tag = "rerank" if config.get("rerank") else "norerank"
    # A --limit run is a smoke test and MUST NOT be able to land on the filename
    # of a full run: same config + same day produces the same name, so a 2-question
    # sanity check silently replaced a finished 99-question result with a
    # 2-question file. The analysis had already been written up, but the raw data
    # was gone. The tag makes the collision impossible rather than merely unlikely.
    smoke_tag = f"_smoke{args.limit}" if args.limit is not None else ""
    out_name = (
        f"resultado_chunk{chunk_size}_{chunk_strategy}_{query_mode}_"
        f"{chunk_top_k_tag}_{rerank_tag}_{model_slug}_{prompt_tag}_{today}{smoke_tag}.json"
    )
    out_path = Path(args.out_dir) / out_name

    # Belt and braces on top of smoke_tag: never replace a result that covered
    # MORE questions than this one, whatever the filename says. Side-steps
    # rather than aborts — the answers are already paid for (~25 min, ~$8.5),
    # so refusing to write would destroy the very thing this guard protects.
    if out_path.exists():
        try:
            previous = json.loads(out_path.read_text(encoding="utf-8")).get("n_preguntas", 0)
        except Exception:  # noqa: BLE001 — unreadable/corrupt file is not worth protecting
            previous = 0
        if previous > len(results):
            out_path = out_path.with_name(f"{out_path.stem}_parcial{len(results)}.json")
            print(
                f"  (aviso: el destino ya tenia {previous} preguntas y esta corrida solo "
                f"{len(results)}; NO se sobrescribe. Guardando en {out_path.name})",
                file=sys.stderr,
            )

    output = {
        "corpus": eval_set.get("corpus"),
        "fecha_ejecucion": date.today().isoformat(),
        "agent_id": args.agent_id,
        "agent_name": config["agent_name"],
        "app_id": args.app_id,
        "config": {
            "chunk_size": chunk_size,
            "chunk_strategy": chunk_strategy,
            "query_mode": query_mode,
            "rag_k": config["rag_k"],
            "rag_chunk_top_k": config["rag_chunk_top_k"],
            "effective_chunk_top_k": config["effective_chunk_top_k"],
            "rerank": config.get("rerank"),
            "rerank_model": config.get("rerank_model"),
            "llm_model": llm_model,
            "has_system_prompt": has_system_prompt,
        },
        "n_preguntas": len(results),
        "doc_recall": aggregate_doc_recall(results),
        "coste_estimado": {
            "total_tokens": total_tokens,
            "cost_usd_upper_bound": round(total_cost, 4),
            "pricing_used": {"model_matched": pricing_key, "input_per_1m": input_per_1m, "output_per_1m": output_per_1m},
            "nota": (
                "Upper bound: suma tokens de TODOS los modelos usados en cada "
                "llamada (incluye el modelo local de keywords, que es gratis) y "
                "aplica el precio del modelo de sintesis del agente (ver "
                "pricing_used) al total. Requiere el middleware 'monitoring' "
                "en el agente; si no esta activo, sale en 0."
            ),
        },
        "resultados": results,
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
