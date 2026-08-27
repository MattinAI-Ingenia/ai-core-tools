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


def introspect_config(app_id: int, agent_id: int) -> dict:
    """Pull the agent+silo LightRAG config from the running backend container."""
    snippet = f"""
import json
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


def call_agent(base_url: str, app_id: int, agent_id: int, api_key: str, message: str) -> dict:
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
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--app-id", type=int, default=1)
    parser.add_argument("--agent-id", type=int, default=1)
    parser.add_argument("--base-url", default="http://localhost")
    parser.add_argument("--eval-set", default=str(REPO_ROOT / "benchmark" / "data" / "eval_set.json"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "benchmark" / "resultados"))
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions (smoke test)")
    args = parser.parse_args()

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
    if args.limit:
        questions = questions[: args.limit]

    run_started_at = datetime.now(timezone.utc)
    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']}: {q['pregunta'][:70]}...", file=sys.stderr)
        t0 = time.monotonic()
        try:
            answer = call_agent(args.base_url, args.app_id, args.agent_id, args.api_key, q["pregunta"])
            error = None
        except Exception as exc:  # noqa: BLE001
            answer = None
            error = str(exc)
        results.append({
            **q,
            "respuesta_agente": answer["response"] if answer else None,
            "error": error,
            "latency_s": round(time.monotonic() - t0, 2),
        })

    usage_events = fetch_monitoring_usage(
        args.agent_id, run_started_at, len(results), input_per_1m, output_per_1m,
    )
    total_cost = 0.0
    total_tokens = 0
    for r, usage in zip(results, usage_events):
        r["usage"] = usage or None
        total_cost += (usage or {}).get("cost_usd_upper_bound", 0.0)
        total_tokens += (usage or {}).get("total_tokens", 0)

    today = date.today().isoformat().replace("-", "")
    out_name = (
        f"resultado_chunk{chunk_size}_{chunk_strategy}_{query_mode}_"
        f"{model_slug}_{prompt_tag}_{today}.json"
    )
    out_path = Path(args.out_dir) / out_name

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
            "llm_model": llm_model,
            "has_system_prompt": has_system_prompt,
        },
        "n_preguntas": len(results),
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
