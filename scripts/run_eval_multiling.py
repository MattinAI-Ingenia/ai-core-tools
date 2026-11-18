#!/usr/bin/env python3
"""Run the multilingual eval set against one agent and score routing + citations.

For each question: call the agent through the public streaming API and take the
answer and its LightRAG graph from the final ``done`` event (no conversation
memory needed), then compute, with no LLM involved:

  silos_retrieved / silos_cited  silo ids of the retrieved / cited chunks
                                 (router payloads carry silo_id on every chunk)
  routing_ok                     ES/EN/IT/FR -> only that language's silo,
                                 ALL/NS -> every silo retrieved (router agents only)
  bad_cites                      [N] with no chunk N in the payload
  doc_recall                     expected docs found in the cited chunks' file_path
                                 or in the answer text

    python scripts/run_eval_multiling.py --api-key K --agent-id 6 [--only ALL,ES,EN]
        [--limit N] [--workers 4] [--es-silo 37] [--en-silo 39]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CITE_RE = re.compile(r"cite://(\d+)")


def parse_sse_done(lines, stats: dict | None = None) -> tuple[str, dict | None]:
    """(response, lightrag_graph) from the ``done`` event of an SSE stream.

    *stats*, when given, is filled with the tool calls seen on the way: ``tool_calls``
    maps "<sub-agent or router>:<tool>" to how many times it ran."""
    for line in lines:
        line = line.decode("utf-8").strip() if isinstance(line, bytes) else line.strip()
        if not line.startswith("data: "):
            continue
        event = json.loads(line[6:])
        if stats is not None and event.get("type") == "tool_start":
            who = event["data"].get("subagent_name") or "router"
            key = f"{who}:{event['data'].get('tool_name')}"
            stats.setdefault("tool_calls", {})
            stats["tool_calls"][key] = stats["tool_calls"].get(key, 0) + 1
        if event.get("type") == "error":
            raise RuntimeError(event.get("data", {}).get("message", "agent error"))
        if event.get("type") == "done":
            data = event["data"]
            return str(data.get("response")), data.get("lightrag_graph")
    raise RuntimeError("stream ended without a done event")


def ask(base: str, app_id: int, agent_id: int, api_key: str, message: str, timeout: int, stats: dict | None = None) -> tuple[str, dict | None]:
    boundary = "----evalmultilingboundary"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"message\"\r\n\r\n{message}\r\n--{boundary}--\r\n").encode()
    req = urllib.request.Request(f"{base}/public/v1/app/{app_id}/chat/{agent_id}/call/stream", data=body, method="POST")
    req.add_header("X-API-KEY", api_key)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return parse_sse_done(resp, stats)


def score(q: dict, text: str, graph: dict | None, silos: dict) -> dict:
    chunks = ((graph or {}).get("data") or {}).get("chunks") or []
    cites = sorted({int(n) for n in CITE_RE.findall(text)})
    bad = [n for n in cites if not 1 <= n <= len(chunks)]
    cited = [chunks[n - 1] for n in cites if 1 <= n <= len(chunks)]
    retrieved_silos = sorted({c.get("silo_id") for c in chunks if c.get("silo_id") is not None})
    cited_silos = sorted({c.get("silo_id") for c in cited if c.get("silo_id") is not None})
    want = q["idioma_esperado"]
    wanted = {silos[lang] for lang in want}  # one silo per expected language
    tagged = bool(retrieved_silos)
    if not tagged:
        routing_ok = None  # not a router payload
    elif len(want) > 1:
        routing_ok = set(retrieved_silos) == wanted
    else:
        routing_ok = set(retrieved_silos) <= wanted and bool(retrieved_silos)
    haystack = text + " " + " ".join(str(c.get("file_path", "")) for c in cited)
    expected = q.get("docs_esperados") or []
    found = [d for d in expected if d in haystack]
    return {
        "cites": cites, "bad_cites": bad, "n_chunks": len(chunks),
        "silos_retrieved": retrieved_silos, "silos_cited": cited_silos, "routing_ok": routing_ok,
        "doc_recall": round(len(found) / len(expected), 3) if expected else None,
        "docs_found": found,
        "wrong_silo_cited": [s for s in cited_silos if tagged and len(want) == 1 and s not in wanted],
    }


_claimed: set = set()  # router executions already attached to a question (parallel workers)
_claim_lock = threading.Lock()


def _psql(sql: str) -> list[list[str]]:
    """Rows of a query against the stack's Postgres (docker exec), or [] if it cannot be reached."""
    try:
        out = subprocess.run(
            ["docker", "exec", "docker-postgres-1", "sh", "-c",
             'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -At -F "|" -c "$0"', sql],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception:  # noqa: BLE001 - usage is best effort, never fails the question
        return []
    return [line.split("|") for line in out.splitlines() if line]


def fetch_usage(agent_id: int, started_at: str) -> dict | None:
    """Tokens and time of this question from agent_execution_event: the router execution that
    started closest to *started_at* (UTC) plus its specialists (parent_execution_id). The recorder
    writes in the background, so retry a few times. Tokens are None when the model did not report."""
    ts = datetime.fromisoformat(started_at).strftime("%Y-%m-%d %H:%M:%S.%f")
    for _ in range(4):
        time.sleep(2)
        rows = _psql(
            "select event_id, status, duration_ms, input_tokens, output_tokens, total_tokens, llm_calls "
            f"from agent_execution_event where agent_id={int(agent_id)} "
            f"and started_at between timestamp '{ts}' - interval '3 seconds' and timestamp '{ts}' + interval '20 seconds' "
            f"order by abs(extract(epoch from started_at - timestamp '{ts}'))"
        )
        with _claim_lock:
            row = next((r for r in rows if r[0] not in _claimed), None)
            if row:
                _claimed.add(row[0])
        if row:
            break
    else:
        return None
    num = lambda v: int(v) if v not in ("", None) else None  # noqa: E731
    kids = _psql(
        "select a.name, e.status, e.duration_ms, e.input_tokens, e.output_tokens, e.total_tokens, e.llm_calls "
        f"from agent_execution_event e join \"Agent\" a on a.agent_id=e.agent_id where e.parent_execution_id='{row[0]}'"
    )
    specialists = [{"agent": k[0], "status": k[1], "duration_s": round(int(k[2]) / 1000, 1), "input_tokens": num(k[3]),
                    "output_tokens": num(k[4]), "total_tokens": num(k[5]), "llm_calls": num(k[6])} for k in kids]
    parts = [{"input_tokens": num(row[3]), "output_tokens": num(row[4]), "total_tokens": num(row[5])}, *specialists]
    total = {k: sum(p[k] or 0 for p in parts) for k in ("input_tokens", "output_tokens", "total_tokens")}
    return {"router_duration_s": round(int(row[2]) / 1000, 1), "router": {"input_tokens": num(row[3]), "output_tokens": num(row[4]),
            "llm_calls": num(row[6])}, "specialists": specialists, **total}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--agent-id", type=int, required=True)
    ap.add_argument("--app-id", type=int, default=1)
    ap.add_argument("--base-url", default="http://localhost")
    ap.add_argument("--eval-set", default=str(REPO_ROOT / "benchmark/preguntas/multiling/eval_set_multiling_v2.json"))
    ap.add_argument("--only", default="ALL,ES,EN,IT,FR,NS", help="comma-separated blocks to run")
    ap.add_argument("--limit", type=int, default=None, help="max questions per block")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--es-silo", type=int, default=37)
    ap.add_argument("--en-silo", type=int, default=39)
    ap.add_argument("--fr-silo", type=int, default=66)
    ap.add_argument("--it-silo", type=int, default=67)
    ap.add_argument("--label", default="")
    ap.add_argument("--ids", default=None, help="comma-separated question ids to run (ignores --only/--limit)")
    ap.add_argument("--retry-errors-from", default=None, help="only re-run the questions that errored in this result file")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "benchmark/resultados/multiling"))
    args = ap.parse_args()

    questions = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))["preguntas"]
    blocks = [b.strip() for b in args.only.split(",")]
    selected = []
    for b in blocks:
        group = [q for q in questions if q["id"].startswith(f"ML-{b}-")]
        selected += group[: args.limit] if args.limit else group
    if args.ids:
        wanted = [i.strip() for i in args.ids.split(",") if i.strip()]
        by_id = {q["id"]: q for q in questions}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            raise SystemExit(f"unknown question ids: {missing}")
        selected, blocks = [by_id[i] for i in wanted], sorted({i.split("-")[1] for i in wanted})
    if args.retry_errors_from:
        failed = {r["id"] for r in json.loads(Path(args.retry_errors_from).read_text(encoding="utf-8"))["results"] if r.get("error")}
        selected = [q for q in selected if q["id"] in failed]

    def run(q: dict) -> dict:
        stats: dict = {}
        t0 = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")  # joins with agent_execution_event.started_at
        try:
            text, graph = ask(args.base_url, args.app_id, args.agent_id, args.api_key, q["pregunta"], args.timeout, stats)
            return {**q, "started_at": started_at, "usage": fetch_usage(args.agent_id, started_at), "respuesta_agente": text, "error": None, **score(q, text, graph, {"es": args.es_silo, "en": args.en_silo, "fr": args.fr_silo, "it": args.it_silo}),
                    "latency_s": round(time.monotonic() - t0, 1), "tool_calls": stats.get("tool_calls", {}),
                    "graph_chunks": [{k: c.get(k) for k in ("id", "file_path", "silo_id", "silo_name", "resource_id", "page")}
                                     for c in ((graph or {}).get("data") or {}).get("chunks") or []]}
        except Exception as exc:  # noqa: BLE001
            return {**q, "started_at": started_at, "usage": fetch_usage(args.agent_id, started_at), "respuesta_agente": None, "error": str(exc), "latency_s": round(time.monotonic() - t0, 1),
                    "tool_calls": stats.get("tool_calls", {})}

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"multiling_agent{args.agent_id}{('_' + args.label) if args.label else ''}_{date.today():%Y%m%d}.json"
    partial = path.with_suffix(".jsonl")  # one line per finished question: a crash loses nothing
    by_id: dict = {}
    if partial.exists():  # resume: keep what already finished without error
        for line in partial.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if not r["error"]:
                by_id[r["id"]] = r
    todo = [q for q in selected if q["id"] not in by_id]
    print(f"{len(by_id)} already done, {len(todo)} to run ({partial})", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool, partial.open("a", encoding="utf-8") as fh:
        for fut in as_completed([pool.submit(run, q) for q in todo]):
            r = fut.result()
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            by_id[r["id"]] = r
            print(f"[{len(by_id)}/{len(selected)}] {r['id']} {'ERROR' if r['error'] else 'ok'} {r['latency_s']}s", flush=True)
    results = [by_id[q["id"]] for q in selected]

    summary = {}
    for b in blocks:
        rs = [r for r in results if r["id"].startswith(f"ML-{b}-")]
        ok = [r for r in rs if not r["error"]]
        rec = [r["doc_recall"] for r in ok if r["doc_recall"] is not None]
        summary[b] = {
            "n": len(rs), "errors": len(rs) - len(ok),
            "routing_ok": sum(1 for r in ok if r["routing_ok"] is True),
            "routing_checked": sum(1 for r in ok if r["routing_ok"] is not None),
            "with_cites": sum(1 for r in ok if r["cites"]),
            "bad_cites_total": sum(len(r["bad_cites"]) for r in ok),
            "wrong_silo_cited_total": sum(len(r["wrong_silo_cited"]) for r in ok),
            "doc_recall_mean": round(sum(rec) / len(rec), 3) if rec else None,
        }
    path.write_text(json.dumps({"agent_id": args.agent_id, "summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(path)


if __name__ == "__main__":
    main()
