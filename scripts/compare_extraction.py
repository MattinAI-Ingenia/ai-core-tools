#!/usr/bin/env python3
"""Compara extracción de entidades/relaciones entre modelos (open source vs cloud).

Usa los prompts REALES de LightRAG (`lightrag.prompt.PROMPTS`), así que hay que
ejecutarlo donde `lightrag` esté instalado — la imagen del backend de la rama
lightrag, no la de develop:

    docker compose exec -T backend python /app/scripts/compare_extraction.py \
        /app/docs/documento.pdf \
        -m openai:gpt-4.1-mini -m anthropic:claude-sonnet-5 -m ollama:qwen3:8b

Claves de API: las del entorno del backend (OPENAI_API_KEY, etc.).
Modelos locales:
    Ollama → OLLAMA_BASE_URL=http://host.docker.internal:11434
    vLLM   → -m openai:Qwen/Qwen3-8B@http://vllm:8000/v1  (clave: VLLM_API_KEY, default "EMPTY")

Salida en --out: summary.csv (una fila por modelo×chunk) y raw.jsonl (entidades y
relaciones completas, para comparar calidad a ojo o calcular solapamiento).

Si un modelo open-source no respeta el límite de filas del prompt (columna
`truncated=True` en el CSV) prueba --json: activa el modo JSON de LightRAG
(ENTITY_EXTRACTION_USE_JSON, no activado por defecto en este repo), que sus
propios docs describen como mejora de compatibilidad con modelos pequeños.

También compara el rol `keyword` (LIGHTRAG_KEYWORD_MODEL — extrae
high/low-level keywords de una consulta de usuario, no del documento):

    echo "¿Cuál es la presión máxima del circuito primario?" > queries.txt
    python scripts/compare_extraction.py queries.txt --role keyword \
        -m openai:gpt-4.1-mini -m ollama:qwen3:8b

    python scripts/compare_extraction.py --self-check   # no necesita lightrag ni red
"""

import argparse
import asyncio
import csv
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")


# ponytail: troceo por caracteres, no por tokens como LightRAG. Para comparar
# modelos solo importa que el trozo sea idéntico para todos.
def chunk_text(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def fill(template: str, values: dict, name: str) -> str:
    # No usa str.format(): las plantillas JSON (--json) tienen `{` literales
    # (p.ej. `{"entities": [`) que .format() confundiría con placeholders.
    fields = set(re.findall(r"{(\w+)}", template))
    missing = fields - values.keys()
    if missing:
        sys.exit(
            f"{name}: placeholders desconocidos {sorted(missing)}. "
            "Esta versión de LightRAG cambió el prompt; añádelos a `values`."
        )
    return re.sub(r"{(\w+)}", lambda m: str(values[m.group(1)]), template)


def load_prompts() -> dict:
    """PROMPTS de LightRAG. Fuera del contenedor, apunta LIGHTRAG_PROMPT_PY al
    prompt.py de la versión que use el backend (el wheel se baja con
    `pip download lightrag-hku==X --no-deps`); solo necesita PyYAML."""
    path = os.getenv("LIGHTRAG_PROMPT_PY")
    if not path:
        from lightrag.prompt import PROMPTS

        return PROMPTS
    spec = importlib.util.spec_from_file_location("lightrag_prompt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PROMPTS


def build_prompt_values(PROMPTS: dict, language: str, entity_types: str | None, json_mode: bool) -> dict:
    guidance = entity_types or PROMPTS.get("default_entity_types_guidance", "")
    values = {
        "language": language,
        "entity_types_guidance": guidance,
        "entity_types": guidance,
        "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        "record_delimiter": PROMPTS.get("DEFAULT_RECORD_DELIMITER") or "\n",
        "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        # Defaults de LightRAG 1.5.6 (constants.py): tope de filas por respuesta.
        "max_total_records": 100,
        "max_entity_records": 40,
        "heading_context_block": "",  # sin heading_path, como al indexar texto plano
        "input_text": "",
    }
    examples_key = "entity_extraction_json_examples" if json_mode else "entity_extraction_examples"
    examples = PROMPTS.get(examples_key, "")
    values["examples"] = fill(
        "\n".join(examples) if isinstance(examples, list) else examples, values, "examples"
    )
    return values


def parse_records(text: str, tuple_delim: str, record_delim: str) -> tuple[list, list]:
    """Parsea la salida delimitada de LightRAG 1.5.6: un registro por línea/record_delim,
    sin paréntesis ni comillas (ese formato es de versiones anteriores de LightRAG):
    `entity<TD>name<TD>type<TD>description`
    `relation<TD>source<TD>target<TD>keywords<TD>description`
    """
    entities, relations = [], []
    for line in re.split(r"\n|" + re.escape(record_delim), text):
        line = line.strip()
        if tuple_delim not in line:
            continue
        parts = [p.strip().strip('"').strip() for p in line.split(tuple_delim)]
        kind = parts[0].lower()
        if kind == "entity" and len(parts) >= 4:
            entities.append({"name": parts[1], "type": parts[2], "description": parts[3]})
        elif kind == "relation" and len(parts) >= 5:
            relations.append(
                {
                    "source": parts[1],
                    "target": parts[2],
                    "keywords": parts[3],
                    "description": parts[4],
                }
            )
    return entities, relations


def parse_json_output(text: str) -> tuple[list, list]:
    """Parsea el modo JSON de LightRAG (`ENTITY_EXTRACTION_USE_JSON=true`): un objeto
    {"entities": [...], "relationships": [...]}, con fallback a json_repair porque
    los modelos a veces envuelven la respuesta en ```json o la dejan a medias."""
    text = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import json_repair

        data = json_repair.loads(text)
    if not isinstance(data, dict):
        return [], []
    entities = [
        {"name": e.get("name", ""), "type": e.get("type", ""), "description": e.get("description", "")}
        for e in data.get("entities", []) or []
        if isinstance(e, dict)
    ]
    relations = [
        {
            "source": r.get("source", ""),
            "target": r.get("target", ""),
            "keywords": r.get("keywords", ""),
            "description": r.get("description", ""),
        }
        for r in data.get("relationships", []) or []
        if isinstance(r, dict)
    ]
    return entities, relations


def parse_keywords_output(text: str) -> tuple[list, list, bool]:
    """Parsea la salida del rol `keyword` de LightRAG (`keywords_extraction`,
    siempre JSON, no depende de ENTITY_EXTRACTION_USE_JSON): {"high_level_keywords":
    [...], "low_level_keywords": [...]}. `valid_shape` marca si el modelo respetó
    el esquema exacto que le pide el prompt (solo esas dos claves, ambas listas
    de strings) — el propio prompt lo exige explícitamente en la regla 2."""
    text = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import json_repair

        data = json_repair.loads(text)
    if not isinstance(data, dict):
        return [], [], False
    hl, ll = data.get("high_level_keywords"), data.get("low_level_keywords")
    valid_shape = (
        set(data.keys()) == {"high_level_keywords", "low_level_keywords"}
        and isinstance(hl, list)
        and isinstance(ll, list)
        and all(isinstance(x, str) for x in hl + ll)
    )
    hl = [x for x in hl if isinstance(x, str)] if isinstance(hl, list) else []
    ll = [x for x in ll if isinstance(x, str)] if isinstance(ll, list) else []
    return hl, ll, valid_shape


def is_reasoning_model(spec: str) -> bool:
    """gpt-5*/o1/o3/o4 vía Chat Completions: no aceptan temperature, max_tokens ni stop."""
    return bool(re.search(r"(gpt-5|:o[134])", spec.partition("@")[0]))


def production_json_schema() -> dict:
    """El esquema que el backend manda de verdad, importado del propio adapter.

    Así el benchmark mide producción en vez de una copia que puede desviarse.
    """
    from tools.vector_stores.lightrag.adapters import _EXTRACTION_JSON_SCHEMA

    return _EXTRACTION_JSON_SCHEMA


def make_llm(
    spec: str,
    temperature: float,
    num_ctx: int,
    max_tokens: int,
    json_mode: bool,
    response_format: str = "object",
):
    from langchain.chat_models import init_chat_model

    spec, _, base_url = spec.partition("@")  # openai:Qwen/Qwen3-8B@http://vllm:8000/v1
    kwargs = {} if is_reasoning_model(spec) else {"temperature": temperature}
    # max_tokens=0 → no se manda: reproduce el comportamiento sin tope, donde un
    # servidor OpenAI-compatible usa "lo que queda de ventana" (~30k en vLLM).
    if max_tokens and not is_reasoning_model(spec):
        kwargs["max_tokens"] = max_tokens
    if json_mode and spec.split(":", 1)[0] == "openai" and response_format != "none":
        # "object" = solo JSON valido (lo que medía el benchmark original).
        # "schema" = el esquema de produccion, decodificacion restringida.
        rf = (
            {"type": "json_object"}
            if response_format == "object"
            else {
                "type": "json_schema",
                "json_schema": {
                    "name": "lightrag_entity_extraction",
                    "schema": production_json_schema(),
                },
            }
        )
        kwargs["model_kwargs"] = {"response_format": rf}
    if base_url:
        # No mandamos la clave cloud a un endpoint local aunque OPENAI_API_KEY esté puesta.
        kwargs |= {"base_url": base_url, "api_key": os.getenv("VLLM_API_KEY", "EMPTY")}
    if spec.split(":", 1)[0] == "ollama":
        # Ollama trunca silenciosamente con num_ctx por defecto (2048) — sin esto
        # los modelos locales parecen peores de lo que son. vLLM no lo necesita:
        # usa el contexto real del modelo y da 400 si te pasas.
        kwargs["num_ctx"] = num_ctx
        kwargs.setdefault("base_url", os.getenv("OLLAMA_BASE_URL") or None)
    return init_chat_model(spec, **{k: v for k, v in kwargs.items() if v is not None})


def _self_check() -> None:
    sample = (
        "entity<|#|>ACME<|#|>Organization<|#|>Fabricante de widgets (desde 1950)\n"
        "entity<|#|>Bilbao<|#|>Location<|#|>Sede central\n"
        "relation<|#|>ACME<|#|>Bilbao<|#|>sede, ubicación<|#|>Tiene su sede en Bilbao\n"
        "ruido que no es un registro\n"
        "<|COMPLETE|>"
    )
    ents, rels = parse_records(sample, "<|#|>", "\n")
    assert [e["name"] for e in ents] == ["ACME", "Bilbao"], ents
    assert ents[0]["description"].endswith("(desde 1950)"), ents[0]
    assert len(rels) == 1 and rels[0]["keywords"] == "sede, ubicación", rels
    assert fill("hola {a}", {"a": 1, "b": 2}, "t") == "hola 1"

    json_sample = '```json\n{"entities": [{"name": "ACME", "type": "Organization", "description": "x"}], "relationships": [{"source": "ACME", "target": "Bilbao", "keywords": "sede", "description": "y"}]}\n```'
    jents, jrels = parse_json_output(json_sample)
    assert [e["name"] for e in jents] == ["ACME"], jents
    assert len(jrels) == 1 and jrels[0]["target"] == "Bilbao", jrels
    assert is_reasoning_model("openai:gpt-5.4-mini") and not is_reasoning_model("openai:gpt-4.1-mini")

    hl, ll, valid = parse_keywords_output('{"high_level_keywords": ["presión"], "low_level_keywords": ["BT Duo 500", "7 bar"]}')
    assert hl == ["presión"] and ll == ["BT Duo 500", "7 bar"] and valid
    _, _, invalid = parse_keywords_output('{"keywords": ["x"], "high_level_keywords": []}')
    assert not invalid

    print("self-check OK")


async def process_chunk(sem, llm, spec, i, chunk, system_prompt, user_template, values, args):
    user_prompt = fill(user_template, {**values, "input_text": chunk}, "user_prompt")
    row = {"model": spec, "chunk": i, "chars": len(chunk)}
    invoke_kwargs = {"config": {"metadata": {"lc_source": "extraction-bench"}}}
    if not args.json and not is_reasoning_model(spec):
        # Modelos de razonamiento rechazan `stop`; en modo JSON no hay
        # delimitador de fin que buscar.
        invoke_kwargs["stop"] = [values["completion_delimiter"]]
    salvaged = False
    async with sem:
        start = time.perf_counter()
        try:
            resp = await llm.ainvoke([("system", system_prompt), ("human", user_prompt)], **invoke_kwargs)
        except Exception as exc:
            # Con response_format, el SDK de OpenAI lanza LengthFinishReasonError en
            # vez de devolver el texto cortado. Produccion rescata el parcial (ver
            # adapters._salvage_length_limit), asi que el benchmark hace lo mismo o
            # mediria un comportamiento que ya no existe.
            resp = None
            try:
                from tools.vector_stores.lightrag.adapters import _salvage_length_limit

                resp = _salvage_length_limit(exc)
            except ImportError:
                pass
            if resp is None:
                row |= {"error": f"{type(exc).__name__}: {exc}"}
                print(f"[{spec}] chunk {i} ERROR: {exc}", file=sys.stderr)
                return row, None
            salvaged = True
        elapsed = time.perf_counter() - start

    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    ents, rels = (
        parse_json_output(content)
        if args.json
        else parse_records(content, values["tuple_delimiter"], values["record_delimiter"])
    )
    usage = resp.usage_metadata or {}
    finish_reason = (resp.response_metadata or {}).get("finish_reason", "")
    # Si no paró por stop/fin natural, el modelo no respetó los límites del
    # prompt (max_total_records) y lo cortamos con max_tokens — la cuenta de
    # entidades/relaciones de este chunk es un mínimo, no el resultado real.
    truncated = finish_reason not in ("stop", "")
    row |= {
        "entities": len(ents),
        "relations": len(rels),
        "entity_types": len({e["type"] for e in ents}),
        "latency_s": round(elapsed, 2),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "truncated": truncated or salvaged,
        "salvaged": salvaged,
        "error": "",
    }
    if truncated:
        print(
            f"[{spec}] chunk {i}: TRUNCADO (finish_reason={finish_reason}) — "
            "no respetó max_total_records, ver raw.jsonl",
            file=sys.stderr,
        )
    else:
        print(f"[{spec}] chunk {i}: {len(ents)} entidades, {len(rels)} relaciones, {elapsed:.1f}s")
    raw_record = {**row, "entities_detail": ents, "relations_detail": rels, "output": content}
    return row, raw_record


async def run_all(args, values, system_prompt, user_template, chunks):
    llms = {}
    for spec in args.model:
        try:
            llms[spec] = make_llm(spec, args.temperature, args.num_ctx, args.max_tokens, args.json, args.response_format)
        except Exception as exc:
            print(f"[{spec}] no se pudo crear el modelo: {exc}", file=sys.stderr)

    # Un semáforo por modelo: igual que LightRAG, que limita cuántas llamadas
    # concurrentes al LLM de extracción manda por *rol*, no en total.
    sems = {spec: asyncio.Semaphore(args.concurrency) for spec in llms}
    tasks = [
        process_chunk(sems[spec], llm, spec, i, chunk, system_prompt, user_template, values, args)
        for spec, llm in llms.items()
        for i, chunk in enumerate(chunks)
    ]
    return await asyncio.gather(*tasks)


async def process_keyword_query(sem, llm, spec, i, query, prompt_template, values):
    # Sin system prompt: LightRAG manda `keywords_extraction` como un único
    # mensaje (operate.py, extract_keywords_only), no system+human.
    prompt = fill(prompt_template, {**values, "query": query}, "keyword_prompt")
    row = {"model": spec, "query_idx": i, "query": query}
    async with sem:
        start = time.perf_counter()
        try:
            resp = await llm.ainvoke(
                [("human", prompt)], config={"metadata": {"lc_source": "extraction-bench"}}
            )
        except Exception as exc:
            row |= {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[{spec}] query {i} ERROR: {exc}", file=sys.stderr)
            return row, None
        elapsed = time.perf_counter() - start

    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    hl, ll, valid_shape = parse_keywords_output(content)
    usage = resp.usage_metadata or {}
    row |= {
        "high_level_keywords": "; ".join(hl),
        "low_level_keywords": "; ".join(ll),
        "hl_count": len(hl),
        "ll_count": len(ll),
        "valid_shape": valid_shape,
        "latency_s": round(elapsed, 2),
        "output_tokens": usage.get("output_tokens"),
        "error": "",
    }
    flag = "" if valid_shape else " ESQUEMA INVÁLIDO"
    print(f"[{spec}] query {i}:{flag} hl={hl} ll={ll} ({elapsed:.1f}s)")
    raw_record = {**row, "output": content}
    return row, raw_record


async def run_all_keywords(args, values, prompt_template, queries):
    llms = {}
    for spec in args.model:
        try:
            # response_format JSON siempre para este rol, sea cual sea --json:
            # LightRAG lo fuerza incondicionalmente en extract_keywords_only.
            llms[spec] = make_llm(spec, args.temperature, args.num_ctx, args.max_tokens, True)
        except Exception as exc:
            print(f"[{spec}] no se pudo crear el modelo: {exc}", file=sys.stderr)

    sems = {spec: asyncio.Semaphore(args.concurrency) for spec in llms}
    tasks = [
        process_keyword_query(sems[spec], llm, spec, i, query, prompt_template, values)
        for spec, llm in llms.items()
        for i, query in enumerate(queries)
    ]
    return await asyncio.gather(*tasks)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "document",
        nargs="?",
        type=Path,
        help="--role extract: documento .txt/.md/.pdf a trocear. "
        "--role keyword: fichero de texto con una consulta por línea.",
    )
    p.add_argument(
        "--role",
        choices=["extract", "keyword"],
        default="extract",
        help="Rol de LightRAG a comparar: `extract` (entidades/relaciones sobre chunks "
        "del documento, LIGHTRAG_EXTRACT_MODEL) o `keyword` (high/low-level keywords "
        "de una consulta, LIGHTRAG_KEYWORD_MODEL — siempre JSON, sin --json).",
    )
    p.add_argument(
        "-m",
        "--model",
        action="append",
        default=[],
        help="provider:model[@base_url] (repetible). base_url para vLLM u otro endpoint OpenAI-compatible",
    )
    p.add_argument("--language", default="Spanish")
    p.add_argument("--entity-types", help="Guía de tipos de entidad (default: la de LightRAG)")
    p.add_argument("--chunk-chars", type=int, default=4000)
    p.add_argument("--max-chunks", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--num-ctx", type=int, default=8192, help="Contexto de Ollama")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Tope de salida por chunk; 0 = sin tope (el servidor usa lo que quede de "
        "ventana). Si un modelo no respeta max_total_records y no para, esto evita "
        "quemar minutos por chunk (ver columna `truncated`)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Modo ENTITY_EXTRACTION_USE_JSON de LightRAG: salida JSON en vez de texto "
        "delimitado. El propio LightRAG dice que mejora la calidad y compatibilidad "
        "con modelos pequeños/open-source — el repo no lo tiene activado por defecto.",
    )
    p.add_argument(
        "--response-format",
        choices=["none", "object", "schema"],
        default="object",
        help="Con --json, que se manda al servidor: 'none' = nada (solo el prompt lo "
        "pide, como hacia LightRAG por defecto), 'object' = json_object (JSON valido, "
        "sin esquema), 'schema' = el esquema de produccion (decodificacion restringida)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Chunks concurrentes por modelo. Default 4 = DEFAULT_MAX_ASYNC de LightRAG "
        "(límite de llamadas concurrentes al LLM de extracción); el repo no lo sobreescribe.",
    )
    p.add_argument("--out", type=Path, default=Path("extraction-bench"))
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()

    if args.self_check:
        return _self_check()
    if not args.document or not args.model:
        p.error("hacen falta un documento y al menos un -m/--model")

    args.out.mkdir(parents=True, exist_ok=True)
    PROMPTS = load_prompts()

    if args.role == "keyword":
        return _run_keyword_role(args, PROMPTS)
    return _run_extract_role(args, PROMPTS)


def _run_extract_role(args, PROMPTS) -> None:
    values = build_prompt_values(PROMPTS, args.language, args.entity_types, args.json)
    system_key = "entity_extraction_json_system_prompt" if args.json else "entity_extraction_system_prompt"
    user_key = "entity_extraction_json_user_prompt" if args.json else "entity_extraction_user_prompt"
    system_prompt = fill(PROMPTS[system_key], values, "system_prompt")
    user_template = PROMPTS[user_key]

    chunks = chunk_text(read_text(args.document), args.chunk_chars)[: args.max_chunks]

    wall_start = time.perf_counter()
    results = asyncio.run(run_all(args, values, system_prompt, user_template, chunks))
    wall_elapsed = time.perf_counter() - wall_start
    rows = [row for row, _ in results]
    raw_records = [rec for _, rec in results if rec is not None]
    rows.sort(key=lambda r: (r["model"], r["chunk"]))
    raw_records.sort(key=lambda r: (r["model"], r["chunk"]))

    with (args.out / "raw.jsonl").open("w", encoding="utf-8") as raw:
        for rec in raw_records:
            raw.write(json.dumps(rec, ensure_ascii=False) + "\n")

    totals = {}
    for row in rows:
        agg = totals.setdefault(
            row["model"], {"entities": 0, "relations": 0, "latency_s": 0.0, "output_tokens": 0}
        )
        agg["entities"] += row.get("entities") or 0
        agg["relations"] += row.get("relations") or 0
        agg["latency_s"] += row.get("latency_s") or 0.0
        agg["output_tokens"] += row.get("output_tokens") or 0

    fields = ["model", "chunk", "chars", "entities", "relations", "entity_types", "latency_s", "input_tokens", "output_tokens", "truncated", "salvaged", "error"]
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'modelo':<40} {'ents':>6} {'rels':>6} {'seg (suma)':>10} {'out_tok':>8}")
    for spec, agg in totals.items():
        print(f"{spec:<40} {agg['entities']:>6} {agg['relations']:>6} {agg['latency_s']:>10.1f} {agg['output_tokens']:>8}")
    print(f"\nTiempo de pared total (concurrency={args.concurrency}): {wall_elapsed:.1f}s")
    print(f"{args.out / 'summary.csv'} · {args.out / 'raw.jsonl'}")


def _run_keyword_role(args, PROMPTS) -> None:
    values = {"language": args.language, "query": ""}
    examples = PROMPTS.get("keywords_extraction_examples", "")
    values["examples"] = "\n".join(examples) if isinstance(examples, list) else examples
    prompt_template = PROMPTS["keywords_extraction"]

    queries = [q.strip() for q in args.document.read_text(encoding="utf-8").splitlines() if q.strip()]

    wall_start = time.perf_counter()
    results = asyncio.run(run_all_keywords(args, values, prompt_template, queries))
    wall_elapsed = time.perf_counter() - wall_start
    rows = [row for row, _ in results]
    raw_records = [rec for _, rec in results if rec is not None]
    rows.sort(key=lambda r: (r["model"], r["query_idx"]))
    raw_records.sort(key=lambda r: (r["model"], r["query_idx"]))

    with (args.out / "raw.jsonl").open("w", encoding="utf-8") as raw:
        for rec in raw_records:
            raw.write(json.dumps(rec, ensure_ascii=False) + "\n")

    totals = {}
    for row in rows:
        agg = totals.setdefault(
            row["model"], {"hl_count": 0, "ll_count": 0, "invalid_shape": 0, "latency_s": 0.0, "output_tokens": 0}
        )
        agg["hl_count"] += row.get("hl_count") or 0
        agg["ll_count"] += row.get("ll_count") or 0
        agg["invalid_shape"] += 0 if row.get("valid_shape", True) else 1
        agg["latency_s"] += row.get("latency_s") or 0.0
        agg["output_tokens"] += row.get("output_tokens") or 0

    fields = ["model", "query_idx", "query", "high_level_keywords", "low_level_keywords", "hl_count", "ll_count", "valid_shape", "latency_s", "output_tokens", "error"]
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'modelo':<40} {'hl':>4} {'ll':>4} {'inválidos':>10} {'seg (suma)':>10} {'out_tok':>8}")
    for spec, agg in totals.items():
        print(
            f"{spec:<40} {agg['hl_count']:>4} {agg['ll_count']:>4} {agg['invalid_shape']:>10} "
            f"{agg['latency_s']:>10.1f} {agg['output_tokens']:>8}"
        )
    print(f"\nTiempo de pared total (concurrency={args.concurrency}): {wall_elapsed:.1f}s")
    print(f"{args.out / 'summary.csv'} · {args.out / 'raw.jsonl'}")


if __name__ == "__main__":
    main()
