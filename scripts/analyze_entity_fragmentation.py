#!/usr/bin/env python3
"""Muestra para clasificar a mano la fragmentación de entidades por subsunción.

LightRAG fusiona solo por nombre EXACTO. Cuando un nombre extraído contiene a
otro ('bt duo' ⊂ 'tapa elíptica bt duo 500-1000'), quedan como nodos separados.
El benchmark de JSON/tope de tokens midió 127-145 de estos pares por corpus
(docs/testing/lightrag_json_and_token_cap_benchmark.md, §6 y recomendación 5)
pero solo contó cuántos había — no si fusionarlos ayuda. La subsunción por
substring mezcla tres casos distintos que requieren tres acciones distintas:

  1. variant  — variante real del mismo nombre ('mcf 40' / 'quemador mcf 40')
               → debería fusionarse
  2. hyponym  — parte-de/hiperónimo sin extraer del todo
               ('bt duo' ⊂ 'tapa elíptica bt duo 500-1000': una pieza, no el
               equipo) → NO fusionar; como mucho una arista
  3. noise    — ruido puro (grado 0-1, entrada de una tabla de despiece)
               → ni fusionar ni conectar, es candidato a descartar

Esto no se puede automatizar sin decidir primero cuál de las tres domina. Este
script no clasifica nada: agrupa los pares subsumidos por su grado (que es lo
único medible sin ojos humanos) y saca una muestra estratificada para que una
persona rellene la columna `class` a mano. Si la muestra sale mayoría "noise",
fusionar no es la solución — hay que limpiar en la extracción, no en el grafo
(ver docs/dependencies/lightrag.md o el propio benchmark, recomendación 4:
enum de tipos de entidad).

Uso:
    python scripts/analyze_entity_fragmentation.py backend/data/bench/qwen_esquema_8192 \
        --sample 30 --out fragmentation_sample.csv

    python scripts/analyze_entity_fragmentation.py --self-check

Entrada: uno o más `raw.jsonl` (los que escribe compare_extraction.py en --out),
buscados recursivamente bajo el directorio dado. Cada línea debe tener
`entities_detail` / `relations_detail` (formato --json de compare_extraction.py).
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


def norm(s: str) -> str:
    return " ".join((s or "").split()).strip().lower()


def iter_raw_records(root: Path):
    """Yield (variant, model, entities_detail, relations_detail) por línea de
    cada raw.jsonl bajo `root`. `variant` es el subdirectorio inmediato de
    `root` (o `root.name` si el fichero está directamente en `root`) — así una
    corrida con varias configuraciones (una carpeta por variante, como
    backend/data/bench/) se etiqueta igual que en graph_analysis.py/dupes.py.
    """
    for f in sorted(root.rglob("raw.jsonl")):
        rel = f.relative_to(root)
        variant = rel.parts[0] if len(rel.parts) > 1 else root.name
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            model = (rec.get("model") or "?").split("@")[0].replace("openai:", "")
            yield variant, model, rec.get("entities_detail") or [], rec.get("relations_detail") or []


def load_groups(root: Path, model_filter: str | None):
    """(variant, model) -> (entities_detail acumulados, relations_detail acumulados)."""
    groups: dict[tuple[str, str], tuple[list, list]] = defaultdict(lambda: ([], []))
    for variant, model, ents, rels in iter_raw_records(root):
        if model_filter and model_filter not in model:
            continue
        acc_ents, acc_rels = groups[(variant, model)]
        acc_ents += ents
        acc_rels += rels
    return groups


def degrees(rels: list[dict]) -> dict[str, int]:
    """Vecinos únicos por nombre normalizado — mismo criterio que graph_analysis.py."""
    adj: dict[str, set] = defaultdict(set)
    for r in rels:
        a, b = norm(r.get("source")), norm(r.get("target"))
        if a and b and a != b:
            adj[a].add(b)
            adj[b].add(a)
    return {n: len(v) for n, v in adj.items()}


def find_subsumed_pairs(names: set[str], min_short_len: int = 6) -> list[tuple[str, str]]:
    """Pares (short, long) donde `short` aparece como frase completa (delimitada
    por espacios) dentro de `long`. Filtro heredado de dupes.py: si `short` es
    una única palabra que coincide con un token entero de `long`
    ('quemador' ⊂ 'quemador mcf 40'), se descarta — es el nombre genérico del
    tipo de entidad conviviendo con el nombre específico, no fragmentación.
    Solo sobreviven las frases de varias palabras o los identificadores
    compuestos, que es lo que de verdad reporta el benchmark ('bt duo',
    'mcf 40'). `min_short_len` descarta fragmentos triviales (siglas de 2-3
    letras que aparecen por casualidad dentro de cualquier nombre largo).
    """
    keys = sorted(names, key=len)
    pairs = []
    for i, short in enumerate(keys):
        if len(short) < min_short_len:
            continue
        for long in keys[i + 1:]:
            if short == long:
                continue
            if short in long.split():  # palabra exacta contenida: no cuenta
                continue
            if f" {short} " in f" {long} ":
                pairs.append((short, long))
                break  # el primer (= más corto) superconjunto basta
    return pairs


def degree_tier(degree_long: int) -> str:
    """Solo para ESTRATIFICAR la muestra — no es la clasificación final.
    El grado del nombre largo es lo único medible sin leer el par; separar en
    bandas garantiza que la muestra de 30 no se quede solo con los casos
    obvios de un extremo."""
    if degree_long <= 1:
        return "0-1 (probable ruido)"
    if degree_long <= 3:
        return "2-3 (ambiguo)"
    return "4+ (conectado)"


def build_rows(root: Path, model_filter: str | None, min_short_len: int) -> list[dict]:
    rows = []
    for (variant, model), (ents, rels) in load_groups(root, model_filter).items():
        if not ents:
            continue
        mentions = Counter(norm(e.get("name")) for e in ents if norm(e.get("name")))
        deg = degrees(rels)
        for short, long in find_subsumed_pairs(set(mentions), min_short_len):
            rows.append({
                "variant": variant,
                "model": model,
                "short": short,
                "long": long,
                "degree_short": deg.get(short, 0),
                "degree_long": deg.get(long, 0),
                "mentions_short": mentions[short],
                "mentions_long": mentions[long],
                "tier": degree_tier(deg.get(long, 0)),
                "class": "",  # a rellenar a mano: variant | hyponym | noise
            })
    return rows


def stratified_sample(rows: list[dict], sample_size: int, seed: int) -> list[dict]:
    """Reparte `sample_size` a partes iguales entre los tiers presentes, para
    que la muestra no sea toda del tier mayoritario. Si un tier tiene menos de
    su cupo, el resto se redistribuye entre los otros."""
    rng = random.Random(seed)
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_tier[row["tier"]].append(row)
    for bucket in by_tier.values():
        rng.shuffle(bucket)

    tiers = list(by_tier)
    if not tiers:
        return []
    quota = {t: sample_size // len(tiers) for t in tiers}
    quota[tiers[0]] += sample_size - sum(quota.values())  # resto al primero

    picked: list[dict] = []
    leftover = 0
    for t in tiers:
        take = min(quota[t], len(by_tier[t]))
        picked += by_tier[t][:take]
        leftover += quota[t] - take
    if leftover:
        remaining = [row for t in tiers for row in by_tier[t][quota[t]:]]
        rng.shuffle(remaining)
        picked += remaining[:leftover]
    return picked


def _self_check() -> None:
    # 'bt duo' es un hub real (grado alto); 'tapa elíptica bt duo 500-1000' es
    # una pieza mencionada una vez, sin relaciones — el caso "hyponym"/"noise"
    # que motiva este script.
    entities = [
        {"name": "bt duo", "type": "Artifact", "description": ""},
        {"name": "tapa elíptica bt duo 500-1000", "type": "Artifact", "description": ""},
        {"name": "quemador", "type": "Artifact", "description": ""},
        {"name": "quemador mcf 40", "type": "Artifact", "description": ""},
        {"name": "mcf 40", "type": "Artifact", "description": ""},
    ]
    relations = [
        {"source": "bt duo", "target": "quemador", "keywords": "", "description": ""},
        {"source": "bt duo", "target": "mcf 40", "keywords": "", "description": ""},
        {"source": "quemador mcf 40", "target": "mcf 40", "keywords": "", "description": ""},
    ]
    names = {norm(e["name"]) for e in entities}
    pairs = find_subsumed_pairs(names)
    assert ("bt duo", "tapa elíptica bt duo 500-1000") in pairs, pairs
    assert ("mcf 40", "quemador mcf 40") in pairs, pairs
    # 'quemador' es palabra completa dentro de 'quemador mcf 40' → descartado
    assert not any(short == "quemador" for short, _ in pairs), pairs

    deg = degrees(relations)
    assert deg["bt duo"] == 2, deg
    assert deg.get("tapa elíptica bt duo 500-1000", 0) == 0, deg

    assert degree_tier(0) == "0-1 (probable ruido)"
    assert degree_tier(2) == "2-3 (ambiguo)"
    assert degree_tier(5) == "4+ (conectado)"

    sample_rows = [
        {"tier": "0-1 (probable ruido)", "short": "a", "long": "aa"},
        {"tier": "0-1 (probable ruido)", "short": "b", "long": "bb"},
        {"tier": "4+ (conectado)", "short": "c", "long": "cc"},
    ]
    sample = stratified_sample(sample_rows, sample_size=2, seed=0)
    assert len(sample) == 2, sample
    assert {row["tier"] for row in sample} == {"0-1 (probable ruido)", "4+ (conectado)"}, sample

    print("self-check OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", nargs="?", type=Path, help="Directorio con raw.jsonl (recursivo)")
    p.add_argument("--model", help="Filtra a modelos cuyo nombre contenga esta subcadena")
    p.add_argument("--min-short-len", type=int, default=6, help="Longitud mínima del nombre corto (default: 6, como dupes.py)")
    p.add_argument("--sample", type=int, default=30, help="Tamaño de la muestra a clasificar a mano (default: 30)")
    p.add_argument("--seed", type=int, default=0, help="Semilla del muestreo, para reproducibilidad (default: 0)")
    p.add_argument("--out", type=Path, default=Path("entity_fragmentation_sample.csv"))
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()

    if args.self_check:
        return _self_check()
    if not args.root:
        p.error("hace falta un directorio con raw.jsonl (o --self-check)")
    if not args.root.exists():
        p.error(f"no existe: {args.root}")

    rows = build_rows(args.root, args.model, args.min_short_len)
    if not rows:
        print("Ningún par subsumido encontrado (¿directorio correcto? ¿--model demasiado restrictivo?)", file=sys.stderr)
        return

    by_tier = Counter(row["tier"] for row in rows)
    print(f"{len(rows)} pares subsumidos totales")
    for tier, n in sorted(by_tier.items()):
        print(f"  {tier}: {n} ({100 * n / len(rows):.0f}%)")

    sample = stratified_sample(rows, args.sample, args.seed)
    fields = ["variant", "model", "short", "long", "degree_short", "degree_long",
              "mentions_short", "mentions_long", "tier", "class"]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sample)

    print(f"\nMuestra de {len(sample)} pares en {args.out}")
    print("Rellena la columna `class` a mano con uno de:")
    print("  variant  — variante real del mismo nombre → debería fusionarse")
    print("  hyponym  — parte-de/hiperónimo sin extraer → NO fusionar")
    print("  noise    — ruido puro (tabla de despiece, etc.) → descartar")


if __name__ == "__main__":
    main()
