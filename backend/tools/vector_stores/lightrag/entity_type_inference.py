"""Propose LightRAG entity types from a corpus, before anything is indexed.

The problem this solves: ``Silo.lightrag_entity_types`` shapes the whole
knowledge graph and cannot change once extraction has run, but at silo-creation
time nobody knows what the corpus is about. Leaving it blank falls back to
LightRAG's generic categories (``Persona, Criatura, Lugar, …``), which are
useless for, say, boiler manuals — every parameter code and error code lands in
``Concepto``.

Feeding whole documents to an LLM is not an option: a 92-page manual is ~52k
tokens and a self-hosted model's window is commonly 20k. But a document's
**cover page plus its table of contents** is ~580 tokens on average and is
almost pure domain vocabulary::

    1.5 ADVERTENCIAS SOBRE EL REFRIGERANTE DE LA BOMBA DE CALOR
    13.1 PARÁMETROS DEL SISTEMA
    16 BLOQUEOS DE SEGURIDAD
    21 CARACTERÍSTICAS TÉCNICAS

So 30 documents fit comfortably in one call, which matters more than depth:
entity *types* are a genre-level vocabulary that saturates after a couple of
documents, while what keeps appearing is per-family vocabulary (pellet/cenicero
for biomass, refrigerante/compresor for heat pumps). Breadth beats depth, hence
:func:`select_diverse`.

This module does the extraction and prompt building only — no LLM call, no DB.
"""

from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# A table-of-contents line ends in a page number after a run of dot leaders:
#   "13.1 PARÁMETROS DEL SISTEMA ...................... 47"
# Some manuals emit the leaders as two runs with a space between them, which a
# lazy group happily swallows into the title, so the tail is stripped after the
# match rather than trusted to the pattern (one real manual went from 47 to
# 276 chars per line that way).
_TOC_LINE_RE = re.compile(r"^(.*?)\.{3,}\s*(\d{1,4})\s*$")
_TOC_TAIL_RE = re.compile(r"[\s.]+$")

# Cover pages repeat the same title block in every manual; it carries no
# product information and would dominate the family key.
_COVER_BOILERPLATE_RE = re.compile(
    r"INSTRUCCIONES\s+DE\s+INSTALACI[OÓ]N\s+Y\s+FUNCIONAMIENTO"
    r"|INSTRUC[CT]IONS?\s+D[´'`]?INSTALLATION(?:\s+ET\s+DE\s+FONCTIONNEMENT)?"
    r"|INSTALLATION\s+AND\s+OPERATING\s+INSTRUCTIONS"
    r"|MANUAL\s+DE\s+SERVICIO"
    r"|SERVICE\s+MANUAL",
    re.IGNORECASE,
)

# Certification stamps ("CGM-04/392", "ER-0170/1996") and language markers
# sit on the cover before the product name in some manuals.
_COVER_NOISE_RE = re.compile(
    r"\b(?:CGM|ER)-[\d/]+\b"
    r"|\b(?:ES|EN|GB|FR|PT|IT|EU|DE|NL|PL|RU|HU|RO|BG)\s*\|"
    r"|[|]",
    re.IGNORECASE,
)

# Words that can lead a cover page but never identify a product family.
_NOT_A_FAMILY = frozenset({
    "SISTEMA", "MODULO", "MÓDULO", "KIT", "DEPOSITO", "DEPÓSITO", "CALDERA",
    "BOMBA", "ACUMULADOR", "UNIDAD", "GRUPO", "CONJUNTO", "MANUAL",
    "INSTRUCCIONES", "DOCUMENTO", "ANEXO", "SUPLEMENTO", "ADVERTENCIAS",
})

# How much of page 1 to keep. The product name is always in the first lines;
# past that comes legal boilerplate.
_COVER_CHARS = 300

# Pages 1..N are where a table of contents lives when there is one.
_TOC_SCAN_PAGES = 6

_MIN_TOC_LINES = 3

# Sections worth opening: the ones whose body is a table of literal instances
# (P20, E20-4, 32 bar). A table of contents proves such a section exists but
# never shows an instance, and a model asked for examples off an index alone
# invents plausible ones — which is exactly what the user would be shown to
# decide on.
#
# Grouped into buckets rather than one alternation for two reasons: the bucket
# is the dedup key (without it "BLOQUEOS DE SEGURIDAD" and its "BLOQUEO DE
# SEGURIDAD POR TEMPERATURA" subsection read as two different sections and eat
# both slots on the same chapter), and the declared order is the priority when
# a document offers more sections than the budget allows.
#
# \b matters: without it "BLOQUEO" matches inside "FUNCIÓN ANTIBLOQUEO DE
# BOMBAS", and "BLOQUEO DE TECLADO" is a keypad feature, not a fault table.
_SECTION_BUCKETS = (
    ("especificaciones", re.compile(
        r"\bCARACTER[IÍ]STICAS?\s+T[EÉ]CNICAS?|\bTECHNICAL\s+DATA|\bSPECIFICATION",
        re.IGNORECASE)),
    ("parametros", re.compile(
        r"\bPAR[AÁ]METRO|\bMEN[UÚ]\s+T[EÉ]CNICO|\bPARAMETER", re.IGNORECASE)),
    ("errores", re.compile(
        r"\bC[OÓ]DIGO|\bERROR|\bAVER[IÍ]A|\bALARMA|\bFAULT|\bALARM", re.IGNORECASE)),
    ("bloqueos", re.compile(r"\bBLOQUEOS?\s+DE\s+SEGURIDAD", re.IGNORECASE)),
)

# The printed page number in a TOC is not the PDF page index — front matter
# shifts it, usually by a few pages and never consistently across publishers.
# So the number is only a hint: the title is searched for in a window around
# it, and the number is used as-is only when the search comes up empty.
_PAGE_SEARCH_BACK = 3
_PAGE_SEARCH_FORWARD = 8

# Enough to catch a table's header row and its first entries, which is all it
# takes to show what an instance of the section looks like.
_SAMPLE_CHARS = 700

# The search window reaches back a few pages, which for a section on page 2 or
# 3 lands on the table of contents itself — where the title obviously appears.
# A page carrying this many dot-leader lines is an index, not a body page.
_TOC_PAGE_LEADERS = 3

# Fallback only, for when tiktoken is unavailable — approx_tokens counts for
# real otherwise (see _count_tokens). Measured against a real payload of
# Spanish manual outlines and spec tables: 50.836 characters tokenised to
# 22.156 tokens. The usual "4 chars per token" rule of thumb is for prose;
# accented words, model codes (THERMAPRO 16 HTT) and figures with units all
# split much harder. That measurement is what set this ratio, and it still
# missed budget on a later run (prompt_tokens=12342 against a 10500 budget) —
# an approximation, however well-calibrated, drifts with whatever corpus is
# selected this time. Real counting doesn't.
_CHARS_PER_TOKEN = 2.3


@functools.lru_cache(maxsize=1)
def _token_encoder():
    """tiktoken encoder matching silo_service._token_encoder — not imported
    from there: this tools-layer module does not depend on the services
    layer. Same encoding choice for consistency, not because this call goes
    through gpt-4o-mini specifically (the configured ai_service may differ);
    it is an approximation of whatever model actually runs either way."""
    import tiktoken
    try:
        return tiktoken.encoding_for_model("gpt-4o-mini")
    except Exception:
        return tiktoken.get_encoding("o200k_base")


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        return len(_token_encoder().encode(text))
    except Exception:
        return int(len(text) / _CHARS_PER_TOKEN)  # ponytail: tiktoken unavailable

# Cap per document so one 172-page manual cannot eat the whole context window.
# Top-level entries are kept first: they carry the document's outline, while
# the third-level ones repeat the same vocabulary at finer grain.
_MAX_TOC_LINES = 60

# A top-level entry starts with a bare section number: "21 CARACTERÍSTICAS…"
_TOP_LEVEL_RE = re.compile(r"^\d{1,2}\s+\S")


@dataclass
class DocumentOutline:
    """The cheap, high-signal skeleton of one document."""

    doc_id: str
    cover: str
    toc: List[str] = field(default_factory=list)
    family: str = ""
    pages: int = 0
    # (section title, body excerpt) pairs, filled by sample_sections.
    samples: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def approx_tokens(self) -> int:
        body = "\n".join(f"{t}\n{x}" for t, x in self.samples)
        text = self.cover + "\n".join(self.toc) + body
        return _count_tokens(text)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# A cover word carried by more than this share of the corpus is boilerplate,
# not a product name. 0.4 leaves room for a genuinely dominant product line.
_COMMON_TOKEN_RATIO = 0.4


def _cover_tokens(cover: str) -> List[str]:
    """Uppercase word tokens of a cover, boilerplate phrases already removed."""
    text = _COVER_NOISE_RE.sub(" ", _COVER_BOILERPLATE_RE.sub(" ", cover))
    return [t.upper() for t in re.findall(r"[A-Za-zÁÉÍÓÚÑáéíóúñ][\w\-]{1,}", text)]


def _family_key(cover: str) -> str:
    """Best-effort product-family label for diversity sampling.

    Not authoritative and not stored — its only job is to stop the sampler from
    picking thirty manuals of the same product line. A wrong key costs one
    redundant sample, never a wrong entity type.
    """
    for token in _cover_tokens(cover):
        if len(token) > 2 and token not in _NOT_A_FAMILY:
            return token
    return "?"


def _cap_toc(toc: List[str]) -> List[str]:
    """Trim a table of contents to _MAX_TOC_LINES, keeping document order.

    Top-level entries are protected; the subsections that survive are the ones
    that appeared first, so the kept slice still reads as a coherent outline.
    """
    if len(toc) <= _MAX_TOC_LINES:
        return toc
    keep = {i for i, line in enumerate(toc) if _TOP_LEVEL_RE.match(line)}
    for i in range(len(toc)):
        if len(keep) >= _MAX_TOC_LINES:
            break
        keep.add(i)
    return [line for i, line in enumerate(toc) if i in keep][:_MAX_TOC_LINES]


def extract_outline(pdf_path: str, doc_id: str) -> Optional[DocumentOutline]:
    """Read one PDF's cover and table of contents. None if it has no text layer.

    Scanned-only documents (image despieces, for instance) yield nothing here,
    which is correct: they contribute no vocabulary to infer types from.
    """
    import pypdf  # noqa: PLC0415 — heavy import, only needed on this path

    try:
        reader = pypdf.PdfReader(pdf_path)
    except Exception as exc:  # noqa: BLE001 — a corrupt PDF must not abort the batch
        logger.warning("Cannot read %s for outline extraction: %s", doc_id, exc)
        return None

    pages = reader.pages
    if not pages:
        return None

    cover = _clean(pages[0].extract_text() or "")[:_COVER_CHARS]

    toc: List[str] = []
    for page in pages[:_TOC_SCAN_PAGES]:
        for line in (page.extract_text() or "").split("\n"):
            match = _TOC_LINE_RE.match(line.strip())
            if match:
                title = _TOC_TAIL_RE.sub("", _clean(match.group(1)))
                if title:
                    toc.append(f"{title} — p.{match.group(2)}")

    toc = _cap_toc(toc)

    if not cover and len(toc) < _MIN_TOC_LINES:
        logger.info("%s has no usable text layer; skipping.", doc_id)
        return None

    return DocumentOutline(
        doc_id=doc_id,
        cover=cover,
        toc=toc,
        family=_family_key(cover),
        pages=len(pages),
    )


def assign_families(outlines: List[DocumentOutline]) -> None:
    """Recompute every ``family`` using the corpus as its own stopword list.

    :func:`_family_key` alone cannot know that "INSTRUCCIONS" is boilerplate in
    a French-Spanish manual — the hand-written regex only covers the phrasings
    someone thought of. Words carried by most covers in the corpus are, by
    definition, not what tells one product from another, so they are dropped
    here. Self-tuning, and it works in any language without a wordlist.
    """
    if not outlines:
        return
    seen_in: dict[str, int] = {}
    per_doc = []
    for outline in outlines:
        tokens = _cover_tokens(outline.cover)
        per_doc.append(tokens)
        for token in set(tokens):
            seen_in[token] = seen_in.get(token, 0) + 1

    ubiquitous = {t for t, n in seen_in.items() if n / len(outlines) > _COMMON_TOKEN_RATIO}
    for outline, tokens in zip(outlines, per_doc):
        outline.family = next(
            (t for t in tokens if t not in ubiquitous and t not in _NOT_A_FAMILY),
            outline.family or "?",
        )


_TOC_PAGE_RE = re.compile(r"—\s*p\.(\d+)\s*$")
_SECTION_NUMBER_RE = re.compile(r"^[\d.]+\s+")


def _match_key(title: str) -> str:
    """Uppercased title without its section number, for page matching."""
    return _SECTION_NUMBER_RE.sub("", title).upper().strip()


def sample_sections(
    pdf_path: str,
    outline: DocumentOutline,
    limit: int = 2,
) -> None:
    """Attach body excerpts from the outline's instance-bearing sections.

    Uses the table of contents as a map: it already names the interesting
    sections *and* where they are, so no page has to be guessed or scored. The
    printed page number is treated as a hint rather than an index — the section
    title is searched for in a window around it, because front matter shifts
    the numbering by a few pages in most manuals.

    Mutates ``outline.samples``. Silent on failure: a missing excerpt costs
    grounding for one section, never the whole inference.
    """
    import pypdf  # noqa: PLC0415

    # First TOC entry per bucket, then the buckets in priority order.
    first_per_bucket: dict[str, Tuple[str, int]] = {}
    for line in outline.toc:
        page_match = _TOC_PAGE_RE.search(line)
        if not page_match:
            continue
        for bucket, pattern in _SECTION_BUCKETS:
            if bucket not in first_per_bucket and pattern.search(line):
                first_per_bucket[bucket] = (
                    _TOC_PAGE_RE.sub("", line).strip(),
                    int(page_match.group(1)),
                )
                break

    wanted = [first_per_bucket[b] for b, _ in _SECTION_BUCKETS if b in first_per_bucket][:limit]
    if not wanted:
        return

    try:
        pages = pypdf.PdfReader(pdf_path).pages
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cannot reopen %s to sample sections: %s", outline.doc_id, exc)
        return

    for title, printed_page in wanted:
        key = _match_key(title)[:25]
        start = max(0, printed_page - 1 - _PAGE_SEARCH_BACK)
        stop = min(len(pages), printed_page + _PAGE_SEARCH_FORWARD)
        excerpt = ""
        for index in range(start, stop):
            raw = pages[index].extract_text() or ""
            if raw.count("...") >= _TOC_PAGE_LEADERS:
                continue
            text = _clean(raw)
            if key and key in text.upper():
                excerpt = text[:_SAMPLE_CHARS]
                break
        if not excerpt and 0 < printed_page <= len(pages):
            raw = pages[printed_page - 1].extract_text() or ""
            if raw.count("...") < _TOC_PAGE_LEADERS:
                excerpt = _clean(raw)[:_SAMPLE_CHARS]
        if excerpt:
            outline.samples.append((title, excerpt))


def select_diverse(outlines: List[DocumentOutline], limit: int = 30) -> List[DocumentOutline]:
    """Pick up to *limit* outlines, spreading them across product families.

    Round-robins over families so the first pass takes one document per family
    before any family gets a second. Within a family the richest table of
    contents wins, since that is the one carrying the most vocabulary.
    """
    assign_families(outlines)
    if len(outlines) <= limit:
        return list(outlines)

    by_family: dict[str, List[DocumentOutline]] = {}
    for outline in outlines:
        by_family.setdefault(outline.family, []).append(outline)
    for group in by_family.values():
        group.sort(key=lambda o: len(o.toc), reverse=True)

    # Deterministic family order — richest family first, then alphabetical, so
    # the same corpus always yields the same sample.
    families = sorted(by_family, key=lambda f: (-len(by_family[f][0].toc), f))

    picked: List[DocumentOutline] = []
    depth = 0
    while len(picked) < limit:
        added = False
        for family in families:
            group = by_family[family]
            if depth < len(group):
                picked.append(group[depth])
                added = True
                if len(picked) == limit:
                    break
        if not added:
            break
        depth += 1
    return picked


def fit_budget(outlines: List[DocumentOutline], max_tokens: int) -> List[DocumentOutline]:
    """Trim the payload until it fits *max_tokens*, cheapest content first.

    A self-hosted model's window is small and the provider rejects the whole
    call when it is exceeded, so the budget has to be enforced here rather than
    hoped for. Second excerpts go first (the specs table already carries most
    of the instance vocabulary), then whole documents from the tail — which is
    the least diverse end, since select_diverse front-loads one per family.

    Returns copies with trimmed samples; the inputs are left alone.
    """
    from copy import copy  # noqa: PLC0415

    kept = [copy(o) for o in outlines]
    if sum(o.approx_tokens for o in kept) <= max_tokens:
        return kept

    for outline in reversed(kept):
        if sum(o.approx_tokens for o in kept) <= max_tokens:
            break
        outline.samples = outline.samples[:1]

    while kept and sum(o.approx_tokens for o in kept) > max_tokens:
        kept.pop()
    return kept


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# Kept as two hand-written prompts rather than one translated at runtime: the
# rules are about wording ("singular", "CamelCase") and a machine translation
# of them drifts. Selected by Silo.lightrag_language, the same setting that
# picks LightRAG's own extraction language.
_PROMPT_ES = """\
Eres un analista de documentación técnica.

Abajo tienes, de {n} documentos del mismo fondo documental: la portada, el
ÍNDICE, y algún extracto literal de las secciones con tablas.

Tu tarea NO es resumir de qué hablan los capítulos. Es deducir qué CLASES DE
COSAS CONCRETAS se nombran dentro, para tiparlas como nodos de un grafo.

NO quiero: "Instalacion", "Mantenimiento", "Esquema". Son capítulos, no cosas.
Un técnico no busca "un Mantenimiento"; busca "el código E20-4" o "la válvula
de seguridad".

SÍ quiero: si un capítulo se llama "CÓDIGOS DE ERROR", la clase es
"CodigoError" con ejemplos como "E20-4". Si se llama "CARACTERÍSTICAS
TÉCNICAS", dentro hay modelos, magnitudes con unidades y normativas citadas:
son clases distintas.

Reglas:
- Entre {lo} y {hi} clases. Un solo sustantivo, español, singular, CamelCase.
- Cada "examples" debe ser una INSTANCIA copiada literalmente de los extractos
  de abajo: un código, un modelo, una cifra con unidad, un nombre propio.
  Nunca un título de capítulo, y nunca un ejemplo inventado.
- Si no encuentras 2 instancias literales de una clase, bórrala.
- No propongas una clase cajón de sastre ("Otro", "General", "Elemento"):
  LightRAG ya clasifica como `Other` lo que no encaja en ninguna, y ponerla en
  la lista solo la hace más atractiva y vacía a las demás.
- Sin solapamientos: fusiona las clases que un técnico buscaría juntas.
  "ModeloCaldera" y "ModeloBomba" son una sola clase, "Modelo".
- Una FILA de una tabla no es una clase. Presión, temperatura, potencia,
  caudal y rendimiento son todos lo mismo: una cifra con su unidad. Una única
  clase para todas ellas, no una por fila.
- Antes de responder, repasa si el corpus nombra: quién fabrica o distribuye,
  con qué normas cumple, qué sustancias contiene, y qué funciones se activan o
  desactivan. Si alguna de esas aparece con instancias literales, es una clase.

Responde únicamente con este JSON, sin texto alrededor:
{{"types": [{{"name": "...", "why": "...", "examples": ["...", "..."]}}]}}

--- DOCUMENTOS ---
{documents}"""

_PROMPT_EN = """\
You are a technical-documentation analyst.

Below, for {n} documents from one corpus: the cover page, the TABLE OF
CONTENTS, and some verbatim excerpts from the sections that hold tables.

Your job is NOT to summarise what the chapters discuss. It is to work out what
KINDS OF CONCRETE THINGS are named inside them, to type as graph nodes.

Not wanted: "Installation", "Maintenance", "Diagram". Those are chapters, not
things. A technician does not look up "a Maintenance"; they look up "error code
E20-4" or "the safety valve".

Wanted: if a chapter is called "ERROR CODES", the class is "ErrorCode" with
examples like "E20-4". If it is called "TECHNICAL DATA", inside there are
models, quantities with units and cited standards: those are separate classes.

Rules:
- Between {lo} and {hi} classes. One noun, English, singular, CamelCase.
- Every "examples" entry must be an INSTANCE copied verbatim from the excerpts
  below: a code, a model, a figure with its unit, a proper name. Never a
  chapter title, and never an invented example.
- If you cannot find 2 literal instances of a class, drop it.
- Do not propose a catch-all class ("Other", "General", "Item"): LightRAG
  already files anything that fits nothing under `Other`, and listing it only
  makes it more attractive and drains the real classes.
- No overlap: merge classes a technician would look up together.
  "BoilerModel" and "PumpModel" are one class, "Model".
- A table ROW is not a class. Pressure, temperature, power, flow rate and
  efficiency are all the same thing: a figure with a unit. One class for all of
  them, not one per row.
- Before answering, check whether the corpus names: who makes or distributes
  it, which standards it complies with, what substances it contains, and which
  functions can be switched on or off. Where those appear with literal
  instances, they are classes.

Reply with this JSON only, no surrounding text:
{{"types": [{{"name": "...", "why": "...", "examples": ["...", "..."]}}]}}

--- DOCUMENTS ---
{documents}"""

# Second pass. The first one reads a whole corpus and has to abstract at the
# same time; asked to do both it reliably over-splits — one class per row of a
# specifications table (Presion, Temperatura, Potencia, Caudal…) — because the
# evidence in front of it is row-shaped. Merging is a much smaller job, and
# giving it its own call with nothing but the candidate list is both cheap
# (~500 tokens) and markedly better than another rule in the first prompt.
_CONSOLIDATE_ES = """\
Estas son las clases de entidad que un primer análisis propuso para un fondo
documental técnico. Están fragmentadas de más.

{candidates}

Fusiona únicamente las que se solapan de verdad. Como mucho pueden quedar
{hi}, pero quedarse en menos es mejor que forzar el número: NUNCA añadas una
clase para llegar a la cifra.

El "name" de cada clase que devuelvas tiene que ser uno de los nombres de la
lista de arriba. No inventes nombres nuevos.

Criterio de fusión: dos clases son la misma si un técnico las buscaría en el
mismo sitio. "Presion" y "Temperatura" son ambas una cifra con su unidad:
una sola clase. "ModeloCaldera" y "ModeloBomba" son "Modelo".

Criterio para NO fusionar: si las instancias tienen formas distintas y se
consultan por motivos distintos, déjalas separadas. Un código de error y un
parámetro configurable se parecen, pero uno diagnostica y el otro se ajusta.

Al fusionar, conserva los mejores ejemplos de las clases originales. No
inventes clases nuevas ni ejemplos nuevos: solo fusiona, renombra y descarta.

Responde únicamente con este JSON:
{{"types": [{{"name": "...", "why": "...", "examples": ["...", "..."]}}]}}"""

_CONSOLIDATE_EN = """\
These are the entity classes a first pass proposed for a technical corpus.
They are over-fragmented.

{candidates}

Merge only the ones that genuinely overlap. At most {hi} may remain, but
fewer is better than forcing the number: NEVER add a class to reach the count.

Every "name" you return must be one of the names listed above. Do not invent
new names.

Merge test: two classes are the same if a technician would look them up in the
same place. "Pressure" and "Temperature" are both a figure with a unit: one
class. "BoilerModel" and "PumpModel" are "Model".

Do NOT merge when the instances have different shapes and are consulted for
different reasons. An error code and a settable parameter look alike, but one
diagnoses and the other is adjusted.

When merging, keep the best examples from the originals. Do not invent new
classes or new examples: only merge, rename and drop.

Reply with this JSON only:
{{"types": [{{"name": "...", "why": "...", "examples": ["...", "..."]}}]}}"""


def build_consolidation_prompt(
    types: List[dict],
    language: Optional[str] = None,
    max_types: int = 10,
) -> str:
    """Second-pass prompt that merges over-fragmented candidate classes."""
    template = _CONSOLIDATE_ES if (language or "").strip().lower() == "spanish" else _CONSOLIDATE_EN
    candidates = "\n".join(
        f"- {t.get('name')}: {', '.join(t.get('examples', [])[:4])}" for t in types
    )
    return template.format(candidates=candidates, hi=max_types)


# The JSON contract above, as a schema for providers that support constrained
# decoding (vLLM/OpenAI `response_format`) — same trick as
# LIGHTRAG_EXTRACT_GUIDED_JSON, so the format stops being a failure mode.
ENTITY_TYPES_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "types": {
            "type": "array",
            "minItems": 4,
            "maxItems": 20,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "why": {"type": "string"},
                    # minItems is load-bearing: without it a model under
                    # pressure to hit a target count emits a class with an
                    # empty examples list, and "no instances" is exactly the
                    # signal that the class should not exist.
                    "examples": {"type": "array", "minItems": 2, "items": {"type": "string"}},
                },
                "required": ["name", "why", "examples"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["types"],
    "additionalProperties": False,
}


def render_documents(outlines: List[DocumentOutline]) -> str:
    """Serialize outlines into the block the prompt embeds."""
    blocks = []
    for outline in outlines:
        lines = [f"### {outline.doc_id} ({outline.pages} pág.)"]
        if outline.cover:
            lines.append(f"Portada: {outline.cover}")
        lines.extend(outline.toc)
        for title, excerpt in outline.samples:
            lines.append(f"[extracto · {title}] {excerpt}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_prompt(
    outlines: List[DocumentOutline],
    language: Optional[str] = None,
    min_types: int = 8,
    max_types: int = 12,
    max_tokens: int = 14000,
) -> str:
    """Build the inference prompt in the silo's configured language.

    ``max_tokens`` budgets the document payload only; leave headroom in it for
    the model's window minus the instructions and the completion.
    """
    outlines = fit_budget(outlines, max_tokens)
    template = _PROMPT_ES if (language or "").strip().lower() == "spanish" else _PROMPT_EN
    return template.format(
        n=len(outlines),
        lo=min_types,
        hi=max_types,
        documents=render_documents(outlines),
    )
