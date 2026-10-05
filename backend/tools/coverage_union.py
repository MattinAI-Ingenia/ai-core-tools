"""Union of the literal coverage hits with the semantic hits, per document.

The coverage tool finds documents by the literal phrase, so a manual that writes
"EN 303- 5" (extra space) or "pressure relief valve" is missed. Semantic search
finds those, but returns only a few dozen chunks. This keeps EVERY literal
document and adds the semantic-only ones the reranker scores as relevant.

The reranker decides which snippet represents each document and which extra
documents are worth showing; it never cuts a literal document, because the
phrase itself is the evidence. For an enumeration the unit of the answer is the
document, not the chunk: a top-K cut over chunks would favour documents with
many relevant chunks and drop the ones that mention the thing once.
"""
from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

# ponytail: set by eye on a handful of questions; re-tune with a labelled set.
SEMANTIC_MIN_SCORE = 0.3
# Between the two thresholds a document is listed as "possible" (one snippet) and
# the agent decides: the reranker scores a short mention inside a long chunk at
# 0.01-0.04 (an "Anti-frost function" section), so a single cut would drop it.
POSSIBLE_MIN_SCORE = 0.05
MAX_ADDED_DOCS = 15
MAX_POSSIBLE_DOCS = 10
_SNIPPET_CHARS = 1500
EXCERPT_CHARS = 300  # what the LLM reads of a page: a window on the part that matters

Snippet = Tuple[str, str, Optional[int]]  # (file_path, content, page)
Grouped = Dict[Any, List[Snippet]]


def excerpt_around(text: str, term: str, width: int = EXCERPT_CHARS) -> str:
    """A *width*-character window of *text* centred on the first occurrence of *term*.

    The LLM used to be shown the first 240 characters of a page, which is where
    the mention is not: a "possible" document was judged on text that did not
    contain it. Separators in the term match any run of spaces, hyphens and line
    breaks, like the literal search does ("EN 303-5" finds "EN 303-⏎5"). With no
    match the start of the page is returned.
    """
    flat = " ".join(text.split())
    tokens = [t for t in re.split(r"[\s\-]+", term) if t]
    match = re.search(r"[\s-]+".join(re.escape(t) for t in tokens), flat, re.IGNORECASE) if tokens else None
    start = max(0, match.start() - width // 3) if match else 0
    end = start + width
    return ("…" if start > 0 else "") + flat[start:end] + ("…" if end < len(flat) else "")


def _best_window(flat: str, scores: List[float], windows: List[Tuple[int, int]]) -> str:
    start, end = windows[max(range(len(windows)), key=lambda i: scores[i])]
    return ("…" if start > 0 else "") + flat[start:end] + ("…" if end < len(flat) else "")


async def _fill_best_windows(term, snippets_by_path, rerank_fn, excerpts_out) -> None:
    """Rerank half-overlapping windows of each document's best page and keep, per
    page, the window the reranker likes best. Best effort: a failure leaves the
    excerpts out and the caller falls back to the start of the page."""
    flats: Dict[str, str] = {}
    spans: Dict[str, List[Tuple[int, int]]] = {}
    texts: List[str] = []
    owner: List[Tuple[str, int]] = []
    for path, content in snippets_by_path.items():
        flat = " ".join(content.split())
        step = EXCERPT_CHARS // 2
        flats[path] = flat
        spans[path] = [(a, a + EXCERPT_CHARS) for a in range(0, max(1, len(flat) - step), step)]
        for i, (a, b) in enumerate(spans[path]):
            texts.append(flat[a:b])
            owner.append((path, i))
    if not texts:
        return
    try:
        ranked = await rerank_fn(term, texts)
    except Exception:
        logger.warning("[coverage] window rerank failed; excerpts start at the top of the page", exc_info=True)
        return
    by_path: Dict[str, List[float]] = {p: [0.0] * len(w) for p, w in spans.items()}
    for r in ranked:
        path, i = owner[r["index"]]
        by_path[path][i] = r["relevance_score"]
    for path in snippets_by_path:
        excerpts_out[path] = _best_window(flats[path], by_path[path], spans[path])


async def augment_with_semantic(
    grouped: Grouped,
    term: str,
    *,
    semantic_chunks_fn: Callable[[str], Awaitable[List[dict]]],
    rerank_fn: Optional[Callable[[str, List[str]], Awaitable[List[dict]]]],
    max_added: int = MAX_ADDED_DOCS,
    max_possible: int = MAX_POSSIBLE_DOCS,
    excerpts_out: Optional[Dict[str, str]] = None,
) -> Tuple[Grouped, Dict[Any, str]]:
    """Return ``(grouped, tiers)``: *grouped* extended with semantic-only
    documents, *tiers* maps each of those resource ids to ``"semantic"``
    (score >= SEMANTIC_MIN_SCORE) or ``"possible"`` (weaker, one snippet).

    The reranker scores every candidate (it also picks each document's best
    snippet) and its score decides the tier.

    Falls back to the untouched literal result when there is no reranker (no
    scores, no way to judge) or when the semantic search or the reranker fails.
    """
    if rerank_fn is None:
        return grouped, {}
    try:
        chunks = await semantic_chunks_fn(term)
    except Exception:
        logger.warning("[coverage] semantic search failed; literal result only", exc_info=True)
        return grouped, {}

    # The literal search keys its result by str ("311"), semantic chunks carry
    # int ids: compare as int or every literal document would be re-added.
    known = {int(k) for k in grouped}
    candidates: List[Tuple[Any, Snippet, bool]] = []  # (resource_id, snippet, is_semantic_only)
    for rid, snippets in grouped.items():
        candidates.extend((rid, s, False) for s in snippets)
    for c in chunks:
        rid = c.get("resource_id")
        if rid is None or int(rid) in known:
            continue  # a chunk of a document already found literally adds nothing
        candidates.append((rid, (c.get("file_path", ""), c.get("content", ""), c.get("page")), True))

    if not candidates:  # the rerank API rejects an empty list (422) and retries before failing
        return grouped, {}
    try:
        ranked = await rerank_fn(term, [s[1][:_SNIPPET_CHARS] for _, s, _ in candidates])
    except Exception:
        logger.warning("[coverage] rerank failed; literal result only", exc_info=True)
        return grouped, {}
    score = {r["index"]: r["relevance_score"] for r in ranked}

    sem_min, pos_min = SEMANTIC_MIN_SCORE, POSSIBLE_MIN_SCORE

    def best_first(items: List[Tuple[int, Snippet]]) -> List[Snippet]:
        return [s for _, s in sorted(items, key=lambda it: -score.get(it[0], 0.0))]

    by_doc: Dict[Any, List[Tuple[int, Snippet]]] = {}
    extras: Dict[Any, None] = {}  # insertion-ordered set of semantic-only ids
    for i, (rid, snippet, extra) in enumerate(candidates):
        if extra and score.get(i, 0.0) < pos_min:
            continue
        by_doc.setdefault(rid, []).append((i, snippet))
        if extra:
            extras[rid] = None

    result: Grouped = {rid: best_first(by_doc[rid]) for rid in grouped}
    top = {rid: max(score.get(i, 0.0) for i, _ in by_doc[rid]) for rid in extras}
    ranked_extras = sorted(extras, key=lambda rid: -top[rid])
    semantic = [rid for rid in ranked_extras if top[rid] >= sem_min][:max_added]
    possible = [rid for rid in ranked_extras if top[rid] < sem_min][:max_possible]
    result.update({rid: best_first(by_doc[rid]) for rid in semantic})
    result.update({rid: best_first(by_doc[rid])[:1] for rid in possible})
    tiers = {**dict.fromkeys(semantic, "semantic"), **dict.fromkeys(possible, "possible")}
    if excerpts_out is not None and tiers:
        # excerpts_out is keyed by file_path ("DOC.pdf (p. 7)"), unique per page
        await _fill_best_windows(
            term, {result[rid][0][0]: result[rid][0][1] for rid in tiers}, rerank_fn, excerpts_out,
        )
    return result, tiers
