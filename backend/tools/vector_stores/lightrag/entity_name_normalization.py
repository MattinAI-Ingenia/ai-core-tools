"""Entity-name canonicalization for LightRAG (merge-key folding + variant tracking).

LightRAG (lightrag-hku 1.5.6) merges entities only when the extracted
``entity_name`` matches EXACTLY after its own ``normalize_entity_name()``
cleanup (HTML tags, full-width chars, quotes, NBSP, short-numeric filter).
Names that differ only in case, accents, spaces or hyphens ("MCF-40" /
"mcf 40" / "Mcf 40") stay as separate graph nodes — documented in
``docs/dependencies/lightrag.md`` §6.1 and measured in
``docs/testing/lightrag_json_and_token_cap_benchmark.md`` (recomendación 5).

This module closes that gap with two pieces:

* ``canonicalize_entity_name()`` — folds a LightRAG-cleaned name into a
  canonical merge key: NFKC, accents stripped, casefolded, whitespace and
  hyphen-like separators collapsed to single spaces. Deterministic and
  side-effect free.
* A monkeypatch of LightRAG's ``normalize_entity_name`` (the single choke
  point ``operate.py`` uses for every extracted entity/relation endpoint)
  that returns the canonical form as the merge key. Because the key is
  identical for every spelling variant, LightRAG's own
  ``_merge_nodes_then_upsert()`` fuses them natively — no library patching
  of the merge itself.

The display name is NOT the canonical key (it would show as lowercase,
accent-less text). While the patch is active, every call records which raw
(cleanup-preserving) spelling was seen for which canonical key in a
per-run :class:`EntityNameVariantCollector`. The wrapper drains the
collector after each indexing run and persists the counts in the
``entity_name_variant`` table (one row per (silo, canonical, variant),
mention-count incremented on conflict). At graph-render time the store
picks the most-mentioned variant per canonical key as the display name;
entities with no recorded variant keep the canonical name.

Mention-count semantics: ``normalize_entity_name()`` is called once per
entity record and once per relation endpoint, so counts are *mentions*
(a form used by one entity record plus three relations counts four).
That bias is harmless for picking a display name — both spellings come
from the same extraction — and keeps the patch a single call site.

Limitations (documented in §6.1): counts are lost if the indexing process
crashes mid-run (display name is cosmetic); graphs indexed BEFORE this
change keep their non-canonical names, so legacy duplicates remain until
a reindex/retroactive cleanup pass.
"""

from __future__ import annotations

import contextvars
import logging
import re
import threading
import unicodedata
from collections import Counter, defaultdict
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Hyphen-like codepoints that should be treated as plain separators:
# ASCII hyphen-minus, soft hyphen, Armenian/Hebrew hyphens, Unicode hyphen
# family (U+2010..U+2015 incl. non-breaking hyphen and horizontal bar),
# minus sign, double-oblique and small hyphens, and fullwidth hyphen.
_HYPHEN_LIKE_RE = re.compile(
    "[\u002d\u00ad\u058a\u05be\u1400\u1806\u2010-\u2015\u2212\u2e3a\u2e3b\ufe58\ufe63\uff0d]+"
)
_WHITESPACE_RE = re.compile(r"\s+")

# contextvars slot: holds the active collector during an indexing run,
# or None outside one (e.g. during query). Same pattern as
# ``adapters._active_accumulator``.
_active_collector: contextvars.ContextVar[Optional["EntityNameVariantCollector"]] = (
    contextvars.ContextVar("_lightrag_entity_name_variant_collector", default=None)
)

_INSTALL_LOCK = threading.Lock()
_PATCH_INSTALLED = False

# LightRAG truncates entity names after normalization (lightrag.constants.
# DEFAULT_ENTITY_NAME_MAX_LENGTH = 256, enforced via _truncate_entity_identifier
# before the merge), so the canonical key recorded here MUST be clamped to the
# same limit — otherwise a >256-char name would get variant rows that can never
# match the actual node name. Resolved at install time; 256 as fallback keeps
# the module importable without the library.
_MERGE_KEY_MAX_LENGTH = 256


def canonicalize_entity_name(raw: str) -> str:
    """Fold *raw* into the canonical merge key LightRAG fuses nodes on.

    Order matters: NFKC first (full-width/compatibility chars → ASCII),
    NFKD + combining-mark strip (accents → base letter), casefold, then
    hyphen-like runs → single space and whitespace runs → single space.
    Returns ``""`` only when the input was empty; a name that folds to
    nothing (e.g. ``"-"``) is passed through as-is by the caller.
    """
    if not raw:
        return ""
    text = unicodedata.normalize("NFKC", raw)
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    text = unicodedata.normalize("NFC", stripped).casefold()
    text = _HYPHEN_LIKE_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


class EntityNameVariantCollector:
    """Per-indexing-run map of canonical key → {raw spelling → mention count}.

    Shared by reference across the asyncio tasks spawned inside one
    ``apipeline_process_enqueue_documents()`` run (contextvars pass the
    object by reference, mutations happen on the collection's single event
    loop, so no locking is needed). Drained once, after the run, by the
    wrapper that created it.
    """

    def __init__(self) -> None:
        self._counts: Dict[str, Counter[str]] = defaultdict(Counter)

    def record(self, canonical: str, raw_variant: str) -> None:
        """Count one mention of *raw_variant* for merge key *canonical*."""
        if canonical and raw_variant:
            self._counts[canonical][raw_variant] += 1

    def drain(self) -> Dict[str, Dict[str, int]]:
        """Return and clear the accumulated counts as plain dicts."""
        drained = {canonical: dict(counter) for canonical, counter in self._counts.items()}
        self._counts.clear()
        return drained

    def has_data(self) -> bool:
        return bool(self._counts)


def set_active_variant_collector(collector: Optional[EntityNameVariantCollector]) -> contextvars.Token:
    """Set the active variant collector; returns a token for reset."""
    return _active_collector.set(collector)


def reset_active_variant_collector(token: contextvars.Token) -> None:
    """Restore the collector context var to its previous value."""
    _active_collector.reset(token)


def get_active_variant_collector() -> Optional[EntityNameVariantCollector]:
    """Return the active collector, or None outside an indexing run."""
    return _active_collector.get()


def _make_canonicalizing_normalize(original, merge_key_max_length: int) -> Callable[[str], str]:
    """Build the patched ``normalize_entity_name`` around LightRAG's original.

    Keeps LightRAG's contract exactly: its own cleaning (encoding,
    quotes, short-numeric filter) runs first — an empty result means the
    caller drops the record, so we return it untouched. The canonical key is
    clamped to LightRAG's own merge-key length limit so the variant rows can
    match the node name LightRAG will actually store.
    """
    def normalize_entity_name_canonical(input_text: str) -> str:
        cleaned = original(input_text)
        if not cleaned:
            return cleaned
        canonical = canonicalize_entity_name(cleaned)
        if not canonical:
            return cleaned
        if len(canonical) > merge_key_max_length:
            canonical = canonical[:merge_key_max_length]
        if canonical != cleaned:
            collector = _active_collector.get()
            if collector is not None:
                collector.record(canonical, cleaned)
            return canonical
        return cleaned

    return normalize_entity_name_canonical


def ensure_entity_name_normalization_patch() -> None:
    """Monkeypatch LightRAG's ``normalize_entity_name`` (idempotent, lock-guarded).

    ``lightrag.operate`` imports the function by name from ``lightrag.utils``
    at module load (``from lightrag.utils import normalize_entity_name``), so
    rebinding only ``lightrag.utils`` would leave such modules calling the
    original. Rather than hard-coding the known import sites (they can change
    with a library upgrade), every already-imported ``lightrag.*`` module whose
    ``normalize_entity_name`` attribute still points at the original is
    rebound to the canonicalizing wrapper.

    Concurrency: ``_build_rag`` runs on several threads (per-collection init
    locks), so installation is guarded by a lock and ordered so a module that
    imports DURING the patch gets the wrapper: ``lightrag.utils`` is rebound
    first, then the ``sys.modules`` snapshot is scanned.
    """
    global _PATCH_INSTALLED, _MERGE_KEY_MAX_LENGTH
    if _PATCH_INSTALLED:
        return
    import sys  # noqa: WPS433 - local to keep module import cheap

    with _INSTALL_LOCK:
        if _PATCH_INSTALLED:
            return
        try:
            import lightrag.utils as _utils  # noqa: WPS433
            from lightrag.constants import (  # noqa: WPS433
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
            )
        except ImportError:
            return
        _MERGE_KEY_MAX_LENGTH = int(DEFAULT_ENTITY_NAME_MAX_LENGTH)

        original = _utils.normalize_entity_name
        if getattr(original, "_mattin_canonicalizes", False):
            # Another call site (or a re-import under reload) installed it already.
            _PATCH_INSTALLED = True
            return

        patched = _make_canonicalizing_normalize(original, _MERGE_KEY_MAX_LENGTH)
        patched._mattin_canonicalizes = True  # type: ignore[attr-defined]

        # 1) Rebind the source module FIRST: every `from lightrag.utils import
        #    normalize_entity_name` executed after this point gets the wrapper.
        _utils.normalize_entity_name = patched

        # 2) Rebind modules already imported that still hold the original.
        rebound = 0
        for module_name, module in list(sys.modules.items()):
            if not module_name or not module_name.startswith("lightrag"):
                continue
            try:
                bound = getattr(module, "normalize_entity_name", None)
            except Exception:  # noqa: WPS433 - module objects are heterogeneous
                continue
            if bound is original:
                module.normalize_entity_name = patched
                rebound += 1

        _PATCH_INSTALLED = True
        logger.info(
            "LightRAG entity-name canonicalization installed: merge keys folded "
            "(NFKC + accents + casefold + space/hyphen collapse, clamp %d chars), "
            "variants counted for display-name selection (%d lightrag module(s) rebound)",
            _MERGE_KEY_MAX_LENGTH,
            rebound,
        )
