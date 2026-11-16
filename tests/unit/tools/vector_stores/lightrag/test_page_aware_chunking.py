"""Unit tests for page-aware chunking (lightrag-pdf-document-schema step_003).

Covers the page-marker helpers, per-strategy option resolution, the
``_achunk_page_aware`` legacy ``chunking_func`` hook, and the removal of
LightRAG's explicit chunking selector (``process_options``) from the
enqueue/insert paths — every document now goes through the hook.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

lightrag = pytest.importorskip("lightrag")

pytestmark = pytest.mark.asyncio

from tools.vector_stores import lightrag_store
from tools.vector_stores.lightrag_store import (
    LightRAGStore,
    _achunk_page_aware,
    _aenqueue,
    _ainsert,
    _assign_start_pages,
    _resolve_strategy_options,
    page_marked_text,
    strip_page_markers,
)


def _char_tokenizer():
    """Hermetic char-level tokenizer — same pattern as test_lightrag_contract.py."""
    from lightrag.utils import Tokenizer

    class _CharTokenModel:
        def encode(self, s):
            return [ord(c) for c in s]

        def decode(self, t):
            return "".join(map(chr, t))

    return Tokenizer("char", _CharTokenModel())


def _fake_rag(chunk_token_size=120, fixed_token_overrides=None):
    """Minimal fake ``rag`` carrying everything ``_achunk_page_aware`` reads.

    ``default_chunker_config`` IS importable on lightrag-hku 1.5.6 (confirmed
    live), so no hand-built fallback was needed.
    """
    from lightrag.parser.routing import default_chunker_config

    cfg = default_chunker_config()
    cfg["chunk_token_size"] = chunk_token_size
    if fixed_token_overrides:
        cfg["fixed_token"] = {**cfg["fixed_token"], **fixed_token_overrides}
    return SimpleNamespace(
        addon_params={"chunker": cfg},
        chunk_token_size=chunk_token_size,
        embedding_func=AsyncMock(),
        workspace="silo_t",
    )


def _expected_size_opts(rag, option):
    """Independently re-derive (size, opts) straight from the library's own
    ``resolve_chunk_options``/``normalize_r_separators`` — used to cross-check
    ``_achunk_page_aware``'s real call args without trusting our own
    ``_resolve_strategy_options`` circularly.
    """
    from lightrag.chunker.recursive_character import normalize_r_separators
    from lightrag.parser.routing import resolve_chunk_options

    key = {
        "F": "fixed_token", "R": "recursive_character",
        "V": "semantic_vector", "P": "paragraph_semantic",
    }[option]
    chunk_opts = resolve_chunk_options(rag.addon_params, process_options=option)
    resolved = int(chunk_opts.get("chunk_token_size") or rag.chunk_token_size)
    opts = dict(chunk_opts.get(key) or {})
    size = int(opts.pop("chunk_token_size", resolved))
    if option == "R" and "separators" in opts:
        seps = normalize_r_separators(opts["separators"])
        if seps:
            opts["separators"] = seps
        else:
            opts.pop("separators")
    return size, opts


# ---------------------------------------------------------------------------
# Helpers: page_marked_text / strip_page_markers
# ---------------------------------------------------------------------------


class TestPageMarkerHelpers:
    def test_round_trip(self):
        marked = page_marked_text([(1, "abc"), (2, "def")])
        assert strip_page_markers(marked) == ("abc\n\ndef", [(0, 1), (5, 2)])

    def test_page_number_gaps_are_preserved(self):
        marked = page_marked_text([(1, "a"), (2, "b"), (4, "c")])
        _, starts = strip_page_markers(marked)
        assert [p for _, p in starts] == [1, 2, 4]

    def test_lookalike_marker_in_page_text_is_neutralised(self):
        marked = page_marked_text([(1, "hello\n<<<page:99>>>\nworld")])
        clean, starts = strip_page_markers(marked)
        assert "<<<page:99>>>" not in clean
        assert [p for _, p in starts] == [1]
        # Neutralised, not deleted: exactly one '<' was dropped.
        assert "<<page:99>>>" in clean

    def test_unmarked_text_is_returned_unchanged(self):
        text = "just some text\nwith lines\n"
        assert strip_page_markers(text) == (text, [])

    def test_marker_lookalike_mid_text_not_at_start_is_unchanged(self):
        text = "intro line\n<<<page:3>>>\nmore text after"
        assert strip_page_markers(text) == (text, [])


class TestPageNumberDigitCap:
    """Review-board fix B: an unbounded digit group risks int()'s string-
    conversion limit (5000+ digits) and chunk-id column overflow (~255
    chars) once prefixed. The marker regexes cap at 6 digits.
    """

    def test_huge_page_number_does_not_match_and_does_not_raise(self):
        huge = "9" * 6000
        text = f"<<<page:{huge}>>>\ncontent"
        # Must not match (so no attempt to int() a 6000-digit string), and
        # must not raise — falls through to "just unmarked content".
        clean, starts = strip_page_markers(text)
        assert clean == text
        assert starts == []


class TestMarkerForgeryViaSanitizer:
    """Review-board fix A: a payload that doesn't LOOK like a marker yet (so
    _PAGE_MARKER_LOOKALIKE_RE doesn't touch it) must not sanitize INTO a real
    one after LightRAG's own ``sanitize_text_for_encoding`` runs on the
    stored text. ``page_marked_text`` must pre-apply that same sanitizer so
    neutralisation happens on the text LightRAG will actually parse.
    """

    @pytest.mark.parametrize(
        "payload",
        [
            "&lt;&lt;&lt;page:999&gt;&gt;&gt;",
            "<<\x01<page:999>>>",
            "<<<page:999>>>\x00",
            "<<<page:999>>>\x7f",
            # Multi-layer-encoded: sanitize_text_for_encoding only peels ONE
            # layer per pass, so these survive a single sanitize-and-check
            # unresolved and would only forge a marker on a SECOND pass
            # (which page_marked_text's single application does not perform
            # — the "&" re-escape blocks LightRAG's later second pass too).
            "&amp;lt;&amp;lt;&amp;lt;page:999&amp;gt;&amp;gt;&amp;gt;",
            "&\x01lt;&\x01lt;&\x01lt;page:999&\x01gt;&\x01gt;&\x01gt;",
        ],
    )
    def test_sanitizer_forgeable_payload_does_not_become_a_real_marker(self, payload):
        from lightrag.utils import sanitize_text_for_encoding

        marked = page_marked_text([(1, f"before\n{payload}\nafter"), (2, "page two text")])
        # Simulate LightRAG applying its own sanitizer again to the stored
        # text (apipeline_enqueue_documents does this) before the hook's
        # strip_page_markers ever sees it.
        clean, starts = strip_page_markers(sanitize_text_for_encoding(marked))

        assert [p for _, p in starts] == [1, 2]
        assert "<<<page:999>>>" not in clean

    def test_legitimate_entity_in_page_text_is_unescaped_exactly_once(self):
        """Content-fidelity side effect of the same fix: real page text
        containing a literal HTML entity must come out matching what
        LightRAG's OWN single sanitize pass would produce on that text
        directly — not double-unescaped by an extra pass of ours.
        """
        from lightrag.utils import sanitize_text_for_encoding

        text = "intro &amp;lt;b&amp;gt; outro"
        baseline = sanitize_text_for_encoding(text)  # what LightRAG alone would do

        marked = page_marked_text([(1, text)])
        # Simulate LightRAG's own later sanitize pass on the whole enqueued doc.
        clean, _ = strip_page_markers(sanitize_text_for_encoding(marked))

        assert clean == baseline


# ---------------------------------------------------------------------------
# Real chunking, F strategy
# ---------------------------------------------------------------------------


class TestRealFixedTokenChunking:
    async def test_contiguous_ids_and_independent_page_match(self):
        tokenizer = _char_tokenizer()
        pages = [
            (1, "Page one. " + ("alpha beta gamma delta. " * 8)),
            (2, "Page two. " + ("epsilon zeta eta theta. " * 8)),
            (3, "Page three. " + ("iota kappa lambda mu. " * 8)),
        ]
        marked = page_marked_text(pages)
        clean, starts = strip_page_markers(marked)
        offsets = [o for o, _ in starts]

        from bisect import bisect_right

        rag = _fake_rag(chunk_token_size=40, fixed_token_overrides={"chunk_overlap_token_size": 5})
        chunks = await _achunk_page_aware(rag, "F", tokenizer, marked)

        assert len(chunks) > 3
        checked = 0
        for order, chunk in enumerate(chunks):
            assert chunk["chunk_id"].endswith(f"-chunk-{order:03d}")
            span = chunk.get("_source_span")
            if span is None:
                # The real chunker's trailing chunk can omit _source_span
                # (observed live on lightrag-hku 1.5.6) — _assign_start_pages
                # carries the previous page forward for it, nothing to
                # independently verify here.
                continue
            expected_page = starts[bisect_right(offsets, span["start"]) - 1][1]
            assert chunk["chunk_id"] == f"p{expected_page}-chunk-{order:03d}"
            checked += 1
        assert checked > 0, "expected at least one chunk with a _source_span to verify"

    async def test_chunk_spanning_multiple_pages_attributed_to_start_page(self):
        tokenizer = _char_tokenizer()
        pages = [(1, "A" * 150), (2, "B" * 10), (3, "C" * 10), (4, "D" * 10)]
        marked = page_marked_text(pages)
        clean, starts = strip_page_markers(marked)
        offsets = [o for o, _ in starts]
        size = offsets[1]  # a window boundary lands exactly at page 2's start

        rag = _fake_rag(chunk_token_size=size, fixed_token_overrides={"chunk_overlap_token_size": 0})
        chunks = await _achunk_page_aware(rag, "F", tokenizer, marked)

        spanning = [c for c in chunks if c["content"].startswith("B" * 10)]
        assert len(spanning) == 1
        chunk = spanning[0]
        assert chunk["chunk_id"].startswith("p2-chunk-")
        assert "B" * 10 in chunk["content"]
        assert "C" * 10 in chunk["content"]
        assert "D" * 10 in chunk["content"]
        assert "<<<" not in chunk["content"]


# ---------------------------------------------------------------------------
# .find() fallback (no _source_span) — tested directly against
# _assign_start_pages, since the real R/P chunkers on lightrag-hku 1.5.6 DO
# emit _source_span (see report) so this path is exercised synthetically.
# ---------------------------------------------------------------------------


class TestFindFallback:
    def test_unfindable_first_chunk_takes_first_kept_page(self):
        clean = "xxxxxxxxxx"
        starts = [(0, 3)]  # first kept page starts at page 3 (1-2 filtered upstream)
        chunks = [{"content": "not in clean at all"}]
        _assign_start_pages(chunks, clean, starts)
        assert chunks[0]["chunk_id"] == "p3-chunk-000"

    def test_unfindable_chunk_carries_forward_previous_page(self):
        clean = "AAAA" + "BBBB"
        starts = [(0, 1), (4, 2)]
        chunks = [
            {"content": "BBBB"},               # found at page 2
            {"content": "not findable at all"},  # carries page 2 forward
        ]
        _assign_start_pages(chunks, clean, starts)
        assert chunks[0]["chunk_id"] == "p2-chunk-000"
        assert chunks[1]["chunk_id"] == "p2-chunk-001"

    def test_cursor_never_steps_backward(self):
        clean = "AAAA" + "BBBB" + "CCCC"
        starts = [(0, 1), (4, 2), (8, 3)]
        # Deliberately out of order: chunk 0 is found late in the text, chunk
        # 1's content coincidentally also matches near the start — the
        # cursor must not let chunk 1 be (mis)found before chunk 0's offset.
        chunks = [
            {"content": "CCCC"},  # found at offset 8 -> page 3, cursor -> 8
            {"content": "AAAA"},  # not findable from offset 8 onward -> carry forward page 3
        ]
        _assign_start_pages(chunks, clean, starts)
        assert chunks[0]["chunk_id"] == "p3-chunk-000"
        assert chunks[1]["chunk_id"] == "p3-chunk-001"


# ---------------------------------------------------------------------------
# Strategy dispatch
# ---------------------------------------------------------------------------


class TestStrategyDispatch:
    @pytest.mark.parametrize(
        "option,func_name",
        [
            ("F", "chunking_by_fixed_token"),
            ("R", "chunking_by_recursive_character"),
            ("P", "chunking_by_paragraph_semantic"),
        ],
    )
    async def test_sync_strategy_called_once_with_expected_args(self, option, func_name):
        tokenizer = _char_tokenizer()
        content = "plain unmarked text with no page markers at all " * 3
        rag = _fake_rag(chunk_token_size=50)
        expected_size, expected_opts = _expected_size_opts(rag, option)

        with patch(f"lightrag.chunker.{func_name}", return_value=[]) as mock_fn:
            chunks = await _achunk_page_aware(rag, option, tokenizer, content)

        assert chunks == []
        mock_fn.assert_called_once()
        args, kwargs = mock_fn.call_args
        assert args[0] is tokenizer
        assert args[1] == content
        assert args[2] == expected_size
        if option == "F":
            assert kwargs.pop("_emit_source_span") is True
        assert kwargs == expected_opts

    async def test_v_strategy_awaited_once_with_expected_args(self):
        tokenizer = _char_tokenizer()
        content = "plain unmarked text with no page markers at all " * 3
        rag = _fake_rag(chunk_token_size=50)
        expected_size, expected_opts = _expected_size_opts(rag, "V")

        with patch(
            "lightrag.chunker.chunking_by_semantic_vector",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_fn:
            chunks = await _achunk_page_aware(rag, "V", tokenizer, content)

        assert chunks == []
        mock_fn.assert_awaited_once()
        args, kwargs = mock_fn.call_args
        assert args[0] is tokenizer
        assert args[1] == content
        assert args[2] == expected_size
        assert kwargs.pop("embedding_func") is rag.embedding_func
        assert kwargs == expected_opts

    def test_chunk_strategy_maps_to_option_and_defaults_to_f(self):
        store = LightRAGStore(
            db=None, ai_service=None, embedding_service=None,
            lightrag_chunk_strategy="token_window",
        )
        assert store._chunk_process_option == "F"

        store_default = LightRAGStore(db=None, ai_service=None, embedding_service=None)
        assert store_default._chunk_process_option == "F"


# ---------------------------------------------------------------------------
# Threading boundary
# ---------------------------------------------------------------------------


class TestThreadingBoundary:
    async def test_sync_strategies_run_only_inside_to_thread(self):
        tokenizer = _char_tokenizer()
        content = "plain unmarked text " * 5
        rag = _fake_rag(chunk_token_size=20)
        real_to_thread = asyncio.to_thread
        seen = []

        async def spy(func, *a, **kw):
            seen.append(func)
            return await real_to_thread(func, *a, **kw)

        with patch("lightrag.chunker.chunking_by_fixed_token", return_value=[]) as mock_fn, \
             patch("asyncio.to_thread", side_effect=spy):
            await _achunk_page_aware(rag, "F", tokenizer, content)

        assert mock_fn.called
        assert mock_fn not in seen, "chunking_by_fixed_token must run INSIDE to_thread, not be passed to it"
        assert len(seen) == 1

    async def test_v_strategy_never_passed_to_to_thread(self):
        tokenizer = _char_tokenizer()
        pages = [(1, "alpha beta " * 10), (2, "gamma delta " * 10)]
        marked = page_marked_text(pages)
        rag = _fake_rag(chunk_token_size=20)
        mock_v = AsyncMock(return_value=[])
        real_to_thread = asyncio.to_thread
        seen = []

        async def spy(func, *a, **kw):
            seen.append(func)
            return await real_to_thread(func, *a, **kw)

        with patch("lightrag.chunker.chunking_by_semantic_vector", mock_v), \
             patch("asyncio.to_thread", side_effect=spy):
            await _achunk_page_aware(rag, "V", tokenizer, marked)

        mock_v.assert_awaited_once()
        assert mock_v not in seen
        assert strip_page_markers in seen


# ---------------------------------------------------------------------------
# Unmarked-text / backward-compat regression
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    async def test_unmarked_text_f_has_no_chunk_id_and_matches_direct_call(self):
        from lightrag.chunker import chunking_by_fixed_token

        tokenizer = _char_tokenizer()
        content = "Some unmarked plain document text. " * 20
        rag = _fake_rag(chunk_token_size=40, fixed_token_overrides={"chunk_overlap_token_size": 5})
        size, opts = _resolve_strategy_options(rag, "F")

        direct = chunking_by_fixed_token(tokenizer, content, size, _emit_source_span=True, **opts)
        hooked = await _achunk_page_aware(rag, "F", tokenizer, content)

        assert [c["content"] for c in hooked] == [c["content"] for c in direct]
        assert all("chunk_id" not in c for c in hooked)


# ---------------------------------------------------------------------------
# Selector removal
# ---------------------------------------------------------------------------


def _store(**advanced):
    return LightRAGStore(
        db=MagicMock(),
        ai_service=SimpleNamespace(
            provider="OpenAI", name="t", description="gpt-4o", api_key="k", endpoint=None,
        ),
        embedding_service=SimpleNamespace(
            provider="OpenAI", name="e", description="d", api_key="k",
            endpoint=None, api_version=None,
        ),
        **advanced,
    )


def _patched_build_rag(store, collection_name="silo_1"):
    with patch("lightrag.LightRAG") as mock_lightrag_cls, \
         patch("tools.vector_stores.lightrag.adapters.build_embedding_func"), \
         patch("tools.vector_stores.lightrag.adapters.build_llm_model_func"), \
         patch("tools.vector_stores.lightrag.adapters.build_role_llm_configs"), \
         patch(
             "tools.vector_stores.lightrag.storage_config.build_storage_config",
             return_value={
                 "graph_storage": "Neo4JStorage",
                 "vector_storage": "QdrantVectorDBStorage",
                 "kv_storage": "PGKVStorage",
                 "doc_status_storage": "PGDocStatusStorage",
             },
         ):
        rag = store._build_rag(collection_name)
        return rag, mock_lightrag_cls


class TestSelectorRemoval:
    async def test_ainsert_has_no_process_options_or_chunk_options_kwarg(self):
        rag = SimpleNamespace(
            apipeline_enqueue_documents=AsyncMock(),
            apipeline_process_enqueue_documents=AsyncMock(),
        )
        await _ainsert(rag, ["a", "b"], file_paths=["f1", "f2"], ids=["i1", "i2"])
        rag.apipeline_enqueue_documents.assert_awaited_once_with(
            ["a", "b"], ids=["i1", "i2"], file_paths=["f1", "f2"]
        )

    async def test_aenqueue_has_no_process_options_or_chunk_options_kwarg(self):
        rag = SimpleNamespace(apipeline_enqueue_documents=AsyncMock())
        await _aenqueue(rag, ["a"], file_paths=["f1"], ids=["i1"])
        rag.apipeline_enqueue_documents.assert_awaited_once_with(
            ["a"], ids=["i1"], file_paths=["f1"]
        )

    def test_build_rag_chunking_func_delegates_to_achunk_page_aware(self):
        store = _store(lightrag_chunk_strategy="recursive_character")
        rag, mock_lightrag_cls = _patched_build_rag(store)
        chunking_func = mock_lightrag_cls.call_args.kwargs["chunking_func"]
        assert callable(chunking_func)

        with patch("tools.vector_stores.lightrag_store._achunk_page_aware") as mock_achunk:
            tokenizer = object()
            # The full legacy 6-positional-arg calling convention LightRAG
            # itself uses (see pipeline.py's legacy_args tuple).
            chunking_func(tokenizer, "some content", None, False, 100, 1200)

        mock_achunk.assert_called_once_with(rag, "R", tokenizer, "some content")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    async def test_same_input_produces_identical_chunk_ids(self):
        tokenizer = _char_tokenizer()
        pages = [
            (1, "alpha beta gamma " * 10),
            (2, "delta epsilon zeta " * 10),
            (3, "eta theta iota " * 10),
        ]
        marked = page_marked_text(pages)
        rag = _fake_rag(chunk_token_size=30, fixed_token_overrides={"chunk_overlap_token_size": 5})

        chunks1 = await _achunk_page_aware(rag, "F", tokenizer, marked)
        chunks2 = await _achunk_page_aware(rag, "F", tokenizer, marked)

        assert [c["chunk_id"] for c in chunks1] == [c["chunk_id"] for c in chunks2]
