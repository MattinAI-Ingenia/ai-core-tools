"""Union of the literal coverage hits with the semantic hits, per document.

Why it exists: the literal search only matches the exact wording, so a manual
that writes "EN 303- 5" (extra space) or "pressure relief valve" is missed.
Semantic search finds those, but only returns a handful of chunks. The union
keeps every literal document (the evidence is the phrase itself) and adds the
semantic-only ones the reranker scores as relevant, flagged as such.
"""
import asyncio

from tools.coverage_union import augment_with_semantic


def _run(coro):
    return asyncio.run(coro)


def _chunk(rid, text, page=1, name=None):
    return {"resource_id": rid, "file_path": name or f"DOC{rid}.pdf (p. {page})", "content": text, "page": page}


def _scores(by_text):
    """A fake rerank function: relevance by exact text, 0.0 for unknown text."""
    async def rerank(query, documents):
        return [{"index": i, "relevance_score": by_text.get(d, 0.0)} for i, d in enumerate(documents)]
    return rerank


def _semantic(chunks):
    async def fn(term):
        return chunks
    return fn


def _literal():
    return {1: [("DOC1.pdf (p. 5)", "weak mention", 5), ("DOC1.pdf (p. 9)", "strong mention", 9)],
            2: [("DOC2.pdf (p. 3)", "only mention", 3)]}


def test_adds_a_semantic_only_document_and_flags_it():
    grouped, added = _run(augment_with_semantic(
        _literal(), "EN 303-5",
        semantic_chunks_fn=_semantic([_chunk(3, "Boiler class (according to EN 303- 5)")]),
        rerank_fn=_scores({"Boiler class (according to EN 303- 5)": 0.9}),
    ))
    assert added == {3: "semantic"}
    assert list(grouped) == [1, 2, 3]
    assert grouped[3][0][1] == "Boiler class (according to EN 303- 5)"


def test_literal_documents_are_never_dropped_or_reordered_even_with_low_scores():
    grouped, added = _run(augment_with_semantic(
        _literal(), "x", semantic_chunks_fn=_semantic([]), rerank_fn=_scores({}),
    ))
    assert list(grouped) == [1, 2] and added == {}


def test_the_best_scoring_snippet_comes_first_for_each_document():
    grouped, _ = _run(augment_with_semantic(
        _literal(), "x", semantic_chunks_fn=_semantic([]),
        rerank_fn=_scores({"weak mention": 0.2, "strong mention": 0.8, "only mention": 0.5}),
    ))
    assert [s[1] for s in grouped[1]] == ["strong mention", "weak mention"]


def test_semantic_documents_below_the_threshold_are_left_out():
    grouped, added = _run(augment_with_semantic(
        _literal(), "x", semantic_chunks_fn=_semantic([_chunk(3, "unrelated")]),
        rerank_fn=_scores({"unrelated": 0.01}),
    ))
    assert 3 not in grouped and added == {}


def test_only_the_best_scoring_extra_documents_are_kept_when_there_are_too_many():
    chunks = [_chunk(10 + i, f"extra {i}") for i in range(5)]
    grouped, added = _run(augment_with_semantic(
        {}, "x", semantic_chunks_fn=_semantic(chunks),
        rerank_fn=_scores({f"extra {i}": 0.5 + i / 10 for i in range(5)}), max_added=2,
    ))
    assert set(added) == {13, 14}, "the two highest scores"


def test_a_semantic_chunk_of_a_document_already_found_literally_adds_nothing_new():
    grouped, added = _run(augment_with_semantic(
        _literal(), "x", semantic_chunks_fn=_semantic([_chunk(1, "another chunk of doc 1")]),
        rerank_fn=_scores({"another chunk of doc 1": 0.9}),
    ))
    assert added == {} and len(grouped[1]) == 2, "no extra snippet, no new document"


def test_without_a_reranker_the_literal_result_is_returned_untouched():
    literal = _literal()
    grouped, added = _run(augment_with_semantic(
        literal, "x", semantic_chunks_fn=_semantic([_chunk(3, "t")]), rerank_fn=None,
    ))
    assert grouped == literal and added == {}


def test_a_failing_semantic_search_or_reranker_falls_back_to_the_literal_result():
    async def boom(*_a, **_k):
        raise RuntimeError("down")

    literal = _literal()
    g1, a1 = _run(augment_with_semantic(literal, "x", semantic_chunks_fn=boom, rerank_fn=_scores({})))
    g2, a2 = _run(augment_with_semantic(
        literal, "x", semantic_chunks_fn=_semantic([_chunk(3, "t")]), rerank_fn=boom,
    ))
    assert g1 == literal and a1 == {} and g2 == literal and a2 == {}


def test_string_resource_ids_from_the_literal_search_match_the_integer_ids_of_semantic_chunks():
    """find_chunks_mentioning keys its result by str ("311"); the semantic chunk
    ids are ints. Comparing them raw would re-add every literal document as a
    semantic-only one."""
    literal = {"311": [("DOC311.pdf (p. 2)", "literal hit", 2)]}
    grouped, added = _run(augment_with_semantic(
        literal, "x", semantic_chunks_fn=_semantic([_chunk(311, "same doc, semantic")]),
        rerank_fn=_scores({"literal hit": 0.9, "same doc, semantic": 0.9}),
    ))
    assert added == {} and len(grouped) == 1


# --- two tiers for the semantic-only documents --------------------------------
# A score cut alone drops mentions the reranker undervalues (a 3-line "Anti-frost
# function" section inside a 2000-token chunk scores 0.01-0.04). Between the two
# thresholds a document is shown as "possible", with one snippet, and the agent
# reads it and decides, instead of a number deciding for it.

def test_a_document_between_the_two_thresholds_is_listed_as_possible_with_one_snippet():
    chunks = [_chunk(3, "dim mention", 1), _chunk(3, "another dim mention", 2)]
    grouped, added = _run(augment_with_semantic(
        _literal(), "x", semantic_chunks_fn=_semantic(chunks),
        rerank_fn=_scores({"dim mention": 0.12, "another dim mention": 0.10}),
    ))
    assert added == {3: "possible"}
    assert [s[1] for s in grouped[3]] == ["dim mention"], "only the best snippet"


def test_a_document_above_the_semantic_threshold_is_tier_semantic():
    _, added = _run(augment_with_semantic(
        _literal(), "x", semantic_chunks_fn=_semantic([_chunk(3, "good")]),
        rerank_fn=_scores({"good": 0.8}),
    ))
    assert added == {3: "semantic"}


def test_possible_documents_are_capped_and_the_best_ones_kept():
    chunks = [_chunk(20 + i, f"p{i}") for i in range(6)]
    _, added = _run(augment_with_semantic(
        {}, "x", semantic_chunks_fn=_semantic(chunks),
        rerank_fn=_scores({f"p{i}": 0.06 + i / 100 for i in range(6)}), max_possible=2,
    ))
    assert set(added) == {24, 25} and set(added.values()) == {"possible"}


def test_no_candidates_means_no_rerank_call():
    """The rerank API rejects an empty document list (422) and its retry policy
    waits before failing: with nothing literal and nothing semantic, don't call it."""
    calls = []

    async def rerank(query, documents):
        calls.append(documents)
        return []

    grouped, added = _run(augment_with_semantic(
        {}, "x", semantic_chunks_fn=_semantic([]), rerank_fn=rerank,
    ))
    assert calls == [] and grouped == {} and added == {}


# --- excerpts: show the reader the part of the page that matters --------------
# The LLM used to get the first 240 characters of each page. The mention can be
# further down, so a "possible" document was judged on text that did not contain it.

from tools.coverage_union import excerpt_around  # noqa: E402


def _page(before, needle, after=""):
    return ("intro " * before) + needle + (" outro" * after)


def test_the_excerpt_is_centred_on_the_first_occurrence_of_the_term():
    ex = excerpt_around(_page(200, "NEEDLE here", 200), "NEEDLE", width=300)
    assert "NEEDLE" in ex and len(ex) <= 300 + 2  # plus the two ellipses
    assert ex.startswith("…") and ex.endswith("…")


def test_the_term_is_found_across_a_line_break_or_hyphen_like_the_search_does():
    page = ("x " * 300) + "Boiler class (according to EN 303-\n5) Clase 5" + (" y" * 300)
    assert "303-" in excerpt_around(page, "EN 303-5", width=300)


def test_without_a_match_the_excerpt_is_the_start_of_the_page():
    ex = excerpt_around("Start of the page " + "filler " * 100, "absent term", width=60)
    assert ex.startswith("Start of the page") and not ex.startswith("…")


def test_a_short_page_is_returned_whole_without_ellipses():
    assert excerpt_around("tiny page with the term", "term", width=300) == "tiny page with the term"


def test_semantic_only_documents_get_their_best_window_as_excerpt():
    needle = "ANTI-FROST FUNCTION protects the boiler"
    long_page = ("unrelated words " * 60) + needle + (" other unrelated words" * 60)

    async def rerank(query, documents):  # scores a window by whether it holds the needle
        return [{"index": i, "relevance_score": 0.9 if "ANTI-FROST" in d else 0.01}
                for i, d in enumerate(documents)]

    excerpts = {}
    _run(augment_with_semantic(
        {}, "antifreeze", semantic_chunks_fn=_semantic([_chunk(3, long_page, 7, name="DOC3.pdf (p. 7)")]),
        rerank_fn=lambda q, d: rerank(q, d) if len(d) > 1 else _scores({long_page[:1500]: 0.8})(q, d),
        excerpts_out=excerpts,
    ))
    assert "ANTI-FROST" in excerpts["DOC3.pdf (p. 7)"]
    assert len(excerpts["DOC3.pdf (p. 7)"]) <= 300 + 2


def test_a_failing_window_rerank_leaves_the_excerpt_out_not_the_document():
    calls = {"n": 0}

    async def rerank(query, documents):
        calls["n"] += 1
        if calls["n"] == 1:
            return [{"index": 0, "relevance_score": 0.9}]
        raise RuntimeError("down")

    excerpts = {}
    grouped, added = _run(augment_with_semantic(
        {}, "x", semantic_chunks_fn=_semantic([_chunk(3, "word " * 400, 1, name="DOC3.pdf (p. 1)")]),
        rerank_fn=rerank, excerpts_out=excerpts,
    ))
    assert added == {3: "semantic"} and excerpts == {}


def test_the_citation_block_uses_the_excerpt_when_a_chunk_has_one():
    from tools.agentTools import _append_lightrag_citation_sources
    from langchain_core.documents import Document

    doc = Document(page_content="x", metadata={"lightrag_raw_data": {"data": {"chunks": [
        {"file_path": "A.pdf (p. 1)", "content": "start of the page " * 40, "excerpt": "…the middle part…"},
        {"file_path": "B.pdf (p. 2)", "content": "plain page " * 40},
    ]}}})
    out = _append_lightrag_citation_sources("body", [doc])
    assert "…the middle part…" in out and "start of the page" not in out
    assert "[2] (source: B.pdf (p. 2)) plain page" in out, "chunks without an excerpt behave as before"
