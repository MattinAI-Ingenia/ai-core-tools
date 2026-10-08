"""Locks the inline-citation SOURCES block appended to LightRAG tool output.

The numbering must be 1-based and mirror ``lightrag_raw_data.data.chunks`` order
1:1 — the frontend resolves ``cite://N`` against ``chunks[N - 1]``, so any drift
would open the wrong source. Non-LightRAG docs must pass through untouched.
"""
from langchain_core.documents import Document

from tools.agentTools import _append_lightrag_citation_sources, _CITATION_INSTRUCTION


def _lightrag_doc(chunks):
    return Document(
        page_content="fused context",
        metadata={"source": "lightrag", "lightrag_raw_data": {"data": {"chunks": chunks}}},
    )


def test_appends_numbered_sources_in_chunk_order():
    doc = _lightrag_doc([
        {"id": "c1", "content": "alpha content", "file_path": "a.pdf"},
        {"id": "c2", "content": "beta content", "file_path": "b.pdf"},
    ])
    out = _append_lightrag_citation_sources("ANSWER", [doc])

    assert out.startswith("ANSWER")
    assert _CITATION_INSTRUCTION in out
    # 1-based, in order — cite://1 -> first chunk, cite://2 -> second chunk.
    assert "[1] (source: a.pdf) alpha content" in out
    assert "[2] (source: b.pdf) beta content" in out
    assert out.index("[1] (source: a.pdf)") < out.index("[2] (source: b.pdf)")


def test_missing_file_path_falls_back_to_unknown_source():
    doc = _lightrag_doc([{"id": "c1", "content": "x", "file_path": ""}])
    out = _append_lightrag_citation_sources("A", [doc])
    assert "[1] (source: Unknown source) x" in out


def test_noop_without_lightrag_chunks():
    # PGVector/Qdrant docs carry no lightrag_raw_data — must be returned verbatim.
    plain = Document(page_content="text", metadata={"source": "pgvector"})
    assert _append_lightrag_citation_sources("A", [plain]) == "A"
    # No chunks list at all.
    empty = _lightrag_doc([])
    assert _append_lightrag_citation_sources("A", [empty]) == "A"


def test_offset_continues_numbering_across_calls_in_the_same_turn():
    """A turn with 2 retrieval calls (e.g. multi-silo) must number 1,2,3,4 —
    not restart at 1 on the second call — passing the SAME counter cell,
    same pattern as _call_count."""
    doc1 = _lightrag_doc([
        {"id": "a1", "content": "alpha", "file_path": "a.pdf"},
        {"id": "a2", "content": "beta", "file_path": "a.pdf"},
    ])
    doc2 = _lightrag_doc([{"id": "b1", "content": "gamma", "file_path": "b.pdf"}])

    offset = [0]
    out1 = _append_lightrag_citation_sources("PART A", [doc1], offset)
    out2 = _append_lightrag_citation_sources("PART B", [doc2], offset)

    assert "[1] (source: a.pdf) alpha" in out1
    assert "[2] (source: a.pdf) beta" in out1
    assert "[3] (source: b.pdf) gamma" in out2  # continues, doesn't restart at 1


def test_offset_omitted_keeps_old_single_call_behavior():
    doc = _lightrag_doc([{"id": "c1", "content": "x", "file_path": "a.pdf"}])
    assert "[1] (source: a.pdf) x" in _append_lightrag_citation_sources("A", [doc])


# --- LightRAG's own reference numbering must not compete with ours -------------
# Measured 2026-10-07 (9 traces): every citation mismatch came from the model citing
# with LightRAG's "Reference Document List" numbers instead of our SOURCES numbers
# (e.g. LightRAG [30] = CDOC001961 while our [30] was THERMAPRO).

_LIGHTRAG_CONTEXT = (
    "Content: Knowledge Graph Data (Entity):\n\n```json\n"
    '{"entity": "Función Anti-Hielo", "type": "función", "description": "protege"}\n```\n\n'
    "Document Chunks (Each entry has a reference_id refer to the `Reference Document List`; "
    "the optional `content_headings` field gives the chunk's heading path):\n\n```json\n"
    '{"reference_id": "1", "content": "Protección antihielo ..."}\n'
    '{"reference_id": "2", "content": "Precaución contra heladas ..."}\n```\n\n'
    "Reference Document List (Each entry starts with a [reference_id] that corresponds to "
    "entries in the Document Chunks):\n\n```\n[1] DSAT000008.pdf (p. 20)\n[2] CDOC001961.pdf (p. 23)\n```\n"
    'Metadata: {"source": "lightrag", "query_mode": "global"}'
)


def _lightrag_docs():
    return [Document(page_content="x", metadata={"lightrag_raw_data": {"data": {"chunks": [
        {"file_path": "DSAT000008.pdf (p. 20)", "content": "a"},
        {"file_path": "CDOC001961.pdf (p. 23)", "content": "b"},
    ]}}})]


def test_lightrag_reference_list_and_ids_are_removed_from_what_the_model_sees():
    out = _append_lightrag_citation_sources(_LIGHTRAG_CONTEXT, _lightrag_docs(), [40])

    assert "Reference Document List" not in out
    assert "reference_id" not in out
    assert "[1] DSAT000008.pdf (p. 20)" not in out
    # our own numbering is the only one left, continuing from the offset
    assert "[41] (source: DSAT000008.pdf (p. 20))" in out
    assert "[42] (source: CDOC001961.pdf (p. 23))" in out


def test_chunk_json_stays_valid_and_graph_context_untouched():
    import json
    import re as _re

    out = _append_lightrag_citation_sources(_LIGHTRAG_CONTEXT, _lightrag_docs(), [0])

    chunk_lines = [l for l in out.splitlines() if l.startswith('{"content"')]
    assert [json.loads(l)["content"] for l in chunk_lines] == ["Protección antihielo ...", "Precaución contra heladas ..."]
    assert '"entity": "Función Anti-Hielo"' in out
    assert not _re.search(r"Reference Document List|reference_id", out)


def test_non_lightrag_content_is_left_alone():
    plain = Document(page_content="p", metadata={})
    assert _append_lightrag_citation_sources(_LIGHTRAG_CONTEXT, [plain], [0]) == _LIGHTRAG_CONTEXT
