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
