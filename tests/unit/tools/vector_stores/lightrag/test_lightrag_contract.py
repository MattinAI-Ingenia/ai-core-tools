"""Contract test pinning lightrag-hku 1.5.6.

Fails loudly on a lightrag-hku bump; every assertion is an assumption the
page-aware chunking / progress code relies on.
"""

from __future__ import annotations

import inspect

import pytest

lightrag = pytest.importorskip("lightrag")

pytestmark = pytest.mark.asyncio


def _char_tokenizer():
    from lightrag.utils import Tokenizer

    class _CharTokenModel:
        def encode(self, s):
            return [ord(c) for c in s]

        def decode(self, t):
            return "".join(map(chr, t))

    return Tokenizer("char", _CharTokenModel())


# ---------------------------------------------------------------------------
# 1. Chunker signatures
# ---------------------------------------------------------------------------


class TestChunkerSignatures:
    def test_chunking_by_fixed_token(self):
        from lightrag.chunker import chunking_by_fixed_token

        sig = inspect.signature(chunking_by_fixed_token)
        params = sig.parameters
        names = list(params)
        assert names[:3] == ["tokenizer", "content", "chunk_token_size"]
        assert params["chunk_token_size"].default == 1200
        for kw, default in (
            ("chunk_overlap_token_size", 100),
            ("split_by_character", None),
            ("split_by_character_only", False),
            ("_emit_source_span", False),
        ):
            assert kw in params, kw
            assert params[kw].kind == inspect.Parameter.KEYWORD_ONLY, kw
            assert params[kw].default == default, kw

    def test_chunking_by_recursive_character(self):
        from lightrag.chunker import chunking_by_recursive_character

        sig = inspect.signature(chunking_by_recursive_character)
        params = sig.parameters
        names = list(params)
        assert names[:3] == ["tokenizer", "content", "chunk_token_size"]
        assert params["chunk_token_size"].default == 1200
        for kw, default in (
            ("chunk_overlap_token_size", 100),
            ("separators", None),
        ):
            assert kw in params, kw
            assert params[kw].kind == inspect.Parameter.KEYWORD_ONLY, kw
            assert params[kw].default == default, kw

    def test_chunking_by_semantic_vector(self):
        from lightrag.chunker import chunking_by_semantic_vector

        assert inspect.iscoroutinefunction(chunking_by_semantic_vector)
        sig = inspect.signature(chunking_by_semantic_vector)
        assert "embedding_func" in sig.parameters

    def test_chunking_by_paragraph_semantic(self):
        from lightrag.chunker import chunking_by_paragraph_semantic

        sig = inspect.signature(chunking_by_paragraph_semantic)
        params = sig.parameters
        names = list(params)
        assert names[:3] == ["tokenizer", "content", "chunk_token_size"]
        assert params["chunk_token_size"].default == 2000
        for kw in ("blocks_path", "chunk_overlap_token_size", "doc_id"):
            assert kw in params, kw

    def test_other_importables(self):
        from lightrag.parser.routing import resolve_chunk_options
        from lightrag.chunker.recursive_character import normalize_r_separators
        from lightrag.utils_pipeline import build_chunks_dict_from_chunking_result
        from lightrag.utils import sanitize_text_for_encoding

        assert "process_options" in inspect.signature(resolve_chunk_options).parameters
        assert callable(normalize_r_separators)
        assert callable(build_chunks_dict_from_chunking_result)
        assert callable(sanitize_text_for_encoding)

    def test_lightrag_dataclass_fields(self):
        import dataclasses
        from lightrag import LightRAG

        field_names = {f.name for f in dataclasses.fields(LightRAG)}
        for name in ("chunking_func", "chunk_token_size", "chunk_overlap_token_size"):
            assert name in field_names, name


# ---------------------------------------------------------------------------
# 2. _source_span
# ---------------------------------------------------------------------------


def _three_page_text():
    pages = [
        "Page one. " + ("lorem ipsum dolor sit amet. " * 35),
        "Page two. " + ("consectetur adipiscing elit. " * 35),
        "Page three. " + ("sed do eiusmod tempor incididunt. " * 35),
    ]
    return "\n\n".join(pages)


class TestSourceSpan:
    def test_fixed_token_source_span(self):
        from lightrag.chunker import chunking_by_fixed_token

        tokenizer = _char_tokenizer()
        text = _three_page_text()
        chunks = chunking_by_fixed_token(
            tokenizer, text, 200, chunk_overlap_token_size=20, _emit_source_span=True
        )
        assert chunks
        for chunk in chunks:
            span = chunk["_source_span"]
            start = span["start"]
            assert isinstance(start, int)
            content = chunk["content"]
            assert text[start : start + len(content)] == content

    def test_recursive_character_source_span(self):
        from lightrag.chunker import chunking_by_recursive_character

        tokenizer = _char_tokenizer()
        text = _three_page_text()
        chunks = chunking_by_recursive_character(
            tokenizer, text, 200, chunk_overlap_token_size=20
        )
        assert chunks
        for chunk in chunks:
            span = chunk["_source_span"]
            start = span["start"]
            assert isinstance(start, int)
            content = chunk["content"]
            assert text[start : start + len(content)] == content


# ---------------------------------------------------------------------------
# 3. Marker survives the sanitizer
# ---------------------------------------------------------------------------


class TestSanitizerPreservesMarkers:
    def test_sanitize_text_for_encoding(self):
        from lightrag.utils import sanitize_text_for_encoding

        text = "<<<page:1>>>\nabc\n\n<<<page:2>>>\ndef"
        assert sanitize_text_for_encoding(text) == text


# ---------------------------------------------------------------------------
# 4. {doc_id}- prefixing
# ---------------------------------------------------------------------------


class TestDocIdPrefixing:
    def test_build_chunks_dict_from_chunking_result(self):
        from lightrag.utils_pipeline import build_chunks_dict_from_chunking_result

        result = build_chunks_dict_from_chunking_result(
            [
                {
                    "content": "x",
                    "tokens": 1,
                    "chunk_order_index": 0,
                    "chunk_id": "p3-chunk-000",
                    "_source_span": {"start": 0, "end": 1},
                }
            ],
            doc_id="res7",
            file_path="m.pdf",
        )
        assert "res7-p3-chunk-000" in result
        entry = result["res7-p3-chunk-000"]
        assert entry["full_doc_id"] == "res7"
        assert entry["file_path"] == "m.pdf"
        assert "_source_span" not in entry

    def test_already_prefixed_id_is_not_double_prefixed(self):
        from lightrag.utils_pipeline import build_chunks_dict_from_chunking_result

        result = build_chunks_dict_from_chunking_result(
            [
                {
                    "content": "x",
                    "tokens": 1,
                    "chunk_order_index": 0,
                    "chunk_id": "res7-p3-chunk-000",
                }
            ],
            doc_id="res7",
            file_path="m.pdf",
        )
        assert "res7-p3-chunk-000" in result
        assert "res7-res7-p3-chunk-000" not in result

    def test_enforce_chunk_token_limit_before_embedding(self):
        # Confirmed importable on lightrag-hku 1.5.6 -- a plain import, so a
        # future bump that removes/renames it fails this test loudly instead
        # of skipping it green.
        from lightrag.utils import enforce_chunk_token_limit_before_embedding

        tokenizer = _char_tokenizer()
        chunk = {
            "content": "alpha beta gamma delta epsilon zeta eta theta",
            "tokens": 46,
            "chunk_order_index": 0,
            "chunk_id": "p3-chunk-000",
        }
        children = enforce_chunk_token_limit_before_embedding(
            [chunk], tokenizer=tokenizer, max_tokens=10
        )
        assert len(children) > 1
        for child in children:
            assert child["chunk_id"].startswith("p3-chunk-000-s")


# ---------------------------------------------------------------------------
# 5. Legacy hook end-to-end (hermetic pipeline)
# ---------------------------------------------------------------------------


async def _make_rag(tmp_path, chunking_func, llm_model_func=None):
    import numpy as np
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    async def fake_embedding(texts):
        return np.zeros((len(texts), 8))

    calls = {"n": 0}

    async def default_llm(prompt, *args, **kwargs):
        calls["n"] += 1
        return '{"entities": [], "relationships": []}'

    # lightrag.kg.shared_storage keys pipeline status by workspace in
    # process-global dicts, not by working_dir -- give every call (so every
    # test) its own workspace to avoid cross-test state leakage.
    workspace = f"contract-{tmp_path.name}"

    rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=workspace,
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        doc_status_storage="JsonDocStatusStorage",
        embedding_func=EmbeddingFunc(embedding_dim=8, max_token_size=8192, func=fake_embedding),
        llm_model_func=llm_model_func or default_llm,
        entity_extraction_use_json=True,
        chunking_func=chunking_func,
    )
    await rag.initialize_storages()

    from lightrag.kg.shared_storage import initialize_pipeline_status

    await initialize_pipeline_status()
    return rag, calls


class TestLegacyHookEndToEnd:
    async def test_six_positional_args_and_prefixing(self, tmp_path):
        spy_calls = []

        async def spy(*args, **kwargs):
            spy_calls.append((args, kwargs))
            return [
                {
                    "content": "alpha beta gamma",
                    "tokens": 3,
                    "chunk_order_index": 0,
                    "chunk_id": "p3-chunk-000",
                }
            ]

        rag, llm_calls = await _make_rag(tmp_path, spy)

        input_text = "<<<page:3>>>\nalpha beta gamma"
        await rag.apipeline_enqueue_documents(
            [input_text], ids=["res7"], file_paths=["m.pdf"]
        )
        await rag.apipeline_process_enqueue_documents()

        assert len(spy_calls) == 1
        args, kwargs = spy_calls[0]
        assert kwargs == {}
        assert len(args) == 6
        assert args[1] == input_text

        chunk = await rag.text_chunks.get_by_id("res7-p3-chunk-000")
        assert chunk is not None
        assert chunk["full_doc_id"] == "res7"
        assert chunk["file_path"] == "m.pdf"

        status = await rag.doc_status.get_by_id("res7")
        assert status is not None
        assert status["status"] == "processed"
        assert status["chunks_count"] == 1

        llm_cache_list = chunk.get("llm_cache_list")
        assert llm_cache_list, "llm_cache_list must be non-empty after extraction"

        # Second doc, same chunk content -> extraction should hit the LLM cache.
        calls_before_second = llm_calls["n"]
        await rag.apipeline_enqueue_documents(
            ["<<<page:1>>>\ndifferent text"], ids=["res8"], file_paths=["n.pdf"]
        )
        await rag.apipeline_process_enqueue_documents()

        chunk2 = await rag.text_chunks.get_by_id("res8-p3-chunk-000")
        assert chunk2 is not None
        assert chunk2.get("llm_cache_list"), "llm_cache_list must be non-empty for the 2nd doc"
        assert llm_calls["n"] == calls_before_second, "the 2nd extraction should be a cache hit"
