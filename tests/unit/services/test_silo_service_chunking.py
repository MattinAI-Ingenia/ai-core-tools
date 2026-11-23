"""Unit tests for chunking in SiloService: extract_documents_from_file (FR-10) and index_single_content (FR-11).

All external dependencies are mocked — no DB, filesystem, or vector store access.
"""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from tools.pdf_loader import PdfLoader
from services.silo_service import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SiloService,
)


def _make_silo(silo_id: int = 7) -> MagicMock:
    silo = MagicMock()
    silo.silo_id = silo_id
    silo.embedding_service_id = 1
    silo.embedding_service = MagicMock(name="mock-embedding-service", end_of_service_at=None)
    return silo


def _make_vector_store() -> MagicMock:
    vs = MagicMock()
    vs.index_documents.return_value = None
    return vs


def _long_text(n_chars: int = CHUNK_SIZE * 3) -> str:
    """Return a realistic multi-paragraph text longer than CHUNK_SIZE."""
    paragraph = (
        "The quick brown fox jumped over the lazy dog near the riverbank. "
        "Scientists discovered a new species in the Amazon rainforest yesterday. "
    )
    full = "\n\n".join([paragraph.strip()] * (n_chars // len(paragraph) + 2))
    return full[:n_chars]


class TestDomainContentChunking:
    """FR-11: Long web-page text must be chunked; metadata propagated to every chunk."""

    def _call_index_single(self, content: str, metadata: dict, silo_id: int = 7):
        silo = _make_silo(silo_id)
        vs = _make_vector_store()
        db = MagicMock()

        with (
            patch("services.silo_service.SiloRepository.get_by_id", return_value=silo),
            patch("services.silo_service.SiloRepository.get_embedding_service_by_id", return_value=silo.embedding_service),
            patch("services.silo_service._get_vector_store", return_value=vs),
        ):
            SiloService.index_single_content(silo_id, content, metadata, db)

        assert vs.index_documents.called, "index_documents was never called"
        call_args = vs.index_documents.call_args
        indexed_docs: list[Document] = call_args[0][1]
        return indexed_docs

    def test_long_content_produces_multiple_chunks(self):
        """A text longer than CHUNK_SIZE must be split into >1 chunk."""
        content = _long_text(CHUNK_SIZE * 3)
        metadata = {"url": "https://example.com/page", "domain_id": 42}

        docs = self._call_index_single(content, metadata, silo_id=7)

        assert len(docs) > 1, (
            f"Expected >1 chunk for {len(content)}-char text with CHUNK_SIZE={CHUNK_SIZE}, "
            f"got {len(docs)}"
        )

    def test_all_chunks_carry_url_metadata(self):
        """Every chunk produced from domain content must include the 'url' key."""
        content = _long_text(CHUNK_SIZE * 3)
        metadata = {"url": "https://example.com/page", "domain_id": 42}

        docs = self._call_index_single(content, metadata, silo_id=7)

        for i, doc in enumerate(docs):
            assert "url" in doc.metadata, f"Chunk {i} missing 'url' in metadata"
            assert doc.metadata["url"] == "https://example.com/page", (
                f"Chunk {i} has wrong 'url': {doc.metadata['url']}"
            )

    def test_all_chunks_carry_domain_id_metadata(self):
        """Every chunk produced from domain content must include the 'domain_id' key."""
        content = _long_text(CHUNK_SIZE * 3)
        metadata = {"url": "https://example.com/page", "domain_id": 42}

        docs = self._call_index_single(content, metadata, silo_id=7)

        for i, doc in enumerate(docs):
            assert "domain_id" in doc.metadata, f"Chunk {i} missing 'domain_id'"
            assert doc.metadata["domain_id"] == 42, (
                f"Chunk {i} has wrong 'domain_id': {doc.metadata['domain_id']}"
            )

    def test_all_chunks_carry_silo_id_metadata(self):
        """Every chunk produced must include 'silo_id' matching the target silo."""
        content = _long_text(CHUNK_SIZE * 3)
        metadata = {"url": "https://example.com/page", "domain_id": 42}

        docs = self._call_index_single(content, metadata, silo_id=7)

        for i, doc in enumerate(docs):
            assert "silo_id" in doc.metadata, f"Chunk {i} missing 'silo_id'"
            assert doc.metadata["silo_id"] == 7, (
                f"Chunk {i} has wrong 'silo_id': {doc.metadata['silo_id']}"
            )

    def test_chunk_content_is_non_empty(self):
        """No chunk should be empty after splitting."""
        content = _long_text(CHUNK_SIZE * 3)
        metadata = {"url": "https://example.com/page", "domain_id": 42}

        docs = self._call_index_single(content, metadata, silo_id=7)

        for i, doc in enumerate(docs):
            assert doc.page_content.strip(), f"Chunk {i} has empty page_content"

    def test_short_content_produces_single_chunk(self):
        """Text shorter than CHUNK_SIZE must produce exactly 1 chunk with intact metadata."""
        short_content = "This is a short web-page snippet that fits in one chunk."
        assert len(short_content) < CHUNK_SIZE
        metadata = {"url": "https://example.com/short", "domain_id": 99}

        docs = self._call_index_single(short_content, metadata, silo_id=5)

        assert len(docs) == 1
        assert docs[0].metadata["url"] == "https://example.com/short"
        assert docs[0].metadata["domain_id"] == 99
        assert docs[0].metadata["silo_id"] == 5
        assert docs[0].page_content == short_content

    def test_chunk_at_exact_size_boundary_is_not_resplit(self):
        """Idempotency invariant: a chunk of exactly CHUNK_SIZE chars produces 1 chunk.

        The file-upload path re-feeds already-split chunks (each <= CHUNK_SIZE)
        through this splitter; this test pins the boundary so a future change to
        the constants or the splitter class cannot silently break that invariant.
        """
        content_at_limit = "A" * CHUNK_SIZE
        metadata = {"url": "https://example.com/boundary", "domain_id": 1}

        docs = self._call_index_single(content_at_limit, metadata, silo_id=7)

        assert len(docs) == 1, (
            f"A chunk of exactly CHUNK_SIZE={CHUNK_SIZE} chars must not be re-split, "
            f"got {len(docs)} chunks"
        )
        assert docs[0].page_content == content_at_limit

    def test_caller_metadata_cannot_override_silo_id(self):
        """A 'silo_id' key inside caller metadata must not override the silo_id parameter."""
        metadata = {"url": "https://example.com/x", "silo_id": 999}

        docs = self._call_index_single("Some short content.", metadata, silo_id=7)

        assert docs[0].metadata["silo_id"] == 7

    def test_index_multiple_content_chunks_each_document(self):
        """index_multiple_content with two long documents must produce >2 total chunks."""
        silo = _make_silo(3)
        vs = _make_vector_store()
        db = MagicMock()

        long_text = _long_text(CHUNK_SIZE * 2)
        documents = [
            {"content": long_text, "metadata": {"url": "https://example.com/a", "domain_id": 1}},
            {"content": long_text, "metadata": {"url": "https://example.com/b", "domain_id": 1}},
        ]

        with (
            patch("services.silo_service.SiloRepository.get_by_id", return_value=silo),
            patch("services.silo_service.SiloRepository.get_embedding_service_by_id", return_value=silo.embedding_service),
            patch("services.silo_service._get_vector_store", return_value=vs),
        ):
            SiloService.index_multiple_content(3, documents, db)

        indexed_docs = vs.index_documents.call_args[0][1]
        assert len(indexed_docs) > 2, (
            f"Expected >2 total chunks from 2 long documents, got {len(indexed_docs)}"
        )

    def test_metadata_isolation_between_documents(self):
        """Chunks from document A must not carry the URL of document B."""
        silo = _make_silo(3)
        vs = _make_vector_store()
        db = MagicMock()

        long_text = _long_text(CHUNK_SIZE * 2)
        documents = [
            {"content": long_text, "metadata": {"url": "https://example.com/a", "domain_id": 10}},
            {"content": long_text, "metadata": {"url": "https://example.com/b", "domain_id": 20}},
        ]

        with (
            patch("services.silo_service.SiloRepository.get_by_id", return_value=silo),
            patch("services.silo_service.SiloRepository.get_embedding_service_by_id", return_value=silo.embedding_service),
            patch("services.silo_service._get_vector_store", return_value=vs),
        ):
            SiloService.index_multiple_content(3, documents, db)

        indexed_docs = vs.index_documents.call_args[0][1]
        urls = {doc.metadata["url"] for doc in indexed_docs}
        assert urls == {"https://example.com/a", "https://example.com/b"}
        for doc in indexed_docs:
            if doc.metadata["url"] == "https://example.com/a":
                assert doc.metadata["domain_id"] == 10
            else:
                assert doc.metadata["domain_id"] == 20


class TestExtractDocumentsFromFileSplitter:
    """PDF blocks are reconstructed into page- and section-aware chunks."""

    def test_pdf_loader_implements_langchain_base_loader(self):
        assert issubclass(PdfLoader, BaseLoader)

    @staticmethod
    def _make_block(text: str, y: float = 72, size: float = 11, bold: bool = False) -> dict:
        lines = []
        for line_number, line_text in enumerate(text.splitlines() or [text]):
            top = y + line_number * 12
            lines.append({
                "bbox": (72, top, 400, top + 10),
                "spans": [{
                    "text": line_text,
                    "size": size,
                    "flags": 16 if bold else 0,
                }],
            })
        return {"type": 0, "lines": lines}

    class _FakePdf:
        def __init__(self, pages: list[list[dict]]):
            self.pages = pages

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def __iter__(self):
            for blocks in self.pages:
                page = MagicMock()
                page.get_text.return_value = {"blocks": blocks}
                yield page

    def _extract(self, pages: list[list[dict]], base_metadata: dict | None = None):
        fake_pdf = self._FakePdf(pages)
        with patch("fitz.open", return_value=fake_pdf):
            return SiloService.extract_documents_from_file(
                file_path="fake.pdf",
                file_extension=".pdf",
                base_metadata=base_metadata,
            )

    def test_visual_line_breaks_are_joined_within_a_pdf_block(self):
        docs = self._extract([[
            self._make_block("A visual line ends here\nand continues on the next line."),
        ]])

        assert len(docs) == 1
        assert docs[0].page_content == (
            "A visual line ends here and continues on the next line."
        )

    def test_paragraphs_are_grouped_without_crossing_section_boundaries(self):
        first_paragraph = "First paragraph. " * 12
        section_heading = "Second section"
        second_paragraph = "Second section content. " * 8
        docs = self._extract([[
            self._make_block(first_paragraph),
            self._make_block(section_heading, y=100, size=16, bold=True),
            self._make_block(second_paragraph, y=130),
        ]])

        assert len(docs) == 2
        assert first_paragraph.strip() in docs[0].page_content
        assert section_heading in docs[1].page_content
        assert second_paragraph.strip() in docs[1].page_content
        assert "section" not in docs[0].metadata
        assert docs[1].metadata["section"] == section_heading

    def test_oversized_paragraph_is_kept_intact_above_target(self):
        sentence_a = "Alpha " * 100 + "ends here."
        sentence_b = "Beta " * 100 + "ends there."
        paragraph = f"{sentence_a} {sentence_b}"
        docs = self._extract([[self._make_block(paragraph)]])

        assert len(docs) == 1
        assert len(docs[0].page_content) > CHUNK_SIZE
        assert docs[0].page_content == paragraph

    def test_single_long_sentence_is_not_split_without_embedding_limit(self):
        paragraph = "meaningful " * (CHUNK_SIZE * 2)
        docs = self._extract([[self._make_block(paragraph)]])

        assert len(docs) == 1
        assert docs[0].page_content == paragraph.strip()

    def test_complete_paragraphs_are_grouped_up_to_target_size(self):
        first = "First paragraph has a complete idea and ends clearly."
        second = "Second paragraph carries another complete idea."
        docs = self._extract([[
            self._make_block(first),
            self._make_block(second, y=100),
        ]])

        assert len(docs) == 1
        assert docs[0].page_content == f"{first}\n{second}"
        assert docs[0].metadata["pdf_blocks"] == [0, 1]

    def test_second_pass_preserves_a_pdf_unit_above_target_size(self):
        text = "A complete PDF text block. " * 60
        docs = self._extract([[self._make_block(text)]])

        indexed = SiloService._create_documents_for_indexing(
            4,
            [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs],
        )

        assert len(indexed) == 1
        assert indexed[0].page_content == docs[0].page_content
        assert len(indexed[0].page_content) > CHUNK_SIZE

    def test_page_section_and_base_metadata_survive_second_chunking_pass(self):
        docs = self._extract(
            [
                [
                    self._make_block("First page body. " * 4),
                ],
                [
                    self._make_block("Overview", y=100, size=16, bold=True),
                    self._make_block("Second page section body. " * 4, y=130),
                ],
            ],
            base_metadata={"resource_id": 42, "silo_id": 9},
        )
        indexed = SiloService._create_documents_for_indexing(
            9,
            [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs],
        )

        assert len(indexed) == len(docs)
        assert [doc.page_content for doc in indexed] == [doc.page_content for doc in docs]
        assert [doc.metadata["page"] for doc in indexed] == [1, 2]
        assert all(doc.metadata["resource_id"] == 42 for doc in indexed)
        assert all(doc.metadata["silo_id"] == 9 for doc in indexed)
        assert indexed[-1].metadata["section"] == "Overview"
        assert indexed[-1].metadata["pdf_blocks"] == [0, 1]
        assert all("_aict_structured_pdf_chunk" not in doc.metadata for doc in indexed)

    def test_pdf_loader_fallback_returns_unsplit_pages(self):
        fake_pdf = MagicMock()
        fake_pdf.__enter__.return_value = fake_pdf
        fake_pdf.__exit__.return_value = False
        fake_pdf.__iter__.return_value = iter([
            MagicMock(get_text=MagicMock(return_value={"not_blocks": []})),
        ])
        loaded_page = Document(
            page_content="Fallback text " * 100,
            metadata={"page": 0, "source": "fallback.pdf"},
        )
        with (
            patch("fitz.open", return_value=fake_pdf),
            patch("langchain_community.document_loaders.PyMuPDFLoader") as loader,
        ):
            loader.return_value.load.return_value = [loaded_page]
            pdf_loader = PdfLoader("fake.pdf", chunk_size=CHUNK_SIZE)
            docs = pdf_loader.load()

        assert docs == [loaded_page]
        assert not pdf_loader.structured
        loader.return_value.load.assert_called_once()

    def test_silo_service_splits_pdf_fallback_documents(self):
        fake_pdf = MagicMock()
        fake_pdf.__enter__.return_value = fake_pdf
        fake_pdf.__exit__.return_value = False
        fake_pdf.__iter__.return_value = iter([
            MagicMock(get_text=MagicMock(return_value={"not_blocks": []})),
        ])
        loaded_page = Document(
            page_content="Fallback text " * 100,
            metadata={"page": 0, "source": "fallback.pdf"},
        )
        with (
            patch("fitz.open", return_value=fake_pdf),
            patch("langchain_community.document_loaders.PyMuPDFLoader") as loader,
        ):
            loader.return_value.load.return_value = [loaded_page]
            docs = SiloService.extract_documents_from_file("fake.pdf", ".pdf")

        assert len(docs) > 1
        assert all(doc.metadata["page"] == 1 for doc in docs)
        assert all("_aict_structured_pdf_chunk" not in doc.metadata for doc in docs)

    def test_text_file_keeps_existing_loader_and_split_behavior(self):
        source = Document(page_content="A short text file with readable content.", metadata={})
        with patch("langchain_community.document_loaders.TextLoader") as loader:
            loader.return_value.load.return_value = [source]
            docs = SiloService.extract_documents_from_file("fake.txt", ".txt")

        assert len(docs) == 1
        assert docs[0].page_content == source.page_content
        loader.assert_called_once_with("fake.txt", encoding="utf-8")


class TestExtractDocumentsFromFileWholeFile:
    """step_004: split=False returns exactly one Document per file.

    PDFs get <<<page:N>>> markers embedded in their content; Resource-linked
    documents (resource_id in base_metadata) get id=res{N} stamped on it.
    """

    GOOD_PAGES = [
        "This is the first real page with plenty of alphanumeric content to pass the filter, page one.",
        "This is the second real page with plenty of alphanumeric content to pass the filter, page two.",
        "This is the fourth real page with plenty of alphanumeric content to pass the filter, page four.",
        "This is the fifth real page with plenty of alphanumeric content to pass the filter, page five.",
    ]
    GARBAGE_PAGE = "@@@@"  # symbol-only -> dropped by _clean_chunk_text / _is_meaningful_chunk

    @staticmethod
    def _single_page_doc(extension: str, text: str) -> Document:
        metadata = {"source": f"/tmp/test{extension}"}
        if extension == ".pdf":
            metadata["page"] = 0
        return Document(page_content=text, metadata=metadata)

    @staticmethod
    def _loader_name(extension: str) -> str:
        return {".pdf": "PyMuPDFLoader", ".docx": "Docx2txtLoader", ".txt": "TextLoader", ".md": "TextLoader"}[extension]

    def _mock_pdf_pages(self, texts: list[str]):
        return [Document(page_content=t, metadata={"page": i, "source": "/tmp/test.pdf"}) for i, t in enumerate(texts)]

    @staticmethod
    def _unstructured_fake_pdf():
        """A fitz.open() fake whose pages have no 'blocks': PdfLoader sees an
        unstructured PDF and falls back to PyMuPDFLoader, which the test mocks."""
        fake_pdf = MagicMock()
        fake_pdf.__enter__.return_value = fake_pdf
        fake_pdf.__exit__.return_value = False
        fake_pdf.__iter__.return_value = iter([
            MagicMock(get_text=MagicMock(return_value={"not_blocks": []})),
        ])
        return fake_pdf

    def test_pdf_whole_file_skips_filtered_page_keeps_marker_order(self):
        from tools.vector_stores.lightrag_store import strip_page_markers

        texts = [self.GOOD_PAGES[0], self.GOOD_PAGES[1], self.GARBAGE_PAGE, self.GOOD_PAGES[2], self.GOOD_PAGES[3]]
        with (
            patch("fitz.open", return_value=self._unstructured_fake_pdf()),
            patch("langchain_community.document_loaders.PyMuPDFLoader") as MockLoader,
        ):
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = self._mock_pdf_pages(texts)
            MockLoader.return_value = mock_loader_instance

            docs = SiloService.extract_documents_from_file(
                file_path="/tmp/fake.pdf", file_extension=".pdf",
                base_metadata={"resource_id": 7}, split=False,
            )

        assert len(docs) == 1
        content = docs[0].page_content
        assert content.startswith("<<<page:1>>>")
        assert "<<<page:3>>>" not in content
        positions = [content.index(f"<<<page:{n}>>>") for n in (1, 2, 4, 5)]
        assert positions == sorted(positions)

        clean, _ = strip_page_markers(content)
        assert clean == "\n\n".join(self.GOOD_PAGES)
        assert docs[0].id == "res7"
        assert docs[0].metadata["resource_id"] == 7

    def test_pdf_whole_file_all_pages_filtered_returns_empty(self):
        with (
            patch("fitz.open", return_value=self._unstructured_fake_pdf()),
            patch("langchain_community.document_loaders.PyMuPDFLoader") as MockLoader,
        ):
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = self._mock_pdf_pages([self.GARBAGE_PAGE] * 3)
            MockLoader.return_value = mock_loader_instance

            docs = SiloService.extract_documents_from_file(
                file_path="/tmp/fake.pdf", file_extension=".pdf",
                base_metadata={"resource_id": 7}, split=False,
            )

        assert docs == []

    def test_pdf_split_true_unchanged(self):
        """Backward-compat: split=True PDF output is unaffected by this diff."""
        texts = [self.GOOD_PAGES[0], self.GARBAGE_PAGE, self.GOOD_PAGES[1]]
        with (
            patch("fitz.open", return_value=self._unstructured_fake_pdf()),
            patch("langchain_community.document_loaders.PyMuPDFLoader") as MockLoader,
        ):
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = self._mock_pdf_pages(texts)
            MockLoader.return_value = mock_loader_instance

            docs = SiloService.extract_documents_from_file(
                file_path="/tmp/fake.pdf", file_extension=".pdf",
                base_metadata={"resource_id": 7}, split=True,
            )

        assert len(docs) == 2
        assert docs[0].page_content == self.GOOD_PAGES[0]
        assert docs[0].metadata["page"] == 1
        assert docs[1].page_content == self.GOOD_PAGES[1]
        assert docs[1].metadata["page"] == 3  # original page index 2 -> +1
        for doc in docs:
            assert doc.id is None
            assert doc.metadata["resource_id"] == 7

    def test_txt_split_false_single_document_with_id(self):
        text = "Short plain text content, long enough to pass the meaningful-chunk filter."
        with patch("langchain_community.document_loaders.TextLoader") as MockLoader:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = [self._single_page_doc(".txt", text)]
            MockLoader.return_value = mock_loader_instance

            docs = SiloService.extract_documents_from_file(
                file_path="/tmp/fake.txt", file_extension=".txt",
                base_metadata={"resource_id": 7}, split=False,
            )

        assert len(docs) == 1
        assert docs[0].page_content == text
        assert docs[0].id == "res7"

    def test_docx_split_false_single_document_with_id(self):
        text = "Short docx content, long enough to pass the meaningful-chunk filter easily."
        with patch("langchain_community.document_loaders.Docx2txtLoader") as MockLoader:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = [self._single_page_doc(".docx", text)]
            MockLoader.return_value = mock_loader_instance

            docs = SiloService.extract_documents_from_file(
                file_path="/tmp/fake.docx", file_extension=".docx",
                base_metadata={"resource_id": 7}, split=False,
            )

        assert len(docs) == 1
        assert docs[0].page_content == text
        assert docs[0].id == "res7"

    def test_txt_whole_file_marker_lookalike_is_neutralized(self):
        """A non-PDF whole-file Document now gets a res{N}-shaped id too (unlike
        the old doc-<hash> id), so a line that merely looks like a page marker
        in its own content must not survive to be mistaken for a real one."""
        text = "<<<page:7>>>\nThe rest of the document has plenty of real content, long enough to pass the filter."
        with patch("langchain_community.document_loaders.TextLoader") as MockLoader:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = [self._single_page_doc(".txt", text)]
            MockLoader.return_value = mock_loader_instance

            docs = SiloService.extract_documents_from_file(
                file_path="/tmp/fake.txt", file_extension=".txt",
                base_metadata={"resource_id": 7}, split=False,
            )

        assert len(docs) == 1
        assert docs[0].id == "res7"
        assert "<<<page:7>>>" not in docs[0].page_content

    def test_txt_whole_file_html_encoded_marker_lookalike_is_neutralized(self):
        """Same as above for an HTML-entity-encoded lookalike — must not sanitize
        into a real marker on LightRAG's later second sanitize pass."""
        text = (
            "&lt;&lt;&lt;page:7&gt;&gt;&gt;\n"
            "The rest of the document has plenty of real content, long enough to pass the filter."
        )
        with patch("langchain_community.document_loaders.TextLoader") as MockLoader:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = [self._single_page_doc(".txt", text)]
            MockLoader.return_value = mock_loader_instance

            docs = SiloService.extract_documents_from_file(
                file_path="/tmp/fake.txt", file_extension=".txt",
                base_metadata={"resource_id": 7}, split=False,
            )

        assert len(docs) == 1
        assert docs[0].id == "res7"

        from lightrag.utils import sanitize_text_for_encoding
        # Simulate LightRAG's own later sanitize pass on the stored text.
        resanitized = sanitize_text_for_encoding(docs[0].page_content)
        assert "<<<page:7>>>" not in resanitized

    def test_csv_page_key_in_base_metadata_does_not_shift_markers(self):
        """A caller-supplied 'page' key (e.g. CSV extra_metadata) must not change
        which page numbers the markers use — those come from PyMuPDFLoader's own
        per-Document metadata, read before base_metadata is merged in. It must
        also be dropped from the output Document's own metadata (not just kept
        out of the markers) — left in, it would bake a stale "(p. 999)" suffix
        into the citation label that _page_label never corrects."""
        from tools.vector_stores.lightrag_store import _source_label_from_metadata

        texts = [self.GOOD_PAGES[0], self.GOOD_PAGES[1]]
        with (
            patch("fitz.open", return_value=self._unstructured_fake_pdf()),
            patch("langchain_community.document_loaders.PyMuPDFLoader") as MockLoader,
        ):
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = self._mock_pdf_pages(texts)
            MockLoader.return_value = mock_loader_instance

            docs = SiloService.extract_documents_from_file(
                file_path="/tmp/fake.pdf", file_extension=".pdf",
                base_metadata={"resource_id": 7, "page": 999, "name": "report.pdf"}, split=False,
            )

        assert len(docs) == 1
        content = docs[0].page_content
        assert "<<<page:1>>>" in content
        assert "<<<page:2>>>" in content
        assert "<<<page:999>>>" not in content
        # The caller-supplied "page" key must not survive into the whole-file
        # Document's own metadata either — a whole-file Resource Document has
        # no "page" key of its own (see _source_label_from_metadata's docstring).
        assert "page" not in docs[0].metadata
        assert _source_label_from_metadata(docs[0].metadata) == "report.pdf"

    def _mock_and_extract(self, extension, text, base_metadata):
        with patch(f"langchain_community.document_loaders.{self._loader_name(extension)}") as MockLoader:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = [self._single_page_doc(extension, text)]
            MockLoader.return_value = mock_loader_instance
            if extension == ".pdf":
                with patch("fitz.open", return_value=self._unstructured_fake_pdf()):
                    return SiloService.extract_documents_from_file(
                        file_path=f"/tmp/fake{extension}", file_extension=extension,
                        base_metadata=base_metadata, split=False,
                    )
            return SiloService.extract_documents_from_file(
                file_path=f"/tmp/fake{extension}", file_extension=extension,
                base_metadata=base_metadata, split=False,
            )

    @pytest.mark.parametrize("extension", [".pdf", ".docx", ".txt", ".md"])
    def test_id_is_res_prefixed_resource_id_for_every_extension(self, extension):
        text = "Enough alphanumeric content here to pass the meaningful-chunk filter easily."
        docs = self._mock_and_extract(extension, text, {"resource_id": 7})

        assert len(docs) == 1
        assert docs[0].id == "res7"

    @pytest.mark.parametrize("extension", [".pdf", ".docx", ".txt", ".md"])
    def test_id_is_none_without_resource_id(self, extension):
        text = "Enough alphanumeric content here to pass the meaningful-chunk filter easily."
        docs = self._mock_and_extract(extension, text, {})

        assert len(docs) == 1
        assert docs[0].id is None

    def test_id_is_none_when_split_true_regardless_of_resource_id(self):
        text = "Short text, long enough to pass the meaningful-chunk filter easily."
        with patch("langchain_community.document_loaders.TextLoader") as MockLoader:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = [self._single_page_doc(".txt", text)]
            MockLoader.return_value = mock_loader_instance

            docs = SiloService.extract_documents_from_file(
                file_path="/tmp/fake.txt", file_extension=".txt",
                base_metadata={"resource_id": 7}, split=True,
            )

        assert len(docs) == 1
        assert docs[0].id is None


def test_chunk_size_constant_is_positive():
    assert CHUNK_SIZE > 0


def test_chunk_overlap_less_than_chunk_size():
    assert CHUNK_OVERLAP < CHUNK_SIZE
