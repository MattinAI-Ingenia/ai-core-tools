"""Unit tests for the PDF chunk text cleaning in SiloService.

Regression guard for the bug where ``_clean_chunk_text`` deleted every numeric
cell of a technical-characteristics table: PyMuPDF emits tables one cell per
line, and the old "drop short lines with <3 letters" rules removed 41% of all
digits in the DOMUSA corpus — so the RAG agent kept answering "I can find the
row but not the value", for values that were in the PDF but never in the index.

No database and no PDF file needed: these exercise the two pure functions on
text shaped exactly like PyMuPDF's real output.
"""

import pytest

from services.silo_service import _clean_chunk_text, _is_meaningful_chunk


# PyMuPDF's real output shape for a spec table: one cell per line, values in
# column order after the label/symbol/unit. Taken from CDOC001961.pdf p.24
# (MINNY DUO 30) and CDOC001018.pdf p.16 (SH, four model columns).
TABLE_ROW = (
    "Emisiones de óxidos de nitrógeno \n"
    "NOx \n"
    "mg/kWh \n"
    "76 \n"
    "73 \n"
    "Perfil de carga declarado \n"
    "- \n"
    "XL \n"
    "Peso bruto \n"
    "Kg \n"
    "145"
)


class TestCleanChunkTextKeepsTableValues:
    """The values, units and short codes of a table row must survive cleaning."""

    @pytest.mark.parametrize("value", ["76", "73", "145", "XL", "mg/kWh", "Kg"])
    def test_value_survives(self, value):
        assert value in _clean_chunk_text(TABLE_ROW)

    def test_labels_also_survive(self):
        cleaned = _clean_chunk_text(TABLE_ROW)
        assert "Emisiones de óxidos de nitrógeno" in cleaned
        assert "Peso bruto" in cleaned

    def test_no_digits_are_lost(self):
        assert sum(c.isdigit() for c in _clean_chunk_text(TABLE_ROW)) == sum(
            c.isdigit() for c in TABLE_ROW
        )

    @pytest.mark.parametrize(
        "value",
        ["0,032", "30,1", "0,0132", "97,5", "1.000", "220 V", "3 mm", "ºC", "P20", "E01"],
    )
    def test_decimal_values_units_and_codes_survive(self, value):
        assert value in _clean_chunk_text(f"Alguna etiqueta \n{value}\nOtra etiqueta")

    def test_multi_column_order_is_preserved(self):
        """Four model columns must stay in order so the LLM can map value→column."""
        raw = "Potencia calorífico nominal \nPrated \nkW \n29 \n40 \n29 \n40"
        cleaned = _clean_chunk_text(raw)
        assert [l.strip() for l in cleaned.splitlines() if l.strip().isdigit()] == [
            "29", "40", "29", "40",
        ]


class TestCleanChunkTextStillDropsGarbage:
    """The Type3-font artifacts the filter was written for must still go."""

    @pytest.mark.parametrize("garbage", ["@@", "??", "•", "(+) (-)", "‐", "((("])
    def test_symbol_only_lines_dropped(self, garbage):
        cleaned = _clean_chunk_text(f"Texto normal de la caldera \n{garbage}\nMás texto normal")
        assert garbage not in cleaned
        assert "Texto normal de la caldera" in cleaned

    def test_long_repeated_glyph_runs_dropped(self):
        assert "@@@@@@@@" not in _clean_chunk_text("Texto válido \n@@@@@@@@@@@@\nMás texto")

    def test_raw_codepoint_sequences_dropped(self):
        assert "64/64/64" not in _clean_chunk_text("Texto válido \n" + "64/" * 10 + "\nMás")

    @pytest.mark.parametrize("marker", ["- 15 -", "- 1 -", "- 132", "132 -"])
    def test_dash_decorated_page_markers_dropped(self, marker):
        cleaned = _clean_chunk_text(f"Instrucciones de instalación \n{marker}\nCapítulo siguiente")
        assert marker not in cleaned.splitlines()

    def test_bare_number_is_kept_even_if_it_might_be_a_page_number(self):
        """Deliberate: "15" alone is indistinguishable from a table value cell.

        Dropping a value costs an answer; keeping a stray page number costs a
        few tokens, so the ambiguous case resolves in favour of keeping it.
        """
        assert "15" in _clean_chunk_text("Presión máxima de trabajo \nbar \n15").splitlines()

    @pytest.mark.parametrize("noise", ["r . * * * ! ? *", "^ i * * : * ;", "* * * * !"])
    def test_symbol_dominated_ocr_noise_dropped(self, noise):
        cleaned = _clean_chunk_text(f"Texto real del manual \n{noise}\nOtro texto real")
        assert noise not in cleaned
        assert "Texto real del manual" in cleaned

    @pytest.mark.parametrize("survivor", ["-'Jl"])
    def test_known_limitation_balanced_noise_survives(self, survivor):
        """Documented trade-off, not an oversight.

        These have as many alphanumerics as symbols. Tightening the rule to
        ``symbols >= alnum`` would also delete legitimate cells such as the
        hydraulic size ``1/2"`` (2 alphanumerics, 2 symbols), which appears
        throughout these manuals. A few stray noise tokens are cheaper than a
        lost measurement, so these are allowed through.
        """
        cleaned = _clean_chunk_text(f"Texto real del manual \n{survivor}\nOtro texto real")
        assert survivor in cleaned

    def test_hydraulic_size_with_punctuation_survives(self):
        assert '1/2"' in _clean_chunk_text('Válvula de seguridad \n1/2"\nRosca hembra')


class TestIsMeaningfulChunk:
    """Digit-dense pages are data, not noise — they must not be discarded."""

    def test_digit_dense_table_is_kept(self):
        """CDOC004191 p.45 shape: an R-T sensor conversion table, ~65% digits."""
        rows = "\n".join(f"{t} \n{r},{t}" for t, r in zip(range(-20, 60), range(100, 180)))
        text = "Tabla de conversión R-T del sensor de temperatura \n" + rows
        assert _is_meaningful_chunk(text) is True

    def test_normal_prose_is_kept(self):
        assert _is_meaningful_chunk(
            "La caldera dispone de dos tipos de bloqueo de seguridad de funcionamiento."
        ) is True

    def test_too_short_is_rejected(self):
        assert _is_meaningful_chunk("76 73") is False

    def test_symbol_soup_is_rejected(self):
        assert _is_meaningful_chunk("@@ ?? @@ ?? @@ ?? @@ ?? @@ ?? @@ ??") is False

    def test_digits_only_without_any_label_is_rejected(self):
        """Numbers with no word anywhere carry nothing retrievable on their own."""
        assert _is_meaningful_chunk("76 73 145 0,032 29 40 29 40 12,1 8,9 97,5") is False


class TestTableOfContentsPages:
    """Dot leaders must not cost us the whole table of contents."""

    TOC = (
        "ÍNDICE \nPág. \n"
        "1. ENUMERACIÓN DE COMPONENTES ....................................................... 2 \n"
        "2. COMPONENTES DE MANDO ............................................................. 3 \n"
        "3. INSTRUCCIONES PARA LA INSTALACIÓN ................................................ 4 \n"
        "23. CARACTERÍSTICAS TÉCNICAS ....................................................... 26"
    )

    def test_toc_page_is_indexed(self):
        cleaned = _clean_chunk_text(self.TOC)
        assert _is_meaningful_chunk(cleaned) is True

    def test_toc_keeps_section_names_and_page_numbers(self):
        cleaned = _clean_chunk_text(self.TOC)
        assert "CARACTERÍSTICAS TÉCNICAS" in cleaned
        assert "26" in cleaned  # the page number the section lives on

    def test_dot_leaders_are_removed(self):
        assert "....." not in _clean_chunk_text(self.TOC)
