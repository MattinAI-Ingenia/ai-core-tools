"""Locks the entity-type inference extractor.

The two failures worth guarding are silent ones: a table-of-contents line whose
dot leaders come in two runs used to be kept whole (one real manual went from
47 to 276 chars per line, quadrupling the prompt), and a sampler that picks
thirty manuals of the same product line looks like it worked.
"""
import pytest

from backend.tools.vector_stores.lightrag.entity_type_inference import (  # noqa: E402
    ENTITY_TYPES_JSON_SCHEMA,
    DocumentOutline,
    build_consolidation_prompt,
    fit_budget,
    _TOC_LINE_RE,
    _TOC_TAIL_RE,
    _cap_toc,
    _clean,
    _count_tokens,
    assign_families,
    build_prompt,
    select_diverse,
)


def _parse_toc_line(raw: str) -> str | None:
    """Mirror of the parsing extract_outline does per line."""
    match = _TOC_LINE_RE.match(raw.strip())
    if not match:
        return None
    return _TOC_TAIL_RE.sub("", _clean(match.group(1)))


def test_single_run_of_dot_leaders():
    assert _parse_toc_line("21 CARACTERÍSTICAS TÉCNICAS ......... 80") == "21 CARACTERÍSTICAS TÉCNICAS"


def test_dot_leaders_split_in_two_runs_do_not_leak_into_the_title():
    raw = "22.8 BOQUILLAS " + "." * 90 + " " + "." * 70 + " 61"
    assert _parse_toc_line(raw) == "22.8 BOQUILLAS"


def test_non_toc_line_is_ignored():
    assert _parse_toc_line("El quemador debe revisarse cada 2 años.") is None


def test_cap_keeps_top_level_entries_and_document_order():
    toc = [f"{i} SECCION {i} — p.{i}" for i in range(1, 6)]
    toc += [f"9.{i} SUBSECCION {i} — p.{i}" for i in range(1, 200)]

    capped = _cap_toc(toc)

    assert len(capped) == 60
    assert capped == sorted(capped, key=toc.index), "original order must survive"
    for top_level in toc[:5]:
        assert top_level in capped, "top-level entries must never be dropped"


def _outline(doc_id, cover, toc_lines=3):
    return DocumentOutline(
        doc_id=doc_id,
        cover=cover,
        toc=[f"{i} SECCION — p.{i}" for i in range(toc_lines)],
        pages=10,
    )


def test_ubiquitous_cover_words_are_not_treated_as_families():
    # "MANUAL DE INSTALACION" is on every cover, so it cannot distinguish them.
    outlines = [
        _outline("A", "MANUAL DE INSTALACION THERMAPRO HT 290"),
        _outline("B", "MANUAL DE INSTALACION BIOCLASS HC 66"),
        _outline("C", "MANUAL DE INSTALACION HYDREA 150"),
    ]

    assign_families(outlines)

    assert [o.family for o in outlines] == ["THERMAPRO", "BIOCLASS", "HYDREA"]


def test_select_diverse_takes_one_per_family_before_a_second():
    outlines = [_outline(f"DUAL{i}", f"DUAL CLIMA {i}R") for i in range(5)]
    outlines += [_outline("BIO1", "BIOCLASS HC 66"), _outline("HYD1", "HYDREA 150")]

    picked = select_diverse(outlines, limit=3)

    assert len({o.family for o in picked}) == 3


def test_select_diverse_returns_everything_below_the_limit():
    outlines = [_outline(f"D{i}", f"PRODUCTO{i}") for i in range(4)]
    assert len(select_diverse(outlines, limit=30)) == 4


@pytest.mark.parametrize(
    "language,expected,unexpected",
    [
        ("Spanish", "analista de documentación técnica", "documentation analyst"),
        ("English", "documentation analyst", "analista de documentación"),
        (None, "documentation analyst", "analista de documentación"),
    ],
)
def test_prompt_language_follows_the_silo_setting(language, expected, unexpected):
    prompt = build_prompt([_outline("A", "THERMAPRO HT 290")], language=language)

    assert expected in prompt
    assert unexpected not in prompt
    assert '"types"' in prompt, "the JSON contract must be in the prompt"


def test_fit_budget_drops_second_excerpts_before_whole_documents():
    outlines = []
    for i in range(4):
        outline = _outline(f"D{i}", f"PRODUCTO{i}", toc_lines=10)
        outline.samples = [("ESPECIFICACIONES", "x" * 700), ("ERRORES", "y" * 700)]
        outlines.append(outline)
    full = sum(o.approx_tokens for o in outlines)

    kept = fit_budget(outlines, max_tokens=int(full * 0.8))

    assert len(kept) == 4, "trimming excerpts must be tried before dropping documents"
    assert sum(o.approx_tokens for o in kept) <= int(full * 0.8)
    assert [len(o.samples) for o in kept][0] == 2, "the first document keeps both"


def test_fit_budget_leaves_a_payload_that_already_fits_alone():
    outlines = [_outline("D0", "PRODUCTO", toc_lines=3)]
    outlines[0].samples = [("ESPECIFICACIONES", "x" * 100)]

    kept = fit_budget(outlines, max_tokens=100_000)

    assert len(kept) == 1
    assert len(kept[0].samples) == 1


def test_a_class_with_no_examples_is_rejected_by_the_schema():
    # A model padding up to a target count emits exactly this, and "no
    # instances" is the signal that the class should not exist.
    examples_schema = ENTITY_TYPES_JSON_SCHEMA["properties"]["types"]["items"]
    assert examples_schema["properties"]["examples"]["minItems"] == 2


def test_consolidation_prompt_lists_the_candidates_with_their_examples():
    prompt = build_consolidation_prompt(
        [{"name": "Presion", "why": "…", "examples": ["3 bar", "0,3 MPa"]}],
        language="Spanish",
    )

    assert "Presion" in prompt
    assert "3 bar" in prompt
    assert "No inventes nombres nuevos" in prompt


class TestApproxTokensUsesRealCounting:
    """approx_tokens used to be a chars/2.3 heuristic, calibrated once against
    a real payload — and still missed budget on a later run
    (prompt_tokens=12342 against a 10500 budget it should have fit under).
    Any fixed ratio drifts with whatever corpus is selected; a real tiktoken
    count does not.
    """

    def test_matches_a_direct_tiktoken_count(self):
        outline = DocumentOutline(
            doc_id="d1",
            cover="THERMAPRO 16 HTT — manual técnico",
            toc=["21 CARACTERÍSTICAS TÉCNICAS", "13.1 PARÁMETROS DEL SISTEMA"],
            samples=[("21 CARACTERÍSTICAS TÉCNICAS", "Potencia nominal 16 kW, 230V, 45ºC máx.")],
        )
        expected = _count_tokens(
            outline.cover
            + "\n".join(outline.toc)
            + "\n".join(f"{t}\n{x}" for t, x in outline.samples)
        )
        assert outline.approx_tokens == expected

    def test_scales_with_content_length(self):
        short = DocumentOutline(doc_id="d1", cover="A", toc=[])
        long = DocumentOutline(doc_id="d2", cover="A " * 500, toc=[])
        assert long.approx_tokens > short.approx_tokens

    def test_empty_outline_costs_nothing(self):
        assert DocumentOutline(doc_id="d1", cover="").approx_tokens == 0
