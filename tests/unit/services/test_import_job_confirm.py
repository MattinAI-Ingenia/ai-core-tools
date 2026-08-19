import pytest

from services.import_job_confirm import _pdf_filename


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://example.com/docs/CDOC004235.pdf", "CDOC004235.pdf"),
        ("https://example.com/docs/CDOC004235.PDF", "CDOC004235.PDF"),
        ("https://example.com/docs/report", "report.pdf"),
        ("https://example.com/", "document.pdf"),
    ],
)
def test_pdf_filename_never_double_extends(url, expected):
    assert _pdf_filename(url) == expected
