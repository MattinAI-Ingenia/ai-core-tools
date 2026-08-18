import io
import pytest

from services.csv_import_parser import read_csv_headers, parse_and_dedupe_csv, ParsedRow


def _csv(text: str) -> io.BytesIO:
    return io.BytesIO(text.encode('utf-8'))


def test_read_csv_headers():
    headers = read_csv_headers(_csv("link,title,category\nhttp://a.com/x.pdf,A,Finance\n"))
    assert headers == ['link', 'title', 'category']


def test_dedupe_identical_url_and_metadata_merges():
    csv_text = (
        "link,title\n"
        "http://a.com/x.pdf,A\n"
        "http://a.com/x.pdf,A\n"
    )
    rows, no_link_count = parse_and_dedupe_csv(_csv(csv_text), link_column='link')
    assert rows == [ParsedRow(url='http://a.com/x.pdf', row_metadata={'title': 'A'})]
    assert no_link_count == 0


def test_same_url_different_metadata_stays_separate():
    csv_text = (
        "link,title\n"
        "http://a.com/x.pdf,A\n"
        "http://a.com/x.pdf,B\n"
    )
    rows, _ = parse_and_dedupe_csv(_csv(csv_text), link_column='link')
    assert len(rows) == 2
    assert {r.row_metadata['title'] for r in rows} == {'A', 'B'}


def test_empty_link_excluded_and_counted():
    csv_text = (
        "link,title\n"
        "http://a.com/x.pdf,A\n"
        " ,B\n"
        ",C\n"
    )
    rows, no_link_count = parse_and_dedupe_csv(_csv(csv_text), link_column='link')
    assert len(rows) == 1
    assert no_link_count == 2


def test_unknown_link_column_raises():
    with pytest.raises(ValueError, match="link"):
        parse_and_dedupe_csv(_csv("a,b\n1,2\n"), link_column='nope')
