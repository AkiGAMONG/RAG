"""1.2 — parsing: pages, dehyphenation, header/footer filter, offsets, title."""

import pytest

from paper_rag.errors import PaperRagError
from paper_rag.ingest import parse_pdf, rects_for_range


@pytest.fixture(scope="module")
def parsed(sample_pdf):
    return parse_pdf(str(sample_pdf))


def test_page_count_and_numbers(parsed):
    assert parsed.n_pages == 3
    assert [p.number for p in parsed.pages] == [1, 2, 3]


def test_dehyphenation_joins_syllable_split(parsed):
    assert "interaction between queries" in parsed.pages[0].text
    assert "inter- action" not in parsed.pages[0].text


def test_headers_footers_filtered(parsed):
    for page in parsed.pages:
        assert "Sample Conference 2024" not in page.text
    # standalone page numbers in the footer zone are dropped too
    assert not parsed.pages[0].text.strip().endswith("\n1")


def test_paragraphs_are_exact_slices(parsed):
    page = parsed.pages[0]
    assert page.paragraphs, "paragraph rebuild produced nothing"
    for para in page.paragraphs:
        assert page.text[para.start:para.end].strip()
    # paragraphs are non-overlapping and ordered
    for a, b in zip(page.paragraphs, page.paragraphs[1:]):
        assert a.end <= b.start


def test_line_spans_cover_text_with_rects(parsed):
    page = parsed.pages[0]
    idx = page.text.find("standard benchmarks")
    assert idx >= 0
    rects = rects_for_range(page, idx, idx + 10)
    assert rects and all(r.page == 1 for r in rects)
    assert all(r.x1 > r.x0 and r.y1 > r.y0 for r in rects)


def test_two_body_paragraphs_detected_on_page1(parsed):
    texts = [parsed.pages[0].text[p.start:p.end] for p in parsed.pages[0].paragraphs]
    assert any("interaction between queries" in t for t in texts)
    assert any("encodes queries independently" in t for t in texts)
    # the vertical gap separates the two paragraphs
    assert not any("interaction between queries" in t
                   and "encodes queries independently" in t for t in texts)


def test_title_extraction(parsed):
    assert parsed.title == "A Study of Late Interaction"


def test_unreadable_pdf_raises_ingest_failed(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all")
    with pytest.raises(PaperRagError) as err:
        parse_pdf(str(bad))
    assert err.value.code == "INGEST_FAILED"
