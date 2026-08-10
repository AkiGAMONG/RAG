"""1.5 — bibliography entries, field extraction, inline marker parsing."""

import pytest

from paper_rag.ingest import (extract_references, parse_author_year_markers,
                              parse_numeric_markers, parse_pdf,
                              resolve_inline_citations)


@pytest.fixture(scope="module")
def refs(sample_pdf):
    parsed = parse_pdf(str(sample_pdf))
    return extract_references(parsed, "p1")


def test_entries_split_with_sequence_discipline(refs):
    entries, _, _ = refs
    assert [e.key for e in entries] == ["1", "2"]
    assert entries[0].raw.startswith("[1] Alice Smith")
    assert "Proceedings of Tests" in entries[0].raw


def test_field_extraction(refs):
    entries, _, _ = refs
    assert entries[0].year == 2020
    assert "Alice Smith" in entries[0].authors
    assert "Foundational" in entries[0].title
    assert entries[1].year == 2021
    assert "Carol White" in entries[1].authors


def test_reference_chunks(refs):
    _, chunks, _ = refs
    assert [c.chunk_id for c in chunks] == ["p1:r1", "p1:r2"]
    for chunk in chunks:
        assert chunk.type == "reference"
        assert chunk.page_start == 3
        assert chunk.char_end > chunk.char_start


def test_reference_region_excluded_from_body(refs):
    _, _, exclusions = refs
    assert 3 in exclusions
    assert len(exclusions[3]) >= 2   # heading + entries


def _fake_paper(pages: list[list[str]]):
    """Build a ParsedPaper directly from paragraph lists (deterministic)."""
    from paper_rag.ingest.pdf_parser import (LineSpan, Paragraph, ParsedPage,
                                             ParsedPaper)
    from paper_rag.models import Rect
    out = []
    for number, paras in enumerate(pages, start=1):
        text = "\n\n".join(paras)
        offsets, pos = [], 0
        for p in paras:
            offsets.append(Paragraph(pos, pos + len(p)))
            pos += len(p) + 2
        spans = [LineSpan(o.start, o.end, Rect(number, 0, 10 * i, 100, 10 * i + 9))
                 for i, o in enumerate(offsets)]
        out.append(ParsedPage(number=number, text=text, paragraphs=offsets,
                              line_spans=spans))
    return ParsedPaper(title="T", n_pages=len(out), pages=out)


def test_numdot_style_offsets_and_page_breaks():
    parsed = _fake_paper([
        ["Body text."],
        ["References",
         "1. Alice Aaa. 2020. First Title. Venue One.",
         "2. Bob Bbb. 2021. Second Title. Venue Two."],
        ["3. Carol Ccc. 2022. Third Title. Venue Three."],
    ])
    entries, chunks, _ = extract_references(parsed, "p1")
    assert [e.key for e in entries] == ["1", "2", "3"]
    # entry 2 starts exactly at its own paragraph, not in the "\n\n" gap
    page2 = parsed.pages[1]
    assert page2.text[entries[1].char_start:].startswith("2. Bob")
    # entry 3 lives on page 3 with a valid range and rects
    assert entries[2].page == 3
    chunk3 = next(c for c in chunks if c.chunk_id == "p1:r3")
    assert chunk3.text.startswith("3. Carol")
    assert chunk3.rects


def test_reference_chunks_are_exact_page_slices(refs, sample_pdf):
    from paper_rag.ingest import parse_pdf
    parsed = parse_pdf(str(sample_pdf))
    _, chunks, _ = refs
    for chunk in chunks:
        page = next(p for p in parsed.pages if p.number == chunk.page_start)
        assert chunk.text == page.text[chunk.char_start:chunk.char_end]


def test_numeric_marker_parsing():
    assert parse_numeric_markers("see [12] and [3, 5]") == [3, 5, 12]
    assert parse_numeric_markers("range [4-6] works") == [4, 5, 6]
    assert parse_numeric_markers("no markers") == []
    assert parse_numeric_markers("empty [] and words [abc]") == []


def test_author_year_marsing():
    found = parse_author_year_markers(
        "As shown in (Khattab & Zaharia, 2020) and by Devlin et al. (2018).")
    assert ("Khattab & Zaharia", 2020) in found
    assert ("Devlin et al.", 2018) in found


def test_resolve_inline_citations():
    references = {
        "1": {"raw": "[1] ...", "authors": "Alice Smith and Bob Jones",
              "year": 2020, "title": "A Foundational Method"},
        "2": {"raw": "[2] ...", "authors": "Carol White", "year": 2021,
              "title": "Another Approach"},
    }
    resolved = resolve_inline_citations("Prior work [1] and (White, 2021).",
                                        references)
    assert [key for key, _ in resolved] == ["1", "2"]
    assert resolve_inline_citations("[7] does not exist", references) == []
