"""1.4 — caption detection, region screenshots, degradation to full page."""

import pymupdf
import pytest

from paper_rag.ingest import extract_figures, find_figure_mentions, parse_pdf


@pytest.fixture()
def extracted(sample_pdf, tmp_path):
    parsed = parse_pdf(str(sample_pdf))
    records, chunks, exclusions = extract_figures(
        str(sample_pdf), parsed, tmp_path / "paper", "p1")
    return parsed, records, chunks, exclusions, tmp_path / "paper"


def test_captions_detected(extracted):
    _, records, _, _, _ = extracted
    ids = {r.figure_id for r in records}
    assert ids == {"fig:1", "tab:1"}
    fig = next(r for r in records if r.figure_id == "fig:1")
    assert fig.label == "Figure 1"
    assert fig.caption.startswith("Figure 1:")
    assert fig.page == 2
    tab = next(r for r in records if r.figure_id == "tab:1")
    assert tab.label == "Table 1"


def test_images_rendered_and_regions_not_full_page(extracted):
    _, records, _, _, paper_dir = extracted
    for record in records:
        image = paper_dir / record.image_path
        assert image.is_file() and image.stat().st_size > 0
        assert not record.full_page, \
            f"{record.figure_id} should have found a drawing-based region"
    fig = next(r for r in records if r.figure_id == "fig:1")
    x0, y0, x1, y1 = fig.rect
    # the drawn rectangle sits at (100,110)-(300,240); margin is 4pt
    assert 90 <= x0 <= 100 and 240 <= y1 <= 250


def test_caption_chunks_and_exclusions(extracted):
    _, _, chunks, exclusions, _ = extracted
    assert {c.figure_id for c in chunks} == {"fig:1", "tab:1"}
    for chunk in chunks:
        assert chunk.type == "caption"
        assert chunk.rects, "caption chunks must carry highlight rects"
    assert 2 in exclusions and len(exclusions[2]) == 2


def test_full_page_fallback_without_drawings(tmp_path):
    pdf = tmp_path / "plain.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 300), "Figure 7: A caption with no drawable content nearby.",
                     fontsize=10)
    doc.save(str(pdf))
    doc.close()
    parsed = parse_pdf(str(pdf))
    records, _, _ = extract_figures(str(pdf), parsed, tmp_path / "p", "p1")
    assert len(records) == 1
    assert records[0].full_page
    assert (tmp_path / "p" / records[0].image_path).is_file()


def test_inline_mentions():
    text = "As shown in Figure 3 and Fig. 4, unlike Table 2. figure 3 again."
    assert find_figure_mentions(text) == ["fig:3", "fig:4", "tab:2"]
    assert find_figure_mentions("No mentions here.") == []
