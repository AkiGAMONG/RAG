"""1.3 — D18 paragraph-aware chunking: budget, boundaries, overlap, fallback."""

from paper_rag.ingest.chunker import chunk_pages
from paper_rag.ingest.pdf_parser import ParsedPage, Paragraph, LineSpan
from paper_rag.models import Rect
from conftest import fake_config


def _page(text: str, paragraphs: list[tuple[int, int]], number: int = 1,
          fallback: bool = False) -> ParsedPage:
    return ParsedPage(
        number=number, text=text,
        paragraphs=[Paragraph(s, e) for s, e in paragraphs],
        line_spans=[LineSpan(0, len(text), Rect(number, 0, 0, 100, 100))],
        fallback=fallback)


def _paras_from_text(text: str) -> list[tuple[int, int]]:
    out, pos = [], 0
    for part in text.split("\n\n"):
        out.append((pos, pos + len(part)))
        pos += len(part) + 2
    return out


def test_chunks_are_exact_slices_and_within_budget():
    text = "\n\n".join(f"Paragraph {i} " + "x" * 120 for i in range(10))
    page = _page(text, _paras_from_text(text))
    chunks = chunk_pages("p1", [page], fake_config())
    assert chunks
    for c in chunks:
        assert c.text == page.text[c.char_start:c.char_end]
        assert len(c.text) <= 800
        assert c.page_start == c.page_end == 1


def test_short_paragraphs_merge_and_break_on_boundaries():
    text = "\n\n".join("Para " + "a" * 200 for _ in range(6))  # ~205 chars each
    paras = _paras_from_text(text)
    page = _page(text, paras)
    chunks = chunk_pages("p1", [page], fake_config())
    starts = {c.char_start for c in chunks}
    # every chunk starts exactly at a paragraph start (no overlap at
    # paragraph-boundary breaks)
    assert starts <= {s for s, _ in paras}
    assert 1 < len(chunks) < 6  # merging happened


def test_forced_split_never_reemits_a_boundary():
    # A >700-char sentence right after a boundary used to make the splitter
    # re-select the same boundary ~100 times, spraying sliver chunks.
    text = "A" * 599 + ". " + "B" * 748 + "."       # 1350 chars, two sentences
    page = _page(text, [(0, len(text))])
    chunks = chunk_pages("p1", [page], fake_config())
    assert len(chunks) <= 4
    ends = [c.char_end for c in chunks]
    assert ends == sorted(set(ends)), "cut points must strictly advance"
    # tail merge may exceed the budget by up to _MIN_TAIL (150) chars
    assert all(len(c.text) <= 800 + 150 for c in chunks)
    assert chunks[-1].char_end == len(text)


def test_long_paragraph_forced_split_has_overlap():
    sentences = " ".join("This is sentence number %d of the very long paragraph." % i
                         for i in range(40))   # ~2000 chars, one paragraph
    page = _page(sentences, [(0, len(sentences))])
    config = fake_config()
    chunks = chunk_pages("p1", [page], config)
    assert len(chunks) >= 2
    for a, b in zip(chunks, chunks[1:]):
        assert b.char_start < a.char_end, "forced splits must overlap (D18)"
        assert a.char_end - b.char_start <= config.chunk_overlap + 80


def test_sentence_over_budget_hard_cut_progresses():
    text = "y" * 3000   # no sentence boundaries at all
    page = _page(text, [(0, len(text))])
    chunks = chunk_pages("p1", [page], fake_config())
    assert all(len(c.text) <= 800 for c in chunks)
    assert chunks[-1].char_end == 3000


def test_fallback_page_uses_fixed_windows():
    text = "z" * 2000
    page = _page(text, [(0, len(text))], fallback=True)
    chunks = chunk_pages("p1", [page], fake_config())
    assert chunks[0].char_end - chunks[0].char_start == 800
    assert chunks[1].char_start == 700   # step = size - overlap


def test_fixed_strategy_config():
    text = "w" * 1600
    page = _page(text, _paras_from_text(text))
    chunks = chunk_pages("p1", [page], fake_config(chunking_strategy="fixed"))
    assert chunks[0].char_end == 800 and chunks[1].char_start == 700


def test_exclusions_are_skipped():
    text = "A" * 300 + "\n\n" + "CAPTION" * 20 + "\n\n" + "B" * 300
    paras = _paras_from_text(text)
    page = _page(text, paras)
    excl = {1: [paras[1]]}
    chunks = chunk_pages("p1", [page], fake_config(), excl)
    joined = " ".join(c.text for c in chunks)
    assert "CAPTION" not in joined
    assert "A" * 300 in joined and "B" * 300 in joined


def test_chunk_ids_deterministic():
    text = "\n\n".join("Para " + "a" * 300 for _ in range(4))
    page = _page(text, _paras_from_text(text))
    ids1 = [c.chunk_id for c in chunk_pages("p1", [page], fake_config())]
    ids2 = [c.chunk_id for c in chunk_pages("p1", [page], fake_config())]
    assert ids1 == ids2 and len(set(ids1)) == len(ids1)
