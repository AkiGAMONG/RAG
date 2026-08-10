"""1.2 — PDF -> per-page text with positions (plan.md §3, D18 pre-processing).

Responsibilities:
  * extract text per page in reading order (PyMuPDF "dict" mode, sort=True);
  * drop rotated lines (e.g. the arXiv sidebar) and repeated headers/footers;
  * undo end-of-line hyphenation;
  * rebuild paragraphs from block structure + line-gap/indent heuristics;
  * keep, for every line, its contribution range in the page text and its
    bounding rectangle — this is what makes char offsets and highlight
    rects possible downstream (api_contract §6).

The page text produced here is canonical: it is written verbatim to
``pages/{n}.txt`` at ingest time, and every chunk's ``char_start/char_end``
indexes into it. Body/caption/reference chunks are always exact slices of
this text.

Pages where paragraph rebuilding misbehaves are flagged ``fallback=True``
(D18): the chunker then uses fixed-size windows for them, and rects degrade
to the whole-page rectangle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf

from ..errors import INGEST_FAILED, PaperRagError
from ..models import Rect

# Header/footer heuristics: candidate zones as a fraction of page height,
# and how many pages a repeated line must appear on to be dropped.
_HF_TOP_FRAC = 0.07
_HF_BOTTOM_FRAC = 0.93
_HF_MIN_REPEATS_FRAC = 0.4

_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
_PARA_GAP_FACTOR = 1.6      # vertical gap > factor * median line height -> new paragraph
_PARA_INDENT_PT = 8.0       # first-line indent that signals a new paragraph


@dataclass
class LineSpan:
    start: int               # char offset into the page text
    end: int
    rect: Rect


@dataclass
class Paragraph:
    start: int               # slice of the page text: text[start:end]
    end: int


@dataclass
class ParsedPage:
    number: int              # 1-based
    text: str
    paragraphs: list[Paragraph] = field(default_factory=list)
    line_spans: list[LineSpan] = field(default_factory=list)
    fallback: bool = False   # True -> paragraph rebuild failed; use fixed chunking
    width: float = 0.0
    height: float = 0.0


@dataclass
class ParsedPaper:
    title: str               # "" if extraction failed (caller falls back to filename)
    n_pages: int
    pages: list[ParsedPage]


@dataclass
class _RawLine:
    text: str
    bbox: tuple[float, float, float, float]
    block: int


def rects_for_range(page: ParsedPage, start: int, end: int) -> list[Rect]:
    """Rectangles of all lines overlapping [start, end) — the highlight shape."""
    return [ls.rect for ls in page.line_spans if ls.start < end and ls.end > start]


def parse_pdf(path: str) -> ParsedPaper:
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise PaperRagError(INGEST_FAILED, f"Cannot open PDF: {exc}",
                            {"path": path}) from exc
    try:
        if doc.page_count == 0:
            raise PaperRagError(INGEST_FAILED, "PDF has no pages", {"path": path})
        raw_pages = [_extract_raw_lines(page) for page in doc]
        dropped = _detect_headers_footers(raw_pages, [p.rect.height for p in doc])
        pages = []
        for i, page in enumerate(doc):
            kept = [ln for ln in raw_pages[i] if id(ln) not in dropped]
            pages.append(_build_page(page, i + 1, kept))
        if all(not p.text.strip() for p in pages):
            raise PaperRagError(
                INGEST_FAILED,
                "No extractable text on any page (scanned/image-only PDF?)",
                {"path": path})
        title = _extract_title(doc)
        return ParsedPaper(title=title, n_pages=doc.page_count, pages=pages)
    finally:
        doc.close()


# --------------------------------------------------------------------------
# line extraction

def _extract_raw_lines(page: pymupdf.Page) -> list[_RawLine]:
    lines: list[_RawLine] = []
    try:
        data = page.get_text("dict", sort=True)
    except Exception:
        return lines
    for bno, block in enumerate(data.get("blocks", [])):
        if block.get("type") != 0:
            continue  # image blocks handled by figures.py
        for line in block.get("lines", []):
            dx, dy = line.get("dir", (1.0, 0.0))
            if abs(dx - 1.0) > 0.01 or abs(dy) > 0.01:
                continue  # rotated text (arXiv sidebar, watermarks)
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            lines.append(_RawLine(text=text, bbox=tuple(line["bbox"]), block=bno))
    return lines


def _normalize_hf(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def _detect_headers_footers(raw_pages: list[list[_RawLine]],
                            heights: list[float]) -> set[int]:
    """Return id()s of lines to drop (repeated headers/footers, page numbers)."""
    n_pages = len(raw_pages)
    threshold = max(2, round(_HF_MIN_REPEATS_FRAC * n_pages))
    seen: dict[str, list[int]] = {}   # normalized text -> line ids in candidate zones
    counts: dict[str, set[int]] = {}  # normalized text -> page indexes
    dropped: set[int] = set()
    for pno, lines in enumerate(raw_pages):
        h = heights[pno] or 1.0
        for ln in lines:
            y_mid = (ln.bbox[1] + ln.bbox[3]) / 2.0
            in_zone = y_mid < _HF_TOP_FRAC * h or y_mid > _HF_BOTTOM_FRAC * h
            if not in_zone:
                continue
            if _PAGE_NUMBER_RE.match(ln.text):
                dropped.add(id(ln))
                continue
            key = _normalize_hf(ln.text)
            seen.setdefault(key, []).append(id(ln))
            counts.setdefault(key, set()).add(pno)
    for key, page_set in counts.items():
        if len(page_set) >= threshold:
            dropped.update(seen[key])
    return dropped


# --------------------------------------------------------------------------
# paragraph rebuild + page text layout

def _build_page(page: pymupdf.Page, number: int, lines: list[_RawLine]) -> ParsedPage:
    width, height = page.rect.width, page.rect.height
    try:
        para_groups = _group_paragraphs(lines)
        text, paragraphs, line_spans = _layout(para_groups, number)
        if lines and not paragraphs:
            raise ValueError("paragraph rebuild produced nothing")
        return ParsedPage(number=number, text=text, paragraphs=paragraphs,
                          line_spans=line_spans, width=width, height=height)
    except Exception:
        plain = page.get_text(sort=True) or ""
        plain = plain.strip("\n")
        span = LineSpan(0, len(plain),
                        Rect(page=number, x0=0.0, y0=0.0, x1=width, y1=height))
        return ParsedPage(number=number, text=plain,
                          paragraphs=[Paragraph(0, len(plain))] if plain else [],
                          line_spans=[span] if plain else [],
                          fallback=True, width=width, height=height)


def _group_paragraphs(lines: list[_RawLine]) -> list[list[_RawLine]]:
    """Split each block's lines into paragraphs by gap and indent heuristics."""
    paragraphs: list[list[_RawLine]] = []
    by_block: dict[int, list[_RawLine]] = {}
    order: list[int] = []
    for ln in lines:
        if ln.block not in by_block:
            order.append(ln.block)
        by_block.setdefault(ln.block, []).append(ln)
    for bno in order:
        blines = by_block[bno]
        heights = sorted(l.bbox[3] - l.bbox[1] for l in blines)
        median_h = heights[len(heights) // 2] if heights else 10.0
        left = min(l.bbox[0] for l in blines)
        current = [blines[0]]
        for prev, ln in zip(blines, blines[1:]):
            gap = ln.bbox[1] - prev.bbox[3]
            indented = (ln.bbox[0] - left) > _PARA_INDENT_PT
            same_row = gap < -0.5 * median_h  # column artifacts: overlapping rows
            new_para = gap > _PARA_GAP_FACTOR * median_h or (
                indented and not same_row and not prev.text.endswith("-"))
            if new_para:
                paragraphs.append(current)
                current = [ln]
            else:
                current.append(ln)
        paragraphs.append(current)
    return paragraphs


def _layout(para_groups: list[list[_RawLine]],
            page_number: int) -> tuple[str, list[Paragraph], list[LineSpan]]:
    """Join lines into page text; track per-line spans and paragraph ranges.

    Dehyphenation: a line ending in "-" whose successor starts lowercase is
    joined directly with the hyphen removed (syllable split); otherwise lines
    join with a single space. Paragraphs are separated by a blank line.
    """
    parts: list[str] = []
    pos = 0
    paragraphs: list[Paragraph] = []
    line_spans: list[LineSpan] = []
    for group in para_groups:
        if not group:
            continue
        if parts:
            parts.append("\n\n")
            pos += 2
        para_start = pos
        contribs: list[str] = [ln.text for ln in group]
        joiners: list[str] = [""]
        for k in range(1, len(group)):
            nxt = contribs[k]
            if contribs[k - 1].endswith("-") and nxt[:1].islower():
                contribs[k - 1] = contribs[k - 1][:-1]
                joiners.append("")
            else:
                joiners.append(" ")
        for k, ln in enumerate(group):
            if joiners[k]:
                parts.append(joiners[k])
                pos += len(joiners[k])
            start = pos
            parts.append(contribs[k])
            pos += len(contribs[k])
            line_spans.append(LineSpan(start, pos, Rect(
                page=page_number, x0=ln.bbox[0], y0=ln.bbox[1],
                x1=ln.bbox[2], y1=ln.bbox[3])))
        paragraphs.append(Paragraph(para_start, pos))
    return "".join(parts), paragraphs, line_spans


# --------------------------------------------------------------------------
# title extraction (best effort)

def _extract_title(doc: pymupdf.Document) -> str:
    meta_title = (doc.metadata or {}).get("title", "").strip()
    try:
        page = doc[0]
        data = page.get_text("dict", sort=True)
        best_size = 0.0
        best_lines: list[str] = []
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                dx, dy = line.get("dir", (1.0, 0.0))
                if abs(dx - 1.0) > 0.01 or abs(dy) > 0.01:
                    continue  # rotated (arXiv sidebar) — never the title
                if line["bbox"][1] > page.rect.height * 0.5:
                    continue
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text or len(text) < 3:
                        continue
                    size = round(span.get("size", 0.0), 1)
                    if size > best_size + 0.4:
                        best_size, best_lines = size, [text]
                    elif abs(size - best_size) <= 0.4:
                        best_lines.append(text)
        candidate = re.sub(r"\s+", " ", " ".join(best_lines)).strip()
        if 8 <= len(candidate) <= 300:
            return candidate
    except Exception:
        pass
    return meta_title
