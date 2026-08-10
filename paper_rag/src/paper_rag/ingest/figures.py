"""1.4 — Figure/table captions and page-region screenshots (plan.md D3/D4).

Captions are detected by regex at paragraph starts ("Figure 3: ...",
"Fig. 2. ...", "Table 1: ..."); tables are treated exactly like figures
(D3 — no structured table parsing). For each caption we screenshot the page
region occupied by nearby image blocks / vector drawings (figures usually
sit above their caption, tables below); when no plausible region is found
the whole page is captured instead (the D3 degradation path).

Outputs:
  * ``FigureRecord`` per figure -> meta.json "figures" (local ids "fig:3");
  * a caption Chunk per figure (type="caption", the retrieval index of the
    figure per D3);
  * exclusion ranges so caption text is not re-chunked as body text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from ..models import Rect
from .chunker import Chunk, Exclusions
from .pdf_parser import ParsedPaper, rects_for_range

_CAPTION_RE = re.compile(
    r"^\s*(Figure|Fig\.|FIG\.?|Table|TABLE|Tab\.)\s*(\d+)\s*[.:：]\s+\S",
    re.IGNORECASE)

_MAX_REGION_DISTANCE = 380.0   # pt: how far from the caption we look for content
_REGION_MARGIN = 4.0
_MIN_REGION_SIZE = 40.0
_MIN_CANDIDATE_SIZE = 8.0
_RENDER_DPI = 150


@dataclass
class FigureRecord:
    figure_id: str            # paper-local id: "fig:3" / "tab:2"
    label: str                # "Figure 3" / "Table 2"
    caption: str
    page: int                 # 1-based
    image_path: str           # relative to the paper dir, e.g. "figures/fig_3.png"
    rect: list[float]         # captured region [x0, y0, x1, y1] on that page
    full_page: bool           # True when we degraded to a whole-page screenshot


def _normalize_label(kind: str, number: str) -> tuple[str, str]:
    """('Figure'|'Table', local figure id)."""
    if kind.lower().startswith(("table", "tab")):
        return f"Table {number}", f"tab:{number}"
    return f"Figure {number}", f"fig:{number}"


def extract_figures(pdf_path: str, parsed: ParsedPaper, paper_dir: Path,
                    paper_id: str) -> tuple[list[FigureRecord], list[Chunk], Exclusions]:
    records: list[FigureRecord] = []
    caption_chunks: list[Chunk] = []
    exclusions: Exclusions = {}
    seen_ids: set[str] = set()
    figures_dir = paper_dir / "figures"

    doc = pymupdf.open(pdf_path)
    try:
        for page in parsed.pages:
            if page.fallback:
                continue  # no reliable paragraph ranges to anchor a caption on
            for para in page.paragraphs:
                text = page.text[para.start:para.end]
                m = _CAPTION_RE.match(text)
                if not m:
                    continue
                label, fid = _normalize_label(m.group(1), m.group(2))
                if fid in seen_ids:
                    continue  # e.g. the caption re-detected on a later page
                seen_ids.add(fid)

                caption_rects = rects_for_range(page, para.start, para.end)
                pdf_page = doc[page.number - 1]
                region, full_page = _locate_region(
                    pdf_page, caption_rects, is_table=fid.startswith("tab:"))
                figures_dir.mkdir(parents=True, exist_ok=True)
                image_name = f"{fid.replace(':', '_')}.png"
                _render_region(pdf_page, region, figures_dir / image_name)

                records.append(FigureRecord(
                    figure_id=fid, label=label, caption=text,
                    page=page.number, image_path=f"figures/{image_name}",
                    rect=[region.x0, region.y0, region.x1, region.y1],
                    full_page=full_page))
                caption_chunks.append(Chunk(
                    chunk_id=f"{paper_id}:{page.number:04d}:c{len(caption_chunks):03d}",
                    paper_id=paper_id, type="caption", text=text,
                    page_start=page.number, page_end=page.number,
                    char_start=para.start, char_end=para.end,
                    figure_id=fid, rects=caption_rects))
                exclusions.setdefault(page.number, []).append((para.start, para.end))
    finally:
        doc.close()
    return records, caption_chunks, exclusions


# --------------------------------------------------------------------------
# region location

def _locate_region(page: pymupdf.Page, caption_rects: list[Rect],
                   is_table: bool) -> tuple[pymupdf.Rect, bool]:
    """Bounding box of the visual content belonging to a caption.

    Figures prefer candidates above the caption, then below. Tables (ACM
    style: caption above the table) only look below — falling back to the
    other side too easily captures a neighboring figure's plot, and a
    full-page screenshot is the safer degradation (D3).
    """
    if not caption_rects:
        return page.rect, True
    cap = pymupdf.Rect(min(r.x0 for r in caption_rects),
                       min(r.y0 for r in caption_rects),
                       max(r.x1 for r in caption_rects),
                       max(r.y1 for r in caption_rects))
    candidates = _visual_candidates(page)
    sides = ["below"] if is_table else ["above", "below"]
    for side in sides:
        rects = [c for c in candidates if _on_side(c, cap, side)]
        if not rects:
            continue
        # Manual min/max union: pymupdf's `|` ignores empty rects, and ruled
        # table lines are zero-height rects that must still count.
        region = pymupdf.Rect(min(r.x0 for r in rects) - _REGION_MARGIN,
                              min(r.y0 for r in rects) - _REGION_MARGIN,
                              max(r.x1 for r in rects) + _REGION_MARGIN,
                              max(r.y1 for r in rects) + _REGION_MARGIN)
        region &= page.rect
        if region.width >= _MIN_REGION_SIZE and region.height >= _MIN_REGION_SIZE:
            return region, False
    return page.rect, True


def _visual_candidates(page: pymupdf.Page) -> list[pymupdf.Rect]:
    rects: list[pymupdf.Rect] = []
    try:
        for info in page.get_image_info():
            rects.append(pymupdf.Rect(info["bbox"]))
    except Exception:
        pass
    try:
        for drawing in page.get_drawings():
            rects.append(pymupdf.Rect(drawing["rect"]))
    except Exception:
        pass
    return [r for r in rects
            if r.width >= _MIN_CANDIDATE_SIZE or r.height >= _MIN_CANDIDATE_SIZE]


def _on_side(rect: pymupdf.Rect, cap: pymupdf.Rect, side: str) -> bool:
    if rect.x1 <= cap.x0 - 2 or rect.x0 >= cap.x1 + 2:
        return False  # other column: no horizontal overlap with the caption
    if side == "above":
        ok = rect.y1 <= cap.y0 + 2 and (cap.y0 - rect.y0) <= _MAX_REGION_DISTANCE
    else:
        ok = rect.y0 >= cap.y1 - 2 and (rect.y1 - cap.y1) <= _MAX_REGION_DISTANCE
    return ok


def _render_region(page: pymupdf.Page, region: pymupdf.Rect, out_path: Path) -> None:
    pixmap = page.get_pixmap(clip=region, dpi=_RENDER_DPI)
    pixmap.save(out_path)


# --------------------------------------------------------------------------
# inline figure mentions (used by retrieval, D4 regex-first)

_INLINE_FIGURE_RE = re.compile(
    r"\b(Figure|Fig\.|Figs?\.|Table|Tab\.)\s*(\d+)", re.IGNORECASE)


def find_figure_mentions(text: str) -> list[str]:
    """Local figure ids ("fig:3") mentioned in free text, in order, deduped."""
    out: list[str] = []
    for kind, number in _INLINE_FIGURE_RE.findall(text):
        _, fid = _normalize_label(kind, number)
        if fid not in out:
            out.append(fid)
    return out
