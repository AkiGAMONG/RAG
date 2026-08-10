"""1.3 — Paragraph-aware recursive chunking (plan.md D18).

Strategy "paragraph" (default):
  * pack whole paragraphs greedily into a ``chunk_size`` character budget
    (short paragraphs merge; breaks land on paragraph boundaries, no overlap);
  * a paragraph over budget is force-split at sentence boundaries, and a
    sentence over budget is hard-cut — ``chunk_overlap`` is applied ONLY at
    these forced splits;
  * pages flagged ``fallback`` by the parser use the fixed strategy.

Strategy "fixed" (fallback + A/B baseline): sliding windows of
``chunk_size`` with ``chunk_overlap``, page-aware.

Invariant either way: ``chunk.text == page.text[char_start:char_end]`` — a
chunk is always an exact slice of the canonical page text, so offsets and
rects stay meaningful (api_contract §6). Chunks never span pages; a
paragraph cut by a page break simply yields one chunk per page (recorded as
a trade-off in PROGRESS.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import Config
from ..models import Rect
from .pdf_parser import ParsedPage, rects_for_range

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")
_MIN_TAIL = 150   # a trailing forced-split piece smaller than this merges back


@dataclass
class Chunk:
    """Internal unit shared by ingest and store (not part of the public API)."""
    chunk_id: str
    paper_id: str
    type: str                 # "body" | "caption" | "reference"
    text: str
    page_start: int
    page_end: int
    char_start: int           # offsets into pages/{page_start}.txt
    char_end: int
    figure_id: str = ""       # non-empty only for type="caption"
    section: str = ""         # "" in MVP (stretch)
    rects: list[Rect] = field(default_factory=list)


Exclusions = dict[int, list[tuple[int, int]]]   # page number -> [(start, end)]


def chunk_pages(paper_id: str, pages: list[ParsedPage], config: Config,
                exclusions: Exclusions | None = None) -> list[Chunk]:
    """Body chunks for all pages, skipping excluded ranges (captions, refs)."""
    exclusions = exclusions or {}
    chunks: list[Chunk] = []
    for page in pages:
        ranges = _chunk_ranges_for_page(page, config, exclusions.get(page.number, []))
        for start, end in ranges:
            text = page.text[start:end]
            if not text.strip():
                continue
            chunks.append(Chunk(
                chunk_id=f"{paper_id}:{page.number:04d}:b{len(chunks):04d}",
                paper_id=paper_id, type="body", text=text,
                page_start=page.number, page_end=page.number,
                char_start=start, char_end=end,
                rects=rects_for_range(page, start, end)))
    return chunks


def _chunk_ranges_for_page(page: ParsedPage, config: Config,
                           excluded: list[tuple[int, int]]) -> list[tuple[int, int]]:
    regions = _free_regions(len(page.text), excluded)
    use_fixed = config.chunking_strategy == "fixed" or page.fallback
    ranges: list[tuple[int, int]] = []
    for r_start, r_end in regions:
        if use_fixed:
            ranges.extend(_fixed_ranges(r_start, r_end, config))
        else:
            paras = [(max(p.start, r_start), min(p.end, r_end))
                     for p in page.paragraphs
                     if p.start < r_end and p.end > r_start]
            paras = [(s, e) for s, e in paras if e > s]
            if paras:
                ranges.extend(_paragraph_ranges(page.text, paras, config))
            else:
                ranges.extend(_fixed_ranges(r_start, r_end, config))
    return ranges


def _free_regions(text_len: int, excluded: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Complement of the excluded ranges within [0, text_len)."""
    regions: list[tuple[int, int]] = []
    pos = 0
    for s, e in sorted(excluded):
        if s > pos:
            regions.append((pos, min(s, text_len)))
        pos = max(pos, e)
    if pos < text_len:
        regions.append((pos, text_len))
    return [(s, e) for s, e in regions if e > s]


def _fixed_ranges(start: int, end: int, config: Config) -> list[tuple[int, int]]:
    size, overlap = config.chunk_size, config.chunk_overlap
    step = max(1, size - overlap)
    out = []
    pos = start
    while pos < end:
        out.append((pos, min(pos + size, end)))
        if pos + size >= end:
            break
        pos += step
    return out


def _paragraph_ranges(text: str, paras: list[tuple[int, int]],
                      config: Config) -> list[tuple[int, int]]:
    budget = config.chunk_size
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(paras):
        p_start, p_end = paras[i]
        if p_end - p_start > budget:
            out.extend(_split_long(text, p_start, p_end, config))
            i += 1
            continue
        j = i
        while j + 1 < len(paras) and paras[j + 1][1] - p_start <= budget:
            j += 1
        out.append((p_start, paras[j][1]))
        i = j + 1
    return out


def _split_long(text: str, start: int, end: int,
                config: Config) -> list[tuple[int, int]]:
    """Force-split an over-budget paragraph; overlap applies here (D18).

    Cut points advance strictly: each piece ends at the last sentence
    boundary AFTER the previous cut that fits the budget, or at a hard cut
    when no boundary qualifies. (A naive "step back by overlap and retry"
    re-selects the same boundary whenever the next sentence exceeds
    budget - overlap, spraying near-duplicate sliver chunks.)
    """
    budget, overlap = config.chunk_size, config.chunk_overlap
    # Sentence boundary positions (absolute offsets), always including `end`.
    boundaries = [start + m.end() for m in _SENTENCE_SPLIT_RE.finditer(text[start:end])]
    boundaries = [b for b in boundaries if start < b < end] + [end]
    pieces: list[tuple[int, int]] = []
    prev_cut = start
    while prev_cut < end:
        piece_start = start if not pieces else max(prev_cut - overlap, start)
        limit = piece_start + budget
        if limit >= end:
            cut = end
        else:
            fitting = [b for b in boundaries if prev_cut < b <= limit]
            cut = fitting[-1] if fitting else limit         # hard cut mid-sentence
        cut = max(cut, prev_cut + 1)   # progress even if overlap >= budget
        pieces.append((piece_start, cut))
        prev_cut = cut
    if len(pieces) >= 2 and pieces[-1][1] - pieces[-1][0] < _MIN_TAIL:
        prev_s, _ = pieces[-2]
        merged_end = pieces[-1][1]
        if merged_end - prev_s <= budget + _MIN_TAIL:
            pieces[-2:] = [(prev_s, merged_end)]
    return pieces
