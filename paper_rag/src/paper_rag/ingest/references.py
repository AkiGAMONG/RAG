"""1.5 — Bibliography parsing + inline citation-marker parsing (L1, plan.md D5).

Bibliography: locate the References heading, slice the trailing region into
entries. Numbered styles ("[12] ..." and "12. ...") are parsed with sequence
discipline (each entry number must follow the previous one, which keeps
inline "[n]" inside an entry from starting a new one). Unnumbered
author-year bibliographies fall back to one-entry-per-paragraph, keyed by
position.

Each entry becomes a chunk (type="reference") so bibliography text is
retrievable, and a structured record for meta.json ("references") so inline
markers can be resolved to *which work* is cited. matched_paper_id is always
null here (L2 linkage is stretch).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .chunker import Chunk, Exclusions
from .pdf_parser import ParsedPaper, rects_for_range

_HEADING_RE = re.compile(
    r"^\s*(?:\d+\.?\s+)?(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*:?\s*$")
_STOP_RE = re.compile(r"^\s*(?:[A-Z]\.?\s+)?(Appendix|APPENDIX|Supplementary)\b")
_BRACKET_ENTRY_RE = re.compile(r"\[(\d{1,3})\]\s")
# Lookbehind (not a consuming group) so m.start() lands on the digit itself,
# never inside the "\n\n" paragraph separator — entry offsets map through
# offset_map only when they point at real paragraph text.
_NUMDOT_ENTRY_RE = re.compile(r"(?:^|(?<=\n\n))(\d{1,3})\.\s+")
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})[a-z]?\b")
_MAX_ENTRY_CHARS = 1200


@dataclass
class ReferenceEntry:
    key: str                  # "12" for numbered styles; ordinal for author-year
    raw: str
    authors: str | None
    year: int | None
    title: str | None
    page: int                 # 1-based page where the entry starts
    char_start: int           # offsets on that page's text
    char_end: int


def extract_references(parsed: ParsedPaper,
                       paper_id: str) -> tuple[list[ReferenceEntry], list[Chunk], Exclusions]:
    region = _locate_region(parsed)
    if region is None:
        return [], [], {}
    stream, offset_map, exclusions = _build_stream(parsed, region)
    entries = _split_entries(stream, offset_map)
    chunks = [c for e in entries
              if (c := _entry_chunk(parsed, paper_id, e)) is not None]
    return entries, chunks, exclusions


# --------------------------------------------------------------------------
# region location: (page_idx, paragraph_idx) of heading -> end of doc / stop

def _locate_region(parsed: ParsedPaper) -> tuple[int, int] | None:
    """Return (page_index, paragraph_index) of the LAST References heading."""
    found = None
    for pi, page in enumerate(parsed.pages):
        for qi, para in enumerate(page.paragraphs):
            if _HEADING_RE.match(page.text[para.start:para.end]):
                found = (pi, qi)
    return found


def _build_stream(parsed: ParsedPaper, region: tuple[int, int]):
    """Concatenate reference-section paragraphs into one string.

    Returns (stream, offset_map, exclusions) where offset_map maps each
    stream offset range to (page_number, page_char_offset) via a list of
    (stream_start, stream_end, page_number, page_start).
    """
    start_pi, start_qi = region
    stream_parts: list[str] = []
    offset_map: list[tuple[int, int, int, int]] = []
    exclusions: Exclusions = {}
    pos = 0
    stopped = False
    for pi in range(start_pi, len(parsed.pages)):
        page = parsed.pages[pi]
        first_q = start_qi if pi == start_pi else 0
        for qi in range(first_q, len(page.paragraphs)):
            para = page.paragraphs[qi]
            text = page.text[para.start:para.end]
            is_heading = pi == start_pi and qi == start_qi
            if not is_heading and _STOP_RE.match(text):
                stopped = True
                break
            exclusions.setdefault(page.number, []).append((para.start, para.end))
            if is_heading:
                continue  # the heading itself is excluded but not entry text
            if stream_parts:
                stream_parts.append("\n\n")
                pos += 2
            stream_parts.append(text)
            offset_map.append((pos, pos + len(text), page.number, para.start))
            pos += len(text)
        if stopped:
            break
    return "".join(stream_parts), offset_map, exclusions


def _stream_to_page(offset_map: list[tuple[int, int, int, int]],
                    stream_offset: int) -> tuple[int, int]:
    """Map a stream offset to (page_number, page_char_offset).

    Offsets falling into the "\n\n" separators between paragraphs clamp to
    the end of the nearest PRECEDING paragraph (an entry's exclusive end may
    land there)."""
    prev = None
    for s, e, page, page_start in offset_map:
        if stream_offset < s:
            break
        if stream_offset < e:
            return page, page_start + (stream_offset - s)
        prev = (s, e, page, page_start)
    if prev is not None:
        s, e, page, page_start = prev
        return page, page_start + (e - s)
    if offset_map:
        s, e, page, page_start = offset_map[0]
        return page, page_start
    return 1, 0


def _stream_end_to_page(offset_map, stream_end: int) -> tuple[int, int]:
    """Map an EXCLUSIVE stream end offset to (page, exclusive page offset).

    Ends landing in a "\n\n" separator clamp to the end of the preceding
    paragraph — never past it (the separator is not part of any entry)."""
    prev = None
    for s, e, page, page_start in offset_map:
        if stream_end <= s:
            break
        prev = (s, e, page, page_start)
        if stream_end <= e:
            return page, page_start + (stream_end - s)
    if prev is not None:
        s, e, page, page_start = prev
        return page, page_start + (e - s)
    return 1, 0


# --------------------------------------------------------------------------
# entry splitting

def _split_entries(stream: str, offset_map) -> list[ReferenceEntry]:
    if not stream.strip():
        return []
    marks = _sequence_marks(_BRACKET_ENTRY_RE, stream)
    if not marks:
        marks = _sequence_marks(_NUMDOT_ENTRY_RE, stream)
    entries: list[ReferenceEntry] = []
    if marks:
        for i, (num, start) in enumerate(marks):
            end = marks[i + 1][1] if i + 1 < len(marks) else len(stream)
            raw = re.sub(r"\s+", " ", stream[start:end]).strip()
            if not raw:
                continue
            entries.append(_make_entry(str(num), raw, start, end, offset_map))
    else:
        # Author-year style: one paragraph per entry, positional keys.
        pos = 0
        for i, para_text in enumerate(stream.split("\n\n")):
            if para_text.strip():
                entries.append(_make_entry(
                    str(len(entries) + 1),
                    re.sub(r"\s+", " ", para_text).strip(),
                    pos, pos + len(para_text), offset_map))
            pos += len(para_text) + 2
    return entries


def _sequence_marks(pattern: re.Pattern, stream: str) -> list[tuple[int, int]]:
    """(number, stream_offset) of entry starts, enforcing n, n+1, n+2, ..."""
    marks: list[tuple[int, int]] = []
    expected: int | None = None
    for m in pattern.finditer(stream):
        num = int(m.group(1))
        if expected is None:
            if num != 1:
                continue  # a bibliography starts at 1; ignore stray matches
            marks.append((num, m.start()))
            expected = 2
        elif num == expected:
            marks.append((num, m.start()))
            expected += 1
    return marks


def _make_entry(key: str, raw: str, stream_start: int, stream_end: int,
                offset_map) -> ReferenceEntry:
    page, char_start = _stream_to_page(offset_map, stream_start)
    end_page, char_end = _stream_end_to_page(offset_map, stream_end)
    if end_page != page:
        # Entry continues onto a later page; clip the range to the end of its
        # start page's portion of the reference region (contract §5: offsets
        # are into the page_start page's text).
        page_ends = [ps + (e - s) for s, e, p, ps in offset_map if p == page]
        char_end = max(page_ends) if page_ends else char_start
    authors, year, title = _parse_fields(raw)
    return ReferenceEntry(key=key, raw=raw[:_MAX_ENTRY_CHARS], authors=authors,
                          year=year, title=title, page=page,
                          char_start=char_start, char_end=char_end)


def _parse_fields(raw: str) -> tuple[str | None, int | None, str | None]:
    """Best-effort (authors, year, title) from an entry's raw text."""
    body = re.sub(r"^\[?\d{1,3}[\].]\s*", "", raw)   # strip the leading marker
    year_match = _YEAR_RE.search(body)
    year = int(year_match.group(1)) if year_match else None
    authors = None
    title = None
    if year_match:
        head = body[:year_match.start()].strip(" .,;:(")
        if head:
            authors = head[:200]
        tail = body[year_match.end():].lstrip(" .,;:)")
        sentence = re.split(r"\.(?=\s)", tail, maxsplit=1)[0].strip()
        if sentence:
            title = sentence[:300]
    return authors, year, title


def _entry_chunk(parsed: ParsedPaper, paper_id: str,
                 entry: ReferenceEntry) -> Chunk | None:
    """The retrievable chunk for an entry: an EXACT slice of its start page's
    text (the invariant every chunk type obeys — contract §5/§6). The
    normalized full raw text lives only in meta.json's references record."""
    page = next(p for p in parsed.pages if p.number == entry.page)
    char_start = min(entry.char_start, len(page.text))
    char_end = min(entry.char_end, len(page.text))
    text = page.text[char_start:char_end]
    if not text.strip():
        return None   # degenerate range (e.g. entry fully on a later page)
    return Chunk(
        chunk_id=f"{paper_id}:r{entry.key}",
        paper_id=paper_id, type="reference", text=text,
        page_start=entry.page, page_end=entry.page,
        char_start=char_start, char_end=char_end,
        rects=rects_for_range(page, char_start, char_end))


# --------------------------------------------------------------------------
# inline citation markers (used by retrieval; L1)

_INLINE_NUMERIC_RE = re.compile(r"\[(\d{1,3}(?:\s*[,;–-]\s*\d{1,3})*)\]")
_PAREN_AUTHOR_YEAR_RE = re.compile(
    r"\(\s*([A-Z][\w'’\-]+(?:\s+(?:and|&)\s+[A-Z][\w'’\-]+)?(?:\s+et\s+al\.?)?)"
    r"\s*,?\s+((?:19|20)\d{2})[a-z]?\s*\)")
_NARRATIVE_AUTHOR_YEAR_RE = re.compile(
    r"([A-Z][\w'’\-]+(?:\s+(?:and|&)\s+[A-Z][\w'’\-]+)?(?:\s+et\s+al\.?)?)"
    r"\s*\(\s*((?:19|20)\d{2})[a-z]?\s*\)")


def parse_numeric_markers(text: str) -> list[int]:
    """[12], [3, 5], [3-6] -> sorted unique entry numbers."""
    numbers: set[int] = set()
    for group in _INLINE_NUMERIC_RE.findall(text):
        parts = re.split(r"[,;]", group)
        for part in parts:
            rng = re.split(r"[–-]", part)
            if len(rng) == 2 and all(x.strip().isdigit() for x in rng):
                lo, hi = int(rng[0]), int(rng[1])
                if 0 < hi - lo <= 15:
                    numbers.update(range(lo, hi + 1))
                    continue
            if part.strip().isdigit():
                numbers.add(int(part.strip()))
    return sorted(n for n in numbers if n < 1000)


def parse_author_year_markers(text: str) -> list[tuple[str, int]]:
    """"(Khattab & Zaharia, 2020)" / "Khattab and Zaharia (2020)" mentions."""
    out: list[tuple[str, int]] = []
    for regex in (_PAREN_AUTHOR_YEAR_RE, _NARRATIVE_AUTHOR_YEAR_RE):
        for name, year in regex.findall(text):
            item = (name.strip(), int(year))
            if item not in out:
                out.append(item)
    return out


def resolve_inline_citations(text: str, references: dict[str, dict]) -> list[tuple[str, dict]]:
    """Match inline markers in `text` against a paper's references dict.

    `references` is the meta.json shape: key -> {raw, authors, year, ...}.
    Returns (key, entry) pairs in mention order, deduped.
    """
    resolved: list[tuple[str, dict]] = []

    def add(key: str, entry: dict) -> None:
        if all(k != key for k, _ in resolved):
            resolved.append((key, entry))

    for num in parse_numeric_markers(text):
        entry = references.get(str(num))
        if entry:
            add(str(num), entry)
    for name, year in parse_author_year_markers(text):
        surname = re.split(r"\s+(?:and|&|et)\s+", name)[0].strip().lower()
        for key, entry in references.items():
            authors = (entry.get("authors") or "").lower()
            if entry.get("year") == year and surname and surname in authors:
                add(key, entry)
                break
    return resolved
