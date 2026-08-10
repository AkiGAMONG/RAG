"""Public data model — api_contract.md §3 and §4, verbatim.

Everything here is a plain dataclass (or str enum) that serializes 1:1 to
JSON via ``dataclasses.asdict`` — no numpy arrays, no provider SDK types
(contract §1, "JSON-clean boundary"). The only exception is
``FigureAsset.data`` (raw bytes), which the contract explicitly designates
as the single place raw bytes appear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Literal


@dataclass
class Anchor:
    paper_id: str                  # required: which paper the user is reading
    page: int | None = None        # 1-based
    selection: str | None = None   # verbatim selected text (may be empty)
    # Three information tiers, by how much is filled in:
    #   paper_id only        -> general question about this paper
    #   + page               -> question about this page
    #   + selection          -> "what does this passage mean" — selection feeds
    #                           query rewriting AND anchor-chunk resolution.
    # The anchor is an input SIGNAL (what the question is about), never a hard
    # restriction on where answers may come from (decision D1 in plan.md).


@dataclass
class Turn:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class Rect:                        # api_contract §6 for coordinate semantics
    page: int
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class Citation:
    marker: int                    # 1-based, as printed in text
    chunk_id: str
    paper_id: str
    page: int                      # 1-based, page where the quote starts
    quote: str                     # verbatim supporting text (may be truncated)
    rects: list[Rect]              # for highlight; may be empty if unresolved


@dataclass
class FigureRef:
    figure_id: str
    paper_id: str
    label: str                     # "Figure 3" / "Table 2"
    caption: str
    page: int
    image_path: str


@dataclass
class FigureAsset:                 # FigureRef + the bytes
    ref: FigureRef
    mime: str                      # "image/png"
    data: bytes                    # only place raw bytes appear; wire = endpoint


@dataclass
class Answer:
    text: str                      # contains inline markers like [1], [2]
    citations: list[Citation]      # marker n <-> citations[n-1]
    figures: list[FigureRef]       # figures actually shown to the model
    abstained: bool                # True iff ABSTAIN_MARKER found in text
    scope_used: str                # "paper:<id>" | "library"
    debug: dict = field(default_factory=dict)


@dataclass
class PaperInfo:
    paper_id: str
    title: str                     # best-effort extraction; falls back to filename
    n_pages: int
    n_chunks: int
    n_figures: int
    ingested_at: str               # ISO 8601


@dataclass
class IngestProgress:
    stage: Literal["parsing", "figures", "references", "embedding", "indexing"]
    done: int
    total: int


ProgressFn = Callable[[IngestProgress], None]


class Scope(str, Enum):            # api_contract §4
    AUTO = "auto"        # anchor present -> PAPER(anchor.paper_id); else LIBRARY
    PAPER = "paper"      # requires anchor.paper_id; INVALID_ANCHOR otherwise
    LIBRARY = "library"  # whole collection; citations must carry paper_id
