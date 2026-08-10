"""1.7 — Anchor resolution + scope pre-filtering + reference resolution.

One pipeline, two optional inputs (D1): the anchor is a SIGNAL for what the
question is about, never a hard restriction on where answers come from.

Steps (api_contract §11):
  1. resolve scope (AUTO -> PAPER when anchored, else LIBRARY; §4);
  2. resolve the anchor selection to its chunk(s) — exact match against the
     stored page text, then normalized-whitespace fuzzy match (§6);
  3. rewrite the question into standalone retrieval queries using the
     selection (deterministic in MVP: no LLM, history-aware rewriting is
     stretch per D9);
  4. retrieve with a paper_id metadata pre-filter (D2);
  5. resolve figure/table mentions (regex first, LLM fallback — D4) and
     inline citations (L1 — D5) found in the question / selection / anchored
     chunks.

The result object carries everything generation needs; generation decides
final context ordering and deduplication.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .config import Config
from .errors import INVALID_ANCHOR, PAPER_NOT_FOUND, PaperRagError
from .ingest.figures import find_figure_mentions
from .ingest.references import resolve_inline_citations
from .models import Anchor, FigureRef, Scope
from .providers.base import LLMClient, TextPart
from .store import PaperMetaStore, VectorStore

_MAX_ANCHORED_CHUNKS = 3
_MAX_FIGURES = 3
_MAX_CITATION_NOTES = 5
_SELECTION_QUERY_CHARS = 500
_FIGURE_WORDS_RE = re.compile(r"\b(figure|figures|fig|table|tables|diagram|plot|chart)\b|图|表",
                              re.IGNORECASE)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    distance: float | None = None
    anchored: bool = False


@dataclass
class RetrievalResult:
    scope_used: str                       # "paper:<id>" | "library"
    queries: list[str]
    anchored_chunks: list[RetrievedChunk]
    retrieved_chunks: list[RetrievedChunk]
    reference_chunks: list[RetrievedChunk]      # resolved inline citations (L1)
    figures: list[FigureRef]              # figures to attach as images
    debug: dict = field(default_factory=dict)


class Retriever:
    def __init__(self, store: VectorStore, metas: PaperMetaStore,
                 embedder, config: Config, llm: LLMClient | None = None):
        self.store = store
        self.metas = metas
        self.embedder = embedder
        self.config = config
        self.llm = llm    # only used for the D4 figure-reference fallback

    # ------------------------------------------------------------------ main
    def retrieve(self, question: str, anchor: Anchor | None,
                 scope: Scope, top_k: int) -> RetrievalResult:
        t0 = time.monotonic()
        debug: dict = {}
        paper_filter = self._resolve_scope(anchor, scope)
        scope_used = f"paper:{paper_filter}" if paper_filter else "library"

        anchored_chunks, anchor_debug = self._resolve_anchor(anchor)
        debug["anchor"] = anchor_debug

        queries = self._rewrite_queries(question, anchor)
        debug["queries"] = queries
        retrieved = self._search(queries, top_k, paper_filter)

        figures, figure_debug = self._resolve_figures(
            question, anchor, anchored_chunks, retrieved, paper_filter)
        debug["figures"] = figure_debug

        reference_chunks, citation_debug = self._resolve_citations(
            anchor, anchored_chunks, paper_filter)
        debug["inline_citations"] = citation_debug
        debug["timings"] = {"retrieval_s": round(time.monotonic() - t0, 3)}

        return RetrievalResult(
            scope_used=scope_used, queries=queries,
            anchored_chunks=anchored_chunks, retrieved_chunks=retrieved,
            reference_chunks=reference_chunks, figures=figures, debug=debug)

    # ------------------------------------------------------------------ scope
    def _resolve_scope(self, anchor: Anchor | None, scope: Scope) -> str | None:
        """Returns the paper_id to pre-filter on, or None for library-wide."""
        if scope == Scope.PAPER:
            if anchor is None or not anchor.paper_id:
                raise PaperRagError(INVALID_ANCHOR,
                                    "scope=PAPER requires an anchor with paper_id")
            self._require_paper(anchor.paper_id)
            return anchor.paper_id
        if scope == Scope.AUTO and anchor is not None and anchor.paper_id:
            self._require_paper(anchor.paper_id)
            return anchor.paper_id
        if anchor is not None and anchor.paper_id:
            self._require_paper(anchor.paper_id)   # LIBRARY scope, anchored
        return None

    def _require_paper(self, paper_id: str) -> dict:
        meta = self.metas.read_meta(paper_id)
        if meta is None:
            raise PaperRagError(PAPER_NOT_FOUND, f"Unknown paper_id {paper_id!r}",
                                {"paper_id": paper_id})
        return meta

    # ------------------------------------------------------------------ anchor
    def _resolve_anchor(self, anchor: Anchor | None) -> tuple[list[RetrievedChunk], dict]:
        if anchor is None:
            return [], {"present": False}
        info: dict = {"present": True, "paper_id": anchor.paper_id,
                      "page": anchor.page, "match": None}
        selection = (anchor.selection or "").strip()
        if not selection:
            return [], info

        pages = ([anchor.page] if anchor.page else
                 self._all_pages(anchor.paper_id))
        for page_number in pages:
            page_text = self.metas.read_page_text(anchor.paper_id, page_number)
            if page_text is None:
                continue
            span = _match_selection(page_text, selection)
            if span is None:
                continue
            info["match"] = {"page": page_number, "start": span[0], "end": span[1],
                            "kind": span[2]}
            chunks = self._chunks_covering(anchor.paper_id, page_number, span[0], span[1])
            return chunks, info
        info["match"] = "miss"   # degrade gracefully (api_contract §6)
        return [], info

    def _all_pages(self, paper_id: str) -> list[int]:
        meta = self.metas.read_meta(paper_id) or {}
        return list(range(1, int(meta.get("n_pages", 0)) + 1))

    def _chunks_covering(self, paper_id: str, page: int,
                         start: int, end: int) -> list[RetrievedChunk]:
        rows = self.store.get_page_chunks(paper_id, page)
        hits = [r for r in rows
                if r["metadata"]["char_start"] < end
                and r["metadata"]["char_end"] > start]
        hits.sort(key=lambda r: r["metadata"]["char_start"])
        return [RetrievedChunk(chunk_id=r["chunk_id"], text=r["text"],
                               metadata=r["metadata"], anchored=True)
                for r in hits[:_MAX_ANCHORED_CHUNKS]]

    # ------------------------------------------------------------------ queries
    @staticmethod
    def _rewrite_queries(question: str, anchor: Anchor | None) -> list[str]:
        """Deterministic MVP rewriting: the selection tells retrieval what the
        question is about even when the question alone is context-free."""
        queries = [question.strip()]
        selection = (anchor.selection or "").strip() if anchor else ""
        if selection:
            queries.append(f"{question.strip()}\n{selection[:_SELECTION_QUERY_CHARS]}")
        return queries

    def _search(self, queries: list[str], top_k: int,
                paper_filter: str | None) -> list[RetrievedChunk]:
        best: dict[str, dict] = {}
        for query in queries:
            embedding = self.embedder.embed_query(query)
            for row in self.store.query(embedding, top_k, paper_filter):
                seen = best.get(row["chunk_id"])
                if seen is None or row["distance"] < seen["distance"]:
                    best[row["chunk_id"]] = row
        rows = sorted(best.values(), key=lambda r: r["distance"])[:top_k]
        return [RetrievedChunk(chunk_id=r["chunk_id"], text=r["text"],
                               metadata=r["metadata"], distance=r["distance"])
                for r in rows]

    # ------------------------------------------------------------------ figures
    def _resolve_figures(self, question: str, anchor: Anchor | None,
                         anchored: list[RetrievedChunk],
                         retrieved: list[RetrievedChunk],
                         paper_filter: str | None) -> tuple[list[FigureRef], dict]:
        refs: list[FigureRef] = []
        dbg: dict = {"regex": [], "llm_fallback": None, "from_captions": []}

        def add(paper_id: str, local_id: str) -> None:
            if len(refs) >= _MAX_FIGURES:
                return
            ref = self._figure_ref(paper_id, local_id)
            if ref and all(r.figure_id != ref.figure_id for r in refs):
                refs.append(ref)

        # 1) explicit "Figure 3" mentions in question / selection / anchored text
        mention_text = question
        if anchor and anchor.selection:
            mention_text += "\n" + anchor.selection
        mention_text += "\n" + "\n".join(c.text for c in anchored)
        mentioned = find_figure_mentions(mention_text)
        dbg["regex"] = mentioned
        target_papers = self._candidate_papers(anchor, paper_filter, retrieved)
        for local_id in mentioned:
            for pid in target_papers:
                if self._figure_exists(pid, local_id):
                    add(pid, local_id)
                    break

        # 2) LLM fallback for indirect mentions (D4), only when regex found nothing
        if not mentioned and self.llm is not None and _FIGURE_WORDS_RE.search(question):
            local = self._llm_figure_fallback(question, target_papers)
            dbg["llm_fallback"] = local
            if local:
                add(local[0], local[1])

        # 3) caption chunks that retrieval surfaced are linked figures (§11) —
        #    used only when nothing explicit was resolved, so an explicitly
        #    named figure is never diluted by unrelated captions.
        explicit_found = bool(refs)
        for chunk in retrieved:
            if chunk.metadata.get("type") == "caption" and chunk.metadata.get("figure_id"):
                dbg["from_captions"].append(chunk.metadata["figure_id"])
                if not explicit_found:
                    add(chunk.metadata["paper_id"], chunk.metadata["figure_id"])
        return refs, dbg

    def _candidate_papers(self, anchor: Anchor | None, paper_filter: str | None,
                          retrieved: list[RetrievedChunk]) -> list[str]:
        if paper_filter:
            return [paper_filter]
        if anchor and anchor.paper_id:
            return [anchor.paper_id]
        out: list[str] = []
        for chunk in retrieved:
            pid = chunk.metadata.get("paper_id")
            if pid and pid not in out:
                out.append(pid)
        return out

    def _figure_exists(self, paper_id: str, local_id: str) -> bool:
        meta = self.metas.read_meta(paper_id) or {}
        return local_id in meta.get("figures", {})

    def _figure_ref(self, paper_id: str, local_id: str) -> FigureRef | None:
        meta = self.metas.read_meta(paper_id) or {}
        record = meta.get("figures", {}).get(local_id)
        if not record:
            return None
        return FigureRef(
            figure_id=f"{paper_id}:{local_id}",     # globally unique public id
            paper_id=paper_id, label=record["label"],
            caption=record["caption"], page=record["page"],
            image_path=str(self.metas.paper_dir(paper_id) / record["image_path"]))

    def _llm_figure_fallback(self, question: str,
                             papers: list[str]) -> tuple[str, str] | None:
        """Ask the LLM which known figure an indirect mention refers to."""
        options: list[tuple[str, str, str]] = []   # (paper_id, local_id, line)
        for pid in papers:
            meta = self.metas.read_meta(pid) or {}
            for local_id, rec in meta.get("figures", {}).items():
                options.append((pid, local_id,
                                f"{pid}|{local_id}: {rec['label']} — "
                                f"{rec['caption'][:120]}"))
        if not options:
            return None
        prompt = ("A reader asked a question that seems to reference a figure or "
                  "table indirectly. Pick which ONE of the listed figures it "
                  "refers to, or answer NONE.\n\n"
                  f"QUESTION: {question}\n\nFIGURES:\n" +
                  "\n".join(line for _, _, line in options) +
                  "\n\nAnswer with exactly the `paper_id|figure_id` prefix of "
                  "one line, or NONE.")
        try:
            reply = self.llm.complete(
                "You resolve figure references. Output only the identifier or NONE.",
                [TextPart(text=prompt)], temperature=0.0).strip()
        except PaperRagError:
            return None   # fallback is best-effort; never fail retrieval on it
        # Longest key first: "p1|fig:1" is a substring of "p1|fig:12", so a
        # naive first-match would resolve the wrong figure.
        matches = [(pid, local_id) for pid, local_id, _ in options
                   if f"{pid}|{local_id}" in reply]
        if matches:
            return max(matches, key=lambda m: len(f"{m[0]}|{m[1]}"))
        return None

    # ------------------------------------------------------------------ inline citations
    def _resolve_citations(self, anchor: Anchor | None,
                           anchored: list[RetrievedChunk],
                           paper_filter: str | None) -> tuple[list[RetrievedChunk], dict]:
        """L1: inline markers in the selection/anchored text -> reference chunks."""
        source_text = ""
        if anchor and anchor.selection:
            source_text += anchor.selection + "\n"
        source_text += "\n".join(c.text for c in anchored)
        if not source_text.strip():
            return [], {"resolved": []}
        paper_id = paper_filter or (anchor.paper_id if anchor else None)
        if not paper_id:
            return [], {"resolved": []}
        meta = self.metas.read_meta(paper_id) or {}
        resolved = resolve_inline_citations(source_text, meta.get("references", {}))
        resolved = resolved[:_MAX_CITATION_NOTES]
        chunk_ids = [f"{paper_id}:r{key}" for key, _ in resolved]
        rows = self.store.get_by_ids(chunk_ids)
        chunks = [RetrievedChunk(chunk_id=r["chunk_id"], text=r["text"],
                                 metadata=r["metadata"]) for r in rows]
        return chunks, {"resolved": [key for key, _ in resolved]}


# --------------------------------------------------------------------------
# selection matching (api_contract §6)

def _match_selection(page_text: str, selection: str) -> tuple[int, int, str] | None:
    """Exact match, then normalized-whitespace fuzzy match. Returns
    (start, end, kind) in page-text offsets, or None."""
    idx = page_text.find(selection)
    if idx >= 0:
        return idx, idx + len(selection), "exact"
    # Fuzzy: collapse all whitespace runs, then map the match back through a
    # position table.
    norm_chars: list[str] = []
    positions: list[int] = []
    prev_space = True
    for i, ch in enumerate(page_text):
        if ch.isspace():
            if not prev_space:
                norm_chars.append(" ")
                positions.append(i)
            prev_space = True
        else:
            norm_chars.append(ch)
            positions.append(i)
            prev_space = False
    norm_text = "".join(norm_chars)
    norm_sel = re.sub(r"\s+", " ", selection).strip()
    if not norm_sel:
        return None
    idx = norm_text.find(norm_sel)
    if idx < 0:
        return None
    start = positions[idx]
    end = positions[idx + len(norm_sel) - 1] + 1
    return start, end, "fuzzy"
