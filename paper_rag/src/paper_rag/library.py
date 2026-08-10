"""PaperLibrary — the facade (api_contract §2). Wires the parts, holds almost
no logic of its own (plan.md §6).

ingest():  pdf_parser -> figures -> references -> chunker -> Embedder ->
           vector_store + meta.json/pages/source.pdf. Blocking, minutes-long
           under free-tier limits; reports via the progress callback (§1.6).
ask():     retrieval -> generation -> Answer.
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .config import Config
from .errors import (FIGURE_NOT_FOUND, INGEST_FAILED, NO_PAPERS,
                     PAPER_NOT_FOUND, PaperRagError)
from .generation import Generator
from .ingest import chunk_pages, extract_figures, extract_references, parse_pdf
from .models import (Anchor, Answer, FigureAsset, FigureRef, IngestProgress,
                     PaperInfo, ProgressFn, Scope, Turn)
from .providers import create_embedder, create_llm
from .retrieval import Retriever
from .store import PaperMetaStore, VectorStore

_EMBED_PROGRESS_BATCH = 20


class PaperLibrary:
    def __init__(self, data_dir: str = "./paper_rag_data",
                 config: Config | None = None):
        load_dotenv()   # keys live in .env / the environment, never in code
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or Config.from_env()
        self.metas = PaperMetaStore(self.data_dir)
        try:
            self.store = VectorStore(self.data_dir,
                                     self.config.embedder_fingerprint())
        except PaperRagError as exc:
            if exc.code == "EMBEDDER_MISMATCH":
                exc.details["papers"] = [
                    {"paper_id": m.get("paper_id"), "embedder": m.get("embedder")}
                    for m in self.metas.list_metas()]
                # The constructor raises (contract §5), so re-ingest through
                # this instance is impossible — give the manual escape hatch.
                exc.details["fix"] = (
                    f"either restore the old embed config, or delete "
                    f"{self.data_dir / 'chroma'} to drop the stale index and "
                    f"re-run ingest() per paper (each paper dir keeps a "
                    f"source.pdf copy)")
            raise
        self._embedder = None
        self._llm = None

    # -- providers are created lazily so papers()/get_figure() work without keys
    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = create_embedder(self.config)
        return self._embedder

    @property
    def llm(self):
        if self._llm is None:
            self._llm = create_llm(self.config)
        return self._llm

    # ------------------------------------------------------------------ ingest
    def ingest(self, pdf_path: str, *, paper_id: str | None = None,
               progress: ProgressFn | None = None) -> str:
        """Parse, chunk, embed and index one PDF. Returns paper_id.
        Idempotent per paper_id (re-ingest replaces)."""
        src = Path(pdf_path)
        if not src.is_file():
            raise PaperRagError(INGEST_FAILED, f"File not found: {pdf_path}",
                                {"path": str(pdf_path)})
        pid = paper_id or self._derive_paper_id(src)

        def report(stage: str, done: int, total: int) -> None:
            if progress is not None:
                progress(IngestProgress(stage=stage, done=done, total=total))

        report("parsing", 0, 1)
        parsed = parse_pdf(str(src))
        report("parsing", parsed.n_pages, parsed.n_pages)

        # All artifacts are staged in a temp dir first; the previous version
        # of the paper is only removed AFTER the expensive/fragile stages
        # (notably rate-limited embedding) have succeeded, so a failed
        # re-ingest leaves the old data intact.
        paper_dir = self.metas.paper_dir(pid)
        staging_dir = paper_dir.with_name(paper_dir.name + ".tmp")
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

        report("figures", 0, 1)
        figure_records, caption_chunks, fig_excl = extract_figures(
            str(src), parsed, staging_dir, pid)
        n_figs = max(len(figure_records), 1)
        report("figures", n_figs, n_figs)   # done == total even with 0 figures

        report("references", 0, 1)
        ref_entries, ref_chunks, ref_excl = extract_references(parsed, pid)
        report("references", 1, 1)

        exclusions = {**fig_excl}
        for page, ranges in ref_excl.items():
            exclusions.setdefault(page, []).extend(ranges)
        body_chunks = chunk_pages(pid, parsed.pages, self.config, exclusions)
        chunks = body_chunks + caption_chunks + ref_chunks
        if not chunks:
            raise PaperRagError(INGEST_FAILED, "No chunks extracted from PDF",
                                {"path": str(pdf_path)})

        embeddings: list[list[float]] = []
        texts = [c.text for c in chunks]
        report("embedding", 0, len(texts))
        for i in range(0, len(texts), _EMBED_PROGRESS_BATCH):
            embeddings.extend(
                self.embedder.embed_documents(texts[i:i + _EMBED_PROGRESS_BATCH]))
            report("embedding", min(i + _EMBED_PROGRESS_BATCH, len(texts)), len(texts))

        report("indexing", 0, len(chunks))
        # The source PDF is copied into staging BEFORE the swap: `src` may
        # itself be the stored copy data_dir/papers/<pid>/source.pdf (the
        # contract's self-contained re-ingest flow, §5), which the swap is
        # about to delete.
        try:
            staging_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, staging_dir / "source.pdf")
        except OSError as exc:
            raise PaperRagError(INGEST_FAILED,
                                f"Cannot stage source PDF: {exc}",
                                {"path": str(src)}) from exc
        # Swap point: replace the old version, then index. meta.json is
        # written LAST — it is the commit marker papers() relies on, so a
        # failure part-way leaves no half-registered paper.
        self.store.delete_paper(pid)
        self.metas.delete_paper(pid)
        staging_dir.rename(paper_dir)
        for page in parsed.pages:
            self.metas.write_page_text(pid, page.number, page.text)
        self.store.add_chunks(chunks, embeddings)
        self.metas.write_meta(pid, {
            "paper_id": pid,
            "title": parsed.title or src.stem,
            "source_file": str(src),
            "source_copy": "source.pdf",
            "embedder": self.config.embedder_fingerprint(),
            "n_pages": parsed.n_pages,
            "n_chunks": len(chunks),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "figures": {
                f.figure_id: {"label": f.label, "caption": f.caption,
                              "page": f.page, "image_path": f.image_path,
                              "rect": f.rect, "full_page": f.full_page}
                for f in figure_records},
            "references": {
                e.key: {"raw": e.raw, "authors": e.authors, "year": e.year,
                        "title": e.title, "matched_paper_id": None}
                for e in ref_entries},
        })
        report("indexing", len(chunks), len(chunks))
        return pid

    @staticmethod
    def _derive_paper_id(src: Path) -> str:
        digest = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
        return f"p{digest}"

    # ------------------------------------------------------------------ read API
    def papers(self) -> list[PaperInfo]:
        return [PaperInfo(
                    paper_id=m["paper_id"], title=m.get("title", ""),
                    n_pages=int(m.get("n_pages", 0)),
                    n_chunks=int(m.get("n_chunks", 0)),
                    n_figures=len(m.get("figures", {})),
                    ingested_at=m.get("ingested_at", ""))
                for m in self.metas.list_metas()]

    def ask(self, question: str, *, anchor: Anchor | None = None,
            scope: Scope = Scope.AUTO, history: list[Turn] | None = None,
            top_k: int = 8) -> Answer:
        # `history` is reserved in the signature (D9); implemented in stretch.
        if not self.metas.list_metas():
            raise PaperRagError(NO_PAPERS, "ask() on an empty library — "
                                           "ingest a paper first")
        retriever = Retriever(self.store, self.metas, self.embedder,
                              self.config, llm=self.llm)
        retrieval = retriever.retrieve(question, anchor, scope, top_k)
        answer = Generator(self.llm).generate(
            question, retrieval, self._load_figure_bytes)
        if history:
            answer.debug["history"] = "ignored (multi-turn is a stretch goal, D9)"
        return answer

    def get_figure(self, figure_id: str) -> FigureAsset:
        located = self._locate_figure(figure_id)
        if located is None:
            raise PaperRagError(FIGURE_NOT_FOUND,
                                f"Unknown figure_id {figure_id!r}",
                                {"figure_id": figure_id})
        paper_id, local_id, record = located
        image = self.metas.paper_dir(paper_id) / record["image_path"]
        if not image.is_file():
            raise PaperRagError(FIGURE_NOT_FOUND,
                                f"Image file missing for {figure_id!r}",
                                {"figure_id": figure_id})
        ref = FigureRef(figure_id=f"{paper_id}:{local_id}", paper_id=paper_id,
                        label=record["label"], caption=record["caption"],
                        page=record["page"], image_path=str(image))
        return FigureAsset(ref=ref, mime="image/png", data=image.read_bytes())

    def _locate_figure(self, figure_id: str):
        """Accepts global ids ("<paper_id>:fig:3"); a bare local id ("fig:3")
        also resolves when it is unambiguous across the library."""
        matches = []
        for meta in self.metas.list_metas():
            pid = meta["paper_id"]
            for local_id, record in meta.get("figures", {}).items():
                if figure_id == f"{pid}:{local_id}" or figure_id == local_id:
                    matches.append((pid, local_id, record))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise PaperRagError(
                FIGURE_NOT_FOUND,
                f"figure_id {figure_id!r} is ambiguous across papers; use the "
                f"paper-qualified id",
                {"figure_id": figure_id,
                 "candidates": [f"{p}:{l}" for p, l, _ in matches]})
        return None

    def _load_figure_bytes(self, figure: FigureRef) -> bytes | None:
        path = Path(figure.image_path)
        try:
            return path.read_bytes()
        except OSError:
            return None
