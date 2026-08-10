"""1.9 — end-to-end smoke test (plan.md §3): ingest papers, ask the four
question types (concept / figure reference / anchored / negative control),
write everything to smoke_output/. Corpus-agnostic: questions are built from
whatever papers were ingested.

    uv run python scripts/run_smoke.py                       # default corpus (papers/)
    uv run python scripts/run_smoke.py --papers ../test_paper
    uv run python scripts/run_smoke.py --fake                # force offline providers

Providers come from PAPER_RAG_* env / .env (e.g. Claude generation + Gemini
embeddings). Embedding ladder (D17): configured model -> gemini-embedding-001
-> fake. The run records the plan.md §4 verification probes:
  1. does the configured generation model accept image input?
  2. can this account call gemini-embedding-2?
API keys are only ever read by the SDKs from the environment; this script
never prints them. The smoke index under smoke_output/ is disposable: a
stale embedder fingerprint wipes and rebuilds it automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from paper_rag import Anchor, Config, PaperLibrary, PaperRagError  # noqa: E402
from paper_rag.config import FALLBACK_EMBED_MODEL  # noqa: E402

OUT_DIR = ROOT / "smoke_output"
DATA_DIR = OUT_DIR / "library_data"
DEFAULT_PAPERS = [ROOT.parent / "papers" / "ColBERT.pdf",
                  ROOT.parent / "papers" / "ColPali.pdf"]


def fake_config() -> Config:
    return Config(llm_provider="fake", llm_model="fake-llm",
                  embed_provider="fake", embed_model="fake-embed", embed_dim=32)


def _missing_keys(config: Config) -> list[str]:
    needed = set()
    if "gemini" in (config.embed_provider, config.llm_provider):
        needed.add("GOOGLE_API_KEY")
    if config.llm_provider == "anthropic":
        needed.add("ANTHROPIC_API_KEY")
    return sorted(k for k in needed if not os.environ.get(k))


def pick_config(force_fake: bool, notes: dict) -> Config:
    """Choose the best available provider config, recording why (D17)."""
    if force_fake:
        notes["provider_decision"] = "forced fake via --fake"
        return fake_config()
    base = Config.from_env()   # PAPER_RAG_* selections apply
    missing = _missing_keys(base)
    if missing:
        notes["provider_decision"] = (
            f"missing {'/'.join(missing)} — real-API smoke impossible; "
            f"falling back to fake providers")
        return fake_config()
    if base.embed_provider != "gemini":
        notes["provider_decision"] = (f"real API: llm={base.llm_provider}"
                                      f":{base.llm_model} embed={base.embed_model}")
        return base
    from paper_rag.providers import create_embedder
    for model in dict.fromkeys([base.embed_model, FALLBACK_EMBED_MODEL]):
        config = Config.from_env()
        config.embed_model = model
        try:
            create_embedder(config).embed_documents(["availability probe"])
            if base.embed_model == "gemini-embedding-2":
                notes["embedding_2_available"] = (model == base.embed_model)
            notes["provider_decision"] = (f"real API: llm={config.llm_provider}"
                                          f":{config.llm_model} embed={model}")
            return config
        except PaperRagError as exc:
            notes.setdefault("embed_probe_errors", {})[model] = \
                f"{exc.code}: {exc.message}"
    notes["provider_decision"] = ("real API unusable for embeddings "
                                  "(see embed_probe_errors); using fake")
    return fake_config()


def open_library(config: Config, notes: dict) -> PaperLibrary:
    try:
        return PaperLibrary(data_dir=str(DATA_DIR), config=config)
    except PaperRagError as exc:
        if exc.code != "EMBEDDER_MISMATCH":
            raise
        # The smoke index is disposable; a fingerprint change just rebuilds it.
        notes["index_wiped"] = (f"stale embedder {exc.details.get('stored')!r} "
                                f"-> rebuilding with {exc.details.get('configured')!r}")
        print(f"wiping stale smoke index: {notes['index_wiped']}")
        shutil.rmtree(DATA_DIR, ignore_errors=True)
        try:
            # chroma caches clients per path in-process; without this the
            # rebuilt library would still see the deleted collection.
            from chromadb.api.client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
        except Exception:
            pass
        return PaperLibrary(data_dir=str(DATA_DIR), config=config)


def find_selection(lib: PaperLibrary, paper_id: str) -> tuple[int, str]:
    """Pick a real mid-document sentence to use as the anchor selection —
    mimics a reader selecting text. Returns exact page-text substrings so
    exact-match anchor resolution is exercised."""
    info = next(p for p in lib.papers() if p.paper_id == paper_id)
    for page in range(max(1, info.n_pages // 3), info.n_pages + 1):
        text = lib.metas.read_page_text(paper_id, page) or ""
        for para in text.split("\n\n"):
            if len(para) >= 150 and para[:1].isupper():
                cut = para.find(". ", 60)
                return page, (para[:cut + 1] if 0 < cut < 350 else para[:200])
    return 1, (lib.metas.read_page_text(paper_id, 1) or "")[:150]


def build_questions(lib: PaperLibrary, ingested: list[str]) -> list[dict]:
    first = ingested[0]
    fig_spec = None
    for pid in ingested:
        figures = (lib.metas.read_meta(pid) or {}).get("figures", {})
        if figures:
            fig_spec = (pid, list(figures.values())[0]["label"])
            break
    anchor_page, selection = find_selection(lib, first)
    questions = [
        {"kind": "concept",
         "question": "What is the main contribution of this paper?",
         "kwargs": {"anchor": Anchor(paper_id=first)}},
    ]
    if fig_spec is not None:
        questions.append(
            {"kind": "figure",
             "question": f"What does {fig_spec[1]} show?",
             "kwargs": {"anchor": Anchor(paper_id=fig_spec[0])}})
    questions += [
        {"kind": "anchored",
         "question": "这段话在说什么？请解释其中的关键概念。",
         "kwargs": {"anchor": Anchor(paper_id=first, page=anchor_page,
                                     selection=selection)}},
        {"kind": "negative",
         "question": "What is the best recipe for chocolate cake?",
         "kwargs": {}},
    ]
    return questions


def run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--papers", help="directory of PDFs to ingest "
                                         "(default: ../papers ColBERT+ColPali)")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    load_dotenv()
    OUT_DIR.mkdir(exist_ok=True)

    pdfs = (sorted(Path(args.papers).glob("*.pdf")) if args.papers
            else DEFAULT_PAPERS)
    if not pdfs:
        print(f"no PDFs found in {args.papers}")
        return 1

    notes: dict = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "corpus": [p.name for p in pdfs]}
    config = pick_config(args.fake, notes)
    print(f"provider decision: {notes['provider_decision']}")
    lib = open_library(config, notes)

    ingested: list[str] = []
    for pdf in pdfs:
        t0 = time.monotonic()
        pid = lib.ingest(str(pdf), progress=lambda p: print(
            f"  [{pdf.name}] {p.stage}: {p.done}/{p.total}".ljust(60), end="\r"))
        print()
        info = next(p for p in lib.papers() if p.paper_id == pid)
        ingested.append(pid)
        notes.setdefault("ingest", {})[pdf.name] = {
            "paper_id": pid, "seconds": round(time.monotonic() - t0, 1),
            "n_pages": info.n_pages, "n_chunks": info.n_chunks,
            "n_figures": info.n_figures, "title": info.title}
        print(f"  {pdf.name}: {pid} pages={info.n_pages} "
              f"chunks={info.n_chunks} figures={info.n_figures}")

    results = []
    for spec in build_questions(lib, ingested):
        print(f"\n=== [{spec['kind']}] {spec['question']}")
        record: dict = {"kind": spec["kind"], "question": spec["question"]}
        anchor = spec["kwargs"].get("anchor")
        if anchor is not None:
            record["anchor"] = asdict(anchor)
        try:
            answer = lib.ask(spec["question"], **spec["kwargs"])
            record["answer"] = asdict(answer)
            print(answer.text[:600])
            print(f"    citations={len(answer.citations)} "
                  f"figures={[f.figure_id for f in answer.figures]} "
                  f"abstained={answer.abstained} scope={answer.scope_used}")
            if spec["kind"] == "figure":
                notes["generation_image_input_ok"] = bool(answer.figures)
        except PaperRagError as exc:
            record["error"] = {"code": exc.code, "message": exc.message}
            print(f"    ERROR {exc.code}: {exc.message}")
            if spec["kind"] == "figure":
                notes["generation_image_input_ok"] = f"failed: {exc.code}"
        results.append(record)
        if config.llm_provider == "gemini":
            time.sleep(10)   # 15 RPM free tier

    notes["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    payload = {"notes": notes, "config": asdict(config), "results": results}
    (OUT_DIR / "smoke_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(payload)
    print(f"\nwrote {OUT_DIR / 'smoke_results.json'} and smoke_report.md")

    negative = next(r for r in results if r["kind"] == "negative")
    ok = all("answer" in r for r in results)
    abstained_ok = negative.get("answer", {}).get("abstained", False)
    print(f"\nSMOKE {'PASS' if ok else 'PARTIAL'}; "
          f"negative-control abstained: {abstained_ok}")
    return 0 if ok else 1


def _write_report(payload: dict) -> None:
    lines = ["# Smoke run report (1.9)", ""]
    lines.append(f"- provider: `{payload['config']['llm_provider']}"
                 f":{payload['config']['llm_model']}` / "
                 f"embed `{payload['config']['embed_model']}`")
    for key, value in payload["notes"].items():
        if key not in ("ingest",):
            lines.append(f"- {key}: {value}")
    lines.append("")
    for name, info in payload["notes"].get("ingest", {}).items():
        lines.append(f"- **{name}** -> `{info['paper_id']}` — {info['n_pages']} pages, "
                     f"{info['n_chunks']} chunks, {info['n_figures']} figures, "
                     f"{info['seconds']}s ({info['title'][:60]})")
    for record in payload["results"]:
        lines += ["", f"## [{record['kind']}] {record['question']}", ""]
        if "anchor" in record:
            sel = (record["anchor"].get("selection") or "")[:120]
            lines.append(f"*anchor: paper `{record['anchor']['paper_id']}` "
                         f"page {record['anchor']['page']}; selection: “{sel}…”*\n")
        if "error" in record:
            lines.append(f"**ERROR** `{record['error']['code']}`: "
                         f"{record['error']['message']}")
            continue
        answer = record["answer"]
        lines.append(answer["text"])
        lines.append("")
        for c in answer["citations"]:
            lines.append(f"- [{c['marker']}] paper `{c['paper_id']}` "
                         f"p.{c['page']}: “{c['quote'][:100]}…”")
        for f in answer["figures"]:
            lines.append(f"- figure: `{f['figure_id']}` ({f['label']}, "
                         f"p.{f['page']})")
        lines.append(f"- abstained: {answer['abstained']}; "
                     f"scope: {answer['scope_used']}")
    (OUT_DIR / "smoke_report.md").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(run())
