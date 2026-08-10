# paper_rag

RAG backend for an academic-paper reading companion: anchored, figure- and
reference-aware, citation-grounded Q&A over paper PDFs.

> Built as the backend of a (future) AI-assisted PDF reader. Every answer
> carries `[n]` citations that map back to page numbers and rectangle
> coordinates for click-to-highlight; unanswerable questions get an explicit
> abstention instead of a guess. Roadmap: [STRETCH_LIST.md](STRETCH_LIST.md).

## Quickstart

```bash
uv sync                 # install deps
cp .env.example .env    # then fill in your API keys (never commit .env)
uv run pytest           # unit tests (offline, fake providers)
```

```python
from paper_rag import PaperLibrary, Anchor

lib = PaperLibrary(data_dir="./paper_rag_data")
pid = lib.ingest("paper.pdf")                    # minutes under free-tier rate limits
ans = lib.ask("What is the main contribution?")  # ask in any language — the answer
print(ans.text)                                  #   follows the question's language
for c in ans.citations:
    print(f"  [{c.marker}] p.{c.page}: {c.quote[:60]}")

# anchored question: reading page 5, selected a sentence
ans2 = lib.ask("Which baseline is this?",
               anchor=Anchor(paper_id=pid, page=5, selection="..."))
```

CLI: `uv run python -m paper_rag ingest paper.pdf` / `... ask "question"` /
`... papers`. Dev web chat: `uv run python scripts/chat_ui.py`.

## Providers

Generation and embedding are pluggable via env vars
(`PAPER_RAG_LLM_PROVIDER`, `PAPER_RAG_EMBED_PROVIDER`, `PAPER_RAG_LLM_MODEL`,
`PAPER_RAG_EMBED_MODEL`). Reference setup: Claude (`anthropic`) for
generation + Gemini embeddings (`gemini-embedding-2`); keys via `.env`
(`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`). Swapping the generation model is a
config-only change; swapping the embedding model requires re-ingesting — the
index stores an embedder fingerprint and fails loudly on mismatch.

## Offline mode

Set `PAPER_RAG_LLM_PROVIDER=fake` and `PAPER_RAG_EMBED_PROVIDER=fake`
(plus `PAPER_RAG_EMBED_DIM=32`) to run the full pipeline without any API —
used by the test suite and the smoke fallback
(`uv run python scripts/run_smoke.py --fake`).

## Known macOS issue

If `python -m paper_rag` fails with `No module named paper_rag` on macOS
(iCloud "Desktop & Documents" sync can stamp the venv's files with the
`hidden` flag, and recent CPython skips hidden `.pth` files), run once:

```bash
uv sync --no-editable
```

`uv run pytest` and the scripts are immune (they put `src` on `sys.path`
directly).
