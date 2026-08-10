# Stretch List — deferred work, each item with an explicit trigger

> Discipline: **evaluate, not guess.** Items live here because they are worth
> doing but out of the current scope. Quality items carry a *trigger
> condition* — we first measure the problem with the evaluation framework
> (`eval/`), and only build the fix once the measurement proves it matters.
> Item IDs (S1…S24) are stable and referenced across the project's history.

## A. Retrieval & generation quality

| ID | Item | What & why | Trigger / status |
|----|------|-----------|------------------|
| S1 | **In-library citation linkage (L2)** | Match parsed bibliography entries to papers already in the library (arXiv id / DOI / fuzzy title) and fill the reserved `matched_paper_id` field; when an anchored passage cites an in-library paper, extend retrieval into that paper so the answer can quote both. The most demo-worthy single feature. | High priority after the evaluation baseline. |
| S2 | **Multi-turn conversation** | The `history` parameter is already in `ask()`'s signature. Implement follow-up vs. new-topic classification, history-aware query rewriting, retrieval-need routing, and history truncation. | Needed before wiring into the reader's sidebar; low value for single-shot CLI use. |
| S3 | **LLM query rewriting (single-turn)** | Current rewriting is deterministic (question / question + selection). Upgrade to LLM-generated diverse queries for vague questions. | Only if evaluation shows recall is the bottleneck (adds one LLM call per question). |
| S4 | **Semantic deduplication** | Context dedup is currently by chunk id; upgrade to embedding-similarity dedup of near-duplicates. | Matters only when the corpus contains multiple versions of the same paper. |
| S5 | **Hybrid retrieval + reranking** | Add a keyword/full-text filter layer, widen the candidate pool, rerank by additional signals (optionally cross-encoder). | Trigger: evaluation shows pure vector search failing on exact-terminology questions. |
| S6 | **Quote-first mode** | Config switch: for high-risk questions, output supporting QUOTES before the answer to shrink the room for fabrication. | Cheap; adopt if grounding metrics ever dip. |
| S7 | **Abstention polish** | Observed in smoke tests: an abstained answer may still carry an unrelated figure picked up by retrieval before the model decided to refuse. Suppress figures on abstained answers. | Cosmetic; quick fix, do alongside any generation change. |
| S19 | **Structured reads for enumeration** | Enumeration/aggregation requests ("list all references", "all figures") mismatch top-k semantic search — observed live: asking for references returned 8 of 31 entries, all genuine yet *looking* fabricated. Add structured read APIs (`references(paper_id)`, `figures(paper_id)`) over metadata already stored; optionally route enumeration intents automatically. | The API half is near-mandatory for reader integration; auto-routing is a separate decision. |
| S20 | **Cross-lingual retrieval hardening** | Measured weakness from evaluation round 1: one Chinese query scored R@8 = 0 against the English corpus while its English twin scored 1.0. Candidates: tune the embedder's query-instruction prefix, translate-then-embed fallback, bilingual dual queries. | **Evidence in hand and metrics ready** — high priority; the parity metrics quantify any fix immediately. |

## B. Parsing & ingestion quality

| ID | Item | What & why | Trigger / status |
|----|------|-----------|------------------|
| S8 | **Numbered display equations as images** | Detect numbered display equations (right-edge "(n)" anchor + math-glyph density + centering), reuse the figure-screenshot machinery to register them as `eq:n`, resolve "Eq. (n)" mentions, attach as image input; exclude equation regions from chunking. Inline math stays as text. | Trigger: equation questions in the gold set score poorly with the current text-only treatment. |
| S9 | **Better table region detection** | Borderless tables often fall back to full-page screenshots. Use text-block negative space / rule-line detection to tighten regions. | Trigger: table questions underperform in evaluation. |
| S10 | **Separate adjacent figure/table regions** | Figures and tables that sit close on a page can end up in one crop. Needs region-boundary arbitration. | Do together with S9. |
| S11 | **Exclude in-figure text from body chunks** | Glyph text inside figures currently leaks into body chunks (and has been retrieved as a "source"). Add figure rectangles to the chunker's exclusion zones. | Cheap and clearly correct; batch with the next parsing change. |
| S12 | **Section-path metadata** | Heading heuristics (font size / numbering) → each chunk carries `section` (e.g. "3.2 Late Interaction"); enables section filtering and much friendlier citation display. The metadata slot already exists (empty). | Before reader integration; big citation-UX win. |
| S13 | **Ligature repair** | fi/ffi/Th ligatures are lost by extraction in some older PDFs ("Te paper"). Needs font-level mapping. | Low value-for-effort; parked. |
| S14 | **Cross-column/page paragraph merging** | Paragraph rebuilding doesn't merge across columns/pages; a page-leading half-sentence becomes its own paragraph. Merge heuristics risk false joins. | Only if evaluation shows real impact (expected small). |

## C. Product integration (reader-facing)

| ID | Item | What & why | Trigger / status |
|----|------|-----------|------------------|
| S15 | **REST sidecar** | Thin FastAPI wrapper per the wire mapping the API was designed for: `POST /papers` (async job + polling), `POST /ask` (later SSE streaming), `GET /figures/{id}`. | The public API was shaped for this from day one; the wrapper is thin. |
| S16 | **Auto-fetch cited papers (L3)** | Citations not matched in-library could be fetched via arXiv etc. Copyright/paywall boundaries apply. | Product phase; explicitly out of scope for now. |
| S17 | **Anchor carries rectangles** | Let the reader pass the selection as rects rather than text, so anchoring inside figures/equations isn't hostage to garbled extracted text. Requires an API extension (optional `rects` on `Anchor`). | High synergy with S8. |
| S18 | **Multimodal embeddings for figures** | If an A/B shows image-vector retrieval beating caption-text retrieval, promote image embeddings as the figure-search path. | Decided by the A/B result. |
| S21 | **Sentence-level citation highlight** | Today `Citation.rects` covers the whole chunk (~a paragraph); clicking `[n]` highlights the block, not the exact supporting sentence. Upgrade: prompt the model to quote its evidence verbatim → locate the quote with the existing exact/fuzzy matcher → take the sub-range rectangles; fall back to block-level on match failure. All machinery exists. | Reader-integration polish; high synergy with S17. |
| S22 | **Public `remove_paper(paper_id)`** | The public API has no delete. Internals exist (`vector_store.delete_paper` + `paper_meta.delete_paper`) but **both stores must be deleted together** — deleting only the paper directory leaves ghost vectors in the index (still retrieved; jump-to-source broken). Needs an API addition, `PAPER_NOT_FOUND` on unknown id, and delete-then-verify tests. | Must-have for reader integration (basic library management). |
| S23 | **Query decomposition for multi-topic questions** | A question mixing unrelated topics currently produces one blended embedding, diluting retrieval for both; in the worst case one topic's material is entirely absent (abstention rules prevent fabrication, but the user gets half an answer). Downstream plumbing already accepts multiple queries; the missing piece is one cheap LLM call to split the question. Evaluation blind spot: all current gold-set questions are single-topic — add the question type when implementing. | Do together with S3. |
| S24 | **Abstention localization** | The abstention sentence opens with a fixed English constant (the anchor for mechanical detection), optionally followed by an explanation in the question's language. Non-English users see an English first sentence. Options: (a) presentation layer renders its own localized message keyed off the structured `abstained` flag — zero library change; (b) per-language constant set — requires spec and evaluation changes. | (a) is nearly free during reader integration; (b) only if bare-API use demands it. |
