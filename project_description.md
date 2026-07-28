# PaperRAG — A Grounded Q&A Backend for an Academic-Paper Reading Companion

*(Working name; Python package: `paper_rag`. Rename freely.)*

**One-liner:** A Python library that answers a reader's questions about academic papers with rigorously grounded, citation-backed responses — aware of *where in the paper* the reader currently is, and of the figures, tables, and references the text points to.

## Background & Product Vision (context only, out of scope here)

The long-term product is a PDF reader built for the "ultimate paper-reading experience": a reading pane plus an AI chat sidebar that answers questions while the user reads. The reader's other components (UI, PDF rendering, backend services) will be built separately in Rust. **This project builds exactly one component: the RAG system behind the sidebar**, delivered as a standalone Python library with tests and an evaluation report. Integration happens later through a thin service wrapper (see `api_contract.md`); nothing in this project depends on the frontend existing.

## Why

Reading academic papers is disproportionately painful for non-native English speakers: on top of the language barrier, papers are dense, assume background knowledge, and constantly point elsewhere — "as shown in Figure 3", "following [12]" — forcing slow, fragmented reading.

Generic chatbots do not solve this. They hallucinate plausible-sounding claims, cannot show *where in the paper* an answer comes from, and know nothing about the reader's current position — yet most real reading questions ("what does this passage mean?", "which baseline is this referring to?") only make sense relative to that position. A trustworthy reading companion needs answers that are (a) grounded exclusively in authoritative sources, (b) verifiable down to the page and sentence, and (c) anchored to the reading context.

## Who

Primary users: university students and researchers reading English-language papers in a non-native language, starting with the author himself. Their needs: instant explanations of the passage in front of them, in their own language; answers they can trust and verify against the paper itself; figures/tables and cited works surfaced at the moment the text mentions them.

## What (knowledge sources)

The knowledge base consists of academic-paper PDFs — uploaded by the user or acquired by the backend. **Acquisition is out of scope: this project assumes the PDFs are already on disk.** Papers are ingested as:

- **Body text**, extracted per page with positions preserved (page number + character offsets + bounding rectangles) so every answer can be traced back to an exact location.
- **Figures and tables**, located via their numbered captions ("Figure 3: …", "Table 2: …"). The caption text is embedded as the retrieval index for the figure; the figure itself is stored as a page-region screenshot and passed to a multimodal LLM when relevant. Tables are treated as images (no structured table parsing).
- **References**, parsed from the bibliography into structured entries so inline citation markers ("[12]", "(Khattab & Zaharia, 2020)") can be resolved to *which* work is being cited.

Development/evaluation corpus: 2–3 papers the author knows well (criteria: text-native PDFs, numbered figure captions, familiar enough to hand-label a gold answer set).

## How

**Answering pipeline (one pipeline, two optional inputs).** A request carries a question, an optional *anchor* (paper id + page + selected text = the reading position), and a *scope* (current paper or whole library — user-selectable). The pipeline: (1) resolve the anchor to its chunk(s) and surrounding context, including any figure/table/citation references inside it; (2) rewrite the question into standalone retrieval queries using the anchor text — the anchor tells us *what the question is about*, retrieval finds *where the answer lives*, which matters precisely when the anchored passage only mentions the topic in passing; (3) retrieve with metadata pre-filtering by scope; (4) assemble context (anchored chunks first, retrieved chunks, linked figures as images, resolved citation info) with deduplication; (5) generate under strict rules.

**Figure/table awareness.** References like "Figure 3" in the question or in retrieved text are resolved deterministically (regex + metadata) first, with LLM fallback for indirect mentions; the resolved image is attached to the multimodal LLM call.

**Citation following (leveled).** L1 (core): resolve inline markers to bibliography entries and tell the model what is being cited. L2 (stretch): if the cited paper is already in the library, retrieve from it too. L3 (excluded): automatically downloading missing cited papers — future work; the API only signals "cited paper not in library".

**Handling conflicting, incomplete, or outdated information.** Papers legitimately disagree. In cross-paper mode, answers must attribute claims to their specific papers ("Paper A reports X [1]; Paper B reports Y [3]") and never silently merge conflicting claims. For anchored questions the paper being read takes precedence. Incomplete evidence triggers abstention rather than guessing.

**When no reliable answer exists.** The model must output a fixed, machine-checkable refusal marker (exact sentence defined in the API contract) instead of guessing, optionally followed by a localized explanation of what is missing.

**How users know an answer is trustworthy.** Every factual sentence carries a citation marker mapped to paper, page, verbatim quote, and bounding rectangles — enabling click-to-highlight in the future reader. Decoding uses temperature 0 for reproducibility. Trust is *measured*, not asserted: the evaluation suite scores retrieval quality (precision@k, recall@k, MRR, nDCG@k on a hand-labeled gold set), citation validity (mechanical checks), groundedness (LLM-as-judge), abstention correctness on negative controls, and Chinese-vs-English query parity (answers follow the question's language; the corpus is English).

**Provider-agnostic by design.** Generation and embedding go through thin `LLMClient` / `Embedder` abstractions; the default configuration uses Google Gemini (free tier), but providers are swappable via config (note: Anthropic offers no embedding API, so a Claude configuration still needs a separate embedder).

## Deliverables (course scope)

A working Python library (`paper_rag`) with the public API defined in `api_contract.md`; unit and end-to-end tests; an evaluation report on the gold set; project documentation. **Stretch:** L2 citation-linkage, multi-turn support (history-aware query rewriting), caption-embedding vs multimodal-embedding A/B comparison. **Non-goals for this project:** any UI, PDF acquisition/auto-download, structured table parsing, section-path detection (stretch), REST wrapper, streaming.

## Success criteria

End-to-end answers on the dev corpus with: zero fabricated citations in mechanical checks; ≥ the naive-baseline scores on all retrieval metrics with the anchored pipeline; correct abstention on all negative controls; groundedness confirmed by LLM-as-judge on the gold set; comparable retrieval quality for Chinese and English phrasings of the same questions.
