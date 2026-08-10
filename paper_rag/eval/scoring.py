"""Phase 2 scoring logic — pure functions, unit-testable offline.

Design note: gold evidence is stored as (page, verbatim quote), NOT chunk
ids — chunk boundaries change with the chunking strategy (D18 A/B), so the
gold set must stay chunking-independent. `evidence_chunk_ids` maps evidence
onto whatever index is being evaluated, at scoring time.
"""

from __future__ import annotations

import json
import math
import re

# --------------------------------------------------------------------------
# retrieval metrics (binary relevance, ranked list)

def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for cid in ranked[:k] if cid in gold) / k

def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return sum(1 for cid in ranked[:k] if cid in gold) / len(gold)

def mrr(ranked: list[str], gold: set[str], k: int) -> float:
    for i, cid in enumerate(ranked[:k], start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0

def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1)
              for i, cid in enumerate(ranked[:k], start=1) if cid in gold)
    ideal = sum(1.0 / math.log2(i + 1)
                for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0

def retrieval_metrics(ranked: list[str], gold: set[str], k: int) -> dict:
    return {"P@k": precision_at_k(ranked, gold, k),
            "R@k": recall_at_k(ranked, gold, k),
            "MRR": mrr(ranked, gold, k),
            "nDCG@k": ndcg_at_k(ranked, gold, k)}


def mean_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {key: sum(r[key] for r in rows) / len(rows) for key in rows[0]}


# --------------------------------------------------------------------------
# evidence -> chunk mapping (chunking-independent gold, D18 A/B)

def evidence_chunk_ids(evidence: list[dict], page_text_fn, page_chunks_fn,
                       match_fn) -> tuple[set[str], list[str]]:
    """Map (page, quote) evidence onto the current index's chunk ids.

    page_text_fn(page) -> str | None; page_chunks_fn(page) -> rows with
    metadata char_start/char_end; match_fn = the pipeline's selection matcher
    (exact -> fuzzy, api_contract §6). Returns (ids, warnings)."""
    ids: set[str] = set()
    warnings: list[str] = []
    for ev in evidence:
        text = page_text_fn(ev["page"])
        if text is None:
            warnings.append(f"page {ev['page']}: no stored text")
            continue
        span = match_fn(text, ev["quote"])
        if span is None:
            warnings.append(f"page {ev['page']}: quote unresolvable: "
                            f"{ev['quote'][:50]!r}")
            continue
        start, end = span[0], span[1]
        hit = False
        for row in page_chunks_fn(ev["page"]):
            meta = row["metadata"]
            if meta["char_start"] < end and meta["char_end"] > start:
                ids.add(row["chunk_id"])
                hit = True
        if not hit:
            warnings.append(f"page {ev['page']}: no chunk overlaps the quote")
    return ids, warnings


# --------------------------------------------------------------------------
# mechanical checks

_CJK_RE = re.compile(r"[一-鿿]")

def has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))

def language_ok(question: str, answer_text: str, abstained: bool,
                abstain_marker: str) -> bool:
    """Answer language must follow the question language (契约 §9/D8).
    Abstentions are exempt: the marker is a fixed English constant and the
    localized explanation after it is optional."""
    if abstained and abstain_marker in answer_text:
        return True
    if has_cjk(question):
        return has_cjk(answer_text)
    return True   # English question: no mechanical constraint we can trust


def citation_support(citation_chunk_ids: list[str], gold: set[str]) -> float:
    """Fraction of cited chunks that belong to the gold evidence set."""
    if not citation_chunk_ids:
        return 0.0
    return sum(1 for cid in citation_chunk_ids if cid in gold) / len(citation_chunk_ids)


# --------------------------------------------------------------------------
# LLM-as-judge

JUDGE_SYSTEM = """You grade an AI assistant's answer about an academic paper against gold criteria.
Judge ONLY from the provided gold evidence and expected points — not your own knowledge.
Output ONLY a JSON object, no prose, no code fences."""

def judge_prompt(question: str, answer_text: str,
                 expected_points: list[str], evidence_quotes: list[str]) -> str:
    points = "\n".join(f"  {i}. {p}" for i, p in enumerate(expected_points))
    quotes = "\n".join(f"  - {q}" for q in evidence_quotes) or "  (none)"
    return f"""QUESTION:
{question}

GOLD EVIDENCE (verbatim from the paper):
{quotes}

EXPECTED POINTS (0-indexed):
{points}

ANSWER TO GRADE:
{answer_text}

Return JSON:
{{"covered": [indices of expected points the answer conveys, paraphrase ok],
 "missing": [indices not conveyed],
 "contradicted": true/false  (answer states something the gold evidence contradicts),
 "unsupported": true/false  (answer makes factual claims not backed by the evidence),
 "comment": "one short sentence"}}"""


def parse_judge_json(reply: str) -> dict | None:
    """Extract the first JSON object from a judge reply. None if unusable."""
    text = reply.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text, flags=re.S)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "covered" not in obj:
        return None
    return obj


def judge_score(verdict: dict, n_points: int) -> float:
    if n_points == 0:
        return 0.0
    covered = {i for i in verdict.get("covered", [])
               if isinstance(i, int) and 0 <= i < n_points}
    return len(covered) / n_points
