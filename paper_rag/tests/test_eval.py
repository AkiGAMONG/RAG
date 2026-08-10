"""Offline tests for the Phase 2 scoring logic (eval/scoring.py)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import scoring  # noqa: E402


def test_retrieval_metrics_known_values():
    gold = {"a", "b", "c"}
    ranked = ["x", "a", "y", "b", "z", "w", "v", "u"]
    m = scoring.retrieval_metrics(ranked, gold, k=8)
    assert m["P@k"] == 2 / 8
    assert m["R@k"] == 2 / 3
    assert m["MRR"] == 1 / 2          # first hit at rank 2
    # DCG = 1/log2(3) + 1/log2(5); IDCG = 1/log2(2)+1/log2(3)+1/log2(4)
    import math
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / 2
    assert abs(m["nDCG@k"] - dcg / idcg) < 1e-9


def test_retrieval_metrics_edge_cases():
    assert scoring.retrieval_metrics([], {"a"}, 8)["MRR"] == 0.0
    assert scoring.retrieval_metrics(["a"], {"a"}, 8)["nDCG@k"] == 1.0
    assert scoring.recall_at_k(["a"], set(), 8) == 0.0
    perfect = scoring.retrieval_metrics(["a", "b"], {"a", "b"}, 8)
    assert perfect["R@k"] == 1.0 and perfect["MRR"] == 1.0


def test_mean_metrics():
    rows = [{"P@k": 0.5, "R@k": 1.0}, {"P@k": 0.25, "R@k": 0.0}]
    mean = scoring.mean_metrics(rows)
    assert mean == {"P@k": 0.375, "R@k": 0.5}
    assert scoring.mean_metrics([]) == {}


def test_evidence_chunk_ids_mapping():
    page_text = "Alpha beta gamma.\n\nDelta epsilon zeta."
    rows = [
        {"chunk_id": "c1", "metadata": {"char_start": 0, "char_end": 17}},
        {"chunk_id": "c2", "metadata": {"char_start": 19, "char_end": 39}},
    ]

    def match(text, quote):
        idx = text.find(quote)
        return (idx, idx + len(quote), "exact") if idx >= 0 else None

    ids, warnings = scoring.evidence_chunk_ids(
        [{"page": 1, "quote": "Delta epsilon"}],
        page_text_fn=lambda p: page_text,
        page_chunks_fn=lambda p: rows,
        match_fn=match)
    assert ids == {"c2"} and not warnings

    ids, warnings = scoring.evidence_chunk_ids(
        [{"page": 1, "quote": "NOT PRESENT"}],
        page_text_fn=lambda p: page_text,
        page_chunks_fn=lambda p: rows,
        match_fn=match)
    assert ids == set() and len(warnings) == 1


def test_language_checks():
    marker = "I don't have that information in the provided documents."
    assert scoring.language_ok("为什么？", "因为欧氏距离是 Bregman 散度。", False, marker)
    assert not scoring.language_ok("为什么？", "Because it is a Bregman divergence.",
                                   False, marker)
    # abstention is exempt: English constant allowed for a Chinese question
    assert scoring.language_ok("为什么？", marker, True, marker)
    assert scoring.language_ok("Why?", "Because.", False, marker)


def test_judge_json_parsing_and_score():
    reply = '```json\n{"covered": [0, 2], "missing": [1], "contradicted": false}\n```'
    verdict = scoring.parse_judge_json(reply)
    assert verdict["covered"] == [0, 2]
    assert scoring.judge_score(verdict, 3) == 2 / 3
    # out-of-range and junk indices are ignored
    assert scoring.judge_score({"covered": [0, 7, "x"]}, 3) == 1 / 3
    assert scoring.parse_judge_json("no json here") is None
    assert scoring.parse_judge_json('{"other": 1}') is None


def test_citation_support():
    assert scoring.citation_support(["a", "b"], {"a"}) == 0.5
    assert scoring.citation_support([], {"a"}) == 0.0
