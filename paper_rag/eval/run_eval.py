"""Phase 2 evaluation runner (plan.md §3 Phase 2).

For every gold-set question:
  * retrieval scoring — the anchored pipeline vs a naive baseline (question
    text only, no anchor), P@k / R@k / MRR / nDCG@k against gold evidence
    mapped onto the current index;
  * generation — lib.ask(), mechanical checks (abstention correctness,
    fabricated-citation count, figure attachment, answer language), and
    LLM-as-judge coverage of expected_points;
  * zh/en parity for questions carrying question_zh;
  * a small answer-stability probe (same question 3×) for the D12/temperature
    question.

Answers and judge verdicts are cached write-through in eval/results/cache.json
(keyed by question id + variant), so a rate-limited or interrupted run resumes
where it stopped. Retrieval is recomputed every run (cheap, index-dependent).

Usage:
    uv run python eval/run_eval.py [--data-dir smoke_output/library_data]
        [--gold eval/gold_set.json] [--k 8] [--no-judge] [--fresh]
        [--tag paragraph]      # label for this index/config in the report

Cost note: one full run ≈ ~30 Claude calls (25 ask + parity + stability) +
~25 judge calls + ~60 single-text query embeddings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from dotenv import load_dotenv  # noqa: E402

import scoring  # noqa: E402
from paper_rag import Anchor, Config, PaperLibrary, PaperRagError  # noqa: E402
from paper_rag.generation import ABSTAIN_MARKER  # noqa: E402
from paper_rag.models import Scope  # noqa: E402
from paper_rag.providers.base import TextPart  # noqa: E402
from paper_rag.retrieval import Retriever, _match_selection  # noqa: E402

RESULTS_DIR = ROOT / "eval" / "results"
# Figure-attachment expectations encoded by the gold set's feature tags/notes.
EXPECTED_FIGURES = {"q13": "fig:1", "q14": "tab:1", "q16": "fig:3"}
STABILITY_QID = "q01"
STABILITY_RUNS = 3


# --------------------------------------------------------------------------
# cache

class Cache:
    def __init__(self, path: Path, fresh: bool):
        self.path = path
        self.data: dict = {}
        if path.is_file() and not fresh:
            self.data = json.loads(path.read_text(encoding="utf-8"))

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value) -> None:
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1),
                             encoding="utf-8")


# --------------------------------------------------------------------------
# helpers

def make_anchor(paper_id: str, spec: dict | None) -> Anchor | None:
    if spec is None:
        return None
    return Anchor(paper_id=paper_id, page=spec.get("page"),
                  selection=spec.get("selection"))


def gold_ids_for(question: dict, lib: PaperLibrary, paper_id: str,
                 k: int) -> tuple[set[str], list[str]]:
    return scoring.evidence_chunk_ids(
        question["gold_evidence"],
        page_text_fn=lambda p: lib.metas.read_page_text(paper_id, p),
        page_chunks_fn=lambda p: lib.store.get_page_chunks(paper_id, p),
        match_fn=_match_selection)


def ranked_retrieval_ids(lib: PaperLibrary, question_text: str,
                         anchor: Anchor | None, k: int) -> list[str]:
    retriever = Retriever(lib.store, lib.metas, lib.embedder, lib.config,
                          llm=None)   # no LLM: pure retrieval, no D4 fallback
    result = retriever.retrieve(question_text, anchor, Scope.AUTO, k)
    return [c.chunk_id for c in result.retrieved_chunks]


def cached_ask(lib: PaperLibrary, cache: Cache, key: str, question_text: str,
               anchor: Anchor | None, k: int) -> dict:
    hit = cache.get(key)
    if hit is not None:
        return hit
    answer = lib.ask(question_text, anchor=anchor, top_k=k)
    record = asdict(answer)
    cache.put(key, record)
    return record


def cached_judge(lib: PaperLibrary, cache: Cache, key: str, question_text: str,
                 answer_text: str, expected_points: list[str],
                 quotes: list[str]) -> dict:
    hit = cache.get(key)
    if hit is not None:
        return hit
    prompt = scoring.judge_prompt(question_text, answer_text,
                                  expected_points, quotes)
    verdict = None
    for attempt in range(2):
        reply = lib.llm.complete(scoring.JUDGE_SYSTEM, [TextPart(text=prompt)])
        verdict = scoring.parse_judge_json(reply)
        if verdict is not None:
            break
        prompt += "\n\nReturn ONLY the JSON object."
    record = verdict if verdict is not None else {"judge_error": True,
                                                  "covered": [], "missing": []}
    record["score"] = scoring.judge_score(record, len(expected_points))
    cache.put(key, record)
    return record


# --------------------------------------------------------------------------
# per-question evaluation

def evaluate_question(q: dict, lib: PaperLibrary, paper_id: str, k: int,
                      cache: Cache, use_judge: bool) -> dict:
    qid = q["id"]
    out: dict = {"id": qid, "kind": q["kind"], "features": q.get("features", [])}
    anchor = make_anchor(paper_id, q.get("anchor"))
    gold, warnings = gold_ids_for(q, lib, paper_id, k)
    out["gold_chunk_ids"] = sorted(gold)
    out["gold_warnings"] = warnings

    # ---- retrieval: pipeline vs naive baseline (non-negative questions)
    if q["gold_evidence"]:
        pipeline_ids = ranked_retrieval_ids(lib, q["question"], anchor, k)
        naive_ids = ranked_retrieval_ids(lib, q["question"], None, k)
        out["retrieval"] = {
            "pipeline": scoring.retrieval_metrics(pipeline_ids, gold, k),
            "naive": scoring.retrieval_metrics(naive_ids, gold, k),
            "pipeline_ids": pipeline_ids, "naive_ids": naive_ids}

    # ---- generation + mechanical checks
    answer = cached_ask(lib, cache, f"{qid}:ask", q["question"], anchor, k)
    out["answer_text"] = answer["text"]
    out["abstained"] = answer["abstained"]
    dropped = answer.get("debug", {}).get("dropped_markers", [])
    citation_ids = [c["chunk_id"] for c in answer["citations"]]
    mech = {
        "abstain_correct": answer["abstained"] == q["expect_abstain"],
        "fabricated_markers": len(dropped),
        "has_citations": bool(citation_ids),
        "citation_support": scoring.citation_support(citation_ids, gold),
        "language_ok": scoring.language_ok(q["question"], answer["text"],
                                           answer["abstained"], ABSTAIN_MARKER),
    }
    expected_fig = EXPECTED_FIGURES.get(qid)
    if expected_fig:
        shown = [f["figure_id"] for f in answer["figures"]]
        mech["figure_attached"] = any(f.endswith(expected_fig) for f in shown)
        mech["figures_shown"] = shown
    if q.get("anchor"):
        mech["anchor_match"] = (answer.get("debug", {})
                                .get("anchor", {}).get("match"))
    out["mechanical"] = mech

    # ---- LLM-as-judge (skip expected-abstain questions)
    if use_judge and not q["expect_abstain"] and not answer["abstained"]:
        quotes = [ev["quote"] for ev in q["gold_evidence"]]
        out["judge"] = cached_judge(lib, cache, f"{qid}:judge", q["question"],
                                    answer["text"], q["expected_points"], quotes)

    # ---- zh/en parity
    if q.get("question_zh"):
        zh_anchor = anchor
        zh_pipeline = ranked_retrieval_ids(lib, q["question_zh"], zh_anchor, k)
        zh_answer = cached_ask(lib, cache, f"{qid}:ask_zh", q["question_zh"],
                               zh_anchor, k)
        en_set = set(out.get("retrieval", {}).get("pipeline_ids", []))
        zh_set = set(zh_pipeline)
        union = en_set | zh_set
        out["parity"] = {
            "zh_metrics": scoring.retrieval_metrics(zh_pipeline, gold, k),
            "jaccard_top_k": (len(en_set & zh_set) / len(union)) if union else 0.0,
            "zh_language_ok": scoring.language_ok(q["question_zh"],
                                                  zh_answer["text"],
                                                  zh_answer["abstained"],
                                                  ABSTAIN_MARKER),
            "zh_abstained": zh_answer["abstained"],
        }
    return out


def stability_probe(gold: dict, lib: PaperLibrary, paper_id: str, k: int,
                    cache: Cache) -> dict:
    q = next(x for x in gold["questions"] if x["id"] == STABILITY_QID)
    texts = []
    for i in range(STABILITY_RUNS):
        rec = cached_ask(lib, cache, f"{STABILITY_QID}:stability{i}",
                         q["question"], make_anchor(paper_id, q.get("anchor")), k)
        texts.append(rec["text"])
    return {"question_id": STABILITY_QID, "runs": STABILITY_RUNS,
            "all_identical": len(set(texts)) == 1,
            "distinct_answers": len(set(texts))}


# --------------------------------------------------------------------------
# aggregation + report

def aggregate(rows: list[dict], stability: dict, tag: str, k: int,
              config: Config) -> dict:
    scored = [r for r in rows if "retrieval" in r]
    negatives = [r for r in rows if r["kind"] == "negative"]
    judged = [r for r in rows if "judge" in r and not r["judge"].get("judge_error")]
    parity_rows = [r for r in rows if "parity" in r]
    summary = {
        "tag": tag, "k": k,
        "config": {"llm": f"{config.llm_provider}:{config.llm_model}",
                   "embedder": config.embedder_fingerprint(),
                   "chunking": config.chunking_strategy},
        "n_questions": len(rows),
        "retrieval_pipeline": scoring.mean_metrics(
            [r["retrieval"]["pipeline"] for r in scored]),
        "retrieval_naive": scoring.mean_metrics(
            [r["retrieval"]["naive"] for r in scored]),
        "fabricated_markers_total": sum(r["mechanical"]["fabricated_markers"]
                                        for r in rows),
        "abstention_correct": sum(1 for r in rows
                                  if r["mechanical"]["abstain_correct"]),
        "abstention_total_negatives": len(negatives),
        "negatives_all_abstained": all(r["abstained"] for r in negatives),
        "language_ok_all": all(r["mechanical"]["language_ok"] for r in rows),
        "figure_checks": {r["id"]: r["mechanical"].get("figure_attached")
                          for r in rows if "figure_attached" in r["mechanical"]},
        "judge_mean_score": (sum(r["judge"]["score"] for r in judged)
                             / len(judged)) if judged else None,
        "judge_contradicted": [r["id"] for r in judged
                               if r["judge"].get("contradicted")],
        "parity_mean_jaccard": (sum(r["parity"]["jaccard_top_k"]
                                    for r in parity_rows) / len(parity_rows))
                               if parity_rows else None,
        "stability": stability,
    }
    pipe, naive = summary["retrieval_pipeline"], summary["retrieval_naive"]
    summary["success_criteria"] = {
        "zero_fabricated_citations": summary["fabricated_markers_total"] == 0,
        "pipeline_geq_naive_all_metrics": all(
            pipe[m] >= naive[m] - 1e-9 for m in pipe) if pipe else None,
        "all_negatives_abstained": summary["negatives_all_abstained"],
        "abstentions_all_correct": (summary["abstention_correct"] == len(rows)),
        "judge_groundedness_ok": (summary["judge_mean_score"] is not None
                                  and summary["judge_mean_score"] >= 0.7
                                  and not summary["judge_contradicted"]),
    }
    return summary


def write_report(summary: dict, rows: list[dict], out_path: Path) -> None:
    s = summary
    lines = [f"# PaperRAG 评测报告（Phase 2） — run `{s['tag']}`", ""]
    lines.append(f"- 配置: llm=`{s['config']['llm']}` / embed=`{s['config']['embedder']}` "
                 f"/ chunking=`{s['config']['chunking']}` / top_k={s['k']}")
    lines.append(f"- 题目数: {s['n_questions']}（gold set v0.1，25 题六类）")
    lines += ["", "## 检索指标（锚定管线 vs 朴素基线，二值相关，k=8）", "",
              "| 指标 | 管线 | 朴素基线 | Δ |", "|---|---|---|---|"]
    for m in ("P@k", "R@k", "MRR", "nDCG@k"):
        p, n = s["retrieval_pipeline"].get(m, 0), s["retrieval_naive"].get(m, 0)
        lines.append(f"| {m} | {p:.3f} | {n:.3f} | {p - n:+.3f} |")
    lines += ["", "## 机械校验", "",
              f"- 虚构引用标记（fabricated markers）: **{s['fabricated_markers_total']}**",
              f"- 弃答正确率: {s['abstention_correct']}/{s['n_questions']}"
              f"（负对照全部弃答: {s['negatives_all_abstained']}）",
              f"- 回答语言跟随问题语言: {s['language_ok_all']}",
              f"- 图表附加检查: {s['figure_checks']}"]
    if s["judge_mean_score"] is not None:
        lines += ["", "## LLM-as-judge（要点覆盖率）", "",
                  f"- 平均覆盖率: **{s['judge_mean_score']:.2f}**",
                  f"- 与证据矛盾的答案: {s['judge_contradicted'] or '无'}"]
    if s["parity_mean_jaccard"] is not None:
        lines += ["", "## 中英 parity", "",
                  f"- 平均 top-k Jaccard 重叠: {s['parity_mean_jaccard']:.2f}"]
        for r in rows:
            if "parity" in r:
                lines.append(f"  - {r['id']}: jaccard={r['parity']['jaccard_top_k']:.2f}, "
                             f"zh R@k={r['parity']['zh_metrics']['R@k']:.2f}, "
                             f"zh 语言={r['parity']['zh_language_ok']}")
    st = s["stability"]
    stability_desc = ("逐字一致" if st["all_identical"]
                      else f"{st['distinct_answers']} 种不同答案")
    lines += ["", "## 稳定性（D12/temperature 弃用检验）", "",
              f"- 同题 {st['runs']} 次: {stability_desc}"]
    lines += ["", "## Success criteria（project_description）", ""]
    for key, val in s["success_criteria"].items():
        mark = "✅" if val else ("❓" if val is None else "❌")
        lines.append(f"- {mark} {key}: {val}")
    lines += ["", "## 逐题明细", ""]
    for r in rows:
        mech = r["mechanical"]
        parts = [f"**{r['id']}** ({r['kind']})"]
        if "retrieval" in r:
            pm = r["retrieval"]["pipeline"]
            parts.append(f"R@k={pm['R@k']:.2f} MRR={pm['MRR']:.2f}")
        if "judge" in r:
            parts.append(f"judge={r['judge']['score']:.2f}")
        parts.append(f"abstain_ok={mech['abstain_correct']}")
        if mech["fabricated_markers"]:
            parts.append(f"fabricated={mech['fabricated_markers']}")
        if r["gold_warnings"]:
            parts.append(f"warnings={r['gold_warnings']}")
        lines.append("- " + " | ".join(parts))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "smoke_output/library_data"))
    parser.add_argument("--gold", default=str(ROOT / "eval/gold_set.json"))
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore the answer/judge cache")
    parser.add_argument("--tag", default="paragraph",
                        help="label for this index/config in the report")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    load_dotenv()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    paper_id = gold["paper_id"]
    lib = PaperLibrary(data_dir=args.data_dir)
    if not any(p.paper_id == paper_id for p in lib.papers()):
        print(f"paper {paper_id} not in {args.data_dir}; ingest it first "
              f"(scripts/run_smoke.py --papers ../test_paper)")
        return 1
    print(f"eval config: llm={lib.config.llm_provider}:{lib.config.llm_model} "
          f"embed={lib.config.embedder_fingerprint()} tag={args.tag}")

    cache = Cache(RESULTS_DIR / f"cache_{args.tag}.json", args.fresh)
    rows: list[dict] = []
    for q in gold["questions"]:
        t0 = time.monotonic()
        try:
            row = evaluate_question(q, lib, paper_id, args.k, cache,
                                    use_judge=not args.no_judge)
        except PaperRagError as exc:
            row = {"id": q["id"], "kind": q["kind"], "error": exc.code,
                   "mechanical": {"abstain_correct": False,
                                  "fabricated_markers": 0, "has_citations": False,
                                  "citation_support": 0.0, "language_ok": False},
                   "abstained": False, "gold_warnings": []}
        rows.append(row)
        note = row.get("error", "")
        print(f"  {q['id']} ({q['kind']}) done in "
              f"{time.monotonic() - t0:.1f}s {note}")

    stability = stability_probe(gold, lib, paper_id, args.k, cache)
    summary = aggregate(rows, stability, args.tag, args.k, lib.config)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_json = RESULTS_DIR / f"eval_{args.tag}_{stamp}.json"
    out_json.write_text(json.dumps({"summary": summary, "questions": rows},
                                   ensure_ascii=False, indent=1),
                        encoding="utf-8")
    report_path = ROOT / "docs" / "eval_report.md"
    write_report(summary, rows, report_path)
    print(f"\nwrote {out_json}\nwrote {report_path}\n")
    print(json.dumps(summary["success_criteria"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
