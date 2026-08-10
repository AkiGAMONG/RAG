"""Validate eval/gold_set.json against an ingested paper's stored pages.

Checks (all machine-verifiable; Jeff reviews content quality separately):
  * schema basics: unique ids, known kinds, anchored/negative invariants;
  * every gold_evidence quote resolves in its page's stored text via the
    SAME exact->fuzzy matcher the anchor pipeline uses (api_contract §6);
  * every anchor selection resolves likewise (page-scoped or whole-paper);
  * parity pairs (question_zh) exist for the zh-vs-en metric.

Usage:
    uv run python eval/validate_gold_set.py \
        [--data-dir smoke_output/library_data] [--gold eval/gold_set.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from paper_rag.retrieval import _match_selection  # noqa: E402
from paper_rag.store import PaperMetaStore  # noqa: E402

KINDS = {"concept", "detail", "figure", "anchored", "chinese", "negative"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "smoke_output/library_data"))
    parser.add_argument("--gold", default=str(ROOT / "eval/gold_set.json"))
    args = parser.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    metas = PaperMetaStore(args.data_dir)
    paper_id = gold["paper_id"]
    meta = metas.read_meta(paper_id)
    if meta is None:
        print(f"FAIL: paper {paper_id} not ingested under {args.data_dir}")
        return 1

    def page_text(page: int) -> str | None:
        return metas.read_page_text(paper_id, page)

    errors: list[str] = []
    ids = [q["id"] for q in gold["questions"]]
    if len(ids) != len(set(ids)):
        errors.append("duplicate question ids")
    parity = [q["id"] for q in gold["questions"] if q.get("question_zh")]

    for q in gold["questions"]:
        qid = q["id"]
        if q["kind"] not in KINDS:
            errors.append(f"{qid}: unknown kind {q['kind']!r}")
        if q["kind"] == "negative" and not q["expect_abstain"]:
            errors.append(f"{qid}: negative question must expect_abstain")
        if q["kind"] == "anchored" and not q.get("anchor"):
            errors.append(f"{qid}: anchored question without anchor")
        for ev in q["gold_evidence"]:
            text = page_text(ev["page"])
            if text is None:
                errors.append(f"{qid}: page {ev['page']} has no stored text")
                continue
            match = _match_selection(text, ev["quote"])
            if match is None:
                errors.append(f"{qid}: evidence not found on page {ev['page']}: "
                              f"{ev['quote'][:60]!r}...")
            elif match[2] != "exact":
                print(f"note: {qid} evidence on p.{ev['page']} matched via "
                      f"{match[2]} (ok)")
        anchor = q.get("anchor")
        if anchor and anchor.get("selection"):
            pages = ([anchor["page"]] if anchor.get("page")
                     else range(1, int(meta["n_pages"]) + 1))
            hits = [(p, _match_selection(page_text(p) or "", anchor["selection"]))
                    for p in pages]
            hits = [(p, m) for p, m in hits if m is not None]
            if not hits:
                errors.append(f"{qid}: anchor selection does not match any page")
            else:
                kinds = {m[2] for _, m in hits}
                print(f"note: {qid} anchor matched on p.{hits[0][0]} ({'/'.join(kinds)})")

    print(f"\n{len(gold['questions'])} questions; kinds: "
          f"{ {k: sum(1 for q in gold['questions'] if q['kind'] == k) for k in KINDS} }")
    print(f"parity pairs (question_zh): {parity}")
    if errors:
        print("\nFAILURES:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nGOLD SET VALID ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
