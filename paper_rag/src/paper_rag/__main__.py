"""Minimal CLI (plan.md D15) — development self-test and demo, not a frontend.

    uv run python -m paper_rag ingest paper.pdf
    uv run python -m paper_rag ask "What is late interaction?" [--paper ID] [--page N] [--selection TEXT]
    uv run python -m paper_rag papers
"""

import argparse
import json
from dataclasses import asdict

from . import Anchor, PaperLibrary, PaperRagError, Scope


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS: absent flags leave no attribute, so a subcommand's parse can
    # never clobber a --data-dir given before the subcommand.
    common.add_argument("--data-dir", default=argparse.SUPPRESS)
    parser = argparse.ArgumentParser(prog="paper_rag", parents=[common])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ingest = sub.add_parser("ingest", parents=[common])
    p_ingest.add_argument("pdf")
    p_ingest.add_argument("--paper-id")
    p_ask = sub.add_parser("ask", parents=[common])
    p_ask.add_argument("question")
    p_ask.add_argument("--paper")
    p_ask.add_argument("--page", type=int)
    p_ask.add_argument("--selection")
    p_ask.add_argument("--scope", choices=[s.value for s in Scope], default="auto")
    p_ask.add_argument("--top-k", type=int, default=8)
    sub.add_parser("papers", parents=[common])
    args = parser.parse_args(argv)

    try:
        lib = PaperLibrary(data_dir=getattr(args, "data_dir", "./paper_rag_data"))
        if args.cmd == "ingest":
            pid = lib.ingest(args.pdf, paper_id=args.paper_id,
                             progress=lambda p: print(f"  {p.stage}: {p.done}/{p.total}"))
            print(f"ingested: {pid}")
        elif args.cmd == "ask":
            if not args.paper and (args.page or args.selection):
                print("error: --page/--selection require --paper")
                return 1
            anchor = (Anchor(paper_id=args.paper, page=args.page,
                             selection=args.selection) if args.paper else None)
            answer = lib.ask(args.question, anchor=anchor,
                             scope=Scope(args.scope), top_k=args.top_k)
            print(answer.text)
            for c in answer.citations:
                print(f"  [{c.marker}] {c.paper_id} p.{c.page}: {c.quote[:80]}...")
            for f in answer.figures:
                print(f"  figure: {f.figure_id} ({f.label})")
        else:
            print(json.dumps([asdict(p) for p in lib.papers()],
                             ensure_ascii=False, indent=2))
    except PaperRagError as exc:
        print(f"error {exc.code}: {exc.message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
