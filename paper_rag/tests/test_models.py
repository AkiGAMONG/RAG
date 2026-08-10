"""§3: every public dataclass must be JSON-clean."""

import json
from dataclasses import asdict

from paper_rag import (Anchor, Answer, Citation, FigureRef, IngestProgress,
                       PaperInfo, Rect, Scope, Turn)


def test_dataclasses_serialize_to_json():
    answer = Answer(
        text="Late interaction is cheap [1].",
        citations=[Citation(marker=1, chunk_id="p1:0001:b0000", paper_id="p1",
                            page=1, quote="late interaction ...",
                            rects=[Rect(page=1, x0=0, y0=0, x1=10, y1=10)])],
        figures=[FigureRef(figure_id="p1:fig:1", paper_id="p1",
                           label="Figure 1", caption="cap", page=2,
                           image_path="figures/fig_1.png")],
        abstained=False, scope_used="paper:p1", debug={"queries": ["q"]})
    payload = json.dumps(asdict(answer))
    assert "Late interaction" in payload

    for obj in (Anchor(paper_id="p1", page=3, selection="text"),
                Turn(role="user", content="hi"),
                PaperInfo(paper_id="p1", title="T", n_pages=3, n_chunks=10,
                          n_figures=1, ingested_at="2026-01-01T00:00:00Z"),
                IngestProgress(stage="parsing", done=1, total=3)):
        json.dumps(asdict(obj))


def test_scope_values():
    assert Scope.AUTO.value == "auto"
    assert Scope.PAPER.value == "paper"
    assert Scope.LIBRARY.value == "library"
    assert Scope("paper") is Scope.PAPER
