"""1.7 — anchors, scope, query rewriting, figure/citation resolution.

Uses the ingested fake-provider library from conftest; the sample paper has
a figure (fig:1), a table (tab:1) and two numbered references.
"""

import pytest

from paper_rag import Anchor, PaperRagError, Scope
from paper_rag.providers.fake import FakeLLM
from paper_rag.retrieval import Retriever, _match_selection


def _retriever(lib, llm=None) -> Retriever:
    return Retriever(lib.store, lib.metas, lib.embedder, lib.config,
                     llm=llm if llm is not None else lib.llm)


# ---------------------------------------------------------------- selection
def test_match_selection_exact_and_fuzzy():
    page = "Our method encodes queries independently of documents."
    assert _match_selection(page, "encodes queries") == (11, 26, "exact")
    fuzzy = _match_selection(page, "encodes   queries\nindependently")
    assert fuzzy is not None and fuzzy[2] == "fuzzy"
    assert page[fuzzy[0]:fuzzy[1]].startswith("encodes")
    assert _match_selection(page, "totally absent text") is None


# ---------------------------------------------------------------- scope
def test_scope_auto_without_anchor_is_library(lib):
    result = _retriever(lib).retrieve("What is studied?", None, Scope.AUTO, 4)
    assert result.scope_used == "library"


def test_scope_auto_with_anchor_is_paper(lib):
    pid = lib.sample_paper_id
    result = _retriever(lib).retrieve(
        "What is studied?", Anchor(paper_id=pid), Scope.AUTO, 4)
    assert result.scope_used == f"paper:{pid}"
    assert all(c.metadata["paper_id"] == pid for c in result.retrieved_chunks)


def test_scope_paper_without_anchor_raises(lib):
    with pytest.raises(PaperRagError) as err:
        _retriever(lib).retrieve("q", None, Scope.PAPER, 4)
    assert err.value.code == "INVALID_ANCHOR"


def test_unknown_paper_raises(lib):
    with pytest.raises(PaperRagError) as err:
        _retriever(lib).retrieve("q", Anchor(paper_id="ghost"), Scope.AUTO, 4)
    assert err.value.code == "PAPER_NOT_FOUND"


# ---------------------------------------------------------------- anchor
def test_anchored_selection_resolves_chunks(lib):
    pid = lib.sample_paper_id
    anchor = Anchor(paper_id=pid, page=1,
                    selection="encodes queries independently")
    result = _retriever(lib).retrieve("What does this mean?", anchor,
                                      Scope.AUTO, 4)
    assert result.anchored_chunks
    assert all(c.anchored for c in result.anchored_chunks)
    assert result.debug["anchor"]["match"]["kind"] == "exact"
    assert len(result.queries) == 2   # question + question-with-selection


def test_anchor_miss_degrades(lib):
    pid = lib.sample_paper_id
    anchor = Anchor(paper_id=pid, page=1, selection="text that is nowhere")
    result = _retriever(lib).retrieve("q", anchor, Scope.AUTO, 4)
    assert result.anchored_chunks == []
    assert result.debug["anchor"]["match"] == "miss"
    assert result.scope_used == f"paper:{pid}"   # still scopes retrieval


def test_anchor_selection_without_page_searches_paper(lib):
    pid = lib.sample_paper_id
    anchor = Anchor(paper_id=pid, selection="encodes queries independently")
    result = _retriever(lib).retrieve("q", anchor, Scope.AUTO, 4)
    assert result.anchored_chunks
    assert result.debug["anchor"]["match"]["page"] == 1


# ---------------------------------------------------------------- figures
def test_explicit_figure_mention_resolves(lib):
    pid = lib.sample_paper_id
    result = _retriever(lib).retrieve(
        "What does Figure 1 show?", Anchor(paper_id=pid), Scope.AUTO, 4)
    assert [f.figure_id for f in result.figures] == [f"{pid}:fig:1"]
    assert result.figures[0].label == "Figure 1"


def test_table_mention_resolves(lib):
    pid = lib.sample_paper_id
    result = _retriever(lib).retrieve(
        "Explain Table 1 please", Anchor(paper_id=pid), Scope.AUTO, 4)
    assert [f.figure_id for f in result.figures] == [f"{pid}:tab:1"]


def test_llm_fallback_for_indirect_mention(lib):
    pid = lib.sample_paper_id
    llm = FakeLLM(responses=[f"{pid}|fig:1"])
    result = _retriever(lib, llm=llm).retrieve(
        "What does the diagram illustrate?", Anchor(paper_id=pid),
        Scope.AUTO, 4)
    assert result.debug["figures"]["llm_fallback"] == (pid, "fig:1")
    assert [f.figure_id for f in result.figures] == [f"{pid}:fig:1"]


def test_no_figure_words_no_fallback_call(lib):
    pid = lib.sample_paper_id
    llm = FakeLLM(responses=[])   # would raise if consulted
    result = _retriever(lib, llm=llm).retrieve(
        "What is the main contribution?", Anchor(paper_id=pid), Scope.AUTO, 4)
    assert result.debug["figures"]["llm_fallback"] is None


# ---------------------------------------------------------------- citations
def test_inline_citation_resolution_from_anchor(lib):
    pid = lib.sample_paper_id
    anchor = Anchor(paper_id=pid, page=2,
                    selection="Prior work [1] introduced the foundational method.")
    result = _retriever(lib).retrieve("What prior work?", anchor, Scope.AUTO, 4)
    assert result.debug["inline_citations"]["resolved"] == ["1"]
    assert [c.chunk_id for c in result.reference_chunks] == [f"{pid}:r1"]
    assert result.reference_chunks[0].metadata["type"] == "reference"
