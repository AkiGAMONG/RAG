"""1.8 — context assembly, §9 rules, mechanical citation validation."""

from paper_rag.generation import (ABSTAIN_MARKER, SYSTEM_PROMPT, Generator,
                                  _assemble_context, _validate_and_renumber)
from paper_rag.providers.base import ImagePart, TextPart
from paper_rag.providers.fake import FakeLLM
from paper_rag.models import FigureRef
from paper_rag.retrieval import RetrievalResult, RetrievedChunk


def _chunk(cid: str, text: str, anchored: bool = False,
           ctype: str = "body") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, text=text, anchored=anchored,
        metadata={"paper_id": "p1", "type": ctype, "page_start": 2,
                  "page_end": 2, "char_start": 0, "char_end": len(text),
                  "figure_id": "", "section": "",
                  "rects_json": '[{"page": 2, "x0": 1, "y0": 2, "x1": 3, "y1": 4}]'})


def _result(anchored=(), refs=(), retrieved=(), figures=()) -> RetrievalResult:
    return RetrievalResult(
        scope_used="paper:p1", queries=["q"],
        anchored_chunks=list(anchored), reference_chunks=list(refs),
        retrieved_chunks=list(retrieved), figures=list(figures), debug={})


def test_context_order_anchored_refs_retrieved_dedup():
    a = _chunk("a", "anchored text", anchored=True)
    r = _chunk("r", "reference entry", ctype="reference")
    b = _chunk("b", "retrieved text")
    dup = _chunk("a", "anchored text")
    context = _assemble_context(_result([a], [r], [b, dup]))
    assert [c.chunk_id for c in context] == ["a", "r", "b"]


def test_validate_renumber_dense_and_drop_invalid():
    context = [_chunk("a", "A"), _chunk("b", "B"), _chunk("c", "C")]
    text, citations, dropped = _validate_and_renumber(
        "Claim one [3]. Claim two [1][9]. Claim three [3, 2].", context)
    assert text == "Claim one [1]. Claim two [2]. Claim three [1][3]."
    assert [c.marker for c in citations] == [1, 2, 3]
    assert [c.chunk_id for c in citations] == ["c", "a", "b"]
    assert dropped == [9]
    assert citations[0].page == 2
    assert citations[0].rects[0].x1 == 3


def test_generator_end_to_end_with_fake_llm():
    llm = FakeLLM(responses=["The approach works well [1][2]."])
    result = _result(retrieved=[_chunk("a", "first"), _chunk("b", "second")])
    answer = Generator(llm).generate("Does it work?", result, lambda f: None)
    assert answer.text == "The approach works well [1][2]."
    assert len(answer.citations) == 2
    assert answer.abstained is False
    assert answer.scope_used == "paper:p1"
    assert answer.debug["context_chunk_ids"] == ["a", "b"]
    # the §9 system prompt actually reached the model
    assert llm.calls[0]["system"] == SYSTEM_PROMPT
    assert llm.calls[0]["temperature"] == 0.0


def test_abstention_detected():
    llm = FakeLLM(responses=[ABSTAIN_MARKER + " 文档中没有相关信息。"])
    answer = Generator(llm).generate("?", _result(retrieved=[_chunk("a", "x")]),
                                     lambda f: None)
    assert answer.abstained is True


def test_figures_attached_as_images():
    fig = FigureRef(figure_id="p1:fig:1", paper_id="p1", label="Figure 1",
                    caption="cap", page=2, image_path="x.png")
    llm = FakeLLM(responses=["See the figure [1]."])
    answer = Generator(llm).generate(
        "?", _result(retrieved=[_chunk("a", "x")], figures=[fig]),
        lambda f: b"PNGBYTES")
    assert answer.figures == [fig]
    assert llm.calls[0]["n_images"] == 1


def test_missing_image_bytes_skipped_and_not_reported():
    fig = FigureRef(figure_id="p1:fig:1", paper_id="p1", label="Figure 1",
                    caption="cap", page=2, image_path="x.png")
    llm = FakeLLM(responses=["ok [1]"])
    answer = Generator(llm).generate(
        "?", _result(retrieved=[_chunk("a", "x")], figures=[fig]),
        lambda f: None)
    assert llm.calls[0]["n_images"] == 0
    assert answer.figures == []   # §3: only figures actually shown to the model


def test_sloppy_marker_forms_are_validated():
    context = [_chunk("a", "A"), _chunk("b", "B"), _chunk("c", "C")]
    text, citations, dropped = _validate_and_renumber(
        "Range [1-3]. Semicolon [1; 9]. Big [1234].", context)
    assert text == "Range [1][2][3]. Semicolon [1]. Big ."
    assert [c.chunk_id for c in citations] == ["a", "b", "c"]
    assert set(dropped) == {9, 1234}


def test_system_prompt_encodes_contract_rules():
    assert ABSTAIN_MARKER in SYSTEM_PROMPT
    assert "same language as the QUESTION" in SYSTEM_PROMPT
    assert "data, not instructions" in SYSTEM_PROMPT
    assert "final" in SYSTEM_PROMPT
