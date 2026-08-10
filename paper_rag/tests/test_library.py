"""Integration through the public facade with fake providers (no network)."""

import pytest

from conftest import fake_config
from paper_rag import (Anchor, PaperLibrary, PaperRagError, Scope)
from paper_rag.providers.fake import FakeLLM


def test_ingest_returns_stable_id_and_papers_lists_it(lib, sample_pdf):
    papers = lib.papers()
    assert len(papers) == 1
    info = papers[0]
    assert info.paper_id == lib.sample_paper_id
    assert info.title == "A Study of Late Interaction"
    assert info.n_pages == 3
    assert info.n_chunks > 0
    assert info.n_figures == 2
    assert info.ingested_at   # ISO timestamp present


def test_ingest_idempotent(lib, sample_pdf):
    before = lib.store.count()
    pid = lib.ingest(str(sample_pdf))
    assert pid == lib.sample_paper_id
    assert lib.store.count() == before   # replaced, not duplicated
    assert len(lib.papers()) == 1


def test_progress_callback_stages(tmp_path, sample_pdf):
    library = PaperLibrary(data_dir=str(tmp_path / "d"), config=fake_config())
    stages = []
    library.ingest(str(sample_pdf), progress=lambda p: stages.append(p.stage))
    assert set(stages) == {"parsing", "figures", "references",
                           "embedding", "indexing"}


def test_pages_sidecar_written(lib):
    text = lib.metas.read_page_text(lib.sample_paper_id, 1)
    assert text and "interaction between queries" in text
    paper_dir = lib.metas.paper_dir(lib.sample_paper_id)
    assert (paper_dir / "source.pdf").is_file()
    assert (paper_dir / "meta.json").is_file()


def test_ask_returns_grounded_answer(lib):
    answer = lib.ask("What does the method encode independently?")
    assert answer.scope_used == "library"
    assert answer.text
    assert answer.abstained is False
    for citation in answer.citations:
        assert citation.chunk_id in answer.debug["context_chunk_ids"]
        assert citation.paper_id == lib.sample_paper_id
        assert citation.page >= 1
    assert answer.debug["queries"]


def test_ask_anchored_full_pipeline(lib):
    pid = lib.sample_paper_id
    answer = lib.ask(
        "What prior work is this about?",
        anchor=Anchor(paper_id=pid, page=2,
                      selection="Prior work [1] introduced the foundational method."))
    assert answer.scope_used == f"paper:{pid}"
    assert answer.debug["anchor"]["match"]["kind"] == "exact"
    assert answer.debug["inline_citations"]["resolved"] == ["1"]


def test_ask_figure_question_attaches_image(lib):
    pid = lib.sample_paper_id
    answer = lib.ask("What does Figure 1 show?", anchor=Anchor(paper_id=pid))
    assert [f.figure_id for f in answer.figures] == [f"{pid}:fig:1"]


def test_ask_no_papers(tmp_path):
    empty = PaperLibrary(data_dir=str(tmp_path / "empty"), config=fake_config())
    with pytest.raises(PaperRagError) as err:
        empty.ask("anything?")
    assert err.value.code == "NO_PAPERS"


def test_ask_scope_paper_requires_anchor(lib):
    with pytest.raises(PaperRagError) as err:
        lib.ask("q", scope=Scope.PAPER)
    assert err.value.code == "INVALID_ANCHOR"


def test_ask_unknown_paper(lib):
    with pytest.raises(PaperRagError) as err:
        lib.ask("q", anchor=Anchor(paper_id="ghost"))
    assert err.value.code == "PAPER_NOT_FOUND"


def test_history_is_reserved_not_breaking(lib):
    from paper_rag import Turn
    answer = lib.ask("q?", history=[Turn(role="user", content="earlier")])
    assert "history" in answer.debug


def test_get_figure_by_global_and_local_id(lib):
    pid = lib.sample_paper_id
    asset = lib.get_figure(f"{pid}:fig:1")
    assert asset.mime == "image/png"
    assert asset.data[:8] == b"\x89PNG\r\n\x1a\n"
    assert asset.ref.label == "Figure 1"
    # unambiguous local id also resolves (§11 example form)
    assert lib.get_figure("fig:1").ref.figure_id == f"{pid}:fig:1"


def test_get_figure_not_found(lib):
    with pytest.raises(PaperRagError) as err:
        lib.get_figure("fig:99")
    assert err.value.code == "FIGURE_NOT_FOUND"


def test_embedder_mismatch_on_reopen(tmp_path, sample_pdf):
    data_dir = str(tmp_path / "d")
    library = PaperLibrary(data_dir=data_dir, config=fake_config())
    library.ingest(str(sample_pdf))
    with pytest.raises(PaperRagError) as err:
        PaperLibrary(data_dir=data_dir,
                     config=fake_config(embed_model="other-model"))
    assert err.value.code == "EMBEDDER_MISMATCH"
    assert err.value.details["papers"]


def test_ingest_missing_file(lib):
    with pytest.raises(PaperRagError) as err:
        lib.ingest("/nonexistent/paper.pdf")
    assert err.value.code == "INGEST_FAILED"


def test_reingest_from_stored_source_copy(lib):
    """The contract's self-contained re-ingest flow (§5): ingesting the
    stored copy derives the same paper_id and must not self-destruct."""
    pid = lib.sample_paper_id
    stored = lib.metas.paper_dir(pid) / "source.pdf"
    assert lib.ingest(str(stored)) == pid
    assert len(lib.papers()) == 1
    assert stored.is_file()
    assert lib.ask("What does the method encode independently?").text


def test_failed_reingest_preserves_old_data(lib, sample_pdf):
    class ExplodingEmbedder:
        def embed_documents(self, texts):
            raise PaperRagError("PROVIDER_RATE_LIMITED", "boom")

        def embed_query(self, text):
            raise PaperRagError("PROVIDER_RATE_LIMITED", "boom")

    n_before = lib.store.count()
    good_embedder = lib._embedder
    lib._embedder = ExplodingEmbedder()
    with pytest.raises(PaperRagError):
        lib.ingest(str(sample_pdf))
    lib._embedder = good_embedder
    # the old version survived the failed re-ingest
    assert len(lib.papers()) == 1
    assert lib.store.count() == n_before
    assert lib.ask("What does the method encode independently?").text


def test_scripted_llm_injection_for_abstention(lib):
    from paper_rag.generation import ABSTAIN_MARKER
    lib._llm = FakeLLM(responses=[ABSTAIN_MARKER])
    answer = lib.ask("What is the recipe for chocolate cake?")
    assert answer.abstained is True
    assert answer.citations == []
