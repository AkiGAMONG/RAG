"""D15 — the minimal CLI drives ingest / ask / papers with env-selected fakes."""

import json

from paper_rag.__main__ import main


def test_cli_roundtrip(tmp_path, sample_pdf, monkeypatch, capsys):
    monkeypatch.setenv("PAPER_RAG_LLM_PROVIDER", "fake")
    monkeypatch.setenv("PAPER_RAG_EMBED_PROVIDER", "fake")
    monkeypatch.setenv("PAPER_RAG_EMBED_MODEL", "fake-embed")
    monkeypatch.setenv("PAPER_RAG_EMBED_DIM", "32")
    data_dir = str(tmp_path / "cli_data")

    assert main(["--data-dir", data_dir, "ingest", str(sample_pdf)]) == 0
    out = capsys.readouterr().out
    assert "ingested: " in out

    assert main(["--data-dir", data_dir, "papers"]) == 0
    papers = json.loads(capsys.readouterr().out)
    assert len(papers) == 1 and papers[0]["n_pages"] == 3

    assert main(["--data-dir", data_dir, "ask",
                 "What does Figure 1 show?", "--paper",
                 papers[0]["paper_id"]]) == 0
    out = capsys.readouterr().out
    assert "figure:" in out


def test_cli_error_paths(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PAPER_RAG_LLM_PROVIDER", "fake")
    monkeypatch.setenv("PAPER_RAG_EMBED_PROVIDER", "fake")
    data_dir = str(tmp_path / "cli_data2")
    assert main(["--data-dir", data_dir, "ask", "hello?"]) == 1
    assert "NO_PAPERS" in capsys.readouterr().out
    # --page/--selection without --paper is an input error, not silence
    assert main(["--data-dir", data_dir, "ask", "q", "--page", "3"]) == 1
    assert "--paper" in capsys.readouterr().out


def test_cli_data_dir_after_subcommand(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PAPER_RAG_LLM_PROVIDER", "fake")
    monkeypatch.setenv("PAPER_RAG_EMBED_PROVIDER", "fake")
    data_dir = str(tmp_path / "cli_data3")
    assert main(["papers", "--data-dir", data_dir]) == 0
    assert json.loads(capsys.readouterr().out) == []
