"""Shared fixtures: a synthetic three-page paper PDF and a fake-provider
library. Unit tests never touch the network (plan.md D14)."""

from __future__ import annotations

import pymupdf
import pytest

from paper_rag import Config, PaperLibrary


def build_sample_pdf(path) -> None:
    """Three pages: title + hyphenated body; a figure, a table and inline
    references; a numbered bibliography. Repeated header + page numbers to
    exercise the header/footer filter."""
    doc = pymupdf.open()
    w, h = 595, 842

    p1 = doc.new_page(width=w, height=h)
    p1.insert_text((200, 30), "Sample Conference 2024", fontsize=9)
    p1.insert_text((100, 110), "A Study of Late Interaction", fontsize=18)
    p1.insert_text((72, 170), "This paper studies late inter-", fontsize=11)
    p1.insert_text((72, 184), "action between queries and documents.", fontsize=11)
    p1.insert_text((72, 198), "We evaluate on standard benchmarks.", fontsize=11)
    p1.insert_text((72, 250), "Our method encodes queries independently of documents.", fontsize=11)
    p1.insert_text((72, 264), "This enables offline processing of the collection.", fontsize=11)
    p1.insert_text((295, 830), "1", fontsize=9)

    p2 = doc.new_page(width=w, height=h)
    p2.insert_text((200, 30), "Sample Conference 2024", fontsize=9)
    p2.insert_text((72, 70), "As shown in Figure 1, the approach outperforms baselines.", fontsize=11)
    p2.insert_text((72, 84), "Prior work [1] introduced the foundational method.", fontsize=11)
    p2.draw_rect(pymupdf.Rect(100, 110, 300, 240), color=(0, 0, 0), width=1)
    p2.insert_text((100, 265), "Figure 1: A sample diagram illustrating the approach.", fontsize=10)
    p2.insert_text((72, 330), "Additional analysis appears in the results table below.", fontsize=11)
    p2.insert_text((100, 395), "Table 1: Sample results on the benchmark.", fontsize=10)
    for y in (420, 450, 478):
        p2.draw_line(pymupdf.Point(100, y), pymupdf.Point(300, y), width=1)
    p2.insert_text((110, 440), "method A: 0.71   method B: 0.89", fontsize=9)
    p2.insert_text((295, 830), "2", fontsize=9)

    p3 = doc.new_page(width=w, height=h)
    p3.insert_text((200, 30), "Sample Conference 2024", fontsize=9)
    p3.insert_text((72, 70), "References", fontsize=13)
    p3.insert_text((72, 108), "[1] Alice Smith and Bob Jones. 2020. A Foundational", fontsize=10)
    p3.insert_text((72, 121), "Method for Testing. In Proceedings of Tests.", fontsize=10)
    p3.insert_text((72, 141), "[2] Carol White. 2021. Another Approach Entirely.", fontsize=10)
    p3.insert_text((72, 154), "Journal of Examples, 5(2):1-10.", fontsize=10)
    p3.insert_text((295, 830), "3", fontsize=9)

    doc.save(str(path))
    doc.close()


def fake_config(**overrides) -> Config:
    defaults = dict(llm_provider="fake", llm_model="fake-llm",
                    embed_provider="fake", embed_model="fake-embed",
                    embed_dim=32)
    defaults.update(overrides)
    return Config(**defaults)


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdfs") / "sample.pdf"
    build_sample_pdf(path)
    return path


@pytest.fixture()
def lib(tmp_path, sample_pdf):
    """A fresh fake-provider library with the sample paper ingested."""
    library = PaperLibrary(data_dir=str(tmp_path / "data"), config=fake_config())
    pid = library.ingest(str(sample_pdf))
    library.sample_paper_id = pid
    return library
