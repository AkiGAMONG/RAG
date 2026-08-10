"""1.6 — Chroma roundtrip, §5 metadata schema, fingerprint binding."""

import pytest

from paper_rag.errors import PaperRagError
from paper_rag.ingest.chunker import Chunk
from paper_rag.models import Rect
from paper_rag.providers.fake import FakeEmbedder
from paper_rag.store import PaperMetaStore, VectorStore, rects_from_metadata

FP = "fake:fake-embed:32"


def _chunk(cid: str, paper: str, text: str, page: int = 1,
           ctype: str = "body", figure_id: str = "") -> Chunk:
    return Chunk(chunk_id=cid, paper_id=paper, type=ctype, text=text,
                 page_start=page, page_end=page, char_start=0,
                 char_end=len(text), figure_id=figure_id,
                 rects=[Rect(page, 1, 2, 3, 4)])


@pytest.fixture()
def store(tmp_path):
    emb = FakeEmbedder(dim=32)
    vs = VectorStore(tmp_path, FP)
    chunks = [
        _chunk("a:0001:b0", "a", "late interaction ranking model"),
        _chunk("a:0001:c0", "a", "Figure 1: ranking effectiveness plot",
               ctype="caption", figure_id="fig:1"),
        _chunk("b:0001:b0", "b", "vision language retrieval with images"),
    ]
    vs.add_chunks(chunks, emb.embed_documents([c.text for c in chunks]))
    return vs, emb


def test_query_roundtrip_and_metadata_schema(store):
    vs, emb = store
    rows = vs.query(emb.embed_query("late interaction ranking"), top_k=2)
    assert rows and rows[0]["chunk_id"] == "a:0001:b0"
    meta = rows[0]["metadata"]
    for key in ("paper_id", "type", "page_start", "page_end",
                "char_start", "char_end", "figure_id", "section", "rects_json"):
        assert key in meta
    rects = rects_from_metadata(meta)
    assert rects == [Rect(1, 1, 2, 3, 4)]


def test_paper_prefilter(store):
    vs, emb = store
    rows = vs.query(emb.embed_query("retrieval"), top_k=5, paper_id="b")
    assert rows and all(r["metadata"]["paper_id"] == "b" for r in rows)


def test_page_chunks_and_get_by_ids(store):
    vs, _ = store
    rows = vs.get_page_chunks("a", 1)
    assert {r["chunk_id"] for r in rows} == {"a:0001:b0", "a:0001:c0"}
    assert vs.get_by_ids(["b:0001:b0"])[0]["metadata"]["paper_id"] == "b"


def test_delete_paper_and_counts(store):
    vs, _ = store
    assert vs.count() == 3 and vs.count("a") == 2
    vs.delete_paper("a")
    assert vs.count() == 1 and vs.count("a") == 0


def test_fingerprint_mismatch_raises(tmp_path):
    VectorStore(tmp_path, FP)
    with pytest.raises(PaperRagError) as err:
        VectorStore(tmp_path, "gemini:gemini-embedding-2:3072")
    assert err.value.code == "EMBEDDER_MISMATCH"
    assert err.value.details["stored"] == FP


def test_add_chunks_count_mismatch_is_structured_error(tmp_path):
    vs = VectorStore(tmp_path / "g", FP)
    with pytest.raises(PaperRagError) as err:
        vs.add_chunks([_chunk("x:1", "x", "text one"),
                       _chunk("x:2", "x", "text two")],
                      [[0.0] * 32])   # one vector for two chunks
    assert err.value.code == "PROVIDER_ERROR"
    assert err.value.details["n_chunks"] == 2


def test_empty_store_query_ok(tmp_path):
    vs = VectorStore(tmp_path / "fresh", FP)
    assert vs.query([0.0] * 32, top_k=5) == []


def test_paper_meta_roundtrip(tmp_path):
    metas = PaperMetaStore(tmp_path)
    metas.write_meta("p1", {"paper_id": "p1", "title": "T"})
    metas.write_page_text("p1", 1, "page one text")
    assert metas.read_meta("p1")["title"] == "T"
    assert metas.read_page_text("p1", 1) == "page one text"
    assert metas.read_page_text("p1", 9) is None
    assert [m["paper_id"] for m in metas.list_metas()] == ["p1"]
    metas.delete_paper("p1")
    assert metas.read_meta("p1") is None
