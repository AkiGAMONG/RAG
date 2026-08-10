"""1.6 — Chroma-backed vector store (api_contract §5).

One collection, cosine space. Embeddings are computed by the caller (via the
Embedder protocol) and passed in explicitly, so documents and queries are
guaranteed to come from the same model. The collection is stamped with the
embedder fingerprint ``provider:model:dim``; opening it with a different
fingerprint raises EMBEDDER_MISMATCH (D16 — vectors from different embedding
models live in unrelated spaces and comparing them silently returns garbage).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import chromadb
from chromadb.config import Settings

from ..errors import EMBEDDER_MISMATCH, PROVIDER_ERROR, PaperRagError
from ..ingest.chunker import Chunk
from ..models import Rect

_COLLECTION = "chunks"


class VectorStore:
    def __init__(self, data_dir: str | Path, fingerprint: str):
        self._client = chromadb.PersistentClient(
            path=str(Path(data_dir) / "chroma"),
            settings=Settings(anonymized_telemetry=False))  # tests stay offline
        self.fingerprint = fingerprint
        try:
            self._collection = self._client.get_collection(_COLLECTION)
            stored = (self._collection.metadata or {}).get("embedder", "")
            if stored != fingerprint:
                raise PaperRagError(
                    EMBEDDER_MISMATCH,
                    f"Index was built with embedder {stored!r} but the "
                    f"configured embedder is {fingerprint!r}. Re-ingest every "
                    f"paper (ingest is idempotent) or restore the old config.",
                    {"stored": stored, "configured": fingerprint})
        except PaperRagError:
            raise
        except Exception:
            self._collection = self._client.create_collection(
                _COLLECTION,
                metadata={"hnsw:space": "cosine", "embedder": fingerprint})

    # ------------------------------------------------------------------ write
    def add_chunks(self, chunks: list[Chunk],
                   embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(embeddings) != len(chunks):
            # e.g. an embedder whose batch semantics collapsed several texts
            # into one vector — fail with a structured error, not deep inside
            # chroma validation.
            raise PaperRagError(
                PROVIDER_ERROR,
                f"Embedder returned {len(embeddings)} vectors for "
                f"{len(chunks)} chunks — provider batch-semantics bug.",
                {"n_chunks": len(chunks), "n_embeddings": len(embeddings)})
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[self._metadata(c) for c in chunks])

    @staticmethod
    def _metadata(chunk: Chunk) -> dict:
        # api_contract §5: scalar values only; complex values JSON-encoded (_json).
        return {
            "paper_id": chunk.paper_id,
            "type": chunk.type,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "figure_id": chunk.figure_id,
            "section": chunk.section,
            "rects_json": json.dumps([asdict(r) for r in chunk.rects]),
        }

    def delete_paper(self, paper_id: str) -> None:
        self._collection.delete(where={"paper_id": paper_id})

    # ------------------------------------------------------------------ read
    def query(self, embedding: list[float], top_k: int,
              paper_id: str | None = None) -> list[dict]:
        """Nearest chunks as [{chunk_id, text, metadata, distance}], best first."""
        total = self._collection.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, total),
            where={"paper_id": paper_id} if paper_id else None,
            include=["documents", "metadatas", "distances"])
        return [
            {"chunk_id": cid, "text": doc, "metadata": meta, "distance": dist}
            for cid, doc, meta, dist in zip(
                result["ids"][0], result["documents"][0],
                result["metadatas"][0], result["distances"][0])
        ]

    def get_page_chunks(self, paper_id: str, page: int) -> list[dict]:
        """All chunks covering a page (for anchor resolution)."""
        result = self._collection.get(
            where={"$and": [
                {"paper_id": {"$eq": paper_id}},
                {"page_start": {"$lte": page}},
                {"page_end": {"$gte": page}},
            ]},
            include=["documents", "metadatas"])
        return self._repack_get(result)

    def get_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        if not chunk_ids:
            return []
        result = self._collection.get(ids=chunk_ids,
                                      include=["documents", "metadatas"])
        return self._repack_get(result)

    @staticmethod
    def _repack_get(result: dict) -> list[dict]:
        return [
            {"chunk_id": cid, "text": doc, "metadata": meta, "distance": None}
            for cid, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"])
        ]

    def count(self, paper_id: str | None = None) -> int:
        if paper_id is None:
            return self._collection.count()
        return len(self._collection.get(where={"paper_id": paper_id},
                                        include=[])["ids"])


def rects_from_metadata(metadata: dict) -> list[Rect]:
    """Decode the rects_json metadata field back into Rect objects."""
    try:
        return [Rect(**r) for r in json.loads(metadata.get("rects_json", "[]"))]
    except (json.JSONDecodeError, TypeError):
        return []
