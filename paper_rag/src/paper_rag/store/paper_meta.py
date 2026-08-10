"""Per-paper sidecar storage: meta.json, pages/{n}.txt, source.pdf, figures/.

Directory layout under ``data_dir``:

    data_dir/
    ├── chroma/                  # vector store (vector_store.py)
    └── papers/
        └── {paper_id}/
            ├── meta.json        # api_contract §5 sidecar
            ├── source.pdf       # copy of the ingested PDF (D16: self-contained re-ingest)
            ├── pages/1.txt ...  # canonical page text; char offsets index into these
            └── figures/*.png

meta.json follows the api_contract §5 shape plus bookkeeping fields
(n_chunks, n_figures) that PaperInfo needs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


class PaperMetaStore:
    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir) / "papers"

    def paper_dir(self, paper_id: str) -> Path:
        return self.root / paper_id

    def exists(self, paper_id: str) -> bool:
        return (self.paper_dir(paper_id) / "meta.json").is_file()

    def write_meta(self, paper_id: str, meta: dict) -> None:
        d = self.paper_dir(paper_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def read_meta(self, paper_id: str) -> dict | None:
        path = self.paper_dir(paper_id) / "meta.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_metas(self) -> list[dict]:
        metas = []
        if self.root.is_dir():
            for meta_path in sorted(self.root.glob("*/meta.json")):
                try:
                    metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue  # damaged sidecar; skip rather than fail the listing
        return metas

    def write_page_text(self, paper_id: str, page_number: int, text: str) -> None:
        d = self.paper_dir(paper_id) / "pages"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{page_number}.txt").write_text(text, encoding="utf-8")

    def read_page_text(self, paper_id: str, page_number: int) -> str | None:
        path = self.paper_dir(paper_id) / "pages" / f"{page_number}.txt"
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def delete_paper(self, paper_id: str) -> None:
        shutil.rmtree(self.paper_dir(paper_id), ignore_errors=True)
