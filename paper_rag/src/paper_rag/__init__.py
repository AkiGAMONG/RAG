"""paper_rag — grounded, anchored, figure/reference-aware Q&A over paper PDFs.

Public API per project_docs/api_contract.md. Quickstart:

    from paper_rag import PaperLibrary, Anchor

    lib = PaperLibrary(data_dir="./mydata")
    pid = lib.ingest("colbert.pdf")
    ans = lib.ask("What is the core contribution?")
"""

from .config import Config
from .errors import PaperRagError
from .generation import ABSTAIN_MARKER
from .library import PaperLibrary
from .models import (Anchor, Answer, Citation, FigureAsset, FigureRef,
                     IngestProgress, PaperInfo, ProgressFn, Rect, Scope, Turn)

__version__ = "0.1.0"

__all__ = [
    "PaperLibrary", "Config", "PaperRagError", "ABSTAIN_MARKER",
    "Anchor", "Answer", "Citation", "FigureAsset", "FigureRef",
    "IngestProgress", "PaperInfo", "ProgressFn", "Rect", "Scope", "Turn",
    "__version__",
]
