"""Structured errors (api_contract.md §7).

Callers switch on ``PaperRagError.code``, never on exception type.
"""

from __future__ import annotations

# Error codes (the contract; exception types are not).
PAPER_NOT_FOUND = "PAPER_NOT_FOUND"
FIGURE_NOT_FOUND = "FIGURE_NOT_FOUND"
INVALID_ANCHOR = "INVALID_ANCHOR"
INGEST_FAILED = "INGEST_FAILED"
PROVIDER_RATE_LIMITED = "PROVIDER_RATE_LIMITED"
PROVIDER_ERROR = "PROVIDER_ERROR"
NO_PAPERS = "NO_PAPERS"
EMBEDDER_MISMATCH = "EMBEDDER_MISMATCH"


class PaperRagError(Exception):
    """The one public exception. ``code`` is from the constants above."""

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.details = details or {}
