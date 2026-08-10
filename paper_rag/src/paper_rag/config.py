"""Config: provider/model selection and pipeline knobs (plan.md D7/D16/D17/D18).

Four provider fields resolved by name through the provider registry
(``providers/registry.py``). The embedder fingerprint ``provider:model:dim``
binds an index to the embedding model that built it (api_contract §5).

Environment variables (also read from a ``.env`` file by ``PaperLibrary``)
override the defaults so the CLI and tests can switch providers without code:

    PAPER_RAG_LLM_PROVIDER / PAPER_RAG_LLM_MODEL
    PAPER_RAG_EMBED_PROVIDER / PAPER_RAG_EMBED_MODEL / PAPER_RAG_EMBED_DIM
    PAPER_RAG_CHUNKING            ("paragraph" | "fixed")

API keys (e.g. GOOGLE_API_KEY) are read from the environment by the provider
implementations; their values are never stored on Config and never logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Defaults per plan.md D17 (embedding) and §5 (generation model used in the
# course assignments). The generation model is hot-swappable at any time; the
# embedding model is bound to the index (D16).
DEFAULT_LLM_PROVIDER = "gemini"
DEFAULT_LLM_MODEL = "gemini-3.1-flash-lite"
DEFAULT_EMBED_PROVIDER = "gemini"
DEFAULT_EMBED_MODEL = "gemini-embedding-2"
DEFAULT_EMBED_DIM = 3072
# D17 fallback if the account's tier cannot call gemini-embedding-2:
FALLBACK_EMBED_MODEL = "gemini-embedding-001"


@dataclass
class Config:
    llm_provider: str = DEFAULT_LLM_PROVIDER
    llm_model: str = DEFAULT_LLM_MODEL
    embed_provider: str = DEFAULT_EMBED_PROVIDER
    embed_model: str = DEFAULT_EMBED_MODEL
    embed_dim: int = DEFAULT_EMBED_DIM

    # Chunking (plan.md D18): paragraph-aware recursive by default,
    # "fixed" kept as fallback and A/B baseline.
    chunking_strategy: str = "paragraph"   # "paragraph" | "fixed"
    chunk_size: int = 800
    chunk_overlap: int = 100

    def embedder_fingerprint(self) -> str:
        """api_contract §5: the string an index is stamped with."""
        return f"{self.embed_provider}:{self.embed_model}:{self.embed_dim}"

    @classmethod
    def from_env(cls) -> "Config":
        """Defaults overridden by PAPER_RAG_* environment variables."""
        return cls(
            llm_provider=os.environ.get("PAPER_RAG_LLM_PROVIDER", DEFAULT_LLM_PROVIDER),
            llm_model=os.environ.get("PAPER_RAG_LLM_MODEL", DEFAULT_LLM_MODEL),
            embed_provider=os.environ.get("PAPER_RAG_EMBED_PROVIDER", DEFAULT_EMBED_PROVIDER),
            embed_model=os.environ.get("PAPER_RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL),
            embed_dim=int(os.environ.get("PAPER_RAG_EMBED_DIM", str(DEFAULT_EMBED_DIM))),
            chunking_strategy=os.environ.get("PAPER_RAG_CHUNKING", "paragraph"),
        )
