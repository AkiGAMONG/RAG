"""Provider abstractions (api_contract §8) and the message part types.

Two methods on the Embedder on purpose — do NOT merge them: providers like
Gemini apply asymmetric optimizations to the document side vs the query side.
Providers without this distinction simply route both methods to the same
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class TextPart:
    text: str


@dataclass
class ImagePart:
    mime: str        # e.g. "image/png"
    data: bytes


@runtime_checkable
class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, system: str, parts: list[TextPart | ImagePart],
                 temperature: float = 0.0) -> str: ...
