"""Deterministic offline providers for unit tests and API-free smoke runs.

Registered under the provider name "fake" (plan.md D14: unit tests never
touch the network). FakeEmbedder is a bag-of-words hash embedding: texts
sharing tokens land near each other, so retrieval order is meaningful in
tests. FakeLLM either replays scripted responses or synthesizes a grounded-
looking answer from the context it is shown.
"""

from __future__ import annotations

import math
import re

from .base import ImagePart, TextPart

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DOC_TAG_RE = re.compile(r'<doc id="(\d+)"')

# Must equal generation.ABSTAIN_MARKER; duplicated here (string constant, not
# an import) to keep providers free of pipeline imports.
_ABSTAIN = "I don't have that information in the provided documents."

# Question words that carry no topical signal for the overlap heuristic.
_STOPWORDS = frozenset(
    "what which best does show this that how why where when the is are was "
    "were in of a an to for with about and or not it its their there".split())


class FakeEmbedder:
    """Hash-bucket bag-of-words embedding, L2-normalized. Deterministic."""

    def __init__(self, dim: int = 32):
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            # Stable across processes (unlike builtin hash()).
            bucket = sum((i + 1) * b for i, b in enumerate(token.encode())) % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class FakeLLM:
    """Scripted (pass ``responses=[...]``) or heuristic context-echo answers."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses) if responses else None
        self.calls: list[dict] = []   # inspection hook for tests

    def complete(self, system: str, parts: list[TextPart | ImagePart],
                 temperature: float = 0.0) -> str:
        text = "\n".join(p.text for p in parts if isinstance(p, TextPart))
        self.calls.append({
            "system": system,
            "text": text,
            "n_images": sum(1 for p in parts if isinstance(p, ImagePart)),
            "temperature": temperature,
        })
        if self.responses is not None:
            if not self.responses:
                raise AssertionError("FakeLLM ran out of scripted responses")
            return self.responses.pop(0)
        doc_ids = _DOC_TAG_RE.findall(text)
        if not doc_ids:
            return _ABSTAIN
        # Deterministic abstention heuristic: an un-anchored question whose
        # topical tokens never appear in the context is unanswerable — this
        # lets negative controls exercise the abstention path offline.
        context, _, question = text.rpartition("QUESTION:")
        if context and 'anchored="true"' not in context:
            q_tokens = {t for t in _TOKEN_RE.findall(question.lower())
                        if len(t) > 3 and t not in _STOPWORDS}
            c_tokens = set(_TOKEN_RE.findall(context.lower()))
            if q_tokens and len(q_tokens & c_tokens) < min(2, len(q_tokens)):
                return _ABSTAIN
        cites = "".join(f"[{i}]" for i in dict.fromkeys(doc_ids[:2]))
        return (f"FAKE ANSWER (offline provider): based on {len(doc_ids)} context "
                f"documents {cites}. This text is a placeholder produced without "
                f"any language model.")
