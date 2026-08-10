"""Gemini reference implementations of Embedder and LLMClient (plan.md D7/D17).

Rate handling (numbers measured in the course assignments, plan.md §5):
free-tier embedding allows ~100 texts/minute, so documents are embedded in
batches of 20 with >=20s spacing between requests; generation retries 429s.
Retries exhausted on 429 -> PROVIDER_RATE_LIMITED; other provider failures
-> PROVIDER_ERROR (api_contract §7).

Model quirk (D17): ``gemini-embedding-2`` has NO task_type parameter — the
document/query asymmetry is expressed by prefixing a task instruction on the
query text instead. ``gemini-embedding-001`` keeps the classic task_type
config. Both are handled inside this module; the two-method Embedder protocol
is unchanged.

The API key is read from the GOOGLE_API_KEY environment variable at client
construction. Its value is never stored on our objects, logged, or echoed.
"""

from __future__ import annotations

import os
import time

from ..errors import PROVIDER_ERROR, PROVIDER_RATE_LIMITED, PaperRagError
from .base import ImagePart, TextPart

# Instruction prefix for query-side embedding on models without task_type
# (D17). Documents are embedded raw. NOTE for Jeff: exact wording pending
# verification against official gemini-embedding-2 docs (see PROGRESS.md).
QUERY_INSTRUCTION_PREFIX = (
    "Instruct: Given a search query, retrieve passages from academic papers "
    "that answer the query.\nQuery: "
)

_EMBED_BATCH_SIZE = 20
_EMBED_MIN_INTERVAL = 20.0   # seconds between BATCH embedding requests
_EMBED_SINGLE_INTERVAL = 0.7  # seconds between single-text requests (<100 RPM)
_MAX_RETRIES = 4
_RETRY_WAIT = 60.0           # server-suggested delays can exceed 30s


def _model_uses_task_type(model: str) -> bool:
    """gemini-embedding-001 style models take task_type; -2 does not."""
    return "embedding-001" in model


def _require_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise PaperRagError(
            PROVIDER_ERROR,
            "GOOGLE_API_KEY is not set. Put it in .env (see .env.example) "
            "or the environment.",
        )
    return key


def _is_rate_limit(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code == 429 or "RESOURCE_EXHAUSTED" in str(exc)


class GeminiEmbedder:
    def __init__(self, model: str, dim: int,
                 batch_size: int = _EMBED_BATCH_SIZE,
                 min_interval: float = _EMBED_MIN_INTERVAL,
                 single_interval: float = _EMBED_SINGLE_INTERVAL):
        from google import genai  # deferred so "fake"-only runs never import the SDK
        self._genai = genai
        self._client = genai.Client(api_key=_require_api_key())
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.min_interval = min_interval
        self.single_interval = single_interval
        self._last_request_at = 0.0
        # gemini-embedding-2 treats a multi-text request as ONE multimodal
        # input and returns a single vector (observed live, 2026-07-28) —
        # once detected, all further texts are embedded one request each.
        self._single_mode = False

    # -- pacing & retry ----------------------------------------------------
    def _pace(self, interval: float) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < interval:
            time.sleep(interval - elapsed)

    def _request(self, contents, config, interval: float) -> list[list[float]]:
        """One embed_content API call with pacing and 429 retries."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            self._pace(interval)
            try:
                self._last_request_at = time.monotonic()
                result = self._client.models.embed_content(
                    model=self.model, contents=contents, config=config)
                vectors = [list(e.values) for e in result.embeddings]
                if not vectors:
                    raise PaperRagError(
                        PROVIDER_ERROR, "Embedding response contained no vectors",
                        {"model": self.model})
                return vectors
            except PaperRagError:
                raise
            except Exception as exc:  # SDK raises several error types
                last_exc = exc
                if _is_rate_limit(exc) and attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_WAIT)
                    continue
                if _is_rate_limit(exc):
                    raise PaperRagError(
                        PROVIDER_RATE_LIMITED,
                        f"Embedding rate limit persisted after {_MAX_RETRIES} retries.",
                        {"model": self.model}) from exc
                raise PaperRagError(
                    PROVIDER_ERROR, f"Embedding request failed: {exc}",
                    {"model": self.model}) from exc
        raise PaperRagError(PROVIDER_ERROR, f"Embedding failed: {last_exc}",
                            {"model": self.model})

    def _embed_many(self, texts: list[str], task_type: str | None) -> list[list[float]]:
        from google.genai import types
        # output_dimensionality is always passed: the fingerprint promises
        # `dim`, so the API must be held to it rather than silently using a
        # model default.
        config = types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=self.dim)
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            if len(batch) > 1 and not self._single_mode:
                got = self._request(batch, config, self.min_interval)
                if len(got) == len(batch):
                    vectors.extend(got)
                    continue
                self._single_mode = True   # batch collapsed -> re-embed singly
            for text in batch:
                interval = (self.single_interval if self._single_mode or len(batch) == 1
                            else self.min_interval)
                vectors.append(self._request([text], config, interval)[0])
        return vectors

    # -- Embedder protocol -------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        task = "RETRIEVAL_DOCUMENT" if _model_uses_task_type(self.model) else None
        return self._embed_many(texts, task)

    def embed_query(self, text: str) -> list[float]:
        if _model_uses_task_type(self.model):
            return self._embed_many([text], "RETRIEVAL_QUERY")[0]
        return self._embed_many([prepare_query_text(self.model, text)], None)[0]


def prepare_query_text(model: str, text: str) -> str:
    """Query text as actually sent to the embedding model (unit-testable)."""
    if _model_uses_task_type(model):
        return text
    return QUERY_INSTRUCTION_PREFIX + text


class GeminiLLM:
    def __init__(self, model: str, max_output_tokens: int = 1024):
        from google import genai
        self._client = genai.Client(api_key=_require_api_key())
        self.model = model
        self.max_output_tokens = max_output_tokens

    def complete(self, system: str, parts: list[TextPart | ImagePart],
                 temperature: float = 0.0) -> str:
        from google.genai import types
        contents: list = []
        for part in parts:
            if isinstance(part, TextPart):
                contents.append(types.Part.from_text(text=part.text))
            else:
                contents.append(types.Part.from_bytes(
                    data=part.data, mime_type=part.mime))
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=temperature,
                        max_output_tokens=self.max_output_tokens,
                    ),
                )
                return (resp.text or "").strip()
            except Exception as exc:
                if _is_rate_limit(exc) and attempt < _MAX_RETRIES:
                    time.sleep(30.0)
                    continue
                if _is_rate_limit(exc):
                    raise PaperRagError(
                        PROVIDER_RATE_LIMITED,
                        f"Generation rate limit persisted after {_MAX_RETRIES} retries.",
                        {"model": self.model}) from exc
                raise PaperRagError(
                    PROVIDER_ERROR, f"Generation request failed: {exc}",
                    {"model": self.model}) from exc
        raise PaperRagError(PROVIDER_ERROR, "Generation failed after retries",
                            {"model": self.model})
