"""Anthropic (Claude) LLMClient implementation (contract §8, D7/D16).

Anthropic has no embedding API, so a Claude generation config always pairs
with a non-Anthropic embedder (e.g. gemini-embedding-2) — exactly the mixed
configuration the provider registry exists for. All current Claude models
accept image input, so figure attachment works unchanged.

Select via environment / .env:

    PAPER_RAG_LLM_PROVIDER=anthropic
    PAPER_RAG_LLM_MODEL=claude-sonnet-5

The API key is read from ANTHROPIC_API_KEY at client construction. Its value
is never stored on our objects, logged, or echoed. NOTE: the Anthropic API
is pay-per-token (no free tier).
"""

from __future__ import annotations

import base64
import os
import time

from ..errors import PROVIDER_ERROR, PROVIDER_RATE_LIMITED, PaperRagError
from .base import ImagePart, TextPart

# Used when the provider is switched to "anthropic" but llm_model still holds
# the Gemini default — a friendlier outcome than sending a Gemini model id to
# the Anthropic API.
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"

_MAX_RETRIES = 4
_RETRY_WAIT = 30.0


def build_content(parts: list[TextPart | ImagePart]) -> list[dict]:
    """Convert protocol parts to Anthropic message content blocks
    (unit-testable without a client or key)."""
    content: list[dict] = []
    for part in parts:
        if isinstance(part, TextPart):
            content.append({"type": "text", "text": part.text})
        else:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": part.mime,
                    "data": base64.b64encode(part.data).decode("ascii"),
                },
            })
    return content


def _require_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise PaperRagError(
            PROVIDER_ERROR,
            "ANTHROPIC_API_KEY is not set. Put it in .env (see .env.example) "
            "or the environment.",
        )
    return key


def _is_rate_limit(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status in (429, 529) or "rate_limit" in str(exc) or "overloaded" in str(exc)


class AnthropicLLM:
    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL,
                 max_tokens: int = 1024):
        import anthropic  # deferred: only real Anthropic runs need the SDK
        self._client = anthropic.Anthropic(api_key=_require_api_key())
        self.model = model
        self.max_tokens = max_tokens
        # Claude 5 models reject the temperature parameter outright
        # ("`temperature` is deprecated for this model", HTTP 400) — detected
        # live and remembered so we stop sending it. Impact on D12's
        # "temperature=0 for reproducibility" is recorded in PROGRESS.md.
        self._temperature_unsupported = False

    def complete(self, system: str, parts: list[TextPart | ImagePart],
                 temperature: float = 0.0) -> str:
        messages = [{"role": "user", "content": build_content(parts)}]
        for attempt in range(_MAX_RETRIES + 1):
            kwargs: dict = dict(model=self.model, max_tokens=self.max_tokens,
                                system=system, messages=messages)
            if not self._temperature_unsupported:
                kwargs["temperature"] = temperature
            try:
                resp = self._client.messages.create(**kwargs)
                return "".join(block.text for block in resp.content
                               if getattr(block, "type", "") == "text").strip()
            except Exception as exc:
                if (not self._temperature_unsupported
                        and getattr(exc, "status_code", None) == 400
                        and "temperature" in str(exc)):
                    self._temperature_unsupported = True
                    continue   # immediate retry without the parameter
                if _is_rate_limit(exc) and attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_WAIT)
                    continue
                if _is_rate_limit(exc):
                    raise PaperRagError(
                        PROVIDER_RATE_LIMITED,
                        f"Anthropic rate limit persisted after {_MAX_RETRIES} retries.",
                        {"model": self.model}) from exc
                raise PaperRagError(
                    PROVIDER_ERROR, f"Anthropic request failed: {exc}",
                    {"model": self.model}) from exc
        raise PaperRagError(PROVIDER_ERROR, "Anthropic generation failed after retries",
                            {"model": self.model})
