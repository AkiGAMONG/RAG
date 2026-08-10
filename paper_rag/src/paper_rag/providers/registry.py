"""Provider registry (plan.md D16): name -> factory.

Adding a provider = implement the Protocols in base.py + register one entry
here (or call the register_* functions at runtime). Factories receive the
full Config so they can read model names and dimensions.
"""

from __future__ import annotations

from typing import Callable

from ..config import Config
from ..errors import PROVIDER_ERROR, PaperRagError
from .base import Embedder, LLMClient

EmbedderFactory = Callable[[Config], Embedder]
LLMFactory = Callable[[Config], LLMClient]


def _gemini_embedder(config: Config) -> Embedder:
    from .gemini import GeminiEmbedder
    return GeminiEmbedder(model=config.embed_model, dim=config.embed_dim)


def _gemini_llm(config: Config) -> LLMClient:
    from .gemini import GeminiLLM
    return GeminiLLM(model=config.llm_model)


def _anthropic_llm(config: Config) -> LLMClient:
    from .anthropic import DEFAULT_ANTHROPIC_MODEL, AnthropicLLM
    # Guard against a half-switched config (provider=anthropic but llm_model
    # still the Gemini default): fall back to the Anthropic default model.
    model = config.llm_model
    if model.startswith("gemini"):
        model = DEFAULT_ANTHROPIC_MODEL
    return AnthropicLLM(model=model)


def _fake_embedder(config: Config) -> Embedder:
    from .fake import FakeEmbedder
    return FakeEmbedder(dim=config.embed_dim)


def _fake_llm(config: Config) -> LLMClient:
    from .fake import FakeLLM
    return FakeLLM()


_EMBEDDERS: dict[str, EmbedderFactory] = {
    "gemini": _gemini_embedder,
    "fake": _fake_embedder,
}
_LLMS: dict[str, LLMFactory] = {
    "gemini": _gemini_llm,
    "anthropic": _anthropic_llm,
    "fake": _fake_llm,
}


def register_embedder(name: str, factory: EmbedderFactory) -> None:
    _EMBEDDERS[name] = factory


def register_llm(name: str, factory: LLMFactory) -> None:
    _LLMS[name] = factory


def create_embedder(config: Config) -> Embedder:
    try:
        factory = _EMBEDDERS[config.embed_provider]
    except KeyError:
        raise PaperRagError(
            PROVIDER_ERROR,
            f"Unknown embed_provider {config.embed_provider!r}; "
            f"registered: {sorted(_EMBEDDERS)}")
    return factory(config)


def create_llm(config: Config) -> LLMClient:
    try:
        factory = _LLMS[config.llm_provider]
    except KeyError:
        raise PaperRagError(
            PROVIDER_ERROR,
            f"Unknown llm_provider {config.llm_provider!r}; "
            f"registered: {sorted(_LLMS)}")
    return factory(config)
