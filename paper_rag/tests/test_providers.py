"""1.6 — fake providers, registry resolution, gemini query-side preparation."""

import pytest

from conftest import fake_config
from paper_rag.errors import PaperRagError
from paper_rag.providers import create_embedder, create_llm
from paper_rag.providers.base import TextPart
from paper_rag.providers.fake import FakeEmbedder, FakeLLM
from paper_rag.providers.gemini import (QUERY_INSTRUCTION_PREFIX,
                                        prepare_query_text)


def test_fake_embedder_deterministic_and_normalized():
    emb = FakeEmbedder(dim=32)
    v1 = emb.embed_query("late interaction over BERT")
    v2 = emb.embed_query("late interaction over BERT")
    assert v1 == v2 and len(v1) == 32
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-6


def test_fake_embedder_similarity_orders_by_token_overlap():
    emb = FakeEmbedder(dim=64)
    q = emb.embed_query("late interaction ranking")
    near = emb.embed_documents(["late interaction is a ranking paradigm"])[0]
    far = emb.embed_documents(["chocolate cake recipe with butter"])[0]
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert dot(q, near) > dot(q, far)


def test_fake_llm_scripted_and_heuristic():
    scripted = FakeLLM(responses=["first", "second"])
    assert scripted.complete("sys", [TextPart(text="x")]) == "first"
    assert scripted.complete("sys", [TextPart(text="x")]) == "second"

    heuristic = FakeLLM()
    reply = heuristic.complete("sys", [TextPart(text='<doc id="1">a</doc>')])
    assert "[1]" in reply
    empty = heuristic.complete("sys", [TextPart(text="no docs")])
    assert empty.startswith("I don't have that information")


def test_fake_llm_abstains_when_question_off_topic():
    llm = FakeLLM()
    context = ('CONTEXT:\n<doc id="1">late interaction ranking over BERT '
               'embeddings</doc>\nQUESTION: ')
    off_topic = llm.complete("sys", [TextPart(text=context +
                                              "chocolate cake recipe?")])
    assert off_topic.startswith("I don't have that information")
    on_topic = llm.complete("sys", [TextPart(text=context +
                                             "how does late interaction rank?")])
    assert "[1]" in on_topic


def test_registry_resolves_fakes():
    config = fake_config()
    assert isinstance(create_embedder(config), FakeEmbedder)
    assert isinstance(create_llm(config), FakeLLM)


def test_registry_unknown_provider():
    with pytest.raises(PaperRagError) as err:
        create_embedder(fake_config(embed_provider="nope"))
    assert err.value.code == "PROVIDER_ERROR"


def test_anthropic_content_building():
    from paper_rag.providers.anthropic import build_content
    from paper_rag.providers.base import ImagePart
    content = build_content([TextPart(text="hello"),
                             ImagePart(mime="image/png", data=b"PNG")])
    assert content[0] == {"type": "text", "text": "hello"}
    assert content[1]["type"] == "image"
    assert content[1]["source"]["media_type"] == "image/png"
    import base64
    assert base64.b64decode(content[1]["source"]["data"]) == b"PNG"


def test_anthropic_registry_and_model_guard(monkeypatch):
    from paper_rag.providers.anthropic import (DEFAULT_ANTHROPIC_MODEL,
                                               AnthropicLLM)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    llm = create_llm(fake_config(llm_provider="anthropic",
                                 llm_model="claude-sonnet-5"))
    assert isinstance(llm, AnthropicLLM)
    assert llm.model == "claude-sonnet-5"
    # provider switched but llm_model left at the Gemini default -> guard kicks in
    guarded = create_llm(fake_config(llm_provider="anthropic",
                                     llm_model="gemini-3.1-flash-lite"))
    assert guarded.model == DEFAULT_ANTHROPIC_MODEL


def test_anthropic_temperature_deprecation_fallback(monkeypatch):
    """claude-sonnet-5 rejects `temperature` with HTTP 400; the client must
    drop the parameter and retry, then stop sending it."""
    from types import SimpleNamespace
    from paper_rag.providers.anthropic import AnthropicLLM

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    llm = AnthropicLLM(model="claude-sonnet-5")

    class TemperatureDeprecated(Exception):
        status_code = 400

        def __str__(self):
            return "`temperature` is deprecated for this model."

    seen_kwargs = []

    def fake_create(**kwargs):
        seen_kwargs.append(kwargs)
        if "temperature" in kwargs:
            raise TemperatureDeprecated()
        return SimpleNamespace(content=[SimpleNamespace(type="text",
                                                        text="answer [1]")])

    llm._client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    assert llm.complete("sys", [TextPart(text="q")]) == "answer [1]"
    assert llm._temperature_unsupported is True
    assert "temperature" in seen_kwargs[0] and "temperature" not in seen_kwargs[1]
    # subsequent calls never send it again
    llm.complete("sys", [TextPart(text="q2")])
    assert "temperature" not in seen_kwargs[2]


def test_anthropic_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(PaperRagError) as err:
        create_llm(fake_config(llm_provider="anthropic",
                               llm_model="claude-sonnet-5"))
    assert err.value.code == "PROVIDER_ERROR"
    assert "ANTHROPIC_API_KEY" in err.value.message


def test_gemini_batch_collapse_falls_back_to_single(monkeypatch):
    """gemini-embedding-2 returns ONE vector for a multi-text request
    (treats the batch as one multimodal input); the embedder must detect
    this and re-embed text by text."""
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key-not-real")
    from paper_rag.providers.gemini import GeminiEmbedder
    emb = GeminiEmbedder(model="gemini-embedding-2", dim=8, batch_size=3,
                         min_interval=0.0, single_interval=0.0)
    calls = []

    def fake_request(contents, config, interval):
        calls.append(list(contents))
        if len(contents) > 1:
            return [[0.5] * 8]          # collapsed: one vector for the batch
        return [[float(len(contents[0]))] * 8]

    monkeypatch.setattr(emb, "_request", fake_request)
    vectors = emb.embed_documents(["aa", "bbb", "c", "dd"])
    assert len(vectors) == 4            # every text got its own vector
    assert emb._single_mode is True
    assert vectors[0][0] == 2.0 and vectors[1][0] == 3.0
    # exactly one wasted batch call, then singles only
    assert [len(c) for c in calls] == [3, 1, 1, 1, 1]


def test_gemini_healthy_batching_kept(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key-not-real")
    from paper_rag.providers.gemini import GeminiEmbedder
    emb = GeminiEmbedder(model="gemini-embedding-001", dim=8, batch_size=3,
                         min_interval=0.0)
    monkeypatch.setattr(emb, "_request",
                        lambda contents, config, interval:
                        [[1.0] * 8 for _ in contents])
    assert len(emb.embed_documents(["a", "b", "c", "d", "e"])) == 5
    assert emb._single_mode is False


def test_gemini_query_preparation_by_model():
    # gemini-embedding-2 has no task_type: instruction goes into the text (D17)
    prepared = prepare_query_text("gemini-embedding-2", "what is colbert?")
    assert prepared.startswith(QUERY_INSTRUCTION_PREFIX)
    assert prepared.endswith("what is colbert?")
    # gemini-embedding-001 keeps task_type: text must stay untouched
    assert prepare_query_text("gemini-embedding-001", "q") == "q"
