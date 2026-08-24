from __future__ import annotations

from app.llm.client import FakeLLMClient, compute_cost
from app.llm.models import Message


class TestComputeCost:
    def test_known_model_input_only(self) -> None:
        cost = compute_cost("text-embedding-3-small", prompt_tokens=1_000_000)
        assert cost == 0.02

    def test_known_model_input_and_output(self) -> None:
        cost = compute_cost(
            "gpt-4o-2024-08-06", prompt_tokens=1_000_000, completion_tokens=1_000_000
        )
        assert cost == 12.50

    def test_unknown_model_returns_zero(self) -> None:
        cost = compute_cost("unknown-model", prompt_tokens=1000, completion_tokens=500)
        assert cost == 0.0

    def test_zero_tokens(self) -> None:
        cost = compute_cost("gpt-4o-2024-08-06", prompt_tokens=0, completion_tokens=0)
        assert cost == 0.0

    def test_small_token_count(self) -> None:
        cost = compute_cost("gpt-4o-mini-2024-07-18", prompt_tokens=100, completion_tokens=50)
        expected = (100 / 1_000_000) * 0.15 + (50 / 1_000_000) * 0.60
        assert abs(cost - round(expected, 8)) < 1e-10


class TestFakeLLMClientChat:
    def test_returns_canned_response(self) -> None:
        client = FakeLLMClient(chat_response="Hello from fake!")
        result = client.chat([Message(role="user", content="Hi")])
        assert result.content == "Hello from fake!"

    def test_tracks_calls(self) -> None:
        client = FakeLLMClient()
        msgs = [Message(role="user", content="Test")]
        client.chat(msgs)
        client.chat(msgs, temperature=0.5)
        assert len(client.chat_calls) == 2
        assert client.chat_calls[1]["temperature"] == 0.5

    def test_includes_token_counts(self) -> None:
        client = FakeLLMClient()
        result = client.chat([Message(role="user", content="one two three")])
        assert result.prompt_tokens > 0
        assert result.completion_tokens > 0

    def test_includes_cost(self) -> None:
        client = FakeLLMClient()
        result = client.chat([Message(role="user", content="test")])
        assert result.cost_usd >= 0.0

    def test_includes_request_id(self) -> None:
        client = FakeLLMClient()
        r1 = client.chat([Message(role="user", content="a")])
        r2 = client.chat([Message(role="user", content="b")])
        assert r1.request_id != r2.request_id

    def test_passes_kwargs_through(self) -> None:
        client = FakeLLMClient()
        client.chat(
            [Message(role="user", content="x")],
            max_tokens=100,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        call = client.chat_calls[0]
        assert call["max_tokens"] == 100
        assert call["temperature"] == 0.1
        assert call["response_format"] == {"type": "json_object"}


class TestFakeLLMClientEmbed:
    def test_returns_correct_count(self) -> None:
        client = FakeLLMClient(embeddings_dim=8)
        result = client.embed(["hello", "world", "test"])
        assert len(result.embeddings) == 3

    def test_returns_correct_dimensions(self) -> None:
        client = FakeLLMClient(embeddings_dim=64)
        result = client.embed(["hello"])
        assert len(result.embeddings[0]) == 64

    def test_tracks_calls(self) -> None:
        client = FakeLLMClient()
        client.embed(["a", "b"])
        client.embed(["c"])
        assert len(client.embed_calls) == 2
        assert client.embed_calls[0] == ["a", "b"]

    def test_includes_usage(self) -> None:
        client = FakeLLMClient()
        result = client.embed(["hello world"])
        assert result.usage.prompt_tokens > 0
        assert result.usage.cost_usd >= 0.0
        assert result.usage.request_id != ""

    def test_empty_list(self) -> None:
        client = FakeLLMClient()
        result = client.embed([])
        assert result.embeddings == []


class TestFakeLLMClientModerate:
    def test_not_flagged_by_default(self) -> None:
        client = FakeLLMClient()
        result = client.moderate("safe text")
        assert result.flagged is False

    def test_flagged_when_configured(self) -> None:
        client = FakeLLMClient(flagged=True)
        result = client.moderate("anything")
        assert result.flagged is True

    def test_tracks_calls(self) -> None:
        client = FakeLLMClient()
        client.moderate("text one")
        client.moderate("text two")
        assert len(client.moderate_calls) == 2
        assert client.moderate_calls[0] == "text one"


class TestFakeLLMClientProtocol:
    def test_satisfies_protocol(self) -> None:
        client = FakeLLMClient()
        assert hasattr(client, "chat")
        assert hasattr(client, "embed")
        assert hasattr(client, "moderate")
