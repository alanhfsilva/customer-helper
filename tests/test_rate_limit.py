from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.api import dependencies
from app.api.rate_limit import RateLimitConfig, RateLimiter
from app.llm.client import FakeLLMClient
from app.main import create_app
from app.models import Chunk
from app.retrieval.keyword import KeywordIndex
from app.retrieval.memory_store import InMemoryVectorStore
from app.retrieval.retriever import HybridRetriever
from app.settings import get_settings

DIM = 8
API_KEY = "test-key-123"


def _emb(idx: int) -> list[float]:
    v = [0.0] * DIM
    v[idx % DIM] = 1.0
    return v


def _setup_app(rate_limiter: RateLimiter) -> TestClient:
    settings = get_settings()
    store = InMemoryVectorStore()
    chunks = [
        Chunk(
            id="billing:0", document_id="billing", ordinal=0,
            text="Refunds are processed within 5-10 business days",
            heading_path=["Billing"], token_count=8,
            embedding=_emb(0),
            metadata={
                "source_uri": "/billing",
                "title": "Billing",
                "source_type": "help_article",
            },
        ),
    ]
    store.upsert(chunks)

    response_json = json.dumps({
        "answer": "Refunds take 5-10 business days [billing:0]",
        "used_sources": ["billing:0"],
    })
    llm = FakeLLMClient(embeddings_dim=DIM, chat_response=response_json)
    default_emb = _emb(0)
    llm.embed = lambda texts, e=default_emb: type(  # type: ignore[method-assign, misc]
        "R", (), {
            "embeddings": [e for _ in texts],
            "usage": type("U", (), {
                "prompt_tokens": 10, "cost_usd": 0.0, "request_id": "x",
            })(),
        }
    )()

    idx = KeywordIndex()
    idx.add(chunks)
    retriever = HybridRetriever(store, llm, settings.retrieval, keyword_index=idx)

    dependencies.configure(settings, llm, retriever, rate_limiter=rate_limiter)
    application = create_app()
    return TestClient(application)


class TestRateLimiter:
    def test_allows_within_limit(self) -> None:
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=5))
        for _ in range(5):
            assert limiter.is_allowed("caller-1")

    def test_blocks_over_limit(self) -> None:
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=3))
        for _ in range(3):
            assert limiter.is_allowed("caller-1")
        assert not limiter.is_allowed("caller-1")

    def test_separate_callers(self) -> None:
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=2))
        assert limiter.is_allowed("caller-a")
        assert limiter.is_allowed("caller-a")
        assert not limiter.is_allowed("caller-a")
        assert limiter.is_allowed("caller-b")

    def test_default_config(self) -> None:
        config = RateLimitConfig()
        assert config.requests_per_minute == 60
        assert config.window_seconds == 60


class TestRateLimitEndpoint:
    def test_chat_returns_429_when_rate_limited(self) -> None:
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=1))
        client = _setup_app(limiter)

        response = client.post(
            "/chat",
            json={"message": "first request"},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 200

        response = client.post(
            "/chat",
            json={"message": "second request"},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 429
        assert "Rate limit" in response.json()["detail"]

    def test_feedback_returns_429_when_rate_limited(self) -> None:
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=1))
        client = _setup_app(limiter)

        response = client.post(
            "/feedback",
            json={"request_id": "r1", "signal": "thumbs_up"},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 200

        response = client.post(
            "/feedback",
            json={"request_id": "r2", "signal": "thumbs_up"},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 429

    def test_healthz_not_rate_limited(self) -> None:
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=1))
        client = _setup_app(limiter)

        for _ in range(5):
            response = client.get("/healthz")
            assert response.status_code == 200

    def test_metrics_not_rate_limited(self) -> None:
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=1))
        client = _setup_app(limiter)

        for _ in range(5):
            response = client.get("/metrics")
            assert response.status_code == 200
