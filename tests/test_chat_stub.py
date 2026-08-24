from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.api import dependencies
from app.llm.client import FakeLLMClient
from app.main import create_app
from app.models import Chunk
from app.orchestrator import redact_pii
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


def _setup_app() -> TestClient:
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
        Chunk(
            id="shipping:0", document_id="shipping", ordinal=0,
            text="Standard shipping takes 5-7 business days",
            heading_path=["Shipping"], token_count=7,
            embedding=_emb(1),
            metadata={
                "source_uri": "/shipping",
                "title": "Shipping",
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

    dependencies.configure(settings, llm, retriever)

    application = create_app()
    return TestClient(application)


client = _setup_app()


def test_chat_returns_valid_envelope() -> None:
    response = client.post(
        "/chat",
        json={"message": "How do I get a refund?"},
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 200

    data = response.json()
    assert "conversation_id" in data
    assert "answer" in data
    assert "request_id" in data
    assert "latency_ms" in data
    assert "usage" in data
    assert data["status"] in ("answered", "abstained", "escalated", "blocked")


def test_chat_with_conversation_id() -> None:
    cid = "conv-123"
    response = client.post(
        "/chat",
        json={"message": "Follow up question", "conversation_id": cid},
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 200
    assert response.json()["conversation_id"] == cid


def test_chat_rejects_empty_message() -> None:
    response = client.post(
        "/chat",
        json={"message": ""},
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 422


def test_chat_rejects_oversized_message() -> None:
    response = client.post(
        "/chat",
        json={"message": "x" * 4001},
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 422


def test_chat_requires_api_key() -> None:
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 401
    assert "Missing" in response.json()["detail"]


def test_chat_includes_citations_when_answered() -> None:
    response = client.post(
        "/chat",
        json={"message": "refund policy"},
        headers={"X-API-Key": API_KEY},
    )
    data = response.json()
    if data["status"] == "answered":
        assert len(data["citations"]) > 0


def test_chat_includes_grounding_score() -> None:
    response = client.post(
        "/chat",
        json={"message": "billing question"},
        headers={"X-API-Key": API_KEY},
    )
    data = response.json()
    assert "grounding_score" in data
    assert 0.0 <= data["grounding_score"] <= 1.0


def test_chat_includes_confidence() -> None:
    response = client.post(
        "/chat",
        json={"message": "test"},
        headers={"X-API-Key": API_KEY},
    )
    data = response.json()
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0


def test_healthz_returns_ok() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


class TestPIIRedaction:
    def test_redacts_email(self) -> None:
        assert "[REDACTED]" in redact_pii("user@example.com")

    def test_redacts_phone(self) -> None:
        assert "[REDACTED]" in redact_pii("Call 555-123-4567")

    def test_redacts_ssn(self) -> None:
        assert "[REDACTED]" in redact_pii("SSN 123-45-6789")

    def test_preserves_normal_text(self) -> None:
        text = "How do I get a refund?"
        assert redact_pii(text) == text
