from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_chat_returns_valid_envelope() -> None:
    response = client.post("/chat", json={"message": "How do I reset my password?"})
    assert response.status_code == 200

    data = response.json()
    assert "conversation_id" in data
    assert "answer" in data
    assert data["status"] == "abstained"
    assert data["needs_human"] is True
    assert "request_id" in data
    assert "latency_ms" in data
    assert "usage" in data


def test_chat_with_conversation_id() -> None:
    cid = "conv-123"
    response = client.post(
        "/chat", json={"message": "Follow up question", "conversation_id": cid}
    )
    assert response.status_code == 200
    assert response.json()["conversation_id"] == cid


def test_chat_rejects_empty_message() -> None:
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_rejects_oversized_message() -> None:
    response = client.post("/chat", json={"message": "x" * 4001})
    assert response.status_code == 422
