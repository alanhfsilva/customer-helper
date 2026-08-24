from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path

from app.api import dependencies
from app.feedback.metrics import MetricsCollector
from app.feedback.store import (
    FeedbackRecord,
    FeedbackSignal,
    InMemoryFeedbackStore,
    export_feedback_to_golden,
)
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


def _setup_app() -> tuple[TestClient, InMemoryFeedbackStore, MetricsCollector]:
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

    feedback_store = InMemoryFeedbackStore()
    metrics_collector = MetricsCollector()

    dependencies.configure(
        settings, llm, retriever,
        feedback_store=feedback_store,
        metrics_collector=metrics_collector,
    )

    application = create_app()
    return TestClient(application), feedback_store, metrics_collector


client, fb_store, metrics_coll = _setup_app()


class TestFeedbackStore:
    def test_save_and_retrieve(self) -> None:
        store = InMemoryFeedbackStore()
        record = FeedbackRecord(
            request_id="req-1",
            signal=FeedbackSignal.THUMBS_UP,
        )
        store.save(record)
        results = store.get_by_request_id("req-1")
        assert len(results) == 1
        assert results[0].signal == FeedbackSignal.THUMBS_UP

    def test_list_all_with_filter(self) -> None:
        store = InMemoryFeedbackStore()
        store.save(FeedbackRecord(request_id="a", signal=FeedbackSignal.THUMBS_UP))
        store.save(FeedbackRecord(request_id="b", signal=FeedbackSignal.THUMBS_DOWN))
        store.save(FeedbackRecord(request_id="c", signal=FeedbackSignal.THUMBS_UP))

        ups = store.list_all(signal=FeedbackSignal.THUMBS_UP)
        assert len(ups) == 2

        downs = store.list_all(signal=FeedbackSignal.THUMBS_DOWN)
        assert len(downs) == 1

    def test_list_all_limit(self) -> None:
        store = InMemoryFeedbackStore()
        for i in range(10):
            store.save(FeedbackRecord(
                request_id=f"req-{i}",
                signal=FeedbackSignal.THUMBS_UP,
            ))
        results = store.list_all(limit=3)
        assert len(results) == 3

    def test_count(self) -> None:
        store = InMemoryFeedbackStore()
        assert store.count == 0
        store.save(FeedbackRecord(request_id="x", signal=FeedbackSignal.RESOLVED))
        assert store.count == 1

    def test_immutability(self) -> None:
        store = InMemoryFeedbackStore()
        record = FeedbackRecord(request_id="x", signal=FeedbackSignal.THUMBS_UP)
        store.save(record)
        old_records = store.list_all()
        store.save(FeedbackRecord(request_id="y", signal=FeedbackSignal.THUMBS_DOWN))
        assert len(old_records) == 1


class TestFeedbackEndpoint:
    def test_submit_feedback(self) -> None:
        response = client.post(
            "/feedback",
            json={
                "request_id": "req-abc",
                "signal": "thumbs_up",
            },
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "req-abc"
        assert data["status"] == "recorded"

    def test_feedback_requires_api_key(self) -> None:
        response = client.post(
            "/feedback",
            json={"request_id": "req-x", "signal": "thumbs_up"},
        )
        assert response.status_code == 401

    def test_feedback_with_agent_edit(self) -> None:
        response = client.post(
            "/feedback",
            json={
                "request_id": "req-edit",
                "signal": "agent_edit",
                "corrected_answer": "The correct answer is 3-5 days.",
                "agent_id": "agent-42",
            },
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 200

    def test_feedback_invalid_signal(self) -> None:
        response = client.post(
            "/feedback",
            json={"request_id": "req-x", "signal": "invalid"},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422

    def test_feedback_stores_record(self) -> None:
        rid = "req-stored"
        client.post(
            "/feedback",
            json={"request_id": rid, "signal": "thumbs_down", "comment": "wrong"},
            headers={"X-API-Key": API_KEY},
        )
        records = fb_store.get_by_request_id(rid)
        assert len(records) >= 1
        assert records[-1].comment == "wrong"

    def test_feedback_with_tags(self) -> None:
        response = client.post(
            "/feedback",
            json={
                "request_id": "req-tagged",
                "signal": "thumbs_up",
                "tags": ["billing", "helpful"],
            },
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 200

    def test_feedback_empty_request_id_rejected(self) -> None:
        response = client.post(
            "/feedback",
            json={"request_id": "", "signal": "thumbs_up"},
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 422


class TestMetricsCollector:
    def test_empty_snapshot(self) -> None:
        collector = MetricsCollector()
        snap = collector.snapshot()
        assert snap.total_requests == 0
        assert snap.deflection_rate == 0.0
        assert snap.escalation_rate == 0.0

    def test_record_answered(self) -> None:
        collector = MetricsCollector()
        collector.record("answered", 100, 0.001)
        snap = collector.snapshot()
        assert snap.total_requests == 1
        assert snap.answered == 1
        assert snap.deflection_rate == 1.0

    def test_record_mixed(self) -> None:
        collector = MetricsCollector()
        collector.record("answered", 100, 0.001)
        collector.record("answered", 200, 0.002)
        collector.record("escalated", 300, 0.003)
        collector.record("abstained", 50, 0.0)
        snap = collector.snapshot()
        assert snap.total_requests == 4
        assert snap.answered == 2
        assert snap.escalated == 1
        assert snap.abstained == 1
        assert snap.deflection_rate == 0.5
        assert snap.escalation_rate == 0.25

    def test_avg_latency(self) -> None:
        collector = MetricsCollector()
        collector.record("answered", 100, 0.0)
        collector.record("answered", 300, 0.0)
        snap = collector.snapshot()
        assert snap.avg_latency_ms == 200.0

    def test_reset(self) -> None:
        collector = MetricsCollector()
        collector.record("answered", 100, 0.001)
        collector.reset()
        snap = collector.snapshot()
        assert snap.total_requests == 0


class TestMetricsEndpoint:
    def test_metrics_returns_data(self) -> None:
        client.post(
            "/chat",
            json={"message": "refund policy"},
            headers={"X-API-Key": API_KEY},
        )
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] >= 1
        assert "deflection_rate" in data
        assert "escalation_rate" in data
        assert "avg_latency_ms" in data
        assert "avg_cost_usd" in data


class TestExportToGolden:
    def test_exports_thumbs_up_with_correction(self, tmp_path: Path) -> None:
        output = tmp_path / "golden.jsonl"
        records = [
            FeedbackRecord(
                request_id="r1",
                signal=FeedbackSignal.THUMBS_UP,
                corrected_answer="The right answer.",
            ),
        ]
        count = export_feedback_to_golden(records, output)
        assert count == 1
        lines = output.read_text().strip().split("\n")
        entry = json.loads(lines[0])
        assert entry["reference_answer"] == "The right answer."
        assert entry["source_request_id"] == "r1"

    def test_exports_agent_edit(self, tmp_path: Path) -> None:
        output = tmp_path / "golden.jsonl"
        records = [
            FeedbackRecord(
                request_id="r2",
                signal=FeedbackSignal.AGENT_EDIT,
                corrected_answer="Agent-corrected answer.",
            ),
        ]
        count = export_feedback_to_golden(records, output)
        assert count == 1

    def test_skips_thumbs_down(self, tmp_path: Path) -> None:
        output = tmp_path / "golden.jsonl"
        records = [
            FeedbackRecord(
                request_id="r3",
                signal=FeedbackSignal.THUMBS_DOWN,
                corrected_answer="Should not export.",
            ),
        ]
        count = export_feedback_to_golden(records, output)
        assert count == 0

    def test_skips_empty_corrected_answer(self, tmp_path: Path) -> None:
        output = tmp_path / "golden.jsonl"
        records = [
            FeedbackRecord(
                request_id="r4",
                signal=FeedbackSignal.THUMBS_UP,
            ),
        ]
        count = export_feedback_to_golden(records, output)
        assert count == 0

    def test_appends_to_existing_file(self, tmp_path: Path) -> None:
        output = tmp_path / "golden.jsonl"
        output.write_text('{"existing": true}\n')
        records = [
            FeedbackRecord(
                request_id="r5",
                signal=FeedbackSignal.AGENT_EDIT,
                corrected_answer="New entry.",
            ),
        ]
        export_feedback_to_golden(records, output)
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2
