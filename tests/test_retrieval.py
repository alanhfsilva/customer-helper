from __future__ import annotations

from dataclasses import dataclass

from app.llm.client import FakeLLMClient
from app.llm.models import Message
from app.models import Chunk, RetrievedChunk
from app.retrieval.condenser import condense_query
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.keyword import KeywordIndex
from app.retrieval.memory_store import InMemoryVectorStore
from app.retrieval.retriever import HybridRetriever
from app.settings import RetrievalConfig

DIM = 8


def _emb(idx: int) -> list[float]:
    v = [0.0] * DIM
    v[idx % DIM] = 1.0
    return v


def _chunk(
    doc_id: str,
    ordinal: int,
    text: str,
    embedding: list[float],
    source_type: str = "help_article",
) -> Chunk:
    return Chunk(
        id=f"{doc_id}:{ordinal}",
        document_id=doc_id,
        ordinal=ordinal,
        text=text,
        heading_path=[doc_id.replace("-", " ").title()],
        token_count=len(text.split()),
        embedding=embedding,
        metadata={
            "source_uri": f"/{doc_id}",
            "title": doc_id.replace("-", " ").title(),
            "source_type": source_type,
        },
    )


CORPUS = [
    _chunk("billing", 0, "Update your payment method in account settings", _emb(0)),
    _chunk("billing", 1, "Refunds are processed within 5-10 business days", _emb(0)),
    _chunk("shipping", 0, "Standard shipping takes 5-7 business days", _emb(1)),
    _chunk("shipping", 1, "Track your order with the tracking number", _emb(1)),
    _chunk("returns", 0, "Return items within 30 days for a full refund", _emb(2)),
    _chunk("account", 0, "Reset your password from the login page", _emb(3)),
    _chunk("account", 1, "Enable two-factor authentication for security", _emb(3)),
    _chunk("pricing", 0, "Enterprise plans start at $99 per month", _emb(4)),
    _chunk("internal", 0, "Internal escalation procedures for agents", _emb(5),
           source_type="policy"),
]

@dataclass(frozen=True)
class _LabeledQuery:
    query: str
    expected: list[str]
    embedding: list[float]


LABELED_QUERIES = [
    _LabeledQuery("payment method", ["billing:0", "billing:1"], _emb(0)),
    _LabeledQuery("shipping delivery time", ["shipping:0", "shipping:1"], _emb(1)),
    _LabeledQuery("return policy", ["returns:0"], _emb(2)),
    _LabeledQuery("reset password", ["account:0", "account:1"], _emb(3)),
    _LabeledQuery("pricing enterprise", ["pricing:0"], _emb(4)),
]


def _make_store_and_index() -> tuple[InMemoryVectorStore, KeywordIndex]:
    store = InMemoryVectorStore()
    store.upsert(CORPUS)
    idx = KeywordIndex()
    idx.add(CORPUS)
    return store, idx


def _config(**overrides: object) -> RetrievalConfig:
    defaults: dict[str, object] = {
        "k": 5,
        "fetch_n": 20,
        "score_threshold": 0.0,
        "low_confidence_floor": 0.4,
        "hybrid_alpha": 0.7,
        "rerank_enabled": False,
    }
    defaults.update(overrides)
    return RetrievalConfig(**defaults)  # type: ignore[arg-type]


class TestCondenser:
    def test_no_history_returns_original(self) -> None:
        llm = FakeLLMClient()
        result = condense_query("How do I pay?", None, llm)
        assert result == "How do I pay?"
        assert len(llm.chat_calls) == 0

    def test_with_history_calls_llm(self) -> None:
        llm = FakeLLMClient(chat_response="standalone billing query")
        history = [
            Message(role="user", content="I have a billing issue"),
            Message(role="assistant", content="I can help with that"),
        ]
        result = condense_query("What are my options?", history, llm)
        assert result == "standalone billing query"
        assert len(llm.chat_calls) == 1


class TestKeywordIndex:
    def test_basic_search(self) -> None:
        idx = KeywordIndex()
        idx.add(CORPUS)
        results = idx.search("payment method", k=3)
        assert len(results) > 0
        assert results[0].chunk_id == "billing:0"

    def test_filters(self) -> None:
        idx = KeywordIndex()
        idx.add(CORPUS)
        results = idx.search(
            "escalation", k=5, filters={"source_type": "help_article"}
        )
        assert all(r.chunk_id != "internal:0" for r in results)

    def test_no_match_returns_empty(self) -> None:
        idx = KeywordIndex()
        idx.add(CORPUS)
        results = idx.search("xyznonexistent", k=5)
        assert results == []


class TestFusion:
    def test_rrf_combines_rankings(self) -> None:
        r1 = [
            RetrievedChunk("a", "d1", "t1", 0.9, "/a", "A", []),
            RetrievedChunk("b", "d2", "t2", 0.8, "/b", "B", []),
        ]
        r2 = [
            RetrievedChunk("b", "d2", "t2", 0.9, "/b", "B", []),
            RetrievedChunk("c", "d3", "t3", 0.7, "/c", "C", []),
        ]
        fused = reciprocal_rank_fusion(r1, r2, k=3)
        assert fused[0].chunk_id == "b"

    def test_rrf_respects_k(self) -> None:
        results = [
            RetrievedChunk(f"c{i}", f"d{i}", f"t{i}", 0.5, f"/{i}", f"T{i}", [])
            for i in range(10)
        ]
        fused = reciprocal_rank_fusion(results, k=3)
        assert len(fused) == 3


def _make_fake_llm(
    emb: list[float],
    chat_response: str = "fake response",
) -> FakeLLMClient:
    llm = FakeLLMClient(embeddings_dim=DIM, chat_response=chat_response)
    llm.embed = lambda texts, e=emb: type(  # type: ignore[method-assign, misc]
        "R", (), {
            "embeddings": [e for _ in texts],
            "usage": type("U", (), {
                "prompt_tokens": 10, "cost_usd": 0.0, "request_id": "x",
            })(),
        }
    )()
    return llm


class TestHybridRetriever:
    def test_retrieves_relevant_chunks(self) -> None:
        store, idx = _make_store_and_index()
        llm = _make_fake_llm(_emb(0))
        retriever = HybridRetriever(store, llm, _config(), keyword_index=idx)
        results = retriever.retrieve("payment billing", k=3)
        assert len(results) > 0
        assert any("billing" in r.chunk_id for r in results)

    def test_metadata_filter_excludes(self) -> None:
        store, idx = _make_store_and_index()
        llm = _make_fake_llm(_emb(5))
        retriever = HybridRetriever(store, llm, _config(), keyword_index=idx)
        results = retriever.retrieve(
            "internal escalation",
            filters={"source_type": "help_article"},
        )
        assert all(r.chunk_id != "internal:0" for r in results)

    def test_score_threshold_filters_low(self) -> None:
        store, idx = _make_store_and_index()
        llm = _make_fake_llm([0.01] * DIM)
        retriever = HybridRetriever(
            store, llm, _config(score_threshold=0.9), keyword_index=idx
        )
        results = retriever.retrieve("nonexistent topic")
        assert len(results) == 0

    def test_pure_semantic_without_keyword_index(self) -> None:
        store, _ = _make_store_and_index()
        llm = _make_fake_llm(_emb(2))
        retriever = HybridRetriever(store, llm, _config())
        results = retriever.retrieve("return policy", k=2)
        assert len(results) > 0
        assert results[0].chunk_id == "returns:0"

    def test_condenses_with_history(self) -> None:
        store, idx = _make_store_and_index()
        llm = _make_fake_llm(_emb(0), chat_response="billing payment")
        retriever = HybridRetriever(store, llm, _config(), keyword_index=idx)
        history = [Message(role="user", content="billing question")]
        results = retriever.retrieve(
            "what options?", history=history, k=3
        )
        assert len(llm.chat_calls) == 1
        assert len(results) > 0


class TestRetrievalQuality:
    def test_recall_at_5_meets_threshold(self) -> None:
        store, idx = _make_store_and_index()
        hits = 0
        total_expected = 0

        for case in LABELED_QUERIES:
            llm = _make_fake_llm(case.embedding)
            retriever = HybridRetriever(
                store, llm, _config(), keyword_index=idx
            )
            results = retriever.retrieve(case.query, k=5)
            retrieved_ids = {r.chunk_id for r in results}

            for expected_id in case.expected:
                total_expected += 1
                if expected_id in retrieved_ids:
                    hits += 1

        recall = hits / total_expected if total_expected > 0 else 0
        assert recall >= 0.85, f"Recall@5 = {recall:.2f}, need >= 0.85"

    def test_mrr_meets_threshold(self) -> None:
        store, idx = _make_store_and_index()
        reciprocal_ranks: list[float] = []

        for case in LABELED_QUERIES:
            llm = _make_fake_llm(case.embedding)
            retriever = HybridRetriever(
                store, llm, _config(), keyword_index=idx
            )
            results = retriever.retrieve(case.query, k=5)
            retrieved_ids = [r.chunk_id for r in results]

            rr = 0.0
            for expected_id in case.expected:
                if expected_id in retrieved_ids:
                    rank = retrieved_ids.index(expected_id) + 1
                    rr = max(rr, 1.0 / rank)
            reciprocal_ranks.append(rr)

        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        assert mrr >= 0.70, f"MRR = {mrr:.2f}, need >= 0.70"

    def test_low_confidence_returns_empty(self) -> None:
        store = InMemoryVectorStore()
        store.upsert(CORPUS)
        llm = _make_fake_llm(_emb(7))
        retriever = HybridRetriever(
            store, llm, _config(score_threshold=0.5)
        )
        results = retriever.retrieve("completely unrelated xyz")
        assert len(results) == 0
