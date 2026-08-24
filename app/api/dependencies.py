from __future__ import annotations

from typing import TYPE_CHECKING

from app.api.rate_limit import RateLimiter
from app.feedback.metrics import MetricsCollector
from app.feedback.store import InMemoryFeedbackStore

if TYPE_CHECKING:
    from app.feedback.store import FeedbackStore
    from app.llm.client import LLMClient
    from app.retrieval.keyword import KeywordIndex
    from app.retrieval.retriever import HybridRetriever
    from app.settings import Settings

_settings: Settings | None = None
_llm_client: LLMClient | None = None
_retriever: HybridRetriever | None = None
_keyword_index: KeywordIndex | None = None
_feedback_store: FeedbackStore | None = None
_metrics_collector: MetricsCollector | None = None
_rate_limiter: RateLimiter | None = None


def configure(
    settings: Settings,
    llm_client: LLMClient,
    retriever: HybridRetriever,
    *,
    feedback_store: FeedbackStore | None = None,
    metrics_collector: MetricsCollector | None = None,
    rate_limiter: RateLimiter | None = None,
) -> None:
    global _settings, _llm_client, _retriever  # noqa: PLW0603
    global _feedback_store, _metrics_collector, _rate_limiter  # noqa: PLW0603
    _settings = settings
    _llm_client = llm_client
    _retriever = retriever
    _feedback_store = feedback_store or InMemoryFeedbackStore()
    _metrics_collector = metrics_collector or MetricsCollector()
    _rate_limiter = rate_limiter or RateLimiter()


def get_settings() -> Settings:
    assert _settings is not None, "Settings not configured"
    return _settings


def get_llm_client() -> LLMClient:
    assert _llm_client is not None, "LLM client not configured"
    return _llm_client


def get_retriever() -> HybridRetriever:
    assert _retriever is not None, "Retriever not configured"
    return _retriever


def get_feedback_store() -> FeedbackStore:
    assert _feedback_store is not None, "Feedback store not configured"
    return _feedback_store


def get_metrics_collector() -> MetricsCollector:
    assert _metrics_collector is not None, "Metrics collector not configured"
    return _metrics_collector


def get_rate_limiter() -> RateLimiter:
    assert _rate_limiter is not None, "Rate limiter not configured"
    return _rate_limiter
