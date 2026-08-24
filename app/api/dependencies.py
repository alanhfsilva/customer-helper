from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.llm.client import LLMClient
    from app.retrieval.keyword import KeywordIndex
    from app.retrieval.retriever import HybridRetriever
    from app.settings import Settings

_settings: Settings | None = None
_llm_client: LLMClient | None = None
_retriever: HybridRetriever | None = None
_keyword_index: KeywordIndex | None = None


def configure(
    settings: Settings,
    llm_client: LLMClient,
    retriever: HybridRetriever,
) -> None:
    global _settings, _llm_client, _retriever  # noqa: PLW0603
    _settings = settings
    _llm_client = llm_client
    _retriever = retriever


def get_settings() -> Settings:
    assert _settings is not None, "Settings not configured"
    return _settings


def get_llm_client() -> LLMClient:
    assert _llm_client is not None, "LLM client not configured"
    return _llm_client


def get_retriever() -> HybridRetriever:
    assert _retriever is not None, "Retriever not configured"
    return _retriever
