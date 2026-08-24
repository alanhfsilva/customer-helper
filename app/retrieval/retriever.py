from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from app.retrieval.condenser import condense_query
from app.retrieval.fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from app.llm.client import LLMClient
    from app.llm.models import Message
    from app.models import RetrievedChunk
    from app.retrieval.keyword import KeywordIndex
    from app.retrieval.store import VectorStore
    from app.settings import RetrievalConfig

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        llm: LLMClient,
        config: RetrievalConfig,
        *,
        keyword_index: KeywordIndex | None = None,
    ) -> None:
        self._store = vector_store
        self._llm = llm
        self._config = config
        self._keyword = keyword_index

    def retrieve(
        self,
        query: str,
        *,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
        history: list[Message] | None = None,
    ) -> list[RetrievedChunk]:
        effective_k = k or self._config.k
        fetch_n = self._config.fetch_n

        search_query = condense_query(query, history, self._llm) if history else query

        embed_result = self._llm.embed([search_query])
        query_embedding = embed_result.embeddings[0]

        semantic_results = self._store.search(
            query_embedding, k=fetch_n, filters=filters
        )

        if self._keyword is not None:
            keyword_results = self._keyword.search(
                search_query, k=fetch_n, filters=filters
            )
            fused = reciprocal_rank_fusion(
                semantic_results, keyword_results, k=fetch_n
            )
        else:
            logger.debug("Keyword index unavailable; pure semantic retrieval")
            fused = semantic_results

        filtered = [
            r for r in fused
            if r.score >= self._config.score_threshold
        ]

        return filtered[:effective_k]
