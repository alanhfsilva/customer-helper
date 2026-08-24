from __future__ import annotations

import math
from typing import Any

from app.models import Chunk, RetrievedChunk


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    def upsert(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk

    def search(
        self,
        embedding: list[float],
        *,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        scored: list[tuple[float, Chunk]] = []

        for chunk in self._chunks.values():
            if filters and not self._matches_filters(chunk, filters):
                continue
            score = _cosine_similarity(embedding, chunk.embedding)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]

        return [
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                text=chunk.text,
                score=score,
                source_uri=chunk.metadata.get("source_uri", ""),
                title=chunk.metadata.get("title", ""),
                heading_path=chunk.heading_path,
            )
            for score, chunk in top
        ]

    def delete_by_document(self, document_id: str) -> int:
        to_delete = [
            cid for cid, c in self._chunks.items() if c.document_id == document_id
        ]
        for cid in to_delete:
            del self._chunks[cid]
        return len(to_delete)

    def get_document_chunk_ids(self, document_id: str) -> list[str]:
        return [
            cid for cid, c in self._chunks.items() if c.document_id == document_id
        ]

    @staticmethod
    def _matches_filters(chunk: Chunk, filters: dict[str, Any]) -> bool:
        return all(
            chunk.metadata.get(key) == value for key, value in filters.items()
        )
