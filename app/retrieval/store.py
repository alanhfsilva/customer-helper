from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from app.models import Chunk, RetrievedChunk


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk]) -> None: ...

    def search(
        self,
        embedding: list[float],
        *,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...

    def delete_by_document(self, document_id: str) -> int: ...

    def get_document_chunk_ids(self, document_id: str) -> list[str]: ...
