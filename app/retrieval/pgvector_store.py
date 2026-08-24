from __future__ import annotations

import json
import logging
from typing import Any

from app.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    heading_path JSONB NOT NULL DEFAULT '[]',
    token_count INTEGER NOT NULL,
    embedding vector({dims}),
    metadata JSONB NOT NULL DEFAULT '{{}}'
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
"""

UPSERT_SQL = """
INSERT INTO chunks (id, document_id, ordinal, text, heading_path, token_count, embedding, metadata)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    text = EXCLUDED.text,
    heading_path = EXCLUDED.heading_path,
    token_count = EXCLUDED.token_count,
    embedding = EXCLUDED.embedding,
    metadata = EXCLUDED.metadata
"""

SEARCH_SQL = """
SELECT id, document_id, text, heading_path, metadata,
       1 - (embedding <=> %s::vector) AS score
FROM chunks
{where_clause}
ORDER BY embedding <=> %s::vector
LIMIT %s
"""

DELETE_BY_DOC_SQL = "DELETE FROM chunks WHERE document_id = %s"

GET_CHUNK_IDS_SQL = "SELECT id FROM chunks WHERE document_id = %s"


class PgVectorStore:
    def __init__(self, connection_string: str, *, dimensions: int = 1536) -> None:
        import psycopg

        self._conn = psycopg.connect(connection_string)
        self._dimensions = dimensions
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(SCHEMA_SQL.format(dims=self._dimensions))
        self._conn.commit()

    def upsert(self, chunks: list[Chunk]) -> None:
        with self._conn.cursor() as cur:
            for chunk in chunks:
                embedding_str = "[" + ",".join(str(v) for v in chunk.embedding) + "]"
                cur.execute(
                    UPSERT_SQL,
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.ordinal,
                        chunk.text,
                        json.dumps(chunk.heading_path),
                        chunk.token_count,
                        embedding_str,
                        json.dumps(chunk.metadata),
                    ),
                )
        self._conn.commit()

    def search(
        self,
        embedding: list[float],
        *,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
        where_clause, params = self._build_where(filters)
        query_params: list[Any] = [embedding_str, *params, embedding_str, k]

        with self._conn.cursor() as cur:
            cur.execute(SEARCH_SQL.format(where_clause=where_clause), query_params)
            rows = cur.fetchall()

        return [self._row_to_retrieved_chunk(row) for row in rows]

    @staticmethod
    def _row_to_retrieved_chunk(row: tuple[Any, ...]) -> RetrievedChunk:
        meta_raw = row[4]
        meta = meta_raw if isinstance(meta_raw, dict) else json.loads(meta_raw)
        heading_raw = row[3]
        heading = heading_raw if isinstance(heading_raw, list) else json.loads(heading_raw)
        return RetrievedChunk(
            chunk_id=row[0],
            document_id=row[1],
            text=row[2],
            heading_path=heading,
            score=float(row[5]),
            source_uri=meta.get("source_uri", ""),
            title=meta.get("title", ""),
        )

    def delete_by_document(self, document_id: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute(DELETE_BY_DOC_SQL, (document_id,))
            count = cur.rowcount
        self._conn.commit()
        return count

    def get_document_chunk_ids(self, document_id: str) -> list[str]:
        with self._conn.cursor() as cur:
            cur.execute(GET_CHUNK_IDS_SQL, (document_id,))
            return [row[0] for row in cur.fetchall()]

    @staticmethod
    def _build_where(
        filters: dict[str, Any] | None,
    ) -> tuple[str, list[Any]]:
        if not filters:
            return "", []
        conditions: list[str] = []
        params: list[Any] = []
        for key, value in filters.items():
            conditions.append("metadata->>%s = %s")
            params.extend([key, str(value)])
        return "WHERE " + " AND ".join(conditions), params

    def close(self) -> None:
        self._conn.close()
