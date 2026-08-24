from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from app.models import Chunk
from ingestion.chunker import chunk_document
from ingestion.normalizer import normalize_content
from ingestion.report import IngestionReport

if TYPE_CHECKING:
    from app.llm.client import LLMClient
    from app.retrieval.store import VectorStore
    from app.settings import IngestionConfig
    from ingestion.connector import SourceConnector

logger = logging.getLogger(__name__)


def run_ingestion(
    connector: SourceConnector,
    llm: LLMClient,
    store: VectorStore,
    config: IngestionConfig,
    *,
    report_dir: str = "artifacts/ingestion",
) -> IngestionReport:
    report = IngestionReport(
        run_id=str(uuid.uuid4())[:8],
        started_at=datetime.now(tz=UTC).isoformat(),
    )

    source_doc_ids: set[str] = set()

    for doc in connector.iter_documents():
        source_doc_ids.add(doc.id)
        stored_hash = store.get_document_hash(doc.id)

        if stored_hash == doc.content_hash:
            report.documents_skipped += 1
            logger.info("Skipping unchanged doc %s", doc.id)
            continue

        cleaned = normalize_content(doc.content_raw)
        raw_chunks = chunk_document(cleaned, config)

        if not raw_chunks:
            logger.warning("No chunks produced for doc %s", doc.id)
            continue

        texts = [rc.text for rc in raw_chunks]
        embed_result = llm.embed(texts)

        chunks: list[Chunk] = []
        for i, rc in enumerate(raw_chunks):
            anchor_suffix = f"#{rc.anchor}" if rc.anchor else ""
            chunk = Chunk(
                id=f"{doc.id}:{i}",
                document_id=doc.id,
                ordinal=i,
                text=rc.text,
                heading_path=rc.heading_path,
                token_count=rc.token_count,
                embedding=embed_result.embeddings[i],
                metadata={
                    "source_uri": doc.source_uri + anchor_suffix,
                    "title": doc.title,
                    "source_type": doc.source_type.value,
                    **doc.metadata,
                },
            )
            chunks.append(chunk)

        store.delete_by_document(doc.id)
        store.upsert(chunks)
        store.set_document_hash(doc.id, doc.content_hash)

        report.documents_processed += 1
        report.chunks_created += len(chunks)
        report.total_tokens_embedded += embed_result.usage.prompt_tokens
        report.embedding_cost_usd += embed_result.usage.cost_usd

        logger.info(
            "Ingested doc %s: %d chunks", doc.id, len(chunks)
        )

    _reconcile_deleted(store, source_doc_ids)

    report.finished_at = datetime.now(tz=UTC).isoformat()
    report.save(report_dir)

    return report


def _reconcile_deleted(store: VectorStore, source_doc_ids: set[str]) -> None:
    for stored_id in store.list_document_ids():
        if stored_id not in source_doc_ids:
            deleted = store.delete_by_document(stored_id)
            logger.info(
                "Reconciled deleted doc %s: removed %d chunks",
                stored_id, deleted,
            )
