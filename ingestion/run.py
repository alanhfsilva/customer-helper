from __future__ import annotations

import logging
import sys

from app.llm.client import FakeLLMClient
from app.retrieval.memory_store import InMemoryVectorStore
from app.settings import get_settings
from ingestion.connector import MarkdownConnector
from ingestion.pipeline import run_ingestion


def main(corpus_dir: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = get_settings()

    source_dir = corpus_dir or "data/corpus"
    connector = MarkdownConnector(
        source_dir, base_url="https://help.example.com"
    )

    llm = FakeLLMClient(embeddings_dim=settings.models.embedding_dimensions)
    store = InMemoryVectorStore()

    report = run_ingestion(
        connector=connector,
        llm=llm,
        store=store,
        config=settings.ingestion,
    )

    logging.info(
        "Done: %d processed, %d skipped, %d chunks",
        report.documents_processed,
        report.documents_skipped,
        report.chunks_created,
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
