from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from app.models import Chunk
from ingestion.chunker import chunk_document
from ingestion.normalizer import normalize_content

if TYPE_CHECKING:
    from app.llm.client import LLMClient
    from app.settings import IngestionConfig

CORPUS_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "corpus"


def load_eval_corpus(
    llm: LLMClient,
    config: IngestionConfig,
    corpus_dir: Path | None = None,
) -> list[Chunk]:
    root = corpus_dir or CORPUS_DIR
    chunks: list[Chunk] = []

    for path in sorted(root.rglob("*.md")):
        doc_id = path.stem
        content = path.read_text(encoding="utf-8")
        cleaned = normalize_content(content)
        raw_chunks = chunk_document(cleaned, config)

        if not raw_chunks:
            continue

        texts = [rc.text for rc in raw_chunks]
        embed_result = llm.embed(texts)

        for i, rc in enumerate(raw_chunks):
            chunk = Chunk(
                id=f"{doc_id}:{i}",
                document_id=doc_id,
                ordinal=i,
                text=rc.text,
                heading_path=rc.heading_path,
                token_count=rc.token_count,
                embedding=embed_result.embeddings[i],
                metadata={
                    "source_uri": f"/{doc_id}",
                    "title": doc_id.capitalize(),
                    "source_type": "help_article",
                },
            )
            chunks.append(chunk)

    return chunks
