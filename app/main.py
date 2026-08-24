from __future__ import annotations

import json
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

from app.api import dependencies

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
from app.api.routes import router
from app.llm.client import FakeLLMClient, compute_cost
from app.llm.fake_embeddings import bag_of_words_embed
from app.llm.models import ChatResult, Message
from app.models import Chunk
from app.retrieval.keyword import KeywordIndex
from app.retrieval.memory_store import InMemoryVectorStore
from app.retrieval.retriever import HybridRetriever
from app.settings import RetrievalConfig, get_settings
from ingestion.chunker import chunk_document
from ingestion.normalizer import normalize_content

logger = logging.getLogger(__name__)

CORPUS_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "corpus"
_CHUNK_ID_PATTERN = re.compile(r"--- \[([^\]]+)\]")


def _demo_chat(
    messages: list[Message],
    *,
    stream: bool = False,
    max_tokens: int | None = None,
    temperature: float | None = None,
    response_format: dict[str, Any] | None = None,
) -> ChatResult:
    system = next((m.content for m in messages if m.role == "system"), "")
    source_ids = _CHUNK_ID_PATTERN.findall(system)

    context_lines = []
    for line in system.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("[Source:") and not stripped.startswith("---"):
            context_lines.append(stripped)

    context_text = " ".join(context_lines[-10:]) if context_lines else ""

    answer = (
        f"Based on our documentation, here is what I found about your question: "
        f"{context_text[:200]}"
    )

    response = json.dumps({
        "answer": answer,
        "used_sources": source_ids[:3],
    })

    prompt_tokens = sum(len(m.content.split()) * 2 for m in messages)
    completion_tokens = len(response.split()) * 2

    return ChatResult(
        content=response,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=compute_cost("gpt-4o-2024-08-06", prompt_tokens, completion_tokens),
        request_id="demo",
        model="fake-model",
    )


def _seed_corpus(
    store: InMemoryVectorStore,
    llm: FakeLLMClient,
    keyword_index: KeywordIndex,
) -> int:
    settings = get_settings()
    corpus_dir = CORPUS_DIR
    if not corpus_dir.is_dir():
        logger.warning("Corpus directory not found: %s", corpus_dir)
        return 0

    all_chunks: list[Chunk] = []
    for path in sorted(corpus_dir.rglob("*.md")):
        doc_id = path.stem
        content = path.read_text(encoding="utf-8")
        cleaned = normalize_content(content)
        raw_chunks = chunk_document(cleaned, settings.ingestion)
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
            all_chunks.append(chunk)

    store.upsert(all_chunks)
    keyword_index.add(all_chunks)
    return len(all_chunks)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    offline = not settings.openai_api_key

    llm = FakeLLMClient(embeddings_dim=256)
    llm.embed = bag_of_words_embed  # type: ignore[method-assign]
    llm.chat = _demo_chat  # type: ignore[method-assign]

    if offline:
        retrieval_config = RetrievalConfig(
            k=settings.retrieval.k,
            fetch_n=settings.retrieval.fetch_n,
            score_threshold=0.0,
            low_confidence_floor=0.0,
            hybrid_alpha=settings.retrieval.hybrid_alpha,
            rerank_enabled=settings.retrieval.rerank_enabled,
        )
        settings = settings.model_copy(update={"retrieval": retrieval_config})

    store = InMemoryVectorStore()
    keyword_index = KeywordIndex()
    retriever = HybridRetriever(
        store, llm, settings.retrieval, keyword_index=keyword_index,
    )

    dependencies.configure(settings, llm, retriever)

    count = _seed_corpus(store, llm, keyword_index)
    logger.info(
        "Seeded %d chunks from corpus (mode=%s)",
        count, "offline" if offline else "live",
    )

    yield


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


app = create_app()
