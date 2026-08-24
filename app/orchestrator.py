from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.api.models import AnswerStatus, Citation, UsageInfo
from app.generation.generator import generate_answer

if TYPE_CHECKING:
    from app.llm.client import LLMClient
    from app.llm.models import Message
    from app.models import RetrievedChunk
    from app.retrieval.retriever import HybridRetriever
    from app.settings import Settings

logger = logging.getLogger(__name__)

PII_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    r"|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
    r"|\b\d{3}-\d{2}-\d{4}\b"
)


def redact_pii(text: str) -> str:
    return PII_PATTERN.sub("[REDACTED]", text)


@dataclass(frozen=True)
class OrchestratorResult:
    answer: str
    citations: list[Citation]
    status: AnswerStatus
    grounding_score: float
    confidence: float
    needs_human: bool
    usage: UsageInfo
    request_id: str
    latency_ms: int


def _build_citations(
    gen_citations: list[dict[str, object]],
) -> list[Citation]:
    result: list[Citation] = []
    for c in gen_citations:
        raw_ids = c.get("chunk_ids", [])
        ids = list(raw_ids) if isinstance(raw_ids, list) else []
        result.append(Citation(
            title=str(c.get("title", "")),
            source_uri=str(c.get("source_uri", "")),
            chunk_ids=[str(cid) for cid in ids],
        ))
    return result


def _compute_confidence(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    return chunks[0].score


def run_chat(
    query: str,
    retriever: HybridRetriever,
    llm: LLMClient,
    settings: Settings,
    *,
    history: list[Message] | None = None,
    conversation_id: str | None = None,
) -> OrchestratorResult:
    start = time.monotonic()
    request_id = str(uuid.uuid4())
    effective_cid = conversation_id or str(uuid.uuid4())

    chunks = retriever.retrieve(
        query,
        k=settings.retrieval.k,
        history=history,
    )

    confidence = _compute_confidence(chunks)

    if confidence < settings.retrieval.low_confidence_floor or not chunks:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        _log_request(
            request_id, effective_cid, query,
            AnswerStatus.ABSTAINED, elapsed_ms, UsageInfo(),
        )
        return OrchestratorResult(
            answer="I don't have enough information to answer that. "
                   "Let me connect you with a human agent.",
            citations=[],
            status=AnswerStatus.ABSTAINED,
            grounding_score=0.0,
            confidence=confidence,
            needs_human=True,
            usage=UsageInfo(),
            request_id=request_id,
            latency_ms=elapsed_ms,
        )

    gen_result = generate_answer(
        query, chunks, llm, settings.thresholds,
        history=history,
        company_name=settings.app_name,
    )

    citations = _build_citations(gen_result.citations)
    grounding_score = 1.0 if gen_result.used_sources else 0.0

    usage = UsageInfo(
        prompt_tokens=gen_result.prompt_tokens,
        completion_tokens=gen_result.completion_tokens,
        cost_usd=gen_result.cost_usd,
    )

    status = (
        AnswerStatus.ANSWERED if gen_result.used_sources
        else AnswerStatus.ABSTAINED
    )
    needs_human = status != AnswerStatus.ANSWERED

    elapsed_ms = int((time.monotonic() - start) * 1000)

    _log_request(
        request_id, effective_cid, query,
        status, elapsed_ms, usage,
    )

    return OrchestratorResult(
        answer=gen_result.answer,
        citations=citations,
        status=status,
        grounding_score=grounding_score,
        confidence=confidence,
        needs_human=needs_human,
        usage=usage,
        request_id=request_id,
        latency_ms=elapsed_ms,
    )


def _log_request(
    request_id: str,
    conversation_id: str,
    query: str,
    status: AnswerStatus,
    latency_ms: int,
    usage: UsageInfo,
) -> None:
    logger.info(
        "chat.request",
        extra={
            "request_id": request_id,
            "conversation_id": conversation_id,
            "query_redacted": redact_pii(query),
            "status": status.value,
            "latency_ms": latency_ms,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cost_usd": usage.cost_usd,
        },
    )
