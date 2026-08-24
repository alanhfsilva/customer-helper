from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Header, HTTPException

from app.api.dependencies import (
    get_feedback_store,
    get_llm_client,
    get_metrics_collector,
    get_rate_limiter,
    get_retriever,
    get_settings,
)
from app.api.models import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    MetricsResponse,
)
from app.feedback.store import FeedbackRecord
from app.feedback.store import FeedbackSignal as StoreFeedbackSignal
from app.llm.models import Message
from app.orchestrator import run_chat

router = APIRouter()


def _verify_api_key(
    x_api_key: str | None = Header(default=None),
) -> str:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    return x_api_key


def _check_rate_limit(
    api_key: str = Depends(_verify_api_key),
) -> str:
    limiter = get_rate_limiter()
    if not limiter.is_allowed(api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return api_key


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    _api_key: str = Depends(_check_rate_limit),
) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid.uuid4())

    settings = get_settings()
    llm = get_llm_client()
    retriever = get_retriever()
    metrics = get_metrics_collector()

    history: list[Message] | None = None
    if request.history:
        history = [
            Message(role=t.role, content=t.content) for t in request.history
        ]

    result = run_chat(
        request.message,
        retriever,
        llm,
        settings,
        history=history,
        conversation_id=conversation_id,
    )

    metrics.record(
        status=result.status.value,
        latency_ms=result.latency_ms,
        cost_usd=result.usage.cost_usd,
    )

    return ChatResponse(
        conversation_id=conversation_id,
        answer=result.answer,
        citations=result.citations,
        status=result.status,
        grounding_score=result.grounding_score,
        confidence=result.confidence,
        needs_human=result.needs_human,
        usage=result.usage,
        request_id=result.request_id,
        latency_ms=result.latency_ms,
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(
    request: FeedbackRequest,
    _api_key: str = Depends(_check_rate_limit),
) -> FeedbackResponse:
    store = get_feedback_store()

    record = FeedbackRecord(
        request_id=request.request_id,
        signal=StoreFeedbackSignal(request.signal.value),
        comment=request.comment,
        corrected_answer=request.corrected_answer,
        agent_id=request.agent_id,
        tags=tuple(request.tags),
        created_at=datetime.now(tz=UTC).isoformat(),
    )
    store.save(record)

    return FeedbackResponse(request_id=request.request_id)


@router.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    collector = get_metrics_collector()
    snap = collector.snapshot()

    return MetricsResponse(
        total_requests=snap.total_requests,
        answered=snap.answered,
        abstained=snap.abstained,
        escalated=snap.escalated,
        blocked=snap.blocked,
        total_cost_usd=snap.total_cost_usd,
        total_latency_ms=snap.total_latency_ms,
        deflection_rate=snap.deflection_rate,
        escalation_rate=snap.escalation_rate,
        avg_latency_ms=snap.avg_latency_ms,
        avg_cost_usd=snap.avg_cost_usd,
    )
