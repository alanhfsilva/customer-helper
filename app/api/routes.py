from __future__ import annotations

import time
import uuid

from fastapi import APIRouter

from app.api.models import AnswerStatus, ChatRequest, ChatResponse, UsageInfo

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    start = time.monotonic()
    request_id = str(uuid.uuid4())
    conversation_id = request.conversation_id or str(uuid.uuid4())

    elapsed_ms = int((time.monotonic() - start) * 1000)

    return ChatResponse(
        conversation_id=conversation_id,
        answer="This endpoint is not yet implemented. Please check back later.",
        citations=[],
        status=AnswerStatus.ABSTAINED,
        grounding_score=0.0,
        confidence=0.0,
        needs_human=True,
        usage=UsageInfo(),
        request_id=request_id,
        latency_ms=elapsed_ms,
    )
