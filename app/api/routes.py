from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Header, HTTPException

from app.api.dependencies import get_llm_client, get_retriever, get_settings
from app.api.models import ChatRequest, ChatResponse
from app.llm.models import Message
from app.orchestrator import run_chat

router = APIRouter()


def _verify_api_key(
    x_api_key: str | None = Header(default=None),
) -> str:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    return x_api_key


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    _api_key: str = Depends(_verify_api_key),
) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid.uuid4())

    settings = get_settings()
    llm = get_llm_client()
    retriever = get_retriever()

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
