from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(..., min_length=1, max_length=4000)
    history: list[Turn] | None = None
    metadata: dict[str, str] | None = None


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    ESCALATED = "escalated"
    BLOCKED = "blocked"


class Citation(BaseModel):
    title: str
    source_uri: str
    chunk_ids: list[str]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    status: AnswerStatus
    grounding_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human: bool
    usage: UsageInfo = Field(default_factory=UsageInfo)
    request_id: str
    latency_ms: int
