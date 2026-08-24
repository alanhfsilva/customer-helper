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


class FeedbackSignal(StrEnum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    AGENT_EDIT = "agent_edit"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=1)
    signal: FeedbackSignal
    comment: str = ""
    corrected_answer: str = ""
    agent_id: str = ""
    tags: list[str] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    request_id: str
    status: str = "recorded"


class MetricsResponse(BaseModel):
    total_requests: int
    answered: int
    abstained: int
    escalated: int
    blocked: int
    total_cost_usd: float
    total_latency_ms: int
    deflection_rate: float
    escalation_rate: float
    avg_latency_ms: float
    avg_cost_usd: float
