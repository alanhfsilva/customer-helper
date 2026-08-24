from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class ChatResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    request_id: str
    model: str


@dataclass(frozen=True)
class EmbeddingUsage:
    prompt_tokens: int
    cost_usd: float
    request_id: str


@dataclass(frozen=True)
class EmbeddingResult:
    embeddings: list[list[float]]
    usage: EmbeddingUsage


@dataclass(frozen=True)
class ModerationResult:
    flagged: bool
    categories: dict[str, bool] = field(default_factory=dict)
    category_scores: dict[str, float] = field(default_factory=dict)
