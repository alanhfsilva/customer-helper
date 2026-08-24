from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class SourceType(StrEnum):
    HELP_ARTICLE = "help_article"
    DOC = "doc"
    POLICY = "policy"
    TICKET = "ticket"
    FAQ = "faq"


@dataclass(frozen=True)
class Document:
    id: str
    source_uri: str
    title: str
    source_type: SourceType
    content_raw: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    fetched_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    ordinal: int
    text: str
    heading_path: list[str]
    token_count: int
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    score: float
    source_uri: str
    title: str
    heading_path: list[str]
