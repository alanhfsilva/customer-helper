from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.generation.prompt import render_system_prompt
from app.llm.models import Message

if TYPE_CHECKING:
    from app.llm.client import LLMClient
    from app.models import RetrievedChunk
    from app.settings import ThresholdsConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    used_sources: list[str]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    model: str
    request_id: str
    raw_content: str = ""
    citations: list[dict[str, object]] = field(default_factory=list)


def _parse_structured_response(content: str) -> tuple[str, list[str]]:
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        answer = data.get("answer", cleaned)
        used_sources = data.get("used_sources", [])
        return str(answer), [str(s) for s in used_sources]
    except (json.JSONDecodeError, AttributeError):
        return content.strip(), []


def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    llm: LLMClient,
    thresholds: ThresholdsConfig,
    *,
    history: list[Message] | None = None,
    company_name: str = "our company",
) -> GenerationResult:
    system_prompt = render_system_prompt(
        chunks,
        token_budget=thresholds.context_token_budget,
        company_name=company_name,
    )

    messages: list[Message] = [Message(role="system", content=system_prompt)]

    if history:
        messages = [*messages, *history]

    messages = [*messages, Message(role="user", content=query)]

    result = llm.chat(
        messages,
        temperature=thresholds.generation_temperature,
        max_tokens=thresholds.max_output_tokens,
    )

    answer, used_sources = _parse_structured_response(result.content)

    chunk_map: dict[str, RetrievedChunk] = {c.chunk_id: c for c in chunks}
    citations: list[dict[str, object]] = []
    seen_sources: set[str] = set()
    for source_id in used_sources:
        if source_id in chunk_map and source_id not in seen_sources:
            seen_sources.add(source_id)
            chunk = chunk_map[source_id]
            citations.append({
                "title": chunk.title,
                "source_uri": chunk.source_uri,
                "chunk_ids": [source_id],
            })

    return GenerationResult(
        answer=answer,
        used_sources=used_sources,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=result.cost_usd,
        model=result.model,
        request_id=result.request_id,
        raw_content=result.content,
        citations=citations,
    )
