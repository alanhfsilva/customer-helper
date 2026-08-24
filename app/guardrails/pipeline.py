from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.llm.client import LLMClient
    from app.models import RetrievedChunk
    from app.settings import Settings

SENSITIVE_TOPICS = frozenset({
    "billing dispute", "legal", "lawsuit", "attorney",
    "security breach", "hack", "unauthorized access",
    "refund dispute", "chargeback",
})

ESCALATION_PHRASES = frozenset({
    "speak to a human", "talk to a person", "human agent",
    "real person", "speak to someone", "transfer me",
})

PII_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    r"|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
    r"|\b\d{3}-\d{2}-\d{4}\b"
    r"|\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
)


class GuardrailAction(StrEnum):
    PASS = "pass"
    MODIFY = "modify"
    BLOCK = "block"


@dataclass(frozen=True)
class GuardrailResult:
    action: GuardrailAction
    reason: str
    modified_answer: str | None = None
    grounding_score: float = 1.0
    needs_human: bool = False


def check_input_moderation(
    message: str,
    llm: LLMClient,
) -> GuardrailResult:
    result = llm.moderate(message)
    if result.flagged:
        flagged_cats = [
            cat for cat, flagged in result.categories.items() if flagged
        ]
        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            reason=f"Input flagged by moderation: {', '.join(flagged_cats)}",
        )
    return GuardrailResult(action=GuardrailAction.PASS, reason="Input passed moderation")


def check_retrieval_confidence(
    chunks: list[RetrievedChunk],
    low_confidence_floor: float,
) -> GuardrailResult:
    if not chunks:
        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            reason="No retrieval results",
            needs_human=True,
        )
    top_score = chunks[0].score
    if top_score < low_confidence_floor:
        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            reason=f"Top score {top_score:.2f} below floor {low_confidence_floor}",
            needs_human=True,
        )
    return GuardrailResult(action=GuardrailAction.PASS, reason="Retrieval confidence OK")


def check_grounding(
    answer: str,
    used_sources: list[str],
    chunks: list[RetrievedChunk],
) -> GuardrailResult:
    if not used_sources:
        return GuardrailResult(
            action=GuardrailAction.MODIFY,
            reason="No sources cited; answer may be ungrounded",
            grounding_score=0.0,
            needs_human=True,
        )

    chunk_ids = {c.chunk_id for c in chunks}
    valid_sources = [s for s in used_sources if s in chunk_ids]
    score = len(valid_sources) / len(used_sources) if used_sources else 0.0

    if score < 0.5:
        return GuardrailResult(
            action=GuardrailAction.MODIFY,
            reason=f"Grounding score {score:.2f} below 0.5",
            grounding_score=score,
            needs_human=True,
        )

    return GuardrailResult(
        action=GuardrailAction.PASS,
        reason="Answer is grounded",
        grounding_score=score,
    )


def check_output_moderation(
    answer: str,
    llm: LLMClient,
) -> GuardrailResult:
    result = llm.moderate(answer)
    if result.flagged:
        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            reason="Generated answer flagged by moderation",
        )
    return GuardrailResult(
        action=GuardrailAction.PASS,
        reason="Output passed moderation",
    )


def check_pii_in_answer(answer: str) -> GuardrailResult:
    if PII_PATTERN.search(answer):
        cleaned = PII_PATTERN.sub("[REDACTED]", answer)
        return GuardrailResult(
            action=GuardrailAction.MODIFY,
            reason="PII detected and redacted from answer",
            modified_answer=cleaned,
        )
    return GuardrailResult(action=GuardrailAction.PASS, reason="No PII in answer")


def check_escalation(
    message: str,
    confidence: float,
    low_confidence_floor: float,
) -> GuardrailResult:
    lower_msg = message.lower()

    for phrase in ESCALATION_PHRASES:
        if phrase in lower_msg:
            return GuardrailResult(
                action=GuardrailAction.MODIFY,
                reason=f"User requested human: '{phrase}'",
                needs_human=True,
            )

    for topic in SENSITIVE_TOPICS:
        if topic in lower_msg:
            return GuardrailResult(
                action=GuardrailAction.MODIFY,
                reason=f"Sensitive topic detected: '{topic}'",
                needs_human=True,
            )

    if confidence < low_confidence_floor:
        return GuardrailResult(
            action=GuardrailAction.MODIFY,
            reason=f"Low confidence {confidence:.2f}",
            needs_human=True,
        )

    return GuardrailResult(action=GuardrailAction.PASS, reason="No escalation needed")


@dataclass(frozen=True)
class PipelineResult:
    answer: str
    blocked: bool
    needs_human: bool
    grounding_score: float
    block_reason: str
    stages: list[GuardrailResult]


def run_guardrails(
    message: str,
    answer: str,
    used_sources: list[str],
    chunks: list[RetrievedChunk],
    confidence: float,
    llm: LLMClient,
    settings: Settings,
) -> PipelineResult:
    stages: list[GuardrailResult] = []
    current_answer = answer
    needs_human = False
    grounding_score = 1.0

    input_mod = check_input_moderation(message, llm)
    stages.append(input_mod)
    if input_mod.action == GuardrailAction.BLOCK:
        return PipelineResult(
            answer="", blocked=True, needs_human=False,
            grounding_score=0.0,
            block_reason=input_mod.reason, stages=stages,
        )

    retrieval_gate = check_retrieval_confidence(
        chunks, settings.retrieval.low_confidence_floor,
    )
    stages.append(retrieval_gate)
    if retrieval_gate.action == GuardrailAction.BLOCK:
        return PipelineResult(
            answer=current_answer, blocked=True,
            needs_human=True, grounding_score=0.0,
            block_reason=retrieval_gate.reason, stages=stages,
        )

    grounding = check_grounding(current_answer, used_sources, chunks)
    stages.append(grounding)
    grounding_score = grounding.grounding_score
    if grounding.needs_human:
        needs_human = True
    if grounding.action == GuardrailAction.MODIFY and grounding.modified_answer:
        current_answer = grounding.modified_answer

    output_mod = check_output_moderation(current_answer, llm)
    stages.append(output_mod)
    if output_mod.action == GuardrailAction.BLOCK:
        return PipelineResult(
            answer="", blocked=True, needs_human=False,
            grounding_score=grounding_score,
            block_reason=output_mod.reason, stages=stages,
        )

    pii_check = check_pii_in_answer(current_answer)
    stages.append(pii_check)
    if pii_check.modified_answer:
        current_answer = pii_check.modified_answer

    escalation = check_escalation(
        message, confidence, settings.retrieval.low_confidence_floor,
    )
    stages.append(escalation)
    if escalation.needs_human:
        needs_human = True

    return PipelineResult(
        answer=current_answer,
        blocked=False,
        needs_human=needs_human,
        grounding_score=grounding_score,
        block_reason="",
        stages=stages,
    )
