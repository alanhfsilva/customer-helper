from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path


class FeedbackSignal(StrEnum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    AGENT_EDIT = "agent_edit"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


@dataclass(frozen=True)
class FeedbackRecord:
    request_id: str
    signal: FeedbackSignal
    comment: str = ""
    corrected_answer: str = ""
    agent_id: str = ""
    tags: tuple[str, ...] = ()
    created_at: str = ""


class FeedbackStore(Protocol):
    def save(self, record: FeedbackRecord) -> None: ...

    def get_by_request_id(self, request_id: str) -> list[FeedbackRecord]: ...

    def list_all(
        self,
        *,
        signal: FeedbackSignal | None = None,
        limit: int = 100,
    ) -> list[FeedbackRecord]: ...


class InMemoryFeedbackStore:
    def __init__(self) -> None:
        self._records: list[FeedbackRecord] = []

    def save(self, record: FeedbackRecord) -> None:
        self._records = [*self._records, record]

    def get_by_request_id(self, request_id: str) -> list[FeedbackRecord]:
        return [r for r in self._records if r.request_id == request_id]

    def list_all(
        self,
        *,
        signal: FeedbackSignal | None = None,
        limit: int = 100,
    ) -> list[FeedbackRecord]:
        filtered = self._records
        if signal is not None:
            filtered = [r for r in filtered if r.signal == signal]
        return filtered[:limit]

    @property
    def count(self) -> int:
        return len(self._records)


def export_feedback_to_golden(
    records: list[FeedbackRecord],
    output_path: Path,
) -> int:
    exported = 0
    entries: list[dict[str, object]] = []
    for r in records:
        if r.signal not in (FeedbackSignal.THUMBS_UP, FeedbackSignal.AGENT_EDIT):
            continue
        answer = r.corrected_answer if r.corrected_answer else ""
        if not answer:
            continue
        entry: dict[str, object] = {
            "question": "",
            "reference_answer": answer,
            "expected_source_ids": [],
            "category": "feedback",
            "should_abstain": False,
            "source_request_id": r.request_id,
        }
        entries.append(entry)
        exported += 1

    with open(output_path, "a") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return exported
