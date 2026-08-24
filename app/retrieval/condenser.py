from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.llm.client import LLMClient
    from app.llm.models import Message


PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "prompts" / "query_condenser.md"
)


def _load_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def condense_query(
    message: str,
    history: list[Message] | None,
    llm: LLMClient,
) -> str:
    if not history:
        return message

    template = _load_template()
    history_text = "\n".join(
        f"{m.role}: {m.content}" for m in history
    )
    prompt = template.replace("{{history}}", history_text).replace(
        "{{message}}", message
    )

    from app.llm.models import Message as Msg

    result = llm.chat(
        [Msg(role="user", content=prompt)],
        temperature=0.0,
        max_tokens=200,
    )
    return result.content.strip()
