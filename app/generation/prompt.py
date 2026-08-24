from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models import RetrievedChunk

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "prompts"


def load_template(name: str) -> str:
    path = PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")


def assemble_context(
    chunks: list[RetrievedChunk],
    token_budget: int,
) -> str:
    lines: list[str] = []
    tokens_used = 0

    for chunk in chunks:
        estimated_tokens = int(len(chunk.text.split()) * 1.33)
        if tokens_used + estimated_tokens > token_budget and lines:
            break
        source_label = f"[{chunk.chunk_id}] (source: {chunk.source_uri})"
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else ""
        entry = f"--- {source_label}\n"
        if heading:
            entry += f"Section: {heading}\n"
        entry += chunk.text + "\n"
        lines.append(entry)
        tokens_used += estimated_tokens

    return "\n".join(lines)


def render_system_prompt(
    chunks: list[RetrievedChunk],
    token_budget: int,
    company_name: str = "our company",
) -> str:
    template = load_template("system.md")
    context = assemble_context(chunks, token_budget)
    return (
        template
        .replace("{{context}}", context)
        .replace("{{company_name}}", company_name)
    )
