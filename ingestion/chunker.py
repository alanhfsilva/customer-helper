from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.settings import IngestionConfig


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.33))


@dataclass(frozen=True)
class RawChunk:
    text: str
    heading_path: list[str]
    anchor: str
    token_count: int


def _heading_to_anchor(heading: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", heading.lower())
    return re.sub(r"[\s]+", "-", slug).strip("-")


def _split_into_sections(
    content: str,
) -> list[tuple[list[str], str]]:
    lines = content.split("\n")
    sections: list[tuple[list[str], str]] = []
    current_path: list[str] = []
    current_lines: list[str] = []
    heading_stack: list[tuple[int, str]] = []

    for line in lines:
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            if current_lines:
                text = "\n".join(current_lines).strip()
                if text:
                    sections.append((list(current_path), text))
                current_lines = []

            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_path = [h[1] for h in heading_stack]
        else:
            current_lines.append(line)

    if current_lines:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append((list(current_path), text))

    return sections


def _split_long_text(
    text: str, max_tokens: int, overlap_tokens: int
) -> list[str]:
    paragraphs = re.split(r"\n\n+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)
        if current_tokens + para_tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            overlap_text = _get_overlap_text(current, overlap_tokens)
            current = [overlap_text] if overlap_text else []
            current_tokens = estimate_tokens(overlap_text) if overlap_text else 0

        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _get_overlap_text(paragraphs: list[str], overlap_tokens: int) -> str:
    result: list[str] = []
    tokens = 0
    for para in reversed(paragraphs):
        para_tokens = estimate_tokens(para)
        if tokens + para_tokens > overlap_tokens and result:
            break
        result.insert(0, para)
        tokens += para_tokens
    return "\n\n".join(result) if result else ""


def chunk_document(
    content: str,
    config: IngestionConfig,
) -> list[RawChunk]:
    sections = _split_into_sections(content)
    overlap_tokens = int(config.chunk_target_tokens * config.chunk_overlap_fraction)
    chunks: list[RawChunk] = []

    for heading_path, section_text in sections:
        section_tokens = estimate_tokens(section_text)
        anchor = _heading_to_anchor(heading_path[-1]) if heading_path else ""

        if section_tokens <= config.chunk_max_tokens:
            chunks.append(RawChunk(
                text=section_text,
                heading_path=heading_path,
                anchor=anchor,
                token_count=section_tokens,
            ))
        else:
            sub_texts = _split_long_text(
                section_text, config.chunk_target_tokens, overlap_tokens
            )
            for sub in sub_texts:
                chunks.append(RawChunk(
                    text=sub,
                    heading_path=heading_path,
                    anchor=anchor,
                    token_count=estimate_tokens(sub),
                ))

    return chunks
