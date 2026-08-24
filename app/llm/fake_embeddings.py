from __future__ import annotations

import uuid

from app.llm.models import EmbeddingResult, EmbeddingUsage

EMBED_DIM = 256

_VOCAB: dict[str, int] = {}


def _word_index(word: str) -> int:
    if word not in _VOCAB:
        _VOCAB[word] = len(_VOCAB) % EMBED_DIM
    return _VOCAB[word]


def text_to_embedding(text: str) -> list[float]:
    vec = [0.0] * EMBED_DIM
    for w in text.lower().split():
        vec[_word_index(w)] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def bag_of_words_embed(texts: list[str]) -> EmbeddingResult:
    return EmbeddingResult(
        embeddings=[text_to_embedding(t) for t in texts],
        usage=EmbeddingUsage(
            prompt_tokens=sum(len(t.split()) * 2 for t in texts),
            cost_usd=0.0,
            request_id=str(uuid.uuid4()),
        ),
    )
