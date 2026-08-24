from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from app.models import Chunk, RetrievedChunk

STOP_WORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "of", "to", "and", "or",
    "for", "on", "with", "at", "by", "from", "that", "this", "was",
    "are", "be", "have", "has", "had", "do", "does", "did", "will",
    "can", "could", "would", "should", "not", "but", "if", "so",
    "my", "your", "our", "their", "i", "you", "we", "they", "he",
    "she", "what", "how", "when", "where", "which", "who",
})


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"\w+", text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


@dataclass
class _DocEntry:
    chunk: Chunk
    terms: list[str]
    term_freq: dict[str, int]


class KeywordIndex:
    def __init__(self) -> None:
        self._docs: dict[str, _DocEntry] = {}
        self._doc_freq: dict[str, int] = {}
        self._total_docs = 0
        self._avg_dl = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            terms = _tokenize(chunk.text)
            tf: dict[str, int] = {}
            for t in terms:
                tf[t] = tf.get(t, 0) + 1

            if chunk.id in self._docs:
                old_entry = self._docs[chunk.id]
                for t in set(old_entry.terms):
                    self._doc_freq[t] = max(0, self._doc_freq.get(t, 1) - 1)
                self._total_docs -= 1

            self._docs[chunk.id] = _DocEntry(chunk=chunk, terms=terms, term_freq=tf)
            for t in set(terms):
                self._doc_freq[t] = self._doc_freq.get(t, 0) + 1
            self._total_docs += 1

        if self._total_docs > 0:
            self._avg_dl = sum(len(d.terms) for d in self._docs.values()) / self._total_docs

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        scored: list[tuple[float, _DocEntry]] = []
        k1 = 1.5
        b = 0.75

        for entry in self._docs.values():
            if filters and not _matches_filters(entry.chunk, filters):
                continue

            score = 0.0
            dl = len(entry.terms)

            for qt in query_terms:
                df = self._doc_freq.get(qt, 0)
                if df == 0:
                    continue
                idf = math.log(
                    (self._total_docs - df + 0.5) / (df + 0.5) + 1.0
                )
                tf = entry.term_freq.get(qt, 0)
                denom = tf + k1 * (1 - b + b * dl / max(self._avg_dl, 1))
                score += idf * (tf * (k1 + 1)) / denom

            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            RetrievedChunk(
                chunk_id=e.chunk.id,
                document_id=e.chunk.document_id,
                text=e.chunk.text,
                score=s,
                source_uri=e.chunk.metadata.get("source_uri", ""),
                title=e.chunk.metadata.get("title", ""),
                heading_path=e.chunk.heading_path,
            )
            for s, e in scored[:k]
        ]

    def remove_by_document(self, document_id: str) -> None:
        to_remove = [
            cid for cid, e in self._docs.items()
            if e.chunk.document_id == document_id
        ]
        for cid in to_remove:
            entry = self._docs.pop(cid)
            for t in set(entry.terms):
                self._doc_freq[t] = max(0, self._doc_freq.get(t, 1) - 1)
            self._total_docs -= 1

        if self._total_docs > 0:
            self._avg_dl = (
                sum(len(d.terms) for d in self._docs.values()) / self._total_docs
            )


def _matches_filters(chunk: Chunk, filters: dict[str, Any]) -> bool:
    return all(
        chunk.metadata.get(key) == value for key, value in filters.items()
    )
