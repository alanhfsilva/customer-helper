from __future__ import annotations

from app.models import RetrievedChunk

RRF_K = 60


def reciprocal_rank_fusion(
    *result_lists: list[RetrievedChunk],
    k: int = 5,
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results):
            rrf_score = 1.0 / (RRF_K + rank + 1)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + rrf_score
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        RetrievedChunk(
            chunk_id=chunk_map[cid].chunk_id,
            document_id=chunk_map[cid].document_id,
            text=chunk_map[cid].text,
            score=score,
            source_uri=chunk_map[cid].source_uri,
            title=chunk_map[cid].title,
            heading_path=chunk_map[cid].heading_path,
        )
        for cid, score in ranked[:k]
    ]
