from __future__ import annotations

import math

from app.models import Chunk
from app.retrieval.memory_store import InMemoryVectorStore, _cosine_similarity


def _make_chunk(
    doc_id: str = "doc1",
    ordinal: int = 0,
    text: str = "chunk text",
    embedding: list[float] | None = None,
    metadata: dict[str, str] | None = None,
    heading_path: list[str] | None = None,
) -> Chunk:
    return Chunk(
        id=f"{doc_id}:{ordinal}",
        document_id=doc_id,
        ordinal=ordinal,
        text=text,
        heading_path=heading_path or [],
        token_count=len(text.split()),
        embedding=embedding or [0.0] * 4,
        metadata=metadata or {},
    )


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine_similarity(a, b)) < 1e-9

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-9

    def test_zero_vector(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


class TestUpsert:
    def test_upsert_single_chunk(self) -> None:
        store = InMemoryVectorStore()
        chunk = _make_chunk()
        store.upsert([chunk])
        ids = store.get_document_chunk_ids("doc1")
        assert ids == ["doc1:0"]

    def test_upsert_multiple_chunks(self) -> None:
        store = InMemoryVectorStore()
        chunks = [_make_chunk(ordinal=i) for i in range(3)]
        store.upsert(chunks)
        ids = store.get_document_chunk_ids("doc1")
        assert len(ids) == 3

    def test_upsert_overwrites_existing(self) -> None:
        store = InMemoryVectorStore()
        store.upsert([_make_chunk(text="old")])
        store.upsert([_make_chunk(text="new")])
        results = store.search([1.0, 0.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        assert results[0].text == "new"


class TestSearch:
    def test_knn_returns_nearest(self) -> None:
        store = InMemoryVectorStore()
        store.upsert([
            _make_chunk(ordinal=0, text="close", embedding=[1.0, 0.0, 0.0, 0.0]),
            _make_chunk(ordinal=1, text="far", embedding=[0.0, 0.0, 0.0, 1.0]),
            _make_chunk(ordinal=2, text="medium", embedding=[0.7, 0.7, 0.0, 0.0]),
        ])
        results = store.search([1.0, 0.0, 0.0, 0.0], k=2)
        assert len(results) == 2
        assert results[0].text == "close"
        assert results[0].score == 1.0

    def test_knn_respects_k(self) -> None:
        store = InMemoryVectorStore()
        chunks = [
            _make_chunk(ordinal=i, embedding=[float(i), 1.0, 0.0, 0.0])
            for i in range(5)
        ]
        store.upsert(chunks)
        results = store.search([1.0, 0.0, 0.0, 0.0], k=3)
        assert len(results) == 3

    def test_knn_scores_descending(self) -> None:
        store = InMemoryVectorStore()
        store.upsert([
            _make_chunk(ordinal=0, embedding=[1.0, 0.0, 0.0, 0.0]),
            _make_chunk(ordinal=1, embedding=[0.5, 0.5, 0.0, 0.0]),
            _make_chunk(ordinal=2, embedding=[0.0, 1.0, 0.0, 0.0]),
        ])
        results = store.search([1.0, 0.0, 0.0, 0.0], k=3)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_search_empty_store(self) -> None:
        store = InMemoryVectorStore()
        results = store.search([1.0, 0.0, 0.0, 0.0], k=5)
        assert results == []

    def test_search_populates_retrieved_chunk_fields(self) -> None:
        store = InMemoryVectorStore()
        store.upsert([
            _make_chunk(
                embedding=[1.0, 0.0, 0.0, 0.0],
                metadata={"source_uri": "https://help.example.com/billing", "title": "Billing FAQ"},
                heading_path=["Billing", "Refunds"],
            )
        ])
        results = store.search([1.0, 0.0, 0.0, 0.0], k=1)
        r = results[0]
        assert r.chunk_id == "doc1:0"
        assert r.document_id == "doc1"
        assert r.source_uri == "https://help.example.com/billing"
        assert r.title == "Billing FAQ"
        assert r.heading_path == ["Billing", "Refunds"]
        assert r.score == 1.0


class TestFilters:
    def test_filter_by_metadata(self) -> None:
        store = InMemoryVectorStore()
        emb_0 = [1.0, 0.0, 0.0, 0.0]
        emb_1 = [0.9, 0.1, 0.0, 0.0]
        emb_2 = [0.8, 0.2, 0.0, 0.0]
        store.upsert([
            _make_chunk(ordinal=0, embedding=emb_0, metadata={"source_type": "help_article"}),
            _make_chunk(ordinal=1, embedding=emb_1, metadata={"source_type": "policy"}),
            _make_chunk(ordinal=2, embedding=emb_2, metadata={"source_type": "help_article"}),
        ])
        results = store.search(
            [1.0, 0.0, 0.0, 0.0], k=5, filters={"source_type": "help_article"}
        )
        assert len(results) == 2
        assert all(r.chunk_id.endswith(("0", "2")) for r in results)

    def test_filter_excludes_non_matching(self) -> None:
        store = InMemoryVectorStore()
        store.upsert([
            _make_chunk(embedding=[1.0, 0.0, 0.0, 0.0], metadata={"visibility": "internal"}),
        ])
        results = store.search(
            [1.0, 0.0, 0.0, 0.0], k=5, filters={"visibility": "external"}
        )
        assert results == []


class TestDeleteByDocument:
    def test_delete_removes_all_chunks(self) -> None:
        store = InMemoryVectorStore()
        store.upsert([_make_chunk(ordinal=i) for i in range(3)])
        deleted = store.delete_by_document("doc1")
        assert deleted == 3
        assert store.get_document_chunk_ids("doc1") == []

    def test_delete_only_target_document(self) -> None:
        store = InMemoryVectorStore()
        store.upsert([_make_chunk(doc_id="doc1", ordinal=0)])
        store.upsert([_make_chunk(doc_id="doc2", ordinal=0)])
        store.delete_by_document("doc1")
        assert store.get_document_chunk_ids("doc1") == []
        assert store.get_document_chunk_ids("doc2") == ["doc2:0"]

    def test_delete_nonexistent_returns_zero(self) -> None:
        store = InMemoryVectorStore()
        assert store.delete_by_document("nonexistent") == 0


class TestRoundTrip:
    def test_upsert_then_knn_round_trip(self) -> None:
        store = InMemoryVectorStore()

        dim = 8
        billing_embedding = [1.0] + [0.0] * (dim - 1)
        shipping_embedding = [0.0, 1.0] + [0.0] * (dim - 2)
        returns_embedding = [0.7, 0.3] + [0.0] * (dim - 2)

        billing_meta = {
            "source_uri": "/billing",
            "title": "Billing Help",
            "source_type": "help_article",
        }
        shipping_meta = {
            "source_uri": "/shipping",
            "title": "Shipping Help",
            "source_type": "help_article",
        }
        returns_meta = {
            "source_uri": "/returns",
            "title": "Returns Help",
            "source_type": "policy",
        }

        store.upsert([
            Chunk(
                id="billing-doc:0",
                document_id="billing-doc",
                ordinal=0,
                text="How to update your billing information",
                heading_path=["Billing"],
                token_count=7,
                embedding=billing_embedding,
                metadata=billing_meta,
            ),
            Chunk(
                id="shipping-doc:0",
                document_id="shipping-doc",
                ordinal=0,
                text="Track your shipment status",
                heading_path=["Shipping"],
                token_count=5,
                embedding=shipping_embedding,
                metadata=shipping_meta,
            ),
            Chunk(
                id="returns-doc:0",
                document_id="returns-doc",
                ordinal=0,
                text="How to return an item",
                heading_path=["Returns"],
                token_count=5,
                embedding=returns_embedding,
                metadata=returns_meta,
            ),
        ])

        query_embedding = [0.9, 0.1] + [0.0] * (dim - 2)
        results = store.search(query_embedding, k=2)

        assert len(results) == 2
        assert results[0].chunk_id == "billing-doc:0"
        assert results[0].title == "Billing Help"
        assert results[0].source_uri == "/billing"
        assert results[0].score > results[1].score

    def test_update_then_search_reflects_change(self) -> None:
        store = InMemoryVectorStore()

        store.upsert([_make_chunk(text="version 1", embedding=[1.0, 0.0, 0.0, 0.0])])
        store.delete_by_document("doc1")
        store.upsert([_make_chunk(text="version 2", embedding=[0.0, 1.0, 0.0, 0.0])])

        results = store.search([0.0, 1.0, 0.0, 0.0], k=1)
        assert results[0].text == "version 2"
        assert math.isclose(results[0].score, 1.0)
