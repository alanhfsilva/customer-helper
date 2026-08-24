from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from app.llm.client import FakeLLMClient
from app.retrieval.memory_store import InMemoryVectorStore
from app.settings import IngestionConfig
from ingestion.chunker import RawChunk, chunk_document, estimate_tokens
from ingestion.connector import MarkdownConnector
from ingestion.normalizer import normalize_content
from ingestion.pipeline import run_ingestion

if TYPE_CHECKING:
    from ingestion.report import IngestionReport

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"


class TestEstimateTokens:
    def test_short_text(self) -> None:
        assert estimate_tokens("hello world") >= 2

    def test_empty(self) -> None:
        assert estimate_tokens("") >= 1


class TestNormalizer:
    def test_strips_html_comments(self) -> None:
        text = "Hello <!-- hidden --> world"
        assert "hidden" not in normalize_content(text)

    def test_strips_nav(self) -> None:
        text = "Before <nav>menu items</nav> After"
        result = normalize_content(text)
        assert "menu" not in result
        assert "Before" in result

    def test_collapses_blank_lines(self) -> None:
        text = "a\n\n\n\n\nb"
        result = normalize_content(text)
        assert "\n\n\n" not in result


class TestChunker:
    def _config(self, target: int = 50, max_t: int = 80) -> IngestionConfig:
        return IngestionConfig(
            chunk_target_tokens=target,
            chunk_max_tokens=max_t,
            chunk_overlap_fraction=0.15,
        )

    def test_produces_chunks(self) -> None:
        content = "# Title\n\nSome content here."
        chunks = chunk_document(content, self._config())
        assert len(chunks) >= 1
        assert isinstance(chunks[0], RawChunk)

    def test_heading_path_extracted(self) -> None:
        content = "# Top\n## Sub\nContent under sub."
        chunks = chunk_document(content, self._config())
        assert any(c.heading_path == ["Top", "Sub"] for c in chunks)

    def test_anchor_generated(self) -> None:
        content = "# Billing\n## Payment Methods\nSome text."
        chunks = chunk_document(content, self._config())
        assert any(c.anchor == "payment-methods" for c in chunks)

    def test_long_section_splits(self) -> None:
        long_text = "# Title\n\n" + "\n\n".join(
            [f"Paragraph {i} with enough words to count." for i in range(30)]
        )
        chunks = chunk_document(long_text, self._config(target=30, max_t=40))
        assert len(chunks) > 1


class TestMarkdownConnector:
    def test_iterates_documents(self) -> None:
        connector = MarkdownConnector(FIXTURES)
        docs = list(connector.iter_documents())
        assert len(docs) == 2

    def test_extracts_title(self) -> None:
        connector = MarkdownConnector(FIXTURES)
        docs = {d.title: d for d in connector.iter_documents()}
        assert "Billing" in docs
        assert "Shipping" in docs

    def test_computes_content_hash(self) -> None:
        connector = MarkdownConnector(FIXTURES)
        docs = list(connector.iter_documents())
        assert all(len(d.content_hash) == 64 for d in docs)

    def test_stable_ids(self) -> None:
        c1 = MarkdownConnector(FIXTURES)
        c2 = MarkdownConnector(FIXTURES)
        ids1 = [d.id for d in c1.iter_documents()]
        ids2 = [d.id for d in c2.iter_documents()]
        assert ids1 == ids2


class TestPipeline:
    def _run(
        self,
        store: InMemoryVectorStore | None = None,
        llm: FakeLLMClient | None = None,
        report_dir: str | None = None,
    ) -> tuple[InMemoryVectorStore, FakeLLMClient, IngestionReport]:
        s = store or InMemoryVectorStore()
        client = llm or FakeLLMClient(embeddings_dim=8)
        rd = report_dir or str(
            Path(__file__).parent / "tmp_reports"
        )
        config = IngestionConfig(
            chunk_target_tokens=50,
            chunk_max_tokens=80,
            chunk_overlap_fraction=0.15,
        )
        connector = MarkdownConnector(FIXTURES)
        report = run_ingestion(
            connector=connector,
            llm=client,
            store=s,
            config=config,
            report_dir=rd,
        )
        return s, client, report

    def test_ingests_all_documents(self) -> None:
        store, _, report = self._run()
        assert report.documents_processed == 2
        assert report.chunks_created > 0
        assert store.list_document_ids()

    def test_idempotent_rerun_zero_embeds(self) -> None:
        store = InMemoryVectorStore()
        llm = FakeLLMClient(embeddings_dim=8)

        self._run(store=store, llm=llm)

        llm2 = FakeLLMClient(embeddings_dim=8)
        _, _, report2 = self._run(store=store, llm=llm2)

        assert len(llm2.embed_calls) == 0
        assert report2.documents_skipped == 2
        assert report2.documents_processed == 0

    def test_changed_doc_reembeds_only_changed(self) -> None:
        store = InMemoryVectorStore()
        llm1 = FakeLLMClient(embeddings_dim=8)
        self._run(store=store, llm=llm1)

        docs = list(MarkdownConnector(FIXTURES).iter_documents())
        store.set_document_hash(
            docs[0].id, "modified_hash_to_force_reembed"
        )

        llm2 = FakeLLMClient(embeddings_dim=8)
        _, _, report2 = self._run(store=store, llm=llm2)

        assert report2.documents_processed == 1
        assert report2.documents_skipped == 1
        assert len(llm2.embed_calls) == 1

    def test_report_produced(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            _, _, report = self._run(report_dir=td)
            files = list(Path(td).glob("report_*"))
            assert len(files) == 2
            assert any(f.suffix == ".json" for f in files)
            assert any(f.suffix == ".md" for f in files)

    def test_report_has_cost_info(self) -> None:
        _, _, report = self._run()
        assert report.total_tokens_embedded > 0
        assert report.embedding_cost_usd >= 0.0

    def test_embedding_uses_llm_client(self) -> None:
        _, llm, _ = self._run()
        assert len(llm.embed_calls) > 0
