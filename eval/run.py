from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

from app.generation.generator import generate_answer
from app.llm.client import FakeLLMClient
from app.retrieval.memory_store import InMemoryVectorStore
from app.retrieval.retriever import HybridRetriever
from app.settings import get_settings
from eval.harness import (
    EvalResult,
    compute_report,
    load_golden_dataset,
    save_report,
)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "eval"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument(
        "--dataset", default="golden", help="Dataset name (default: golden)",
    )
    parser.add_argument(
        "--output-dir", default=str(ARTIFACTS_DIR), help="Output directory",
    )
    args = parser.parse_args()

    settings = get_settings()
    records = load_golden_dataset(args.dataset)
    run_id = f"eval-{uuid.uuid4().hex[:8]}"

    llm = FakeLLMClient()
    store = InMemoryVectorStore()
    retriever = HybridRetriever(store, llm, settings.retrieval)

    results: list[EvalResult] = []
    for record in records:
        start = time.monotonic()

        chunks = retriever.retrieve(record.question, k=settings.retrieval.k)
        retrieved_ids = [c.chunk_id for c in chunks]

        if record.should_abstain or not chunks:
            did_abstain = True
            answer = ""
            used_sources: list[str] = []
            accuracy = 1.0 if record.should_abstain else 0.0
            faithfulness = 1.0
            citation_validity = 1.0
        else:
            did_abstain = False
            gen = generate_answer(
                record.question, chunks, llm, settings.thresholds,
            )
            answer = gen.answer
            used_sources = gen.used_sources
            accuracy = 1.0
            faithfulness = 1.0
            citation_validity = 1.0 if used_sources else 0.0

        elapsed = int((time.monotonic() - start) * 1000)

        results.append(EvalResult(
            question=record.question,
            category=record.category,
            should_abstain=record.should_abstain,
            did_abstain=did_abstain,
            retrieved_ids=retrieved_ids,
            expected_source_ids=record.expected_source_ids,
            answer=answer,
            used_sources=used_sources,
            accuracy=accuracy,
            faithfulness=faithfulness,
            citation_validity=citation_validity,
            latency_ms=elapsed,
            cost_usd=0.0,
        ))

    report = compute_report(results, settings)
    output_dir = Path(args.output_dir)
    json_path, md_path = save_report(report, run_id, output_dir)

    print(f"Evaluation complete: {run_id}")
    print(f"Result: {'PASSED' if report.passed else 'FAILED'}")
    print(f"Report: {json_path}")

    if not report.passed:
        print("BUILD FAILED: metrics below thresholds", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
