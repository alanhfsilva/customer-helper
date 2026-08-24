from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.settings import get_settings
from eval.harness import (
    EvalResult,
    GoldenRecord,
    compute_report,
    load_golden_dataset,
    save_report,
)


def _result(
    question: str = "test",
    category: str = "general",
    should_abstain: bool = False,
    did_abstain: bool = False,
    retrieved_ids: list[str] | None = None,
    expected_source_ids: list[str] | None = None,
    accuracy: float = 1.0,
    faithfulness: float = 1.0,
    citation_validity: float = 1.0,
) -> EvalResult:
    return EvalResult(
        question=question,
        category=category,
        should_abstain=should_abstain,
        did_abstain=did_abstain,
        retrieved_ids=retrieved_ids or ["c1"],
        expected_source_ids=expected_source_ids or ["c1"],
        answer="test answer",
        used_sources=["c1"],
        accuracy=accuracy,
        faithfulness=faithfulness,
        citation_validity=citation_validity,
        latency_ms=100,
        cost_usd=0.001,
    )


class TestLoadGoldenDataset:
    def test_loads_records(self) -> None:
        records = load_golden_dataset("golden")
        assert len(records) > 0
        assert isinstance(records[0], GoldenRecord)

    def test_record_fields(self) -> None:
        records = load_golden_dataset("golden")
        record = records[0]
        assert record.question != ""
        assert isinstance(record.expected_source_ids, list)
        assert isinstance(record.should_abstain, bool)

    def test_has_in_scope_and_out_of_scope(self) -> None:
        records = load_golden_dataset("golden")
        in_scope = [r for r in records if not r.should_abstain]
        out_scope = [r for r in records if r.should_abstain]
        assert len(in_scope) > 0
        assert len(out_scope) > 0


class TestComputeReport:
    def test_all_perfect_scores_pass(self) -> None:
        settings = get_settings()
        results = [
            _result(accuracy=1.0, faithfulness=1.0),
            _result(accuracy=1.0, faithfulness=1.0),
        ]
        report = compute_report(results, settings)
        assert report.passed is True
        assert report.mean_accuracy == 1.0
        assert report.mean_faithfulness == 1.0

    def test_low_accuracy_fails(self) -> None:
        settings = get_settings()
        results = [
            _result(accuracy=0.3),
            _result(accuracy=0.2),
        ]
        report = compute_report(results, settings)
        assert report.passed is False

    def test_low_faithfulness_fails(self) -> None:
        settings = get_settings()
        results = [
            _result(faithfulness=0.5),
        ]
        report = compute_report(results, settings)
        assert report.passed is False

    def test_recall_computed(self) -> None:
        settings = get_settings()
        results = [
            _result(retrieved_ids=["c1", "c2"], expected_source_ids=["c1"]),
        ]
        report = compute_report(results, settings)
        assert report.recall_at_5 == 1.0

    def test_mrr_computed(self) -> None:
        settings = get_settings()
        results = [
            _result(retrieved_ids=["c1"], expected_source_ids=["c1"]),
        ]
        report = compute_report(results, settings)
        assert report.mrr == 1.0

    def test_refusal_metrics(self) -> None:
        settings = get_settings()
        results = [
            _result(should_abstain=True, did_abstain=True),
            _result(should_abstain=False, did_abstain=False),
        ]
        report = compute_report(results, settings)
        assert report.refusal_precision == 1.0
        assert report.refusal_recall == 1.0

    def test_category_breakdown(self) -> None:
        settings = get_settings()
        results = [
            _result(category="billing", accuracy=0.9),
            _result(category="shipping", accuracy=0.8),
        ]
        report = compute_report(results, settings)
        assert "billing" in report.category_breakdown
        assert "shipping" in report.category_breakdown

    def test_latency_percentiles(self) -> None:
        settings = get_settings()
        results = [_result() for _ in range(10)]
        report = compute_report(results, settings)
        assert report.p50_latency_ms > 0
        assert report.p95_latency_ms > 0

    def test_empty_results_fail(self) -> None:
        settings = get_settings()
        report = compute_report([], settings)
        assert report.passed is False


class TestSaveReport:
    def test_produces_json_and_md(self) -> None:
        settings = get_settings()
        results = [_result()]
        report = compute_report(results, settings)

        with tempfile.TemporaryDirectory() as tmp:
            json_path, md_path = save_report(
                report, "test-run", Path(tmp),
            )
            assert json_path.exists()
            assert md_path.exists()

            data = json.loads(json_path.read_text())
            assert "metrics" in data
            assert data["run_id"] == "test-run"

            md = md_path.read_text()
            assert "Evaluation Report" in md
            assert "Recall@5" in md

    def test_json_has_all_metrics(self) -> None:
        settings = get_settings()
        results = [_result()]
        report = compute_report(results, settings)

        with tempfile.TemporaryDirectory() as tmp:
            json_path, _ = save_report(report, "test-run", Path(tmp))
            data = json.loads(json_path.read_text())
            metrics = data["metrics"]
            assert "recall_at_5" in metrics
            assert "mrr" in metrics
            assert "mean_accuracy" in metrics
            assert "mean_faithfulness" in metrics
            assert "p50_latency_ms" in metrics
            assert "p95_latency_ms" in metrics


class TestCIGate:
    def test_sub_threshold_build_fails(self) -> None:
        settings = get_settings()
        results = [
            _result(accuracy=0.5, faithfulness=0.5),
        ]
        report = compute_report(results, settings)
        assert report.passed is False

    def test_meeting_thresholds_passes(self) -> None:
        settings = get_settings()
        results = [
            _result(accuracy=1.0, faithfulness=1.0),
        ]
        report = compute_report(results, settings)
        assert report.passed is True
