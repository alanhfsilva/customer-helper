from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.settings import Settings

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"


@dataclass(frozen=True)
class GoldenRecord:
    question: str
    reference_answer: str
    expected_source_ids: list[str]
    category: str
    should_abstain: bool
    history: list[dict[str, str]] | None = None


@dataclass(frozen=True)
class EvalResult:
    question: str
    category: str
    should_abstain: bool
    did_abstain: bool
    retrieved_ids: list[str]
    expected_source_ids: list[str]
    answer: str
    used_sources: list[str]
    accuracy: float
    faithfulness: float
    citation_validity: float
    latency_ms: int
    cost_usd: float


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)
    recall_at_5: float = 0.0
    mrr: float = 0.0
    mean_accuracy: float = 0.0
    mean_faithfulness: float = 0.0
    mean_citation_validity: float = 0.0
    refusal_precision: float = 0.0
    refusal_recall: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    mean_cost_usd: float = 0.0
    category_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    passed: bool = False


def load_golden_dataset(name: str = "golden") -> list[GoldenRecord]:
    path = DATASETS_DIR / f"{name}.jsonl"
    records: list[GoldenRecord] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(GoldenRecord(
                question=data["question"],
                reference_answer=data.get("reference_answer", ""),
                expected_source_ids=data.get("expected_source_ids", []),
                category=data.get("category", "general"),
                should_abstain=data.get("should_abstain", False),
                history=data.get("history"),
            ))
    return records


def _compute_recall_at_k(
    retrieved: list[str], expected: list[str], k: int = 5,
) -> float:
    if not expected:
        return 1.0
    top_k = set(retrieved[:k])
    hits = sum(1 for eid in expected if eid in top_k)
    return hits / len(expected)


def _compute_rr(retrieved: list[str], expected: list[str]) -> float:
    for i, rid in enumerate(retrieved):
        if rid in expected:
            return 1.0 / (i + 1)
    return 0.0


def _sorted_percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def compute_report(results: list[EvalResult], settings: Settings) -> EvalReport:
    if not results:
        return EvalReport(passed=False)

    recall_scores = [
        _compute_recall_at_k(r.retrieved_ids, r.expected_source_ids)
        for r in results if r.expected_source_ids
    ]
    rr_scores = [
        _compute_rr(r.retrieved_ids, r.expected_source_ids)
        for r in results if r.expected_source_ids
    ]

    answerable = [r for r in results if not r.should_abstain]
    accuracy_scores = [r.accuracy for r in answerable] if answerable else [0.0]
    faith_scores = [r.faithfulness for r in answerable] if answerable else [0.0]
    citation_scores = [
        r.citation_validity for r in answerable
    ] if answerable else [0.0]

    tp = sum(1 for r in results if r.should_abstain and r.did_abstain)
    fp = sum(1 for r in results if not r.should_abstain and r.did_abstain)
    fn = sum(1 for r in results if r.should_abstain and not r.did_abstain)

    refusal_precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    refusal_recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

    latencies = [float(r.latency_ms) for r in results]
    costs = [r.cost_usd for r in results]

    categories: dict[str, list[EvalResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    cat_breakdown: dict[str, dict[str, float]] = {}
    for cat, cat_results in categories.items():
        cat_answerable = [r for r in cat_results if not r.should_abstain]
        cat_breakdown[cat] = {
            "count": len(cat_results),
            "accuracy": (
                sum(r.accuracy for r in cat_answerable) / len(cat_answerable)
                if cat_answerable else 0.0
            ),
            "faithfulness": (
                sum(r.faithfulness for r in cat_answerable) / len(cat_answerable)
                if cat_answerable else 0.0
            ),
        }

    recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    mrr = sum(rr_scores) / len(rr_scores) if rr_scores else 0.0
    mean_acc = sum(accuracy_scores) / len(accuracy_scores)
    mean_faith = sum(faith_scores) / len(faith_scores)
    mean_cit = sum(citation_scores) / len(citation_scores)

    thresholds = settings.thresholds
    passed = (
        mean_acc >= thresholds.answer_accuracy_floor
        and mean_faith >= thresholds.faithfulness_floor
        and recall >= thresholds.retrieval_recall_at_5
        and mrr >= thresholds.retrieval_mrr
    )

    return EvalReport(
        results=results,
        recall_at_5=recall,
        mrr=mrr,
        mean_accuracy=mean_acc,
        mean_faithfulness=mean_faith,
        mean_citation_validity=mean_cit,
        refusal_precision=refusal_precision,
        refusal_recall=refusal_recall,
        p50_latency_ms=_sorted_percentile(latencies, 50),
        p95_latency_ms=_sorted_percentile(latencies, 95),
        mean_cost_usd=sum(costs) / len(costs) if costs else 0.0,
        category_breakdown=cat_breakdown,
        passed=passed,
    )


def save_report(
    report: EvalReport, run_id: str, output_dir: Path,
) -> tuple[Path, Path]:
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    report_data: dict[str, Any] = {
        "run_id": run_id,
        "passed": report.passed,
        "metrics": {
            "recall_at_5": round(report.recall_at_5, 4),
            "mrr": round(report.mrr, 4),
            "mean_accuracy": round(report.mean_accuracy, 4),
            "mean_faithfulness": round(report.mean_faithfulness, 4),
            "mean_citation_validity": round(report.mean_citation_validity, 4),
            "refusal_precision": round(report.refusal_precision, 4),
            "refusal_recall": round(report.refusal_recall, 4),
            "p50_latency_ms": round(report.p50_latency_ms, 1),
            "p95_latency_ms": round(report.p95_latency_ms, 1),
            "mean_cost_usd": round(report.mean_cost_usd, 6),
        },
        "category_breakdown": report.category_breakdown,
        "total_questions": len(report.results),
    }

    json_path = run_dir / "report.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    md_lines = [
        f"# Evaluation Report — {run_id}\n",
        f"**Result: {'PASSED' if report.passed else 'FAILED'}**\n",
        "## Metrics\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Recall@5 | {report.recall_at_5:.4f} |",
        f"| MRR | {report.mrr:.4f} |",
        f"| Mean Accuracy | {report.mean_accuracy:.4f} |",
        f"| Mean Faithfulness | {report.mean_faithfulness:.4f} |",
        f"| Citation Validity | {report.mean_citation_validity:.4f} |",
        f"| Refusal Precision | {report.refusal_precision:.4f} |",
        f"| Refusal Recall | {report.refusal_recall:.4f} |",
        f"| p50 Latency (ms) | {report.p50_latency_ms:.1f} |",
        f"| p95 Latency (ms) | {report.p95_latency_ms:.1f} |",
        f"| Mean Cost (USD) | {report.mean_cost_usd:.6f} |",
        "",
        "## Category Breakdown\n",
    ]
    for cat, metrics in report.category_breakdown.items():
        md_lines.append(
            f"- **{cat}**: {int(metrics['count'])} questions, "
            f"accuracy={metrics['accuracy']:.2f}, "
            f"faithfulness={metrics['faithfulness']:.2f}"
        )

    md_path = run_dir / "report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return json_path, md_path
