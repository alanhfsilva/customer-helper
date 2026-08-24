from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class IngestionReport:
    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    documents_processed: int = 0
    documents_skipped: int = 0
    chunks_created: int = 0
    total_tokens_embedded: int = 0
    embedding_cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)

    def save(self, output_dir: str | Path) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        report_path = out / f"report_{timestamp}.json"
        report_path.write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8"
        )

        md_path = out / f"report_{timestamp}.md"
        md_path.write_text(self._to_markdown(), encoding="utf-8")

        return report_path

    def _to_markdown(self) -> str:
        return (
            f"# Ingestion Report\n\n"
            f"- **Run ID:** {self.run_id}\n"
            f"- **Started:** {self.started_at}\n"
            f"- **Finished:** {self.finished_at}\n"
            f"- **Documents processed:** {self.documents_processed}\n"
            f"- **Documents skipped (unchanged):** {self.documents_skipped}\n"
            f"- **Chunks created:** {self.chunks_created}\n"
            f"- **Tokens embedded:** {self.total_tokens_embedded}\n"
            f"- **Embedding cost:** ${self.embedding_cost_usd:.6f}\n"
            f"- **Errors:** {len(self.errors)}\n"
        )
