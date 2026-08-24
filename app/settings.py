from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_yaml(filename: str) -> dict[str, Any]:
    path = CONFIG_DIR / filename
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return data


class ModelsConfig(BaseModel):
    chat_model: str
    chat_model_fallback: str
    embedding_model: str
    embedding_dimensions: int
    moderation_model: str


class RetrievalConfig(BaseModel):
    k: int = 5
    fetch_n: int = 20
    score_threshold: float = 0.3
    low_confidence_floor: float = 0.4
    hybrid_alpha: float = 0.7
    rerank_enabled: bool = False


class IngestionConfig(BaseModel):
    chunk_target_tokens: int = 400
    chunk_max_tokens: int = 500
    chunk_overlap_fraction: float = 0.15
    embedding_batch_size: int = 100
    embedding_concurrency: int = 5


class ThresholdsConfig(BaseModel):
    answer_accuracy_floor: float = 0.80
    faithfulness_floor: float = 0.95
    retrieval_recall_at_5: float = 0.85
    retrieval_mrr: float = 0.70
    grounding_score_floor: float = 0.80
    max_input_length: int = 4000
    max_output_tokens: int = 1024
    generation_temperature: float = 0.2
    context_token_budget: int = 3000
    p95_latency_ms: int = 4000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "support-assistant"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    openai_api_key: str = Field(default="", description="Injected at runtime; never in repo")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/support_assistant"

    models: ModelsConfig = Field(default_factory=lambda: ModelsConfig(**_load_yaml("models.yaml")))
    retrieval: RetrievalConfig = Field(
        default_factory=lambda: RetrievalConfig(**_load_yaml("retrieval.yaml"))
    )
    ingestion: IngestionConfig = Field(
        default_factory=lambda: IngestionConfig(**_load_yaml("ingestion.yaml"))
    )
    thresholds: ThresholdsConfig = Field(
        default_factory=lambda: ThresholdsConfig(**_load_yaml("thresholds.yaml"))
    )


def get_settings() -> Settings:
    return Settings()
