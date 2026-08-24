from __future__ import annotations

from app.settings import Settings, get_settings


def test_settings_loads_yaml_configs() -> None:
    settings = get_settings()
    assert settings.models.chat_model == "gpt-4o-2024-08-06"
    assert settings.models.embedding_model == "text-embedding-3-small"
    assert settings.models.embedding_dimensions == 1536


def test_settings_loads_thresholds() -> None:
    settings = get_settings()
    assert settings.thresholds.answer_accuracy_floor == 0.80
    assert settings.thresholds.faithfulness_floor == 0.95
    assert settings.thresholds.max_input_length == 4000


def test_settings_loads_retrieval() -> None:
    settings = get_settings()
    assert settings.retrieval.k == 5
    assert settings.retrieval.fetch_n == 20


def test_settings_loads_ingestion() -> None:
    settings = get_settings()
    assert settings.ingestion.chunk_target_tokens == 400
    assert settings.ingestion.chunk_overlap_fraction == 0.15


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.app_name == "support-assistant"
    assert settings.port == 8000
    assert settings.openai_api_key == ""
