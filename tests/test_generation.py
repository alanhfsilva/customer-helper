from __future__ import annotations

import json

from app.generation.generator import GenerationResult, _parse_structured_response, generate_answer
from app.generation.prompt import assemble_context, load_template, render_system_prompt
from app.llm.client import FakeLLMClient
from app.llm.models import Message
from app.models import RetrievedChunk
from app.settings import ThresholdsConfig


def _chunk(
    chunk_id: str,
    text: str,
    source_uri: str = "/docs/test",
    title: str = "Test Doc",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        score=0.9,
        source_uri=source_uri,
        title=title,
        heading_path=["Test", "Section"],
    )


def _thresholds(**overrides: object) -> ThresholdsConfig:
    defaults: dict[str, object] = {
        "answer_accuracy_floor": 0.80,
        "faithfulness_floor": 0.95,
        "retrieval_recall_at_5": 0.85,
        "retrieval_mrr": 0.70,
        "grounding_score_floor": 0.80,
        "max_input_length": 4000,
        "max_output_tokens": 1024,
        "generation_temperature": 0.2,
        "context_token_budget": 3000,
        "p95_latency_ms": 4000,
    }
    defaults.update(overrides)
    return ThresholdsConfig(**defaults)  # type: ignore[arg-type]


CHUNKS = [
    _chunk("billing:0", "Refunds take 5-10 business days", "/billing", "Billing"),
    _chunk("shipping:0", "Standard shipping is 5-7 days", "/shipping", "Shipping"),
]


class TestLoadTemplate:
    def test_loads_system_prompt(self) -> None:
        template = load_template("system.md")
        assert "{{context}}" in template
        assert "{{company_name}}" in template

    def test_loads_condenser_prompt(self) -> None:
        template = load_template("query_condenser.md")
        assert "{{history}}" in template
        assert "{{message}}" in template


class TestAssembleContext:
    def test_includes_all_chunks(self) -> None:
        ctx = assemble_context(CHUNKS, token_budget=5000)
        assert "billing:0" in ctx
        assert "shipping:0" in ctx

    def test_respects_token_budget(self) -> None:
        long_text = "word " * 500
        big_chunks = [
            _chunk("c1", long_text),
            _chunk("c2", long_text),
            _chunk("c3", long_text),
        ]
        ctx = assemble_context(big_chunks, token_budget=700)
        assert "c1" in ctx
        assert "c3" not in ctx

    def test_includes_heading_path(self) -> None:
        ctx = assemble_context(CHUNKS, token_budget=5000)
        assert "Test > Section" in ctx

    def test_includes_source_uri(self) -> None:
        ctx = assemble_context(CHUNKS, token_budget=5000)
        assert "/billing" in ctx


class TestRenderSystemPrompt:
    def test_replaces_placeholders(self) -> None:
        prompt = render_system_prompt(CHUNKS, token_budget=5000, company_name="Acme")
        assert "Acme" in prompt
        assert "{{company_name}}" not in prompt
        assert "{{context}}" not in prompt

    def test_context_contains_chunk_text(self) -> None:
        prompt = render_system_prompt(CHUNKS, token_budget=5000)
        assert "Refunds take 5-10 business days" in prompt


class TestParseStructuredResponse:
    def test_parses_valid_json(self) -> None:
        content = json.dumps({
            "answer": "Refunds take 5-10 days [billing:0]",
            "used_sources": ["billing:0"],
        })
        answer, sources = _parse_structured_response(content)
        assert answer == "Refunds take 5-10 days [billing:0]"
        assert sources == ["billing:0"]

    def test_parses_json_in_code_fence(self) -> None:
        content = '```json\n{"answer": "test", "used_sources": ["a"]}\n```'
        answer, sources = _parse_structured_response(content)
        assert answer == "test"
        assert sources == ["a"]

    def test_fallback_on_invalid_json(self) -> None:
        content = "Just a plain text response"
        answer, sources = _parse_structured_response(content)
        assert answer == "Just a plain text response"
        assert sources == []


class TestGenerateAnswer:
    def test_returns_generation_result(self) -> None:
        response = json.dumps({
            "answer": "Refunds take 5-10 business days [billing:0]",
            "used_sources": ["billing:0"],
        })
        llm = FakeLLMClient(chat_response=response)
        result = generate_answer("refund time?", CHUNKS, llm, _thresholds())
        assert isinstance(result, GenerationResult)
        assert "Refunds" in result.answer
        assert "billing:0" in result.used_sources

    def test_builds_citations(self) -> None:
        response = json.dumps({
            "answer": "See [billing:0]",
            "used_sources": ["billing:0"],
        })
        llm = FakeLLMClient(chat_response=response)
        result = generate_answer("refund?", CHUNKS, llm, _thresholds())
        assert len(result.citations) == 1
        assert result.citations[0]["title"] == "Billing"
        assert result.citations[0]["source_uri"] == "/billing"

    def test_passes_temperature_from_config(self) -> None:
        response = json.dumps({"answer": "ok", "used_sources": []})
        llm = FakeLLMClient(chat_response=response)
        generate_answer("test", CHUNKS, llm, _thresholds(generation_temperature=0.1))
        assert len(llm.chat_calls) == 1
        assert llm.chat_calls[0]["temperature"] == 0.1

    def test_passes_max_tokens_from_config(self) -> None:
        response = json.dumps({"answer": "ok", "used_sources": []})
        llm = FakeLLMClient(chat_response=response)
        generate_answer("test", CHUNKS, llm, _thresholds(max_output_tokens=512))
        assert llm.chat_calls[0]["max_tokens"] == 512

    def test_includes_history(self) -> None:
        response = json.dumps({"answer": "ok", "used_sources": []})
        llm = FakeLLMClient(chat_response=response)
        history = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ]
        generate_answer("follow up", CHUNKS, llm, _thresholds(), history=history)
        messages = llm.chat_calls[0]["messages"]
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[1].content == "hello"
        assert messages[2].role == "assistant"
        assert messages[-1].content == "follow up"

    def test_system_prompt_is_first_message(self) -> None:
        response = json.dumps({"answer": "ok", "used_sources": []})
        llm = FakeLLMClient(chat_response=response)
        generate_answer("question", CHUNKS, llm, _thresholds())
        messages = llm.chat_calls[0]["messages"]
        assert messages[0].role == "system"
        assert messages[-1].role == "user"
        assert messages[-1].content == "question"

    def test_grounded_answer_for_known_question(self) -> None:
        response = json.dumps({
            "answer": "Refunds take 5-10 business days [billing:0]",
            "used_sources": ["billing:0"],
        })
        llm = FakeLLMClient(chat_response=response)
        result = generate_answer("how long for a refund?", CHUNKS, llm, _thresholds())
        assert "billing:0" in result.used_sources
        assert len(result.citations) > 0

    def test_unknown_source_excluded_from_citations(self) -> None:
        response = json.dumps({
            "answer": "See [nonexistent:0]",
            "used_sources": ["nonexistent:0"],
        })
        llm = FakeLLMClient(chat_response=response)
        result = generate_answer("test", CHUNKS, llm, _thresholds())
        assert result.citations == []

    def test_tracks_usage_info(self) -> None:
        response = json.dumps({"answer": "ok", "used_sources": []})
        llm = FakeLLMClient(chat_response=response)
        result = generate_answer("test", CHUNKS, llm, _thresholds())
        assert result.prompt_tokens >= 0
        assert result.completion_tokens >= 0
        assert result.cost_usd >= 0.0
        assert result.request_id != ""

    def test_out_of_context_uses_no_fabrication(self) -> None:
        response = json.dumps({
            "answer": "I don't have that information.",
            "used_sources": [],
        })
        llm = FakeLLMClient(chat_response=response)
        result = generate_answer(
            "what is the meaning of life?", CHUNKS, llm, _thresholds()
        )
        assert result.used_sources == []
        assert result.citations == []
        assert "don't have" in result.answer

    def test_company_name_in_prompt(self) -> None:
        response = json.dumps({"answer": "ok", "used_sources": []})
        llm = FakeLLMClient(chat_response=response)
        generate_answer(
            "test", CHUNKS, llm, _thresholds(), company_name="Acme Corp"
        )
        system_msg = llm.chat_calls[0]["messages"][0]
        assert "Acme Corp" in system_msg.content
