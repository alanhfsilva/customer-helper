from __future__ import annotations

import logging
import math
import uuid
from typing import TYPE_CHECKING, Any, Protocol

from app.llm.models import (
    ChatResult,
    EmbeddingResult,
    EmbeddingUsage,
    Message,
    ModerationResult,
)

if TYPE_CHECKING:
    from app.settings import ModelsConfig

logger = logging.getLogger(__name__)

COST_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.02),
    "text-embedding-3-large": (0.13, 0.13),
}

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_EMBEDDING_BATCH_SIZE = 100


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int = 0) -> float:
    rates = COST_PER_1M_TOKENS.get(model, (0.0, 0.0))
    input_cost = (prompt_tokens / 1_000_000) * rates[0]
    output_cost = (completion_tokens / 1_000_000) * rates[1]
    return round(input_cost + output_cost, 8)


class LLMClient(Protocol):
    def embed(self, texts: list[str]) -> EmbeddingResult: ...

    def chat(
        self,
        messages: list[Message],
        *,
        stream: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult: ...

    def moderate(self, text: str) -> ModerationResult: ...


class OpenAILLMClient:
    def __init__(
        self,
        api_key: str,
        models_config: ModelsConfig,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> None:
        import openai

        self._client = openai.OpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._models = models_config
        self._embedding_batch_size = embedding_batch_size

    def chat(
        self,
        messages: list[Message],
        *,
        stream: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        request_id = str(uuid.uuid4())
        model = self._models.chat_model

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = self._client.chat.completions.create(**kwargs)

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        content = response.choices[0].message.content or ""

        cost = compute_cost(model, prompt_tokens, completion_tokens)

        logger.info(
            "llm.chat",
            extra={
                "request_id": request_id,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost,
            },
        )

        return ChatResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            request_id=request_id,
            model=model,
        )

    def embed(self, texts: list[str]) -> EmbeddingResult:
        model = self._models.embedding_model
        all_embeddings: list[list[float]] = []
        total_tokens = 0
        request_id = str(uuid.uuid4())

        for i in range(0, len(texts), self._embedding_batch_size):
            batch = texts[i : i + self._embedding_batch_size]
            response = self._client.embeddings.create(
                model=model,
                input=batch,
                dimensions=self._models.embedding_dimensions,
            )
            for item in response.data:
                all_embeddings.append(item.embedding)
            if response.usage:
                total_tokens += response.usage.prompt_tokens

        cost = compute_cost(model, total_tokens)

        logger.info(
            "llm.embed",
            extra={
                "request_id": request_id,
                "model": model,
                "prompt_tokens": total_tokens,
                "cost_usd": cost,
                "batch_count": math.ceil(len(texts) / self._embedding_batch_size),
            },
        )

        return EmbeddingResult(
            embeddings=all_embeddings,
            usage=EmbeddingUsage(
                prompt_tokens=total_tokens,
                cost_usd=cost,
                request_id=request_id,
            ),
        )

    def moderate(self, text: str) -> ModerationResult:
        model = self._models.moderation_model
        response = self._client.moderations.create(model=model, input=text)
        result = response.results[0]

        return ModerationResult(
            flagged=result.flagged,
            categories={k: v for k, v in result.categories},
            category_scores={k: v for k, v in result.category_scores},
        )


class FakeLLMClient:
    def __init__(
        self,
        *,
        chat_response: str = "This is a fake response.",
        embeddings_dim: int = 1536,
        flagged: bool = False,
    ) -> None:
        self._chat_response = chat_response
        self._embeddings_dim = embeddings_dim
        self._flagged = flagged
        self.chat_calls: list[dict[str, Any]] = []
        self.embed_calls: list[list[str]] = []
        self.moderate_calls: list[str] = []

    def chat(
        self,
        messages: list[Message],
        *,
        stream: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        request_id = str(uuid.uuid4())
        self.chat_calls.append({
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": response_format,
        })

        prompt_tokens = sum(len(m.content.split()) * 2 for m in messages)
        completion_tokens = len(self._chat_response.split()) * 2

        return ChatResult(
            content=self._chat_response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=compute_cost("gpt-4o-2024-08-06", prompt_tokens, completion_tokens),
            request_id=request_id,
            model="fake-model",
        )

    def embed(self, texts: list[str]) -> EmbeddingResult:
        request_id = str(uuid.uuid4())
        self.embed_calls.append(texts)

        fake_embeddings = [[0.1] * self._embeddings_dim for _ in texts]
        prompt_tokens = sum(len(t.split()) * 2 for t in texts)

        return EmbeddingResult(
            embeddings=fake_embeddings,
            usage=EmbeddingUsage(
                prompt_tokens=prompt_tokens,
                cost_usd=compute_cost("text-embedding-3-small", prompt_tokens),
                request_id=request_id,
            ),
        )

    def moderate(self, text: str) -> ModerationResult:
        self.moderate_calls.append(text)
        return ModerationResult(flagged=self._flagged)
