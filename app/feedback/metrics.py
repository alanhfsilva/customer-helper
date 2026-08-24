from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class OnlineMetrics:
    total_requests: int
    answered: int
    abstained: int
    escalated: int
    blocked: int
    total_cost_usd: float
    total_latency_ms: int
    deflection_rate: float
    escalation_rate: float
    avg_latency_ms: float
    avg_cost_usd: float


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_requests = 0
        self._answered = 0
        self._abstained = 0
        self._escalated = 0
        self._blocked = 0
        self._total_cost_usd = 0.0
        self._total_latency_ms = 0

    def record(
        self,
        status: str,
        latency_ms: int,
        cost_usd: float,
    ) -> None:
        with self._lock:
            self._total_requests += 1
            self._total_latency_ms += latency_ms
            self._total_cost_usd += cost_usd
            if status == "answered":
                self._answered += 1
            elif status == "abstained":
                self._abstained += 1
            elif status == "escalated":
                self._escalated += 1
            elif status == "blocked":
                self._blocked += 1

    def snapshot(self) -> OnlineMetrics:
        with self._lock:
            total = self._total_requests
            answered = self._answered
            escalated = self._escalated
            deflection_rate = answered / total if total > 0 else 0.0
            escalation_rate = escalated / total if total > 0 else 0.0
            avg_latency = self._total_latency_ms / total if total > 0 else 0.0
            avg_cost = self._total_cost_usd / total if total > 0 else 0.0

            return OnlineMetrics(
                total_requests=total,
                answered=answered,
                abstained=self._abstained,
                escalated=escalated,
                blocked=self._blocked,
                total_cost_usd=round(self._total_cost_usd, 8),
                total_latency_ms=self._total_latency_ms,
                deflection_rate=round(deflection_rate, 4),
                escalation_rate=round(escalation_rate, 4),
                avg_latency_ms=round(avg_latency, 2),
                avg_cost_usd=round(avg_cost, 8),
            )

    def reset(self) -> None:
        with self._lock:
            self._total_requests = 0
            self._answered = 0
            self._abstained = 0
            self._escalated = 0
            self._blocked = 0
            self._total_cost_usd = 0.0
            self._total_latency_ms = 0
