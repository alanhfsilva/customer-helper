from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitConfig:
    requests_per_minute: int = 60
    window_seconds: int = 60


class RateLimiter:
    def __init__(self, config: RateLimitConfig | None = None) -> None:
        self._config = config or RateLimitConfig()
        self._lock = threading.Lock()
        self._buckets: dict[str, list[float]] = {}

    def is_allowed(self, caller_id: str) -> bool:
        now = time.monotonic()
        cutoff = now - self._config.window_seconds

        with self._lock:
            timestamps = self._buckets.get(caller_id, [])
            timestamps = [t for t in timestamps if t > cutoff]

            if len(timestamps) >= self._config.requests_per_minute:
                self._buckets[caller_id] = timestamps
                return False

            self._buckets[caller_id] = [*timestamps, now]
            return True
