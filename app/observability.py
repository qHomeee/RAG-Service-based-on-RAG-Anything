import threading
import time
from collections import deque
from typing import Deque


class SLOMetrics:
    def __init__(self, window_size: int = 5000) -> None:
        self._latencies_ms: Deque[float] = deque(maxlen=window_size)
        self._statuses: Deque[int] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._started_at = time.time()

    def record(self, latency_ms: float, status_code: int) -> None:
        with self._lock:
            self._latencies_ms.append(latency_ms)
            self._statuses.append(status_code)

    def snapshot(self) -> dict:
        with self._lock:
            latencies = sorted(self._latencies_ms)
            statuses = list(self._statuses)

        count = len(latencies)
        if count == 0:
            return {
                "window_requests": 0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "error_rate": 0.0,
                "uptime_seconds": round(time.time() - self._started_at, 2),
            }

        def percentile(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            idx = max(0, min(len(values) - 1, int(round((p / 100) * (len(values) - 1)))))
            return float(values[idx])

        errors = sum(1 for code in statuses if code >= 500)
        return {
            "window_requests": count,
            "p95_latency_ms": round(percentile(latencies, 95), 2),
            "p99_latency_ms": round(percentile(latencies, 99), 2),
            "error_rate": round(errors / count, 4),
            "uptime_seconds": round(time.time() - self._started_at, 2),
        }


slo_metrics = SLOMetrics()
