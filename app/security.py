import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from app.config import settings


class InMemoryRateLimiter:
    def __init__(self, limit_per_minute: int) -> None:
        self.limit_per_minute = limit_per_minute
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.time()
        border = now - 60
        with self._lock:
            events = self._events[key]
            while events and events[0] < border:
                events.popleft()
            if len(events) >= self.limit_per_minute:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            events.append(now)


rate_limiter = InMemoryRateLimiter(limit_per_minute=settings.rate_limit_per_minute)


def require_rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    rate_limiter.check(client)
