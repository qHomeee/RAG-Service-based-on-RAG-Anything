import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from app.config import settings

logger = logging.getLogger("rag_service")


class RateLimiterBackend(ABC):
    @abstractmethod
    def check(self, key: str) -> None:
        raise NotImplementedError


class InMemoryRateLimiter(RateLimiterBackend):
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


class RedisRateLimiter(RateLimiterBackend):
    def __init__(self, redis_url: str, limit_per_minute: int) -> None:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("redis package is required for Redis rate limiter") from exc

        self.limit_per_minute = limit_per_minute
        self._redis = redis.Redis.from_url(redis_url)

    def check(self, key: str) -> None:
        bucket = int(time.time() // 60)
        redis_key = f"rag:rl:{key}:{bucket}"
        count = self._redis.incr(redis_key)
        if count == 1:
            self._redis.expire(redis_key, 61)
        if count > self.limit_per_minute:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")


def create_rate_limiter() -> RateLimiterBackend:
    if settings.redis_url:
        try:
            limiter = RedisRateLimiter(settings.redis_url, settings.rate_limit_per_minute)
            logger.info("rate_limiter_backend", extra={"backend": "redis"})
            return limiter
        except Exception:
            logger.exception("redis_rate_limiter_init_failed")

    logger.info("rate_limiter_backend", extra={"backend": "in_memory"})
    return InMemoryRateLimiter(limit_per_minute=settings.rate_limit_per_minute)


rate_limiter: RateLimiterBackend = create_rate_limiter()


def require_rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    rate_limiter.check(client)
