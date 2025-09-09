import time
from collections import deque
from dataclasses import dataclass
import logging
from typing import Deque, Optional

from fastapi import HTTPException, Request, status


@dataclass
class _Bucket:
    timestamps: Deque[float]


class BaseLimiter:
    def allow(self, key: str) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


class SimpleRateLimiter(BaseLimiter):
    """
    In-memory sliding-window rate limiter.

    Not suitable for multi-process deployments. Intended as a safe default for local/WS mode.
    """

    def __init__(self, limit: int, window_sec: int):
        self.limit = int(limit)
        self.window = float(window_sec)
        self._buckets: dict[str, _Bucket] = {}

    def _bucket(self, key: str) -> _Bucket:
        b = self._buckets.get(key)
        if b is None:
            b = _Bucket(deque(maxlen=self.limit))
            self._buckets[key] = b
        return b

    def allow(self, key: str) -> bool:
        now = time.time()
        b = self._bucket(key)
        # Evict old timestamps
        while b.timestamps and (now - b.timestamps[0]) > self.window:
            b.timestamps.popleft()
        if len(b.timestamps) >= self.limit:
            return False
        b.timestamps.append(now)
        return True


class RedisRateLimiter(BaseLimiter):
    """
    Fixed-window rate limiter backed by Redis (best-effort).
    Requires `redis` package if selected via config.
    """

    def __init__(self, limit: int, window_sec: int, redis_url: str, prefix: str):
        try:
            import redis  # type: ignore
        except Exception as e:  # pragma: no cover - optional
            raise RuntimeError("Redis backend requested but 'redis' package not available") from e
        self.limit = int(limit)
        self.window = int(window_sec)
        self.prefix = prefix
        self._r = redis.from_url(redis_url, decode_responses=True)

    def allow(self, key: str) -> bool:
        import time

        now = int(time.time())
        bucket = now // self.window
        redis_key = f"{self.prefix}:{bucket}:{key}"
        pipe = self._r.pipeline()
        pipe.incr(redis_key, 1)
        pipe.expire(redis_key, self.window)
        count, _ = pipe.execute()
        return int(count) <= self.limit


class RedisSlidingWindowLimiter(BaseLimiter):
    """
    Sliding-window limiter using Redis sorted-set.
    Stores timestamps in seconds; trims by window and counts members.
    """

    def __init__(self, limit: int, window_sec: int, redis_url: str, prefix: str):
        try:
            import redis  # type: ignore
        except Exception as e:  # pragma: no cover - optional
            raise RuntimeError("Redis backend requested but 'redis' package not available") from e
        self.limit = int(limit)
        self.window = int(window_sec)
        self.prefix = prefix
        self._r = redis.from_url(redis_url, decode_responses=True)

    def allow(self, key: str) -> bool:
        import time

        now = int(time.time())
        cutoff = now - self.window
        zkey = f"{self.prefix}:sw:{key}"
        pipe = self._r.pipeline()
        pipe.zremrangebyscore(zkey, 0, cutoff)
        pipe.zadd(zkey, {str(now): now})
        pipe.zcard(zkey)
        pipe.expire(zkey, self.window)
        _, _, count, _ = pipe.execute()
        return int(count) <= self.limit


_global_limiters: dict[str, BaseLimiter] = {}
_rl_denials: dict[str, int] = {}
logger = logging.getLogger(__name__)


def _get_backend_limiter(limiter_key: str, limit: int, window_sec: int, backend: str, redis_url: Optional[str]):
    if limiter_key in _global_limiters:
        return _global_limiters[limiter_key]
    if backend == "redis" and redis_url:
        try:
            limiter = RedisRateLimiter(limit, window_sec, redis_url, prefix=limiter_key)
        except Exception:
            limiter = SimpleRateLimiter(limit, window_sec)
    elif backend in ("redis_sliding", "redis_sw") and redis_url:
        try:
            limiter = RedisSlidingWindowLimiter(limit, window_sec, redis_url, prefix=limiter_key)
        except Exception:
            limiter = SimpleRateLimiter(limit, window_sec)
    else:
        limiter = SimpleRateLimiter(limit, window_sec)
    _global_limiters[limiter_key] = limiter
    return limiter


def ip_rate_limiter(limit: int, window_sec: int, *, backend: str = "memory", redis_url: str | None = None):
    limiter_key = f"ip:{limit}:{window_sec}"
    # Limiter is (lazily) created in dependency below to ensure cfg is passed

    async def _dep(request: Request):
        client_ip = request.client.host if request.client else "unknown"
        limiter = _get_backend_limiter(limiter_key, limit, window_sec, backend, redis_url)
        if not limiter.allow(client_ip):
            key = f"{limiter_key}:{client_ip}"
            _rl_denials[key] = _rl_denials.get(key, 0) + 1
            if _rl_denials[key] % 100 == 1:
                logger.warning("Rate limited (IP): %s, count=%s", client_ip, _rl_denials[key])
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limited")
        return True

    return _dep


def token_or_ip_rate_limiter(limit: int, window_sec: int, *, backend: str = "memory", redis_url: str | None = None):
    limiter_key = f"token:{limit}:{window_sec}"

    async def _dep(request: Request):
        # Prefer token; fallback to IP
        auth = request.headers.get("authorization") or ""
        key = auth.split()[1] if auth.lower().startswith("bearer ") and len(auth.split()) > 1 else None
        if not key and request.client:
            key = request.client.host
        key = key or "unknown"
        limiter = _get_backend_limiter(limiter_key, limit, window_sec, backend, redis_url)
        if not limiter.allow(key):
            mkey = f"{limiter_key}:{key}"
            _rl_denials[mkey] = _rl_denials.get(mkey, 0) + 1
            if _rl_denials[mkey] % 100 == 1:
                logger.warning("Rate limited (Token/IP): %s, count=%s", key, _rl_denials[mkey])
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limited")
        return True

    return _dep
