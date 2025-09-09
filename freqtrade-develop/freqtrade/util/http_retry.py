import logging
import random
import time
from typing import Any, Dict, Optional

import requests
from requests import Response
from requests.exceptions import RequestException


logger = logging.getLogger(__name__)


def _sleep_backoff(attempt: int, base: float = 1.0, cap: float = 10.0) -> None:
    delay = min(base * (2 ** (attempt - 1)), cap)
    # Add small jitter to reduce thundering herd
    delay = delay * (0.8 + 0.4 * random.random())
    time.sleep(delay)


def _should_retry(resp: Optional[Response], err: Optional[BaseException]) -> tuple[bool, float | None]:
    if err is not None:
        return True, None
    if resp is None:
        return True, None
    if resp.status_code in (429, 500, 502, 503, 504):
        # Respect Retry-After (seconds)
        ra = resp.headers.get("Retry-After")
        if ra is not None:
            try:
                return True, float(ra)
            except Exception:
                return True, None
        return True, None
    return False, None


def http_get_json(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: int | float = 30,
    retries: int = 3,
) -> Any:
    last_err: Optional[BaseException] = None
    for attempt in range(1, retries + 2):
        resp: Optional[Response] = None
        err: Optional[BaseException] = None
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if 200 <= resp.status_code < 300:
                return resp.json()
            else:
                logger.warning(f"GET {url} -> {resp.status_code}")
        except RequestException as e:
            err = e
            last_err = e
            logger.warning(f"GET {url} raised {type(e).__name__}: {e}")

        should_retry, retry_after = _should_retry(resp, err)
        if not should_retry or attempt > retries:
            if err is not None:
                raise err
            assert resp is not None
            resp.raise_for_status()
        if retry_after is not None:
            time.sleep(retry_after)
        else:
            _sleep_backoff(attempt)
    # Should not reach here
    if last_err:
        raise last_err


def http_get_bytes(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: int | float = 30,
    retries: int = 3,
) -> bytes:
    last_err: Optional[BaseException] = None
    for attempt in range(1, retries + 2):
        resp: Optional[Response] = None
        err: Optional[BaseException] = None
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if 200 <= resp.status_code < 300:
                return resp.content
            else:
                logger.warning(f"GET {url} -> {resp.status_code}")
        except RequestException as e:
            err = e
            last_err = e
            logger.warning(f"GET {url} raised {type(e).__name__}: {e}")

        should_retry, retry_after = _should_retry(resp, err)
        if not should_retry or attempt > retries:
            if err is not None:
                raise err
            assert resp is not None
            resp.raise_for_status()
        if retry_after is not None:
            time.sleep(retry_after)
        else:
            _sleep_backoff(attempt)
    if last_err:
        raise last_err
    raise RuntimeError("http_get_bytes failed without specific error")

