#!/usr/bin/env python3
"""
Flood private API endpoints to validate rate limiting.

Features:
- Optional token login via username/password -> JWT
- Configurable duration and RPS
- Targets a set of private endpoints (GET/POST), reports HTTP status histogram

Examples:
  python scripts/flood_private_endpoints.py \
    --base http://127.0.0.1:8080 \
    --duration 60 \
    --rps 200 \
    --username user --password pass

  python scripts/flood_private_endpoints.py \
    --base http://127.0.0.1:8080 \
    --duration 30 \
    --rps 100 \
    --token eyJhbGciOi...

Note: Run only against your local/test deployment.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from typing import Optional

import httpx


async def get_token(base: str, username: str, password: str) -> str:
    url = f"{base.rstrip('/')}/api/v1/token/login"
    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.post(url, json={"username": username, "password": password})
        r.raise_for_status()
        data = r.json()
        token = data.get("access_token") or data.get("token") or ""
        if not token:
            raise RuntimeError("No access token returned from login endpoint")
        return token


async def worker(base: str, headers: dict[str, str], q: asyncio.Queue, counter: Counter) -> None:
    async with httpx.AsyncClient(timeout=5) as client:
        while True:
            try:
                method, path, payload = await q.get()
            except asyncio.CancelledError:
                break
            try:
                url = f"{base.rstrip('/')}{path}"
                if method == "GET":
                    r = await client.get(url, headers=headers)
                else:
                    r = await client.post(url, headers=headers, json=payload)
                counter[r.status_code] += 1
            except Exception:
                counter[0] += 1
            finally:
                q.task_done()


def build_targets() -> list[tuple[str, str, Optional[dict]]]:
    # Minimal private endpoints set; adjust as needed
    return [
        ("GET", "/api/v1/trades", None),
        ("GET", "/api/v1/strategies", None),
        ("POST", "/api/v1/blacklist", {"pair": "BTC/USDT", "action": "add"}),
        ("GET", "/api/v1/bot/state", None),
    ]


async def main_async(args: argparse.Namespace) -> None:
    token = args.token
    if not token and args.username and args.password:
        token = await get_token(args.base, args.username, args.password)

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    targets = build_targets()

    q: asyncio.Queue = asyncio.Queue(maxsize=args.rps * 2)
    counter: Counter = Counter()

    # Workers
    workers = [asyncio.create_task(worker(args.base, headers, q, counter)) for _ in range(min(32, args.rps))]

    # Producer: enqueue requests at desired RPS for duration
    start = time.monotonic()
    end = start + args.duration
    i = 0
    try:
        while time.monotonic() < end:
            # enqueue args.rps requests per second
            for _ in range(args.rps):
                method, path, payload = targets[i % len(targets)]
                await q.put((method, path, payload))
                i += 1
            await asyncio.sleep(1)
    finally:
        await q.join()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    # Report
    total = sum(counter.values())
    print(json.dumps({"total": total, "by_status": dict(counter)}, indent=2))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Flood private endpoints to validate rate limiting")
    ap.add_argument("--base", required=True, help="Base URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    ap.add_argument("--rps", type=int, default=100, help="Requests per second")
    ap.add_argument("--token", help="Bearer token (if provided, skips login)")
    ap.add_argument("--username", help="Username for login")
    ap.add_argument("--password", help="Password for login")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

