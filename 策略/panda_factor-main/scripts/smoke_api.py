#!/usr/bin/env python3
"""
Panda Factor API Smoke Test

Prerequisites:
- Panda Factor server running (default http://127.0.0.1:8111)
- MongoDB with required market data so factor analysis can run
- Optional: set API_KEY and BASE_URL via env or args

Usage:
  python scripts/smoke_api.py --base http://127.0.0.1:8111 --api-key <KEY>

This will:
  1) Create a factor
  2) Run the factor (async)
  3) Poll status until finished
  4) Fetch a subset of chart endpoints
"""
import os
import sys
import time
import json
import uuid
import argparse

try:
    import requests
except Exception:
    print("This script requires 'requests'. Install via: pip install requests", file=sys.stderr)
    sys.exit(1)


def _headers(api_key: str | None) -> dict:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["X-API-Key"] = api_key
    return h


def create_factor(base: str, api_key: str | None) -> str:
    url = f"{base}/api/v1/factors"
    factor_name = f"smoke_factor_{uuid.uuid4().hex[:8]}"
    payload = {
        "user_id": "1",
        "name": "Smoke因子",
        "factor_name": factor_name,
        "factor_type": "stock",
        "is_persistent": False,
        "cron": None,
        "factor_start_day": None,
        "code": "RANK((CLOSE / DELAY(CLOSE, 20)) - 1)",
        "code_type": "formula",
        "tags": "smoke",
        "status": 0,
        "describe": "smoke test",
        "params": {
            "start_date": "2023-01-01",
            "end_date": "2023-06-01",
            "adjustment_cycle": 5,
            "stock_pool": "000300",
            "factor_direction": True,
            "group_number": 5,
            "include_st": False,
            "extreme_value_processing": "中位数"
        }
    }
    resp = requests.post(url, headers=_headers(api_key), data=json.dumps(payload), timeout=30)
    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(f"Create factor failed: HTTP {resp.status_code} body={resp.text}")
    if data.get("code") != "200":
        raise RuntimeError(f"Create factor failed: {data}")
    factor_id = data.get("data", {}).get("factor_id")
    if not factor_id:
        raise RuntimeError(f"Create factor missing factor_id: {data}")
    print(f"Created factor: {factor_name} id={factor_id}")
    return factor_id


def run_factor(base: str, api_key: str | None, factor_id: str) -> None:
    url = f"{base}/api/v1/factors/{factor_id}/run?is_thread=true"
    resp = requests.post(url, headers=_headers(api_key), timeout=30)
    data = resp.json()
    if data.get("code") != "200":
        raise RuntimeError(f"Run factor failed: {data}")
    print(f"Run started: {data.get('data')}")


def poll_status(base: str, api_key: str | None, factor_id: str, timeout_s: int = 600, interval_s: int = 5) -> tuple[str, int]:
    """Return (task_id, status) when finished. status: 2=success, 3=failed"""
    url = f"{base}/api/v1/factors/{factor_id}/status"
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        resp = requests.get(url, headers=_headers(api_key), timeout=30)
        data = resp.json()
        if data.get("code") != "200":
            raise RuntimeError(f"Query factor status failed: {data}")
        info = data.get("data") or {}
        status = info.get("status")  # 1 running, 2 done, 3 failed
        task_id = info.get("task_id")
        if last != status:
            print(f"Status update: status={status}, task_id={task_id}")
            last = status
        if status in (2, 3) and task_id:
            return task_id, status
        time.sleep(interval_s)
    raise TimeoutError("Timeout waiting for factor to finish")


def fetch_charts(base: str, api_key: str | None, task_id: str) -> None:
    endpoints = [
        f"/api/v1/tasks/{task_id}",
        f"/api/v1/tasks/{task_id}/charts/ic/decay",
        f"/api/v1/tasks/{task_id}/charts/rank-ic/sequence",
        f"/api/v1/tasks/{task_id}/charts/return",
    ]
    for ep in endpoints:
        url = f"{base}{ep}"
        resp = requests.get(url, headers=_headers(api_key), timeout=60)
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"GET {ep} failed: HTTP {resp.status_code} body={resp.text}")
        if data.get("code") != "200":
            raise RuntimeError(f"GET {ep} failed: {data}")
        print(f"OK {ep}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Panda Factor API smoke test")
    parser.add_argument("--base", default=os.getenv("BASE_URL", "http://127.0.0.1:8111"))
    parser.add_argument("--api-key", default=os.getenv("API_KEY"))
    args = parser.parse_args(argv)

    base = args.base.rstrip("/")
    api_key = args.api_key

    print(f"Base URL: {base}")
    try:
        fid = create_factor(base, api_key)
        run_factor(base, api_key, fid)
        task_id, status = poll_status(base, api_key, fid)
        print(f"Task finished: task_id={task_id}, status={status}")
        if status == 2:
            fetch_charts(base, api_key, task_id)
            print("Smoke test completed successfully.")
        else:
            print("Factor run failed, charts will be skipped.")
            sys.exit(2)
    except Exception as e:
        print(f"Smoke test failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

