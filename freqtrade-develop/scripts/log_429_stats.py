#!/usr/bin/env python3
"""
Summarize 429 (rate limited) events from Freqtrade logs per time bucket.

Parses WARNING lines emitted by the limiter (every ~100 denials) and
computes deltas per key (IP/Token/IP), then aggregates per minute or
custom bucket size.

Usage:
  python scripts/log_429_stats.py --logfile /path/to/freqtrade.log --bucket 1 --csv

Outputs CSV with columns: bucket_start,total_denials
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone

LINE_RE = re.compile(r"Rate limited \((?:IP|Token/IP)\):\s*([^,]+), count=(\d+)")


def parse_ts(line: str) -> datetime | None:
    # Expect timestamp at beginning: 2025-09-07 21:52:43
    try:
        ts = line[:19]
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def bucket_floor(ts: datetime, minutes: int) -> datetime:
    delta = timedelta(minutes=minutes)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    seconds = int((ts - epoch).total_seconds())
    bucket = seconds - (seconds % (minutes * 60))
    return epoch + timedelta(seconds=bucket)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize 429 denials from logs")
    ap.add_argument("--logfile", required=True, help="Path to freqtrade log file")
    ap.add_argument("--bucket", type=int, default=1, help="Bucket size in minutes (default: 1)")
    ap.add_argument("--csv", action="store_true", help="Output CSV format")
    args = ap.parse_args()

    last_counts: dict[str, int] = {}
    bucket_sums: defaultdict[datetime, int] = defaultdict(int)

    with open(args.logfile, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Rate limited (" not in line:
                continue
            m = LINE_RE.search(line)
            if not m:
                continue
            key = m.group(1).strip()
            cur = int(m.group(2))
            ts = parse_ts(line)
            if not ts:
                continue
            prev = last_counts.get(key)
            inc = 0
            if prev is None:
                # First observation; counts are cumulative mod logging cadence (~100).
                inc = cur  # approximate
            else:
                inc = max(0, cur - prev)
            last_counts[key] = cur
            b = bucket_floor(ts, args.bucket)
            bucket_sums[b] += inc

    if args.csv:
        print("bucket_start,total_denials")
        for b in sorted(bucket_sums):
            print(f"{b.isoformat()},{bucket_sums[b]}")
    else:
        print("429 denials per bucket:")
        for b in sorted(bucket_sums):
            print(f"{b.isoformat()} -> {bucket_sums[b]}")


if __name__ == "__main__":
    main()

