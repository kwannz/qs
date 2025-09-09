#!/usr/bin/env python3
"""
Derive a small set of tests affected by code changes, to run as a quick pre-check in CI.
Heuristics:
- Map changed python files to tests with matching basename (test_<module>.py) via filesystem search.
- Include any changed tests directly.
Outputs a newline-separated list of test paths to stdout.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def sh(cmd: list[str]) -> str:
    out = subprocess.check_output(cmd, text=True).strip()
    return out


def list_changed_files() -> list[Path]:
    # Prefer using GitHub-provided env for PRs
    base_ref = os.environ.get("GITHUB_BASE_REF")
    head_ref = os.environ.get("GITHUB_SHA")
    try:
        if base_ref and head_ref:
            # Ensure base is fetched
            try:
                subprocess.check_call(["git", "fetch", "origin", base_ref, "--depth=50"]) 
            except Exception:
                pass
            diff = sh(["git", "diff", "--name-only", f"origin/{base_ref}...{head_ref}"])
        else:
            diff = sh(["git", "diff", "--name-only", "HEAD~1..."])
    except Exception:
        return []
    files = [Path(x) for x in diff.splitlines() if x.endswith(".py")]
    return files


def map_to_tests(changed: list[Path]) -> list[Path]:
    testpaths: set[Path] = set()
    repo = Path.cwd()
    # Include directly changed tests
    for p in changed:
        if str(p).startswith("tests/"):
            testpaths.add(p)
    # Map production modules to tests by basename
    for p in changed:
        if str(p).startswith("freqtrade/"):
            name = p.stem
            # possible patterns
            candidates = list(repo.glob(f"tests/**/test_{name}.py"))
            for c in candidates:
                testpaths.add(c)
    return sorted(testpaths)


def main():
    changed = list_changed_files()
    tests = map_to_tests(changed)
    for t in tests:
        print(str(t))


if __name__ == "__main__":
    main()

