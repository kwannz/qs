#!/usr/bin/env bash
set -euo pipefail

# Simple non-interactive installer using pinned requirements.
# Requires Python 3.11+ available as `python3` or `python3.11`/`python3.12`.

pick_python() {
  # Prefer Python 3.12 explicitly, then 3.13, then the default python3, then 3.11
  for py in python3.12 python3.13 python3 python3.11; do
    if command -v "$py" >/dev/null 2>&1; then
      "$py" -c 'import sys; assert sys.version_info[:2] >= (3,11)'
      if [ $? -eq 0 ]; then
        echo "$py"; return 0
      fi
    fi
  done
  echo "Error: Python 3.11+ not found on PATH." >&2
  exit 1
}

PY=$(pick_python)
echo "Using ${PY}"

# Recreate venv to ensure it's bound to Python 3.12
if [ -d .venv ]; then
  echo "Removing existing .venv to recreate with ${PY}"
  rm -rf .venv
fi

${PY} -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools

# Install dev (includes all extras and test deps) with pins
pip install -r requirements-dev.txt

# Install project in editable mode
pip install -e .

# Install/refresh UI assets
freqtrade install-ui || true

echo "Pinned environment installed. Activate with: source .venv/bin/activate"
