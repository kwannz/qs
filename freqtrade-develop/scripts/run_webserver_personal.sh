#!/usr/bin/env bash
set -euo pipefail

CONF="user_data/config_personal.json"

if [ ! -f .venv/bin/activate ]; then
  echo "Virtualenv .venv not found. Run scripts/install-pinned.sh first." >&2
  exit 1
fi

if [ ! -f "$CONF" ]; then
  echo "Config $CONF not found. Create it or run the task that generated it." >&2
  exit 1
fi

source .venv/bin/activate

mkdir -p user_data/logs

exec freqtrade webserver -c "$CONF"
