#!/usr/bin/env bash
set -euo pipefail

# 一键：创建/使用 venv → 启动服务 → 保存工作流 → 运行工作流 → 打开日志页
# 可选参数：
#   5m|minute  使用 1m→5min 重采样示例工作流（默认使用日线示例）
#   --no-open  不自动打开浏览器

HERE="$(cd "$(dirname "$0")" && pwd)"
QF_DIR="$(cd "$HERE/.." && pwd)"
VENV="$QF_DIR/.venv"
PY=${PY:-python3.12}

WORKFLOW_JSON="$QF_DIR/examples/workflows/crypto_backtest_minimal.json"
OPEN_BROWSER=1
if [[ "${1:-}" == "5m" || "${1:-}" == "minute" ]]; then
  WORKFLOW_JSON="$QF_DIR/examples/workflows/crypto_backtest_minute_5m.json"
  shift || true
fi
if [[ "${1:-}" == "--no-open" ]]; then
  OPEN_BROWSER=0
  shift || true
fi

echo "[+] 工作目录: $QF_DIR"
if [[ ! -d "$VENV" ]]; then
  echo "[+] 创建 venv ($PY) ..."
  if ! command -v "$PY" >/dev/null 2>&1; then
    PY=python3
  fi
  "$PY" -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -V

echo "[+] 安装最小依赖..."
pip -q install -U pip >/dev/null
pip -q install fastapi uvicorn motor python-dotenv python-json-logger aiofiles aio-pika pandas numpy pytz cloudpickle uuid6 redis loguru statsmodels tornado >/dev/null

export RUN_MODE=LOCAL
export MONGO_URI="mongodb://127.0.0.1:27017"
export DATABASE_NAME=panda
export MONGO_USER=""
export MONGO_PASSWORD=""
export MONGO_AUTH_DB=""
export MONGO_TYPE=single
export ENABLE_CHAT_ROUTES=0
export PYTHONPATH="$QF_DIR/src"
export LOG_LEVEL=INFO
export LOG_CONSOLE=1
export LOG_FILE=1

echo "[+] 启动/检测服务..."
if ! curl -sf http://127.0.0.1:8000/api/plugins/all >/dev/null 2>&1; then
  echo "    -> 服务未启动，开始启动..."
  nohup python "$QF_DIR/src/panda_server/main.py" > "$QF_DIR/src/panda_server/server.log" 2>&1 &
  for i in {1..40}; do
    sleep 0.5
    if curl -sf http://127.0.0.1:8000/api/plugins/all >/dev/null 2>&1; then
      echo "    -> 服务已就绪。"
      break
    fi
    if [[ $i -eq 40 ]]; then
      echo "[!] 服务启动超时，请检查日志: $QF_DIR/src/panda_server/server.log" >&2
      exit 1
    fi
  done
else
  echo "    -> 服务已在运行。"
fi

echo "[+] 保存工作流: $WORKFLOW_JSON"
WF_SAVE=$(curl -s -X POST http://127.0.0.1:8000/api/workflow/save \
  -H "Content-Type: application/json" -H "uid: local-user" \
  --data-binary @"$WORKFLOW_JSON")
WF_ID=$(printf '%s' "$WF_SAVE" | sed -E 's/.*"workflow_id":"([^"]+)".*/\1/')
if [[ -z "$WF_ID" ]]; then
  echo "[!] 保存工作流失败: $WF_SAVE" >&2
  exit 1
fi
echo "    -> workflow_id=$WF_ID"

echo "[+] 运行工作流..."
RUN_JSON=$(curl -s -X POST http://127.0.0.1:8000/api/workflow/run \
  -H "Content-Type: application/json" -H "uid: local-user" -H "quantflow-auth: 2" \
  -d "{\"workflow_id\":\"$WF_ID\"}")
RUN_ID=$(printf '%s' "$RUN_JSON" | sed -E 's/.*"workflow_run_id":"([^"]+)".*/\1/')
if [[ -z "$RUN_ID" ]]; then
  echo "[!] 运行工作流失败: $RUN_JSON" >&2
  exit 1
fi
echo "    -> workflow_run_id=$RUN_ID"

echo "[+] 拉取运行日志（前20条）..."
sleep 1
curl -s "http://127.0.0.1:8000/api/workflow/run/log?workflow_run_id=$RUN_ID&limit=20" -H "uid: local-user" | sed -e 's/},{/},\n{/g' | head -n 20 || true

echo "[+] 访问日志页面: http://127.0.0.1:8000/logs/"
echo "    提示：填写 uid=local-user 与 workflow_run_id=$RUN_ID，可选填写 req_id/backtest_id 过滤"
if [[ $OPEN_BROWSER -eq 1 ]]; then
  if command -v open >/dev/null 2>&1; then
    open "http://127.0.0.1:8000/logs/" || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://127.0.0.1:8000/logs/" || true
  fi
fi

echo "[√] 完成。"

