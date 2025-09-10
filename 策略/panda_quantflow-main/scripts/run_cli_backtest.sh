#!/usr/bin/env bash
set -euo pipefail

# 一键 CLI 回测：分钟→5min 重采样 + 指定成本/约束 + 关联ID + 导出权益曲线
# 可选参数：
#   --start YYYYMMDD  默认 20240101
#   --end   YYYYMMDD  默认 20240101
#   --symbols "BINANCE:BTCUSDT,BINANCE:ETHUSDT"
#   --resample 5min|15min|30min|1h  默认 5min
#   --fee 1 --slip 2 --maker 0 --taker 5 --max_weight 0.6 --max_turnover 0.5 --min_trade 0.01
#   --out equity_cli_5m.csv

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"   # quantflow root
FACTOR_ROOT="$(cd "$ROOT/../../panda_factor-main" && pwd)" || true
VENV="$FACTOR_ROOT/.venv"

# 默认参数
START=20240101
END=20240101
SYMS="BINANCE:BTCUSDT,BINANCE:ETHUSDT"
RULE=5min
FEE=1
SLIP=2
MAKER=0
TAKER=5
MAXW=0.6
MAXT=0.5
MINTR=0.01
OUT=equity_cli_5m.csv

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START="$2"; shift 2;;
    --end) END="$2"; shift 2;;
    --symbols) SYMS="$2"; shift 2;;
    --resample) RULE="$2"; shift 2;;
    --fee) FEE="$2"; shift 2;;
    --slip) SLIP="$2"; shift 2;;
    --maker) MAKER="$2"; shift 2;;
    --taker) TAKER="$2"; shift 2;;
    --max_weight) MAXW="$2"; shift 2;;
    --max_turnover) MAXT="$2"; shift 2;;
    --min_trade) MINTR="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    *) echo "未知参数: $1"; exit 1;;
  esac
done

if [[ ! -d "$VENV" ]]; then
  echo "[!] 未发现 panda_factor-main venv: $VENV"
  echo "    请先在 策略/panda_factor-main 下创建并安装："
  echo "    python3 -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -e ./panda_common -e ./panda_data_hub -e ./panda_factor -e ./panda_data"
  exit 1
fi

source "$VENV/bin/activate"

# 生成关联ID
RID="RID$(date +%H%M%S)"
BTID="BT$(date +%H%M%S)"

echo "[+] 运行 CLI 回测 (1m→$RULE)"
python "$ROOT/tools/crypto_backtest_cli.py" \
  --symbols "$SYMS" \
  --freq 1m --resample "$RULE" \
  --start "$START" --end "$END" \
  --fee "$FEE" --slip "$SLIP" --maker "$MAKER" --taker "$TAKER" \
  --max_weight "$MAXW" --max_turnover "$MAXT" --min_trade "$MINTR" \
  --req-id "$RID" --bt-id "$BTID" \
  --out "$OUT" -v

echo "[+] 输出曲线: $OUT"
echo "[+] 关联ID: req_id=$RID backtest_id=$BTID"
echo "[√] 完成。"

