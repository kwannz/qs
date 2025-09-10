# Panda Factor — Crypto 快速指引

本指引帮助你在 `策略/panda_factor-main` 下快速完成：数据导入 → 查询/重采样 → 与工作流回测对接。

## 1. 环境与依赖
- 独立虚拟环境（参考仓库 AGENTS.md）
- 安装：`pip install -e .`（本子项目）
- 额外依赖：`pip install ccxt pandas matplotlib pymongo`
- 配置 Mongo：`panda_common/panda_common/config.yaml`

## 2. 导入加密行情（CCXT）
工具脚本：`tools/crypto_ccxt_ingest.py`

示例（BINANCE 日线）：
```
python tools/crypto_ccxt_ingest.py \
  --exchange binance \
  --symbols BTC/USDT,ETH/USDT \
  --timeframe 1d \
  --start 20240101 --end 20240201
```

示例（BINANCE 1 分钟）：
```
python tools/crypto_ccxt_ingest.py \
  --exchange binance \
  --symbols BTC/USDT \
  --timeframe 1m \
  --start 20240101 --end 20240102
```

约定：
- 日线集合：`crypto_market`（字段：`date` YYYYMMDD，`symbol`，`open/high/low/close/volume`）
- 分钟集合：`crypto_market_1m`（字段：`datetime` UTC，`symbol`，OHLCV）
- 符号：`EXCHANGE:BASEQUOTE`，如 `BINANCE:BTCUSDT`

## 3. 查询与重采样（panda_data）
```
import panda_data
panda_data.init()

# 日线
df_day = panda_data.get_crypto_market_data(
    start_date='20240101', end_date='20240110',
    symbols=['BINANCE:BTCUSDT'],
    fields=['date','symbol','open','high','low','close','volume']
)

# 1 分钟
df_1m = panda_data.get_crypto_min_data(
    start_date='20240101', end_date='20240102',
    symbol='BINANCE:BTCUSDT',
    fields=['datetime','symbol','open','high','low','close','volume']
)

# 1 分钟 → 5 分钟重采样
df_5m = panda_data.get_crypto_min_data_resampled(
    start_date='20240101', end_date='20240102',
    symbol='BINANCE:BTCUSDT', rule='5min'
)

# 列出已入库符号
symbols = panda_data.get_crypto_instruments()
```

> Notebook 示例：`notebooks/Crypto_Quickstart.ipynb`

## 4. 回测与工作流
- 启动 QuantFlow 服务：`策略/panda_quantflow-main/src/panda_server/main.py`
- 节点：`加密回测`（支持 1d/1m、多标的等权/信号等权、手续费）
- 现支持：
  - 单标的：`base_symbol`
  - 多标的：`symbols` 逗号分隔
  - 信号输入：因子中包含 `signal` 列（0/1）即可生效
  - 成本参数：`fee_rate`（基础费用bps）、`slippage_bps`（滑点bps）、`maker_bps`、`taker_bps`
- 执行约束（可选）：
  - `max_weight`（单标的最大权重，默认1.0 不限）
  - `max_turnover`（每期最大换手 L1，默认1.0）
  - `min_trade_weight`（最小交易权重阈值，过滤微小调仓）
- 后续将加入：滑点、Maker/Taker 费率、仓位与风控参数

## 5. 常见问题
- 分钟数据时区：存储为 UTC；查询/重采样内部按 UTC 处理。
- 索引：脚本会尝试创建唯一索引 (symbol+date)/(symbol+datetime)，如重复需清理数据。
- 交易对名：各交易所差异较大，脚本含模糊匹配；若失败请核对 `exchange.symbols`。

## 6. 因子（基础版）
- 直接通过 `panda_data.get_factor(..., type='crypto')` 获取：
```
import panda_data
panda_data.init()

factors = panda_data.get_factor(
    factors=['ret_1d','ret_5d','ret_20d','vol_20','mom_20','rsi14','macd','macd_signal','macd_hist','hhv_20','llv_20'],
    start_date='20240101', end_date='20240201',
    symbols=['BINANCE:BTCUSDT'], type='crypto'
)
```
- 若仅需基础字段（OHLCV）：传入 `['open','high','low','close','volume']` 即可直接返回日线数据。

## 7. 本地快速演示（无网络/账号）
- 合成数据写入：
```
python tools/crypto_synthetic_ingest.py --timeframe 1d --start 20240101 --end 20240110
python tools/crypto_synthetic_ingest.py --timeframe 1m --start 20240101 --end 20240102
```
- 开启详细日志：设置环境变量 `LOG_LEVEL=DEBUG` 或在 CCXT 脚本使用 `-v/--verbose`
- 日志格式与输出：
  - 控制台/文件由环境控制：`LOG_CONSOLE=1`、`LOG_FILE=1`、`LOG_DIR=logs`
  - 级别：`LOG_LEVEL=DEBUG|INFO|WARNING|ERROR`
  - 格式：`LOG_FORMAT=plain|json`（需安装 python-json-logger）
  - 轮转：`LOG_ROTATE_SIZE=1048576`（字节），`LOG_BACKUP_COUNT=5`
- 离线回测 CLI：
```
python ../panda_quantflow-main/tools/crypto_backtest_cli.py \
  --symbols BINANCE:BTCUSDT,BINANCE:ETHUSDT \
  --freq 1d --start 20240101 --end 20240110 \
  --fee 1 --slip 2 --maker 0 --taker 5 \
  --max_weight 0.6 --max_turnover 0.5 --min_trade 0.01 \
  --out equity.csv
```
> 提示：在 UI 工作流中运行“加密回测”节点可查看 `metrics` 与 `equity_preview`；详细日志可在服务端设置 `LOG_LEVEL=DEBUG` 查看加载与查询细节。
