# 交易所接入迁移计划（方案B）

目标：将 `freqtrade` 的交易所接入层迁移为 `qs` 内部模块，在不引入整个 freqtrade 的前提下，复用其成熟的 Exchange 抽象、交易所子类与工具，支持 Spot/Futures、行情、下单、WebSocket 等能力。

范围：仅迁移 `exchange` 层及其必要依赖（tools、types、enums、utils），移除与策略、持久化、UI 等非必要耦合。

里程碑与步骤

1. 目录骨架与依赖列表
   - 创建 `qs/exchanges/freqtrade_port/` 作为迁移命名空间。
   - 子目录：
     - `exchange/`：`exchange.py`、各交易所子类（binance/bybit/okx/gate/kraken/htx/hyperliquid/bitget/...）及 `exchange_ws.py`、`exchange_utils*.py`、`common.py`、`exchange_types.py`、`binance_public_data.py`、`binance_leverage_tiers.json`。
     - `enums/`：`candletype.py`、`tradingmode.py`、`marginmode.py`、`pricetype.py`、`runmode.py`（最小集）。
     - `util/`：`datetime_helpers.py`、`periodic_cache.py`、`ft_precise.py`。
     - `configuration/`：`config_secrets.py`（dry-run 清理密钥）。
     - `data/converter/`：`converter.py`、`trade_converter.py`、`orderflow.py` 与 `__init__.py`（OHLCV/Trades 转换）。
     - `misc.py`、`constants.py`、`exceptions.py`（最小集）。

2. 复制与命名空间改写
   - 从 `freqtrade-develop/freqtrade/` 复制上述文件到 `qs/exchanges/freqtrade_port/` 对应位置。
   - 全量替换导入：`from freqtrade.` → `from qs.exchanges.freqtrade_port.`。
   - 移除/改写对 `freqtrade.persistence`、`strategy` 等非必需模块的依赖。

3. 最小可运行闭环
   - 在 `qs` 中提供 `ExchangeResolver` 门面，按 name 实例化交易所类（兼容 `MAP_EXCHANGE_CHILDCLASS`）。
   - 在示例脚本中验证：`load_markets`、`fetch_ticker(s)`、`fetch_ohlcv`、`create/cancel order`（测试网）、`watch_ohlcv`。

4. 覆盖交易所与功能
   - 第一批：Binance、Bybit、OKX、Gate（Spot + Futures + WS）。
   - 第二批：Kraken、HTX、Hyperliquid、Bitget 等。

5. 质量与兼容
   - 保留 freqtrade 的退避与重试（`retrier`/`retrier_async`）。
   - 精度与单位：沿用 `exchange_utils.py`（价格/数量精度、合约数量换算）。
   - 时间与缓存：沿用 `util/datetime_helpers.py`、`PeriodicCache`。
   - 配置：保留 `config_secrets.remove_exchange_credentials` 行为用于 dry-run。

6. 许可与合规
   - `freqtrade` 为 GPLv3；迁移代码需在本仓库中保留原版权与许可说明，且对外分发时符合 GPLv3 要求。

7. 后续演进
   - 统一异常模型与返回结构，压缩 API 面，输出 `QSExchange` 兼容层。
   - 按需扩展交易所子类 `_ft_has` 特性与 WS 订阅类型。

验收标准（每阶段）
 - 阶段1：目录与文件映射表、依赖清单完成；`import` 通过静态检查。
 - 阶段2：本地跑通 `load_markets`、`fetch_ohlcv`、`get_tickers`（Spot）。
 - 阶段3：下单/撤单（测试网）与 WS K线可用；Futures 头寸与 funding 读取可用。
 - 阶段4：新增交易所子类成功加载与运行。

外部依赖（最低）
 - `ccxt`, `ccxt.pro`, `pandas`, `cachetools`, `python-dateutil`, `rapidjson`, `humanize`。

项目内规范
 - 迁移代码置于 `qs/exchanges/freqtrade_port/`，避免污染其他命名空间。
 - 不修改无关模块；保持最小化适配与清晰的替换导入。

