# Sprint: Crypto Analysis Enablement 002

范围
- 在已完成的最小生产能力基础上，补充本地可运行演示与脚手架，方便无网络/无交易所账号快速验证；同时提供离线回测 CLI，便于批量评估。

目标
- 可在本地生成/导入合成数据，快速跑通单/多标的回测，输出指标与曲线到 CSV。

任务清单（WBS）
1) 数据工具
   - [x] T1.1 合成数据写入：生成 BTC/ETH 合成 OHLCV（1d/1m）入库 Mongo（`crypto_market`/`crypto_market_1m`）
2) 回测脚手架
   - [x] T2.1 离线回测 CLI：读取 panda_data 数据，调用引擎（含执行约束），输出 metrics 与 equity.csv
3) 文档
   - [x] T3.1 README_CRYPTO 增补：说明合成数据工具与 CLI 用法

验收标准
- 运行合成脚本后，可在 Mongo 中查询两只币（BINANCE:BTCUSDT/ETHUSDT）日线/分钟数据。
- CLI 支持：单/多标的、1d/1m、成本/约束参数；输出 metrics（stdout）与 equity.csv。

当前进度（滚动更新）
- 已完成：T1.1、T2.1、T3.1。
