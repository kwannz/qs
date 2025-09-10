# Sprint: Qlib Futures (Long/Short) Backtest — End-to-End (2.5–3.5 weeks)

## 目标
- 为 Qlib 增加适配“期货/永续”场景的回测内核：支持做多/做空、杠杆/保证金、资金费、最小变动/交易单位、强平逻辑、24/7 日历（Crypto）。
- 与策略/QuantFlow 和策略/Panda Factor 打通：新增 Qlib 期货回测节点与 CLI；可从 Mongo/DF 拉取行情并落地指标/曲线。
- 提供完整文档、示例与一键脚本。

## 产出物（Deliverables）
- Qlib 扩展
  - 新增 `FuturesExchange`（含 maker/taker/slippage/强平/资金费/IM&MM/contract_multiplier/trade_unit）
  - 新增 `FuturesPosition`（正负仓/均价/名义/权益/保证金/未实现盈亏/资金费结算）
  - 可配置的 backtest config（支持 24/7 日历）
- 数据与日历
  - `panda_provider_futures.py`（从 Panda Data/Mongo 读数据，支持 1m/5m/1h/1d）
  - 24/7 Calendar 实现或适配
- 集成：QuantFlow & CLI
  - 新节点：`qlib_futures_backtest_node`（输入参数与回测内核对齐；输出 metrics + equity_preview）
  - 新 CLI：`qlib_futures_backtest_cli.py`（参数同节点；导出 CSV）
- 可观测性
  - 关键事件日志（rid/wf/bt 贯通）：开/平/强平/资金费结算/保证金校验
  - UI 简易日志页可按 backtest_id/req_id 过滤查询
- 文档与示例
  - `README_FUTURES.md`（原理/参数/注意事项）
  - 示例工作流 JSON（1d/5m）与一键脚本

## 范围与文件（拟）
- Qlib（策略/qlib-main）
  - `qlib/backtest/exchange_futures.py`：FuturesExchange（新增）
  - `qlib/backtest/position_futures.py`：FuturesPosition（新增）
  - `qlib/backtest/decision.py`：可选扩展 reduce-only/close-only（小改）
  - `qlib/backtest/executor.py`：允许注入 FuturesExchange（小改）
  - `qlib/data/providers/panda_provider_futures.py`：数据适配器（新增）
  - `qlib/utils/calendar_24_7.py`：24/7 日历（新增或适配）
  - tests：合成数据的核心用例
- QuantFlow（策略/panda_quantflow-main）
  - `src/panda_plugins/internal/qlib_futures_backtest_node.py`（新增）
  - `tools/qlib_futures_backtest_cli.py`（新增）
  - `examples/workflows/qlib_futures_backtest_5m.json`（新增）
  - `scripts/run_qlib_futures_backtest.sh`（新增）
- Panda Factor（策略/panda_factor-main）
  - 若需要，新增 futures 合约配置（乘数/单位/价格精度）
  - 文档更新：如何配置信息源与合约参数

## 详细设计（要点）
- FuturesPosition
  - 字段：pos（正多负空）、avg_price、contract_multiplier、min_qty、tick_size、cash、equity、IM/MM、unrealized PnL、fee_accum、funding_accum
  - 方法：update_order（开/平处理）、apply_funding（按周期计费）、check_liquidation（强平判定）
- FuturesExchange
  - deal_order(order)：计算成交价/滑点/费用；更新 Position（或 TradeAccount 的 current_position）
  - 费用：开平 maker/taker（bps）、滑点；可选最小费用
  - 杠杆与保证金：通过 leverage 或 IM/MM 计算可开仓量；达到维持保证金阈值时触发强平
  - 资金费：funding_rate 可配置（固定或外部曲线），按周期计费
  - 24/7：支持无交易日历下的连续时间索引
- Backtest Config（示例）
  - exchange_kwargs：`{"type": "futures", "leverage":10, "im":0.1, "mm":0.05, "funding_period":"8h", "funding_rate":0.0001, "maker":0.0, "taker":0.0005, "slippage":0.0002, "trade_unit":0.001, "contract_multiplier":1}`
- Provider
  - 统一读取 `BINANCE:BTCUSDT` 永续行情（来自 Mongo 或 DF），字段齐备（date/datetime/open/high/low/close/volume）
  - 重采样支持在 Provider 或外层转换

## 接口（QuantFlow 节点/CLI）
- 输入
  - symbols（多标的）、start/end、freq/resample
  - leverage / im/mm（两种方式二选一）、funding_period/funding_rate
  - maker/taker/slippage、trade_unit/contract_multiplier、min_notional
  - 风控：stop_loss、max_drawdown
  - 资金：start_capital
- 输出
  - metrics：年化、波动、夏普、最大回撤、胜率、换手
  - equity_preview：最多 500 点（前端绘图）
  - 可选明细路径：成交/资金费/强平事件 CSV

## 计划排期
- Week 1（研发内核：FuturesPosition/FuturesExchange + 单测）
  1) 设计校对与 PoC 用例
  2) 实现 FuturesPosition（方向/PnL/保证金/资金费/强平）
  3) 实现 FuturesExchange（费用/滑点/最小单位/可开仓约束/强平触发/资金费计费钩子）
  4) executor 兼容小改；decision 支撑（如需 reduce-only）
  5) 单测（合成数据）：多空/费用/强平/资金费
- Week 2（数据+集成+CLI/节点）
  6) Provider（panda_provider_futures）
  7) 24/7 日历
  8) CLI：qlib_futures_backtest_cli（导出 CSV + 日志）
  9) QFlow 节点：qlib_futures_backtest_node（输入参数对齐、输出 metrics/曲线、日志贯通）
- Week 3（完善与文档）
  10) 示例工作流 JSON + 一键脚本
  11) README_FUTURES.md + 使用指南
  12) 性能/边界测试、Bug 修复、接口稳定化
  13) 评审/交付与后续优化建议（可选 PR 给 upstream）

## 验收标准
- 功能正确：多空开平、PnL/权益/保证金计算、资金费结算、强平触发均符合预期
- 配置全面：leverage 或 IM/MM、maker/taker/slippage、trade_unit/contract_multiplier
- 数据无缝：可使用 Mongo+PandaData 或 DF 输入，支持 24/7 分钟数据
- 端到端：QuantFlow 节点/CLI 可跑，输出 metrics/equity，日志可按 backtest_id 检索
- 文档齐备：原理、参数、示例、注意事项与常见问题

## 风险与缓解
- 资金费/强平模型差异：以常见交易所（Binance/OKX）为基准，提供可配置参数与扩展点
- 与 Qlib 内核低侵入：尽量新增文件/类；必要修改集中在 executor 的注入与 decision 的小扩展
- 数据边界：分钟数据的缺口/异常处理；提供兜底与日志

## 需要你的确认
- 合约参数来源：是否希望在 `panda_factor-main` 配置文件中维护乘数/单位/精度（建议）
- 优先市场：先做 Crypto 永续；后续可扩展商品/股指期货
- 资金费模型：先做定额/常数版，再接入外部 funding 曲线（可在第二阶段加入）

