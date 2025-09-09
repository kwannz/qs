# 研究/信号与 Freqtrade 实盘解耦方案（加密货币）

## 目标与范围
- 目标：在“策略”子项目中完成研究/回测/信号产出；由 Freqtrade 独立进程执行实盘下单，二者解耦，仅通过统一信号接口对接。
- 范围：
  - 加密货币数据接入（1d/1m）
  - 因子/信号产出与工作流回测（QuantFlow）
  - 信号持久化（Mongo 集合）与 Freqtrade 读取执行
  - 监控、运维与安全
- 不做：直接改造 Freqtrade 内核或在研究侧直接下单。

## 架构与职责划分
- 策略/panda_factor-main（数据/因子/信号计算）
  - 从 Mongo 读取加密行情（日线 crypto_market；分钟 crypto_market_1m）
  - 进行因子计算、策略逻辑，生成信号，并写入信号集合
- 策略/panda_quantflow-main（工作流/回测/导出）
  - 节点化编排：因子构建 → 策略逻辑/信号 → 回测 → 信号导出（写 Mongo）
  - 回测节点：支持 1d/1m，使用外部 0/1 仓位信号或默认 Buy&Hold
- freqtrade-develop（实盘执行）
  - 通过 ccxt 对接交易所账户、风险控制与撮合
  - 自定义策略读取“外部信号集合”，生成 enter/exit 信号并下单

## 数据与接口规范（MongoDB）
- 行情集合：crypto_market（1d）
  - 字段：date(YYYYMMDD, str), symbol(str, 例如 BINANCE:BTCUSDT), open/high/low/close/volume(float)
  - 索引：{symbol:1,date:1}
- 行情集合：crypto_market_1m（1m）
  - 字段：datetime(datetime, UTC), symbol(str), open/high/low/close/volume(float)
  - 索引：{symbol:1,datetime:1}
- 信号集合：crypto_signals（研究输出 → 实盘消费）
  - 必需：
    - ts(datetime, UTC)
    - timeframe('1m'|'1d')
    - symbol('BINANCE:BTCUSDT')
    - pair('BTC/USDT')
    - action('enter_long'|'exit_long') 或 score/weight(float)
    - strategy_id(str)
    - valid_until(datetime, 可选)
  - 建议：note(str), source(str), version(int)
  - 索引：
    - 查询索引：{strategy_id:1, pair:1, timeframe:1, ts:1}
    - 唯一键：{strategy_id:1, pair:1, timeframe:1, ts:1}（幂等/去重）
    - TTL（可选）：{ts:1}

## 环境与配置
- 虚拟环境：各子项目独立（不要共享）
  - panda_factor-main: Python 3.11（建议）
  - panda_quantflow-main: Python 3.12+
  - freqtrade-develop: Python 3.11+
- 配置：
  - Mongo：策略/panda_factor-main/panda_common/panda_common/config.yaml 或 .env
  - Freqtrade：.env 或 config.json 注入 Mongo 读取参数（例如 EXT_SIG_MONGO_URI/DB/COLLECTION/STRATEGY_ID）

## 实施路线图
- 阶段一：数据与读取器（已完成 70%）
  1) [已完成] 日线读取器 market_crypto_reader.py；API：get_market_data(..., type='crypto') / get_crypto_market_data(...)
  2) [已完成] 分钟读取器 market_crypto_minute_reader.py；API：get_crypto_min_data(...)
  3) [进行中] 数据导入脚本与规范（将交易所 OHLCV 入库到 crypto_market / crypto_market_1m）
- 阶段二：回测与信号产出
  4) [已完成] 加密回测节点（1d/1m），简易引擎 simulate_long_flat
  5) [新增] 信号导出节点（QuantFlow）：将 DataFrame(signal) 写入 crypto_signals（幂等 upsert、去重、可选 TTL）
  6) [可选] 策略节点：因子 → 打分 → 生成 0/1 或权重信号
- 阶段三：Freqtrade 实盘对接（独立运行）
  7) [新增] ExternalSignalStrategy 示例（user_data/strategies）：
     - 连接 Mongo，按 timeframe/pair/strategy_id 读取最新信号
     - 将 enter/exit 映射到 DataFrame 列（enter_long/exit_long）
     - 容错：时间窗口匹配、去重、防抖
  8) [配置] Freqtrade config.json：exchange/pairs/timeframe/strategy/风险参数
  9) [联测] 研究端推送模拟信号 → Freqtrade dry-run/tapi 模式回放
- 阶段四：监控与运维
  10) 日志/健康检查：连通性、信号延迟、成交与持仓对账
  11) 告警：信号缺失、连续拒单、滑点异常、余额不足

## 详细任务清单
- panda_factor-main（研究数据侧）
  - 提供数据导入脚本/说明（日线/分钟）
  - 封装符号映射（BINANCE:BTCUSDT ↔ BTC/USDT）
  - 常见技术因子模板与加密特定指标（Funding/基差/深度）
- panda_quantflow-main（工作流侧）
  - backtest_crypto_node.py（已交付）
  - crypto_backtest_engine.py（已交付）
  - signal_export_node.py（新增）
  - 可选：symbol 映射节点、数据合流节点
- freqtrade-develop（执行侧）
  - ExternalSignalStrategy.py（新增示例）
  - 示例 .env 与 config.json（dry-run/实盘）
  - 可选：成交/持仓回写 Mongo（用于监控与报表）

## 接口与对齐细则
- 符号映射：
  - 策略 symbol：EXCHANGE:BASEQUOTE（例 BINANCE:BTCUSDT）
  - Freqtrade pair：BASE/QUOTE（例 BTC/USDT）
  - 在导出/消费两端各提供映射函数，统一处维护
- 时间对齐：
  - 1d：date 为自然日，内部按 00:00 UTC 对齐；Freqtrade 使用相同 timeframe
  - 1m：datetime 为 UTC；Freqtrade 读取信号时设置 ±1 个 bar 的容忍窗口
- 幂等与去重：
  - 唯一键（strategy_id, pair, timeframe, ts），采用 upsert 防重复
  - 新信号覆盖旧信号（同唯一键）
- 延迟控制：
  - 导出延迟 ≤ timeFrame/2；Freqtrade 读取设置最大容忍延迟（如 3×timeframe）

## 测试与验收
- 单元测试：
  - 读取器：边界日期、空结果、大批量
  - 回测引擎：信号对齐、手续费、持仓切换、收益曲线
- 集成测试：
  - 工作流：因子 → 信号 → 回测 → 导出 → Mongo 校验
  - Freqtrade：读取同一信号，dry-run 验证下单逻辑
- UAT/演练：
  - 小资金/纸交易运行 2–3 天，验证延迟、成交、对账、风控
- 验收标准：
  - 数据/信号/下单全链路无报错
  - 信号生成与消费时间差在阈值内
  - Freqtrade 订单、持仓、资金曲线稳定且符合预期

## 运行与部署流程
- 研究侧（独立）
  1) 创建/激活 venv，安装依赖（-e 安装各模块）
  2) 配置 Mongo（config.yaml 或 .env）
  3) 启动 QuantFlow：python src/panda_server/main.py → /quantflow UI
  4) 编排并运行：因子 → 信号 → 回测 → 信号导出
- 实盘侧（独立）
  5) venv + pip install -e .[dev]
  6) freqtrade new-config → 配置交易所/API/pairs/timeframe/策略
  7) 放入 ExternalSignalStrategy.py → trade/dry-run 启动
  8) 监控日志/订单/持仓

## 监控与告警
- 研究侧：导出节点成功率、写入延迟、信号量
- 实盘侧：订单失败、连续拒单、滑点>阈值、余额不足、API 限频
- 告警通道：邮件/IM（含策略ID、pair、时间）

## 安全与合规
- 秘钥：Freqtrade 使用 .env 或密钥管理（绝不入库/入 Git）
- Mongo 权限：最小权限账户；策略库与信号库分库/分角色
- 日志脱敏：屏蔽密钥、订单敏感字段
- 审计：记录策略版本、导出人、回放可追溯

## 风险与缓解
- 数据延迟/缺失：多源冗余、导出重试、空窗告警
- 时间对齐误差：统一 UTC；读取端设置容忍窗口
- 交易所限频：Freqtrade 节流配置与退避重试
- 信号抖动：最小持仓保持/冷却时间、阈值过滤

## 时间计划（示例 5–10 天）
- D1–D2：数据导入脚本与规范、读取器联调（已完成部分）
- D3：QuantFlow 信号导出节点、端到端导出验证
- D4：Freqtrade ExternalSignalStrategy 实现与 dry-run 连通
- D5：回测参数/规则打磨、联测验收
- D6–D7：UAT（纸交易/小资金）、监控与告警上线

## 交付物清单
- 代码与节点：
  - 读取器与 API：market_crypto_reader、market_crypto_minute_reader、get_crypto_market_data/get_crypto_min_data（已交付）
  - 回测：backtest_crypto_node.py、crypto_backtest_engine.py（已交付）
  - 新增：signal_export_node.py（待实现）
  - Freqtrade：ExternalSignalStrategy.py（待实现）
- 文档与配置：
  - AGENTS.md 已补充“策略/ 与加密支持”
  - 示例 .env / config.json（待补充）
  - Mongo 集合与索引说明（本计划文档）

## 下一步
- 新增 QuantFlow “信号导出”节点（写入 crypto_signals，包含幂等/去重）
- 在 freqtrade-develop 提供 ExternalSignalStrategy 示例与配置模板
- 准备联测脚本与最小演示数据，跑通端到端流程

