# QS Monorepo AGENTS.md

简体中文在前，English follows.

## 作用范围（Scope）
- 本文件位于仓库根目录，适用于本仓库下的所有子目录与文件。
- 若某子项目目录内存在自己的 `AGENTS.md`（例如 `freqtrade-develop/AGENTS.md`），则以子项目内的规则优先。
- 修改任意文件前，请先检查对应目录是否有更具体的 `AGENTS.md`。

## 仓库总览（Repository Overview）
本仓库是一个量化研究与交易相关的多项目集合（monorepo）：
- `freqtrade-develop/`：开源加密货币量化交易机器人（Python 3.11+，内含完善开发流程与测试）。
- `panda_factor-main/`：PandaAI 因子库与相关服务（FastAPI、MongoDB、数据清洗、可视化、LLM 集成等）。
- `panda_quantflow-main/`：工作流式量化平台（FastAPI，要求 Python 3.12+，插件化节点，包含 UI 接口）。
- `qlib-main/`：微软开源 AI 量化平台 Qlib（包含 Cython/C++ 扩展模块，需编译）。

建议将每个子项目视作独立仓库维护，分别安装依赖、分别运行与测试，避免环境与依赖冲突。

### Freqtrade（定义与定位）
- 执行交易：对接交易所账户，发单/撤单/查询资产与订单。
- 策略驱动：按自定义策略信号（买/卖）自动执行。
- API 连接：通过 ccxt 统一接入各大交易所（Binance、OKX、Kraken...）。

## 运行时与环境（Runtimes & Environments）
- 强烈建议为每个子项目创建独立虚拟环境（不要共用一个 venv/conda 环境）。
- 推荐 Python 版本：
  - `panda_quantflow-main`：`>= 3.12`（见其 `pyproject.toml`）。
  - `freqtrade-develop`：`>= 3.12`（支持 3.11–3.13）。
  - `panda_factor-main`：建议 `>= 3.12`（具体依赖见各 `setup.py`/`requirements.txt`）。
  - `qlib-main`：常见在 3.12 运行；如使用 3.12+，请按官方文档验证兼容性与编译工具链。

示例（macOS/Linux，Zsh/Bash）：
- Freqtrade（开发安装）
  ```bash
  cd freqtrade-develop
  python3.11 -m venv .venv && source .venv/bin/activate
  pip install -U pip
  pip install -e .[dev]
  # 或使用项目脚本：
  ./setup.sh -i
  ```
- Panda QuantFlow（源码安装）
  ```bash
  cd panda_quantflow-main
  python3.12 -m venv .venv && source .venv/bin/activate
  pip install -U pip
  pip install -e .
  ```
- Panda Factor 系列（需 MongoDB，分模块可编辑安装）
  ```bash
  cd panda_factor-main
  python3.11 -m venv .venv && source .venv/bin/activate
  pip install -U pip
  pip install -r requirements.txt
  pip install -e ./panda_common ./panda_factor ./panda_data \
               ./panda_data_hub ./panda_llm ./panda_factor_server
  ```
- Qlib（带 C/C++ 扩展）
  ```bash
  cd qlib-main
  python3.11 -m venv .venv && source .venv/bin/activate
  pip install -U pip
  pip install -e .
  # 如需手动编译：
  # python setup.py build_ext --inplace
  ```

## 外部依赖与服务（External Services）
- MongoDB：`panda_factor-main` 与 `panda_quantflow-main` 依赖 MongoDB 及预置行情数据库。
  - 连接与认证配置位于：`panda_factor-main/panda_common/panda_common/config.yaml`。
  - 若需要快速体验，可按 `panda_factor-main/README.md` 与 `panda_quantflow-main/README.md` 指引下载预置数据并启动服务。
- 前端静态资源：`panda_factor-main/panda_web/panda_web/static` 由 `panda_factor_server` 通过 FastAPI 静态挂载提供。

## 常用启动方式（Run Targets）
- Freqtrade：
  - 测试：`pytest` 或 `pytest --cov=freqtrade`
  - 其他命令见 `freqtrade-develop/AGENTS.md` 与 `README.md`（如 `freqtrade new-config` 等）。
- Panda QuantFlow：
  - 启动服务：`python src/panda_server/main.py`
  - 打开 UI：工作流 `http://127.0.0.1:8000/quantflow/`，图表 `http://127.0.0.1:8000/charts/`
- Panda Factor Server：
  - 启动（方式一）：`python -m panda_factor_server.panda_factor_server`
  - 启动（方式二）：`uvicorn panda_factor_server.__main__:app --host 0.0.0.0 --port 8111`
- Panda LLM 服务：
  - `uvicorn panda_llm.server:app --reload`
- Qlib：
  - 参考官方文档与 `README.md` 示例（数据准备、训练、回测管线）。

## 代码风格与提交（Code Style & Contribution）
- 通用约定：
  - Python 使用 4 空格缩进，`snake_case`（函数/变量/模块），`CamelCase`（类）。
  - 尽量添加类型标注与简洁 docstring；避免引入未使用依赖。
  - 严禁提交密钥、证书、数据库快照等敏感信息到 Git。
- 子项目特定：
  - `freqtrade-develop` 已配置 Ruff/Mypy/Pre-commit，严格遵循其 `AGENTS.md` 与 `pyproject.toml`。
  - 其他子项目如无统一格式化规则，可遵循 Black + Ruff 的默认风格，但不得擅自大范围重排无关代码。
- 提交建议：
  - 小步提交，提交信息用祈使句并聚焦单一变更（如 `Add X`/`Fix Y`/`Refactor Z`）。
  - 若影响行为，请补充/调整相应测试与文档。

## 工作流与协作（Agent Workflow）
- 在编辑任何文件前：
  1) 先确认对应目录是否存在更具体的 `AGENTS.md`；
  2) 读取 `pyproject.toml`、`setup.cfg`、`requirements*.txt` 等了解依赖与工具链；
  3) 若涉及服务交互，核对外部依赖（MongoDB、端口占用等）。
- 修改范围最小化：
  - 仅在必要范围内改动；避免跨子项目的联动重构，除非明确需要且已评估影响。
  - 不调整文件/目录结构与命名，除非有清晰一致性的改进并获准。
- 测试优先：
  - 以子项目为单位运行、补充测试；优先覆盖变更附近逻辑。
  - 避免真实网络访问，使用 mock/fixture。

## 常见问题（Troubleshooting）
- 依赖冲突：每个子项目使用独立虚拟环境，必要时 pin 版本并记录在相应 `requirements*.txt`。
- Qlib 编译失败：检查本地编译工具链（C/C++/Cython、`numpy` 头文件），尝试 `python setup.py build_ext --inplace`。
- 静态资源无法访问：确保 `panda_factor_server` 启动时挂载到 `panda_web` 的 static 目录路径存在且可读。
- 无法连接 MongoDB：检查 `config.yaml` 连接串与访问凭据，确认数据库副本集/权限已配置。

## 策略 目录与加密货币支持（Strategy Folder & Crypto Support）

本仓库在根目录下包含 `策略/` 子目录，内含三套研究/回测子项目（与根部的同名项目相互独立）：

- `策略/panda_factor-main/`：因子库与数据访问层（MongoDB）。
- `策略/panda_quantflow-main/`：可视化工作流（节点化）与回测/分析服务。
- `策略/qlib-main/`：Qlib 平台源码（需按官方文档编译/初始化）。

默认配置面向 A 股/期货数据。若需要支持“加密货币（Crypto）”，建议按下述最小改造路径实施：

1) 因子与数据（panda_factor-main）
- 数据表：在 MongoDB 新建日线集合 `crypto_market`（字段建议：`date(YYYYMMDD)`、`symbol`、`open`、`high`、`low`、`close`、`volume`，可选 `quote_volume`、`trades`）。
- 读取器：已添加最小读写器 `panda_data/panda_data/market_data/market_crypto_reader.py`；统一入口 `panda_data.get_market_data(..., type='crypto')` 或 `panda_data.get_crypto_market_data(...)`。
- 分钟级（可选）：如有 1m/5m 数据，可仿照 `market_stock_cn_minute_reader.py` 使用/扩展 `MarketCryptoMinReader`（按 `datetime` UTC 查询，24/7 无交易日历），默认读取集合名：`crypto_market_1m`。
- 符号规范：建议 `EXCHANGE:BASEQUOTE`（例如 `BINANCE:BTCUSDT`），以便多交易所并存。

2) 工作流与回测（panda_quantflow-main）
- 节点：已添加“加密回测”占位节点 `src/panda_plugins/internal/backtest_crypto_node.py`，用于 UI 与参数打通；需对接实际回测引擎（撮合、手续费、最小变动、24/7 时钟等）后方可产出真实结果。
- 引擎接入：若已有通用回测库，可在该节点内调用；或复用现有股票/期货回测框架的撮合层进行扩展。

3) Qlib（qlib-main）
- Qlib 原生聚焦股票/期货。若要用于 Crypto，可按 Qlib 的自定义数据接口准备器导入 OHLCV，并选择 24/7 日历（或自定义 market calendar）。该部分不在本仓库直接实现，请遵循 Qlib 官方文档与示例。

4) 实盘交易（建议）
- 本仓库未在上述“策略/”子项目内直接集成交易所连接。推荐使用根目录的 `freqtrade-develop/`（基于 ccxt，支持 Binance/OKX/KuCoin 等）执行实盘，将 `panda_factor-main` 输出的信号/权重对接为策略输入。

5) 配置与运行要点
- 配置文件：`策略/panda_factor-main/panda_common/panda_common/config.yaml`（或 `.env` 环境变量）中设置 Mongo 连接。
- 数据写入：先将加密货币 OHLCV 导入 `crypto_market`，再调用 `panda_data.get_market_data(..., type='crypto')` 验证。
- 日历：Crypto 24/7，无需 `pandas_market_calendars`；涉及自动调度请单独为加密数据通道配置时间窗口。

---

# QS Monorepo AGENTS.md (EN)

## Scope
- This file lives at the repo root and applies to all files under it.
- If a subproject contains its own `AGENTS.md` (e.g. `freqtrade-develop/AGENTS.md`), that file takes precedence for files under that sub-tree.
- Always check for a nested `AGENTS.md` before editing.

## Layout
- `freqtrade-develop/`: Crypto trading bot with mature dev tooling (Py 3.11+).
- `panda_factor-main/`: Factor library and services (FastAPI, MongoDB, data cleaning, LLM).
- `panda_quantflow-main/`: Visual, node-based quant workflow platform (Py 3.12+).
- `qlib-main/`: Microsoft Qlib with C/C++ extensions.

Treat each subproject as an independent repo for env/deps/testing.

## Runtimes & Environments
- Create a dedicated virtualenv per subproject; do not share envs.
- Suggested Python versions:
  - `panda_quantflow-main`: 3.12+
  - `freqtrade-develop`: 3.11–3.13
  - `panda_factor-main`: 3.10+ recommended
  - `qlib-main`: commonly 3.8–3.11; validate for 3.12+

## External Services
- MongoDB required by PandaFactor/QuantFlow; configure `panda_factor-main/panda_common/panda_common/config.yaml`.
- Static UI under `panda_web` is mounted by `panda_factor_server`.

## Run Targets
- Freqtrade: see `freqtrade-develop/AGENTS.md` and README for setup, tests, linting, and CLI usage.
- QuantFlow: `python src/panda_server/main.py` then open `http://127.0.0.1:8000/quantflow/`.
- Panda Factor Server: `python -m panda_factor_server.panda_factor_server` or `uvicorn panda_factor_server.__main__:app`.
- Panda LLM: `uvicorn panda_llm.server:app --reload`.
- Qlib: follow upstream README for data prep, training, and backtesting.

## Style & Contribution
- 4-space indent; snake_case for modules/functions/vars; CamelCase for classes.
- Add type hints and clear docstrings where practical.
- Never commit secrets or large binary data.
- Follow subproject-specific tooling (Ruff/Mypy/Pre-commit in `freqtrade-develop`).

## Workflow
- Read nested `AGENTS.md` and config files before changing code.
- Keep diffs minimal and scoped to a single subproject.
- Write/adjust tests alongside behavior changes.

### Strategy Folder & Crypto Support (EN)
- The repo also contains `策略/` (strategy) with three subprojects:
  - `策略/panda_factor-main/`: factor lib and Mongo-backed data access.
  - `策略/panda_quantflow-main/`: node-based workflow + backtest/analysis services.
  - `策略/qlib-main/`: Qlib source (build/init per upstream).
- To enable crypto:
  1) Data/Factors: create Mongo `crypto_market` (daily OHLCV). Use the added reader `market_crypto_reader.py` via `panda_data.get_market_data(..., type='crypto')` or `get_crypto_market_data(...)`. For minutes, use/extend `MarketCryptoMinReader` (UTC datetime, 24/7), default collection: `crypto_market_1m`.
  2) Workflow/Backtest: a placeholder node `backtest_crypto_node.py` is added; wire to your crypto backtesting engine to get real results.
  3) Qlib: import OHLCV via custom data provider and use a 24/7 calendar (follow upstream docs).
  4) Live trading: use root `freqtrade-develop/` (ccxt) to execute; feed signals from `panda_factor-main`.
  5) Config: set Mongo in `策略/panda_factor-main/panda_common/panda_common/config.yaml` or via `.env`; load data first, then query with the new API.

以上规范旨在帮助在多项目环境中高效、安全地协作与迭代。
