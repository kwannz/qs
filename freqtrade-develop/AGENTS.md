# Repository Guidelines（仓库指南）

Concise bilingual guide for contributors. 面向贡献者的中英双语速览指南。

## Project Structure & Module Organization（项目结构与模块组织）
- Source: `freqtrade/` (CLI `freqtrade.main:main`, strategies, exchange, RPC, templates) — 源码目录，含 CLI 入口、策略、交易所、RPC、模板。
- Tests: `tests/` (pytest; `test_*.py`) — 测试目录，pytest，文件命名 `test_*.py`。
- Docs: `docs/` (mkdocs; `mkdocs.yml`) — 文档与站点配置。
- Client/UI: `ft_client/` — 前端与 UI 资源。
- Examples/User space: `config_examples/`, `user_data/` — 示例与用户空间（勿提交敏感信息）。
- Helpers: `build_helpers/`, `scripts/` — 辅助脚本与工具。

## Build, Test, and Development Commands（构建、测试与开发）
- Setup: `./setup.sh -i` — 引导系统依赖，创建 `.venv`，安装依赖与 freqUI，启用 pre-commit。
- Dev install: `python3.11 -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -e .[dev]` — 手动开发安装。
- Tests: `pytest`（例：`pytest tests/test_<file>.py::test_<name>`）；Coverage：`pytest --cov=freqtrade` — 运行单测与覆盖率。
- Lint/Format: `ruff check .`；`ruff format .`（必要时 `isort .`）— 静态检查与格式化。
- Types/Pre-commit: `mypy freqtrade`；`pre-commit install && pre-commit run -a` — 类型检查与本地钩子。

## Coding Style & Naming Conventions（编码风格与命名）
- Python 3.11+; 4 spaces; max line length 100 — 使用类型标注，公共函数写 reST 风格 docstring。
- Imports follow isort/Black; keep 2 blank lines after imports — 导入遵循 isort/Black，导入后空两行。
- Names: files/modules `snake_case.py`; classes `CamelCase`; funcs/vars `snake_case` — 统一命名约定。
- Follow Ruff rules in `pyproject.toml` (Bugbear, Pyflakes, Pycodestyle, PyUpgrade, Bandit subset, etc.) — 遵循已配置的 Ruff 规则。

## Testing Guidelines（测试指南）
- Pytest with xdist and `pytest-asyncio` — 支持并发与异步测试。
- Mirror package structure in `tests/`; name functions `test_<behavior>()` — 结构对应，命名清晰。
- Prefer fixtures/fakes over network calls — 使用 fixture/假对象，避免真实网络调用。

## Commit & Pull Request Guidelines（提交与 PR 指南）
- Target `develop` (not `stable`) — 目标分支为 `develop`。
- Keep PRs focused; include tests/docs for features — PR 聚焦单一主题，功能需配套测试与文档。
- Commit messages in imperative; link issues (e.g., `Fixes #123`) — 提交信息用祈使句并关联 Issue。
- PR includes description, rationale/repro (for bugs), linked issues, screenshots/logs for UI/RPC — 补充必要说明与证据。

## Security & Configuration Tips（安全与配置）
- Never commit secrets; generate config: `freqtrade new-config -c user_data/config.json` — 切勿提交密钥，使用命令生成配置。
- JSON changes must pass schema via pre-commit — JSON 需通过预提交的架构校验。
