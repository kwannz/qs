# Panda QuantFlow 开发指南（AGENTS）

本指南适用于 `panda_quantflow-main/` 目录及其子目录（服务、插件与前端资源）。

## 项目概述
- 可视化工作流平台：拖拽式节点，覆盖数据处理、特征工程、机器学习、因子与回测流程。
- 插件系统：通过装饰器 `@work_node` 开发自定义节点，位于 `src/panda_plugins/custom/`。
- 服务：FastAPI 提供 REST/页面；内置图表与工作流 UI。

## 环境与依赖
- Python 3.12+（强制）
- 独立虚拟环境；如需访问 Panda 因子/数据，请先运行 `panda_factor-main` 对应服务与 Mongo。

## 安装与运行
```bash
cd panda_quantflow-main
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .

# 启动服务
python src/panda_server/main.py
# 访问 UI
# 工作流: http://127.0.0.1:8000/quantflow/
# 图 表: http://127.0.0.1:8000/charts/
```

## 自定义插件开发
- 目录：`src/panda_plugins/custom/`
- 基类：`BaseWorkNode`，需实现 `input_model`、`output_model`、`run`
- 示例：见 `src/panda_plugins/custom/examples/`

## 与 Panda Factor 集成
- 若节点需要因子/数据：
  - 确保 `panda_factor-main` 已启动（因子服务与数据清洗定时任务）。
  - 配置一致的数据源与数据库（Mongo）。

## 开发规范
- Python 4 空格缩进；`snake_case`（函数/变量/模块），`CamelCase`（类）。
- 修改范围最小化；新增节点优先放入 `panda_plugins/custom/`。
- 勿提交密钥与数据库快照；敏感信息入 `.env` 或本地配置。

## 排错
- UI 无法打开：检查服务日志与端口冲突；确认 FastAPI 正常启动。
- 数据连接失败：确认 Mongo 连接可达；对齐 `panda_factor-main` 的库与凭据。

