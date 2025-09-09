# Panda Factor 系列开发指南（AGENTS）

本指南适用于 `panda_factor-main/` 目录及其子目录，包括：`panda_common/`、`panda_factor/`、`panda_data/`、`panda_data_hub/`、`panda_llm/`、`panda_factor_server/`、`panda_web/`。

## 目标与组件
- 因子库与计算：Python/公式两种模式，内置大量时间序列与技术指标算子。
- 数据中心（DataHub）：对接 Tushare/RiceQuant/XT/TQSDK 等，定时清洗入库。
- 分析与可视化：IC/IR、IC 衰减/密度/序列、分组收益、超额收益等图表数据。
- 服务与前端：FastAPI 服务（含因子 CRUD/运行/查询），静态前端 `/factor`；内置 `/llm` 聊天接口（SSE）。

## 环境准备
- Python 3.11+（推荐）
- 本地 MongoDB（或使用团队提供的预置库）
- 每个子项目使用独立虚拟环境（venv/conda），避免依赖冲突。

## 安装与启动
1) 创建虚拟环境并安装依赖
```bash
cd panda_factor-main
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# 可编辑安装各模块
pip install -e ./panda_common ./panda_factor ./panda_data \
             ./panda_data_hub ./panda_llm ./panda_factor_server
```

2) 配置环境变量（可选，覆盖 config.yaml）
```bash
cp .env.example .env
# 编辑 .env，填入 Mongo、OpenAI/DeepSeek 等配置。环境变量将覆盖 config.yaml 同名键。
```
- 默认配置文件：`panda_common/panda_common/config.yaml`
- 环境变量覆盖：已内置（dotenv 可选，如安装则自动加载 `.env`）

3) 启动服务
- 因子服务与前端（包含 `/api/v1` 与 `/factor` 静态资源）：
```bash
python -m panda_factor_server.panda_factor_server
# 或
uvicorn panda_factor_server.__main__:app --host 0.0.0.0 --port 8111
```
- LLM 聊天服务（如需独立启动）：
```bash
uvicorn panda_llm.server:app --reload
```
- 定时任务（数据/因子清洗）：
```bash
python -m panda_data_hub.panda_data_hub.task.main
```

4) 冒烟测试（可选）
- 运行前提：服务已启动且 MongoDB 有可用市场数据；可选设置 `API_KEY`。
```bash
python scripts/smoke_api.py --base http://127.0.0.1:8111 --api-key "$API_KEY"
# 依赖 requests：pip install requests
```

## 主要接口（前缀 `/api/v1`）
- 因子管理：`/user_factor_list`、`/query_factor`、`/create_factor`、`/update_factor`、`/delete_factor`
- 运行与任务：`/run_factor`、`/query_task_status`、`/task_logs`
- 分析图表：`/query_factor_excess_chart`、`/query_factor_analysis_data`、`/query_group_return_analysis`、
  `/query_ic_*`、`/query_rank_ic_*`、`/query_return_chart`、`/query_simple_return_chart`
- LLM（前缀 `/llm`）：`POST /llm/chat`（SSE）、`GET /llm/chat/sessions?user_id=...`

## 定时任务
- 配置项：
  - `STOCKS_UPDATE_TIME`（每日股票与行情清洗时间，24h 格式）
  - `FACTOR_UPDATE_TIME`（每日因子更新）
  - `DATAHUBSOURCE`（`tushare`/`ricequant`/…）
- 入口：`panda_data_hub/panda_data_hub/task/main.py`
- 注意：调度器使用后台线程，需要保持进程常驻（脚本已处理 Ctrl+C 优雅退出）。

## 配置键（节选）
- Mongo：`MONGO_URI`、`MONGO_USER`、`MONGO_PASSWORD`、`MONGO_AUTH_DB`、`MONGO_DB`、`MONGO_TYPE`、`MONGO_REPLICA_SET`
- 数据源：`DATAHUBSOURCE`、`TS_TOKEN`、`MUSER/MPASSWORD`、`XT_TOKEN`
- LLM：`LLM_API_KEY`、`LLM_MODEL`、`LLM_BASE_URL`

## 开发规范
- Python 4 空格缩进；`snake_case`（函数/变量/模块），`CamelCase`（类）。
- 尽量添加类型标注与简洁 docstring；避免引入未使用依赖。
- 严禁提交密钥、证书、数据库快照等敏感信息到 Git。
- 修改范围最小化：只改必要文件；避免大规模无关重排。

## 测试与排错
- 最小化集成测试：优先 mock 数据源与外部 API。
- Q：定时任务不起效？
  - A：确认通过 `python -m panda_data_hub.panda_data_hub.task.main` 启动；检查配置时间；查看 `logs/` 日志。
- Q：LLM 请求失败？
  - A：检查 `LLM_API_KEY/LLM_BASE_URL/LLM_MODEL`；网络与超时；查看服务日志。
