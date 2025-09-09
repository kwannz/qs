#!/usr/bin/env bash
set -euo pipefail

# Creates GitHub issues from the curated breakdown in sprint-issues.md
# Requires: GitHub CLI (gh) authenticated for this repo.

DRY_RUN=${DRY_RUN:-0}

function need() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1"; exit 1; }
}

need gh

# Verify repo context
if ! gh repo view >/dev/null 2>&1; then
  echo "GitHub CLI not authenticated for this repository. Run: gh auth login"
  exit 1
fi

create_issue() {
  local title="$1"; shift
  local labels_csv="$1"; shift
  local body="$1"; shift || true

  echo "==> Creating issue: $title"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "gh issue create --title \"$title\" --label ${labels_csv//,/ --label } --body <<'EOF'"
    echo "$body"
    echo "EOF"
    echo
  else
    gh issue create --title "$title" $(printf -- "--label %q " ${labels_csv//,/ }) --body "$body" >/dev/null
  fi
}

# 1) API security defaults
create_issue "[Sprint] 强化 API 默认安全（JWT/RBAC/CORS/限流）" \
  "sprint,area:api,type:security,priority:P0" \
  "背景：确保默认安全基线，避免未鉴权访问与跨域风险。
范围/步骤：
- 接入 JWT（签名密钥、过期、刷新）；基于角色/作用域的 RBAC。
- CORS 白名单（可配置），屏蔽通配；统一 401/403 错误响应。
- 接入速率限制（IP/令牌维度）。
验收标准：未携带令牌访问敏感端点返回 401；跨域严格受控；新增集成测试覆盖。
风险/依赖：密钥管理、时钟偏差、反向代理配置。"

# 2) WebSocket limits & heartbeat
create_issue "[Sprint] WebSocket 限流与心跳" \
  "sprint,area:api,type:stability,priority:P0" \
  "背景：WS 长连接可能导致资源泄漏或 DoS。
范围/步骤：心跳检测、空闲断开、最大连接数、消息大小与频率限制、统一错误码。
验收标准：超出限制被拒；空闲自动断连；新增 WS 集成用例。
风险/依赖：代理超时、浏览器兼容。"

# 3) Unauthorized/forbidden e2e tests
create_issue "[Sprint] 未授权/权限不足 e2e 测试" \
  "sprint,area:test,type:e2e,priority:P1" \
  "背景：覆盖 API/WS 未鉴权与越权情形。
范围/步骤：新增 e2e 用例（REST/WS），校验 401/403 与 CORS。
验收标准：CI 通过；问题场景均被测试捕获。"

# 4) Centralized precision/min-stake checks
create_issue "[Sprint] 交易所精度/最小额集中校验" \
  "sprint,area:exchange,type:correctness,priority:P0" \
  "背景：不同交易所精度与最小额差异导致拒单/误差。
范围/步骤：在下单前集中化精度/最小额校验与兜底修正；统一错误信息。
验收标准：边界场景单测齐全；拒单率下降；日志可观测。"

# 5) Precision boundary unit tests
create_issue "[Sprint] 精度边界与舍入策略单测集" \
  "sprint,area:test,type:unit,priority:P1" \
  "背景：补齐四舍五入、截断、最小步进、最小名义金额用例。
范围/步骤：构造多交易所元数据样例，覆盖下单前/后处理。
验收标准：100% 覆盖相关分支；与文档策略一致。"

# 6) Time/concurrency test stabilization
create_issue "[Sprint] 时间与并发用例去抖（time-freezing/隔离）" \
  "sprint,area:test,type:stability,priority:P1" \
  "背景：时间/异步类测试偶发不稳定。
范围/步骤：引入 time-machine；隔离事件循环；清理共享状态。
验收标准：目标用例重复 10 次稳定通过。"

# 7) Coverage gate in CI
create_issue "[Sprint] CI 覆盖率阈值接入" \
  "sprint,area:ci,type:quality,priority:P1" \
  "背景：确保新增改动具备最低覆盖率。
范围/步骤：pytest --cov --cov-fail-under=<阈值>；报告上传与门禁。
验收标准：阈值生效；失败时阻断合并。"

# 8) Backoff/retry and rate limiting for exchanges
create_issue "[Sprint] ccxt/httpx 退避重试与限频器" \
  "sprint,area:exchange,type:resilience,priority:P1" \
  "背景：网络抖动与限频导致间歇失败。
范围/步骤：统一退避重试策略；速率测量与节流；错误分类与上报。
验收标准：长跑任务失败率显著下降；日志可观测。"

# 9) Market metadata TTL cache & validation
create_issue "[Sprint] 市场元数据 TTL 缓存与一致性校验" \
  "sprint,area:exchange,type:performance,priority:P2" \
  "背景：市场精度/最小额变化需及时生效，且避免频繁拉取。
范围/步骤：TTL 缓存；定期对账；异常变化时告警。
验收标准：缓存命中率提升；异常变更可被检测与记录。"

# 10) Data gap detection & dedup
create_issue "[Sprint] 数据缺口检测与去重工具" \
  "sprint,area:data,type:quality,priority:P1" \
  "背景：OHLCV/Trades 存在缺口/重复风险。
范围/步骤：实现缺口扫描与补齐策略；重复去重；质量日志。
验收标准：提供 CLI/日志报告；单测覆盖典型场景。"

# 11) Timezone normalization
create_issue "[Sprint] 时区规范化与边界校验" \
  "sprint,area:data,type:correctness,priority:P2" \
  "背景：多时区导致对齐问题。
范围/步骤：统一 TZ 处理；校验 K 线边界；避免夏令时陷阱。
验收标准：跨时区回测结果一致；新增单测。"

# 12) DB indexing & profiling
create_issue "[Sprint] DB 关键查询索引与性能画像" \
  "sprint,area:db,type:performance,priority:P2" \
  "背景：交易/历史统计查询可能存在慢查询。
范围/步骤：定位热点查询；添加索引；基准对比前后性能。
验收标准：关键查询 P95 改善 ≥30%。"

# 13) Backup/restore drill
create_issue "[Sprint] 备份/恢复流程与演练文档" \
  "sprint,area:db,type:ops,priority:P3" \
  "背景：数据安全与恢复能力。
范围/步骤：形成标准流程；脚本化导出/恢复；演练记录。
验收标准：演练可在限定时间内完成且数据一致。"

# 14) Timeouts/circuit breakers/thread-pool
create_issue "[Sprint] 统一超时/熔断与受限线程池" \
  "sprint,area:runtime,type:resilience,priority:P1" \
  "背景：阻塞 I/O 与异常级联。
范围/步骤：为外部调用设超时/熔断；线程池容量与队列上限；监控指标。
验收标准：在受控故障注入下，系统退化而非崩溃。"

# 15) Pandas/NumPy optimization
create_issue "[Sprint] Pandas/NumPy 向量化与内存优化" \
  "sprint,area:perf,type:optimization,priority:P2" \
  "背景：数据处理热点耗时与内存占用。
范围/步骤：热点定位；向量化替换循环；分类 dtype；减少复制。
验收标准：基准 case 性能提升 ≥20%，内存峰值下降 ≥20%。"

# 16) Parallelism & caching
create_issue "[Sprint] 计算并行化与缓存（joblib/xdist/TTL）" \
  "sprint,area:perf,type:scalability,priority:P2" \
  "背景：CPU 绑定任务串行瓶颈。
范围/步骤：并行/分区处理；TTL 缓存市场/配置；命中指标。
验收标准：整体运行时减少 ≥20%；缓存命中率达标。"

# 17) orjson payload slimming
create_issue "[Sprint] JSON 序列化与载荷瘦身（orjson 调优）" \
  "sprint,area:api,type:performance,priority:P2" \
  "背景：RPC/API 频繁序列化与传输。
范围/步骤：启用 orjson 选项；压缩冗余字段；分页/增量下发。
验收标准：P95 响应时间下降 ≥15%；流量下降 ≥20%。"

# 18) CI split & change-aware tests
create_issue "[Sprint] CI 热点拆分与变更感知测试" \
  "sprint,area:ci,type:productivity,priority:P2" \
  "背景：CI 时长与稳定性。
范围/步骤：-n auto 并行与矩阵拆分；仅跑受影响子集；缓存依赖与构建。
验收标准：CI 总时长下降 ≥20%；波动降低。"

# 19) Docker slimming + scans
create_issue "[Sprint] Docker 多阶段与瘦身 + 漏洞扫描" \
  "sprint,area:devops,type:security,priority:P2" \
  "背景：镜像体积大、潜在 CVE。
范围/步骤：多阶段构建；slim 基镜像；缓存 wheels；引入扫描（GHCR/Trivy）。
验收标准：镜像体积下降 ≥30%；扫描无高危。"

# 20) Upper bounds & dependabot
create_issue "[Sprint] 关键依赖上界与 Dependabot 策略" \
  "sprint,area:deps,type:stability,priority:P2" \
  "背景：依赖突发性破坏更新。
范围/步骤：为 ccxt/fastapi/pandas 等设上界；Dependabot 规则；变更审查与回归。
验收标准：版本升级均在受控 PR 中完成并通过 CI。"

# 21) Secrets scanning in CI
create_issue "[Sprint] Secrets 扫描（Gitleaks）接入 CI" \
  "sprint,area:security,type:ci,priority:P1" \
  "背景：防止密钥意外泄露。
范围/步骤：CI 增加 Gitleaks；false positive 白名单；贡献文档更新。
验收标准：默认扫描生效；误报率可接受；文档可执行。"

# 22) Performance baselines
create_issue "[Sprint] 性能基线与基准脚本" \
  "sprint,area:perf,type:benchmark,priority:P3" \
  "背景：优化需有衡量标准。
范围/步骤：定义关键场景与数据集；编写 benchmark 脚本；CI 按需运行。
验收标准：生成基线报告；新变更与基线对比可视化。"

echo "All done. Set DRY_RUN=1 to preview without creating issues."

