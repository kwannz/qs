Playwright MCP Tool for Codex CLI
=================================

This repo includes a ready-to-use MCP configuration that adds a Playwright-based browser tool to Codex CLI.

Files
- `.codex/mcp.json` — Registers Playwright MCP servers via `npx @modelcontextprotocol/server-playwright`.

Prerequisites
- Node.js 18+ with `npx` available.
- Playwright browsers will be auto-managed by the server package. If needed, you can pre-install: `npx playwright install`.

How to use with Codex CLI
1) Ensure Node is installed and network access allows `npx` to fetch the package on first run.
2) Launch Codex CLI pointing to this MCP config, for example:
   - codex --mcp-config .codex/mcp.json
   - Or add the contents of `.codex/mcp.json` to your Codex MCP tools configuration.

Multiple browsers
- `playwright` (default): Uses Chromium (`PLAYWRIGHT_BROWSER=chromium`).
- `playwright_firefox`: Same server, Firefox backend (`PLAYWRIGHT_BROWSER=firefox`). Disabled by default; enable by setting `disabled: false`.
- `playwright_webkit`: WebKit backend (`PLAYWRIGHT_BROWSER=webkit`). Disabled by default.

Notes
- Headless mode is enabled by default via `PLAYWRIGHT_HEADLESS=1`. Remove or set to `0` to see a browser (if supported in your environment).
- Optional downloads dir: set `PLAYWRIGHT_DOWNLOADS_DIR` (e.g., `/tmp/mcp-downloads`) to control where files are saved.
- If you prefer a pinned or offline install, replace `npx` with a local path to the server binary or install the package globally first.

中文速览
- 该配置为 Codex CLI 增加一个基于 Playwright 的 MCP 浏览器工具。
- 需要 Node.js 18+。使用 `codex --mcp-config .codex/mcp.json` 启动即可。
- 可选浏览器：Chromium（默认）、Firefox、WebKit（后两者在配置里默认禁用，可改为启用）。
- 默认无头模式（`PLAYWRIGHT_HEADLESS=1`）。如需可视浏览器，可改为 `0`（视当前环境支持情况）。
