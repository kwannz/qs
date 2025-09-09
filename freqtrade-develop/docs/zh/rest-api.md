# REST API（简体中文）

## 配置

在配置文件中启用 `api_server.enabled: true` 即可启用 API。

```json
"api_server": {
  "enabled": true,
  "listen_ip_address": "127.0.0.1",
  "listen_port": 8080,
  "verbosity": "error",
  "enable_openapi": false,
  "jwt_secret_key": "somethingrandom",
  "CORS_origins": [],
  "enable_rate_limit": true,
  "rate_limit_backend": "memory",
  "rate_limit_per_minute": 120
}
```

> 个人/本地使用
> 仅在本机使用时，可保持绑定 `127.0.0.1`，必要时可临时设置 `auth_disabled: true`（严禁对公网暴露）。迁移到远程/多用户环境时，请启用 JWT/Basic 鉴权、设置强随机 `jwt_secret_key`、配置 `CORS_origins`，并考虑使用 Redis 限流。

### 健康检查

浏览器访问 `http://127.0.0.1:8080/api/v1/ping` 应返回：`{"status":"pong"}`。

