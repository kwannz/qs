## Enabling Redis Sliding‑Window Rate Limiting

This document shows how to enable and verify Redis‑backed sliding‑window rate limiting for the REST API.

> Personal / Local-only usage
> For strictly personal/local setups that never expose the API publicly, in‑memory limits may be sufficient or can be disabled entirely. When moving to multi-user or distributed deployments, prefer Redis‑backed limits to enforce quotas consistently across processes/instances.

### 1) Start Redis locally

Using Docker:

```
docker run -d --name ft-redis -p 6379:6379 redis:7
```

Ensure Redis is reachable at `redis://127.0.0.1:6379`.

### 2) Configure Freqtrade

In your config (e.g. `user_data/config.json`), set:

```
"api_server": {
  "enabled": true,
  "listen_ip_address": "127.0.0.1",
  "listen_port": 8080,
  "enable_rate_limit": true,
  "rate_limit_backend": "redis_sliding",
  "rate_limit_redis_url": "redis://127.0.0.1:6379",
  "rate_limit_per_minute": 120,
  ...
}
```

Notes:

- `rate_limit_backend` can be `memory`, `redis` (fixed window), or `redis_sliding` (preferred).
- Per‑minute quota applies to private API (token/IP). Public endpoints remain unlimited by default.

### 3) Verify locally

Start API (webserver mode or as part of your run), then flood a private or public endpoint from another terminal to observe 429:

```
# Example: flood public ping endpoint (~ test purposes only)
for i in $(seq 1 300); do curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8080/api/v1/ping; done
```

You should see a mixture of `200` and `429` once limits are exceeded.

Logs will show occasional warnings (every ~100 denials):

```
Rate limited (IP): 127.0.0.1, count=101
Rate limited (Token/IP): <token or ip>, count=201
```

### 4) Observability / Dashboard (optional)

If you are running centralized logging (e.g. Loki/Grafana), you can build a quick panel with:

- Query filter: `|= "Rate limited ("` and group‑by count over time.
- Table panel listing top offending IP/token.

Without a centralized stack, you can still count denials via grep:

```
grep -c "Rate limited (" /path/to/freqtrade.log
```

### 5) Tuning

- Adjust `rate_limit_per_minute` for your environment.
- Switch to `redis` fixed‑window if the sliding variant is too granular (the latter is generally preferred).
- Ensure API is not exposed publicly if authentication is disabled (personal/local mode).
