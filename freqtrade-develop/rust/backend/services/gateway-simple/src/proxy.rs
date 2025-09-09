//! HTTP Proxy Client for upstream services

use anyhow::{anyhow, Result};
use axum::{body::{Body, to_bytes}, http::{HeaderName, Request, Response}};
use axum::http::{Method, StatusCode};
use reqwest::Client;
use std::{collections::HashMap, time::{Duration, Instant}, sync::Arc};
use tokio::sync::{RwLock, Semaphore};
use once_cell::sync::Lazy;
use prometheus::{IntCounter, register_int_counter};

const MAX_BODY_SIZE: usize = 2 * 1024 * 1024; // 2 MiB limit for proxied request bodies

#[derive(Debug, Clone)]
pub struct ProxyClient {
    client: Client,
    base_urls: HashMap<String, String>,
    circuit: Arc<RwLock<HashMap<String, CircuitState>>>, // simple per-service circuit breaker
    rate: Arc<RwLock<HashMap<String, RateWindow>>>,       // simple per-service rate limiter (fixed window)
    semaphore: Arc<Semaphore>,                             // global concurrency limit
    path_blacklist: Vec<String>,                           // simple path-prefix blacklist
    service_rate_limits: HashMap<String, u32>,             // per-service RPS limits
    path_rate_limits: Vec<(String, u32)>,                  // path-prefix RPS limits
    ip_whitelist: Vec<String>,                             // simple IP whitelist (from headers)
}

impl ProxyClient {
    pub fn new(config: crate::config::Config) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;

        let mut base_urls = HashMap::new();
        base_urls.insert("market-data".into(), config.upstream_services.market_data);
        base_urls.insert("trading".into(), config.upstream_services.trading);
        base_urls.insert("risk".into(), config.upstream_services.risk);
        base_urls.insert("strategy".into(), config.upstream_services.strategy);

        // Load optional path blacklist from env
        let path_blacklist = std::env::var("GATEWAY_PATH_BLACKLIST")
            .ok()
            .map(|s| s.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect())
            .unwrap_or_else(|| Vec::new());

        // Per-service rate limits from env (e.g., risk=100,strategy=80)
        let mut service_rate_limits = HashMap::new();
        if let Ok(v) = std::env::var("GATEWAY_SERVICE_RATE_LIMITS") {
            for pair in v.split(',') {
                if let Some((k, val)) = pair.split_once('=') {
                    if let Ok(n) = val.trim().parse::<u32>() {
                        service_rate_limits.insert(k.trim().to_lowercase(), n);
                    }
                }
            }
        }

        // Path rate limits (e.g., /api/v1/risk/=100,/api/v1/strategies/=50)
        let mut path_rate_limits: Vec<(String, u32)> = Vec::new();
        if let Ok(v) = std::env::var("GATEWAY_PATH_RATE_LIMITS") {
            for pair in v.split(',') {
                if let Some((k, val)) = pair.split_once('=') {
                    if let Ok(n) = val.trim().parse::<u32>() {
                        path_rate_limits.push((k.trim().to_string(), n));
                    }
                }
            }
        }

        // IP whitelist (comma-separated entries, compare to X-Forwarded-For / X-Real-IP)
        let ip_whitelist: Vec<String> = std::env::var("GATEWAY_IP_WHITELIST")
            .ok()
            .map(|s| s.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect())
            .unwrap_or_else(|| Vec::new());

        Ok(Self {
            client,
            base_urls,
            circuit: Arc::new(RwLock::new(HashMap::new())),
            rate: Arc::new(RwLock::new(HashMap::new())),
            semaphore: Arc::new(Semaphore::new(256)),
            path_blacklist,
            service_rate_limits,
            path_rate_limits,
            ip_whitelist,
        })
    }

    pub async fn proxy_request(&self, service_name: &str, req: Request<Body>) -> Result<Response<Body>> {
        let base = self
            .base_urls
            .get(service_name)
            .ok_or_else(|| anyhow!("unknown upstream service: {}", service_name))?
            .trim_end_matches('/')
            .to_string();

        let (parts, body) = req.into_parts();
        let path_and_query = parts
            .uri
            .path_and_query()
            .map(|pq| pq.as_str())
            .unwrap_or("/");
        let url = format!("{base}{path_and_query}");

        // Path blacklist check
        if self.path_blacklist.iter().any(|p| path_and_query.starts_with(p)) {
            return Ok(Response::builder()
                .status(StatusCode::FORBIDDEN)
                .body(Body::from("{\"error\":\"path forbidden\"}"))
                .unwrap());
        }

        // IP whitelist (if configured). Read X-Forwarded-For or X-Real-IP
        if !self.ip_whitelist.is_empty() {
            let mut allowed = false;
            if let Some(v) = parts.headers.get("x-forwarded-for") {
                if let Ok(s) = v.to_str() {
                    if let Some(first) = s.split(',').next() { allowed = self.ip_whitelist.iter().any(|ip| ip == &first.trim()); }
                }
            }
            if !allowed {
                if let Some(v) = parts.headers.get("x-real-ip") {
                    if let Ok(s) = v.to_str() { allowed = self.ip_whitelist.iter().any(|ip| ip == &s.trim()); }
                }
            }
            if !allowed {
                return Ok(Response::builder()
                    .status(StatusCode::FORBIDDEN)
                    .body(Body::from("{\"error\":\"ip not allowed\"}"))
                    .unwrap());
            }
        }

        // Circuit breaker: short-circuit if open
        if self.is_circuit_open(service_name).await {
            return Ok(Response::builder()
                .status(StatusCode::SERVICE_UNAVAILABLE)
                .body(Body::from("{\"error\":\"circuit open\"}"))
                .unwrap());
        }

        // Rate limiting: basic fixed-window per-service
        if !self.check_rate_limit(service_name, path_and_query).await {
            return Ok(Response::builder()
                .status(StatusCode::TOO_MANY_REQUESTS)
                .body(Body::from("{\"error\":\"rate limit exceeded\"}"))
                .unwrap());
        }

        // Concurrency limiting (global)
        let _permit = match self.semaphore.clone().acquire_owned().await {
            Ok(p) => p,
            Err(_) => {
                return Ok(Response::builder()
                    .status(StatusCode::SERVICE_UNAVAILABLE)
                    .body(Body::from("{\"error\":\"server busy\"}"))
                    .unwrap());
            }
        };

        // Build reqwest request
        let method = map_method(&parts.method)?;
        let mut builder = self.client.request(method, &url);

        // Copy headers (exclude hop-by-hop)
        for (name, value) in parts.headers.iter() {
            if is_hop_by_hop(name) || name == &HeaderName::from_static("host") {
                continue;
            }
            builder = builder.header(name, value);
        }

        let bytes = to_bytes(body, MAX_BODY_SIZE).await?;
        builder = builder.body(bytes);

        // metrics: count all proxy requests
        REQUESTS_TOTAL.inc();

        // Simple retry for GET on transient upstream errors
        let is_get = parts.method == Method::GET;
        let mut attempt = 0u32;
        let max_attempts = if is_get { 2 } else { 1 };
        loop {
            let res = builder.try_clone()
                .unwrap_or_else(|| self.client.request(map_method(&parts.method).unwrap(), &url))
                .send()
                .await;
            match res {
                Ok(r) => {
                    let status = r.status();
                    let body_bytes = r.bytes().await.unwrap_or_default();
                    // success resets circuit
                    self.on_success(service_name).await;
                    if is_get && (status == StatusCode::BAD_GATEWAY || status == StatusCode::GATEWAY_TIMEOUT || status == StatusCode::SERVICE_UNAVAILABLE) && attempt + 1 < max_attempts {
                        attempt += 1;
                        continue;
                    }
                    return Ok(Response::builder().status(status).body(Body::from(body_bytes)).unwrap());
                }
                Err(err) => {
                    // register failure and maybe open circuit
                    self.on_failure(service_name).await;
                    UPSTREAM_ERRORS_TOTAL.inc();
                    if is_get && attempt + 1 < max_attempts {
                        attempt += 1;
                        continue;
                    }
                    return Ok(Response::builder()
                        .status(StatusCode::BAD_GATEWAY)
                        .body(Body::from(format!("{{\"error\":\"upstream error: {}\"}}", err)))
                        .unwrap());
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct CircuitState {
    failures: u32,
    open_until: Option<Instant>,
}

impl Default for CircuitState {
    fn default() -> Self {
        Self { failures: 0, open_until: None }
    }
}

#[derive(Debug, Clone)]
struct RateWindow {
    window_start: Instant,
    count: u32,
}

impl ProxyClient {
    async fn is_circuit_open(&self, service: &str) -> bool {
        let map = self.circuit.read().await;
        if let Some(state) = map.get(service) {
            if let Some(until) = state.open_until {
                return Instant::now() < until;
            }
        }
        false
    }

    async fn on_success(&self, service: &str) {
        let mut map = self.circuit.write().await;
        let entry = map.entry(service.to_string()).or_default();
        entry.failures = 0;
        entry.open_until = None;
    }

    async fn on_failure(&self, service: &str) {
        let mut map = self.circuit.write().await;
        let entry = map.entry(service.to_string()).or_default();
        entry.failures = entry.failures.saturating_add(1);
        // threshold + cooldown
        if entry.failures >= 5 {
            entry.open_until = Some(Instant::now() + Duration::from_secs(30));
            entry.failures = 0;
        }
    }

    async fn check_rate_limit(&self, service: &str, path: &str) -> bool {
        let default_limit: u32 = 200;
        // Path override
        let mut limit = self
            .service_rate_limits
            .get(&service.to_lowercase())
            .copied()
            .unwrap_or(default_limit);
        for (prefix, n) in &self.path_rate_limits {
            if path.starts_with(prefix) { limit = *n; break; }
        }
        let mut map = self.rate.write().await;
        let now = Instant::now();
        let key = format!("{}:{}", service, path);
        let window = map.entry(key).or_insert_with(|| RateWindow { window_start: now, count: 0 });
        if now.duration_since(window.window_start) >= Duration::from_secs(1) {
            window.window_start = now;
            window.count = 0;
        }
        if window.count >= limit { RATE_LIMITED_TOTAL.inc(); return false; }
        window.count += 1;
        true
    }
}

fn is_hop_by_hop(name: &HeaderName) -> bool {
    const HOP: [&str; 9] = [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "proxy-connection",
    ];
    HOP.iter().any(|h| name == h)
}

fn map_method(method: &Method) -> Result<reqwest::Method> {
    Ok(match *method {
        Method::GET => reqwest::Method::GET,
        Method::POST => reqwest::Method::POST,
        Method::PUT => reqwest::Method::PUT,
        Method::DELETE => reqwest::Method::DELETE,
        Method::PATCH => reqwest::Method::PATCH,
        Method::HEAD => reqwest::Method::HEAD,
        Method::OPTIONS => reqwest::Method::OPTIONS,
        Method::TRACE => reqwest::Method::TRACE,
        Method::CONNECT => reqwest::Method::CONNECT,
        _ => return Err(anyhow!("unsupported HTTP method")),
    })
}

// Prometheus metrics
static REQUESTS_TOTAL: Lazy<IntCounter> = Lazy::new(|| {
    register_int_counter!("gateway_proxy_requests_total", "Total number of proxied requests").unwrap()
});
static UPSTREAM_ERRORS_TOTAL: Lazy<IntCounter> = Lazy::new(|| {
    register_int_counter!("gateway_upstream_errors_total", "Total number of upstream errors").unwrap()
});
static RATE_LIMITED_TOTAL: Lazy<IntCounter> = Lazy::new(|| {
    register_int_counter!("gateway_rate_limited_total", "Total number of rate-limited requests").unwrap()
});
