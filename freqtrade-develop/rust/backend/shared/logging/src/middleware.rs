//! HTTP请求日志中间件

use axum::{
    extract::{ConnectInfo, MatchedPath, State},
    http::{Request, Response, HeaderMap},
    middleware::Next,
    body::Body,
};
use std::{
    net::SocketAddr,
    time::Instant,
};
use tower::{Layer, Service};
use tower_http::trace::{TraceLayer, MakeSpan, OnRequest, OnResponse};
use tracing::{Span, info, warn, error, debug};
use uuid::Uuid;


/// 日志中间件层
#[derive(Clone)]
pub struct LoggingLayer {
    service_name: String,
}

impl LoggingLayer {
    /// 创建新的日志中间件层
    pub fn new(service_name: String) -> Self {
        Self { service_name }
    }
}

impl<S> Layer<S> for LoggingLayer {
    type Service = LoggingService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        LoggingService {
            inner,
            service_name: self.service_name.clone(),
        }
    }
}

/// 日志服务
#[derive(Clone)]
pub struct LoggingService<S> {
    inner: S,
    #[allow(dead_code)]
    service_name: String,
}

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for LoggingService<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>>,
    S::Future: Send + 'static,
    ReqBody: Send + 'static,
    ResBody: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
        let future = self.inner.call(req);
        Box::pin(future)
    }
}

/// HTTP请求追踪中间件
pub async fn http_trace_middleware(
    request: Request<Body>,
    next: Next,
) -> Response<Body> {
    let start = Instant::now();
    let request_id = Uuid::new_v4().to_string();
    
    // 提取请求信息
    let method = request.method().to_string();
    let uri = request.uri().to_string();
    let path = request.uri().path().to_string();
    let query = request.uri().query().map(|q| q.to_string());
    let headers = request.headers().clone();
    
    // 获取客户端IP
    let client_ip = extract_client_ip(&headers, &request);
    let user_agent = headers
        .get("user-agent")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    // 创建追踪span
    let span = tracing::info_span!(
        "http_request",
        method = %method,
        uri = %uri,
        request_id = %request_id,
        client_ip = %client_ip,
    );

    let _guard = span.enter();

    info!(
        method = %method,
        path = %path,
        query = ?query,
        client_ip = %client_ip,
        user_agent = %user_agent,
        "HTTP请求开始"
    );

    // 执行请求
    let mut response = next.run(request).await;
    
    // 计算执行时间
    let duration = start.elapsed();
    let status_code = response.status().as_u16();
    
    // 在响应头注入追踪与耗时
    let _ = response.headers_mut().insert(
        axum::http::header::HeaderName::from_static("x-trace-id"),
        axum::http::HeaderValue::from_str(&request_id).unwrap_or(axum::http::HeaderValue::from_static("invalid")),
    );
    let _ = response.headers_mut().insert(
        axum::http::header::HeaderName::from_static("x-execution-time-ms"),
        axum::http::HeaderValue::from_str(&format!("{}", duration.as_millis())).unwrap_or(axum::http::HeaderValue::from_static("0")),
    );

    // 记录响应日志
    match status_code {
        200..=299 => {
            info!(
                status_code = status_code,
                duration_ms = duration.as_millis(),
                "HTTP请求成功"
            );
        }
        400..=499 => {
            warn!(
                status_code = status_code,
                duration_ms = duration.as_millis(),
                "HTTP请求客户端错误"
            );
        }
        500..=599 => {
            error!(
                status_code = status_code,
                duration_ms = duration.as_millis(),
                "HTTP请求服务器错误"
            );
        }
        _ => {
            debug!(
                status_code = status_code,
                duration_ms = duration.as_millis(),
                "HTTP请求完成"
            );
        }
    }

    response
}

/// 提取客户端IP地址  
fn extract_client_ip(headers: &HeaderMap, request: &Request<Body>) -> String {
    // 尝试从X-Forwarded-For头获取
    if let Some(forwarded) = headers.get("x-forwarded-for") {
        if let Ok(forwarded_str) = forwarded.to_str() {
            if let Some(ip) = forwarded_str.split(',').next() {
                return ip.trim().to_string();
            }
        }
    }

    // 尝试从X-Real-IP头获取
    if let Some(real_ip) = headers.get("x-real-ip") {
        if let Ok(ip_str) = real_ip.to_str() {
            return ip_str.to_string();
        }
    }

    // 从连接信息获取
    if let Some(connect_info) = request.extensions().get::<ConnectInfo<SocketAddr>>() {
        return connect_info.0.ip().to_string();
    }

    "unknown".to_string()
}

/// 创建追踪层
pub fn create_trace_layer() -> impl tower::Layer<axum::Router> {
    TraceLayer::new_for_http()
        .make_span_with(HttpMakeSpan)
        .on_request(HttpOnRequest)
        .on_response(HttpOnResponse)
}

/// Prometheus 指标中间件（基于 MetricsCollector）
/// 记录 http_requests_total 与 http_request_duration_ms 直方图
pub async fn http_metrics_middleware(
    State(metrics): State<std::sync::Arc<crate::metrics::MetricsCollector>>,
    request: Request<Body>,
    next: Next,
) -> Response<Body> {
    use std::time::Instant;
    let start = Instant::now();

    // 预取请求信息
    let method = request.method().to_string();
    let path = request
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_string())
        .unwrap_or_else(|| request.uri().path().to_string());

    // 执行下游
    let response = next.run(request).await;
    let duration = start.elapsed();
    let status_code = response.status().as_u16();

    let http_metrics = crate::metrics::HttpMetrics::new(metrics.clone());
    http_metrics
        .record_request(&method, &path, status_code, duration)
        .await;

    response
}

/// 领域级指标中间件（Execution/Markets/Risk/Factors/Backtests/Strategy/Analytics）
/// 记录 domain_requests_total 与 domain_request_duration_ms
pub async fn domain_metrics_middleware(
    State(metrics): State<std::sync::Arc<crate::metrics::MetricsCollector>>,
    request: Request<Body>,
    next: Next,
) -> Response<Body> {
    use std::time::Instant;
    let start = Instant::now();

    let method = request.method().to_string();
    let path = request
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_string())
        .unwrap_or_else(|| request.uri().path().to_string());

    let response = next.run(request).await;
    let duration = start.elapsed();
    let status_code = response.status().as_u16();

    let domain = match path.as_str() {
        p if p.starts_with("/api/v1/orders") => "execution",
        p if p.starts_with("/api/v1/markets") => "markets",
        p if p.starts_with("/api/v1/risk") => "risk",
        p if p.starts_with("/api/v1/factors") => "factors",
        p if p.starts_with("/api/v1/backtests") => "backtests",
        p if p.starts_with("/api/v1/strategies") => "strategy",
        p if p.starts_with("/api/v1/analytics") => "analytics",
        p if p.starts_with("/api/v1/config") || p.starts_with("/api/v1/alerts") => "config",
        _ => "other",
    };

    // Labels
    let mut labels = std::collections::HashMap::new();
    labels.insert("domain".to_string(), domain.to_string());
    labels.insert("method".to_string(), method);
    labels.insert("status_code".to_string(), status_code.to_string());

    // 提取 symbol（市场路径）
    if domain == "markets" {
        // /api/v1/markets/{symbol}/...
        let segs: Vec<&str> = path.split('/').collect();
        if segs.len() > 4 && segs[3] == "markets" {
            let sym = segs.get(4).map(|s| s.to_string()).unwrap_or_default();
            if !sym.is_empty() { labels.insert("symbol".to_string(), sym); }
        }
    }

    // Counter + Histogram
    metrics.increment_counter("domain_requests_total", labels.clone()).await;
    metrics
        .record_histogram("domain_request_duration_ms", duration.as_millis() as f64, labels)
        .await;

    response
}

/// HTTP请求Span创建器
#[derive(Clone)]
pub struct HttpMakeSpan;

impl<B> MakeSpan<B> for HttpMakeSpan {
    fn make_span(&mut self, request: &Request<B>) -> Span {
        let request_id = Uuid::new_v4();
        
        tracing::info_span!(
            "http_request",
            method = %request.method(),
            uri = %request.uri(),
            version = ?request.version(),
            request_id = %request_id,
            otel.name = format!("{} {}", request.method(), request.uri().path()),
            otel.kind = "server",
            http.method = %request.method(),
            http.url = %request.uri(),
            http.scheme = %request.uri().scheme_str().unwrap_or("http"),
            http.target = %request.uri().path(),
        )
    }
}

/// HTTP请求开始处理器
#[derive(Clone)]
pub struct HttpOnRequest;

impl<B> OnRequest<B> for HttpOnRequest {
    fn on_request(&mut self, request: &Request<B>, span: &Span) {
        let path = request
            .extensions()
            .get::<MatchedPath>()
            .map(|p| p.as_str())
            .unwrap_or(request.uri().path());

        span.record("http.route", path);
        
        info!("开始处理HTTP请求");
    }
}

/// HTTP响应处理器
#[derive(Clone)]
pub struct HttpOnResponse;

impl<B> OnResponse<B> for HttpOnResponse {
    fn on_response(self, response: &Response<B>, latency: std::time::Duration, span: &Span) {
        let status = response.status();
        span.record("http.status_code", status.as_u16());
        
        match status.as_u16() {
            200..=299 => {
                info!(
                    http.status_code = status.as_u16(),
                    duration_ms = latency.as_millis(),
                    "HTTP请求成功完成"
                );
            }
            400..=499 => {
                warn!(
                    http.status_code = status.as_u16(),
                    duration_ms = latency.as_millis(),
                    "HTTP请求客户端错误"
                );
            }
            500..=599 => {
                error!(
                    http.status_code = status.as_u16(),
                    duration_ms = latency.as_millis(),
                    "HTTP请求服务器错误"
                );
            }
            _ => {
                info!(
                    http.status_code = status.as_u16(),
                    duration_ms = latency.as_millis(),
                    "HTTP请求完成"
                );
            }
        }
    }
}

/// 业务操作日志宏
#[macro_export]
macro_rules! business_log {
    ($level:ident, $action:expr, $($key:ident = $value:expr),*) => {
        tracing::$level!(
            action = $action,
            timestamp = %chrono::Utc::now(),
            $($key = $value,)*
        );
    };
}

/// 交易操作日志宏  
#[macro_export]
macro_rules! trading_log {
    ($level:ident, $action:expr, order_id = $order_id:expr, symbol = $symbol:expr, $($key:ident = $value:expr),*) => {
        tracing::$level!(
            action = $action,
            order_id = $order_id,
            symbol = $symbol,
            service = "trading",
            timestamp = %chrono::Utc::now(),
            $($key = $value,)*
        );
    };
}

/// 市场数据日志宏
#[macro_export]
macro_rules! market_log {
    ($level:ident, $action:expr, source = $source:expr, $($key:ident = $value:expr),*) => {
        tracing::$level!(
            action = $action,
            source = $source,
            service = "market",
            timestamp = %chrono::Utc::now(),
            $($key = $value,)*
        );
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Method;

    #[test]
    fn test_logging_layer_creation() {
        let layer = LoggingLayer::new("test-service".to_string());
        assert_eq!(layer.service_name, "test-service");
    }

    #[test]
    fn test_extract_client_ip() {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "192.168.1.1, 10.0.0.1".parse().unwrap());
        
        let request = Request::builder()
            .method(Method::GET)
            .uri("http://example.com/")
            .body(Body::empty())
            .unwrap();
        
        let ip = extract_client_ip(&headers, &request);
        assert_eq!(ip, "192.168.1.1");
    }
}
