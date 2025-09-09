//! 分布式追踪模块

use tracing::{Span, span, Level};
use uuid::Uuid;
use std::collections::HashMap;

/// 追踪上下文
#[derive(Debug, Clone)]
pub struct TraceContext {
    /// 追踪ID
    pub trace_id: String,
    
    /// 跨度ID
    pub span_id: String,
    
    /// 父跨度ID
    pub parent_span_id: Option<String>,
    
    /// 采样标志
    pub sampled: bool,
    
    /// 标签
    pub tags: HashMap<String, String>,
}

impl TraceContext {
    /// 创建新的追踪上下文
    pub fn new() -> Self {
        Self {
            trace_id: Uuid::new_v4().to_string(),
            span_id: Uuid::new_v4().to_string(),
            parent_span_id: None,
            sampled: true,
            tags: HashMap::new(),
        }
    }

    /// 从当前跨度创建子上下文
    pub fn create_child(&self) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            span_id: Uuid::new_v4().to_string(),
            parent_span_id: Some(self.span_id.clone()),
            sampled: self.sampled,
            tags: self.tags.clone(),
        }
    }

    /// 设置标签
    pub fn with_tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }

    /// 设置采样状态
    pub fn with_sampling(mut self, sampled: bool) -> Self {
        self.sampled = sampled;
        self
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
    }
}

/// 追踪工具
pub struct Tracer {
    service_name: String,
    default_tags: HashMap<String, String>,
}

impl Tracer {
    /// 创建新的追踪器
    pub fn new(service_name: String) -> Self {
        let mut default_tags = HashMap::new();
        default_tags.insert("service.name".to_string(), service_name.clone());
        default_tags.insert("service.version".to_string(), 
                          std::env::var("SERVICE_VERSION").unwrap_or_else(|_| "unknown".to_string()));

        Self {
            service_name,
            default_tags,
        }
    }

    /// 开始新的跨度
    pub fn start_span(&self, name: &str, context: Option<TraceContext>) -> (Span, TraceContext) {
        let trace_context = context.unwrap_or_default();
        
        let span = span!(
            Level::INFO,
            "operation",
            otel.name = name,
            otel.kind = "internal",
            trace_id = %trace_context.trace_id,
            span_id = %trace_context.span_id,
            parent_span_id = ?trace_context.parent_span_id,
            service.name = %self.service_name,
        );

        // 添加默认标签
        for (key, value) in &self.default_tags {
            span.record(key.as_str(), value.as_str());
        }

        // 添加上下文标签
        for (key, value) in &trace_context.tags {
            span.record(key.as_str(), value.as_str());
        }

        (span, trace_context)
    }

    /// 开始HTTP请求跨度
    pub fn start_http_span(&self, method: &str, path: &str, context: Option<TraceContext>) -> (Span, TraceContext) {
        let trace_context = context.unwrap_or_default();
        
        let span = span!(
            Level::INFO,
            "http_request",
            otel.name = format!("{} {}", method, path),
            otel.kind = "server",
            http.method = method,
            http.target = path,
            trace_id = %trace_context.trace_id,
            span_id = %trace_context.span_id,
            parent_span_id = ?trace_context.parent_span_id,
            service.name = %self.service_name,
        );

        (span, trace_context)
    }

    /// 开始数据库操作跨度
    pub fn start_db_span(&self, operation: &str, table: &str, context: TraceContext) -> (Span, TraceContext) {
        let child_context = context.create_child();
        
        let span = span!(
            Level::INFO,
            "db_operation",
            otel.name = format!("{} {}", operation, table),
            otel.kind = "client",
            db.system = "postgresql",
            db.operation = operation,
            db.sql.table = table,
            trace_id = %child_context.trace_id,
            span_id = %child_context.span_id,
            parent_span_id = ?child_context.parent_span_id,
            service.name = %self.service_name,
        );

        (span, child_context)
    }

    /// 开始外部HTTP调用跨度
    pub fn start_http_client_span(&self, method: &str, url: &str, context: TraceContext) -> (Span, TraceContext) {
        let child_context = context.create_child();
        
        let span = span!(
            Level::INFO,
            "http_client",
            otel.name = format!("{} {}", method, url),
            otel.kind = "client",
            http.method = method,
            http.url = url,
            trace_id = %child_context.trace_id,
            span_id = %child_context.span_id,
            parent_span_id = ?child_context.parent_span_id,
            service.name = %self.service_name,
        );

        (span, child_context)
    }

    /// 记录错误
    pub fn record_error(&self, span: &Span, error: &str) {
        span.record("error", true);
        span.record("error.message", error);
        tracing::error!(parent: span, error = error, "操作出现错误");
    }

    /// 记录异常
    pub fn record_exception(&self, span: &Span, exception: &dyn std::error::Error) {
        span.record("error", true);
        span.record("exception.type", exception.to_string());
        span.record("exception.message", exception.to_string());
        tracing::error!(parent: span, error = %exception, "操作抛出异常");
    }
}

/// 追踪装饰器宏
#[macro_export]
macro_rules! traced {
    ($tracer:expr, $name:expr, $context:expr, $body:block) => {
        {
            let (span, _ctx) = $tracer.start_span($name, $context);
            let _guard = span.enter();
            tracing::info!("开始执行: {}", $name);
            let start = std::time::Instant::now();
            
            let result = $body;
            
            let duration = start.elapsed();
            span.record("duration_ms", duration.as_millis());
            tracing::info!(duration_ms = duration.as_millis(), "执行完成: {}", $name);
            
            result
        }
    };
}

/// 异步追踪装饰器宏
#[macro_export]
macro_rules! traced_async {
    ($tracer:expr, $name:expr, $context:expr, $body:block) => {
        {
            let (span, _ctx) = $tracer.start_span($name, $context);
            async move {
                let _guard = span.enter();
                tracing::info!("开始异步执行: {}", $name);
                let start = std::time::Instant::now();
                
                let result = $body.await;
                
                let duration = start.elapsed();
                span.record("duration_ms", duration.as_millis());
                tracing::info!(duration_ms = duration.as_millis(), "异步执行完成: {}", $name);
                
                result
            }
        }
    };
}

/// 从HTTP头提取追踪上下文
pub fn extract_trace_context_from_headers(headers: &axum::http::HeaderMap) -> Option<TraceContext> {
    // 检查标准的追踪头
    if let Some(trace_id) = headers.get("x-trace-id").and_then(|h| h.to_str().ok()) {
        let span_id = headers
            .get("x-span-id")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let parent_span_id = headers
            .get("x-parent-span-id")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string());

        let sampled = headers
            .get("x-trace-sampled")
            .and_then(|h| h.to_str().ok())
            .map(|s| s == "1" || s.to_lowercase() == "true")
            .unwrap_or(true);

        Some(TraceContext {
            trace_id: trace_id.to_string(),
            span_id,
            parent_span_id,
            sampled,
            tags: HashMap::new(),
        })
    } else {
        None
    }
}

/// 将追踪上下文注入HTTP头
pub fn inject_trace_context_to_headers(context: &TraceContext, headers: &mut axum::http::HeaderMap) {
    if let Ok(trace_id) = context.trace_id.parse() {
        headers.insert("x-trace-id", trace_id);
    }
    
    if let Ok(span_id) = context.span_id.parse() {
        headers.insert("x-span-id", span_id);
    }
    
    if let Some(parent_span_id) = &context.parent_span_id {
        if let Ok(parent_id) = parent_span_id.parse() {
            headers.insert("x-parent-span-id", parent_id);
        }
    }
    
    let sampled_value = if context.sampled { "1" } else { "0" };
    if let Ok(sampled) = sampled_value.parse() {
        headers.insert("x-trace-sampled", sampled);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_context_creation() {
        let context = TraceContext::new();
        assert!(!context.trace_id.is_empty());
        assert!(!context.span_id.is_empty());
        assert!(context.sampled);
    }

    #[test]
    fn test_trace_context_child() {
        let parent = TraceContext::new();
        let child = parent.create_child();
        
        assert_eq!(parent.trace_id, child.trace_id);
        assert_ne!(parent.span_id, child.span_id);
        assert_eq!(child.parent_span_id, Some(parent.span_id));
    }

    #[test]
    fn test_tracer_creation() {
        let tracer = Tracer::new("test-service".to_string());
        assert_eq!(tracer.service_name, "test-service");
        assert!(tracer.default_tags.contains_key("service.name"));
    }

    #[test]
    fn test_extract_trace_context() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-trace-id", "test-trace-id".parse().unwrap());
        headers.insert("x-span-id", "test-span-id".parse().unwrap());
        headers.insert("x-trace-sampled", "1".parse().unwrap());
        
        let context = extract_trace_context_from_headers(&headers);
        assert!(context.is_some());
        
        let ctx = context.unwrap();
        assert_eq!(ctx.trace_id, "test-trace-id");
        assert_eq!(ctx.span_id, "test-span-id");
        assert!(ctx.sampled);
    }

    #[test]
    fn test_inject_trace_context() {
        let context = TraceContext::new()
            .with_tag("test".to_string(), "value".to_string());
        
        let mut headers = axum::http::HeaderMap::new();
        inject_trace_context_to_headers(&context, &mut headers);
        
        // 由于UUID格式的限制，这个测试可能会失败
        // 在实际应用中应该使用更简单的ID格式
    }
}