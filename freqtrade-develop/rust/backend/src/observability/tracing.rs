//! 分布式链路追踪系统
//! 支持OpenTelemetry、Jaeger等追踪后端

use std::collections::HashMap;
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use anyhow::Result;
use uuid::Uuid;

/// 追踪上下文
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub baggage: HashMap<String, String>,
    pub sampling_priority: Option<f64>,
}

impl TraceContext {
    pub fn new() -> Self {
        Self {
            trace_id: Uuid::new_v4().to_string(),
            span_id: Uuid::new_v4().to_string(),
            parent_span_id: None,
            baggage: HashMap::new(),
            sampling_priority: Some(1.0),
        }
    }

    pub fn child(&self) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            span_id: Uuid::new_v4().to_string(),
            parent_span_id: Some(self.span_id.clone()),
            baggage: self.baggage.clone(),
            sampling_priority: self.sampling_priority,
        }
    }

    pub fn with_baggage(mut self, key: String, value: String) -> Self {
        self.baggage.insert(key, value);
        self
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Span状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SpanStatus {
    Ok,
    Error,
    Timeout,
    Cancelled,
}

/// Span数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    pub context: TraceContext,
    pub operation_name: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub duration_micros: Option<u64>,
    pub status: SpanStatus,
    pub tags: HashMap<String, String>,
    pub logs: Vec<SpanLog>,
}

/// Span日志事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanLog {
    pub timestamp: DateTime<Utc>,
    pub level: LogLevel,
    pub message: String,
    pub fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl Span {
    pub fn new(context: TraceContext, operation_name: String) -> Self {
        Self {
            context,
            operation_name,
            start_time: Utc::now(),
            end_time: None,
            duration_micros: None,
            status: SpanStatus::Ok,
            tags: HashMap::new(),
            logs: Vec::new(),
        }
    }

    pub fn with_tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }

    pub fn set_tag(&mut self, key: String, value: String) {
        self.tags.insert(key, value);
    }

    pub fn set_status(&mut self, status: SpanStatus) {
        self.status = status;
    }

    pub fn log(&mut self, level: LogLevel, message: String, fields: HashMap<String, serde_json::Value>) {
        self.logs.push(SpanLog {
            timestamp: Utc::now(),
            level,
            message,
            fields,
        });
    }

    pub fn log_info(&mut self, message: String) {
        self.log(LogLevel::Info, message, HashMap::new());
    }

    pub fn log_error(&mut self, message: String) {
        self.log(LogLevel::Error, message, HashMap::new());
        self.set_status(SpanStatus::Error);
    }

    pub fn finish(&mut self) {
        self.end_time = Some(Utc::now());
        if let Some(end_time) = self.end_time {
            self.duration_micros = Some(
                (end_time - self.start_time).num_microseconds().unwrap_or(0) as u64
            );
        }
    }
}

/// 追踪器接口
pub trait Tracer: Send + Sync {
    fn start_span(&self, operation_name: &str) -> Span;
    fn start_child_span(&self, parent: &TraceContext, operation_name: &str) -> Span;
    fn inject(&self, span: &Span, format: InjectFormat) -> Result<HashMap<String, String>>;
    fn extract(&self, format: ExtractFormat, carrier: &HashMap<String, String>) -> Result<Option<TraceContext>>;
    fn report(&self, span: &Span) -> Result<()>;
}

/// 注入格式
#[derive(Debug)]
pub enum InjectFormat {
    HttpHeaders,
    TextMap,
    Binary,
}

/// 提取格式  
#[derive(Debug)]
pub enum ExtractFormat {
    HttpHeaders,
    TextMap,
    Binary,
}

/// Jaeger追踪器实现
pub struct JaegerTracer {
    service_name: String,
    agent_endpoint: String,
    sampler: Box<dyn Sampler>,
    reporter: Box<dyn Reporter>,
}

impl JaegerTracer {
    pub fn new(
        service_name: String,
        agent_endpoint: String,
        sampler: Box<dyn Sampler>,
        reporter: Box<dyn Reporter>,
    ) -> Self {
        Self {
            service_name,
            agent_endpoint,
            sampler,
            reporter,
        }
    }
}

impl Tracer for JaegerTracer {
    fn start_span(&self, operation_name: &str) -> Span {
        let context = TraceContext::new();
        let mut span = Span::new(context, operation_name.to_string());
        
        // 添加服务标签
        span.set_tag("service.name".to_string(), self.service_name.clone());
        span.set_tag("service.version".to_string(), env!("CARGO_PKG_VERSION").to_string());
        
        // 应用采样决策
        if let Some(priority) = self.sampler.sample(&span) {
            span.context.sampling_priority = Some(priority);
        }
        
        span
    }

    fn start_child_span(&self, parent: &TraceContext, operation_name: &str) -> Span {
        let context = parent.child();
        let mut span = Span::new(context, operation_name.to_string());
        
        span.set_tag("service.name".to_string(), self.service_name.clone());
        span.set_tag("service.version".to_string(), env!("CARGO_PKG_VERSION").to_string());
        
        span
    }

    fn inject(&self, span: &Span, format: InjectFormat) -> Result<HashMap<String, String>> {
        match format {
            InjectFormat::HttpHeaders => {
                let mut headers = HashMap::new();
                headers.insert("uber-trace-id".to_string(), 
                    format!("{}:{}:{}:01", 
                        span.context.trace_id,
                        span.context.span_id,
                        span.context.parent_span_id.as_deref().unwrap_or("0")
                    )
                );
                
                // 添加baggage
                for (key, value) in &span.context.baggage {
                    headers.insert(format!("uberctx-{}", key), value.clone());
                }
                
                Ok(headers)
            }
            InjectFormat::TextMap => {
                let mut carrier = HashMap::new();
                carrier.insert("trace_id".to_string(), span.context.trace_id.clone());
                carrier.insert("span_id".to_string(), span.context.span_id.clone());
                
                if let Some(parent_id) = &span.context.parent_span_id {
                    carrier.insert("parent_span_id".to_string(), parent_id.clone());
                }
                
                Ok(carrier)
            }
            InjectFormat::Binary => {
                // 二进制格式实现
                todo!("Binary format not implemented")
            }
        }
    }

    fn extract(&self, format: ExtractFormat, carrier: &HashMap<String, String>) -> Result<Option<TraceContext>> {
        match format {
            ExtractFormat::HttpHeaders => {
                if let Some(trace_header) = carrier.get("uber-trace-id") {
                    let parts: Vec<&str> = trace_header.split(':').collect();
                    if parts.len() >= 2 {
                        let trace_id = parts[0].to_string();
                        let span_id = parts[1].to_string();
                        let parent_span_id = if parts.len() > 2 && parts[2] != "0" {
                            Some(parts[2].to_string())
                        } else {
                            None
                        };

                        let mut baggage = HashMap::new();
                        for (key, value) in carrier {
                            if key.starts_with("uberctx-") {
                                let baggage_key = key.strip_prefix("uberctx-").unwrap();
                                baggage.insert(baggage_key.to_string(), value.clone());
                            }
                        }

                        Ok(Some(TraceContext {
                            trace_id,
                            span_id,
                            parent_span_id,
                            baggage,
                            sampling_priority: Some(1.0),
                        }))
                    } else {
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            }
            ExtractFormat::TextMap => {
                if let (Some(trace_id), Some(span_id)) = (carrier.get("trace_id"), carrier.get("span_id")) {
                    Ok(Some(TraceContext {
                        trace_id: trace_id.clone(),
                        span_id: span_id.clone(),
                        parent_span_id: carrier.get("parent_span_id").cloned(),
                        baggage: HashMap::new(),
                        sampling_priority: Some(1.0),
                    }))
                } else {
                    Ok(None)
                }
            }
            ExtractFormat::Binary => {
                todo!("Binary format not implemented")
            }
        }
    }

    fn report(&self, span: &Span) -> Result<()> {
        self.reporter.report(span)
    }
}

/// 采样器接口
pub trait Sampler: Send + Sync {
    fn sample(&self, span: &Span) -> Option<f64>;
}

/// 概率采样器
pub struct ProbabilisticSampler {
    probability: f64,
}

impl ProbabilisticSampler {
    pub fn new(probability: f64) -> Self {
        Self {
            probability: probability.clamp(0.0, 1.0),
        }
    }
}

impl Sampler for ProbabilisticSampler {
    fn sample(&self, _span: &Span) -> Option<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() <= self.probability {
            Some(1.0)
        } else {
            Some(0.0)
        }
    }
}

/// 速率限制采样器
pub struct RateLimitingSampler {
    max_traces_per_second: f64,
    last_sample_time: std::sync::Mutex<Option<std::time::Instant>>,
}

impl RateLimitingSampler {
    pub fn new(max_traces_per_second: f64) -> Self {
        Self {
            max_traces_per_second,
            last_sample_time: std::sync::Mutex::new(None),
        }
    }
}

impl Sampler for RateLimitingSampler {
    fn sample(&self, _span: &Span) -> Option<f64> {
        let mut last_time = self.last_sample_time.lock().unwrap();
        let now = std::time::Instant::now();
        
        if let Some(last) = *last_time {
            let elapsed = now.duration_since(last).as_secs_f64();
            if elapsed < (1.0 / self.max_traces_per_second) {
                return Some(0.0); // 不采样
            }
        }
        
        *last_time = Some(now);
        Some(1.0) // 采样
    }
}

/// 报告器接口
pub trait Reporter: Send + Sync {
    fn report(&self, span: &Span) -> Result<()>;
}

/// HTTP报告器
pub struct HttpReporter {
    endpoint: String,
    client: reqwest::Client,
    buffer: Arc<std::sync::Mutex<Vec<Span>>>,
    buffer_size: usize,
    flush_interval: std::time::Duration,
}

impl HttpReporter {
    pub fn new(endpoint: String, buffer_size: usize, flush_interval: std::time::Duration) -> Self {
        let reporter = Self {
            endpoint,
            client: reqwest::Client::new(),
            buffer: Arc::new(std::sync::Mutex::new(Vec::new())),
            buffer_size,
            flush_interval,
        };

        // 启动定期刷新任务
        reporter.start_flush_task();
        reporter
    }

    fn start_flush_task(&self) {
        let buffer = Arc::clone(&self.buffer);
        let endpoint = self.endpoint.clone();
        let client = self.client.clone();
        let flush_interval = self.flush_interval;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(flush_interval);
            loop {
                interval.tick().await;
                
                let spans_to_send = {
                    let mut buffer_guard = buffer.lock().unwrap();
                    if buffer_guard.is_empty() {
                        continue;
                    }
                    let spans = buffer_guard.drain(..).collect::<Vec<_>>();
                    spans
                };

                if !spans_to_send.is_empty() {
                    if let Err(e) = Self::send_spans(&client, &endpoint, spans_to_send).await {
                        tracing::error!("Failed to send spans: {}", e);
                    }
                }
            }
        });
    }

    async fn send_spans(client: &reqwest::Client, endpoint: &str, spans: Vec<Span>) -> Result<()> {
        let payload = serde_json::to_string(&spans)?;
        
        let response = client
            .post(endpoint)
            .header("Content-Type", "application/json")
            .body(payload)
            .send()
            .await?;

        if response.status().is_success() {
            tracing::debug!("Successfully sent {} spans", spans.len());
        } else {
            tracing::error!("Failed to send spans: {}", response.status());
        }

        Ok(())
    }
}

impl Reporter for HttpReporter {
    fn report(&self, span: &Span) -> Result<()> {
        let mut buffer = self.buffer.lock().unwrap();
        buffer.push(span.clone());
        
        // 如果缓冲区满了，立即发送
        if buffer.len() >= self.buffer_size {
            let spans_to_send = buffer.drain(..).collect::<Vec<_>>();
            drop(buffer); // 释放锁
            
            let client = self.client.clone();
            let endpoint = self.endpoint.clone();
            
            tokio::spawn(async move {
                if let Err(e) = Self::send_spans(&client, &endpoint, spans_to_send).await {
                    tracing::error!("Failed to send spans: {}", e);
                }
            });
        }
        
        Ok(())
    }
}

/// 全局追踪器管理
pub struct TracingManager {
    tracer: Arc<dyn Tracer>,
    active_spans: Arc<std::sync::RwLock<HashMap<String, Span>>>,
}

impl TracingManager {
    pub fn new(tracer: Arc<dyn Tracer>) -> Self {
        Self {
            tracer,
            active_spans: Arc::new(std::sync::RwLock::new(HashMap::new())),
        }
    }

    pub fn start_span(&self, operation_name: &str) -> String {
        let span = self.tracer.start_span(operation_name);
        let span_id = span.context.span_id.clone();
        
        {
            let mut spans = self.active_spans.write().unwrap();
            spans.insert(span_id.clone(), span);
        }
        
        span_id
    }

    pub fn start_child_span(&self, parent_span_id: &str, operation_name: &str) -> Option<String> {
        let parent_context = {
            let spans = self.active_spans.read().unwrap();
            spans.get(parent_span_id).map(|span| span.context.clone())
        }?;

        let span = self.tracer.start_child_span(&parent_context, operation_name);
        let span_id = span.context.span_id.clone();
        
        {
            let mut spans = self.active_spans.write().unwrap();
            spans.insert(span_id.clone(), span);
        }
        
        Some(span_id)
    }

    pub fn finish_span(&self, span_id: &str) -> Result<()> {
        let mut span = {
            let mut spans = self.active_spans.write().unwrap();
            spans.remove(span_id)
                .ok_or_else(|| anyhow::anyhow!("Span {} not found", span_id))?
        };

        span.finish();
        self.tracer.report(&span)?;
        Ok(())
    }

    pub fn set_span_tag(&self, span_id: &str, key: String, value: String) -> Result<()> {
        let mut spans = self.active_spans.write().unwrap();
        let span = spans.get_mut(span_id)
            .ok_or_else(|| anyhow::anyhow!("Span {} not found", span_id))?;
        
        span.set_tag(key, value);
        Ok(())
    }

    pub fn log_span(&self, span_id: &str, level: LogLevel, message: String) -> Result<()> {
        let mut spans = self.active_spans.write().unwrap();
        let span = spans.get_mut(span_id)
            .ok_or_else(|| anyhow::anyhow!("Span {} not found", span_id))?;
        
        span.log(level, message, HashMap::new());
        Ok(())
    }

    pub fn inject_span_context(&self, span_id: &str, format: InjectFormat) -> Result<HashMap<String, String>> {
        let spans = self.active_spans.read().unwrap();
        let span = spans.get(span_id)
            .ok_or_else(|| anyhow::anyhow!("Span {} not found", span_id))?;
        
        self.tracer.inject(span, format)
    }
}

/// 追踪宏
#[macro_export]
macro_rules! trace_span {
    ($tracer:expr, $operation:expr, $code:block) => {{
        let span_id = $tracer.start_span($operation);
        let result = $code;
        let _ = $tracer.finish_span(&span_id);
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_trace_context_creation() {
        let context = TraceContext::new();
        assert!(!context.trace_id.is_empty());
        assert!(!context.span_id.is_empty());
        assert!(context.parent_span_id.is_none());
        assert_eq!(context.sampling_priority, Some(1.0));
    }

    #[test]
    fn test_trace_context_child() {
        let parent = TraceContext::new();
        let child = parent.child();
        
        assert_eq!(child.trace_id, parent.trace_id);
        assert_ne!(child.span_id, parent.span_id);
        assert_eq!(child.parent_span_id, Some(parent.span_id));
    }

    #[test]
    fn test_span_creation_and_finishing() {
        let context = TraceContext::new();
        let mut span = Span::new(context, "test_operation".to_string());
        
        span.set_tag("test.tag".to_string(), "test_value".to_string());
        span.log_info("Test log message".to_string());
        
        assert_eq!(span.operation_name, "test_operation");
        assert_eq!(span.status, SpanStatus::Ok);
        assert!(span.end_time.is_none());
        
        span.finish();
        assert!(span.end_time.is_some());
        assert!(span.duration_micros.is_some());
    }

    #[test]
    fn test_probabilistic_sampler() {
        let sampler = ProbabilisticSampler::new(0.5);
        let context = TraceContext::new();
        let span = Span::new(context, "test".to_string());
        
        // 应该返回0.0或1.0
        let result = sampler.sample(&span);
        assert!(result.is_some());
        let priority = result.unwrap();
        assert!(priority == 0.0 || priority == 1.0);
    }

    #[test]
    fn test_rate_limiting_sampler() {
        let sampler = RateLimitingSampler::new(10.0); // 10 traces per second
        let context = TraceContext::new();
        let span = Span::new(context, "test".to_string());
        
        // 第一个请求应该被采样
        assert_eq!(sampler.sample(&span), Some(1.0));
        
        // 立即第二个请求应该被限制
        assert_eq!(sampler.sample(&span), Some(0.0));
    }

    #[tokio::test]
    async fn test_jaeger_tracer_injection() {
        let sampler = Box::new(ProbabilisticSampler::new(1.0));
        let reporter = Box::new(HttpReporter::new(
            "http://localhost:14268/api/traces".to_string(),
            100,
            Duration::from_secs(1),
        ));
        
        let tracer = JaegerTracer::new(
            "test_service".to_string(),
            "localhost:6832".to_string(),
            sampler,
            reporter,
        );
        
        let span = tracer.start_span("test_operation");
        let headers = tracer.inject(&span, InjectFormat::HttpHeaders).unwrap();
        
        assert!(headers.contains_key("uber-trace-id"));
        
        let extracted = tracer.extract(ExtractFormat::HttpHeaders, &headers).unwrap();
        assert!(extracted.is_some());
        
        let extracted_context = extracted.unwrap();
        assert_eq!(extracted_context.trace_id, span.context.trace_id);
    }

    #[test]
    fn test_tracing_manager() {
        let sampler = Box::new(ProbabilisticSampler::new(1.0));
        let reporter = Box::new(HttpReporter::new(
            "http://localhost:14268/api/traces".to_string(),
            100,
            Duration::from_secs(1),
        ));
        
        let tracer = Arc::new(JaegerTracer::new(
            "test_service".to_string(),
            "localhost:6832".to_string(),
            sampler,
            reporter,
        ));
        
        let manager = TracingManager::new(tracer);
        
        let span_id = manager.start_span("parent_operation");
        assert!(!span_id.is_empty());
        
        let child_span_id = manager.start_child_span(&span_id, "child_operation").unwrap();
        assert!(!child_span_id.is_empty());
        assert_ne!(span_id, child_span_id);
        
        manager.set_span_tag(&span_id, "test.key".to_string(), "test.value".to_string()).unwrap();
        manager.log_span(&span_id, LogLevel::Info, "Test message".to_string()).unwrap();
        
        manager.finish_span(&child_span_id).unwrap();
        manager.finish_span(&span_id).unwrap();
    }
}

// 模拟reqwest模块
#[allow(dead_code)]
mod reqwest {
    use std::collections::HashMap;
    use anyhow::Result;

    #[derive(Clone)]
    pub struct Client;
    
    impl Client {
        pub fn new() -> Self { Self }
        
        pub fn post(&self, _url: &str) -> RequestBuilder {
            RequestBuilder
        }
    }

    pub struct RequestBuilder;
    
    impl RequestBuilder {
        pub fn header(self, _name: &str, _value: &str) -> Self { self }
        pub fn body(self, _body: String) -> Self { self }
        pub async fn send(self) -> Result<Response> {
            Ok(Response)
        }
    }

    pub struct Response;
    
    impl Response {
        pub fn status(&self) -> StatusCode {
            StatusCode::Ok
        }
    }

    #[derive(Debug)]
    pub enum StatusCode {
        Ok,
    }

    impl StatusCode {
        pub fn is_success(&self) -> bool {
            matches!(self, StatusCode::Ok)
        }
    }

    impl std::fmt::Display for StatusCode {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "200 OK")
        }
    }
}