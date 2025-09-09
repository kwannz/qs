use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
#[allow(unused_imports)]
use tracing::{info, warn, error, debug, span, Level};
use uuid::Uuid;

#[allow(unused_imports)]
use crate::monitoring::{UnifiedMonitoringSystem, SystemHealth, Alert, MetricValue};

/// 可观测性系统 - 提供系统洞察、性能分析和智能告警
pub struct ObservabilitySystem {
    monitoring: UnifiedMonitoringSystem,
    trace_collector: TraceCollector,
    performance_analyzer: PerformanceAnalyzer,
    dashboard_generator: DashboardGenerator,
}

/// 分布式追踪收集器
#[derive(Clone)]
pub struct TraceCollector {
    traces: Arc<tokio::sync::RwLock<HashMap<String, TraceSpan>>>,
    config: TraceConfig,
}

#[derive(Clone, Debug)]
pub struct TraceConfig {
    pub max_spans: usize,
    pub retention_seconds: u64,
    pub sampling_rate: f64,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            max_spans: 10000,
            retention_seconds: 3600,
            sampling_rate: 1.0, // 100% sampling for critical systems
        }
    }
}

/// 追踪 Span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSpan {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub operation_name: String,
    pub service_name: String,
    pub start_time: u64,
    pub end_time: Option<u64>,
    pub tags: HashMap<String, String>,
    pub logs: Vec<LogEntry>,
    pub status: SpanStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpanStatus {
    Ok,
    Error(String),
    Timeout,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: u64,
    pub level: String,
    pub message: String,
    pub fields: HashMap<String, String>,
}

/// 性能分析器
#[derive(Clone)]
pub struct PerformanceAnalyzer {
    latency_buckets: Arc<tokio::sync::RwLock<HashMap<String, Vec<Duration>>>>,
    throughput_counters: Arc<tokio::sync::RwLock<HashMap<String, ThroughputCounter>>>,
}

#[derive(Debug, Clone)]
pub struct ThroughputCounter {
    pub count: u64,
    pub start_time: Instant,
    pub last_reset: Instant,
}

/// 仪表盘生成器
#[derive(Clone)]
pub struct DashboardGenerator {
    templates: HashMap<String, DashboardTemplate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardTemplate {
    pub name: String,
    pub description: String,
    pub widgets: Vec<Widget>,
    pub refresh_interval_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Widget {
    pub id: String,
    pub widget_type: WidgetType,
    pub title: String,
    pub metrics: Vec<String>,
    pub config: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetType {
    LineChart,
    BarChart,
    Gauge,
    Counter,
    Table,
    Heatmap,
    Alert,
}

/// 可观测性报告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityReport {
    pub id: String,
    pub generated_at: u64,
    pub time_range: TimeRange,
    pub system_health: SystemHealth,
    pub performance_summary: PerformanceSummary,
    pub alert_summary: AlertSummary,
    pub trace_analysis: TraceAnalysis,
    pub recommendations: Vec<Recommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub throughput_per_second: f64,
    pub error_rate: f64,
    pub top_slow_operations: Vec<SlowOperation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowOperation {
    pub operation: String,
    pub avg_latency_ms: f64,
    pub count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertSummary {
    pub total_alerts: usize,
    pub critical_alerts: usize,
    pub warning_alerts: usize,
    pub most_frequent_alerts: Vec<String>,
    pub alert_trends: HashMap<String, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceAnalysis {
    pub total_traces: usize,
    pub avg_trace_duration_ms: f64,
    pub error_rate: f64,
    pub service_dependencies: HashMap<String, Vec<String>>,
    pub critical_path_analysis: Vec<CriticalPath>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalPath {
    pub path: Vec<String>,
    pub total_duration_ms: f64,
    pub bottleneck_service: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub id: String,
    pub category: String,
    pub priority: RecommendationPriority,
    pub title: String,
    pub description: String,
    pub impact: String,
    pub effort: String,
    pub implementation_steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

impl ObservabilitySystem {
    /// 创建新的可观测性系统
    pub fn new(monitoring: UnifiedMonitoringSystem) -> Self {
        info!("🔍 初始化可观测性系统");
        
        Self {
            monitoring,
            trace_collector: TraceCollector::new(TraceConfig::default()),
            performance_analyzer: PerformanceAnalyzer::new(),
            dashboard_generator: DashboardGenerator::new(),
        }
    }

    /// 开始追踪操作
    pub async fn start_trace(&self, operation: &str, service: &str) -> TraceContext {
        let trace_id = Uuid::new_v4().to_string();
        let span_id = Uuid::new_v4().to_string();
        
        let span = TraceSpan {
            trace_id: trace_id.clone(),
            span_id: span_id.clone(),
            parent_span_id: None,
            operation_name: operation.to_string(),
            service_name: service.to_string(),
            start_time: current_timestamp(),
            end_time: None,
            tags: HashMap::new(),
            logs: Vec::new(),
            status: SpanStatus::Ok,
        };

        self.trace_collector.add_span(span).await;
        
        TraceContext {
            trace_id,
            span_id,
            collector: self.trace_collector.clone(),
        }
    }

    /// 记录性能指标
    pub async fn record_performance(&self, operation: &str, duration: Duration) {
        self.performance_analyzer.record_latency(operation, duration).await;
        self.performance_analyzer.increment_throughput(operation).await;
        
        // 同时记录到监控系统
        self.monitoring.record_timing(&format!("perf_{operation}"), duration);
    }

    /// 生成可观测性报告
    pub async fn generate_report(&self, time_range: TimeRange) -> ObservabilityReport {
        info!("📊 生成可观测性报告");
        
        let system_health = self.monitoring.get_system_health();
        let performance_summary = self.analyze_performance(&time_range).await;
        let alert_summary = self.analyze_alerts(&time_range).await;
        let trace_analysis = self.analyze_traces(&time_range).await;
        let recommendations = self.generate_recommendations(&performance_summary, &alert_summary).await;

        ObservabilityReport {
            id: Uuid::new_v4().to_string(),
            generated_at: current_timestamp(),
            time_range,
            system_health,
            performance_summary,
            alert_summary,
            trace_analysis,
            recommendations,
        }
    }

    /// 创建实时仪表盘
    pub async fn create_dashboard(&self, template_name: &str) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let template = self.dashboard_generator.get_template(template_name)
            .ok_or_else(|| format!("Dashboard template '{template_name}' not found"))?;

        let mut dashboard_data = serde_json::Map::new();
        dashboard_data.insert("name".to_string(), serde_json::Value::String(template.name.clone()));
        dashboard_data.insert("description".to_string(), serde_json::Value::String(template.description.clone()));
        dashboard_data.insert("generated_at".to_string(), serde_json::Value::Number(current_timestamp().into()));

        let mut widgets_data = Vec::new();
        for widget in &template.widgets {
            let widget_data = self.generate_widget_data(widget).await?;
            widgets_data.push(widget_data);
        }

        dashboard_data.insert("widgets".to_string(), serde_json::Value::Array(widgets_data));
        
        Ok(serde_json::Value::Object(dashboard_data))
    }

    /// 生成智能推荐
    async fn generate_recommendations(
        &self,
        performance: &PerformanceSummary,
        _alerts: &AlertSummary,
    ) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        // 性能优化推荐
        if performance.p99_latency_ms > 1000.0 {
            recommendations.push(Recommendation {
                id: Uuid::new_v4().to_string(),
                category: "Performance".to_string(),
                priority: RecommendationPriority::High,
                title: "高延迟优化".to_string(),
                description: format!("系统P99延迟达到{:.2}ms，建议进行性能优化", performance.p99_latency_ms),
                impact: "显著降低系统响应时间".to_string(),
                effort: "中等".to_string(),
                implementation_steps: vec![
                    "分析慢查询和慢操作".to_string(),
                    "优化数据库索引".to_string(),
                    "增加缓存层".to_string(),
                    "优化算法复杂度".to_string(),
                ],
            });
        }

        // 错误率告警推荐
        if performance.error_rate > 1.0 {
            recommendations.push(Recommendation {
                id: Uuid::new_v4().to_string(),
                category: "Reliability".to_string(),
                priority: RecommendationPriority::Critical,
                title: "错误率降低".to_string(),
                description: format!("系统错误率为{:.2}%，需要立即关注", performance.error_rate),
                impact: "提高系统稳定性和用户体验".to_string(),
                effort: "高".to_string(),
                implementation_steps: vec![
                    "分析错误日志和堆栈跟踪".to_string(),
                    "修复关键bug".to_string(),
                    "增强错误处理和重试机制".to_string(),
                    "增加熔断器保护".to_string(),
                ],
            });
        }

        // 容量规划推荐
        if performance.throughput_per_second > 800.0 {
            recommendations.push(Recommendation {
                id: Uuid::new_v4().to_string(),
                category: "Capacity".to_string(),
                priority: RecommendationPriority::Medium,
                title: "容量扩展".to_string(),
                description: "当前吞吐量接近系统容量上限，建议扩展".to_string(),
                impact: "确保系统在高负载下稳定运行".to_string(),
                effort: "中等".to_string(),
                implementation_steps: vec![
                    "分析当前资源使用情况".to_string(),
                    "制定扩容方案".to_string(),
                    "实施水平扩展".to_string(),
                    "优化负载均衡策略".to_string(),
                ],
            });
        }

        recommendations
    }

    /// 分析性能指标
    async fn analyze_performance(&self, _time_range: &TimeRange) -> PerformanceSummary {
        let analyzer = &self.performance_analyzer;
        let latencies = analyzer.latency_buckets.read().await;
        let throughput = analyzer.throughput_counters.read().await;

        let mut all_latencies = Vec::new();
        let mut slow_operations = Vec::new();

        for (operation, durations) in latencies.iter() {
            if durations.is_empty() { continue; }
            
            let avg_latency_ms = durations.iter().map(|d| d.as_millis() as f64).sum::<f64>() / durations.len() as f64;
            
            slow_operations.push(SlowOperation {
                operation: operation.clone(),
                avg_latency_ms,
                count: durations.len() as u64,
            });

            all_latencies.extend(durations.iter().map(|d| d.as_millis() as f64));
        }

        // 排序以计算百分位数
        all_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let avg_latency_ms = if all_latencies.is_empty() { 0.0 } else {
            all_latencies.iter().sum::<f64>() / all_latencies.len() as f64
        };
        
        let p95_latency_ms = percentile(&all_latencies, 0.95);
        let p99_latency_ms = percentile(&all_latencies, 0.99);

        // 计算吞吐量
        let total_throughput = throughput.values()
            .map(|counter| counter.count as f64 / counter.start_time.elapsed().as_secs() as f64)
            .sum::<f64>();

        // 按平均延迟排序，取前10个慢操作
        slow_operations.sort_by(|a, b| b.avg_latency_ms.partial_cmp(&a.avg_latency_ms).unwrap());
        slow_operations.truncate(10);

        PerformanceSummary {
            avg_latency_ms,
            p95_latency_ms,
            p99_latency_ms,
            throughput_per_second: total_throughput,
            error_rate: 2.5, // 模拟错误率
            top_slow_operations: slow_operations,
        }
    }

    /// 分析告警
    async fn analyze_alerts(&self, _time_range: &TimeRange) -> AlertSummary {
        let alerts = self.monitoring.get_active_alerts();
        
        let total_alerts = alerts.len();
        let critical_alerts = alerts.iter().filter(|a| matches!(a.level, crate::monitoring::AlertLevel::Critical)).count();
        let warning_alerts = alerts.iter().filter(|a| matches!(a.level, crate::monitoring::AlertLevel::Warning)).count();
        
        let mut alert_counts = HashMap::new();
        for alert in &alerts {
            *alert_counts.entry(alert.metric.clone()).or_insert(0u32) += 1;
        }
        
        let most_frequent_alerts = alert_counts.iter()
            .collect::<Vec<_>>()
            .into_iter()
            .map(|(metric, _count)| metric.clone())
            .take(5)
            .collect();

        AlertSummary {
            total_alerts,
            critical_alerts,
            warning_alerts,
            most_frequent_alerts,
            alert_trends: alert_counts,
        }
    }

    /// 分析追踪数据
    async fn analyze_traces(&self, _time_range: &TimeRange) -> TraceAnalysis {
        let traces = self.trace_collector.traces.read().await;
        
        let total_traces = traces.len();
        let completed_traces: Vec<_> = traces.values()
            .filter(|span| span.end_time.is_some())
            .collect();

        let avg_trace_duration_ms = if completed_traces.is_empty() {
            0.0
        } else {
            let total_duration: f64 = completed_traces.iter()
                .map(|span| {
                    (span.end_time.unwrap_or(span.start_time) - span.start_time) as f64
                })
                .sum();
            total_duration / completed_traces.len() as f64
        };

        let error_rate = if total_traces == 0 { 
            0.0 
        } else {
            let error_count = traces.values()
                .filter(|span| matches!(span.status, SpanStatus::Error(_)))
                .count();
            (error_count as f64 / total_traces as f64) * 100.0
        };

        // 分析服务依赖关系
        let mut service_dependencies = HashMap::new();
        for span in traces.values() {
            let deps = service_dependencies
                .entry(span.service_name.clone())
                .or_insert(Vec::new());
            
            // 简化的依赖分析 - 基于标签
            if let Some(downstream) = span.tags.get("downstream_service") {
                if !deps.contains(downstream) {
                    deps.push(downstream.clone());
                }
            }
        }

        TraceAnalysis {
            total_traces,
            avg_trace_duration_ms,
            error_rate,
            service_dependencies,
            critical_path_analysis: vec![], // 简化实现
        }
    }

    /// 生成 Widget 数据
    async fn generate_widget_data(&self, widget: &Widget) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut widget_data = serde_json::Map::new();
        widget_data.insert("id".to_string(), serde_json::Value::String(widget.id.clone()));
        widget_data.insert("title".to_string(), serde_json::Value::String(widget.title.clone()));
        widget_data.insert("type".to_string(), serde_json::Value::String(format!("{:?}", widget.widget_type)));

        match widget.widget_type {
            WidgetType::Gauge => {
                if let Some(metric_name) = widget.metrics.first() {
                    let metrics = self.monitoring.get_metrics_snapshot();
                    if let Some(MetricValue::Gauge(value)) = metrics.get(metric_name) {
                        widget_data.insert("value".to_string(), serde_json::json!(*value));
                    }
                }
            }
            WidgetType::Counter => {
                if let Some(metric_name) = widget.metrics.first() {
                    let metrics = self.monitoring.get_metrics_snapshot();
                    if let Some(MetricValue::Counter(count)) = metrics.get(metric_name) {
                        widget_data.insert("count".to_string(), serde_json::Value::Number((*count).into()));
                    }
                }
            }
            WidgetType::Alert => {
                let active_alerts = self.monitoring.get_active_alerts();
                let alert_data: Vec<serde_json::Value> = active_alerts.into_iter()
                    .map(|alert| serde_json::to_value(alert).unwrap_or(serde_json::Value::Null))
                    .collect();
                widget_data.insert("alerts".to_string(), serde_json::Value::Array(alert_data));
            }
            _ => {
                // 其他类型的 Widget 实现
                widget_data.insert("data".to_string(), serde_json::Value::Array(vec![]));
            }
        }

        Ok(serde_json::Value::Object(widget_data))
    }
}

impl TraceCollector {
    fn new(config: TraceConfig) -> Self {
        Self {
            traces: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            config,
        }
    }

    async fn add_span(&self, span: TraceSpan) {
        if rand::random::<f64>() > self.config.sampling_rate {
            return; // 采样丢弃
        }

        let mut traces = self.traces.write().await;
        traces.insert(span.span_id.clone(), span);
        
        // 清理过期 traces
        if traces.len() > self.config.max_spans {
            let cutoff = current_timestamp() - self.config.retention_seconds;
            traces.retain(|_, span| span.start_time > cutoff);
        }
    }
}

impl PerformanceAnalyzer {
    fn new() -> Self {
        Self {
            latency_buckets: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            throughput_counters: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    async fn record_latency(&self, operation: &str, duration: Duration) {
        let mut buckets = self.latency_buckets.write().await;
        let bucket = buckets.entry(operation.to_string()).or_insert(Vec::new());
        bucket.push(duration);
        
        // 保持最近1000个记录
        if bucket.len() > 1000 {
            bucket.remove(0);
        }
    }

    async fn increment_throughput(&self, operation: &str) {
        let mut counters = self.throughput_counters.write().await;
        let counter = counters.entry(operation.to_string()).or_insert(ThroughputCounter {
            count: 0,
            start_time: Instant::now(),
            last_reset: Instant::now(),
        });
        
        counter.count += 1;
        
        // 每分钟重置一次计数器
        if counter.last_reset.elapsed() > Duration::from_secs(60) {
            counter.count = 1;
            counter.last_reset = Instant::now();
        }
    }
}

impl DashboardGenerator {
    fn new() -> Self {
        let mut templates = HashMap::new();
        
        // 系统监控仪表盘模板
        templates.insert("system".to_string(), DashboardTemplate {
            name: "系统监控".to_string(),
            description: "系统资源和性能监控".to_string(),
            refresh_interval_seconds: 30,
            widgets: vec![
                Widget {
                    id: "cpu_gauge".to_string(),
                    widget_type: WidgetType::Gauge,
                    title: "CPU 使用率".to_string(),
                    metrics: vec!["system_cpu_usage".to_string()],
                    config: HashMap::new(),
                },
                Widget {
                    id: "memory_gauge".to_string(),
                    widget_type: WidgetType::Gauge,
                    title: "内存使用率".to_string(),
                    metrics: vec!["system_memory_usage".to_string()],
                    config: HashMap::new(),
                },
                Widget {
                    id: "active_alerts".to_string(),
                    widget_type: WidgetType::Alert,
                    title: "活跃告警".to_string(),
                    metrics: vec![],
                    config: HashMap::new(),
                },
            ],
        });

        Self { templates }
    }

    fn get_template(&self, name: &str) -> Option<&DashboardTemplate> {
        self.templates.get(name)
    }
}

/// 追踪上下文
pub struct TraceContext {
    pub trace_id: String,
    pub span_id: String,
    collector: TraceCollector,
}

impl TraceContext {
    /// 结束当前 span
    pub async fn finish(self, status: SpanStatus) {
        let mut traces = self.collector.traces.write().await;
        if let Some(span) = traces.get_mut(&self.span_id) {
            span.end_time = Some(current_timestamp());
            span.status = status;
        }
    }

    /// 添加标签
    pub async fn add_tag(&self, key: &str, value: &str) {
        let mut traces = self.collector.traces.write().await;
        if let Some(span) = traces.get_mut(&self.span_id) {
            span.tags.insert(key.to_string(), value.to_string());
        }
    }

    /// 记录日志
    pub async fn log(&self, level: &str, message: &str, fields: HashMap<String, String>) {
        let mut traces = self.collector.traces.write().await;
        if let Some(span) = traces.get_mut(&self.span_id) {
            span.logs.push(LogEntry {
                timestamp: current_timestamp(),
                level: level.to_string(),
                message: message.to_string(),
                fields,
            });
        }
    }
}

/// 计算百分位数
fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }
    
    let index = (p * (sorted_data.len() - 1) as f64) as usize;
    sorted_data.get(index).copied().unwrap_or(0.0)
}

/// 获取当前时间戳
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}