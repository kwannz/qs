//! 性能监控和指标收集模块

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

#[cfg(feature = "prometheus")]
use prometheus_client::{
    encoding::text::encode,
    metrics::{counter::Counter, histogram::Histogram, gauge::Gauge, family::Family},
    registry::Registry,
};

/// 指标类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Histogram,
    Gauge,
}

/// 指标值
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    Counter(u64),
    Histogram(Vec<f64>),
    Gauge(f64),
}

/// 指标数据点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    pub name: String,
    pub labels: HashMap<String, String>,
    pub value: MetricValue,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// 性能指标收集器
pub struct MetricsCollector {
    /// 指标注册表
    #[cfg(feature = "prometheus")]
    registry: Registry,
    
    /// 内存中的指标存储
    metrics: Arc<RwLock<HashMap<String, MetricPoint>>>,
    
    /// 计数器
    counters: Arc<RwLock<HashMap<String, u64>>>,
    
    /// 直方图 (用于延迟统计)
    histograms: Arc<RwLock<HashMap<String, Vec<f64>>>>,
    
    /// 仪表盘 (用于实时数值)
    gauges: Arc<RwLock<HashMap<String, f64>>>,
}

impl MetricsCollector {
    /// 创建新的指标收集器
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "prometheus")]
            registry: Registry::default(),
            metrics: Arc::new(RwLock::new(HashMap::new())),
            counters: Arc::new(RwLock::new(HashMap::new())),
            histograms: Arc::new(RwLock::new(HashMap::new())),
            gauges: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 增加计数器
    pub async fn increment_counter(&self, name: &str, labels: HashMap<String, String>) {
        let key = format!("{}:{}", name, serde_json::to_string(&labels).unwrap_or_default());
        
        let mut counters = self.counters.write().await;
        let value = counters.entry(key.clone()).or_insert(0);
        *value += 1;

        // 更新指标存储
        let mut metrics = self.metrics.write().await;
        metrics.insert(key, MetricPoint {
            name: name.to_string(),
            labels,
            value: MetricValue::Counter(*value),
            timestamp: chrono::Utc::now(),
        });

        tracing::debug!(metric = name, value = *value, "计数器更新");
    }

    /// 记录直方图数据
    pub async fn record_histogram(&self, name: &str, value: f64, labels: HashMap<String, String>) {
        let key = format!("{}:{}", name, serde_json::to_string(&labels).unwrap_or_default());
        
        let mut histograms = self.histograms.write().await;
        let values = histograms.entry(key.clone()).or_insert_with(Vec::new);
        values.push(value);

        // 保持最近1000个数据点
        if values.len() > 1000 {
            values.remove(0);
        }

        // 更新指标存储
        let mut metrics = self.metrics.write().await;
        metrics.insert(key, MetricPoint {
            name: name.to_string(),
            labels,
            value: MetricValue::Histogram(values.clone()),
            timestamp: chrono::Utc::now(),
        });

        tracing::debug!(metric = name, value = value, "直方图数据记录");
    }

    /// 设置仪表盘数值
    pub async fn set_gauge(&self, name: &str, value: f64, labels: HashMap<String, String>) {
        let key = format!("{}:{}", name, serde_json::to_string(&labels).unwrap_or_default());
        
        let mut gauges = self.gauges.write().await;
        gauges.insert(key.clone(), value);

        // 更新指标存储
        let mut metrics = self.metrics.write().await;
        metrics.insert(key, MetricPoint {
            name: name.to_string(),
            labels,
            value: MetricValue::Gauge(value),
            timestamp: chrono::Utc::now(),
        });

        tracing::debug!(metric = name, value = value, "仪表盘数值设置");
    }

    /// 获取所有指标
    pub async fn get_all_metrics(&self) -> HashMap<String, MetricPoint> {
        self.metrics.read().await.clone()
    }

    /// 获取指标摘要
    pub async fn get_metrics_summary(&self) -> MetricsSummary {
        let counters = self.counters.read().await;
        let histograms = self.histograms.read().await;
        let gauges = self.gauges.read().await;

        MetricsSummary {
            total_counters: counters.len(),
            total_histograms: histograms.len(),
            total_gauges: gauges.len(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// 导出Prometheus格式指标
    #[cfg(feature = "prometheus")]
    pub fn export_prometheus(&self) -> String {
        let mut buffer = String::new();
        encode(&mut buffer, &self.registry).unwrap_or_default();
        buffer
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// 指标摘要
#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub total_counters: usize,
    pub total_histograms: usize,
    pub total_gauges: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// 性能计时器
pub struct Timer {
    name: String,
    labels: HashMap<String, String>,
    start: Instant,
    metrics: Arc<MetricsCollector>,
}

impl Timer {
    /// 创建新的计时器
    pub fn new(name: String, labels: HashMap<String, String>, metrics: Arc<MetricsCollector>) -> Self {
        Self {
            name,
            labels,
            start: Instant::now(),
            metrics,
        }
    }

    /// 完成计时并记录
    pub async fn finish(self) {
        let duration = self.start.elapsed();
        let duration_ms = duration.as_millis() as f64;
        
        self.metrics.record_histogram(&self.name, duration_ms, self.labels).await;
        
        tracing::debug!(
            metric = %self.name,
            duration_ms = duration_ms,
            "计时器完成"
        );
    }
}

/// HTTP请求指标收集器
pub struct HttpMetrics {
    metrics: Arc<MetricsCollector>,
}

impl HttpMetrics {
    /// 创建HTTP指标收集器
    pub fn new(metrics: Arc<MetricsCollector>) -> Self {
        Self { metrics }
    }

    /// 记录HTTP请求
    pub async fn record_request(&self, method: &str, path: &str, status_code: u16, duration: Duration) {
        let mut labels = HashMap::new();
        labels.insert("method".to_string(), method.to_string());
        labels.insert("path".to_string(), path.to_string());
        labels.insert("status_code".to_string(), status_code.to_string());

        // 计数器: 总请求数
        self.metrics.increment_counter("http_requests_total", labels.clone()).await;

        // 直方图: 请求延迟
        self.metrics.record_histogram("http_request_duration_ms", duration.as_millis() as f64, labels.clone()).await;

        // 按状态码计数
        let mut status_labels = HashMap::new();
        status_labels.insert("status_class".to_string(), format!("{}xx", status_code / 100));
        self.metrics.increment_counter("http_requests_by_status", status_labels).await;
    }

    /// 记录错误
    pub async fn record_error(&self, method: &str, path: &str, error_type: &str) {
        let mut labels = HashMap::new();
        labels.insert("method".to_string(), method.to_string());
        labels.insert("path".to_string(), path.to_string());
        labels.insert("error_type".to_string(), error_type.to_string());

        self.metrics.increment_counter("http_errors_total", labels).await;
    }
}

/// 交易指标收集器
pub struct TradingMetrics {
    metrics: Arc<MetricsCollector>,
}

impl TradingMetrics {
    /// 创建交易指标收集器
    pub fn new(metrics: Arc<MetricsCollector>) -> Self {
        Self { metrics }
    }

    /// 记录订单创建
    pub async fn record_order_created(&self, symbol: &str, side: &str, order_type: &str) {
        let mut labels = HashMap::new();
        labels.insert("symbol".to_string(), symbol.to_string());
        labels.insert("side".to_string(), side.to_string());
        labels.insert("order_type".to_string(), order_type.to_string());

        self.metrics.increment_counter("orders_created_total", labels).await;
    }

    /// 记录订单成交
    pub async fn record_order_filled(&self, symbol: &str, side: &str, quantity: f64, price: f64) {
        let mut labels = HashMap::new();
        labels.insert("symbol".to_string(), symbol.to_string());
        labels.insert("side".to_string(), side.to_string());

        // 计数器: 成交笔数
        self.metrics.increment_counter("orders_filled_total", labels.clone()).await;

        // 仪表盘: 最新价格
        let mut price_labels = HashMap::new();
        price_labels.insert("symbol".to_string(), symbol.to_string());
        self.metrics.set_gauge("latest_price", price, price_labels).await;

        // 直方图: 成交金额分布
        let amount = quantity * price;
        self.metrics.record_histogram("trade_amount_distribution", amount, labels).await;
    }

    /// 记录订单取消
    pub async fn record_order_cancelled(&self, symbol: &str, reason: &str) {
        let mut labels = HashMap::new();
        labels.insert("symbol".to_string(), symbol.to_string());
        labels.insert("reason".to_string(), reason.to_string());

        self.metrics.increment_counter("orders_cancelled_total", labels).await;
    }

    /// 更新持仓信息
    pub async fn update_position(&self, symbol: &str, quantity: f64, unrealized_pnl: f64) {
        let mut labels = HashMap::new();
        labels.insert("symbol".to_string(), symbol.to_string());

        // 仪表盘: 持仓数量
        self.metrics.set_gauge("position_quantity", quantity, labels.clone()).await;

        // 仪表盘: 未实现盈亏
        self.metrics.set_gauge("unrealized_pnl", unrealized_pnl, labels).await;
    }
}

/// 系统指标收集器
pub struct SystemMetrics {
    metrics: Arc<MetricsCollector>,
}

impl SystemMetrics {
    /// 创建系统指标收集器
    pub fn new(metrics: Arc<MetricsCollector>) -> Self {
        Self { metrics }
    }

    /// 启动系统指标收集
    pub async fn start_collection(&self) {
        let metrics = self.metrics.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                // 收集内存使用情况
                #[cfg(target_os = "linux")]
                {
                    if let Ok(memory_usage) = Self::get_memory_usage() {
                        metrics.set_gauge("memory_usage_bytes", memory_usage as f64, HashMap::new()).await;
                    }
                }

                // 收集CPU使用率
                #[cfg(target_os = "linux")]
                {
                    if let Ok(cpu_usage) = Self::get_cpu_usage() {
                        metrics.set_gauge("cpu_usage_percent", cpu_usage, HashMap::new()).await;
                    }
                }

                // 设置运行时间
                let uptime = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as f64;
                metrics.set_gauge("uptime_seconds", uptime, HashMap::new()).await;
            }
        });
    }

    /// 获取内存使用量 (Linux)
    #[cfg(target_os = "linux")]
    fn get_memory_usage() -> Result<u64, std::io::Error> {
        use std::fs;
        let contents = fs::read_to_string("/proc/self/status")?;
        for line in contents.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(size_str) = line.split_whitespace().nth(1) {
                    if let Ok(size_kb) = size_str.parse::<u64>() {
                        return Ok(size_kb * 1024); // 转换为字节
                    }
                }
            }
        }
        Ok(0)
    }

    /// 获取CPU使用率 (简化版本)
    #[cfg(target_os = "linux")]
    fn get_cpu_usage() -> Result<f64, std::io::Error> {
        // 简化实现，实际应该计算CPU使用率
        Ok(0.0)
    }
}

/// 指标中间件
pub async fn metrics_middleware<B>(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let start = Instant::now();
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    
    let response = next.run(request).await;
    
    let duration = start.elapsed();
    let status_code = response.status().as_u16();
    
    // 在实际应用中，这里应该从应用状态中获取metrics实例
    tracing::debug!(
        method = %method,
        path = %path,
        status_code = status_code,
        duration_ms = duration.as_millis(),
        "HTTP请求指标记录"
    );
    
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_metrics_collector() {
        let collector = MetricsCollector::new();
        
        // 测试计数器
        let mut labels = HashMap::new();
        labels.insert("test".to_string(), "value".to_string());
        collector.increment_counter("test_counter", labels.clone()).await;
        
        let metrics = collector.get_all_metrics().await;
        assert!(!metrics.is_empty());
    }

    #[tokio::test]
    async fn test_timer() {
        let collector = Arc::new(MetricsCollector::new());
        let labels = HashMap::new();
        
        let timer = Timer::new("test_timer".to_string(), labels, collector.clone());
        tokio::time::sleep(Duration::from_millis(10)).await;
        timer.finish().await;
        
        let metrics = collector.get_all_metrics().await;
        assert!(!metrics.is_empty());
    }

    #[tokio::test]
    async fn test_http_metrics() {
        let collector = Arc::new(MetricsCollector::new());
        let http_metrics = HttpMetrics::new(collector.clone());
        
        http_metrics.record_request("GET", "/api/test", 200, Duration::from_millis(100)).await;
        
        let metrics = collector.get_all_metrics().await;
        assert!(!metrics.is_empty());
    }
}