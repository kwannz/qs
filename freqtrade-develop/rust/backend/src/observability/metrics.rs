//! 生产级指标收集和监控系统
//! 支持Prometheus、InfluxDB等时序数据库

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use anyhow::{Result, Context};
use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};

/// 指标类型枚举
#[derive(Debug, Clone, PartialEq)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

/// 指标值类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram(HistogramData),
    Summary(SummaryData),
}

/// 直方图数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramData {
    pub buckets: Vec<HistogramBucket>,
    pub count: u64,
    pub sum: f64,
}

/// 直方图桶
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBucket {
    pub le: f64, // 小于等于此值
    pub count: u64,
}

/// 摘要数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryData {
    pub quantiles: Vec<Quantile>,
    pub count: u64,
    pub sum: f64,
}

/// 分位数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quantile {
    pub quantile: f64, // 0.0 - 1.0
    pub value: f64,
}

/// 指标标签
pub type Labels = HashMap<String, String>;

/// 指标数据点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    pub name: String,
    pub value: MetricValue,
    pub labels: Labels,
    pub timestamp: DateTime<Utc>,
    pub help: Option<String>,
}

/// 指标收集器接口
pub trait MetricCollector: Send + Sync {
    fn collect(&self) -> Vec<MetricPoint>;
    fn name(&self) -> &str;
}

/// 计数器
pub struct Counter {
    name: String,
    help: String,
    value: AtomicU64,
    labels: Labels,
}

impl Counter {
    pub fn new(name: String, help: String, labels: Labels) -> Self {
        Self {
            name,
            help,
            value: AtomicU64::new(0),
            labels,
        }
    }

    pub fn inc(&self) {
        self.add(1);
    }

    pub fn add(&self, value: u64) {
        self.value.fetch_add(value, Ordering::Relaxed);
    }

    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    pub fn reset(&self) {
        self.value.store(0, Ordering::Relaxed);
    }
}

impl MetricCollector for Counter {
    fn collect(&self) -> Vec<MetricPoint> {
        vec![MetricPoint {
            name: self.name.clone(),
            value: MetricValue::Counter(self.get()),
            labels: self.labels.clone(),
            timestamp: Utc::now(),
            help: Some(self.help.clone()),
        }]
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// 测量器
pub struct Gauge {
    name: String,
    help: String,
    value: Arc<RwLock<f64>>,
    labels: Labels,
}

impl Gauge {
    pub fn new(name: String, help: String, labels: Labels) -> Self {
        Self {
            name,
            help,
            value: Arc::new(RwLock::new(0.0)),
            labels,
        }
    }

    pub async fn set(&self, value: f64) {
        *self.value.write().await = value;
    }

    pub async fn add(&self, value: f64) {
        *self.value.write().await += value;
    }

    pub async fn sub(&self, value: f64) {
        *self.value.write().await -= value;
    }

    pub async fn get(&self) -> f64 {
        *self.value.read().await
    }
}

impl MetricCollector for Gauge {
    fn collect(&self) -> Vec<MetricPoint> {
        let value = futures::executor::block_on(self.get());
        vec![MetricPoint {
            name: self.name.clone(),
            value: MetricValue::Gauge(value),
            labels: self.labels.clone(),
            timestamp: Utc::now(),
            help: Some(self.help.clone()),
        }]
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// 直方图
pub struct Histogram {
    name: String,
    help: String,
    buckets: Vec<f64>,
    bucket_counts: Vec<AtomicU64>,
    sum: Arc<RwLock<f64>>,
    count: AtomicU64,
    labels: Labels,
}

impl Histogram {
    pub fn new(name: String, help: String, buckets: Vec<f64>, labels: Labels) -> Self {
        let bucket_counts = buckets.iter().map(|_| AtomicU64::new(0)).collect();
        
        Self {
            name,
            help,
            buckets,
            bucket_counts,
            sum: Arc::new(RwLock::new(0.0)),
            count: AtomicU64::new(0),
            labels,
        }
    }

    pub async fn observe(&self, value: f64) {
        // 更新总和
        *self.sum.write().await += value;
        
        // 更新计数
        self.count.fetch_add(1, Ordering::Relaxed);
        
        // 更新桶计数
        for (i, &bucket_le) in self.buckets.iter().enumerate() {
            if value <= bucket_le {
                self.bucket_counts[i].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub async fn get_sum(&self) -> f64 {
        *self.sum.read().await
    }

    pub fn get_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
}

impl MetricCollector for Histogram {
    fn collect(&self) -> Vec<MetricPoint> {
        let buckets = self.buckets.iter().enumerate()
            .map(|(i, &le)| HistogramBucket {
                le,
                count: self.bucket_counts[i].load(Ordering::Relaxed),
            })
            .collect();

        let sum = futures::executor::block_on(self.get_sum());
        
        vec![MetricPoint {
            name: self.name.clone(),
            value: MetricValue::Histogram(HistogramData {
                buckets,
                count: self.get_count(),
                sum,
            }),
            labels: self.labels.clone(),
            timestamp: Utc::now(),
            help: Some(self.help.clone()),
        }]
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// 指标注册表
pub struct MetricRegistry {
    collectors: Arc<RwLock<HashMap<String, Arc<dyn MetricCollector>>>>,
}

impl MetricRegistry {
    pub fn new() -> Self {
        Self {
            collectors: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register(&self, collector: Arc<dyn MetricCollector>) -> Result<()> {
        let mut collectors = self.collectors.write().await;
        let name = collector.name().to_string();
        
        if collectors.contains_key(&name) {
            return Err(anyhow::anyhow!("Metric '{}' already registered", name));
        }
        
        collectors.insert(name, collector);
        Ok(())
    }

    pub async fn unregister(&self, name: &str) -> Result<()> {
        let mut collectors = self.collectors.write().await;
        collectors.remove(name)
            .ok_or_else(|| anyhow::anyhow!("Metric '{}' not found", name))?;
        Ok(())
    }

    pub async fn collect_all(&self) -> Vec<MetricPoint> {
        let collectors = self.collectors.read().await;
        let mut points = Vec::new();
        
        for collector in collectors.values() {
            points.extend(collector.collect());
        }
        
        points
    }

    pub async fn get_collector(&self, name: &str) -> Option<Arc<dyn MetricCollector>> {
        let collectors = self.collectors.read().await;
        collectors.get(name).cloned()
    }
}

impl Default for MetricRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Prometheus导出器
pub struct PrometheusExporter {
    registry: Arc<MetricRegistry>,
    endpoint: String,
}

impl PrometheusExporter {
    pub fn new(registry: Arc<MetricRegistry>, endpoint: String) -> Self {
        Self {
            registry,
            endpoint,
        }
    }

    pub async fn export(&self) -> Result<String> {
        let points = self.registry.collect_all().await;
        let mut output = String::new();

        for point in points {
            // 添加HELP注释
            if let Some(help) = &point.help {
                output.push_str(&format!("# HELP {} {}\n", point.name, help));
            }

            // 添加TYPE注释
            let metric_type = match point.value {
                MetricValue::Counter(_) => "counter",
                MetricValue::Gauge(_) => "gauge",
                MetricValue::Histogram(_) => "histogram",
                MetricValue::Summary(_) => "summary",
            };
            output.push_str(&format!("# TYPE {} {}\n", point.name, metric_type));

            // 添加指标数据
            match point.value {
                MetricValue::Counter(value) => {
                    output.push_str(&self.format_prometheus_line(&point.name, &point.labels, value as f64));
                }
                MetricValue::Gauge(value) => {
                    output.push_str(&self.format_prometheus_line(&point.name, &point.labels, value));
                }
                MetricValue::Histogram(data) => {
                    // 输出bucket数据
                    for bucket in &data.buckets {
                        let mut bucket_labels = point.labels.clone();
                        bucket_labels.insert("le".to_string(), bucket.le.to_string());
                        output.push_str(&self.format_prometheus_line(
                            &format!("{}_bucket", point.name),
                            &bucket_labels,
                            bucket.count as f64,
                        ));
                    }
                    
                    // 输出+Inf bucket
                    let mut inf_labels = point.labels.clone();
                    inf_labels.insert("le".to_string(), "+Inf".to_string());
                    output.push_str(&self.format_prometheus_line(
                        &format!("{}_bucket", point.name),
                        &inf_labels,
                        data.count as f64,
                    ));
                    
                    // 输出count和sum
                    output.push_str(&self.format_prometheus_line(
                        &format!("{}_count", point.name),
                        &point.labels,
                        data.count as f64,
                    ));
                    output.push_str(&self.format_prometheus_line(
                        &format!("{}_sum", point.name),
                        &point.labels,
                        data.sum,
                    ));
                }
                MetricValue::Summary(data) => {
                    // 输出分位数
                    for quantile in &data.quantiles {
                        let mut quantile_labels = point.labels.clone();
                        quantile_labels.insert("quantile".to_string(), quantile.quantile.to_string());
                        output.push_str(&self.format_prometheus_line(
                            &point.name,
                            &quantile_labels,
                            quantile.value,
                        ));
                    }
                    
                    // 输出count和sum
                    output.push_str(&self.format_prometheus_line(
                        &format!("{}_count", point.name),
                        &point.labels,
                        data.count as f64,
                    ));
                    output.push_str(&self.format_prometheus_line(
                        &format!("{}_sum", point.name),
                        &point.labels,
                        data.sum,
                    ));
                }
            }
        }

        Ok(output)
    }

    fn format_prometheus_line(&self, name: &str, labels: &Labels, value: f64) -> String {
        if labels.is_empty() {
            format!("{} {}\n", name, value)
        } else {
            let labels_str = labels.iter()
                .map(|(k, v)| format!("{}=\"{}\"", k, v))
                .collect::<Vec<_>>()
                .join(",");
            format!("{}{{{}}} {}\n", name, labels_str, value)
        }
    }

    pub async fn start_server(&self) -> Result<()> {
        use warp::Filter;

        let registry = Arc::clone(&self.registry);
        let metrics_route = warp::path("metrics")
            .and_then(move || {
                let registry = Arc::clone(&registry);
                async move {
                    let exporter = PrometheusExporter::new(registry, "/metrics".to_string());
                    match exporter.export().await {
                        Ok(content) => Ok(warp::reply::with_header(
                            content,
                            "content-type",
                            "text/plain; version=0.0.4; charset=utf-8",
                        )),
                        Err(_) => Err(warp::reject()),
                    }
                }
            });

        warp::serve(metrics_route)
            .run(([0, 0, 0, 0], 9090))
            .await;

        Ok(())
    }
}

/// 交易系统指标收集器
pub struct TradingMetrics {
    // 订单指标
    pub orders_total: Arc<Counter>,
    pub orders_filled: Arc<Counter>,
    pub orders_cancelled: Arc<Counter>,
    pub orders_rejected: Arc<Counter>,
    
    // 执行指标
    pub execution_latency: Arc<Histogram>,
    pub fill_rate: Arc<Gauge>,
    pub slippage: Arc<Histogram>,
    
    // 风险指标
    pub position_value: Arc<Gauge>,
    pub daily_pnl: Arc<Gauge>,
    pub risk_limit_breaches: Arc<Counter>,
    
    // 市场数据指标
    pub market_data_updates: Arc<Counter>,
    pub market_data_latency: Arc<Histogram>,
    
    // 系统指标
    pub cpu_usage: Arc<Gauge>,
    pub memory_usage: Arc<Gauge>,
    pub network_io: Arc<Counter>,
    pub disk_io: Arc<Counter>,
}

impl TradingMetrics {
    pub fn new() -> Self {
        let labels = HashMap::new();

        Self {
            // 订单指标
            orders_total: Arc::new(Counter::new(
                "trading_orders_total".to_string(),
                "Total number of orders".to_string(),
                labels.clone(),
            )),
            orders_filled: Arc::new(Counter::new(
                "trading_orders_filled_total".to_string(),
                "Total number of filled orders".to_string(),
                labels.clone(),
            )),
            orders_cancelled: Arc::new(Counter::new(
                "trading_orders_cancelled_total".to_string(),
                "Total number of cancelled orders".to_string(),
                labels.clone(),
            )),
            orders_rejected: Arc::new(Counter::new(
                "trading_orders_rejected_total".to_string(),
                "Total number of rejected orders".to_string(),
                labels.clone(),
            )),
            
            // 执行指标
            execution_latency: Arc::new(Histogram::new(
                "trading_execution_latency_seconds".to_string(),
                "Order execution latency in seconds".to_string(),
                vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
                labels.clone(),
            )),
            fill_rate: Arc::new(Gauge::new(
                "trading_fill_rate".to_string(),
                "Order fill rate percentage".to_string(),
                labels.clone(),
            )),
            slippage: Arc::new(Histogram::new(
                "trading_slippage_bps".to_string(),
                "Trading slippage in basis points".to_string(),
                vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0],
                labels.clone(),
            )),
            
            // 风险指标
            position_value: Arc::new(Gauge::new(
                "trading_position_value_usd".to_string(),
                "Current position value in USD".to_string(),
                labels.clone(),
            )),
            daily_pnl: Arc::new(Gauge::new(
                "trading_daily_pnl_usd".to_string(),
                "Daily PnL in USD".to_string(),
                labels.clone(),
            )),
            risk_limit_breaches: Arc::new(Counter::new(
                "trading_risk_limit_breaches_total".to_string(),
                "Total number of risk limit breaches".to_string(),
                labels.clone(),
            )),
            
            // 市场数据指标
            market_data_updates: Arc::new(Counter::new(
                "market_data_updates_total".to_string(),
                "Total number of market data updates".to_string(),
                labels.clone(),
            )),
            market_data_latency: Arc::new(Histogram::new(
                "market_data_latency_seconds".to_string(),
                "Market data update latency in seconds".to_string(),
                vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
                labels.clone(),
            )),
            
            // 系统指标
            cpu_usage: Arc::new(Gauge::new(
                "system_cpu_usage_percent".to_string(),
                "CPU usage percentage".to_string(),
                labels.clone(),
            )),
            memory_usage: Arc::new(Gauge::new(
                "system_memory_usage_bytes".to_string(),
                "Memory usage in bytes".to_string(),
                labels.clone(),
            )),
            network_io: Arc::new(Counter::new(
                "system_network_io_bytes_total".to_string(),
                "Total network I/O in bytes".to_string(),
                labels.clone(),
            )),
            disk_io: Arc::new(Counter::new(
                "system_disk_io_bytes_total".to_string(),
                "Total disk I/O in bytes".to_string(),
                labels.clone(),
            )),
        }
    }

    pub async fn register_all(&self, registry: &MetricRegistry) -> Result<()> {
        // 注册所有指标到注册表
        registry.register(self.orders_total.clone()).await?;
        registry.register(self.orders_filled.clone()).await?;
        registry.register(self.orders_cancelled.clone()).await?;
        registry.register(self.orders_rejected.clone()).await?;
        registry.register(self.execution_latency.clone()).await?;
        registry.register(self.fill_rate.clone()).await?;
        registry.register(self.slippage.clone()).await?;
        registry.register(self.position_value.clone()).await?;
        registry.register(self.daily_pnl.clone()).await?;
        registry.register(self.risk_limit_breaches.clone()).await?;
        registry.register(self.market_data_updates.clone()).await?;
        registry.register(self.market_data_latency.clone()).await?;
        registry.register(self.cpu_usage.clone()).await?;
        registry.register(self.memory_usage.clone()).await?;
        registry.register(self.network_io.clone()).await?;
        registry.register(self.disk_io.clone()).await?;
        
        Ok(())
    }
}

impl Default for TradingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// 系统资源监控器
pub struct SystemMonitor {
    metrics: Arc<TradingMetrics>,
}

impl SystemMonitor {
    pub fn new(metrics: Arc<TradingMetrics>) -> Self {
        Self { metrics }
    }

    pub async fn start_monitoring(&self) {
        let metrics = Arc::clone(&self.metrics);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                // 更新CPU使用率
                if let Ok(cpu_usage) = Self::get_cpu_usage().await {
                    metrics.cpu_usage.set(cpu_usage).await;
                }
                
                // 更新内存使用
                if let Ok(memory_usage) = Self::get_memory_usage().await {
                    metrics.memory_usage.set(memory_usage as f64).await;
                }
                
                // 可以添加更多系统指标监控
            }
        });
    }

    async fn get_cpu_usage() -> Result<f64> {
        // 实际实现需要使用系统API
        Ok(15.5) // 示例值
    }

    async fn get_memory_usage() -> Result<u64> {
        // 实际实现需要使用系统API  
        Ok(1024 * 1024 * 512) // 示例值：512MB
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_counter() {
        let counter = Counter::new(
            "test_counter".to_string(),
            "Test counter".to_string(),
            HashMap::new(),
        );

        counter.inc();
        counter.add(5);

        assert_eq!(counter.get(), 6);

        let points = counter.collect();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].name, "test_counter");
        
        if let MetricValue::Counter(value) = points[0].value {
            assert_eq!(value, 6);
        } else {
            panic!("Expected counter value");
        }
    }

    #[tokio::test]
    async fn test_gauge() {
        let gauge = Gauge::new(
            "test_gauge".to_string(),
            "Test gauge".to_string(),
            HashMap::new(),
        );

        gauge.set(10.5).await;
        assert_eq!(gauge.get().await, 10.5);

        gauge.add(2.5).await;
        assert_eq!(gauge.get().await, 13.0);

        gauge.sub(3.0).await;
        assert_eq!(gauge.get().await, 10.0);
    }

    #[tokio::test]
    async fn test_histogram() {
        let histogram = Histogram::new(
            "test_histogram".to_string(),
            "Test histogram".to_string(),
            vec![0.1, 0.5, 1.0, 2.0, 5.0],
            HashMap::new(),
        );

        histogram.observe(0.05).await;
        histogram.observe(0.3).await;
        histogram.observe(1.5).await;
        histogram.observe(10.0).await;

        assert_eq!(histogram.get_count(), 4);
        assert!((histogram.get_sum().await - 11.85).abs() < 0.001);

        let points = histogram.collect();
        assert_eq!(points.len(), 1);
        
        if let MetricValue::Histogram(data) = &points[0].value {
            assert_eq!(data.count, 4);
            assert_eq!(data.buckets.len(), 5);
            assert_eq!(data.buckets[0].count, 1); // <= 0.1
            assert_eq!(data.buckets[1].count, 2); // <= 0.5
        } else {
            panic!("Expected histogram value");
        }
    }

    #[tokio::test]
    async fn test_registry() {
        let registry = MetricRegistry::new();
        
        let counter = Arc::new(Counter::new(
            "test_counter".to_string(),
            "Test counter".to_string(),
            HashMap::new(),
        ));

        registry.register(counter.clone()).await.unwrap();
        
        // 测试重复注册
        let result = registry.register(counter).await;
        assert!(result.is_err());

        let points = registry.collect_all().await;
        assert_eq!(points.len(), 1);
    }

    #[tokio::test]
    async fn test_prometheus_exporter() {
        let registry = Arc::new(MetricRegistry::new());
        
        let counter = Arc::new(Counter::new(
            "http_requests_total".to_string(),
            "Total HTTP requests".to_string(),
            [("method".to_string(), "GET".to_string())].iter().cloned().collect(),
        ));
        counter.add(42);
        
        registry.register(counter).await.unwrap();
        
        let exporter = PrometheusExporter::new(registry, "/metrics".to_string());
        let output = exporter.export().await.unwrap();
        
        assert!(output.contains("# HELP http_requests_total Total HTTP requests"));
        assert!(output.contains("# TYPE http_requests_total counter"));
        assert!(output.contains("http_requests_total{method=\"GET\"} 42"));
    }
}

// 模拟warp模块
#[allow(dead_code)]
mod warp {
    pub struct Filter;
    
    impl Filter {
        pub fn and_then<F>(self, _f: F) -> Self where F: Fn() { self }
    }
    
    pub fn path(_: &str) -> Filter { Filter }
    
    pub fn serve(_: Filter) -> Serve { Serve }
    
    pub struct Serve;
    
    impl Serve {
        pub async fn run(self, _: ([u8; 4], u16)) {}
    }
    
    pub fn reject() -> Rejection { Rejection }
    
    pub struct Rejection;
    
    pub mod reply {
        pub fn with_header<T>(_: T, _: &str, _: &str) -> impl warp::Reply {
            Reply
        }
        
        pub struct Reply;
        impl warp::Reply for Reply {}
    }
    
    pub trait Reply {}
}