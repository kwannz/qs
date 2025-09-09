use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::dashboard::consistency_monitor::ConsistencyMetrics;

/// 指标聚合配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatorConfig {
    pub aggregation_interval: Duration,    // 聚合间隔
    pub retention_policy: RetentionPolicy, // 保留策略
    pub downsampling_rules: Vec<DownsamplingRule>, // 降采样规则
    pub rollup_rules: Vec<RollupRule>,     // 汇总规则
    pub storage_backend: StorageBackend,   // 存储后端
}

impl Default for AggregatorConfig {
    fn default() -> Self {
        Self {
            aggregation_interval: Duration::from_secs(60),
            retention_policy: RetentionPolicy::default(),
            downsampling_rules: vec![
                DownsamplingRule {
                    source_resolution: Duration::from_secs(5),
                    target_resolution: Duration::from_secs(60),
                    aggregation_function: AggregationFunction::Average,
                    retention_period: Duration::from_secs(24 * 3600), // 1天
                },
                DownsamplingRule {
                    source_resolution: Duration::from_secs(60),
                    target_resolution: Duration::from_secs(3600),
                    aggregation_function: AggregationFunction::Average,
                    retention_period: Duration::from_secs(30 * 24 * 3600), // 30天
                },
            ],
            rollup_rules: Vec::new(),
            storage_backend: StorageBackend::Memory,
        }
    }
}

/// 保留策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    pub raw_data_retention: Duration,      // 原始数据保留时间
    pub aggregated_data_retention: Duration, // 聚合数据保留时间
    pub max_memory_usage_mb: usize,        // 最大内存使用（MB）
    pub compression_enabled: bool,         // 启用压缩
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            raw_data_retention: Duration::from_secs(24 * 3600), // 1天
            aggregated_data_retention: Duration::from_secs(30 * 24 * 3600), // 30天
            max_memory_usage_mb: 1024, // 1GB
            compression_enabled: true,
        }
    }
}

/// 降采样规则
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownsamplingRule {
    pub source_resolution: Duration,
    pub target_resolution: Duration,
    pub aggregation_function: AggregationFunction,
    pub retention_period: Duration,
}

/// 汇总规则
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollupRule {
    pub name: String,
    pub source_metrics: Vec<String>,
    pub target_metric: String,
    pub aggregation_function: AggregationFunction,
    pub grouping_keys: Vec<String>,
    pub filters: Vec<MetricFilter>,
}

/// 聚合函数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationFunction {
    Sum,
    Average,
    Min,
    Max,
    Count,
    P50,
    P95,
    P99,
    StdDev,
    Rate,
    Delta,
}

/// 指标过滤器
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricFilter {
    pub field: String,
    pub operator: FilterOperator,
    pub value: serde_json::Value,
}

/// 过滤操作符
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOperator {
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Contains,
    Regex,
}

/// 存储后端
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageBackend {
    Memory,
    Disk,
    TimeSeries, // InfluxDB, Prometheus等
    Database,   // PostgreSQL, ClickHouse等
}

/// 聚合后的指标点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetric {
    pub timestamp: i64,
    pub metric_name: String,
    pub value: f64,
    pub aggregation_function: AggregationFunction,
    pub sample_count: u64,
    pub resolution: Duration,
    pub tags: HashMap<String, String>,
    pub metadata: MetricMetadata,
}

/// 指标元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricMetadata {
    pub min_value: f64,
    pub max_value: f64,
    pub sum_value: f64,
    pub sum_squares: f64,
    pub first_timestamp: i64,
    pub last_timestamp: i64,
}

/// 时间序列数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    pub metric_name: String,
    pub tags: HashMap<String, String>,
    pub data_points: Vec<DataPoint>,
    pub resolution: Duration,
    pub aggregation_function: AggregationFunction,
}

/// 数据点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub timestamp: i64,
    pub value: f64,
}

/// 查询参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryParams {
    pub metric_names: Vec<String>,
    pub start_time: i64,
    pub end_time: i64,
    pub resolution: Option<Duration>,
    pub aggregation_function: Option<AggregationFunction>,
    pub tags: HashMap<String, String>,
    pub filters: Vec<MetricFilter>,
    pub limit: Option<usize>,
}

/// 查询结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub time_series: Vec<TimeSeries>,
    pub total_points: usize,
    pub execution_time_ms: f64,
    pub cache_hit_ratio: f64,
    pub metadata: QueryMetadata,
}

/// 查询元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetadata {
    pub scanned_points: usize,
    pub filtered_points: usize,
    pub aggregated_points: usize,
    pub memory_usage_mb: f64,
    pub disk_reads: usize,
}

/// 指标聚合器
#[derive(Debug)]
pub struct MetricsAggregator {
    config: AggregatorConfig,
    
    // 数据存储
    raw_metrics: Arc<RwLock<VecDeque<ConsistencyMetrics>>>,
    aggregated_metrics: Arc<RwLock<BTreeMap<String, Vec<AggregatedMetric>>>>,
    time_series_cache: Arc<RwLock<HashMap<String, TimeSeries>>>,
    
    // 聚合状态
    last_aggregation: Arc<RwLock<Instant>>,
    aggregation_stats: Arc<RwLock<AggregationStats>>,
    
    // 查询缓存
    query_cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
}

/// 聚合统计
#[derive(Debug, Clone, Default)]
struct AggregationStats {
    pub total_raw_metrics: u64,
    pub total_aggregated_metrics: u64,
    pub aggregation_latency_ms: f64,
    pub memory_usage_mb: f64,
    pub compression_ratio: f64,
    pub cache_hit_rate: f64,
}

/// 缓存条目
#[derive(Debug, Clone)]
struct CacheEntry {
    pub query_hash: String,
    pub result: QueryResult,
    pub created_at: Instant,
    pub access_count: u64,
}

impl MetricsAggregator {
    pub fn new(config: AggregatorConfig) -> Self {
        Self {
            config,
            raw_metrics: Arc::new(RwLock::new(VecDeque::new())),
            aggregated_metrics: Arc::new(RwLock::new(BTreeMap::new())),
            time_series_cache: Arc::new(RwLock::new(HashMap::new())),
            last_aggregation: Arc::new(RwLock::new(Instant::now())),
            aggregation_stats: Arc::new(RwLock::new(AggregationStats::default())),
            query_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 启动聚合服务
    pub async fn start_service(&self) -> Result<()> {
        info!("Starting metrics aggregation service");
        
        // 启动聚合任务
        self.start_aggregation_task().await?;
        
        // 启动清理任务
        self.start_cleanup_task().await?;
        
        // 启动缓存管理任务
        self.start_cache_management_task().await?;
        
        Ok(())
    }

    /// 启动聚合任务
    async fn start_aggregation_task(&self) -> Result<()> {
        let config = self.config.clone();
        let raw_metrics = self.raw_metrics.clone();
        let aggregated_metrics = self.aggregated_metrics.clone();
        let last_aggregation = self.last_aggregation.clone();
        let aggregation_stats = self.aggregation_stats.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.aggregation_interval);
            
            loop {
                interval.tick().await;
                
                let start_time = Instant::now();
                
                // 执行聚合
                if let Err(e) = Self::perform_aggregation(
                    &config,
                    &raw_metrics,
                    &aggregated_metrics,
                    &aggregation_stats,
                ).await {
                    warn!("Aggregation failed: {}", e);
                    continue;
                }
                
                // 更新最后聚合时间
                {
                    let mut last_agg = last_aggregation.write().await;
                    *last_agg = Instant::now();
                }
                
                // 更新统计信息
                {
                    let mut stats = aggregation_stats.write().await;
                    stats.aggregation_latency_ms = start_time.elapsed().as_secs_f64() * 1000.0;
                }
                
                debug!("Metrics aggregation completed in {:.2}ms", start_time.elapsed().as_secs_f64() * 1000.0);
            }
        });
        
        Ok(())
    }

    /// 启动清理任务
    async fn start_cleanup_task(&self) -> Result<()> {
        let config = self.config.clone();
        let raw_metrics = self.raw_metrics.clone();
        let aggregated_metrics = self.aggregated_metrics.clone();
        let query_cache = self.query_cache.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(3600)); // 每小时清理一次
            
            loop {
                interval.tick().await;
                
                // 清理原始指标
                {
                    let mut raw = raw_metrics.write().await;
                    let cutoff_time = chrono::Utc::now().timestamp_millis() - 
                                      config.retention_policy.raw_data_retention.as_millis() as i64;
                    
                    while let Some(metric) = raw.front() {
                        if metric.timestamp < cutoff_time {
                            raw.pop_front();
                        } else {
                            break;
                        }
                    }
                }
                
                // 清理聚合指标
                {
                    let mut aggregated = aggregated_metrics.write().await;
                    let cutoff_time = chrono::Utc::now().timestamp_millis() - 
                                      config.retention_policy.aggregated_data_retention.as_millis() as i64;
                    
                    for (_metric_name, aggregated_data) in aggregated.iter_mut() {
                        aggregated_data.retain(|m| m.timestamp >= cutoff_time);
                    }
                }
                
                // 清理查询缓存
                {
                    let mut cache = query_cache.write().await;
                    let cache_ttl = Duration::from_secs(1800); // 30分钟TTL
                    
                    cache.retain(|_key, entry| {
                        entry.created_at.elapsed() < cache_ttl
                    });
                }
                
                debug!("Metrics cleanup completed");
            }
        });
        
        Ok(())
    }

    /// 启动缓存管理任务
    async fn start_cache_management_task(&self) -> Result<()> {
        let time_series_cache = self.time_series_cache.clone();
        let aggregation_stats = self.aggregation_stats.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // 每5分钟
            
            loop {
                interval.tick().await;
                
                // 计算缓存统计
                let cache_size = {
                    let cache = time_series_cache.read().await;
                    cache.len()
                };
                
                // 更新统计信息
                {
                    let mut stats = aggregation_stats.write().await;
                    stats.memory_usage_mb = (cache_size * 1024) as f64 / 1024.0 / 1024.0; // 估算
                }
                
                debug!("Cache management: {} series in cache", cache_size);
            }
        });
        
        Ok(())
    }

    /// 接收指标数据
    pub async fn ingest_metric(&self, metric: ConsistencyMetrics) -> Result<()> {
        {
            let mut raw = self.raw_metrics.write().await;
            raw.push_back(metric);
            
            // 限制缓冲区大小
            let max_size = 100000; // 10万条记录
            while raw.len() > max_size {
                raw.pop_front();
            }
        }
        
        // 更新统计信息
        {
            let mut stats = self.aggregation_stats.write().await;
            stats.total_raw_metrics += 1;
        }
        
        Ok(())
    }

    /// 执行聚合
    async fn perform_aggregation(
        config: &AggregatorConfig,
        raw_metrics: &Arc<RwLock<VecDeque<ConsistencyMetrics>>>,
        aggregated_metrics: &Arc<RwLock<BTreeMap<String, Vec<AggregatedMetric>>>>,
        aggregation_stats: &Arc<RwLock<AggregationStats>>,
    ) -> Result<()> {
        // 获取需要聚合的原始指标
        let metrics_to_aggregate = {
            let raw = raw_metrics.read().await;
            let cutoff_time = chrono::Utc::now().timestamp_millis() - 
                              config.aggregation_interval.as_millis() as i64;
            
            raw.iter()
                .filter(|m| m.timestamp >= cutoff_time)
                .cloned()
                .collect::<Vec<_>>()
        };
        
        if metrics_to_aggregate.is_empty() {
            return Ok(());
        }
        
        // 按指标名称和时间窗口分组
        let mut grouped_metrics: HashMap<String, Vec<&ConsistencyMetrics>> = HashMap::new();
        for metric in &metrics_to_aggregate {
            let key = format!("{}_{}", metric.component_id, metric.metric_name);
            grouped_metrics.entry(key).or_default().push(metric);
        }
        
        // 执行聚合
        let mut new_aggregated_metrics = Vec::new();
        for (metric_key, metrics) in grouped_metrics {
            if let Ok(aggregated) = Self::aggregate_metric_group(&metric_key, &metrics) {
                new_aggregated_metrics.extend(aggregated);
            }
        }
        
        // 存储聚合结果
        {
            let mut aggregated = aggregated_metrics.write().await;
            for agg_metric in new_aggregated_metrics {
                let key = agg_metric.metric_name.clone();
                aggregated.entry(key).or_default().push(agg_metric);
            }
        }
        
        // 更新统计信息
        {
            let mut stats = aggregation_stats.write().await;
            stats.total_aggregated_metrics += metrics_to_aggregate.len() as u64;
        }
        
        Ok(())
    }

    /// 聚合指标组
    fn aggregate_metric_group(
        metric_key: &str,
        metrics: &[&ConsistencyMetrics],
    ) -> Result<Vec<AggregatedMetric>> {
        if metrics.is_empty() {
            return Ok(Vec::new());
        }
        
        let mut aggregated_metrics = Vec::new();
        let timestamp = chrono::Utc::now().timestamp_millis();
        
        // 聚合不同的指标字段
        let metric_fields = vec![
            ("consistency_score", metrics.iter().map(|m| m.consistency_score).collect::<Vec<_>>()),
            ("latency_ms", metrics.iter().map(|m| m.latency_ms).collect::<Vec<_>>()),
            ("throughput", metrics.iter().map(|m| m.throughput).collect::<Vec<_>>()),
            ("error_rate", metrics.iter().map(|m| m.error_rate).collect::<Vec<_>>()),
            ("cpu_usage", metrics.iter().map(|m| m.cpu_usage).collect::<Vec<_>>()),
            ("memory_usage", metrics.iter().map(|m| m.memory_usage).collect::<Vec<_>>()),
        ];
        
        for (field_name, values) in metric_fields {
            if values.is_empty() {
                continue;
            }
            
            // 计算各种聚合函数
            let aggregations = vec![
                (AggregationFunction::Average, Self::calculate_average(&values)),
                (AggregationFunction::Min, Self::calculate_min(&values)),
                (AggregationFunction::Max, Self::calculate_max(&values)),
                (AggregationFunction::P95, Self::calculate_percentile(&values, 0.95)),
                (AggregationFunction::P99, Self::calculate_percentile(&values, 0.99)),
            ];
            
            for (agg_func, value) in aggregations {
                let metadata = MetricMetadata {
                    min_value: Self::calculate_min(&values),
                    max_value: Self::calculate_max(&values),
                    sum_value: values.iter().sum(),
                    sum_squares: values.iter().map(|v| v * v).sum(),
                    first_timestamp: metrics.first().unwrap().timestamp,
                    last_timestamp: metrics.last().unwrap().timestamp,
                };
                
                aggregated_metrics.push(AggregatedMetric {
                    timestamp,
                    metric_name: format!("{}_{}", metric_key, field_name),
                    value,
                    aggregation_function: agg_func,
                    sample_count: values.len() as u64,
                    resolution: Duration::from_secs(60),
                    tags: HashMap::new(),
                    metadata,
                });
            }
        }
        
        Ok(aggregated_metrics)
    }

    /// 查询指标数据
    pub async fn query_metrics(&self, params: QueryParams) -> Result<QueryResult> {
        let start_time = Instant::now();
        
        // 生成查询缓存键
        let cache_key = self.generate_cache_key(&params)?;
        
        // 检查缓存
        {
            let cache = self.query_cache.read().await;
            if let Some(entry) = cache.get(&cache_key) {
                let mut result = entry.result.clone();
                result.cache_hit_ratio = 1.0;
                result.execution_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
                return Ok(result);
            }
        }
        
        // 执行查询
        let time_series = self.execute_query(&params).await?;
        
        let result = QueryResult {
            time_series: time_series.clone(),
            total_points: time_series.iter().map(|ts| ts.data_points.len()).sum(),
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            cache_hit_ratio: 0.0,
            metadata: QueryMetadata {
                scanned_points: 0,
                filtered_points: 0,
                aggregated_points: 0,
                memory_usage_mb: 0.0,
                disk_reads: 0,
            },
        };
        
        // 更新缓存
        {
            let mut cache = self.query_cache.write().await;
            cache.insert(cache_key, CacheEntry {
                query_hash: "".to_string(),
                result: result.clone(),
                created_at: Instant::now(),
                access_count: 1,
            });
        }
        
        Ok(result)
    }

    /// 执行查询
    async fn execute_query(&self, params: &QueryParams) -> Result<Vec<TimeSeries>> {
        let aggregated = self.aggregated_metrics.read().await;
        let mut time_series = Vec::new();
        
        for metric_name in &params.metric_names {
            if let Some(aggregated_data) = aggregated.get(metric_name) {
                let filtered_data: Vec<DataPoint> = aggregated_data
                    .iter()
                    .filter(|m| m.timestamp >= params.start_time && m.timestamp <= params.end_time)
                    .map(|m| DataPoint {
                        timestamp: m.timestamp,
                        value: m.value,
                    })
                    .collect();
                
                if !filtered_data.is_empty() {
                    time_series.push(TimeSeries {
                        metric_name: metric_name.clone(),
                        tags: params.tags.clone(),
                        data_points: filtered_data,
                        resolution: Duration::from_secs(60),
                        aggregation_function: AggregationFunction::Average,
                    });
                }
            }
        }
        
        Ok(time_series)
    }

    /// 生成缓存键
    fn generate_cache_key(&self, params: &QueryParams) -> Result<String> {
        let key = format!(
            "{}_{}_{}_{}_{}",
            params.metric_names.join(","),
            params.start_time,
            params.end_time,
            params.resolution.as_ref().map(|d| d.as_secs()).unwrap_or(0),
            serde_json::to_string(&params.tags)?
        );
        Ok(format!("{:x}", md5::compute(key)))
    }

    /// 计算平均值
    fn calculate_average(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }

    /// 计算最小值
    fn calculate_min(values: &[f64]) -> f64 {
        values.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    }

    /// 计算最大值
    fn calculate_max(values: &[f64]) -> f64 {
        values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    }

    /// 计算百分位数
    fn calculate_percentile(values: &[f64], percentile: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (percentile * (sorted_values.len() - 1) as f64).round() as usize;
        sorted_values[index.min(sorted_values.len() - 1)]
    }

    /// 获取聚合统计信息
    pub async fn get_aggregation_stats(&self) -> AggregationStats {
        let stats = self.aggregation_stats.read().await;
        stats.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dashboard::consistency_monitor::ConsistencyMetrics;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_aggregator_creation() {
        let config = AggregatorConfig::default();
        let aggregator = MetricsAggregator::new(config);
        
        // 测试指标接收
        let metric = ConsistencyMetrics {
            timestamp: chrono::Utc::now().timestamp_millis(),
            component_id: "test_component".to_string(),
            metric_name: "test_metric".to_string(),
            value: 1.0,
            expected_value: Some(1.0),
            deviation: 0.0,
            consistency_score: 0.9,
            latency_ms: 50.0,
            throughput: 100.0,
            error_rate: 0.001,
            success_rate: 0.999,
            cpu_usage: 0.5,
            memory_usage: 0.6,
            network_io: 10.0,
            disk_io: 5.0,
            prediction_accuracy: 0.8,
            regime_detection_confidence: 0.7,
            alpha_signal_strength: 0.75,
            risk_adjusted_return: 0.1,
            tags: HashMap::new(),
            attributes: HashMap::new(),
        };
        
        let result = aggregator.ingest_metric(metric).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_query_metrics() {
        let config = AggregatorConfig::default();
        let aggregator = MetricsAggregator::new(config);
        
        let params = QueryParams {
            metric_names: vec!["test_metric".to_string()],
            start_time: chrono::Utc::now().timestamp_millis() - 3600000, // 1小时前
            end_time: chrono::Utc::now().timestamp_millis(),
            resolution: Some(Duration::from_secs(60)),
            aggregation_function: Some(AggregationFunction::Average),
            tags: HashMap::new(),
            filters: Vec::new(),
            limit: None,
        };
        
        let result = aggregator.query_metrics(params).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_statistical_functions() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        assert_eq!(MetricsAggregator::calculate_average(&values), 3.0);
        assert_eq!(MetricsAggregator::calculate_min(&values), 1.0);
        assert_eq!(MetricsAggregator::calculate_max(&values), 5.0);
        assert_eq!(MetricsAggregator::calculate_percentile(&values, 0.5), 3.0);
    }
}