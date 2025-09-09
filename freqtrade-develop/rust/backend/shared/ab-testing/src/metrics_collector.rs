//! 实验指标收集器

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;

/// 指标收集器
pub struct MetricsCollector {
    metrics_buffer: Vec<MetricEvent>,
    aggregated_metrics: HashMap<String, ExperimentMetrics>,
}

/// 指标事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricEvent {
    pub event_id: uuid::Uuid,
    pub user_id: String,
    pub experiment_id: uuid::Uuid,
    pub variant_id: String,
    pub metric_name: String,
    pub metric_value: f64,
    pub timestamp: DateTime<Utc>,
    pub properties: HashMap<String, serde_json::Value>,
}

/// 实验指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMetrics {
    pub experiment_id: uuid::Uuid,
    pub variant_metrics: HashMap<String, VariantMetrics>,
    pub last_updated: DateTime<Utc>,
}

/// 变体指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantMetrics {
    pub variant_id: String,
    pub sample_size: usize,
    pub metrics: HashMap<String, MetricSummary>,
}

/// 指标汇总
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSummary {
    pub metric_name: String,
    pub count: usize,
    pub sum: f64,
    pub mean: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub percentiles: HashMap<String, f64>, // P50, P95, P99等
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics_buffer: Vec::new(),
            aggregated_metrics: HashMap::new(),
        }
    }
    
    /// 记录指标事件
    pub fn record_metric(
        &mut self,
        user_id: String,
        experiment_id: uuid::Uuid,
        variant_id: String,
        metric_name: String,
        metric_value: f64,
        properties: Option<HashMap<String, serde_json::Value>>,
    ) {
        let event = MetricEvent {
            event_id: uuid::Uuid::new_v4(),
            user_id,
            experiment_id,
            variant_id: variant_id.clone(),
            metric_name: metric_name.clone(),
            metric_value,
            timestamp: Utc::now(),
            properties: properties.unwrap_or_default(),
        };
        
        debug!("记录指标事件: {} = {}", event.metric_name, event.metric_value);
        self.metrics_buffer.push(event);
    }
    
    /// 批量记录指标
    pub fn record_batch_metrics(&mut self, events: Vec<MetricEvent>) {
        self.metrics_buffer.extend(events);
    }
    
    /// 聚合指标
    pub async fn aggregate_metrics(&mut self) -> Result<()> {
        // 克隆缓冲区以避免借用冲突
        let events = self.metrics_buffer.clone();
        for event in &events {
            self.process_metric_event(event).await?;
        }
        
        // 清空缓冲区
        self.metrics_buffer.clear();
        
        Ok(())
    }
    
    /// 获取实验指标
    pub fn get_experiment_metrics(&self, experiment_id: &uuid::Uuid) -> Option<&ExperimentMetrics> {
        self.aggregated_metrics.get(&experiment_id.to_string())
    }
    
    async fn process_metric_event(&mut self, event: &MetricEvent) -> Result<()> {
        let experiment_key = event.experiment_id.to_string();
        
        // 获取或创建实验指标
        let experiment_metrics = self.aggregated_metrics
            .entry(experiment_key)
            .or_insert_with(|| ExperimentMetrics {
                experiment_id: event.experiment_id,
                variant_metrics: HashMap::new(),
                last_updated: Utc::now(),
            });
        
        // 获取或创建变体指标
        let variant_metrics = experiment_metrics.variant_metrics
            .entry(event.variant_id.clone())
            .or_insert_with(|| VariantMetrics {
                variant_id: event.variant_id.clone(),
                sample_size: 0,
                metrics: HashMap::new(),
            });
        
        // 更新指标汇总
        Self::update_metric_summary_static(variant_metrics, &event.metric_name, event.metric_value);
        
        experiment_metrics.last_updated = Utc::now();
        
        Ok(())
    }
    
    fn update_metric_summary_static(variant_metrics: &mut VariantMetrics, metric_name: &str, value: f64) {
        let metric_summary = variant_metrics.metrics
            .entry(metric_name.to_string())
            .or_insert_with(|| MetricSummary {
                metric_name: metric_name.to_string(),
                count: 0,
                sum: 0.0,
                mean: 0.0,
                variance: 0.0,
                std_dev: 0.0,
                min: f64::INFINITY,
                max: f64::NEG_INFINITY,
                percentiles: HashMap::new(),
            });
        
        // 更新基本统计
        metric_summary.count += 1;
        metric_summary.sum += value;
        metric_summary.min = metric_summary.min.min(value);
        metric_summary.max = metric_summary.max.max(value);
        
        // 更新均值（在线算法）
        let old_mean = metric_summary.mean;
        metric_summary.mean = metric_summary.sum / metric_summary.count as f64;
        
        // 更新方差（Welford算法）
        if metric_summary.count > 1 {
            let delta = value - old_mean;
            let delta2 = value - metric_summary.mean;
            metric_summary.variance = ((metric_summary.count - 1) as f64 * metric_summary.variance + delta * delta2) / metric_summary.count as f64;
            metric_summary.std_dev = metric_summary.variance.sqrt();
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}