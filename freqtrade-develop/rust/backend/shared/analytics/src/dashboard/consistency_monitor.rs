use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{Duration, Instant};
use tracing::{debug, info, error};
use uuid::Uuid;

// Note: BanditPerformanceMetrics, OPEResults, ShadowTestResult not yet implemented
// Note: MarketRegime, RegimeDetectionService, MicrostructureAlphaFeatures not yet used

/// 一致性监控配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyConfig {
    pub monitoring_window: Duration,       // 监控时间窗口
    pub alert_thresholds: AlertThresholds, // 告警阈值
    pub metric_retention: Duration,        // 指标保留时长
    pub consistency_check_interval: Duration, // 一致性检查间隔
    pub anomaly_detection_window: usize,   // 异常检测窗口大小
    pub correlation_threshold: f64,        // 相关性阈值
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        Self {
            monitoring_window: Duration::from_secs(3600), // 1小时
            alert_thresholds: AlertThresholds::default(),
            metric_retention: Duration::from_secs(24 * 3600), // 24小时
            consistency_check_interval: Duration::from_secs(60), // 1分钟
            anomaly_detection_window: 100,
            correlation_threshold: 0.7,
        }
    }
}

/// 告警阈值配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub performance_degradation: f64,     // 性能降级阈值
    pub consistency_score_min: f64,       // 最低一致性分数
    pub latency_max_ms: f64,             // 最大延迟（毫秒）
    pub error_rate_max: f64,             // 最大错误率
    pub memory_usage_max: f64,           // 最大内存使用率
    pub cpu_usage_max: f64,              // 最大CPU使用率
    pub prediction_accuracy_min: f64,     // 最低预测准确率
    pub regime_detection_confidence_min: f64, // 最低体制检测置信度
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            performance_degradation: 0.1,    // 10%性能降级
            consistency_score_min: 0.8,      // 80%一致性
            latency_max_ms: 100.0,          // 100ms最大延迟
            error_rate_max: 0.01,           // 1%错误率
            memory_usage_max: 0.8,          // 80%内存使用
            cpu_usage_max: 0.8,             // 80%CPU使用
            prediction_accuracy_min: 0.75,  // 75%预测准确率
            regime_detection_confidence_min: 0.6, // 60%体制检测置信度
        }
    }
}

/// 一致性指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyMetrics {
    pub timestamp: i64,
    pub component_id: String,
    pub metric_name: String,
    
    // 核心指标
    pub value: f64,
    pub expected_value: Option<f64>,
    pub deviation: f64,
    pub consistency_score: f64,
    
    // 性能指标
    pub latency_ms: f64,
    pub throughput: f64,
    pub error_rate: f64,
    pub success_rate: f64,
    
    // 资源指标
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_io: f64,
    pub disk_io: f64,
    
    // 业务指标
    pub prediction_accuracy: f64,
    pub regime_detection_confidence: f64,
    pub alpha_signal_strength: f64,
    pub risk_adjusted_return: f64,
    
    // 元数据
    pub tags: HashMap<String, String>,
    pub attributes: HashMap<String, serde_json::Value>,
}

/// 异常事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyEvent {
    pub id: String,
    pub timestamp: i64,
    pub component_id: String,
    pub anomaly_type: AnomalyType,
    pub severity: AlertSeverity,
    pub description: String,
    pub affected_metrics: Vec<String>,
    pub root_cause_analysis: Option<RootCause>,
    pub remediation_suggestion: Option<String>,
    pub resolved: bool,
    pub resolution_time: Option<i64>,
}

/// 异常类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    PerformanceDegradation,
    ConsistencyViolation,
    LatencySpike,
    ErrorRateIncrease,
    ResourceExhaustion,
    PredictionAccuracyDrop,
    RegimeDetectionFailure,
    AlphaSignalAnomaly,
    SystemFailure,
}

impl std::fmt::Display for AnomalyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnomalyType::PerformanceDegradation => write!(f, "Performance Degradation"),
            AnomalyType::ConsistencyViolation => write!(f, "Consistency Violation"),
            AnomalyType::LatencySpike => write!(f, "Latency Spike"),
            AnomalyType::ErrorRateIncrease => write!(f, "Error Rate Increase"),
            AnomalyType::ResourceExhaustion => write!(f, "Resource Exhaustion"),
            AnomalyType::PredictionAccuracyDrop => write!(f, "Prediction Accuracy Drop"),
            AnomalyType::RegimeDetectionFailure => write!(f, "Regime Detection Failure"),
            AnomalyType::AlphaSignalAnomaly => write!(f, "Alpha Signal Anomaly"),
            AnomalyType::SystemFailure => write!(f, "System Failure"),
        }
    }
}

/// 告警严重级别
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// 根因分析
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootCause {
    pub primary_cause: String,
    pub contributing_factors: Vec<String>,
    pub correlation_analysis: HashMap<String, f64>,
    pub time_series_analysis: TimeSeriesAnalysis,
}

/// 时间序列分析结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesAnalysis {
    pub trend: f64,
    pub seasonality: Vec<f64>,
    pub anomaly_score: f64,
    pub changepoint_detected: bool,
    pub forecast_accuracy: f64,
}

/// 系统健康状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub overall_score: f64,
    pub component_scores: HashMap<String, f64>,
    pub active_alerts: Vec<AnomalyEvent>,
    pub recent_incidents: Vec<AnomalyEvent>,
    pub performance_summary: PerformanceSummary,
    pub resource_utilization: ResourceUtilization,
    pub prediction_quality: PredictionQuality,
}

/// 性能摘要
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub throughput_qps: f64,
    pub error_rate: f64,
    pub availability: f64,
}

/// 资源利用率
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_io_mbps: f64,
    pub disk_io_mbps: f64,
    pub connection_count: u64,
    pub thread_count: u64,
}

/// 预测质量
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionQuality {
    pub bandit_performance: HashMap<String, f64>,
    pub ope_accuracy: f64,
    pub regime_detection_accuracy: f64,
    pub alpha_signal_quality: f64,
    pub risk_prediction_accuracy: f64,
}

/// 一致性监控器
pub struct ConsistencyMonitor {
    config: ConsistencyConfig,
    
    // 指标存储
    metrics_buffer: Arc<RwLock<VecDeque<ConsistencyMetrics>>>,
    anomalies: Arc<RwLock<VecDeque<AnomalyEvent>>>,
    component_states: Arc<RwLock<HashMap<String, ComponentState>>>,
    
    // 分析引擎
    anomaly_detector: Arc<AnomalyDetector>,
    correlation_analyzer: Arc<CorrelationAnalyzer>,
    trend_analyzer: Arc<TrendAnalyzer>,
    
    // 运行时状态
    last_check_time: Arc<RwLock<Instant>>,
    alert_callbacks: Arc<RwLock<Vec<Box<dyn Fn(&AnomalyEvent) + Send + Sync>>>>,
}

/// 组件状态
#[derive(Debug, Clone)]
struct ComponentState {
    pub last_update: Instant,
    pub metric_history: VecDeque<ConsistencyMetrics>,
    pub health_score: f64,
    pub status: ComponentStatus,
    pub last_anomaly: Option<AnomalyEvent>,
}

/// 组件状态枚举
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComponentStatus {
    Healthy,
    Warning,
    Critical,
    Offline,
}

impl ConsistencyMonitor {
    pub fn new(config: ConsistencyConfig) -> Self {
        Self {
            config,
            metrics_buffer: Arc::new(RwLock::new(VecDeque::new())),
            anomalies: Arc::new(RwLock::new(VecDeque::new())),
            component_states: Arc::new(RwLock::new(HashMap::new())),
            anomaly_detector: Arc::new(AnomalyDetector::new()),
            correlation_analyzer: Arc::new(CorrelationAnalyzer::new()),
            trend_analyzer: Arc::new(TrendAnalyzer::new()),
            last_check_time: Arc::new(RwLock::new(Instant::now())),
            alert_callbacks: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// 启动监控服务
    pub async fn start_monitoring(&self) -> Result<()> {
        info!("Starting consistency monitoring service");
        
        let config = self.config.clone();
        let metrics_buffer = self.metrics_buffer.clone();
        let anomalies = self.anomalies.clone();
        let component_states = self.component_states.clone();
        let anomaly_detector = self.anomaly_detector.clone();
        let last_check_time = self.last_check_time.clone();
        let alert_callbacks = self.alert_callbacks.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.consistency_check_interval);
            
            loop {
                interval.tick().await;
                
                // 执行一致性检查
                if let Err(e) = Self::perform_consistency_check(
                    &config,
                    &metrics_buffer,
                    &anomalies,
                    &component_states,
                    &anomaly_detector,
                    &alert_callbacks,
                ).await {
                    error!("Consistency check failed: {}", e);
                }
                
                // 清理过期指标
                if let Err(e) = Self::cleanup_expired_metrics(
                    &config,
                    &metrics_buffer,
                    &anomalies,
                ).await {
                    error!("Metric cleanup failed: {}", e);
                }
                
                // 更新最后检查时间
                {
                    let mut last_check = last_check_time.write().await;
                    *last_check = Instant::now();
                }
            }
        });
        
        Ok(())
    }

    /// 上报指标
    pub async fn report_metric(&self, metric: ConsistencyMetrics) -> Result<()> {
        debug!("Reporting metric: {} for component {}", metric.metric_name, metric.component_id);
        
        // 存储指标
        {
            let mut buffer = self.metrics_buffer.write().await;
            buffer.push_back(metric.clone());
            
            // 限制缓冲区大小
            while buffer.len() > 10000 {
                buffer.pop_front();
            }
        }
        
        // 更新组件状态
        self.update_component_state(&metric).await?;
        
        // 实时异常检测
        self.detect_real_time_anomalies(&metric).await?;
        
        Ok(())
    }

    /// 更新组件状态
    async fn update_component_state(&self, metric: &ConsistencyMetrics) -> Result<()> {
        let mut states = self.component_states.write().await;
        
        let state = states.entry(metric.component_id.clone())
            .or_insert_with(|| ComponentState {
                last_update: Instant::now(),
                metric_history: VecDeque::new(),
                health_score: 1.0,
                status: ComponentStatus::Healthy,
                last_anomaly: None,
            });
        
        state.last_update = Instant::now();
        state.metric_history.push_back(metric.clone());
        
        // 限制历史记录大小
        while state.metric_history.len() > 1000 {
            state.metric_history.pop_front();
        }
        
        // 计算健康分数
        state.health_score = self.calculate_component_health_score(&state.metric_history)?;
        
        // 更新状态
        state.status = match state.health_score {
            score if score >= 0.8 => ComponentStatus::Healthy,
            score if score >= 0.6 => ComponentStatus::Warning,
            score if score >= 0.3 => ComponentStatus::Critical,
            _ => ComponentStatus::Offline,
        };
        
        Ok(())
    }

    /// 实时异常检测
    async fn detect_real_time_anomalies(&self, metric: &ConsistencyMetrics) -> Result<()> {
        // 检查阈值违规
        let anomalies = self.check_threshold_violations(metric)?;
        
        for anomaly in anomalies {
            self.handle_anomaly(anomaly).await?;
        }
        
        Ok(())
    }

    /// 检查阈值违规
    fn check_threshold_violations(&self, metric: &ConsistencyMetrics) -> Result<Vec<AnomalyEvent>> {
        let mut anomalies = Vec::new();
        let thresholds = &self.config.alert_thresholds;
        
        // 检查一致性分数
        if metric.consistency_score < thresholds.consistency_score_min {
            anomalies.push(AnomalyEvent {
                id: Uuid::new_v4().to_string(),
                timestamp: metric.timestamp,
                component_id: metric.component_id.clone(),
                anomaly_type: AnomalyType::ConsistencyViolation,
                severity: AlertSeverity::High,
                description: format!("Consistency score {} below threshold {}", 
                                   metric.consistency_score, thresholds.consistency_score_min),
                affected_metrics: vec![metric.metric_name.clone()],
                root_cause_analysis: None,
                remediation_suggestion: Some("Check data synchronization and model consistency".to_string()),
                resolved: false,
                resolution_time: None,
            });
        }
        
        // 检查延迟
        if metric.latency_ms > thresholds.latency_max_ms {
            anomalies.push(AnomalyEvent {
                id: Uuid::new_v4().to_string(),
                timestamp: metric.timestamp,
                component_id: metric.component_id.clone(),
                anomaly_type: AnomalyType::LatencySpike,
                severity: if metric.latency_ms > thresholds.latency_max_ms * 2.0 { 
                    AlertSeverity::Critical 
                } else { 
                    AlertSeverity::High 
                },
                description: format!("Latency {} ms exceeds threshold {} ms", 
                                   metric.latency_ms, thresholds.latency_max_ms),
                affected_metrics: vec![metric.metric_name.clone()],
                root_cause_analysis: None,
                remediation_suggestion: Some("Check system load and optimize critical paths".to_string()),
                resolved: false,
                resolution_time: None,
            });
        }
        
        // 检查错误率
        if metric.error_rate > thresholds.error_rate_max {
            anomalies.push(AnomalyEvent {
                id: Uuid::new_v4().to_string(),
                timestamp: metric.timestamp,
                component_id: metric.component_id.clone(),
                anomaly_type: AnomalyType::ErrorRateIncrease,
                severity: AlertSeverity::High,
                description: format!("Error rate {} exceeds threshold {}", 
                                   metric.error_rate, thresholds.error_rate_max),
                affected_metrics: vec![metric.metric_name.clone()],
                root_cause_analysis: None,
                remediation_suggestion: Some("Investigate error logs and fix underlying issues".to_string()),
                resolved: false,
                resolution_time: None,
            });
        }
        
        // 检查预测准确率
        if metric.prediction_accuracy < thresholds.prediction_accuracy_min {
            anomalies.push(AnomalyEvent {
                id: Uuid::new_v4().to_string(),
                timestamp: metric.timestamp,
                component_id: metric.component_id.clone(),
                anomaly_type: AnomalyType::PredictionAccuracyDrop,
                severity: AlertSeverity::Medium,
                description: format!("Prediction accuracy {} below threshold {}", 
                                   metric.prediction_accuracy, thresholds.prediction_accuracy_min),
                affected_metrics: vec![metric.metric_name.clone()],
                root_cause_analysis: None,
                remediation_suggestion: Some("Retrain models or update feature engineering".to_string()),
                resolved: false,
                resolution_time: None,
            });
        }
        
        Ok(anomalies)
    }

    /// 处理异常事件
    async fn handle_anomaly(&self, anomaly: AnomalyEvent) -> Result<()> {
        info!("Detected anomaly: {} - {}", anomaly.anomaly_type, anomaly.description);
        
        // 存储异常事件
        {
            let mut anomalies = self.anomalies.write().await;
            anomalies.push_back(anomaly.clone());
            
            // 限制异常事件数量
            while anomalies.len() > 1000 {
                anomalies.pop_front();
            }
        }
        
        // 触发告警回调
        {
            let callbacks = self.alert_callbacks.read().await;
            for callback in callbacks.iter() {
                callback(&anomaly);
            }
        }
        
        Ok(())
    }

    /// 执行一致性检查
    async fn perform_consistency_check(
        config: &ConsistencyConfig,
        metrics_buffer: &Arc<RwLock<VecDeque<ConsistencyMetrics>>>,
        anomalies: &Arc<RwLock<VecDeque<AnomalyEvent>>>,
        component_states: &Arc<RwLock<HashMap<String, ComponentState>>>,
        anomaly_detector: &Arc<AnomalyDetector>,
        alert_callbacks: &Arc<RwLock<Vec<Box<dyn Fn(&AnomalyEvent) + Send + Sync>>>>,
    ) -> Result<()> {
        debug!("Performing consistency check");
        
        // 获取最近的指标
        let recent_metrics = {
            let buffer = metrics_buffer.read().await;
            let cutoff_time = chrono::Utc::now().timestamp_millis() - 
                              config.monitoring_window.as_millis() as i64;
            
            buffer.iter()
                .filter(|m| m.timestamp >= cutoff_time)
                .cloned()
                .collect::<Vec<_>>()
        };
        
        if recent_metrics.is_empty() {
            return Ok(());
        }
        
        // 执行高级异常检测
        let detected_anomalies = anomaly_detector.detect_anomalies(&recent_metrics)?;
        
        // 处理检测到的异常
        for anomaly in detected_anomalies {
            // 存储异常事件
            {
                let mut anomalies_buf = anomalies.write().await;
                anomalies_buf.push_back(anomaly.clone());
                
                while anomalies_buf.len() > 1000 {
                    anomalies_buf.pop_front();
                }
            }
            
            // 触发告警回调
            {
                let callbacks = alert_callbacks.read().await;
                for callback in callbacks.iter() {
                    callback(&anomaly);
                }
            }
        }
        
        Ok(())
    }

    /// 清理过期指标
    async fn cleanup_expired_metrics(
        config: &ConsistencyConfig,
        metrics_buffer: &Arc<RwLock<VecDeque<ConsistencyMetrics>>>,
        anomalies: &Arc<RwLock<VecDeque<AnomalyEvent>>>,
    ) -> Result<()> {
        let cutoff_time = chrono::Utc::now().timestamp_millis() - 
                          config.metric_retention.as_millis() as i64;
        
        // 清理指标
        {
            let mut buffer = metrics_buffer.write().await;
            while let Some(metric) = buffer.front() {
                if metric.timestamp < cutoff_time {
                    buffer.pop_front();
                } else {
                    break;
                }
            }
        }
        
        // 清理异常事件
        {
            let mut anomalies_buf = anomalies.write().await;
            while let Some(anomaly) = anomalies_buf.front() {
                if anomaly.timestamp < cutoff_time {
                    anomalies_buf.pop_front();
                } else {
                    break;
                }
            }
        }
        
        Ok(())
    }

    /// 计算组件健康分数
    fn calculate_component_health_score(&self, metrics: &VecDeque<ConsistencyMetrics>) -> Result<f64> {
        if metrics.is_empty() {
            return Ok(0.0);
        }
        
        let recent_metrics: Vec<&ConsistencyMetrics> = metrics.iter().rev().take(10).collect();
        
        let mut total_score = 0.0;
        let mut weight_sum = 0.0;
        
        for (i, metric) in recent_metrics.iter().enumerate() {
            let weight = 1.0 / (i as f64 + 1.0); // 越新的指标权重越高
            
            let score = 
                metric.consistency_score * 0.3 +
                metric.success_rate * 0.2 +
                (1.0 - metric.error_rate.min(1.0)) * 0.2 +
                metric.prediction_accuracy * 0.15 +
                (1.0 - metric.cpu_usage.min(1.0)) * 0.075 +
                (1.0 - metric.memory_usage.min(1.0)) * 0.075;
            
            total_score += score * weight;
            weight_sum += weight;
        }
        
        Ok(if weight_sum > 0.0 { total_score / weight_sum } else { 0.0 })
    }

    /// 获取系统健康状态
    pub async fn get_system_health(&self) -> Result<SystemHealth> {
        let states = self.component_states.read().await;
        let anomalies = self.anomalies.read().await;
        
        // 计算整体分数
        let overall_score = if states.is_empty() {
            0.0
        } else {
            states.values().map(|s| s.health_score).sum::<f64>() / states.len() as f64
        };
        
        // 组件分数
        let component_scores: HashMap<String, f64> = states.iter()
            .map(|(name, state)| (name.clone(), state.health_score))
            .collect();
        
        // 活跃告警
        let active_alerts: Vec<AnomalyEvent> = anomalies.iter()
            .filter(|a| !a.resolved)
            .cloned()
            .collect();
        
        // 最近事件
        let recent_incidents: Vec<AnomalyEvent> = anomalies.iter()
            .rev()
            .take(10)
            .cloned()
            .collect();
        
        // 性能摘要（简化实现）
        let performance_summary = PerformanceSummary {
            avg_latency_ms: 50.0,
            p95_latency_ms: 100.0,
            p99_latency_ms: 200.0,
            throughput_qps: 1000.0,
            error_rate: 0.001,
            availability: 0.999,
        };
        
        // 资源利用率（简化实现）
        let resource_utilization = ResourceUtilization {
            cpu_usage: 0.6,
            memory_usage: 0.7,
            network_io_mbps: 100.0,
            disk_io_mbps: 50.0,
            connection_count: 1000,
            thread_count: 200,
        };
        
        // 预测质量（简化实现）
        let prediction_quality = PredictionQuality {
            bandit_performance: HashMap::new(),
            ope_accuracy: 0.85,
            regime_detection_accuracy: 0.80,
            alpha_signal_quality: 0.75,
            risk_prediction_accuracy: 0.82,
        };
        
        Ok(SystemHealth {
            overall_score,
            component_scores,
            active_alerts,
            recent_incidents,
            performance_summary,
            resource_utilization,
            prediction_quality,
        })
    }

    /// 注册告警回调
    pub async fn register_alert_callback<F>(&self, callback: F) -> Result<()> 
    where
        F: Fn(&AnomalyEvent) + Send + Sync + 'static,
    {
        let mut callbacks = self.alert_callbacks.write().await;
        callbacks.push(Box::new(callback));
        Ok(())
    }
}

/// 异常检测器
#[derive(Debug)]
struct AnomalyDetector;

impl AnomalyDetector {
    fn new() -> Self {
        Self
    }
    
    fn detect_anomalies(&self, _metrics: &[ConsistencyMetrics]) -> Result<Vec<AnomalyEvent>> {
        // 简化的异常检测实现
        Ok(Vec::new())
    }
}

/// 相关性分析器
#[derive(Debug)]
struct CorrelationAnalyzer;

impl CorrelationAnalyzer {
    fn new() -> Self {
        Self
    }
}

/// 趋势分析器
#[derive(Debug)]
struct TrendAnalyzer;

impl TrendAnalyzer {
    fn new() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_consistency_monitor_creation() {
        let config = ConsistencyConfig::default();
        let monitor = ConsistencyMonitor::new(config);
        
        // 测试指标上报
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
        
        let result = monitor.report_metric(metric).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_system_health() {
        let config = ConsistencyConfig::default();
        let monitor = ConsistencyMonitor::new(config);
        
        let health = monitor.get_system_health().await;
        assert!(health.is_ok());
        
        let health = health.unwrap();
        assert_eq!(health.overall_score, 0.0); // No components yet
    }
}