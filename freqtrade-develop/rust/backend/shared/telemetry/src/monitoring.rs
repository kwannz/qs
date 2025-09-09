use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use tracing::{info, warn, error, debug};

/// 统一监控系统 - 核心监控指标收集与分析
#[derive(Clone)]
pub struct UnifiedMonitoringSystem {
    metrics: Arc<RwLock<HashMap<String, MetricValue>>>,
    alerts: Arc<RwLock<Vec<Alert>>>,
    system_health: Arc<RwLock<SystemHealth>>,
    config: MonitoringConfig,
}

/// 监控配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub alert_thresholds: HashMap<String, f64>,
    pub metrics_retention_seconds: u64,
    pub health_check_interval_seconds: u64,
    pub alert_cooldown_seconds: u64,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert("cpu_usage".to_string(), 80.0);
        thresholds.insert("memory_usage".to_string(), 85.0);
        thresholds.insert("latency_ms".to_string(), 1000.0);
        thresholds.insert("error_rate".to_string(), 5.0);
        thresholds.insert("trading_pnl".to_string(), -10000.0);
        
        Self {
            alert_thresholds: thresholds,
            metrics_retention_seconds: 3600, // 1 hour
            health_check_interval_seconds: 30,
            alert_cooldown_seconds: 300, // 5 minutes
        }
    }
}

/// 指标值类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram(Vec<f64>),
    Timer(Duration),
    Rate { value: f64, timestamp: u64 },
}

/// 系统健康状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub overall_status: HealthStatus,
    pub components: HashMap<String, ComponentHealth>,
    pub last_updated: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Critical,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub status: HealthStatus,
    pub metrics: HashMap<String, f64>,
    pub last_check: u64,
    pub error_message: Option<String>,
}

/// 告警信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub level: AlertLevel,
    pub component: String,
    pub metric: String,
    pub value: f64,
    pub threshold: f64,
    pub message: String,
    pub timestamp: u64,
    pub resolved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
    Emergency,
}

impl UnifiedMonitoringSystem {
    /// 创建新的监控系统实例
    pub fn new(config: Option<MonitoringConfig>) -> Self {
        let config = config.unwrap_or_default();
        
        info!("🔍 初始化统一监控系统");
        info!("📊 告警阈值: {:?}", config.alert_thresholds);
        
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            alerts: Arc::new(RwLock::new(Vec::new())),
            system_health: Arc::new(RwLock::new(SystemHealth {
                overall_status: HealthStatus::Unknown,
                components: HashMap::new(),
                last_updated: current_timestamp(),
            })),
            config,
        }
    }

    /// 记录指标
    pub fn record_metric(&self, name: &str, value: MetricValue) {
        let mut metrics = self.metrics.write().unwrap();
        metrics.insert(name.to_string(), value);
        
        // 检查是否触发告警
        if let Err(e) = self.check_alert_conditions(name) {
            error!("告警检查失败: {}", e);
        }
    }

    /// 递增计数器
    pub fn increment_counter(&self, name: &str, value: u64) {
        let mut metrics = self.metrics.write().unwrap();
        let current = match metrics.get(name) {
            Some(MetricValue::Counter(count)) => *count,
            _ => 0,
        };
        metrics.insert(name.to_string(), MetricValue::Counter(current + value));
    }

    /// 设置仪表盘值
    pub fn set_gauge(&self, name: &str, value: f64) {
        self.record_metric(name, MetricValue::Gauge(value));
    }

    /// 记录时间指标
    pub fn record_timing(&self, name: &str, duration: Duration) {
        self.record_metric(name, MetricValue::Timer(duration));
        
        // 同时记录毫秒级别的仪表盘值以便告警
        self.set_gauge(&format!("{name}_ms"), duration.as_millis() as f64);
    }

    /// 记录速率指标
    pub fn record_rate(&self, name: &str, value: f64) {
        self.record_metric(name, MetricValue::Rate {
            value,
            timestamp: current_timestamp(),
        });
    }

    /// 更新组件健康状态
    pub fn update_component_health(&self, component: &str, status: HealthStatus, metrics: HashMap<String, f64>) {
        let mut health = self.system_health.write().unwrap();
        
        health.components.insert(component.to_string(), ComponentHealth {
            status: status.clone(),
            metrics,
            last_check: current_timestamp(),
            error_message: None,
        });

        // 计算整体健康状态
        health.overall_status = self.calculate_overall_health(&health.components);
        health.last_updated = current_timestamp();

        debug!("更新组件健康状态: {} -> {:?}", component, status);
    }

    /// 获取系统健康状态
    pub fn get_system_health(&self) -> SystemHealth {
        self.system_health.read().unwrap().clone()
    }

    /// 获取指标快照
    pub fn get_metrics_snapshot(&self) -> HashMap<String, MetricValue> {
        self.metrics.read().unwrap().clone()
    }

    /// 获取活跃告警
    pub fn get_active_alerts(&self) -> Vec<Alert> {
        self.alerts.read().unwrap()
            .iter()
            .filter(|alert| !alert.resolved)
            .cloned()
            .collect()
    }

    /// 检查告警条件
    fn check_alert_conditions(&self, metric_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let metrics = self.metrics.read().unwrap();
        let metric_value = metrics.get(metric_name);
        
        if let Some(threshold) = self.config.alert_thresholds.get(metric_name) {
            let current_value = match metric_value {
                Some(MetricValue::Gauge(value)) => *value,
                Some(MetricValue::Rate { value, .. }) => *value,
                Some(MetricValue::Timer(duration)) => duration.as_millis() as f64,
                _ => return Ok(()),
            };

            if self.should_trigger_alert(metric_name, current_value, *threshold) {
                self.trigger_alert(metric_name, current_value, *threshold)?;
            }
        }
        
        Ok(())
    }

    /// 判断是否应该触发告警
    fn should_trigger_alert(&self, metric_name: &str, current_value: f64, threshold: f64) -> bool {
        let alerts = self.alerts.read().unwrap();
        
        // 检查冷却期
        let now = current_timestamp();
        let has_recent_alert = alerts.iter().any(|alert| {
            alert.metric == metric_name && 
            !alert.resolved && 
            now - alert.timestamp < self.config.alert_cooldown_seconds
        });

        if has_recent_alert {
            return false;
        }

        // 基本阈值检查
        match metric_name {
            name if name.contains("error") || name.contains("failure") => current_value > threshold,
            name if name.contains("latency") || name.contains("_ms") => current_value > threshold,
            name if name.contains("usage") => current_value > threshold,
            name if name.contains("pnl") => current_value < threshold, // PnL负值告警
            _ => current_value > threshold,
        }
    }

    /// 触发告警
    fn trigger_alert(&self, metric_name: &str, current_value: f64, threshold: f64) -> Result<(), Box<dyn std::error::Error>> {
        let level = self.determine_alert_level(metric_name, current_value, threshold);
        let message = format!(
            "指标 {metric_name} 当前值 {current_value:.2} 超过阈值 {threshold:.2}"
        );

        let alert = Alert {
            id: uuid::Uuid::new_v4().to_string(),
            level: level.clone(),
            component: self.extract_component_from_metric(metric_name),
            metric: metric_name.to_string(),
            value: current_value,
            threshold,
            message: message.clone(),
            timestamp: current_timestamp(),
            resolved: false,
        };

        // 记录告警
        {
            let mut alerts = self.alerts.write().unwrap();
            alerts.push(alert);
        }

        // 记录日志
        match level {
            AlertLevel::Emergency | AlertLevel::Critical => {
                error!("🚨 {} 告警: {}", level_emoji(&level), message);
            }
            AlertLevel::Warning => {
                warn!("⚠️ {} 告警: {}", level_emoji(&level), message);
            }
            AlertLevel::Info => {
                info!("ℹ️ {} 告警: {}", level_emoji(&level), message);
            }
        }

        Ok(())
    }

    /// 确定告警级别
    fn determine_alert_level(&self, metric_name: &str, current_value: f64, threshold: f64) -> AlertLevel {
        let ratio = current_value / threshold;
        
        match metric_name {
            name if name.contains("error") || name.contains("failure") => {
                if ratio > 3.0 { AlertLevel::Critical }
                else if ratio > 1.5 { AlertLevel::Warning }
                else { AlertLevel::Info }
            }
            name if name.contains("latency") || name.contains("_ms") => {
                if ratio > 5.0 { AlertLevel::Emergency }
                else if ratio > 2.0 { AlertLevel::Critical }
                else if ratio > 1.2 { AlertLevel::Warning }
                else { AlertLevel::Info }
            }
            name if name.contains("pnl") => {
                if current_value < threshold * 2.0 { AlertLevel::Emergency }
                else if current_value < threshold * 1.5 { AlertLevel::Critical }
                else { AlertLevel::Warning }
            }
            _ => {
                if ratio > 2.0 { AlertLevel::Critical }
                else if ratio > 1.5 { AlertLevel::Warning }
                else { AlertLevel::Info }
            }
        }
    }

    /// 从指标名提取组件名
    fn extract_component_from_metric(&self, metric_name: &str) -> String {
        metric_name.split('_').next().unwrap_or("system").to_string()
    }

    /// 计算整体健康状态
    fn calculate_overall_health(&self, components: &HashMap<String, ComponentHealth>) -> HealthStatus {
        if components.is_empty() {
            return HealthStatus::Unknown;
        }

        let mut critical_count = 0;
        let mut degraded_count = 0;
        let mut healthy_count = 0;

        for health in components.values() {
            match health.status {
                HealthStatus::Critical => critical_count += 1,
                HealthStatus::Degraded => degraded_count += 1,
                HealthStatus::Healthy => healthy_count += 1,
                HealthStatus::Unknown => {}
            }
        }

        if critical_count > 0 {
            HealthStatus::Critical
        } else if degraded_count > 0 {
            HealthStatus::Degraded
        } else if healthy_count > 0 {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unknown
        }
    }

    /// 启动后台监控任务
    pub async fn start_background_monitoring(&self) {
        let interval_duration = Duration::from_secs(self.config.health_check_interval_seconds);
        let system = self.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval_duration);
            loop {
                interval.tick().await;
                system.perform_health_checks().await;
                system.cleanup_old_metrics();
            }
        });

        info!("🔄 后台监控任务已启动，检查间隔: {}秒", self.config.health_check_interval_seconds);
    }

    /// 执行健康检查
    async fn perform_health_checks(&self) {
        debug!("执行系统健康检查");
        
        // 系统资源检查
        self.check_system_resources().await;
        
        // 应用程序指标检查
        self.check_application_metrics().await;
        
        // 网络连接检查
        self.check_network_health().await;
    }

    /// 检查系统资源
    async fn check_system_resources(&self) {
        let mut metrics = HashMap::new();
        
        // CPU使用率 (模拟)
        let cpu_usage = 45.0 + (rand::random::<f64>() * 30.0); // 45-75%
        metrics.insert("cpu_usage".to_string(), cpu_usage);
        self.set_gauge("system_cpu_usage", cpu_usage);
        
        // 内存使用率 (模拟)
        let memory_usage = 60.0 + (rand::random::<f64>() * 25.0); // 60-85%
        metrics.insert("memory_usage".to_string(), memory_usage);
        self.set_gauge("system_memory_usage", memory_usage);
        
        let status = if cpu_usage > 80.0 || memory_usage > 85.0 {
            HealthStatus::Critical
        } else if cpu_usage > 70.0 || memory_usage > 75.0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };
        
        self.update_component_health("system", status, metrics);
    }

    /// 检查应用程序指标
    async fn check_application_metrics(&self) {
        let mut metrics = HashMap::new();
        
        // 模拟交易系统指标
        let latency_ms = 100.0 + (rand::random::<f64>() * 200.0); // 100-300ms
        metrics.insert("latency_ms".to_string(), latency_ms);
        self.set_gauge("trading_latency_ms", latency_ms);
        
        let error_rate = rand::random::<f64>() * 10.0; // 0-10%
        metrics.insert("error_rate".to_string(), error_rate);
        self.set_gauge("trading_error_rate", error_rate);
        
        let throughput = 1000.0 + (rand::random::<f64>() * 500.0); // 1000-1500 TPS
        metrics.insert("throughput".to_string(), throughput);
        self.set_gauge("trading_throughput", throughput);
        
        let status = if latency_ms > 1000.0 || error_rate > 5.0 {
            HealthStatus::Critical
        } else if latency_ms > 500.0 || error_rate > 2.0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };
        
        self.update_component_health("trading", status, metrics);
    }

    /// 检查网络健康
    async fn check_network_health(&self) {
        let mut metrics = HashMap::new();
        
        // 模拟网络连接状态
        let connection_count = 50.0 + (rand::random::<u32>() % 100) as f64; // 50-150 连接
        metrics.insert("connections".to_string(), connection_count);
        self.set_gauge("network_connections", connection_count);
        
        let packet_loss = rand::random::<f64>() * 5.0; // 0-5% 丢包率
        metrics.insert("packet_loss".to_string(), packet_loss);
        self.set_gauge("network_packet_loss", packet_loss);
        
        let status = if packet_loss > 3.0 {
            HealthStatus::Critical
        } else if packet_loss > 1.0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };
        
        self.update_component_health("network", status, metrics);
    }

    /// 清理旧指标
    fn cleanup_old_metrics(&self) {
        let retention_seconds = self.config.metrics_retention_seconds;
        let cutoff_time = current_timestamp() - retention_seconds;
        
        let mut alerts = self.alerts.write().unwrap();
        alerts.retain(|alert| alert.timestamp > cutoff_time);
        
        debug!("清理了 {} 秒前的旧指标", retention_seconds);
    }
}

/// 获取当前时间戳
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// 获取告警级别对应的emoji
fn level_emoji(level: &AlertLevel) -> &'static str {
    match level {
        AlertLevel::Info => "ℹ️",
        AlertLevel::Warning => "⚠️",
        AlertLevel::Critical => "🚨",
        AlertLevel::Emergency => "🆘",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_monitoring_system_creation() {
        let monitoring = UnifiedMonitoringSystem::new(None);
        let health = monitoring.get_system_health();
        assert!(matches!(health.overall_status, HealthStatus::Unknown));
    }

    #[test]
    fn test_metric_recording() {
        let monitoring = UnifiedMonitoringSystem::new(None);
        monitoring.set_gauge("test_metric", 42.0);
        
        let metrics = monitoring.get_metrics_snapshot();
        match metrics.get("test_metric") {
            Some(MetricValue::Gauge(value)) => assert_eq!(*value, 42.0),
            _ => panic!("Expected gauge metric with value 42.0"),
        }
    }

    #[test]
    fn test_counter_increment() {
        let monitoring = UnifiedMonitoringSystem::new(None);
        monitoring.increment_counter("test_counter", 5);
        monitoring.increment_counter("test_counter", 3);
        
        let metrics = monitoring.get_metrics_snapshot();
        match metrics.get("test_counter") {
            Some(MetricValue::Counter(count)) => assert_eq!(*count, 8),
            _ => panic!("Expected counter with value 8"),
        }
    }
}