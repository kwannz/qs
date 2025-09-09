// Sprint 1: OpenTelemetry集成模块 - 简化实现
pub mod simple;
// 复杂的OpenTelemetry模块暂时禁用以避免API兼容性问题
// pub mod otel;
// pub mod structured_logging;

// 原有模块保持兼容性
pub mod simple_tracing;
pub mod simple_metrics;
pub mod spans;
pub mod health;

// 新增的统一监控和可观测性模块
pub mod monitoring;
pub mod observability;

// Sprint 1: 导出简化的遥测功能
pub use simple::{
    SimpleTelemetryConfig, SimpleTelemetrySystem,
    TraceContext, MetricsCollector
};

// 导出原有的简单功能
pub use simple_tracing::*;
pub use simple_metrics::*;
pub use spans::*;
pub use health::*;

// 导出新的统一功能
pub use monitoring::{
    UnifiedMonitoringSystem, 
    MonitoringConfig, 
    MetricValue, 
    SystemHealth, 
    HealthStatus, 
    ComponentHealth,
    Alert, 
    AlertLevel
};

pub use observability::{
    ObservabilitySystem,
    TraceCollector,
    TraceContext as ObservabilityTraceContext,
    TraceSpan,
    SpanStatus,
    PerformanceAnalyzer,
    DashboardGenerator,
    ObservabilityReport,
    PerformanceSummary,
    AlertSummary,
    TraceAnalysis,
    Recommendation,
    RecommendationPriority,
};

use anyhow::Result;
use std::sync::Arc;
use tracing::{info, error};

/// 集成的遥测系统 - 统一监控、追踪和可观测性的入口点
pub struct TelemetrySystem {
    monitoring: Arc<UnifiedMonitoringSystem>,
    observability: Arc<ObservabilitySystem>,
}

impl TelemetrySystem {
    /// 创建新的遥测系统实例
    pub fn new(monitoring_config: Option<MonitoringConfig>) -> Self {
        info!("🚀 初始化统一遥测系统");
        
        let monitoring = Arc::new(UnifiedMonitoringSystem::new(monitoring_config));
        let observability = Arc::new(ObservabilitySystem::new((*monitoring).clone()));
        
        Self {
            monitoring,
            observability,
        }
    }

    /// 获取监控系统引用
    pub fn monitoring(&self) -> &UnifiedMonitoringSystem {
        &self.monitoring
    }

    /// 获取可观测性系统引用
    pub fn observability(&self) -> &ObservabilitySystem {
        &self.observability
    }

    /// 启动所有后台服务
    pub async fn start_background_services(&self) {
        info!("🔄 启动遥测系统后台服务");
        
        // 启动监控后台任务
        self.monitoring.start_background_monitoring().await;
        
        info!("✅ 遥测系统后台服务已启动");
    }

    /// 记录业务指标 - 便捷方法
    pub fn record_business_metric(&self, operation: &str, value: f64, tags: Option<std::collections::HashMap<String, String>>) {
        self.monitoring.set_gauge(&format!("business_{operation}"), value);
        
        // 记录到追踪系统（如果需要）
        if let Some(tags) = tags {
            // 这里可以添加更多的上下文信息
            tracing::info!("业务指标记录: {} = {}, tags: {:?}", operation, value, tags);
        }
    }

    /// 开始性能追踪 - 简化实现
    pub async fn start_performance_trace(&self, operation: &str, _service: &str) -> TraceContext {
        // 简化实现，仅记录日志
        tracing::info!("开始性能追踪: {}", operation);
        TraceContext
    }

    /// 记录性能数据
    pub async fn record_performance(&self, operation: &str, duration: std::time::Duration) {
        self.observability.record_performance(operation, duration).await;
    }

    /// 生成系统健康报告
    pub async fn generate_health_report(&self) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let time_range = observability::TimeRange {
            start: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs() - 3600, // 最近1小时
            end: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
        };

        let report = self.observability.generate_report(time_range).await;
        
        match serde_json::to_value(report) {
            Ok(json) => Ok(json),
            Err(e) => {
                error!("生成健康报告失败: {}", e);
                Err(Box::new(e))
            }
        }
    }

    /// 获取实时仪表盘数据
    pub async fn get_dashboard(&self, template: &str) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        self.observability.create_dashboard(template).await
    }

    /// 获取系统概览
    pub fn get_system_overview(&self) -> SystemOverview {
        let health = self.monitoring.get_system_health();
        let alerts = self.monitoring.get_active_alerts();
        let metrics = self.monitoring.get_metrics_snapshot();

        SystemOverview {
            health_status: health.overall_status,
            total_components: health.components.len(),
            healthy_components: health.components.values()
                .filter(|c| matches!(c.status, HealthStatus::Healthy))
                .count(),
            active_alerts: alerts.len(),
            critical_alerts: alerts.iter()
                .filter(|a| matches!(a.level, AlertLevel::Critical))
                .count(),
            total_metrics: metrics.len(),
            last_updated: health.last_updated,
        }
    }
}

/// 系统概览信息
#[derive(Debug, Clone, serde::Serialize)]
pub struct SystemOverview {
    pub health_status: HealthStatus,
    pub total_components: usize,
    pub healthy_components: usize,
    pub active_alerts: usize,
    pub critical_alerts: usize,
    pub total_metrics: usize,
    pub last_updated: u64,
}

/// Initialize the telemetry module (保持向后兼容)
pub async fn init_telemetry() -> Result<()> {
    // Initialize simple tracing first
    init_simple_tracing().await?;
    
    // Initialize simple metrics collection
    init_simple_metrics().await?;
    
    tracing::info!("Platform telemetry module initialized successfully");
    Ok(())
}

/// 创建统一遥测系统实例 - 新的推荐方式
pub async fn init_unified_telemetry(config: Option<MonitoringConfig>) -> Result<TelemetrySystem> {
    // 先初始化原有系统以保持兼容性
    init_telemetry().await?;
    
    // 创建新的统一系统
    let telemetry = TelemetrySystem::new(config);
    telemetry.start_background_services().await;
    
    info!("🎯 统一遥测系统初始化完成");
    Ok(telemetry)
}

/// Shutdown telemetry gracefully
pub async fn shutdown_telemetry() -> Result<()> {
    tracing::info!("Shutting down telemetry systems");
    tracing::info!("Telemetry shutdown completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_telemetry_init() {
        let result = init_telemetry().await;
        assert!(result.is_ok());
    }
}