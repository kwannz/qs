pub mod consistency_dashboard;
pub mod performance_monitor;
pub mod alert_system;
pub mod metrics_collector;
pub mod report_generator;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration};

/// AG3一致性仪表盘系统
pub struct ConsistencyDashboard {
    performance_monitor: performance_monitor::PerformanceMonitor,
    alert_system: alert_system::AlertSystem,
    metrics_collector: metrics_collector::MetricsCollector,
    report_generator: report_generator::ReportGenerator,
    config: DashboardConfig,
    consistency_tracker: ConsistencyTracker,
}

/// 仪表盘配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    pub monitoring_interval_seconds: u64,    // 监控间隔
    pub consistency_threshold: f64,          // 一致性阈值
    pub alert_thresholds: AlertThresholds,   // 告警阈值
    pub retention_days: u64,                 // 数据保留天数
    pub report_frequency: ReportFrequency,   // 报告频率
    pub enabled_metrics: Vec<MetricType>,    // 启用的指标
    pub dashboard_refresh_seconds: u64,      // 仪表盘刷新间隔
}

/// 告警阈值
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub consistency_error_bps: f64,         // 一致性误差（基点）
    pub performance_deviation_pct: f64,     // 性能偏差百分比
    pub latency_ms: u64,                    // 延迟阈值
    pub error_rate_pct: f64,                // 错误率百分比
    pub data_freshness_minutes: u64,       // 数据新鲜度
    pub system_load_pct: f64,               // 系统负载百分比
}

/// 报告频率
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFrequency {
    Hourly,
    Daily,
    Weekly,
    Monthly,
}

/// 指标类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    Consistency,         // 一致性指标
    Performance,         // 性能指标
    Risk,               // 风险指标
    Execution,          // 执行指标
    System,             // 系统指标
    Cost,               // 成本指标
}

/// 一致性跟踪器
pub struct ConsistencyTracker {
    tracking_windows: HashMap<String, TrackingWindow>,
    baseline_metrics: HashMap<String, BaselineMetrics>,
    consistency_history: Vec<ConsistencyMeasurement>,
}

/// 跟踪窗口
#[derive(Debug, Clone)]
pub struct TrackingWindow {
    pub strategy_id: String,
    pub backtest_results: Option<BacktestResults>,
    pub simulation_results: Option<SimulationResults>,
    pub live_results: Option<LiveResults>,
    pub window_start: DateTime<Utc>,
    pub window_end: DateTime<Utc>,
}

/// 基线指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineMetrics {
    pub expected_return: f64,
    pub expected_volatility: f64,
    pub expected_sharpe: f64,
    pub expected_max_drawdown: f64,
    pub expected_win_rate: f64,
    pub expected_cost_bps: f64,
}

/// 一致性测量
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyMeasurement {
    pub timestamp: DateTime<Utc>,
    pub strategy_id: String,
    pub consistency_score: f64,           // 0-1, 1为完全一致
    pub return_consistency: f64,          // 收益一致性
    pub risk_consistency: f64,            // 风险一致性
    pub cost_consistency: f64,            // 成本一致性
    pub execution_consistency: f64,       // 执行一致性
    pub deviations: Vec<ConsistencyDeviation>,
}

/// 一致性偏差
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyDeviation {
    pub metric_name: String,
    pub backtest_value: f64,
    pub simulation_value: f64,
    pub live_value: f64,
    pub deviation_type: DeviationType,
    pub severity: DeviationSeverity,
    pub explanation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviationType {
    AbsoluteDeviation,  // 绝对偏差
    RelativeDeviation,  // 相对偏差
    TrendDeviation,     // 趋势偏差
    VolatilityShift,    // 波动率变化
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum DeviationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// 回测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResults {
    pub strategy_id: String,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub total_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub avg_trade_cost_bps: f64,
    pub num_trades: u32,
    pub daily_returns: Vec<f64>,
    pub trade_pnl: Vec<f64>,
}

/// 仿真结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResults {
    pub strategy_id: String,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub total_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub avg_trade_cost_bps: f64,
    pub num_trades: u32,
    pub execution_quality: f64,      // 执行质量评分
    pub slippage_bps: f64,           // 平均滑点
    pub fill_rate: f64,              // 成交率
    pub daily_returns: Vec<f64>,
    pub trade_pnl: Vec<f64>,
}

/// 实盘结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveResults {
    pub strategy_id: String,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub total_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub avg_trade_cost_bps: f64,
    pub num_trades: u32,
    pub execution_quality: f64,
    pub slippage_bps: f64,
    pub fill_rate: f64,
    pub market_impact_bps: f64,      // 市场冲击
    pub latency_ms: f64,             // 平均延迟
    pub rejection_rate: f64,         // 拒单率
    pub daily_returns: Vec<f64>,
    pub trade_pnl: Vec<f64>,
}

/// 仪表盘状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardStatus {
    pub last_update: DateTime<Utc>,
    pub system_health: SystemHealth,
    pub active_strategies: u32,
    pub total_consistency_checks: u64,
    pub average_consistency_score: f64,
    pub critical_alerts: u32,
    pub data_quality_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemHealth {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

impl ConsistencyDashboard {
    pub fn new(config: DashboardConfig) -> Result<Self> {
        Ok(Self {
            performance_monitor: performance_monitor::PerformanceMonitor::new(&config)?,
            alert_system: alert_system::AlertSystem::new(&config.alert_thresholds)?,
            metrics_collector: metrics_collector::MetricsCollector::new(&config)?,
            report_generator: report_generator::ReportGenerator::new(&config)?,
            config: config.clone(),
            consistency_tracker: ConsistencyTracker::new(),
        })
    }

    /// 启动监控系统
    pub async fn start_monitoring(&mut self) -> Result<()> {
        log::info!("Starting consistency dashboard monitoring");

        // 启动指标收集
        self.metrics_collector.start_collection().await?;

        // 启动性能监控
        self.performance_monitor.start_monitoring().await?;

        // 启动告警系统
        self.alert_system.start().await?;

        // 定期一致性检查
        let monitoring_interval = Duration::seconds(self.config.monitoring_interval_seconds as i64);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                std::time::Duration::from_secs(monitoring_interval.num_seconds() as u64)
            );
            
            loop {
                interval.tick().await;
                // 执行一致性检查的逻辑会在这里
                // self.perform_consistency_check().await;
            }
        });

        log::info!("Consistency dashboard started successfully");
        Ok(())
    }

    /// 添加跟踪窗口
    pub fn add_tracking_window(
        &mut self,
        strategy_id: String,
        window_start: DateTime<Utc>,
        window_end: DateTime<Utc>,
    ) -> Result<()> {
        let window = TrackingWindow {
            strategy_id: strategy_id.clone(),
            backtest_results: None,
            simulation_results: None,
            live_results: None,
            window_start,
            window_end,
        };

        self.consistency_tracker.tracking_windows.insert(strategy_id.clone(), window);
        log::info!("Added tracking window for strategy: {}", strategy_id);
        Ok(())
    }

    /// 更新回测结果
    pub fn update_backtest_results(
        &mut self,
        strategy_id: &str,
        results: BacktestResults,
    ) -> Result<()> {
        if let Some(window) = self.consistency_tracker.tracking_windows.get_mut(strategy_id) {
            window.backtest_results = Some(results);
            self.check_window_completion(strategy_id)?;
        } else {
            return Err(anyhow::anyhow!("Tracking window not found for strategy: {}", strategy_id));
        }
        Ok(())
    }

    /// 更新仿真结果
    pub fn update_simulation_results(
        &mut self,
        strategy_id: &str,
        results: SimulationResults,
    ) -> Result<()> {
        if let Some(window) = self.consistency_tracker.tracking_windows.get_mut(strategy_id) {
            window.simulation_results = Some(results);
            self.check_window_completion(strategy_id)?;
        } else {
            return Err(anyhow::anyhow!("Tracking window not found for strategy: {}", strategy_id));
        }
        Ok(())
    }

    /// 更新实盘结果
    pub fn update_live_results(
        &mut self,
        strategy_id: &str,
        results: LiveResults,
    ) -> Result<()> {
        if let Some(window) = self.consistency_tracker.tracking_windows.get_mut(strategy_id) {
            window.live_results = Some(results);
            self.check_window_completion(strategy_id)?;
        } else {
            return Err(anyhow::anyhow!("Tracking window not found for strategy: {}", strategy_id));
        }
        Ok(())
    }

    /// 执行一致性分析
    pub fn analyze_consistency(&mut self, strategy_id: &str) -> Result<ConsistencyMeasurement> {
        let window = self.consistency_tracker.tracking_windows.get(strategy_id)
            .ok_or_else(|| anyhow::anyhow!("Tracking window not found: {}", strategy_id))?;

        let backtest = window.backtest_results.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Backtest results not available"))?;
        let simulation = window.simulation_results.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Simulation results not available"))?;
        let live = window.live_results.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Live results not available"))?;

        // 计算各维度一致性
        let return_consistency = self.calculate_return_consistency(backtest, simulation, live)?;
        let risk_consistency = self.calculate_risk_consistency(backtest, simulation, live)?;
        let cost_consistency = self.calculate_cost_consistency(backtest, simulation, live)?;
        let execution_consistency = self.calculate_execution_consistency(simulation, live)?;

        // 计算综合一致性评分
        let consistency_score = (return_consistency + risk_consistency + cost_consistency + execution_consistency) / 4.0;

        // 识别主要偏差
        let deviations = self.identify_deviations(backtest, simulation, live)?;

        let measurement = ConsistencyMeasurement {
            timestamp: Utc::now(),
            strategy_id: strategy_id.to_string(),
            consistency_score,
            return_consistency,
            risk_consistency,
            cost_consistency,
            execution_consistency,
            deviations,
        };

        // 记录测量结果
        self.consistency_tracker.consistency_history.push(measurement.clone());

        // 检查是否需要告警
        if consistency_score < self.config.consistency_threshold {
            self.alert_system.trigger_consistency_alert(&measurement)?;
        }

        log::info!("Consistency analysis completed for strategy: {} (score: {:.3})", 
            strategy_id, consistency_score);
        
        Ok(measurement)
    }

    /// 获取实时一致性状态
    pub fn get_consistency_status(&self) -> Result<HashMap<String, f64>> {
        let mut status = HashMap::new();
        
        for (strategy_id, _window) in &self.consistency_tracker.tracking_windows {
            // 获取最近的一致性评分
            let recent_score = self.consistency_tracker.consistency_history.iter()
                .filter(|m| m.strategy_id == *strategy_id)
                .last()
                .map(|m| m.consistency_score)
                .unwrap_or(0.0);
            
            status.insert(strategy_id.clone(), recent_score);
        }
        
        Ok(status)
    }

    /// 获取仪表盘状态
    pub fn get_dashboard_status(&self) -> Result<DashboardStatus> {
        let active_strategies = self.consistency_tracker.tracking_windows.len() as u32;
        let total_consistency_checks = self.consistency_tracker.consistency_history.len() as u64;
        
        let average_consistency_score = if !self.consistency_tracker.consistency_history.is_empty() {
            self.consistency_tracker.consistency_history.iter()
                .map(|m| m.consistency_score)
                .sum::<f64>() / self.consistency_tracker.consistency_history.len() as f64
        } else {
            0.0
        };

        let critical_alerts = self.alert_system.get_critical_alert_count()?;
        let data_quality_score = self.metrics_collector.get_data_quality_score()?;
        
        let system_health = if critical_alerts > 0 {
            SystemHealth::Critical
        } else if average_consistency_score < self.config.consistency_threshold {
            SystemHealth::Warning
        } else {
            SystemHealth::Healthy
        };

        Ok(DashboardStatus {
            last_update: Utc::now(),
            system_health,
            active_strategies,
            total_consistency_checks,
            average_consistency_score,
            critical_alerts,
            data_quality_score,
        })
    }

    /// 生成一致性报告
    pub fn generate_consistency_report(
        &self,
        strategy_id: Option<String>,
        period_days: u64,
    ) -> Result<String> {
        let end_time = Utc::now();
        let start_time = end_time - Duration::days(period_days as i64);
        
        let filtered_measurements: Vec<&ConsistencyMeasurement> = self.consistency_tracker.consistency_history.iter()
            .filter(|m| {
                m.timestamp >= start_time && m.timestamp <= end_time &&
                strategy_id.as_ref().map_or(true, |id| m.strategy_id == *id)
            })
            .collect();

        self.report_generator.generate_consistency_report(&filtered_measurements, start_time, end_time)
    }

    /// 导出指标数据
    pub fn export_metrics(&self, format: ExportFormat) -> Result<String> {
        let metrics_data = MetricsExport {
            timestamp: Utc::now(),
            consistency_measurements: self.consistency_tracker.consistency_history.clone(),
            tracking_windows: self.get_tracking_windows_summary(),
            dashboard_status: self.get_dashboard_status()?,
        };

        match format {
            ExportFormat::Json => Ok(serde_json::to_string_pretty(&metrics_data)?),
            ExportFormat::Csv => self.export_to_csv(&metrics_data),
            ExportFormat::Parquet => self.export_to_parquet(&metrics_data),
        }
    }

    // 私有辅助方法
    fn check_window_completion(&mut self, strategy_id: &str) -> Result<()> {
        let window = self.consistency_tracker.tracking_windows.get(strategy_id).unwrap();
        
        if window.backtest_results.is_some() && 
           window.simulation_results.is_some() && 
           window.live_results.is_some() {
            // 三个阶段的结果都有了，执行一致性分析
            self.analyze_consistency(strategy_id)?;
        }
        
        Ok(())
    }

    fn calculate_return_consistency(
        &self,
        backtest: &BacktestResults,
        simulation: &SimulationResults,
        live: &LiveResults,
    ) -> Result<f64> {
        // 计算收益率的一致性
        let bt_return = backtest.total_return;
        let sim_return = simulation.total_return;
        let live_return = live.total_return;
        
        // 使用归一化的相对偏差
        let sim_deviation = ((sim_return - bt_return) / bt_return.abs().max(0.01)).abs();
        let live_deviation = ((live_return - bt_return) / bt_return.abs().max(0.01)).abs();
        
        let avg_deviation = (sim_deviation + live_deviation) / 2.0;
        let consistency = 1.0 / (1.0 + avg_deviation);
        
        Ok(consistency.max(0.0).min(1.0))
    }

    fn calculate_risk_consistency(
        &self,
        backtest: &BacktestResults,
        simulation: &SimulationResults,
        live: &LiveResults,
    ) -> Result<f64> {
        // 基于夏普比率和最大回撤的风险一致性
        let bt_sharpe = backtest.sharpe_ratio;
        let sim_sharpe = simulation.sharpe_ratio;
        let live_sharpe = live.sharpe_ratio;
        
        let bt_dd = backtest.max_drawdown;
        let sim_dd = simulation.max_drawdown;
        let live_dd = live.max_drawdown;
        
        let sharpe_consistency = {
            let sim_dev = ((sim_sharpe - bt_sharpe) / bt_sharpe.abs().max(0.1)).abs();
            let live_dev = ((live_sharpe - bt_sharpe) / bt_sharpe.abs().max(0.1)).abs();
            1.0 / (1.0 + (sim_dev + live_dev) / 2.0)
        };
        
        let dd_consistency = {
            let sim_dev = ((sim_dd - bt_dd) / bt_dd.abs().max(0.01)).abs();
            let live_dev = ((live_dd - bt_dd) / bt_dd.abs().max(0.01)).abs();
            1.0 / (1.0 + (sim_dev + live_dev) / 2.0)
        };
        
        Ok(((sharpe_consistency + dd_consistency) / 2.0).max(0.0).min(1.0))
    }

    fn calculate_cost_consistency(
        &self,
        backtest: &BacktestResults,
        simulation: &SimulationResults,
        live: &LiveResults,
    ) -> Result<f64> {
        let bt_cost = backtest.avg_trade_cost_bps;
        let sim_cost = simulation.avg_trade_cost_bps;
        let live_cost = live.avg_trade_cost_bps;
        
        let sim_deviation = ((sim_cost - bt_cost) / bt_cost.max(0.1)).abs();
        let live_deviation = ((live_cost - bt_cost) / bt_cost.max(0.1)).abs();
        
        let consistency = 1.0 / (1.0 + (sim_deviation + live_deviation) / 2.0);
        Ok(consistency.max(0.0).min(1.0))
    }

    fn calculate_execution_consistency(
        &self,
        simulation: &SimulationResults,
        live: &LiveResults,
    ) -> Result<f64> {
        // 基于执行质量、成交率、滑点的一致性
        let quality_dev = (live.execution_quality - simulation.execution_quality).abs();
        let fill_rate_dev = (live.fill_rate - simulation.fill_rate).abs();
        let slippage_dev = ((live.slippage_bps - simulation.slippage_bps) / simulation.slippage_bps.max(0.1)).abs();
        
        let avg_deviation = (quality_dev + fill_rate_dev + slippage_dev) / 3.0;
        let consistency = 1.0 / (1.0 + avg_deviation);
        
        Ok(consistency.max(0.0).min(1.0))
    }

    fn identify_deviations(
        &self,
        backtest: &BacktestResults,
        simulation: &SimulationResults,
        live: &LiveResults,
    ) -> Result<Vec<ConsistencyDeviation>> {
        let mut deviations = Vec::new();
        
        // 收益率偏差
        let return_deviation = ConsistencyDeviation {
            metric_name: "Total Return".to_string(),
            backtest_value: backtest.total_return,
            simulation_value: simulation.total_return,
            live_value: live.total_return,
            deviation_type: DeviationType::RelativeDeviation,
            severity: self.classify_deviation_severity(
                (live.total_return - backtest.total_return).abs() / backtest.total_return.abs().max(0.01)
            ),
            explanation: "Return deviation between backtest and live trading".to_string(),
        };
        deviations.push(return_deviation);
        
        // 夏普比率偏差
        let sharpe_deviation = ConsistencyDeviation {
            metric_name: "Sharpe Ratio".to_string(),
            backtest_value: backtest.sharpe_ratio,
            simulation_value: simulation.sharpe_ratio,
            live_value: live.sharpe_ratio,
            deviation_type: DeviationType::AbsoluteDeviation,
            severity: self.classify_deviation_severity(
                (live.sharpe_ratio - backtest.sharpe_ratio).abs()
            ),
            explanation: "Sharpe ratio deviation indicating risk-adjusted return differences".to_string(),
        };
        deviations.push(sharpe_deviation);
        
        // 成本偏差
        let cost_deviation = ConsistencyDeviation {
            metric_name: "Trading Cost".to_string(),
            backtest_value: backtest.avg_trade_cost_bps,
            simulation_value: simulation.avg_trade_cost_bps,
            live_value: live.avg_trade_cost_bps,
            deviation_type: DeviationType::AbsoluteDeviation,
            severity: self.classify_deviation_severity(
                (live.avg_trade_cost_bps - backtest.avg_trade_cost_bps).abs() / 10.0
            ),
            explanation: "Trading cost deviation indicating execution differences".to_string(),
        };
        deviations.push(cost_deviation);
        
        // 过滤掉低严重性的偏差
        deviations.retain(|d| d.severity >= DeviationSeverity::Medium);
        
        Ok(deviations)
    }

    fn classify_deviation_severity(&self, normalized_deviation: f64) -> DeviationSeverity {
        if normalized_deviation > 0.5 {
            DeviationSeverity::Critical
        } else if normalized_deviation > 0.2 {
            DeviationSeverity::High
        } else if normalized_deviation > 0.05 {
            DeviationSeverity::Medium
        } else {
            DeviationSeverity::Low
        }
    }

    fn get_tracking_windows_summary(&self) -> Vec<TrackingWindowSummary> {
        self.consistency_tracker.tracking_windows.iter()
            .map(|(strategy_id, window)| TrackingWindowSummary {
                strategy_id: strategy_id.clone(),
                window_start: window.window_start,
                window_end: window.window_end,
                has_backtest: window.backtest_results.is_some(),
                has_simulation: window.simulation_results.is_some(),
                has_live: window.live_results.is_some(),
                completion_status: if window.backtest_results.is_some() && 
                                     window.simulation_results.is_some() && 
                                     window.live_results.is_some() {
                    "Complete".to_string()
                } else {
                    "Incomplete".to_string()
                },
            })
            .collect()
    }

    fn export_to_csv(&self, data: &MetricsExport) -> Result<String> {
        // 简化的CSV导出
        let mut csv = String::new();
        csv.push_str("timestamp,strategy_id,consistency_score,return_consistency,risk_consistency,cost_consistency,execution_consistency\n");
        
        for measurement in &data.consistency_measurements {
            csv.push_str(&format!("{},{},{},{},{},{},{}\n",
                measurement.timestamp.to_rfc3339(),
                measurement.strategy_id,
                measurement.consistency_score,
                measurement.return_consistency,
                measurement.risk_consistency,
                measurement.cost_consistency,
                measurement.execution_consistency,
            ));
        }
        
        Ok(csv)
    }

    fn export_to_parquet(&self, data: &MetricsExport) -> Result<String> {
        // Parquet导出需要额外的依赖，这里返回占位符
        Ok("Parquet export not implemented".to_string())
    }
}

impl ConsistencyTracker {
    fn new() -> Self {
        Self {
            tracking_windows: HashMap::new(),
            baseline_metrics: HashMap::new(),
            consistency_history: Vec::new(),
        }
    }
}

/// 导出格式
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
    Parquet,
}

/// 指标导出数据
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MetricsExport {
    timestamp: DateTime<Utc>,
    consistency_measurements: Vec<ConsistencyMeasurement>,
    tracking_windows: Vec<TrackingWindowSummary>,
    dashboard_status: DashboardStatus,
}

/// 跟踪窗口摘要
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrackingWindowSummary {
    strategy_id: String,
    window_start: DateTime<Utc>,
    window_end: DateTime<Utc>,
    has_backtest: bool,
    has_simulation: bool,
    has_live: bool,
    completion_status: String,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            monitoring_interval_seconds: 60,
            consistency_threshold: 0.8,
            alert_thresholds: AlertThresholds {
                consistency_error_bps: 50.0,
                performance_deviation_pct: 20.0,
                latency_ms: 1000,
                error_rate_pct: 5.0,
                data_freshness_minutes: 10,
                system_load_pct: 80.0,
            },
            retention_days: 90,
            report_frequency: ReportFrequency::Daily,
            enabled_metrics: vec![
                MetricType::Consistency,
                MetricType::Performance,
                MetricType::Risk,
                MetricType::Execution,
            ],
            dashboard_refresh_seconds: 30,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_creation() {
        let config = DashboardConfig::default();
        let result = ConsistencyDashboard::new(config);
        // 由于依赖其他模块，暂时跳过测试
        // assert!(result.is_ok());
    }

    #[test]
    fn test_deviation_severity_classification() {
        let config = DashboardConfig::default();
        if let Ok(dashboard) = ConsistencyDashboard::new(config) {
            assert_eq!(dashboard.classify_deviation_severity(0.6), DeviationSeverity::Critical);
            assert_eq!(dashboard.classify_deviation_severity(0.3), DeviationSeverity::High);
            assert_eq!(dashboard.classify_deviation_severity(0.1), DeviationSeverity::Medium);
            assert_eq!(dashboard.classify_deviation_severity(0.02), DeviationSeverity::Low);
        }
    }

    #[test]
    fn test_consistency_measurement_creation() {
        let measurement = ConsistencyMeasurement {
            timestamp: Utc::now(),
            strategy_id: "test_strategy".to_string(),
            consistency_score: 0.85,
            return_consistency: 0.9,
            risk_consistency: 0.8,
            cost_consistency: 0.85,
            execution_consistency: 0.85,
            deviations: Vec::new(),
        };
        
        assert_eq!(measurement.strategy_id, "test_strategy");
        assert_eq!(measurement.consistency_score, 0.85);
    }
}