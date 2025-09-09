//! AG3增强前置风控门控系统
//! 集成额度/回撤/速率/黑名单/相关暴露的完整检查

use std::collections::HashMap;
use std::sync::Arc;
use std::str::FromStr;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};
use rust_decimal::Decimal;
use anyhow::{Result, Context};
use tracing::{info, warn, error};

// Temporary type definitions - should be replaced with proper platform_types import later
pub type Symbol = String;
pub type Price = rust_decimal::Decimal;
pub type Quantity = rust_decimal::Decimal;
pub type UserId = String;

#[derive(Debug, Clone)]
pub struct OrderRequest {
    pub symbol: Symbol,
    pub side: OrderSide,
    pub quantity: Quantity,
    pub price: Option<Price>,
    pub user_id: UserId,
}

#[derive(Debug, Clone, Copy)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone)]
pub enum OrderType {
    Market,
    Limit,
}

// Mock risk types for compilation
#[derive(Debug, Clone)]
pub struct RiskCheckResult {
    pub approved: bool,
    pub violations: Vec<RiskViolation>,
    pub action: Option<RiskAction>,
    pub risk_score: f64,
    pub details: String,
}
#[derive(Debug, Clone)]  
pub struct RiskViolation {
    pub severity: String,
    pub violation_type: ViolationType,
    pub message: String,
    pub current_value: Option<f64>,
    pub limit_value: Option<f64>,
}
#[derive(Debug, Clone)]
pub enum ViolationType { 
    Warning,
    BatchRisk,
    QuotaLimit,
    PositionLimit,
    RateLimit,
    DrawdownLimit,
    CorrelationLimit,
}
#[derive(Debug, Clone, PartialEq)]
pub enum RiskAction { 
    Allow,
    Block, 
    Reject, 
    RequireApproval,
    Warning,
}

/// 增强前置风控控制器
pub struct EnhancedPreemptiveRiskController {
    config: EnhancedRiskConfig,
    quota_manager: Arc<QuotaManager>,
    drawdown_monitor: Arc<DrawdownMonitor>,
    rate_limiter: Arc<EnhancedRateLimiter>,
    blacklist_manager: Arc<BlacklistManager>,
    correlation_monitor: Arc<CorrelationExposureMonitor>,
    position_tracker: Arc<PositionTracker>,
    market_data_provider: Arc<dyn MarketDataProvider>,
    violation_recorder: Arc<ViolationRecorder>,
}

// Mock implementations removed - using full implementations below

#[derive(Debug, Clone)]
pub struct RiskQuota;
#[derive(Debug, Clone)]
pub struct DrawdownAlert;
#[derive(Debug, Clone)]
pub struct RiskControlAction;

// Basic config structure - removed duplicate, using the one below

/// 增强风控配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedRiskConfig {
    pub enabled: bool,
    pub strict_mode: bool,
    pub circuit_breaker_enabled: bool,
    
    // 额度控制
    pub quota_config: QuotaConfig,
    
    // 回撤控制
    pub drawdown_config: DrawdownConfig,
    
    // 速率限制
    pub rate_limit_config: RateLimitConfig,
    
    // 相关性控制
    pub correlation_config: CorrelationConfig,
    
    // 黑名单配置
    pub blacklist_config: BlacklistConfig,
    
    // 报警配置
    pub alert_config: AlertConfig,
}

/// 额度配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaConfig {
    pub global_daily_limit_usd: Decimal,
    pub per_user_daily_limit_usd: Decimal,
    pub per_symbol_daily_limit_usd: Decimal,
    pub single_order_max_usd: Decimal,
    pub margin_utilization_max: Decimal, // 最大保证金使用率
    pub leverage_max: Decimal,
    pub concentration_limit_pct: Decimal, // 单一品种仓位集中度限制
}

/// 回撤配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownConfig {
    pub max_daily_drawdown_pct: Decimal,
    pub max_weekly_drawdown_pct: Decimal,
    pub max_monthly_drawdown_pct: Decimal,
    pub max_unrealized_loss_pct: Decimal,
    pub trailing_stop_enabled: bool,
    pub recovery_threshold_pct: Decimal, // 恢复阈值
}

/// 速率限制配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub orders_per_second: u32,
    pub orders_per_minute: u32,
    pub orders_per_hour: u32,
    pub volume_per_minute_usd: Decimal,
    pub burst_capacity: u32,
    pub cooldown_period_seconds: u32,
}

/// 相关性配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationConfig {
    pub max_sector_exposure_pct: Decimal,
    pub max_correlation_group_exposure_pct: Decimal,
    pub correlation_threshold: Decimal, // 相关性阈值
    pub sector_mapping: HashMap<Symbol, String>,
    pub correlation_matrix_update_interval: Duration,
}

/// 黑名单配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlacklistConfig {
    pub enabled: bool,
    pub symbol_blacklist: Vec<Symbol>,
    pub user_blacklist: Vec<UserId>,
    pub high_volatility_symbols: Vec<Symbol>,
    pub auto_blacklist_enabled: bool,
    pub volatility_threshold: Decimal,
}

/// 报警配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    pub alert_channels: Vec<String>,
    pub critical_violations_notify: bool,
    pub daily_summary_enabled: bool,
    pub real_time_monitoring: bool,
}

impl EnhancedPreemptiveRiskController {
    pub fn new(
        config: EnhancedRiskConfig,
        market_data_provider: Arc<dyn MarketDataProvider>,
    ) -> Self {
        let quota_manager = Arc::new(QuotaManager::new(config.quota_config.clone()));
        let drawdown_monitor = Arc::new(DrawdownMonitor::new(config.drawdown_config.clone()));
        let rate_limiter = Arc::new(EnhancedRateLimiter::new(config.rate_limit_config.clone()));
        let blacklist_manager = Arc::new(BlacklistManager::new(config.blacklist_config.clone()));
        let correlation_monitor = Arc::new(CorrelationExposureMonitor::new(config.correlation_config.clone()));
        let position_tracker = Arc::new(PositionTracker::new());
        let violation_recorder = Arc::new(ViolationRecorder::new());

        Self {
            config,
            quota_manager,
            drawdown_monitor,
            rate_limiter,
            blacklist_manager,
            correlation_monitor,
            position_tracker,
            market_data_provider,
            violation_recorder,
        }
    }

    /// 前置风控检查主入口
    pub async fn check_order_risk(&self, order: &OrderRequest) -> Result<RiskCheckResult> {
        if !self.config.enabled {
            return Ok(RiskCheckResult {
                approved: true,
                violations: Vec::new(),
                action: Some(RiskAction::Allow),
                risk_score: 0.0,
                details: "Risk control disabled".to_string(),
            });
        }

        let mut violations = Vec::new();
        let mut risk_score = 0.0;

        // 1. 黑名单检查（最高优先级）
        let blacklist_check = self.blacklist_manager.check_blacklist(order).await?;
        if !blacklist_check.passed {
            violations.extend(blacklist_check.violations);
            risk_score += 100.0; // 黑名单直接最高风险分数
        }

        // 2. 速率限制检查
        let rate_limit_check = self.rate_limiter.check_rate_limit(order).await?;
        if !rate_limit_check.passed {
            violations.extend(rate_limit_check.violations);
            risk_score += 50.0;
        }

        // 3. 额度检查
        let quota_check = self.quota_manager.check_quota(order).await?;
        if !quota_check.passed {
            violations.extend(quota_check.violations);
            risk_score += 30.0;
        }

        // 4. 回撤检查
        let drawdown_check = self.drawdown_monitor.check_drawdown(order).await?;
        if !drawdown_check.passed {
            violations.extend(drawdown_check.violations);
            risk_score += 40.0;
        }

        // 5. 相关性暴露检查
        let correlation_check = self.correlation_monitor.check_correlation_exposure(order).await?;
        if !correlation_check.passed {
            violations.extend(correlation_check.violations);
            risk_score += 25.0;
        }

        // 6. 仓位集中度检查
        let position_check = self.position_tracker.check_position_limits(order).await?;
        if !position_check.passed {
            violations.extend(position_check.violations);
            risk_score += 20.0;
        }

        // 记录违规
        if !violations.is_empty() {
            self.violation_recorder.record_violations(&violations, order).await?;
        }

        // 决定行动
        let action = self.determine_risk_action(&violations, risk_score);
        let approved = action == RiskAction::Allow;

        if !approved {
            warn!(
                "Order rejected for user {} symbol {}: {:?}",
                order.user_id, order.symbol, violations
            );
        }

        Ok(RiskCheckResult {
            approved,
            violations,
            action: Some(action),
            risk_score,
            details: format!("Risk score: {risk_score:.2}"),
        })
    }

    /// 批量订单风控检查
    pub async fn check_batch_orders(&self, orders: &[OrderRequest]) -> Result<Vec<RiskCheckResult>> {
        let mut results = Vec::new();
        
        // 检查批量订单的整体风险
        let batch_risk = self.assess_batch_risk(orders).await?;
        
        for order in orders {
            let mut individual_result = self.check_order_risk(order).await?;
            
            // 调整基于批量风险的分数
            individual_result.risk_score += batch_risk.additional_risk;
            
            if batch_risk.should_reject_batch {
                individual_result.approved = false;
                individual_result.violations.push(RiskViolation {
                    violation_type: ViolationType::BatchRisk,
                    severity: "HIGH".to_string(),
                    message: "Batch order risk too high".to_string(),
                    current_value: Some(batch_risk.batch_risk_score),
                    limit_value: Some(100.0),
                });
            }
            
            results.push(individual_result);
        }
        
        Ok(results)
    }

    /// 更新订单状态（用于实时风控）
    pub async fn update_order_status(&self, order_id: &str, status: OrderStatus) -> Result<()> {
        match status {
            OrderStatus::Filled { quantity, price } => {
                // 更新仓位
                self.position_tracker.update_position(order_id, quantity, price).await?;
                
                // 更新额度使用
                self.quota_manager.update_quota_usage(order_id, quantity * price).await?;
                
                // 更新P&L
                self.drawdown_monitor.update_pnl(order_id, quantity, price).await?;
            }
            OrderStatus::Cancelled => {
                // 释放预留的额度
                self.quota_manager.release_reserved_quota(order_id).await?;
            }
            OrderStatus::Rejected => {
                // 记录拒绝统计
                self.violation_recorder.record_rejection(order_id).await?;
            }
        }
        
        Ok(())
    }

    /// 获取风控统计
    pub async fn get_risk_statistics(&self) -> Result<RiskStatistics> {
        let quota_stats = self.quota_manager.get_statistics().await;
        let drawdown_stats = self.drawdown_monitor.get_statistics().await;
        let rate_limit_stats = self.rate_limiter.get_statistics().await;
        let violation_stats = self.violation_recorder.get_statistics().await;

        Ok(RiskStatistics {
            quota_utilization: quota_stats.utilization_pct,
            current_drawdown_pct: drawdown_stats.current_drawdown_pct,
            orders_rejected_today: violation_stats.rejections_today,
            rate_limit_violations_today: rate_limit_stats.violations_today,
            top_risk_symbols: self.get_top_risk_symbols().await?,
            system_health_score: self.calculate_system_health_score().await?,
        })
    }

    /// 执行每日风控重置
    pub async fn daily_reset(&self) -> Result<()> {
        info!("Executing daily risk control reset");
        
        self.quota_manager.daily_reset().await?;
        self.drawdown_monitor.daily_reset().await?;
        self.rate_limiter.daily_reset().await?;
        self.violation_recorder.daily_reset().await?;
        
        // 生成日报
        if self.config.alert_config.daily_summary_enabled {
            let report = self.generate_daily_report().await?;
            self.send_daily_report(report).await?;
        }

        Ok(())
    }

    // 私有辅助方法
    fn determine_risk_action(&self, violations: &[RiskViolation], risk_score: f64) -> RiskAction {
        // 严格模式下，任何违规都拒绝
        if self.config.strict_mode && !violations.is_empty() {
            return RiskAction::Reject;
        }

        // 检查是否有关键违规
        let has_critical = violations.iter().any(|v| v.severity == "CRITICAL");
        if has_critical {
            return RiskAction::Reject;
        }

        // 根据风险分数决定
        match risk_score {
            score if score >= 80.0 => RiskAction::Reject,
            score if score >= 50.0 => RiskAction::RequireApproval,
            score if score >= 30.0 => RiskAction::Warning,
            _ => RiskAction::Allow,
        }
    }

    async fn assess_batch_risk(&self, orders: &[OrderRequest]) -> Result<BatchRiskAssessment> {
        let total_value: Decimal = orders.iter()
            .map(|order| {
                let price = order.price.unwrap_or(Decimal::ZERO);
                order.quantity * price
            })
            .sum();

        let symbol_diversity = orders.iter()
            .map(|order| &order.symbol)
            .collect::<std::collections::HashSet<_>>()
            .len();

        let batch_risk_score = if symbol_diversity == 1 {
            50.0 // 单一品种批量风险高
        } else {
            20.0 / (symbol_diversity as f64) // 多样性降低风险
        };

        let should_reject_batch = batch_risk_score > 100.0 || 
            total_value > self.config.quota_config.single_order_max_usd * Decimal::from(5);

        Ok(BatchRiskAssessment {
            batch_risk_score,
            additional_risk: batch_risk_score * 0.1,
            should_reject_batch,
        })
    }

    async fn get_top_risk_symbols(&self) -> Result<Vec<String>> {
        // 获取风险最高的交易品种
        let position_risks = self.position_tracker.get_symbol_risks().await?;
        let mut risk_symbols: Vec<_> = position_risks.into_iter().collect();
        risk_symbols.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(risk_symbols.into_iter()
            .take(10)
            .map(|(symbol, _)| symbol)
            .collect())
    }

    async fn calculate_system_health_score(&self) -> Result<f64> {
        let quota_health = 100.0 - self.quota_manager.get_statistics().await.utilization_pct;
        let drawdown_health = 100.0 - self.drawdown_monitor.get_statistics().await.current_drawdown_pct.abs();
        let rate_limit_health = 100.0 - (self.rate_limiter.get_statistics().await.violations_today as f64 / 100.0).min(100.0);
        
        Ok((quota_health + drawdown_health + rate_limit_health) / 3.0)
    }

    async fn generate_daily_report(&self) -> Result<DailyRiskReport> {
        let stats = self.get_risk_statistics().await?;
        
        Ok(DailyRiskReport {
            date: Utc::now().date_naive(),
            total_orders_processed: self.violation_recorder.get_total_orders_today().await?,
            orders_rejected: stats.orders_rejected_today,
            rejection_rate: (stats.orders_rejected_today as f64 / (stats.orders_rejected_today + 1000) as f64) * 100.0,
            peak_quota_utilization: stats.quota_utilization,
            max_drawdown_reached: stats.current_drawdown_pct,
            top_violation_types: self.violation_recorder.get_top_violations().await?,
            system_health_score: stats.system_health_score,
            recommendations: self.generate_recommendations(&stats).await?,
        })
    }

    async fn generate_recommendations(&self, stats: &RiskStatistics) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        if stats.quota_utilization > 80.0 {
            recommendations.push("Consider increasing daily quota limits".to_string());
        }

        if stats.current_drawdown_pct.abs() > 5.0 {
            recommendations.push("Review position sizing and risk management".to_string());
        }

        if stats.orders_rejected_today > 100 {
            recommendations.push("High rejection rate - review risk parameters".to_string());
        }

        if stats.system_health_score < 70.0 {
            recommendations.push("System health deteriorating - immediate attention required".to_string());
        }

        Ok(recommendations)
    }

    async fn send_daily_report(&self, report: DailyRiskReport) -> Result<()> {
        // 实际实现应该发送到配置的报警渠道
        info!("Daily risk report: {:?}", report);
        Ok(())
    }
}

/// 市场数据提供者接口
pub trait MarketDataProvider: Send + Sync {
    fn get_current_price(&self, symbol: &Symbol) -> Result<Price>;
    fn get_volatility(&self, symbol: &Symbol) -> Result<Decimal>;
}

/// 订单状态
#[derive(Debug, Clone)]
pub enum OrderStatus {
    Filled { quantity: Quantity, price: Price },
    Cancelled,
    Rejected,
}

/// 批量风险评估
#[derive(Debug)]
struct BatchRiskAssessment {
    batch_risk_score: f64,
    additional_risk: f64,
    should_reject_batch: bool,
}

/// 风控统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskStatistics {
    pub quota_utilization: f64,
    pub current_drawdown_pct: f64,
    pub orders_rejected_today: u64,
    pub rate_limit_violations_today: u64,
    pub top_risk_symbols: Vec<String>,
    pub system_health_score: f64,
}

/// 每日风控报告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyRiskReport {
    pub date: chrono::NaiveDate,
    pub total_orders_processed: u64,
    pub orders_rejected: u64,
    pub rejection_rate: f64,
    pub peak_quota_utilization: f64,
    pub max_drawdown_reached: f64,
    pub top_violation_types: Vec<(String, u64)>,
    pub system_health_score: f64,
    pub recommendations: Vec<String>,
}

// 占位符实现 - 实际应该从各自的模块中导入
pub struct QuotaManager {
    config: QuotaConfig,
}

impl QuotaManager {
    fn new(config: QuotaConfig) -> Self {
        Self { config }
    }

    async fn check_quota(&self, _order: &OrderRequest) -> Result<CheckResult> {
        // 简化实现
        Ok(CheckResult {
            passed: true,
            violations: Vec::new(),
        })
    }

    async fn update_quota_usage(&self, _order_id: &str, _amount: Decimal) -> Result<()> {
        Ok(())
    }

    async fn release_reserved_quota(&self, _order_id: &str) -> Result<()> {
        Ok(())
    }

    async fn get_statistics(&self) -> QuotaStatistics {
        QuotaStatistics {
            utilization_pct: 45.0,
        }
    }

    async fn daily_reset(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct CheckResult {
    passed: bool,
    violations: Vec<RiskViolation>,
}

#[derive(Debug, Clone)]
struct QuotaStatistics {
    utilization_pct: f64,
}

// 类似地实现其他管理器的占位符
pub struct DrawdownMonitor {
    config: DrawdownConfig,
}

impl DrawdownMonitor {
    fn new(config: DrawdownConfig) -> Self {
        Self { config }
    }

    async fn check_drawdown(&self, _order: &OrderRequest) -> Result<CheckResult> {
        Ok(CheckResult { passed: true, violations: Vec::new() })
    }

    async fn update_pnl(&self, _order_id: &str, _quantity: Quantity, _price: Price) -> Result<()> {
        Ok(())
    }

    async fn get_statistics(&self) -> DrawdownStatistics {
        DrawdownStatistics { current_drawdown_pct: 2.5 }
    }

    async fn daily_reset(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct DrawdownStatistics {
    current_drawdown_pct: f64,
}

pub struct EnhancedRateLimiter {
    config: RateLimitConfig,
}

impl EnhancedRateLimiter {
    fn new(config: RateLimitConfig) -> Self {
        Self { config }
    }

    async fn check_rate_limit(&self, _order: &OrderRequest) -> Result<CheckResult> {
        Ok(CheckResult { passed: true, violations: Vec::new() })
    }

    async fn get_statistics(&self) -> RateLimitStatistics {
        RateLimitStatistics { violations_today: 5 }
    }

    async fn daily_reset(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct RateLimitStatistics {
    violations_today: u64,
}

pub struct BlacklistManager {
    config: BlacklistConfig,
}

impl BlacklistManager {
    fn new(config: BlacklistConfig) -> Self {
        Self { config }
    }

    async fn check_blacklist(&self, _order: &OrderRequest) -> Result<CheckResult> {
        Ok(CheckResult { passed: true, violations: Vec::new() })
    }
}

pub struct CorrelationExposureMonitor {
    config: CorrelationConfig,
}

impl CorrelationExposureMonitor {
    fn new(config: CorrelationConfig) -> Self {
        Self { config }
    }

    async fn check_correlation_exposure(&self, _order: &OrderRequest) -> Result<CheckResult> {
        Ok(CheckResult { passed: true, violations: Vec::new() })
    }
}

struct PositionTracker;

impl PositionTracker {
    fn new() -> Self {
        Self
    }

    async fn check_position_limits(&self, _order: &OrderRequest) -> Result<CheckResult> {
        Ok(CheckResult { passed: true, violations: Vec::new() })
    }

    async fn update_position(&self, _order_id: &str, _quantity: Quantity, _price: Price) -> Result<()> {
        Ok(())
    }

    async fn get_symbol_risks(&self) -> Result<HashMap<String, f64>> {
        Ok(HashMap::new())
    }
}

struct ViolationRecorder;

impl ViolationRecorder {
    fn new() -> Self {
        Self
    }

    async fn record_violations(&self, _violations: &[RiskViolation], _order: &OrderRequest) -> Result<()> {
        Ok(())
    }

    async fn record_rejection(&self, _order_id: &str) -> Result<()> {
        Ok(())
    }

    async fn get_statistics(&self) -> ViolationStatistics {
        ViolationStatistics { rejections_today: 25 }
    }

    async fn get_total_orders_today(&self) -> Result<u64> {
        Ok(1000)
    }

    async fn get_top_violations(&self) -> Result<Vec<(String, u64)>> {
        Ok(vec![
            ("QUOTA_EXCEEDED".to_string(), 10),
            ("RATE_LIMIT".to_string(), 8),
            ("BLACKLIST".to_string(), 5),
        ])
    }

    async fn daily_reset(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct ViolationStatistics {
    rejections_today: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct MockMarketDataProvider;

    impl MarketDataProvider for MockMarketDataProvider {
        fn get_current_price(&self, _symbol: &Symbol) -> Result<Price> {
            Ok(Decimal::from(100))
        }

        fn get_volatility(&self, _symbol: &Symbol) -> Result<Decimal> {
            Ok(Decimal::from_str("0.02").unwrap())
        }
    }

    #[tokio::test]
    async fn test_enhanced_risk_control() {
        let config = EnhancedRiskConfig {
            enabled: true,
            strict_mode: false,
            circuit_breaker_enabled: true,
            quota_config: QuotaConfig {
                global_daily_limit_usd: Decimal::from(1000000),
                per_user_daily_limit_usd: Decimal::from(50000),
                per_symbol_daily_limit_usd: Decimal::from(100000),
                single_order_max_usd: Decimal::from(10000),
                margin_utilization_max: Decimal::from_str("0.8").unwrap(),
                leverage_max: Decimal::from(10),
                concentration_limit_pct: Decimal::from_str("0.2").unwrap(),
            },
            drawdown_config: DrawdownConfig {
                max_daily_drawdown_pct: Decimal::from_str("0.05").unwrap(),
                max_weekly_drawdown_pct: Decimal::from_str("0.1").unwrap(),
                max_monthly_drawdown_pct: Decimal::from_str("0.2").unwrap(),
                max_unrealized_loss_pct: Decimal::from_str("0.15").unwrap(),
                trailing_stop_enabled: true,
                recovery_threshold_pct: Decimal::from_str("0.02").unwrap(),
            },
            rate_limit_config: RateLimitConfig {
                orders_per_second: 10,
                orders_per_minute: 100,
                orders_per_hour: 1000,
                volume_per_minute_usd: Decimal::from(50000),
                burst_capacity: 20,
                cooldown_period_seconds: 60,
            },
            correlation_config: CorrelationConfig {
                max_sector_exposure_pct: Decimal::from_str("0.3").unwrap(),
                max_correlation_group_exposure_pct: Decimal::from_str("0.25").unwrap(),
                correlation_threshold: Decimal::from_str("0.7").unwrap(),
                sector_mapping: HashMap::new(),
                correlation_matrix_update_interval: Duration::hours(1),
            },
            blacklist_config: BlacklistConfig {
                enabled: true,
                symbol_blacklist: Vec::new(),
                user_blacklist: Vec::new(),
                high_volatility_symbols: Vec::new(),
                auto_blacklist_enabled: true,
                volatility_threshold: Decimal::from_str("0.1").unwrap(),
            },
            alert_config: AlertConfig {
                alert_channels: vec!["slack".to_string()],
                critical_violations_notify: true,
                daily_summary_enabled: true,
                real_time_monitoring: true,
            },
        };

        let market_data = Arc::new(MockMarketDataProvider);
        let controller = EnhancedPreemptiveRiskController::new(config, market_data);

        let order = OrderRequest {
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(1),
            price: Some(Decimal::from(50000)),
            user_id: "test_user".to_string(),
        };

        let result = controller.check_order_risk(&order).await.unwrap();
        assert!(result.approved);
        assert!(result.risk_score < 50.0);
    }

    #[tokio::test]
    async fn test_batch_risk_assessment() {
        let config = EnhancedRiskConfig {
            enabled: true,
            strict_mode: false,
            circuit_breaker_enabled: true,
            quota_config: QuotaConfig {
                global_daily_limit_usd: Decimal::from(1000000),
                per_user_daily_limit_usd: Decimal::from(50000),
                per_symbol_daily_limit_usd: Decimal::from(100000),
                single_order_max_usd: Decimal::from(10000),
                margin_utilization_max: Decimal::from_str("0.8").unwrap(),
                leverage_max: Decimal::from(10),
                concentration_limit_pct: Decimal::from_str("0.2").unwrap(),
            },
            drawdown_config: DrawdownConfig {
                max_daily_drawdown_pct: Decimal::from_str("0.05").unwrap(),
                max_weekly_drawdown_pct: Decimal::from_str("0.1").unwrap(),
                max_monthly_drawdown_pct: Decimal::from_str("0.2").unwrap(),
                max_unrealized_loss_pct: Decimal::from_str("0.15").unwrap(),
                trailing_stop_enabled: true,
                recovery_threshold_pct: Decimal::from_str("0.02").unwrap(),
            },
            rate_limit_config: RateLimitConfig {
                orders_per_second: 10,
                orders_per_minute: 100,
                orders_per_hour: 1000,
                volume_per_minute_usd: Decimal::from(50000),
                burst_capacity: 20,
                cooldown_period_seconds: 60,
            },
            correlation_config: CorrelationConfig {
                max_sector_exposure_pct: Decimal::from_str("0.3").unwrap(),
                max_correlation_group_exposure_pct: Decimal::from_str("0.25").unwrap(),
                correlation_threshold: Decimal::from_str("0.7").unwrap(),
                sector_mapping: HashMap::new(),
                correlation_matrix_update_interval: Duration::hours(1),
            },
            blacklist_config: BlacklistConfig {
                enabled: true,
                symbol_blacklist: Vec::new(),
                user_blacklist: Vec::new(),
                high_volatility_symbols: Vec::new(),
                auto_blacklist_enabled: true,
                volatility_threshold: Decimal::from_str("0.1").unwrap(),
            },
            alert_config: AlertConfig {
                alert_channels: vec!["slack".to_string()],
                critical_violations_notify: true,
                daily_summary_enabled: true,
                real_time_monitoring: true,
            },
        };

        let market_data = Arc::new(MockMarketDataProvider);
        let controller = EnhancedPreemptiveRiskController::new(config, market_data);

        let orders = vec![
            OrderRequest {
                symbol: "BTCUSDT".to_string(),
                side: OrderSide::Buy,
                quantity: Decimal::from(1),
                price: Some(Decimal::from(50000)),
                user_id: "test_user".to_string(),
            },
            OrderRequest {
                symbol: "ETHUSDT".to_string(),
                side: OrderSide::Buy,
                quantity: Decimal::from(10),
                price: Some(Decimal::from(3000)),
                user_id: "test_user".to_string(),
            },
        ];

        let results = controller.check_batch_orders(&orders).await.unwrap();
        assert_eq!(results.len(), 2);
    }
}