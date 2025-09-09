use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use tokio::sync::RwLock;
use tracing::{info, warn};
use std::sync::Arc;

// Temporary type definitions - should be replaced with proper platform_types import later
pub type Symbol = String;
pub type Price = Decimal;
pub type Quantity = Decimal;
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

/// 前置风控门控系统
pub struct PreemptiveRiskController {
    config: RiskControlConfig,
    account_limits: Arc<RwLock<HashMap<String, AccountRiskLimits>>>,
    position_monitor: Arc<RwLock<PositionTracker>>,
    rate_limiter: Arc<RwLock<RateLimiter>>,
    blacklist_manager: Arc<RwLock<BlacklistManager>>,
    correlation_monitor: Arc<RwLock<CorrelationMonitor>>,
    drawdown_tracker: Arc<RwLock<DrawdownTracker>>,
}

/// 风控配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskControlConfig {
    pub enabled: bool,
    pub strict_mode: bool, // 严格模式：任何违规都拒绝
    pub max_order_value: Decimal,
    pub max_position_concentration: Decimal, // 单一品种最大仓位比例
    pub max_correlation_exposure: Decimal, // 最大相关暴露
    pub rate_limit_per_second: u32,
    pub rate_limit_per_minute: u32,
    pub max_daily_loss_ratio: Decimal, // 最大日亏损比例
    pub max_total_drawdown: Decimal, // 最大总回撤
    pub emergency_stop_loss: Decimal, // 紧急止损水平
}

/// 账户风险限制
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountRiskLimits {
    pub account_id: String,
    pub max_position_size: Decimal,
    pub max_daily_loss: Decimal,
    pub current_daily_loss: Decimal,
    pub max_leverage: Decimal,
    pub allowed_symbols: Option<Vec<Symbol>>,
    pub blocked_symbols: Vec<Symbol>,
    pub daily_order_count: u32,
    pub max_daily_orders: u32,
    pub last_reset: DateTime<Utc>,
    pub is_active: bool,
}

/// 持仓跟踪器
#[derive(Debug, Clone, Default)]
pub struct PositionTracker {
    pub positions: HashMap<Symbol, PositionInfo>,
    pub total_exposure: Decimal,
    pub concentration_ratios: HashMap<Symbol, Decimal>,
}

/// 持仓信息
#[derive(Debug, Clone)]
pub struct PositionInfo {
    pub symbol: Symbol,
    pub size: Decimal,
    pub average_price: Price,
    pub current_value: Decimal,
    pub unrealized_pnl: Decimal,
    pub leverage: Decimal,
    pub last_updated: DateTime<Utc>,
}

/// 频率限制器
#[derive(Debug, Clone)]
pub struct RateLimiter {
    pub requests_per_second: Vec<DateTime<Utc>>,
    pub requests_per_minute: Vec<DateTime<Utc>>,
    pub max_per_second: u32,
    pub max_per_minute: u32,
}

/// 黑名单管理器
#[derive(Debug, Clone, Default)]
pub struct BlacklistManager {
    pub blocked_symbols: Vec<Symbol>,
    pub blocked_accounts: Vec<String>,
    pub temporary_blocks: HashMap<String, DateTime<Utc>>, // 临时封禁
    pub block_reasons: HashMap<String, String>,
}

/// 相关性监控
#[derive(Debug, Clone, Default)]
pub struct CorrelationMonitor {
    pub correlations: HashMap<(Symbol, Symbol), Decimal>,
    pub correlation_groups: HashMap<String, Vec<Symbol>>,
    pub group_exposures: HashMap<String, Decimal>,
    pub last_updated: DateTime<Utc>,
}

/// 回撤跟踪器
#[derive(Debug, Clone)]
pub struct DrawdownTracker {
    pub peak_equity: Decimal,
    pub current_equity: Decimal,
    pub max_drawdown: Decimal,
    pub current_drawdown: Decimal,
    pub daily_pnl: Decimal,
    pub daily_start_equity: Decimal,
    pub last_reset: DateTime<Utc>,
}

/// 风控检查结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskCheckResult {
    pub is_approved: bool,
    pub risk_score: Decimal, // 0.0-1.0
    pub violations: Vec<RiskViolation>,
    pub warnings: Vec<RiskWarning>,
    pub adjusted_quantity: Option<Quantity>,
    pub suggested_actions: Vec<String>,
}

/// 风控违规
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskViolation {
    pub violation_type: RiskViolationType,
    pub severity: RiskSeverity,
    pub description: String,
    pub current_value: Decimal,
    pub limit_value: Decimal,
    pub account_id: Option<String>,
    pub symbol: Option<Symbol>,
}

/// 违规类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RiskViolationType {
    ExcessiveOrderSize,
    PositionConcentration,
    DailyLossExceeded,
    DrawdownExceeded,
    RateLimit,
    BlacklistedSymbol,
    BlacklistedAccount,
    CorrelationLimit,
    LeverageLimit,
    InsufficientMargin,
}

/// 风险警告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskWarning {
    pub warning_type: RiskWarningType,
    pub message: String,
    pub threshold_ratio: Decimal, // 接近限制的比例
}

/// 警告类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RiskWarningType {
    ApproachingLimit,
    HighCorrelation,
    VolatilitySpike,
    LowLiquidity,
    MarketClosed,
}

/// 风险严重程度
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RiskSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl PreemptiveRiskController {
    pub async fn new(config: RiskControlConfig) -> Result<Self> {
        info!("Initializing preemptive risk controller with config: {:?}", config);
        
        let rate_limiter = Arc::new(RwLock::new(RateLimiter::new(
            config.rate_limit_per_second,
            config.rate_limit_per_minute,
        )));

        Ok(Self {
            config,
            account_limits: Arc::new(RwLock::new(HashMap::new())),
            position_monitor: Arc::new(RwLock::new(PositionTracker::default())),
            rate_limiter,
            blacklist_manager: Arc::new(RwLock::new(BlacklistManager::default())),
            correlation_monitor: Arc::new(RwLock::new(CorrelationMonitor::default())),
            drawdown_tracker: Arc::new(RwLock::new(DrawdownTracker::new())),
        })
    }

    /// 主要的前置风控检查函数
    pub async fn check_order_risk(&self, order: &OrderRequest, account_id: &str) -> Result<RiskCheckResult> {
        if !self.config.enabled {
            return Ok(RiskCheckResult::approved());
        }

        let mut result = RiskCheckResult::new();

        // 1. 频率限制检查
        self.check_rate_limit(&mut result).await?;

        // 2. 黑名单检查
        self.check_blacklist(order, account_id, &mut result).await?;

        // 3. 账户限制检查
        self.check_account_limits(order, account_id, &mut result).await?;

        // 4. 持仓集中度检查
        self.check_position_concentration(order, &mut result).await?;

        // 5. 相关性暴露检查
        self.check_correlation_exposure(order, &mut result).await?;

        // 6. 日损失检查
        self.check_daily_loss_limits(order, account_id, &mut result).await?;

        // 7. 回撤检查
        self.check_drawdown_limits(&mut result).await?;

        // 8. 订单规模检查
        self.check_order_size(order, &mut result).await?;

        // 9. 计算综合风险评分
        result.risk_score = self.calculate_risk_score(&result).await;

        // 10. 最终决策
        result.is_approved = self.make_final_decision(&result).await;

        if !result.is_approved {
            warn!("Order rejected by risk control: {:?}", result.violations);
        }

        Ok(result)
    }

    // ============ 各项风控检查实现 ============

    async fn check_rate_limit(&self, result: &mut RiskCheckResult) -> Result<()> {
        let mut limiter = self.rate_limiter.write().await;
        let now = Utc::now();

        // 清理过期记录
        limiter.cleanup_old_requests(now);

        // 检查每秒限制
        if limiter.requests_per_second.len() >= limiter.max_per_second as usize {
            result.add_violation(RiskViolation {
                violation_type: RiskViolationType::RateLimit,
                severity: RiskSeverity::Medium,
                description: "Per-second rate limit exceeded".to_string(),
                current_value: Decimal::from(limiter.requests_per_second.len()),
                limit_value: Decimal::from(limiter.max_per_second),
                account_id: None,
                symbol: None,
            });
        }

        // 检查每分钟限制
        if limiter.requests_per_minute.len() >= limiter.max_per_minute as usize {
            result.add_violation(RiskViolation {
                violation_type: RiskViolationType::RateLimit,
                severity: RiskSeverity::High,
                description: "Per-minute rate limit exceeded".to_string(),
                current_value: Decimal::from(limiter.requests_per_minute.len()),
                limit_value: Decimal::from(limiter.max_per_minute),
                account_id: None,
                symbol: None,
            });
        }

        // 记录本次请求
        limiter.requests_per_second.push(now);
        limiter.requests_per_minute.push(now);

        Ok(())
    }

    async fn check_blacklist(&self, order: &OrderRequest, account_id: &str, result: &mut RiskCheckResult) -> Result<()> {
        let blacklist = self.blacklist_manager.read().await;

        // 检查账户黑名单
        if blacklist.blocked_accounts.contains(&account_id.to_string()) {
            result.add_violation(RiskViolation {
                violation_type: RiskViolationType::BlacklistedAccount,
                severity: RiskSeverity::Critical,
                description: format!("Account {account_id} is blacklisted"),
                current_value: Decimal::ONE,
                limit_value: Decimal::ZERO,
                account_id: Some(account_id.to_string()),
                symbol: None,
            });
        }

        // 检查交易品种黑名单
        if blacklist.blocked_symbols.contains(&order.symbol) {
            result.add_violation(RiskViolation {
                violation_type: RiskViolationType::BlacklistedSymbol,
                severity: RiskSeverity::Critical,
                description: format!("Symbol {:?} is blacklisted", order.symbol),
                current_value: Decimal::ONE,
                limit_value: Decimal::ZERO,
                account_id: Some(account_id.to_string()),
                symbol: Some(order.symbol.clone()),
            });
        }

        Ok(())
    }

    async fn check_account_limits(&self, order: &OrderRequest, account_id: &str, result: &mut RiskCheckResult) -> Result<()> {
        let limits = self.account_limits.read().await;
        
        if let Some(account_limits) = limits.get(account_id) {
            if !account_limits.is_active {
                result.add_violation(RiskViolation {
                    violation_type: RiskViolationType::BlacklistedAccount,
                    severity: RiskSeverity::Critical,
                    description: "Account is inactive".to_string(),
                    current_value: Decimal::ZERO,
                    limit_value: Decimal::ONE,
                    account_id: Some(account_id.to_string()),
                    symbol: None,
                });
            }

            // 检查日订单数量限制
            if account_limits.daily_order_count >= account_limits.max_daily_orders {
                result.add_violation(RiskViolation {
                    violation_type: RiskViolationType::RateLimit,
                    severity: RiskSeverity::Medium,
                    description: "Daily order limit reached".to_string(),
                    current_value: Decimal::from(account_limits.daily_order_count),
                    limit_value: Decimal::from(account_limits.max_daily_orders),
                    account_id: Some(account_id.to_string()),
                    symbol: None,
                });
            }

            // 检查允许的交易品种
            if let Some(ref allowed_symbols) = account_limits.allowed_symbols {
                if !allowed_symbols.contains(&order.symbol) {
                    result.add_violation(RiskViolation {
                        violation_type: RiskViolationType::BlacklistedSymbol,
                        severity: RiskSeverity::Medium,
                        description: "Symbol not in allowed list".to_string(),
                        current_value: Decimal::ZERO,
                        limit_value: Decimal::ONE,
                        account_id: Some(account_id.to_string()),
                        symbol: Some(order.symbol.clone()),
                    });
                }
            }
        }

        Ok(())
    }

    async fn check_position_concentration(&self, order: &OrderRequest, result: &mut RiskCheckResult) -> Result<()> {
        let positions = self.position_monitor.read().await;
        
        if let Some(position) = positions.positions.get(&order.symbol) {
            // 计算新订单后的集中度
            let order_value = order.quantity * order.price.unwrap_or(Decimal::ZERO);
            let new_position_value = position.current_value + order_value;
            let concentration_ratio = new_position_value / positions.total_exposure;

            if concentration_ratio > self.config.max_position_concentration {
                result.add_violation(RiskViolation {
                    violation_type: RiskViolationType::PositionConcentration,
                    severity: RiskSeverity::High,
                    description: format!("Position concentration too high for {}", order.symbol),
                    current_value: concentration_ratio,
                    limit_value: self.config.max_position_concentration,
                    account_id: None,
                    symbol: Some(order.symbol.clone()),
                });
            }
        }

        Ok(())
    }

    async fn check_correlation_exposure(&self, order: &OrderRequest, result: &mut RiskCheckResult) -> Result<()> {
        let correlation_monitor = self.correlation_monitor.read().await;
        
        // 检查相关性组合暴露
        for (group_name, exposure) in &correlation_monitor.group_exposures {
            if *exposure > self.config.max_correlation_exposure {
                result.add_violation(RiskViolation {
                    violation_type: RiskViolationType::CorrelationLimit,
                    severity: RiskSeverity::Medium,
                    description: format!("Correlation group {group_name} exposure too high"),
                    current_value: *exposure,
                    limit_value: self.config.max_correlation_exposure,
                    account_id: None,
                    symbol: Some(order.symbol.clone()),
                });
            }
        }

        Ok(())
    }

    async fn check_daily_loss_limits(&self, _order: &OrderRequest, account_id: &str, result: &mut RiskCheckResult) -> Result<()> {
        let limits = self.account_limits.read().await;
        
        if let Some(account_limits) = limits.get(account_id) {
            let daily_loss_ratio = account_limits.current_daily_loss / account_limits.max_daily_loss;
            
            if daily_loss_ratio > self.config.max_daily_loss_ratio {
                result.add_violation(RiskViolation {
                    violation_type: RiskViolationType::DailyLossExceeded,
                    severity: RiskSeverity::Critical,
                    description: "Daily loss limit exceeded".to_string(),
                    current_value: account_limits.current_daily_loss,
                    limit_value: account_limits.max_daily_loss,
                    account_id: Some(account_id.to_string()),
                    symbol: None,
                });
            }
        }

        Ok(())
    }

    async fn check_drawdown_limits(&self, result: &mut RiskCheckResult) -> Result<()> {
        let drawdown_tracker = self.drawdown_tracker.read().await;
        
        if drawdown_tracker.current_drawdown > self.config.max_total_drawdown {
            result.add_violation(RiskViolation {
                violation_type: RiskViolationType::DrawdownExceeded,
                severity: RiskSeverity::Critical,
                description: "Maximum drawdown exceeded".to_string(),
                current_value: drawdown_tracker.current_drawdown,
                limit_value: self.config.max_total_drawdown,
                account_id: None,
                symbol: None,
            });
        }

        Ok(())
    }

    async fn check_order_size(&self, order: &OrderRequest, result: &mut RiskCheckResult) -> Result<()> {
        let order_value = order.quantity * order.price.unwrap_or(Decimal::ZERO);
        
        if order_value > self.config.max_order_value {
            result.add_violation(RiskViolation {
                violation_type: RiskViolationType::ExcessiveOrderSize,
                severity: RiskSeverity::High,
                description: "Order size too large".to_string(),
                current_value: order_value,
                limit_value: self.config.max_order_value,
                account_id: None,
                symbol: Some(order.symbol.clone()),
            });

            // 建议调整订单数量
            let price_decimal = order.price.unwrap_or(Decimal::ONE);
            let adjusted_qty_decimal = self.config.max_order_value / price_decimal;
            result.adjusted_quantity = Some(adjusted_qty_decimal);
        }

        Ok(())
    }

    async fn calculate_risk_score(&self, result: &RiskCheckResult) -> Decimal {
        let mut score = Decimal::ZERO;
        
        for violation in &result.violations {
            let weight = match violation.severity {
                RiskSeverity::Critical => Decimal::from_parts(4, 0, 0, false, 1), // 0.4
                RiskSeverity::High => Decimal::from_parts(3, 0, 0, false, 1),     // 0.3
                RiskSeverity::Medium => Decimal::from_parts(2, 0, 0, false, 1),   // 0.2
                RiskSeverity::Low => Decimal::from_parts(1, 0, 0, false, 1),      // 0.1
            };
            score += weight;
        }
        
        score.min(Decimal::ONE)
    }

    async fn make_final_decision(&self, result: &RiskCheckResult) -> bool {
        // 严格模式：任何违规都拒绝
        if self.config.strict_mode && !result.violations.is_empty() {
            return false;
        }

        // 检查是否有关键违规
        for violation in &result.violations {
            if violation.severity == RiskSeverity::Critical {
                return false;
            }
        }

        // 基于风险评分决策
        result.risk_score < Decimal::from_parts(8, 0, 0, false, 1) // 0.8
    }

    /// 更新持仓信息
    pub async fn update_position(&self, symbol: Symbol, position_info: PositionInfo) -> Result<()> {
        let mut tracker = self.position_monitor.write().await;
        tracker.positions.insert(symbol, position_info);
        tracker.recalculate_exposures().await;
        Ok(())
    }

    /// 更新账户限制
    pub async fn update_account_limits(&self, account_id: String, limits: AccountRiskLimits) -> Result<()> {
        let mut account_limits = self.account_limits.write().await;
        account_limits.insert(account_id, limits);
        Ok(())
    }
}

// ============ 实现辅助结构的方法 ============

impl Default for RiskCheckResult {
    fn default() -> Self {
        Self::new()
    }
}

impl RiskCheckResult {
    pub fn new() -> Self {
        Self {
            is_approved: true,
            risk_score: Decimal::ZERO,
            violations: Vec::new(),
            warnings: Vec::new(),
            adjusted_quantity: None,
            suggested_actions: Vec::new(),
        }
    }

    pub fn approved() -> Self {
        Self {
            is_approved: true,
            risk_score: Decimal::ZERO,
            violations: Vec::new(),
            warnings: Vec::new(),
            adjusted_quantity: None,
            suggested_actions: Vec::new(),
        }
    }

    pub fn add_violation(&mut self, violation: RiskViolation) {
        self.violations.push(violation);
    }

    pub fn add_warning(&mut self, warning: RiskWarning) {
        self.warnings.push(warning);
    }
}

impl RateLimiter {
    pub fn new(max_per_second: u32, max_per_minute: u32) -> Self {
        Self {
            requests_per_second: Vec::new(),
            requests_per_minute: Vec::new(),
            max_per_second,
            max_per_minute,
        }
    }

    pub fn cleanup_old_requests(&mut self, now: DateTime<Utc>) {
        let one_second_ago = now - Duration::seconds(1);
        let one_minute_ago = now - Duration::minutes(1);

        self.requests_per_second.retain(|&time| time > one_second_ago);
        self.requests_per_minute.retain(|&time| time > one_minute_ago);
    }
}

impl PositionTracker {
    pub async fn recalculate_exposures(&mut self) {
        self.total_exposure = self.positions.values()
            .map(|pos| pos.current_value)
            .sum();

        for (symbol, position) in &self.positions {
            if self.total_exposure > Decimal::ZERO {
                let concentration = position.current_value / self.total_exposure;
                self.concentration_ratios.insert(symbol.clone(), concentration);
            }
        }
    }
}

impl Default for DrawdownTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl DrawdownTracker {
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            peak_equity: Decimal::from(100000), // 假设初始资金
            current_equity: Decimal::from(100000),
            max_drawdown: Decimal::ZERO,
            current_drawdown: Decimal::ZERO,
            daily_pnl: Decimal::ZERO,
            daily_start_equity: Decimal::from(100000),
            last_reset: now,
        }
    }

    pub fn update_equity(&mut self, new_equity: Decimal) {
        self.current_equity = new_equity;
        
        // 更新峰值
        if new_equity > self.peak_equity {
            self.peak_equity = new_equity;
        }

        // 计算当前回撤
        self.current_drawdown = (self.peak_equity - new_equity) / self.peak_equity;
        
        // 更新最大回撤
        if self.current_drawdown > self.max_drawdown {
            self.max_drawdown = self.current_drawdown;
        }

        // 更新日PnL
        self.daily_pnl = new_equity - self.daily_start_equity;
    }
}

impl Default for RiskControlConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strict_mode: false,
            max_order_value: Decimal::from(100000),
            max_position_concentration: Decimal::from_parts(2, 0, 0, false, 1), // 0.2 (20%)
            max_correlation_exposure: Decimal::from_parts(5, 0, 0, false, 1),   // 0.5 (50%)
            rate_limit_per_second: 10,
            rate_limit_per_minute: 100,
            max_daily_loss_ratio: Decimal::from_parts(5, 0, 0, false, 2),       // 0.05 (5%)
            max_total_drawdown: Decimal::from_parts(1, 0, 0, false, 1),         // 0.1 (10%)
            emergency_stop_loss: Decimal::from_parts(2, 0, 0, false, 1),        // 0.2 (20%)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_risk_controller_creation() {
        let config = RiskControlConfig::default();
        let controller = PreemptiveRiskController::new(config).await;
        assert!(controller.is_ok());
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let config = RiskControlConfig {
            rate_limit_per_second: 1,
            rate_limit_per_minute: 2,
            ..RiskControlConfig::default()
        };

        let controller = PreemptiveRiskController::new(config).await.unwrap();
        
        let order = OrderRequest {
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(1),
            price: Some(Decimal::from(50000)),
            user_id: "test_user".to_string(),
        };

        // 第一个订单应该通过
        let result1 = controller.check_order_risk(&order, "test_account").await.unwrap();
        assert!(result1.is_approved);

        // 第二个订单应该被频率限制拒绝
        let result2 = controller.check_order_risk(&order, "test_account").await.unwrap();
        assert!(!result2.is_approved);
        assert!(!result2.violations.is_empty());
    }
}