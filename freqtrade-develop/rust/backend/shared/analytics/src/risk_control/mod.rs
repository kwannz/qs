use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};

pub mod preemptive_rules;
pub mod circuit_breakers;
pub mod exposure_limits;
pub mod velocity_controls;

/// AG3 前置风控门控系统
#[derive(Debug)]
pub struct PretradeRiskGateway {
    config: RiskGatewayConfig,
    rule_engine: Arc<RuleEngine>,
    blacklist_manager: Arc<RwLock<BlacklistManager>>,
    exposure_monitor: Arc<RwLock<ExposureMonitor>>,
    velocity_tracker: Arc<RwLock<VelocityTracker>>,
    circuit_breaker: Arc<RwLock<CircuitBreaker>>,
    statistics: Arc<RwLock<GatewayStatistics>>,
}

/// 风控网关配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskGatewayConfig {
    pub enabled: bool,
    pub fail_safe_mode: bool,
    pub max_check_timeout_ms: u64,
    pub emergency_mode_threshold: u32,
    pub auto_recovery_interval_secs: u64,
    
    // 额度限制
    pub max_position_size: Decimal,
    pub max_daily_loss: Decimal,
    pub max_drawdown_percent: Decimal,
    pub max_leverage: Decimal,
    
    // 速率限制
    pub max_orders_per_second: u32,
    pub max_orders_per_minute: u32,
    pub max_orders_per_hour: u32,
    pub max_volume_per_minute: Decimal,
    
    // 相关性限制
    pub max_correlation_exposure: Decimal,
    pub max_sector_concentration: Decimal,
    pub max_single_position_weight: Decimal,
    
    // 熔断设置
    pub enable_circuit_breakers: bool,
    pub drawdown_circuit_breaker: Decimal,  // 回撤熔断阈值
    pub loss_rate_circuit_breaker: Decimal, // 亏损速率熔断阈值
    pub volatility_circuit_breaker: Decimal, // 波动率熔断阈值
}

/// 前置风控检查请求
#[derive(Debug, Clone)]
pub struct PretradeRiskRequest {
    pub order_id: String,
    pub strategy_id: String,
    pub account_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: Decimal,
    pub price: Option<Decimal>,
    pub order_type: OrderType,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OrderType {
    Market,
    Limit,
    StopMarket,
    StopLimit,
}

/// 前置风控检查响应
#[derive(Debug, Clone)]
pub struct PretradeRiskResponse {
    pub approved: bool,
    pub decision: RiskDecision,
    pub violations: Vec<RiskViolation>,
    pub warnings: Vec<RiskWarning>,
    pub check_duration_ms: f64,
    pub decision_timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RiskDecision {
    Approve,
    Reject,
    ApproveWithWarning,
    RequireConfirmation,
}

#[derive(Debug, Clone)]
pub struct RiskViolation {
    pub rule_type: RiskRuleType,
    pub severity: RiskSeverity,
    pub description: String,
    pub current_value: Decimal,
    pub limit_value: Decimal,
    pub suggested_action: String,
}

#[derive(Debug, Clone)]
pub struct RiskWarning {
    pub rule_type: RiskRuleType,
    pub description: String,
    pub threshold_breach_percent: Decimal,
    pub recommendation: String,
}

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum RiskRuleType {
    PositionLimit,
    DailyLoss,
    DrawdownLimit,
    VelocityControl,
    CorrelationExposure,
    Blacklist,
    LeverageLimit,
    ConcentrationLimit,
    CircuitBreaker,
    LiquidityLimit,
    VolatilityLimit,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RiskSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// 规则引擎
#[derive(Debug)]
pub struct RuleEngine {
    rules: Vec<Box<dyn RiskRule>>,
    rule_config: HashMap<RiskRuleType, serde_json::Value>,
}

/// 风控规则特征
pub trait RiskRule: Send + Sync + std::fmt::Debug {
    fn rule_type(&self) -> RiskRuleType;
    fn check(&self, request: &PretradeRiskRequest, context: &RiskContext) -> Result<RiskCheckResult>;
    fn is_enabled(&self) -> bool;
    fn priority(&self) -> u8;
    fn get_config(&self) -> serde_json::Value;
}

/// 风控上下文
#[derive(Debug, Clone)]
pub struct RiskContext {
    pub current_positions: HashMap<String, Position>,
    pub daily_pnl: Decimal,
    pub total_equity: Decimal,
    pub max_drawdown: Decimal,
    pub correlation_matrix: HashMap<String, HashMap<String, Decimal>>,
    pub market_conditions: MarketConditions,
    pub account_state: AccountState,
}

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: Decimal,
    pub average_price: Decimal,
    pub market_value: Decimal,
    pub unrealized_pnl: Decimal,
    pub weight: Decimal, // 占总权益比重
}

#[derive(Debug, Clone)]
pub struct MarketConditions {
    pub timestamp: DateTime<Utc>,
    pub volatility_index: Decimal,
    pub market_stress_indicator: Decimal,
    pub liquidity_conditions: LiquidityConditions,
}

#[derive(Debug, Clone)]
pub struct LiquidityConditions {
    pub bid_ask_spread_percentile: Decimal,
    pub depth_ratio: Decimal,
    pub market_impact_estimate: Decimal,
}

#[derive(Debug, Clone)]
pub struct AccountState {
    pub account_id: String,
    pub total_equity: Decimal,
    pub available_margin: Decimal,
    pub used_margin: Decimal,
    pub leverage_ratio: Decimal,
    pub daily_pnl: Decimal,
    pub max_drawdown_today: Decimal,
    pub order_count_today: u32,
    pub volume_traded_today: Decimal,
}

/// 风控检查结果
#[derive(Debug, Clone)]
pub struct RiskCheckResult {
    pub passed: bool,
    pub violation: Option<RiskViolation>,
    pub warnings: Vec<RiskWarning>,
    pub suggested_adjustments: Vec<String>,
}

/// 黑名单管理器
#[derive(Debug, Default)]
pub struct BlacklistManager {
    strategy_blacklist: HashSet<String>,
    symbol_blacklist: HashSet<String>,
    account_blacklist: HashSet<String>,
    temporary_blocks: HashMap<String, DateTime<Utc>>,
}

/// 暴露监控器
#[derive(Debug)]
pub struct ExposureMonitor {
    position_limits: HashMap<String, Decimal>,
    sector_limits: HashMap<String, Decimal>,
    correlation_limits: HashMap<String, Decimal>,
    concentration_limits: HashMap<String, Decimal>,
}

/// 速度追踪器
#[derive(Debug)]
pub struct VelocityTracker {
    order_history: VecDeque<OrderEvent>,
    volume_history: VecDeque<VolumeEvent>,
    config: VelocityConfig,
}

#[derive(Debug, Clone)]
pub struct OrderEvent {
    pub timestamp: DateTime<Utc>,
    pub account_id: String,
    pub strategy_id: String,
    pub symbol: String,
    pub quantity: Decimal,
}

#[derive(Debug, Clone)]
pub struct VolumeEvent {
    pub timestamp: DateTime<Utc>,
    pub account_id: String,
    pub volume: Decimal,
}

#[derive(Debug, Clone)]
pub struct VelocityConfig {
    pub max_orders_per_second: u32,
    pub max_orders_per_minute: u32,
    pub max_orders_per_hour: u32,
    pub max_volume_per_minute: Decimal,
    pub max_volume_per_hour: Decimal,
    pub lookback_window_seconds: u64,
}

/// 熔断器
#[derive(Debug)]
pub struct CircuitBreaker {
    is_triggered: bool,
    trigger_conditions: Vec<CircuitBreakerCondition>,
    trigger_time: Option<DateTime<Utc>>,
    recovery_time: Option<DateTime<Utc>>,
    config: CircuitBreakerConfig,
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerCondition {
    pub condition_type: CircuitBreakerType,
    pub threshold: Decimal,
    pub current_value: Decimal,
    pub triggered_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CircuitBreakerType {
    DrawdownLimit,
    LossRate,
    VolatilitySpike,
    LiquidityDrop,
    ErrorRate,
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub enabled: bool,
    pub recovery_time_minutes: u32,
    pub gradual_recovery: bool,
    pub emergency_contacts: Vec<String>,
}

/// 网关统计
#[derive(Debug, Default)]
pub struct GatewayStatistics {
    pub total_checks: u64,
    pub approved_count: u64,
    pub rejected_count: u64,
    pub warning_count: u64,
    pub average_check_time_ms: f64,
    pub violations_by_type: HashMap<RiskRuleType, u64>,
    pub daily_statistics: HashMap<String, DailyStats>, // 按日期分组
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Default)]
pub struct DailyStats {
    pub date: String,
    pub total_checks: u64,
    pub rejections: u64,
    pub top_violations: Vec<(RiskRuleType, u64)>,
    pub avg_check_time_ms: f64,
    pub circuit_breaker_triggers: u32,
}

impl PretradeRiskGateway {
    pub fn new(config: RiskGatewayConfig) -> Result<Self> {
        Ok(Self {
            rule_engine: Arc::new(RuleEngine::new()?),
            blacklist_manager: Arc::new(RwLock::new(BlacklistManager::default())),
            exposure_monitor: Arc::new(RwLock::new(ExposureMonitor::new()?)),
            velocity_tracker: Arc::new(RwLock::new(VelocityTracker::new(config.clone().into())?)),
            circuit_breaker: Arc::new(RwLock::new(CircuitBreaker::new(config.clone().into())?)),
            statistics: Arc::new(RwLock::new(GatewayStatistics::default())),
            config,
        })
    }
    
    /// 执行前置风控检查
    pub async fn check_pretrade_risk(
        &self,
        request: PretradeRiskRequest,
        context: RiskContext,
    ) -> Result<PretradeRiskResponse> {
        let start_time = std::time::Instant::now();
        
        if !self.config.enabled {
            return Ok(PretradeRiskResponse {
                approved: true,
                decision: RiskDecision::Approve,
                violations: vec![],
                warnings: vec![],
                check_duration_ms: 0.0,
                decision_timestamp: Utc::now(),
            });
        }
        
        // 检查熔断器状态
        {
            let circuit_breaker = self.circuit_breaker.read().await;
            if circuit_breaker.is_triggered {
                return Ok(PretradeRiskResponse {
                    approved: false,
                    decision: RiskDecision::Reject,
                    violations: vec![RiskViolation {
                        rule_type: RiskRuleType::CircuitBreaker,
                        severity: RiskSeverity::Critical,
                        description: "Circuit breaker is active".to_string(),
                        current_value: Decimal::ONE,
                        limit_value: Decimal::ZERO,
                        suggested_action: "Wait for circuit breaker recovery".to_string(),
                    }],
                    warnings: vec![],
                    check_duration_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                    decision_timestamp: Utc::now(),
                });
            }
        }
        
        // 执行所有风控规则检查
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        
        // 按优先级执行规则检查
        for rule in &self.rule_engine.rules {
            if !rule.is_enabled() {
                continue;
            }
            
            match rule.check(&request, &context) {
                Ok(result) => {
                    if !result.passed {
                        if let Some(violation) = result.violation {
                            violations.push(violation);
                        }
                    }
                    warnings.extend(result.warnings);
                }
                Err(e) => {
                    error!("Risk rule check failed: {:?}", e);
                    if self.config.fail_safe_mode {
                        violations.push(RiskViolation {
                            rule_type: rule.rule_type(),
                            severity: RiskSeverity::High,
                            description: format!("Rule check failed: {}", e),
                            current_value: Decimal::ZERO,
                            limit_value: Decimal::ZERO,
                            suggested_action: "Manual review required".to_string(),
                        });
                    }
                }
            }
        }
        
        // 决策逻辑
        let decision = self.make_decision(&violations, &warnings)?;
        let approved = decision == RiskDecision::Approve || decision == RiskDecision::ApproveWithWarning;
        
        // 更新统计
        {
            let mut stats = self.statistics.write().await;
            stats.total_checks += 1;
            if approved {
                stats.approved_count += 1;
            } else {
                stats.rejected_count += 1;
            }
            if !warnings.is_empty() {
                stats.warning_count += 1;
            }
            
            // 按类型统计违规
            for violation in &violations {
                *stats.violations_by_type.entry(violation.rule_type.clone()).or_insert(0) += 1;
            }
            
            stats.last_updated = Utc::now();
        }
        
        let check_duration_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        
        info!("Pretrade risk check completed: approved={}, violations={}, warnings={}, duration={:.2}ms",
              approved, violations.len(), warnings.len(), check_duration_ms);
        
        Ok(PretradeRiskResponse {
            approved,
            decision,
            violations,
            warnings,
            check_duration_ms,
            decision_timestamp: Utc::now(),
        })
    }
    
    /// 决策制定
    fn make_decision(
        &self,
        violations: &[RiskViolation],
        warnings: &[RiskWarning],
    ) -> Result<RiskDecision> {
        if violations.is_empty() {
            if warnings.is_empty() {
                Ok(RiskDecision::Approve)
            } else {
                Ok(RiskDecision::ApproveWithWarning)
            }
        } else {
            // 检查是否有关键违规
            let has_critical = violations.iter().any(|v| v.severity == RiskSeverity::Critical);
            if has_critical {
                Ok(RiskDecision::Reject)
            } else {
                // 中等或低风险违规可能需要确认
                Ok(RiskDecision::RequireConfirmation)
            }
        }
    }
    
    /// 获取统计信息
    pub async fn get_statistics(&self) -> GatewayStatistics {
        let stats = self.statistics.read().await;
        GatewayStatistics {
            total_checks: stats.total_checks,
            approved_count: stats.approved_count,
            rejected_count: stats.rejected_count,
            warning_count: stats.warning_count,
            average_check_time_ms: stats.average_check_time_ms,
            violations_by_type: stats.violations_by_type.clone(),
            daily_statistics: stats.daily_statistics.clone(),
            last_updated: stats.last_updated,
        }
    }
    
    /// 添加到黑名单
    pub async fn add_to_blacklist(
        &self,
        item_type: &str,
        item_id: &str,
        duration: Option<Duration>,
    ) -> Result<()> {
        let mut blacklist = self.blacklist_manager.write().await;
        
        match item_type {
            "strategy" => {
                blacklist.strategy_blacklist.insert(item_id.to_string());
            }
            "symbol" => {
                blacklist.symbol_blacklist.insert(item_id.to_string());
            }
            "account" => {
                blacklist.account_blacklist.insert(item_id.to_string());
            }
            _ => return Err(anyhow::anyhow!("Invalid blacklist item type: {}", item_type)),
        }
        
        if let Some(duration) = duration {
            let expiry = Utc::now() + duration;
            blacklist.temporary_blocks.insert(format!("{}:{}", item_type, item_id), expiry);
        }
        
        info!("Added {} {} to blacklist", item_type, item_id);
        Ok(())
    }
    
    /// 触发熔断器
    pub async fn trigger_circuit_breaker(
        &self,
        condition_type: CircuitBreakerType,
        current_value: Decimal,
        threshold: Decimal,
    ) -> Result<()> {
        let mut circuit_breaker = self.circuit_breaker.write().await;
        
        if !circuit_breaker.is_triggered {
            circuit_breaker.is_triggered = true;
            circuit_breaker.trigger_time = Some(Utc::now());
            
            let recovery_duration = Duration::minutes(circuit_breaker.config.recovery_time_minutes as i64);
            circuit_breaker.recovery_time = Some(Utc::now() + recovery_duration);
            
            error!("Circuit breaker triggered: {:?}, current={}, threshold={}", 
                   condition_type, current_value, threshold);
            
            // 这里可以添加告警通知逻辑
        }
        
        Ok(())
    }
}

// 实现各个组件的构造函数
impl RuleEngine {
    fn new() -> Result<Self> {
        let mut rules: Vec<Box<dyn RiskRule>> = Vec::new();
        let mut rule_config = HashMap::new();
        
        // 添加熔断器规则（最高优先级）
        let circuit_breaker_config = circuit_breakers::CircuitBreakerConfig::default();
        rule_config.insert(RiskRuleType::CircuitBreaker, serde_json::to_value(&circuit_breaker_config)?);
        rules.push(Box::new(circuit_breakers::CircuitBreakerRule::new(circuit_breaker_config)?));
        
        // 添加速度控制规则
        let velocity_config = velocity_controls::VelocityControlConfig::default();
        rule_config.insert(RiskRuleType::VelocityControl, serde_json::to_value(&velocity_config)?);
        rules.push(Box::new(velocity_controls::VelocityControlRule::new(velocity_config)?));
        
        // 添加暴露限制规则
        let exposure_config = exposure_limits::ExposureLimitsConfig::default();
        rule_config.insert(RiskRuleType::ConcentrationLimit, serde_json::to_value(&exposure_config)?);
        rules.push(Box::new(exposure_limits::ExposureLimitsRule::new(exposure_config)?));
        
        // 添加预置规则
        let position_limit_rule = preemptive_rules::PositionLimitRule::new(
            Decimal::from(1000000), // $1M limit
            100, // max 100 positions
        );
        rules.push(Box::new(position_limit_rule));
        
        let daily_loss_rule = preemptive_rules::DailyLossRule::new(
            Decimal::from(50000), // $50K daily loss limit
        );
        rules.push(Box::new(daily_loss_rule));
        
        let drawdown_rule = preemptive_rules::DrawdownRule::new(
            Decimal::from_parts(10, 0, 0, false, 2), // 10% drawdown limit
        );
        rules.push(Box::new(drawdown_rule));
        
        let leverage_rule = preemptive_rules::LeverageRule::new(
            Decimal::from(5), // 5x leverage limit
        );
        rules.push(Box::new(leverage_rule));
        
        let blacklist_rule = preemptive_rules::BlacklistRule::new();
        rules.push(Box::new(blacklist_rule));
        
        // 高级规则
        let correlation_rule = preemptive_rules::CorrelationExposureRule::new(
            preemptive_rules::CorrelationConfig::default()
        )?;
        rules.push(Box::new(correlation_rule));
        
        let liquidity_rule = preemptive_rules::LiquidityRule::new(
            preemptive_rules::LiquidityConfig::default()
        );
        rules.push(Box::new(liquidity_rule));
        
        let volatility_rule = preemptive_rules::VolatilityRule::new(
            preemptive_rules::VolatilityConfig::default()
        );
        rules.push(Box::new(volatility_rule));
        
        let concentration_rule = preemptive_rules::ConcentrationRule::new(
            preemptive_rules::ConcentrationConfig::default()
        );
        rules.push(Box::new(concentration_rule));
        
        // 按优先级排序
        rules.sort_by(|a, b| a.priority().cmp(&b.priority()));
        
        Ok(Self {
            rules,
            rule_config,
        })
    }
    
    /// 添加自定义规则
    pub fn add_rule(&mut self, rule: Box<dyn RiskRule>) {
        self.rules.push(rule);
        // 重新排序
        self.rules.sort_by(|a, b| a.priority().cmp(&b.priority()));
    }
    
    /// 移除规则
    pub fn remove_rule(&mut self, rule_type: RiskRuleType) {
        self.rules.retain(|rule| rule.rule_type() != rule_type);
    }
    
    /// 获取规则统计
    pub fn get_rule_statistics(&self) -> HashMap<RiskRuleType, serde_json::Value> {
        self.rules.iter()
            .map(|rule| (rule.rule_type(), rule.get_config()))
            .collect()
    }
}

impl ExposureMonitor {
    fn new() -> Result<Self> {
        Ok(Self {
            position_limits: HashMap::new(),
            sector_limits: HashMap::new(),
            correlation_limits: HashMap::new(),
            concentration_limits: HashMap::new(),
        })
    }
}

impl VelocityTracker {
    fn new(config: VelocityConfig) -> Result<Self> {
        Ok(Self {
            order_history: VecDeque::new(),
            volume_history: VecDeque::new(),
            config,
        })
    }
}

impl CircuitBreaker {
    fn new(config: CircuitBreakerConfig) -> Result<Self> {
        Ok(Self {
            is_triggered: false,
            trigger_conditions: Vec::new(),
            trigger_time: None,
            recovery_time: None,
            config,
        })
    }
}

// 类型转换实现
impl From<RiskGatewayConfig> for VelocityConfig {
    fn from(config: RiskGatewayConfig) -> Self {
        Self {
            max_orders_per_second: config.max_orders_per_second,
            max_orders_per_minute: config.max_orders_per_minute,
            max_orders_per_hour: config.max_orders_per_hour,
            max_volume_per_minute: config.max_volume_per_minute,
            max_volume_per_hour: config.max_volume_per_minute * Decimal::from(60),
            lookback_window_seconds: 3600, // 1小时
        }
    }
}

impl From<RiskGatewayConfig> for CircuitBreakerConfig {
    fn from(config: RiskGatewayConfig) -> Self {
        Self {
            enabled: config.enable_circuit_breakers,
            recovery_time_minutes: (config.auto_recovery_interval_secs / 60) as u32,
            gradual_recovery: true,
            emergency_contacts: Vec::new(),
        }
    }
}

impl From<RiskGatewayConfig> for circuit_breakers::CircuitBreakerConfig {
    fn from(config: RiskGatewayConfig) -> Self {
        Self {
            enabled: config.enable_circuit_breakers,
            priority: 1,
            drawdown_threshold: config.drawdown_circuit_breaker,
            drawdown_window_minutes: 60,
            drawdown_recovery_minutes: (config.auto_recovery_interval_secs / 60) as u32,
            loss_rate_threshold: config.loss_rate_circuit_breaker,
            loss_rate_window_minutes: 60,
            loss_rate_recovery_minutes: (config.auto_recovery_interval_secs / 60) as u32,
            volatility_threshold: config.volatility_circuit_breaker,
            volatility_window_minutes: 30,
            volatility_recovery_minutes: (config.auto_recovery_interval_secs / 60) as u32,
            liquidity_threshold: Decimal::from_parts(30, 0, 0, false, 2), // 30%
            liquidity_window_minutes: 15,
            liquidity_recovery_minutes: 5,
            error_rate_threshold: Decimal::from_parts(10, 0, 0, false, 2), // 10%
            error_rate_window_minutes: 10,
            error_rate_recovery_minutes: 5,
            cascade_protection_enabled: true,
            cascade_threshold: 3,
            cascade_recovery_minutes: (config.auto_recovery_interval_secs / 60) as u32,
            gradual_recovery_enabled: true,
            recovery_step_percent: Decimal::from_parts(25, 0, 0, false, 2),
            recovery_step_interval_minutes: 5,
            alert_enabled: true,
            alert_cooldown_minutes: 10,
            emergency_contacts: Vec::new(),
        }
    }
}

impl Default for RiskGatewayConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            fail_safe_mode: true,
            max_check_timeout_ms: 100,
            emergency_mode_threshold: 10,
            auto_recovery_interval_secs: 300,
            max_position_size: Decimal::from(1000000), // $1M
            max_daily_loss: Decimal::from(50000),       // $50K
            max_drawdown_percent: Decimal::from_parts(10, 0, 0, false, 2), // 10%
            max_leverage: Decimal::from(10),
            max_orders_per_second: 10,
            max_orders_per_minute: 300,
            max_orders_per_hour: 1000,
            max_volume_per_minute: Decimal::from(1000000),
            max_correlation_exposure: Decimal::from_parts(30, 0, 0, false, 2), // 30%
            max_sector_concentration: Decimal::from_parts(25, 0, 0, false, 2), // 25%
            max_single_position_weight: Decimal::from_parts(5, 0, 0, false, 2), // 5%
            enable_circuit_breakers: true,
            drawdown_circuit_breaker: Decimal::from_parts(15, 0, 0, false, 2), // 15%
            loss_rate_circuit_breaker: Decimal::from_parts(5, 0, 0, false, 2),  // 5%/hour
            volatility_circuit_breaker: Decimal::from_parts(50, 0, 0, false, 2), // 50%
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use std::collections::HashMap;

    pub fn create_test_context() -> RiskContext {
        let mut positions = HashMap::new();
        positions.insert("BTCUSDT".to_string(), Position {
            symbol: "BTCUSDT".to_string(),
            quantity: Decimal::from(10),
            average_price: Decimal::from(50000),
            market_value: Decimal::from(500000),
            unrealized_pnl: Decimal::from(25000),
            weight: Decimal::from_parts(50, 0, 0, false, 2), // 50%
        });

        let mut correlation_matrix = HashMap::new();
        let mut btc_correlations = HashMap::new();
        btc_correlations.insert("ETHUSDT".to_string(), Decimal::from_parts(75, 0, 0, false, 2)); // 0.75
        correlation_matrix.insert("BTCUSDT".to_string(), btc_correlations);

        RiskContext {
            current_positions: positions,
            daily_pnl: Decimal::from(10000),
            total_equity: Decimal::from(1000000),
            max_drawdown: Decimal::from_parts(5, 0, 0, false, 2), // 5%
            correlation_matrix,
            market_conditions: MarketConditions {
                timestamp: Utc::now(),
                volatility_index: Decimal::from_parts(25, 0, 0, false, 2), // 25%
                market_stress_indicator: Decimal::from_parts(20, 0, 0, false, 2), // 20%
                liquidity_conditions: LiquidityConditions {
                    bid_ask_spread_percentile: Decimal::from_parts(15, 0, 0, false, 2), // 15%
                    depth_ratio: Decimal::from_parts(80, 0, 0, false, 2), // 80%
                    market_impact_estimate: Decimal::from_parts(2, 0, 0, false, 2), // 2%
                },
            },
            account_state: AccountState {
                account_id: "test_account".to_string(),
                total_equity: Decimal::from(1000000),
                available_margin: Decimal::from(500000),
                used_margin: Decimal::from(500000),
                leverage_ratio: Decimal::from(2), // 2x leverage
                daily_pnl: Decimal::from(10000),
                max_drawdown_today: Decimal::from_parts(5, 0, 0, false, 2), // 5%
                order_count_today: 25,
                volume_traded_today: Decimal::from(750000),
            },
        }
    }

    #[tokio::test]
    async fn test_pretrade_risk_gateway_creation() {
        let config = RiskGatewayConfig::default();
        let gateway = PretradeRiskGateway::new(config);
        assert!(gateway.is_ok());
    }

    #[tokio::test]
    async fn test_pretrade_risk_check_success() {
        let config = RiskGatewayConfig::default();
        let gateway = PretradeRiskGateway::new(config).unwrap();
        
        let request = PretradeRiskRequest {
            order_id: "test_order_1".to_string(),
            strategy_id: "test_strategy".to_string(),
            account_id: "test_account".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(1), // Small order
            price: Some(Decimal::from(50000)),
            order_type: OrderType::Limit,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };
        
        let context = create_test_context();
        let result = gateway.check_pretrade_risk(request, context).await;
        
        assert!(result.is_ok());
        let response = result.unwrap();
        // Small order should generally pass
        println!("Test response: {:?}", response);
    }

    #[tokio::test]
    async fn test_pretrade_risk_check_violation() {
        let mut config = RiskGatewayConfig::default();
        config.max_position_size = Decimal::from(100); // Very small limit
        
        let gateway = PretradeRiskGateway::new(config).unwrap();
        
        let request = PretradeRiskRequest {
            order_id: "test_order_2".to_string(),
            strategy_id: "test_strategy".to_string(),
            account_id: "test_account".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(10), // Large order
            price: Some(Decimal::from(50000)),
            order_type: OrderType::Limit,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };
        
        let context = create_test_context();
        let result = gateway.check_pretrade_risk(request, context).await;
        
        assert!(result.is_ok());
        let response = result.unwrap();
        // Should be rejected due to position size limit
        assert!(!response.approved);
        assert!(!response.violations.is_empty());
    }

    #[tokio::test]
    async fn test_circuit_breaker_trigger() {
        let config = RiskGatewayConfig::default();
        let gateway = PretradeRiskGateway::new(config).unwrap();
        
        // Trigger circuit breaker
        gateway.trigger_circuit_breaker(
            CircuitBreakerType::DrawdownLimit,
            Decimal::from_parts(20, 0, 0, false, 2), // 20%
            Decimal::from_parts(15, 0, 0, false, 2), // 15% threshold
        ).await.unwrap();
        
        let request = PretradeRiskRequest {
            order_id: "test_order_3".to_string(),
            strategy_id: "test_strategy".to_string(),
            account_id: "test_account".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(1),
            price: Some(Decimal::from(50000)),
            order_type: OrderType::Limit,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };
        
        let context = create_test_context();
        let result = gateway.check_pretrade_risk(request, context).await.unwrap();
        
        // Should be rejected due to circuit breaker
        assert!(!result.approved);
        assert_eq!(result.decision, RiskDecision::Reject);
    }

    #[tokio::test]
    async fn test_blacklist_functionality() {
        let config = RiskGatewayConfig::default();
        let gateway = PretradeRiskGateway::new(config).unwrap();
        
        // Add symbol to blacklist
        gateway.add_to_blacklist("symbol", "BTCUSDT", None).await.unwrap();
        
        let request = PretradeRiskRequest {
            order_id: "test_order_4".to_string(),
            strategy_id: "test_strategy".to_string(),
            account_id: "test_account".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(1),
            price: Some(Decimal::from(50000)),
            order_type: OrderType::Limit,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };
        
        let context = create_test_context();
        let result = gateway.check_pretrade_risk(request, context).await.unwrap();
        
        // Should be rejected due to blacklist
        assert!(!result.approved);
    }

    #[tokio::test] 
    async fn test_gateway_statistics() {
        let config = RiskGatewayConfig::default();
        let gateway = PretradeRiskGateway::new(config).unwrap();
        
        // Perform some checks to generate statistics
        let request = PretradeRiskRequest {
            order_id: "test_order_5".to_string(),
            strategy_id: "test_strategy".to_string(),
            account_id: "test_account".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(1),
            price: Some(Decimal::from(50000)),
            order_type: OrderType::Limit,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };
        
        let context = create_test_context();
        
        // Run multiple checks
        for _ in 0..5 {
            gateway.check_pretrade_risk(request.clone(), context.clone()).await.unwrap();
        }
        
        let stats = gateway.get_statistics().await;
        assert!(stats.total_checks >= 5);
    }
}