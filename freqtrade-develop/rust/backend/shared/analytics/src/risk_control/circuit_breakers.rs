use super::*;
use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use rust_decimal::Decimal;
use std::str::FromStr;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};

/// AG3级别的智能熔断器系统
#[derive(Debug)]
pub struct CircuitBreakerRule {
    enabled: bool,
    priority: u8,
    config: CircuitBreakerConfig,
    breakers: Arc<RwLock<HashMap<CircuitBreakerType, CircuitBreakerState>>>,
    metrics: Arc<RwLock<CircuitBreakerMetrics>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub enabled: bool,
    pub priority: u8,
    
    // 回撤熔断
    pub drawdown_threshold: Decimal,        // 最大回撤阈值
    pub drawdown_window_minutes: u32,       // 回撤统计窗口
    pub drawdown_recovery_minutes: u32,     // 回撤恢复时间
    
    // 亏损速率熔断
    pub loss_rate_threshold: Decimal,       // 最大亏损速率 %/hour
    pub loss_rate_window_minutes: u32,      // 亏损速率统计窗口
    pub loss_rate_recovery_minutes: u32,    // 亏损速率恢复时间
    
    // 波动率熔断
    pub volatility_threshold: Decimal,      // 波动率阈值
    pub volatility_window_minutes: u32,     // 波动率统计窗口
    pub volatility_recovery_minutes: u32,   // 波动率恢复时间
    
    // 流动性熔断
    pub liquidity_threshold: Decimal,       // 最小流动性阈值
    pub liquidity_window_minutes: u32,      // 流动性统计窗口
    pub liquidity_recovery_minutes: u32,    // 流动性恢复时间
    
    // 错误率熔断
    pub error_rate_threshold: Decimal,      // 最大错误率阈值
    pub error_rate_window_minutes: u32,     // 错误率统计窗口
    pub error_rate_recovery_minutes: u32,   // 错误率恢复时间
    
    // 级联保护
    pub cascade_protection_enabled: bool,   // 启用级联保护
    pub cascade_threshold: u32,             // 级联触发阈值（多少个熔断器同时触发）
    pub cascade_recovery_minutes: u32,      // 级联恢复时间
    
    // 渐进式恢复
    pub gradual_recovery_enabled: bool,     // 启用渐进式恢复
    pub recovery_step_percent: Decimal,     // 每步恢复的百分比
    pub recovery_step_interval_minutes: u32, // 每步恢复的间隔
    
    // 告警配置
    pub alert_enabled: bool,
    pub alert_cooldown_minutes: u32,        // 告警冷却时间
    pub emergency_contacts: Vec<String>,     // 紧急联系方式
}

/// 熔断器状态
#[derive(Debug, Clone)]
pub struct CircuitBreakerState {
    pub breaker_type: CircuitBreakerType,
    pub state: BreakerState,
    pub trigger_time: Option<DateTime<Utc>>,
    pub recovery_time: Option<DateTime<Utc>>,
    pub trigger_value: Decimal,
    pub threshold_value: Decimal,
    pub consecutive_triggers: u32,
    pub recovery_progress: Decimal,  // 0.0 - 1.0
    pub last_alert_time: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BreakerState {
    Closed,      // 正常状态，允许交易
    HalfOpen,    // 半开状态，部分允许交易
    Open,        // 开路状态，禁止交易
    Recovering,  // 恢复状态，渐进式恢复
}

/// 熔断器指标统计
#[derive(Debug, Default)]
pub struct CircuitBreakerMetrics {
    // 全局统计
    pub total_triggers: AtomicU64,
    pub total_recoveries: AtomicU64,
    pub cascade_events: AtomicU64,
    
    // 按类型统计
    pub trigger_counts: HashMap<CircuitBreakerType, u64>,
    pub avg_trigger_duration: HashMap<CircuitBreakerType, Duration>,
    pub last_trigger_times: HashMap<CircuitBreakerType, DateTime<Utc>>,
    
    // 实时指标
    pub current_drawdown: Decimal,
    pub current_loss_rate: Decimal,
    pub current_volatility: Decimal,
    pub current_liquidity: Decimal,
    pub current_error_rate: Decimal,
    
    // 历史数据窗口
    pub pnl_history: VecDeque<PnlEvent>,
    pub error_history: VecDeque<ErrorEvent>,
    pub market_data_history: VecDeque<MarketDataEvent>,
    
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct PnlEvent {
    pub timestamp: DateTime<Utc>,
    pub pnl: Decimal,
    pub cumulative_pnl: Decimal,
    pub account_id: String,
    pub strategy_id: String,
}

#[derive(Debug, Clone)]
pub struct ErrorEvent {
    pub timestamp: DateTime<Utc>,
    pub error_type: String,
    pub severity: ErrorSeverity,
    pub account_id: String,
    pub strategy_id: String,
}

#[derive(Debug, Clone)]
pub struct MarketDataEvent {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub volatility: Decimal,
    pub liquidity_score: Decimal,
    pub bid_ask_spread: Decimal,
    pub market_depth: Decimal,
}

#[derive(Debug, Clone)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl CircuitBreakerRule {
    pub fn new(config: CircuitBreakerConfig) -> Result<Self> {
        let mut breakers = HashMap::new();
        
        // 初始化所有熔断器状态
        for breaker_type in [
            CircuitBreakerType::DrawdownLimit,
            CircuitBreakerType::LossRate,
            CircuitBreakerType::VolatilitySpike,
            CircuitBreakerType::LiquidityDrop,
            CircuitBreakerType::ErrorRate,
        ] {
            breakers.insert(breaker_type.clone(), CircuitBreakerState {
                breaker_type: breaker_type.clone(),
                state: BreakerState::Closed,
                trigger_time: None,
                recovery_time: None,
                trigger_value: Decimal::ZERO,
                threshold_value: Self::get_threshold_for_type(&config, &breaker_type),
                consecutive_triggers: 0,
                recovery_progress: Decimal::ZERO,
                last_alert_time: None,
            });
        }
        
        Ok(Self {
            enabled: config.enabled,
            priority: config.priority,
            config,
            breakers: Arc::new(RwLock::new(breakers)),
            metrics: Arc::new(RwLock::new(CircuitBreakerMetrics::default())),
        })
    }
    
    /// 获取指定类型的阈值
    fn get_threshold_for_type(config: &CircuitBreakerConfig, breaker_type: &CircuitBreakerType) -> Decimal {
        match breaker_type {
            CircuitBreakerType::DrawdownLimit => config.drawdown_threshold,
            CircuitBreakerType::LossRate => config.loss_rate_threshold,
            CircuitBreakerType::VolatilitySpike => config.volatility_threshold,
            CircuitBreakerType::LiquidityDrop => config.liquidity_threshold,
            CircuitBreakerType::ErrorRate => config.error_rate_threshold,
        }
    }
    
    /// 检查所有熔断器状态
    async fn check_circuit_breakers(&self, request: &PretradeRiskRequest, context: &RiskContext) -> Result<RiskCheckResult> {
        let now = Utc::now();
        
        // 更新指标
        self.update_metrics(request, context, now).await?;
        
        // 检查各个熔断器
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        
        let breakers = self.breakers.read().await;
        let metrics = self.metrics.read().await;
        
        // 检查回撤熔断
        if let Some(violation) = self.check_drawdown_breaker(&metrics, now).await? {
            violations.push(violation);
        }
        
        // 检查亏损速率熔断
        if let Some(violation) = self.check_loss_rate_breaker(&metrics, now).await? {
            violations.push(violation);
        }
        
        // 检查波动率熔断
        if let Some(violation) = self.check_volatility_breaker(&metrics, now).await? {
            violations.push(violation);
        }
        
        // 检查流动性熔断
        if let Some(violation) = self.check_liquidity_breaker(&metrics, now).await? {
            violations.push(violation);
        }
        
        // 检查错误率熔断
        if let Some(violation) = self.check_error_rate_breaker(&metrics, now).await? {
            violations.push(violation);
        }
        
        // 检查级联保护
        if self.config.cascade_protection_enabled {
            let active_breakers = breakers.values()
                .filter(|state| state.state == BreakerState::Open)
                .count() as u32;
                
            if active_breakers >= self.config.cascade_threshold {
                violations.push(RiskViolation {
                    rule_type: RiskRuleType::CircuitBreaker,
                    severity: RiskSeverity::Critical,
                    description: format!("Cascade protection triggered: {} breakers active", active_breakers),
                    current_value: Decimal::from(active_breakers),
                    limit_value: Decimal::from(self.config.cascade_threshold),
                    suggested_action: "System-wide trading halt".to_string(),
                });
            }
        }
        
        // 生成预警
        for (breaker_type, state) in breakers.iter() {
            if state.state == BreakerState::HalfOpen {
                warnings.push(RiskWarning {
                    rule_type: RiskRuleType::CircuitBreaker,
                    description: format!("Circuit breaker {:?} in half-open state", breaker_type),
                    threshold_breach_percent: state.recovery_progress * Decimal::from(100),
                    recommendation: "Monitor system carefully during recovery".to_string(),
                });
            }
        }
        
        drop(breakers);
        drop(metrics);
        
        // 更新熔断器状态
        self.update_breaker_states(now).await?;
        
        if violations.is_empty() {
            Ok(RiskCheckResult {
                passed: true,
                violation: None,
                warnings,
                suggested_adjustments: vec![],
            })
        } else {
            Ok(RiskCheckResult {
                passed: false,
                violation: violations.into_iter().next(), // 返回第一个违规
                warnings,
                suggested_adjustments: vec!["Review risk parameters".to_string()],
            })
        }
    }
    
    /// 检查回撤熔断器
    async fn check_drawdown_breaker(&self, metrics: &CircuitBreakerMetrics, now: DateTime<Utc>) -> Result<Option<RiskViolation>> {
        if metrics.current_drawdown >= self.config.drawdown_threshold {
            Ok(Some(RiskViolation {
                rule_type: RiskRuleType::CircuitBreaker,
                severity: RiskSeverity::Critical,
                description: format!("Drawdown limit breached: {:.2}% >= {:.2}%",
                    metrics.current_drawdown * Decimal::from(100),
                    self.config.drawdown_threshold * Decimal::from(100)),
                current_value: metrics.current_drawdown,
                limit_value: self.config.drawdown_threshold,
                suggested_action: "Halt trading and review positions".to_string(),
            }))
        } else {
            Ok(None)
        }
    }
    
    /// 检查亏损速率熔断器
    async fn check_loss_rate_breaker(&self, metrics: &CircuitBreakerMetrics, now: DateTime<Utc>) -> Result<Option<RiskViolation>> {
        if metrics.current_loss_rate >= self.config.loss_rate_threshold {
            Ok(Some(RiskViolation {
                rule_type: RiskRuleType::CircuitBreaker,
                severity: RiskSeverity::High,
                description: format!("Loss rate limit breached: {:.2}%/hour >= {:.2}%/hour",
                    metrics.current_loss_rate * Decimal::from(100),
                    self.config.loss_rate_threshold * Decimal::from(100)),
                current_value: metrics.current_loss_rate,
                limit_value: self.config.loss_rate_threshold,
                suggested_action: "Reduce position sizes and review strategies".to_string(),
            }))
        } else {
            Ok(None)
        }
    }
    
    /// 检查波动率熔断器
    async fn check_volatility_breaker(&self, metrics: &CircuitBreakerMetrics, now: DateTime<Utc>) -> Result<Option<RiskViolation>> {
        if metrics.current_volatility >= self.config.volatility_threshold {
            Ok(Some(RiskViolation {
                rule_type: RiskRuleType::CircuitBreaker,
                severity: RiskSeverity::Medium,
                description: format!("Volatility spike detected: {:.2}% >= {:.2}%",
                    metrics.current_volatility * Decimal::from(100),
                    self.config.volatility_threshold * Decimal::from(100)),
                current_value: metrics.current_volatility,
                limit_value: self.config.volatility_threshold,
                suggested_action: "Adjust position sizing for high volatility".to_string(),
            }))
        } else {
            Ok(None)
        }
    }
    
    /// 检查流动性熔断器
    async fn check_liquidity_breaker(&self, metrics: &CircuitBreakerMetrics, now: DateTime<Utc>) -> Result<Option<RiskViolation>> {
        if metrics.current_liquidity <= self.config.liquidity_threshold {
            Ok(Some(RiskViolation {
                rule_type: RiskRuleType::CircuitBreaker,
                severity: RiskSeverity::High,
                description: format!("Liquidity drop detected: {:.2} <= {:.2}",
                    metrics.current_liquidity,
                    self.config.liquidity_threshold),
                current_value: metrics.current_liquidity,
                limit_value: self.config.liquidity_threshold,
                suggested_action: "Pause trading in illiquid markets".to_string(),
            }))
        } else {
            Ok(None)
        }
    }
    
    /// 检查错误率熔断器
    async fn check_error_rate_breaker(&self, metrics: &CircuitBreakerMetrics, now: DateTime<Utc>) -> Result<Option<RiskViolation>> {
        if metrics.current_error_rate >= self.config.error_rate_threshold {
            Ok(Some(RiskViolation {
                rule_type: RiskRuleType::CircuitBreaker,
                severity: RiskSeverity::High,
                description: format!("Error rate too high: {:.2}% >= {:.2}%",
                    metrics.current_error_rate * Decimal::from(100),
                    self.config.error_rate_threshold * Decimal::from(100)),
                current_value: metrics.current_error_rate,
                limit_value: self.config.error_rate_threshold,
                suggested_action: "Check system health and connectivity".to_string(),
            }))
        } else {
            Ok(None)
        }
    }
    
    /// 更新熔断器指标
    async fn update_metrics(&self, request: &PretradeRiskRequest, context: &RiskContext, now: DateTime<Utc>) -> Result<()> {
        let mut metrics = self.metrics.write().await;
        
        // 更新回撤指标
        metrics.current_drawdown = context.max_drawdown;
        
        // 计算亏损速率
        let window_start = now - Duration::minutes(self.config.loss_rate_window_minutes as i64);
        let recent_pnl: Decimal = metrics.pnl_history
            .iter()
            .filter(|event| event.timestamp >= window_start)
            .map(|event| event.pnl)
            .sum();
        
        if recent_pnl < Decimal::ZERO {
            let hours = self.config.loss_rate_window_minutes as f64 / 60.0;
            metrics.current_loss_rate = (-recent_pnl / context.total_equity) / Decimal::from_str(&format!("{:.6}", hours)).unwrap_or(Decimal::ONE);
        } else {
            metrics.current_loss_rate = Decimal::ZERO;
        }
        
        // 更新波动率指标
        if let Some(latest_market_data) = metrics.market_data_history.back() {
            metrics.current_volatility = latest_market_data.volatility;
        }
        
        // 更新流动性指标
        if let Some(latest_market_data) = metrics.market_data_history.back() {
            metrics.current_liquidity = latest_market_data.liquidity_score;
        }
        
        // 计算错误率
        let error_window_start = now - Duration::minutes(self.config.error_rate_window_minutes as i64);
        let total_errors = metrics.error_history
            .iter()
            .filter(|event| event.timestamp >= error_window_start)
            .count() as u64;
        
        let total_requests = 100; // 简化：假设每分钟100个请求
        if total_requests > 0 {
            metrics.current_error_rate = Decimal::from(total_errors) / Decimal::from(total_requests);
        }
        
        metrics.last_updated = now;
        
        Ok(())
    }
    
    /// 更新熔断器状态
    async fn update_breaker_states(&self, now: DateTime<Utc>) -> Result<()> {
        let mut breakers = self.breakers.write().await;
        
        for (breaker_type, state) in breakers.iter_mut() {
            match state.state {
                BreakerState::Open => {
                    // 检查是否到恢复时间
                    if let Some(recovery_time) = state.recovery_time {
                        if now >= recovery_time {
                            if self.config.gradual_recovery_enabled {
                                state.state = BreakerState::Recovering;
                                state.recovery_progress = Decimal::ZERO;
                                info!("Circuit breaker {:?} entering recovery phase", breaker_type);
                            } else {
                                state.state = BreakerState::HalfOpen;
                                info!("Circuit breaker {:?} entering half-open state", breaker_type);
                            }
                        }
                    }
                }
                BreakerState::Recovering => {
                    // 渐进式恢复
                    let step_interval = Duration::minutes(self.config.recovery_step_interval_minutes as i64);
                    if let Some(trigger_time) = state.trigger_time {
                        let elapsed_steps = now.signed_duration_since(trigger_time).num_minutes() / 
                                          self.config.recovery_step_interval_minutes as i64;
                        
                        state.recovery_progress = (Decimal::from(elapsed_steps as u64) * 
                                                 self.config.recovery_step_percent).min(Decimal::ONE);
                        
                        if state.recovery_progress >= Decimal::ONE {
                            state.state = BreakerState::Closed;
                            state.trigger_time = None;
                            state.recovery_time = None;
                            state.recovery_progress = Decimal::ZERO;
                            info!("Circuit breaker {:?} fully recovered", breaker_type);
                        }
                    }
                }
                BreakerState::HalfOpen => {
                    // 半开状态处理逻辑可以在这里实现
                    // 例如，如果一段时间内没有新的触发，则转为关闭状态
                }
                BreakerState::Closed => {
                    // 正常状态，无需特殊处理
                }
            }
        }
        
        Ok(())
    }
    
    /// 手动触发熔断器
    pub async fn trigger_breaker(&self, breaker_type: CircuitBreakerType, trigger_value: Decimal) -> Result<()> {
        let mut breakers = self.breakers.write().await;
        let mut metrics = self.metrics.write().await;
        
        if let Some(state) = breakers.get_mut(&breaker_type) {
            let now = Utc::now();
            
            state.state = BreakerState::Open;
            state.trigger_time = Some(now);
            state.trigger_value = trigger_value;
            state.consecutive_triggers += 1;
            
            // 设置恢复时间
            let recovery_minutes = match breaker_type {
                CircuitBreakerType::DrawdownLimit => self.config.drawdown_recovery_minutes,
                CircuitBreakerType::LossRate => self.config.loss_rate_recovery_minutes,
                CircuitBreakerType::VolatilitySpike => self.config.volatility_recovery_minutes,
                CircuitBreakerType::LiquidityDrop => self.config.liquidity_recovery_minutes,
                CircuitBreakerType::ErrorRate => self.config.error_rate_recovery_minutes,
            };
            
            state.recovery_time = Some(now + Duration::minutes(recovery_minutes as i64));
            
            // 更新指标
            metrics.total_triggers.fetch_add(1, Ordering::Relaxed);
            metrics.trigger_counts.entry(breaker_type.clone())
                .and_modify(|count| *count += 1)
                .or_insert(1);
            metrics.last_trigger_times.insert(breaker_type.clone(), now);
            
            error!("Circuit breaker {:?} triggered with value {} (threshold: {})", 
                   breaker_type, trigger_value, state.threshold_value);
        }
        
        Ok(())
    }
    
    /// 获取熔断器统计信息
    pub async fn get_breaker_statistics(&self) -> HashMap<CircuitBreakerType, CircuitBreakerState> {
        let breakers = self.breakers.read().await;
        breakers.clone()
    }
}

impl RiskRule for CircuitBreakerRule {
    fn rule_type(&self) -> RiskRuleType {
        RiskRuleType::CircuitBreaker
    }
    
    fn check(&self, request: &PretradeRiskRequest, context: &RiskContext) -> Result<RiskCheckResult> {
        // 使用异步块处理检查
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            self.check_circuit_breakers(request, context).await
        })
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn priority(&self) -> u8 {
        self.priority
    }
    
    fn get_config(&self) -> serde_json::Value {
        serde_json::to_value(&self.config).unwrap_or(serde_json::Value::Null)
    }
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            priority: 1, // 最高优先级
            drawdown_threshold: Decimal::from_parts(15, 0, 0, false, 2), // 15%
            drawdown_window_minutes: 60,
            drawdown_recovery_minutes: 30,
            loss_rate_threshold: Decimal::from_parts(5, 0, 0, false, 2), // 5%/hour
            loss_rate_window_minutes: 60,
            loss_rate_recovery_minutes: 15,
            volatility_threshold: Decimal::from_parts(50, 0, 0, false, 2), // 50%
            volatility_window_minutes: 30,
            volatility_recovery_minutes: 10,
            liquidity_threshold: Decimal::from_parts(30, 0, 0, false, 2), // 30%
            liquidity_window_minutes: 15,
            liquidity_recovery_minutes: 5,
            error_rate_threshold: Decimal::from_parts(10, 0, 0, false, 2), // 10%
            error_rate_window_minutes: 10,
            error_rate_recovery_minutes: 5,
            cascade_protection_enabled: true,
            cascade_threshold: 3,
            cascade_recovery_minutes: 60,
            gradual_recovery_enabled: true,
            recovery_step_percent: Decimal::from_parts(25, 0, 0, false, 2), // 25% per step
            recovery_step_interval_minutes: 5,
            alert_enabled: true,
            alert_cooldown_minutes: 10,
            emergency_contacts: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_circuit_breaker_drawdown() {
        let config = CircuitBreakerConfig {
            drawdown_threshold: Decimal::from_parts(10, 0, 0, false, 2), // 10%
            ..Default::default()
        };
        
        let rule = CircuitBreakerRule::new(config).unwrap();
        
        let request = PretradeRiskRequest {
            order_id: "test_order".to_string(),
            strategy_id: "test_strategy".to_string(),
            account_id: "test_account".to_string(),
            symbol: "BTCUSD".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(100),
            price: Some(Decimal::from(50000)),
            order_type: OrderType::Limit,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };
        
        let mut context = RiskContext {
            current_positions: HashMap::new(),
            daily_pnl: Decimal::from(-5000),
            total_equity: Decimal::from(100000),
            max_drawdown: Decimal::from_parts(15, 0, 0, false, 2), // 15% drawdown
            correlation_matrix: HashMap::new(),
            market_conditions: MarketConditions {
                timestamp: Utc::now(),
                volatility_index: Decimal::from_parts(25, 0, 0, false, 2),
                market_stress_indicator: Decimal::from_parts(30, 0, 0, false, 2),
                liquidity_conditions: LiquidityConditions {
                    bid_ask_spread_percentile: Decimal::from_parts(5, 0, 0, false, 2),
                    depth_ratio: Decimal::from_parts(80, 0, 0, false, 2),
                    market_impact_estimate: Decimal::from_parts(2, 0, 0, false, 2),
                },
            },
            account_state: AccountState {
                account_id: "test_account".to_string(),
                total_equity: Decimal::from(100000),
                available_margin: Decimal::from(50000),
                used_margin: Decimal::from(50000),
                leverage_ratio: Decimal::from(2),
                daily_pnl: Decimal::from(-5000),
                max_drawdown_today: Decimal::from_parts(15, 0, 0, false, 2),
                order_count_today: 50,
                volume_traded_today: Decimal::from(1000000),
            },
        };
        
        let result = rule.check(&request, &context).unwrap();
        
        // 应该触发回撤熔断
        assert!(!result.passed);
        if let Some(violation) = result.violation {
            assert_eq!(violation.rule_type, RiskRuleType::CircuitBreaker);
            assert_eq!(violation.severity, RiskSeverity::Critical);
        }
    }
    
    #[tokio::test]
    async fn test_circuit_breaker_recovery() {
        let config = CircuitBreakerConfig {
            drawdown_threshold: Decimal::from_parts(10, 0, 0, false, 2),
            drawdown_recovery_minutes: 1, // 1分钟恢复
            gradual_recovery_enabled: false,
            ..Default::default()
        };
        
        let rule = CircuitBreakerRule::new(config).unwrap();
        
        // 手动触发熔断器
        rule.trigger_breaker(CircuitBreakerType::DrawdownLimit, Decimal::from_parts(15, 0, 0, false, 2)).await.unwrap();
        
        // 等待恢复时间
        tokio::time::sleep(tokio::time::Duration::from_secs(65)).await;
        
        // 更新状态
        rule.update_breaker_states(Utc::now()).await.unwrap();
        
        let stats = rule.get_breaker_statistics().await;
        let drawdown_state = stats.get(&CircuitBreakerType::DrawdownLimit).unwrap();
        
        // 应该已经进入半开状态
        assert_eq!(drawdown_state.state, BreakerState::HalfOpen);
    }
    
    #[test]
    fn test_circuit_breaker_config_default() {
        let config = CircuitBreakerConfig::default();
        assert_eq!(config.drawdown_threshold, Decimal::from_parts(15, 0, 0, false, 2));
        assert_eq!(config.cascade_threshold, 3);
        assert!(config.gradual_recovery_enabled);
    }
}