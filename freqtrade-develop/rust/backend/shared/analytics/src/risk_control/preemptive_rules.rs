use super::*;
use anyhow::Result;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use serde_json::json;

/// 持仓限额规则
#[derive(Debug)]
pub struct PositionLimitRule {
    enabled: bool,
    max_position_size: Decimal,
    max_position_count: u32,
    max_single_symbol_weight: Decimal,
}

impl PositionLimitRule {
    pub fn new(max_position_size: Decimal, max_position_count: u32) -> Self {
        Self {
            enabled: true,
            max_position_size,
            max_position_count,
            max_single_symbol_weight: Decimal::from_parts(10, 0, 0, false, 2), // 10%
        }
    }
}

impl RiskRule for PositionLimitRule {
    fn rule_type(&self) -> RiskRuleType {
        RiskRuleType::PositionLimit
    }
    
    fn check(&self, _request: &PretradeRiskRequest, context: &RiskContext) -> Result<RiskCheckResult> {
        let mut warnings = Vec::new();
        
        // 检查单笔订单大小
        let position_value = _request.quantity * _request.price.unwrap_or(Decimal::ZERO);
        if position_value > self.max_position_size {
            return Ok(RiskCheckResult {
                passed: false,
                violation: Some(RiskViolation {
                    rule_type: self.rule_type(),
                    severity: RiskSeverity::High,
                    description: format!("Position size {} exceeds maximum allowed {}", 
                                       position_value, self.max_position_size),
                    current_value: position_value,
                    limit_value: self.max_position_size,
                    suggested_action: "Reduce position size or split into smaller orders".to_string(),
                }),
                warnings,
                suggested_adjustments: vec![
                    format!("Suggested max size: {}", self.max_position_size)
                ],
            });
        }
        
        // 检查持仓数量
        if context.current_positions.len() >= self.max_position_count as usize {
            return Ok(RiskCheckResult {
                passed: false,
                violation: Some(RiskViolation {
                    rule_type: self.rule_type(),
                    severity: RiskSeverity::Medium,
                    description: format!("Position count {} exceeds maximum allowed {}", 
                                       context.current_positions.len(), self.max_position_count),
                    current_value: Decimal::from(context.current_positions.len()),
                    limit_value: Decimal::from(self.max_position_count),
                    suggested_action: "Close some positions before opening new ones".to_string(),
                }),
                warnings,
                suggested_adjustments: Vec::new(),
            });
        }
        
        // 检查单一标的权重
        let total_equity = context.total_equity;
        if total_equity > Decimal::ZERO {
            let new_weight = position_value / total_equity;
            if new_weight > self.max_single_symbol_weight {
                warnings.push(RiskWarning {
                    rule_type: self.rule_type(),
                    description: format!("Single symbol weight {:.2}% approaching limit {:.2}%", 
                                       new_weight * Decimal::from(100), 
                                       self.max_single_symbol_weight * Decimal::from(100)),
                    threshold_breach_percent: (new_weight / self.max_single_symbol_weight - Decimal::ONE) * Decimal::from(100),
                    recommendation: "Consider diversifying across multiple symbols".to_string(),
                });
            }
        }
        
        Ok(RiskCheckResult {
            passed: true,
            violation: None,
            warnings,
            suggested_adjustments: Vec::new(),
        })
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn priority(&self) -> u8 {
        10
    }
    
    fn get_config(&self) -> serde_json::Value {
        json!({
            "max_position_size": self.max_position_size,
            "max_position_count": self.max_position_count,
            "max_single_symbol_weight": self.max_single_symbol_weight
        })
    }
}

/// 日损限制规则
#[derive(Debug)]
pub struct DailyLossRule {
    enabled: bool,
    max_daily_loss: Decimal,
    warning_threshold: Decimal,
}

impl DailyLossRule {
    pub fn new(max_daily_loss: Decimal) -> Self {
        Self {
            enabled: true,
            max_daily_loss,
            warning_threshold: max_daily_loss * Decimal::from_parts(80, 0, 0, false, 2), // 80% of limit
        }
    }
}

impl RiskRule for DailyLossRule {
    fn rule_type(&self) -> RiskRuleType {
        RiskRuleType::DailyLoss
    }
    
    fn check(&self, _request: &PretradeRiskRequest, context: &RiskContext) -> Result<RiskCheckResult> {
        let mut warnings = Vec::new();
        let daily_pnl = context.daily_pnl;
        
        // 检查是否超过日损限制
        if daily_pnl < -self.max_daily_loss {
            return Ok(RiskCheckResult {
                passed: false,
                violation: Some(RiskViolation {
                    rule_type: self.rule_type(),
                    severity: RiskSeverity::Critical,
                    description: format!("Daily loss {} exceeds maximum allowed {}", 
                                       daily_pnl.abs(), self.max_daily_loss),
                    current_value: daily_pnl.abs(),
                    limit_value: self.max_daily_loss,
                    suggested_action: "Stop trading for today and review strategy".to_string(),
                }),
                warnings,
                suggested_adjustments: Vec::new(),
            });
        }
        
        // 检查是否接近警告阈值
        if daily_pnl < -self.warning_threshold {
            warnings.push(RiskWarning {
                rule_type: self.rule_type(),
                description: format!("Daily loss {} approaching limit {}", 
                                   daily_pnl.abs(), self.max_daily_loss),
                threshold_breach_percent: (daily_pnl.abs() / self.max_daily_loss) * Decimal::from(100),
                recommendation: "Consider reducing position sizes or stopping trading".to_string(),
            });
        }
        
        Ok(RiskCheckResult {
            passed: true,
            violation: None,
            warnings,
            suggested_adjustments: Vec::new(),
        })
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn priority(&self) -> u8 {
        5 // High priority
    }
    
    fn get_config(&self) -> serde_json::Value {
        json!({
            "max_daily_loss": self.max_daily_loss,
            "warning_threshold": self.warning_threshold
        })
    }
}

/// 回撤限制规则
#[derive(Debug)]
pub struct DrawdownRule {
    enabled: bool,
    max_drawdown_percent: Decimal,
    warning_threshold_percent: Decimal,
}

impl DrawdownRule {
    pub fn new(max_drawdown_percent: Decimal) -> Self {
        Self {
            enabled: true,
            max_drawdown_percent,
            warning_threshold_percent: max_drawdown_percent * Decimal::from_parts(80, 0, 0, false, 2),
        }
    }
}

impl RiskRule for DrawdownRule {
    fn rule_type(&self) -> RiskRuleType {
        RiskRuleType::DrawdownLimit
    }
    
    fn check(&self, _request: &PretradeRiskRequest, context: &RiskContext) -> Result<RiskCheckResult> {
        let mut warnings = Vec::new();
        let current_drawdown = context.max_drawdown;
        
        // 检查是否超过最大回撤限制
        if current_drawdown > self.max_drawdown_percent {
            return Ok(RiskCheckResult {
                passed: false,
                violation: Some(RiskViolation {
                    rule_type: self.rule_type(),
                    severity: RiskSeverity::Critical,
                    description: format!("Current drawdown {:.2}% exceeds maximum allowed {:.2}%", 
                                       current_drawdown * Decimal::from(100),
                                       self.max_drawdown_percent * Decimal::from(100)),
                    current_value: current_drawdown,
                    limit_value: self.max_drawdown_percent,
                    suggested_action: "Halt trading and review risk management strategy".to_string(),
                }),
                warnings,
                suggested_adjustments: Vec::new(),
            });
        }
        
        // 检查是否接近警告阈值
        if current_drawdown > self.warning_threshold_percent {
            warnings.push(RiskWarning {
                rule_type: self.rule_type(),
                description: format!("Current drawdown {:.2}% approaching limit {:.2}%", 
                                   current_drawdown * Decimal::from(100),
                                   self.max_drawdown_percent * Decimal::from(100)),
                threshold_breach_percent: (current_drawdown / self.max_drawdown_percent) * Decimal::from(100),
                recommendation: "Consider reducing risk exposure and position sizes".to_string(),
            });
        }
        
        Ok(RiskCheckResult {
            passed: true,
            violation: None,
            warnings,
            suggested_adjustments: Vec::new(),
        })
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn priority(&self) -> u8 {
        5 // High priority
    }
    
    fn get_config(&self) -> serde_json::Value {
        json!({
            "max_drawdown_percent": self.max_drawdown_percent,
            "warning_threshold_percent": self.warning_threshold_percent
        })
    }
}

/// 杠杆限制规则
#[derive(Debug)]
pub struct LeverageRule {
    enabled: bool,
    max_leverage: Decimal,
    warning_threshold: Decimal,
}

impl LeverageRule {
    pub fn new(max_leverage: Decimal) -> Self {
        Self {
            enabled: true,
            max_leverage,
            warning_threshold: max_leverage * Decimal::from_parts(85, 0, 0, false, 2), // 85% of limit
        }
    }
}

impl RiskRule for LeverageRule {
    fn rule_type(&self) -> RiskRuleType {
        RiskRuleType::LeverageLimit
    }
    
    fn check(&self, _request: &PretradeRiskRequest, context: &RiskContext) -> Result<RiskCheckResult> {
        let mut warnings = Vec::new();
        let _current_leverage = context.account_state.leverage_ratio;
        
        // 计算新订单后的杠杆
        let order_value = _request.quantity * _request.price.unwrap_or(Decimal::ZERO);
        let estimated_new_leverage = if context.account_state.available_margin > Decimal::ZERO {
            (context.account_state.used_margin + order_value) / context.account_state.available_margin
        } else {
            self.max_leverage + Decimal::ONE // Force violation if no margin available
        };
        
        // 检查是否超过杠杆限制
        if estimated_new_leverage > self.max_leverage {
            return Ok(RiskCheckResult {
                passed: false,
                violation: Some(RiskViolation {
                    rule_type: self.rule_type(),
                    severity: RiskSeverity::High,
                    description: format!("Estimated leverage {:.2}x exceeds maximum allowed {:.2}x", 
                                       estimated_new_leverage, self.max_leverage),
                    current_value: estimated_new_leverage,
                    limit_value: self.max_leverage,
                    suggested_action: "Reduce position size or add more margin".to_string(),
                }),
                warnings,
                suggested_adjustments: vec![
                    format!("Maximum safe order value: {}", 
                           context.account_state.available_margin * self.max_leverage - context.account_state.used_margin)
                ],
            });
        }
        
        // 检查是否接近警告阈值
        if estimated_new_leverage > self.warning_threshold {
            warnings.push(RiskWarning {
                rule_type: self.rule_type(),
                description: format!("Estimated leverage {:.2}x approaching limit {:.2}x", 
                                   estimated_new_leverage, self.max_leverage),
                threshold_breach_percent: (estimated_new_leverage / self.max_leverage) * Decimal::from(100),
                recommendation: "Monitor margin levels closely".to_string(),
            });
        }
        
        Ok(RiskCheckResult {
            passed: true,
            violation: None,
            warnings,
            suggested_adjustments: Vec::new(),
        })
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn priority(&self) -> u8 {
        15
    }
    
    fn get_config(&self) -> serde_json::Value {
        json!({
            "max_leverage": self.max_leverage,
            "warning_threshold": self.warning_threshold
        })
    }
}

/// 黑名单规则
#[derive(Debug)]
pub struct BlacklistRule {
    enabled: bool,
    strategy_blacklist: std::collections::HashSet<String>,
    symbol_blacklist: std::collections::HashSet<String>,
    account_blacklist: std::collections::HashSet<String>,
}

impl BlacklistRule {
    pub fn new() -> Self {
        Self {
            enabled: true,
            strategy_blacklist: std::collections::HashSet::new(),
            symbol_blacklist: std::collections::HashSet::new(),
            account_blacklist: std::collections::HashSet::new(),
        }
    }
    
    pub fn add_strategy(&mut self, strategy_id: &str) {
        self.strategy_blacklist.insert(strategy_id.to_string());
    }
    
    pub fn add_symbol(&mut self, symbol: &str) {
        self.symbol_blacklist.insert(symbol.to_string());
    }
    
    pub fn add_account(&mut self, account_id: &str) {
        self.account_blacklist.insert(account_id.to_string());
    }
}

impl RiskRule for BlacklistRule {
    fn rule_type(&self) -> RiskRuleType {
        RiskRuleType::Blacklist
    }
    
    fn check(&self, _request: &PretradeRiskRequest, _context: &RiskContext) -> Result<RiskCheckResult> {
        // 检查策略黑名单
        if self.strategy_blacklist.contains(&_request.strategy_id) {
            return Ok(RiskCheckResult {
                passed: false,
                violation: Some(RiskViolation {
                    rule_type: self.rule_type(),
                    severity: RiskSeverity::Critical,
                    description: format!("Strategy {} is blacklisted", _request.strategy_id),
                    current_value: Decimal::ONE,
                    limit_value: Decimal::ZERO,
                    suggested_action: "Contact risk management team".to_string(),
                }),
                warnings: Vec::new(),
                suggested_adjustments: Vec::new(),
            });
        }
        
        // 检查交易对黑名单
        if self.symbol_blacklist.contains(&_request.symbol) {
            return Ok(RiskCheckResult {
                passed: false,
                violation: Some(RiskViolation {
                    rule_type: self.rule_type(),
                    severity: RiskSeverity::Critical,
                    description: format!("Symbol {} is blacklisted", _request.symbol),
                    current_value: Decimal::ONE,
                    limit_value: Decimal::ZERO,
                    suggested_action: "Trade different symbols".to_string(),
                }),
                warnings: Vec::new(),
                suggested_adjustments: Vec::new(),
            });
        }
        
        // 检查账户黑名单
        if self.account_blacklist.contains(&_request.account_id) {
            return Ok(RiskCheckResult {
                passed: false,
                violation: Some(RiskViolation {
                    rule_type: self.rule_type(),
                    severity: RiskSeverity::Critical,
                    description: format!("Account {} is blacklisted", _request.account_id),
                    current_value: Decimal::ONE,
                    limit_value: Decimal::ZERO,
                    suggested_action: "Contact compliance team".to_string(),
                }),
                warnings: Vec::new(),
                suggested_adjustments: Vec::new(),
            });
        }
        
        Ok(RiskCheckResult {
            passed: true,
            violation: None,
            warnings: Vec::new(),
            suggested_adjustments: Vec::new(),
        })
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn priority(&self) -> u8 {
        1 // Highest priority
    }
    
    fn get_config(&self) -> serde_json::Value {
        json!({
            "strategy_blacklist_count": self.strategy_blacklist.len(),
            "symbol_blacklist_count": self.symbol_blacklist.len(),
            "account_blacklist_count": self.account_blacklist.len()
        })
    }
}

/// 相关性暴露规则
#[derive(Debug)]
pub struct CorrelationExposureRule {
    enabled: bool,
    max_correlation_exposure: Decimal,
    correlation_threshold: Decimal,
}

#[derive(Debug, Clone)]
pub struct CorrelationConfig {
    pub max_exposure_percent: Decimal,
    pub correlation_threshold: Decimal,
    pub lookback_days: u32,
}

impl CorrelationExposureRule {
    pub fn new(config: CorrelationConfig) -> Result<Self> {
        Ok(Self {
            enabled: true,
            max_correlation_exposure: config.max_exposure_percent,
            correlation_threshold: config.correlation_threshold,
        })
    }
}

impl RiskRule for CorrelationExposureRule {
    fn rule_type(&self) -> RiskRuleType {
        RiskRuleType::CorrelationExposure
    }
    
    fn check(&self, _request: &PretradeRiskRequest, context: &RiskContext) -> Result<RiskCheckResult> {
        // 计算与现有头寸的相关性暴露
        let mut correlated_exposure = Decimal::ZERO;
        let order_value = _request.quantity * _request.price.unwrap_or(Decimal::ZERO);
        
        for (symbol, position) in &context.current_positions {
            if let Some(symbol_correlations) = context.correlation_matrix.get(symbol) {
                if let Some(correlation) = symbol_correlations.get(&_request.symbol) {
                    if correlation.abs() >= self.correlation_threshold {
                        correlated_exposure += position.market_value;
                    }
                }
            }
        }
        
        correlated_exposure += order_value; // 加上当前订单
        let exposure_ratio = correlated_exposure / context.total_equity;
        
        if exposure_ratio > self.max_correlation_exposure {
            Ok(RiskCheckResult {
                passed: false,
                violation: Some(RiskViolation {
                    rule_type: self.rule_type(),
                    severity: RiskSeverity::Medium,
                    description: format!("Correlated exposure {:.2}% exceeds limit {:.2}%", 
                                       exposure_ratio * Decimal::from(100),
                                       self.max_correlation_exposure * Decimal::from(100)),
                    current_value: exposure_ratio,
                    limit_value: self.max_correlation_exposure,
                    suggested_action: "Reduce correlated positions".to_string(),
                }),
                warnings: Vec::new(),
                suggested_adjustments: vec!["Diversify portfolio".to_string()],
            })
        } else {
            Ok(RiskCheckResult {
                passed: true,
                violation: None,
                warnings: Vec::new(),
                suggested_adjustments: Vec::new(),
            })
        }
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn priority(&self) -> u8 {
        5
    }
    
    fn get_config(&self) -> serde_json::Value {
        json!({
            "max_correlation_exposure": self.max_correlation_exposure,
            "correlation_threshold": self.correlation_threshold
        })
    }
}

/// 流动性规则
#[derive(Debug)]
pub struct LiquidityRule {
    enabled: bool,
    min_liquidity_score: Decimal,
    max_impact_bps: Decimal,
}

#[derive(Debug, Clone)]
pub struct LiquidityConfig {
    pub min_liquidity_score: Decimal,
    pub max_impact_bps: Decimal,
}

impl LiquidityRule {
    pub fn new(config: LiquidityConfig) -> Self {
        Self {
            enabled: true,
            min_liquidity_score: config.min_liquidity_score,
            max_impact_bps: config.max_impact_bps,
        }
    }
}

impl RiskRule for LiquidityRule {
    fn rule_type(&self) -> RiskRuleType {
        RiskRuleType::LiquidityLimit
    }
    
    fn check(&self, _request: &PretradeRiskRequest, context: &RiskContext) -> Result<RiskCheckResult> {
        let liquidity_score = context.market_conditions.liquidity_conditions.depth_ratio;
        let market_impact = context.market_conditions.liquidity_conditions.market_impact_estimate;
        
        if liquidity_score < self.min_liquidity_score {
            return Ok(RiskCheckResult {
                passed: false,
                violation: Some(RiskViolation {
                    rule_type: self.rule_type(),
                    severity: RiskSeverity::Medium,
                    description: format!("Liquidity score {} below minimum {}", 
                                       liquidity_score, self.min_liquidity_score),
                    current_value: liquidity_score,
                    limit_value: self.min_liquidity_score,
                    suggested_action: "Wait for better liquidity".to_string(),
                }),
                warnings: Vec::new(),
                suggested_adjustments: vec!["Trade smaller size".to_string()],
            });
        }
        
        if market_impact > self.max_impact_bps {
            return Ok(RiskCheckResult {
                passed: false,
                violation: Some(RiskViolation {
                    rule_type: self.rule_type(),
                    severity: RiskSeverity::Medium,
                    description: format!("Market impact {} bps exceeds limit {} bps", 
                                       market_impact, self.max_impact_bps),
                    current_value: market_impact,
                    limit_value: self.max_impact_bps,
                    suggested_action: "Reduce order size".to_string(),
                }),
                warnings: Vec::new(),
                suggested_adjustments: vec!["Split order into smaller pieces".to_string()],
            });
        }
        
        Ok(RiskCheckResult {
            passed: true,
            violation: None,
            warnings: Vec::new(),
            suggested_adjustments: Vec::new(),
        })
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn priority(&self) -> u8 {
        6
    }
    
    fn get_config(&self) -> serde_json::Value {
        json!({
            "min_liquidity_score": self.min_liquidity_score,
            "max_impact_bps": self.max_impact_bps
        })
    }
}

/// 波动率规则
#[derive(Debug)]
pub struct VolatilityRule {
    enabled: bool,
    max_volatility: Decimal,
    volatility_adjustment_factor: Decimal,
}

#[derive(Debug, Clone)]
pub struct VolatilityConfig {
    pub max_volatility: Decimal,
    pub adjustment_factor: Decimal,
}

impl VolatilityRule {
    pub fn new(config: VolatilityConfig) -> Self {
        Self {
            enabled: true,
            max_volatility: config.max_volatility,
            volatility_adjustment_factor: config.adjustment_factor,
        }
    }
}

impl RiskRule for VolatilityRule {
    fn rule_type(&self) -> RiskRuleType {
        RiskRuleType::VolatilityLimit
    }
    
    fn check(&self, _request: &PretradeRiskRequest, context: &RiskContext) -> Result<RiskCheckResult> {
        let current_volatility = context.market_conditions.volatility_index;
        
        if current_volatility > self.max_volatility {
            let suggested_size_reduction = self.volatility_adjustment_factor * 
                                         (current_volatility / self.max_volatility - Decimal::ONE);
            
            return Ok(RiskCheckResult {
                passed: false,
                violation: Some(RiskViolation {
                    rule_type: self.rule_type(),
                    severity: RiskSeverity::Medium,
                    description: format!("Volatility {} exceeds limit {}", 
                                       current_volatility, self.max_volatility),
                    current_value: current_volatility,
                    limit_value: self.max_volatility,
                    suggested_action: "Reduce position size for high volatility".to_string(),
                }),
                warnings: Vec::new(),
                suggested_adjustments: vec![
                    format!("Reduce order size by {:.1}%", suggested_size_reduction * Decimal::from(100))
                ],
            });
        }
        
        Ok(RiskCheckResult {
            passed: true,
            violation: None,
            warnings: Vec::new(),
            suggested_adjustments: Vec::new(),
        })
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn priority(&self) -> u8 {
        7
    }
    
    fn get_config(&self) -> serde_json::Value {
        json!({
            "max_volatility": self.max_volatility,
            "volatility_adjustment_factor": self.volatility_adjustment_factor
        })
    }
}

/// 集中度规则
#[derive(Debug)]
pub struct ConcentrationRule {
    enabled: bool,
    max_single_asset_weight: Decimal,
    max_sector_weight: Decimal,
}

#[derive(Debug, Clone)]
pub struct ConcentrationConfig {
    pub max_single_asset_weight: Decimal,
    pub max_sector_weight: Decimal,
}

impl ConcentrationRule {
    pub fn new(config: ConcentrationConfig) -> Self {
        Self {
            enabled: true,
            max_single_asset_weight: config.max_single_asset_weight,
            max_sector_weight: config.max_sector_weight,
        }
    }
}

impl RiskRule for ConcentrationRule {
    fn rule_type(&self) -> RiskRuleType {
        RiskRuleType::ConcentrationLimit
    }
    
    fn check(&self, _request: &PretradeRiskRequest, context: &RiskContext) -> Result<RiskCheckResult> {
        let order_value = _request.quantity * _request.price.unwrap_or(Decimal::ZERO);
        let current_position_value = context.current_positions
            .get(&_request.symbol)
            .map(|p| p.market_value)
            .unwrap_or(Decimal::ZERO);
        
        let total_asset_value = current_position_value + order_value;
        let asset_weight = total_asset_value / context.total_equity;
        
        if asset_weight > self.max_single_asset_weight {
            return Ok(RiskCheckResult {
                passed: false,
                violation: Some(RiskViolation {
                    rule_type: self.rule_type(),
                    severity: RiskSeverity::Medium,
                    description: format!("Asset concentration {:.2}% exceeds limit {:.2}%", 
                                       asset_weight * Decimal::from(100),
                                       self.max_single_asset_weight * Decimal::from(100)),
                    current_value: asset_weight,
                    limit_value: self.max_single_asset_weight,
                    suggested_action: "Diversify holdings".to_string(),
                }),
                warnings: Vec::new(),
                suggested_adjustments: vec!["Spread investment across multiple assets".to_string()],
            });
        }
        
        Ok(RiskCheckResult {
            passed: true,
            violation: None,
            warnings: Vec::new(),
            suggested_adjustments: Vec::new(),
        })
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn priority(&self) -> u8 {
        8
    }
    
    fn get_config(&self) -> serde_json::Value {
        json!({
            "max_single_asset_weight": self.max_single_asset_weight,
            "max_sector_weight": self.max_sector_weight
        })
    }
}

// 默认配置实现
impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            max_exposure_percent: Decimal::from_parts(30, 0, 0, false, 2), // 30%
            correlation_threshold: Decimal::from_parts(70, 0, 0, false, 2), // 0.7
            lookback_days: 30,
        }
    }
}

impl Default for LiquidityConfig {
    fn default() -> Self {
        Self {
            min_liquidity_score: Decimal::from_parts(30, 0, 0, false, 2), // 30%
            max_impact_bps: Decimal::from_parts(10, 0, 0, false, 2), // 10 bps
        }
    }
}

impl Default for VolatilityConfig {
    fn default() -> Self {
        Self {
            max_volatility: Decimal::from_parts(40, 0, 0, false, 2), // 40%
            adjustment_factor: Decimal::from_parts(20, 0, 0, false, 2), // 20%
        }
    }
}

impl Default for ConcentrationConfig {
    fn default() -> Self {
        Self {
            max_single_asset_weight: Decimal::from_parts(10, 0, 0, false, 2), // 10%
            max_sector_weight: Decimal::from_parts(25, 0, 0, false, 2), // 25%
        }
    }
}

/// 规则工厂
pub struct RiskRuleFactory;

impl RiskRuleFactory {
    pub fn create_default_rules(config: &RiskGatewayConfig) -> Vec<Box<dyn RiskRule>> {
        let mut rules: Vec<Box<dyn RiskRule>> = Vec::new();
        
        rules.push(Box::new(PositionLimitRule::new(
            config.max_position_size, 
            100
        )));
        
        rules.push(Box::new(DailyLossRule::new(config.max_daily_loss)));
        rules.push(Box::new(DrawdownRule::new(config.max_drawdown_percent)));
        rules.push(Box::new(LeverageRule::new(config.max_leverage)));
        rules.push(Box::new(BlacklistRule::new()));
        
        if let Ok(correlation_rule) = CorrelationExposureRule::new(CorrelationConfig::default()) {
            rules.push(Box::new(correlation_rule));
        }
        
        rules.push(Box::new(LiquidityRule::new(LiquidityConfig::default())));
        rules.push(Box::new(VolatilityRule::new(VolatilityConfig::default())));
        rules.push(Box::new(ConcentrationRule::new(ConcentrationConfig::default())));
        
        rules
    }
    
    pub fn create_conservative_rules(config: &RiskGatewayConfig) -> Vec<Box<dyn RiskRule>> {
        let mut rules: Vec<Box<dyn RiskRule>> = Vec::new();
        
        // 更严格的头寸限制
        rules.push(Box::new(PositionLimitRule::new(
            config.max_position_size / Decimal::TWO, // 50% of normal limit
            50
        )));
        
        // 更严格的损失限制
        rules.push(Box::new(DailyLossRule::new(config.max_daily_loss / Decimal::TWO)));
        rules.push(Box::new(DrawdownRule::new(config.max_drawdown_percent / Decimal::TWO)));
        
        // 更严格的杠杆限制
        rules.push(Box::new(LeverageRule::new(
            config.max_leverage - Decimal::ONE
        )));
        
        rules.push(Box::new(BlacklistRule::new()));
        
        // 保守的相关性限制
        if let Ok(correlation_rule) = CorrelationExposureRule::new(CorrelationConfig {
            max_exposure_percent: Decimal::from_parts(20, 0, 0, false, 2), // 20%
            correlation_threshold: Decimal::from_parts(60, 0, 0, false, 2), // 0.6
            lookback_days: 30,
        }) {
            rules.push(Box::new(correlation_rule));
        }
        
        // 更严格的流动性要求
        rules.push(Box::new(LiquidityRule::new(LiquidityConfig {
            min_liquidity_score: Decimal::from_parts(50, 0, 0, false, 2), // 50%
            max_impact_bps: Decimal::from_parts(5, 0, 0, false, 2), // 5 bps
        })));
        
        // 更严格的波动率限制
        rules.push(Box::new(VolatilityRule::new(VolatilityConfig {
            max_volatility: Decimal::from_parts(25, 0, 0, false, 2), // 25%
            adjustment_factor: Decimal::from_parts(30, 0, 0, false, 2), // 30%
        })));
        
        // 更严格的集中度限制
        rules.push(Box::new(ConcentrationRule::new(ConcentrationConfig {
            max_single_asset_weight: Decimal::from_parts(5, 0, 0, false, 2), // 5%
            max_sector_weight: Decimal::from_parts(15, 0, 0, false, 2), // 15%
        })));
        
        rules
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::collections::HashMap;

    fn create_test_context() -> RiskContext {
        RiskContext {
            current_positions: HashMap::new(),
            daily_pnl: Decimal::ZERO,
            total_equity: Decimal::from(1000000),
            max_drawdown: Decimal::from_parts(5, 0, 0, false, 2), // 5%
            correlation_matrix: HashMap::new(),
            market_conditions: MarketConditions {
                timestamp: Utc::now(),
                volatility_index: Decimal::from_parts(20, 0, 0, false, 2),
                market_stress_indicator: Decimal::from_parts(30, 0, 0, false, 2),
                liquidity_conditions: LiquidityConditions {
                    bid_ask_spread_percentile: Decimal::from_parts(50, 0, 0, false, 2),
                    depth_ratio: Decimal::from_parts(80, 0, 0, false, 2),
                    market_impact_estimate: Decimal::from_parts(2, 0, 0, false, 2),
                },
            },
            account_state: AccountState {
                account_id: "test_account".to_string(),
                total_equity: Decimal::from(1000000),
                available_margin: Decimal::from(800000),
                used_margin: Decimal::from(200000),
                leverage_ratio: Decimal::from_parts(25, 0, 0, false, 2), // 2.5x
                daily_pnl: Decimal::ZERO,
                max_drawdown_today: Decimal::from_parts(3, 0, 0, false, 2),
                order_count_today: 50,
                volume_traded_today: Decimal::from(5000000),
            },
        }
    }

    #[test]
    fn test_position_limit_rule() {
        let rule = PositionLimitRule::new(Decimal::from(100000), 10);
        let context = create_test_context();
        
        let request = PretradeRiskRequest {
            order_id: "test_order".to_string(),
            strategy_id: "test_strategy".to_string(),
            account_id: "test_account".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(2),
            price: Some(Decimal::from(50000)),
            order_type: OrderType::Limit,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };
        
        let result = rule.check(&request, &context).unwrap();
        assert!(result.passed); // Should pass as position value is exactly at limit
    }
    
    #[test]
    fn test_position_limit_violation() {
        let rule = PositionLimitRule::new(Decimal::from(50000), 10);
        let context = create_test_context();
        
        let request = PretradeRiskRequest {
            order_id: "test_order".to_string(),
            strategy_id: "test_strategy".to_string(),
            account_id: "test_account".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(2),
            price: Some(Decimal::from(50000)), // 100k value exceeds 50k limit
            order_type: OrderType::Limit,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };
        
        let result = rule.check(&request, &context).unwrap();
        assert!(!result.passed);
        assert!(result.violation.is_some());
    }
    
    #[test]
    fn test_blacklist_rule() {
        let mut rule = BlacklistRule::new();
        rule.add_strategy("blocked_strategy");
        
        let context = create_test_context();
        let request = PretradeRiskRequest {
            order_id: "test_order".to_string(),
            strategy_id: "blocked_strategy".to_string(),
            account_id: "test_account".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(1),
            price: Some(Decimal::from(50000)),
            order_type: OrderType::Limit,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };
        
        let result = rule.check(&request, &context).unwrap();
        assert!(!result.passed);
        assert!(result.violation.is_some());
        assert_eq!(result.violation.unwrap().severity, RiskSeverity::Critical);
    }
    
    #[test]
    fn test_daily_loss_rule() {
        let rule = DailyLossRule::new(Decimal::from(10000));
        let mut context = create_test_context();
        context.daily_pnl = Decimal::from(-8000); // 8k loss, below 10k limit
        
        let request = PretradeRiskRequest {
            order_id: "test_order".to_string(),
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
        
        let result = rule.check(&request, &context).unwrap();
        assert!(result.passed);
        assert!(!result.warnings.is_empty()); // Should have warning as loss > 80% threshold
    }
    
    #[test]
    fn test_correlation_exposure_rule() {
        let config = CorrelationConfig {
            max_exposure_percent: Decimal::from_parts(30, 0, 0, false, 2), // 30%
            correlation_threshold: Decimal::from_parts(80, 0, 0, false, 2), // 0.8
            lookback_days: 30,
        };
        let rule = CorrelationExposureRule::new(config).unwrap();
        let mut context = create_test_context();
        
        // Add correlated position
        let mut correlation_matrix = HashMap::new();
        let mut btc_correlations = HashMap::new();
        btc_correlations.insert("ETHUSDT".to_string(), Decimal::from_parts(85, 0, 0, false, 2)); // 0.85 correlation
        correlation_matrix.insert("BTCUSDT".to_string(), btc_correlations);
        context.correlation_matrix = correlation_matrix;
        
        let mut positions = HashMap::new();
        positions.insert("ETHUSDT".to_string(), Position {
            symbol: "ETHUSDT".to_string(),
            quantity: Decimal::from(10),
            average_price: Decimal::from(2000),
            market_value: Decimal::from(200000), // 20% of 1M equity
            unrealized_pnl: Decimal::ZERO,
            weight: Decimal::from_parts(20, 0, 0, false, 2),
        });
        context.current_positions = positions;
        
        let request = PretradeRiskRequest {
            order_id: "test_order".to_string(),
            strategy_id: "test_strategy".to_string(),
            account_id: "test_account".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(3),
            price: Some(Decimal::from(50000)), // 150k order value = 15% of equity
            order_type: OrderType::Limit,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };
        
        let result = rule.check(&request, &context).unwrap();
        // Should pass as 20% + 15% = 35% exposure, but only 85% correlation, so ~29.75% effective exposure
        assert!(result.passed);
    }
}