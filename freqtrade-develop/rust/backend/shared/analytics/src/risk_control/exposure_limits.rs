use super::*;
use anyhow::Result;
use chrono::{DateTime, Utc, Duration, Timelike, Datelike};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};

/// AG3级别的暴露限制规则系统
#[derive(Debug)]
pub struct ExposureLimitsRule {
    enabled: bool,
    priority: u8,
    config: ExposureLimitsConfig,
    exposure_tracker: Arc<RwLock<ExposureTracker>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExposureLimitsConfig {
    pub enabled: bool,
    pub priority: u8,
    
    // 总体暴露限制
    pub max_total_exposure: Decimal,          // 最大总暴露
    pub max_net_exposure: Decimal,            // 最大净暴露
    pub max_gross_exposure: Decimal,          // 最大总暴露（多头+空头）
    
    // 单一头寸限制
    pub max_single_position_size: Decimal,    // 单一头寸最大规模
    pub max_single_position_weight: Decimal,  // 单一头寸最大权重（占总权益比例）
    pub max_position_concentration: Decimal,  // 头寸集中度限制
    
    // 行业/板块限制
    pub enable_sector_limits: bool,
    pub max_sector_exposure: Decimal,         // 单一行业最大暴露
    pub max_sector_concentration: Decimal,    // 行业集中度限制
    pub sector_limits: HashMap<String, Decimal>, // 特定行业限制
    
    // 地域限制
    pub enable_geographic_limits: bool,
    pub max_country_exposure: Decimal,        // 单一国家最大暴露
    pub max_region_exposure: Decimal,         // 单一地区最大暴露
    pub country_limits: HashMap<String, Decimal>, // 特定国家限制
    
    // 货币限制
    pub enable_currency_limits: bool,
    pub max_currency_exposure: Decimal,       // 单一货币最大暴露
    pub currency_limits: HashMap<String, Decimal>, // 特定货币限制
    
    // 杠杆限制
    pub max_leverage_ratio: Decimal,          // 最大杠杆比率
    pub leverage_by_asset_class: HashMap<String, Decimal>, // 按资产类别的杠杆限制
    
    // 相关性限制
    pub enable_correlation_limits: bool,
    pub max_correlated_exposure: Decimal,     // 高相关资产最大暴露
    pub correlation_threshold: Decimal,       // 相关性阈值
    pub correlation_window_days: u32,         // 相关性计算窗口
    
    // 流动性限制
    pub enable_liquidity_limits: bool,
    pub min_liquidity_score: Decimal,         // 最小流动性评分
    pub max_illiquid_exposure: Decimal,       // 非流动性资产最大暴露
    pub liquidity_adjustment_factor: Decimal, // 流动性调整因子
    
    // 时间限制
    pub enable_time_limits: bool,
    pub intraday_position_limits: HashMap<String, Decimal>, // 日内头寸限制
    pub overnight_position_limits: HashMap<String, Decimal>, // 隔夜头寸限制
    pub weekend_position_limits: HashMap<String, Decimal>,   // 周末头寸限制
    
    // 预警阈值
    pub warning_threshold_percent: Decimal,   // 达到限制的百分比时预警
    pub critical_threshold_percent: Decimal,  // 关键阈值
}

/// 暴露跟踪器
#[derive(Debug)]
pub struct ExposureTracker {
    // 实时暴露统计
    pub current_exposures: HashMap<String, AssetExposure>,
    pub sector_exposures: HashMap<String, SectorExposure>,
    pub country_exposures: HashMap<String, CountryExposure>,
    pub currency_exposures: HashMap<String, CurrencyExposure>,
    
    // 总体暴露指标
    pub total_long_exposure: Decimal,
    pub total_short_exposure: Decimal,
    pub net_exposure: Decimal,
    pub gross_exposure: Decimal,
    pub current_leverage: Decimal,
    
    // 相关性矩阵
    pub correlation_matrix: HashMap<String, HashMap<String, Decimal>>,
    pub correlation_last_updated: DateTime<Utc>,
    
    // 流动性评分
    pub liquidity_scores: HashMap<String, LiquidityScore>,
    
    // 历史记录
    pub exposure_history: std::collections::VecDeque<ExposureSnapshot>,
    
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct AssetExposure {
    pub symbol: String,
    pub position_size: Decimal,
    pub market_value: Decimal,
    pub weight: Decimal,           // 占总权益比重
    pub unrealized_pnl: Decimal,
    pub beta: Option<Decimal>,     // 相对基准的贝塔值
    pub sector: Option<String>,
    pub country: Option<String>,
    pub currency: String,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct SectorExposure {
    pub sector_name: String,
    pub total_exposure: Decimal,
    pub net_exposure: Decimal,
    pub position_count: u32,
    pub weight: Decimal,
    pub top_holdings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CountryExposure {
    pub country_code: String,
    pub total_exposure: Decimal,
    pub currency_breakdown: HashMap<String, Decimal>,
    pub sector_breakdown: HashMap<String, Decimal>,
    pub weight: Decimal,
}

#[derive(Debug, Clone)]
pub struct CurrencyExposure {
    pub currency_code: String,
    pub total_exposure: Decimal,
    pub hedged_exposure: Decimal,
    pub unhedged_exposure: Decimal,
    pub fx_risk: Decimal,
}

#[derive(Debug, Clone)]
pub struct LiquidityScore {
    pub symbol: String,
    pub score: Decimal,            // 0-100 流动性评分
    pub daily_volume: Decimal,
    pub bid_ask_spread: Decimal,
    pub market_depth: Decimal,
    pub impact_cost: Decimal,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct ExposureSnapshot {
    pub timestamp: DateTime<Utc>,
    pub total_exposure: Decimal,
    pub net_exposure: Decimal,
    pub leverage: Decimal,
    pub risk_metrics: HashMap<String, Decimal>,
}

impl ExposureLimitsRule {
    pub fn new(config: ExposureLimitsConfig) -> Result<Self> {
        Ok(Self {
            enabled: config.enabled,
            priority: config.priority,
            config,
            exposure_tracker: Arc::new(RwLock::new(ExposureTracker::new()?)),
        })
    }
    
    /// 检查所有暴露限制
    async fn check_exposure_limits(&self, request: &PretradeRiskRequest, context: &RiskContext) -> Result<RiskCheckResult> {
        // 更新暴露统计
        self.update_exposure_tracking(request, context).await?;
        
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        
        // 1. 检查总体暴露限制
        if let Some(violation) = self.check_total_exposure_limits().await? {
            violations.push(violation);
        }
        
        // 2. 检查单一头寸限制
        if let Some(violation) = self.check_single_position_limits(request, context).await? {
            violations.push(violation);
        }
        
        // 3. 检查行业暴露限制
        if self.config.enable_sector_limits {
            if let Some(violation) = self.check_sector_limits(request).await? {
                violations.push(violation);
            }
        }
        
        // 4. 检查地域暴露限制
        if self.config.enable_geographic_limits {
            if let Some(violation) = self.check_geographic_limits(request).await? {
                violations.push(violation);
            }
        }
        
        // 5. 检查货币暴露限制
        if self.config.enable_currency_limits {
            if let Some(violation) = self.check_currency_limits(request).await? {
                violations.push(violation);
            }
        }
        
        // 6. 检查杠杆限制
        if let Some(violation) = self.check_leverage_limits(context).await? {
            violations.push(violation);
        }
        
        // 7. 检查相关性限制
        if self.config.enable_correlation_limits {
            if let Some(violation) = self.check_correlation_limits(request).await? {
                violations.push(violation);
            }
        }
        
        // 8. 检查流动性限制
        if self.config.enable_liquidity_limits {
            if let Some(violation) = self.check_liquidity_limits(request).await? {
                violations.push(violation);
            }
        }
        
        // 9. 检查时间限制
        if self.config.enable_time_limits {
            if let Some(violation) = self.check_time_based_limits(request).await? {
                violations.push(violation);
            }
        }
        
        // 生成预警
        warnings.extend(self.generate_warnings().await?);
        
        if violations.is_empty() {
            Ok(RiskCheckResult {
                passed: true,
                violation: None,
                warnings,
                suggested_adjustments: vec![],
            })
        } else {
            let suggested_adjustments = self.generate_suggested_adjustments(&violations).await?;
            Ok(RiskCheckResult {
                passed: false,
                violation: violations.into_iter().next(),
                warnings,
                suggested_adjustments,
            })
        }
    }
    
    /// 检查总体暴露限制
    async fn check_total_exposure_limits(&self) -> Result<Option<RiskViolation>> {
        let tracker = self.exposure_tracker.read().await;
        
        // 检查总暴露限制
        if tracker.gross_exposure > self.config.max_gross_exposure {
            return Ok(Some(RiskViolation {
                rule_type: RiskRuleType::ConcentrationLimit,
                severity: RiskSeverity::High,
                description: format!("Gross exposure {} exceeds limit {}", 
                    tracker.gross_exposure, self.config.max_gross_exposure),
                current_value: tracker.gross_exposure,
                limit_value: self.config.max_gross_exposure,
                suggested_action: "Reduce overall position sizes".to_string(),
            }));
        }
        
        // 检查净暴露限制
        if tracker.net_exposure.abs() > self.config.max_net_exposure {
            return Ok(Some(RiskViolation {
                rule_type: RiskRuleType::ConcentrationLimit,
                severity: RiskSeverity::Medium,
                description: format!("Net exposure {} exceeds limit {}", 
                    tracker.net_exposure, self.config.max_net_exposure),
                current_value: tracker.net_exposure.abs(),
                limit_value: self.config.max_net_exposure,
                suggested_action: "Balance long/short positions".to_string(),
            }));
        }
        
        Ok(None)
    }
    
    /// 检查单一头寸限制
    async fn check_single_position_limits(&self, request: &PretradeRiskRequest, context: &RiskContext) -> Result<Option<RiskViolation>> {
        // 计算新订单的头寸价值
        let order_value = request.quantity * request.price.unwrap_or(Decimal::ZERO);
        
        // 获取当前头寸
        let current_position = context.current_positions
            .get(&request.symbol)
            .map(|pos| pos.market_value)
            .unwrap_or(Decimal::ZERO);
        
        let projected_position = match request.side {
            OrderSide::Buy => current_position + order_value,
            OrderSide::Sell => current_position - order_value,
        };
        
        // 检查单一头寸大小限制
        if projected_position.abs() > self.config.max_single_position_size {
            return Ok(Some(RiskViolation {
                rule_type: RiskRuleType::PositionLimit,
                severity: RiskSeverity::High,
                description: format!("Position size {} for {} exceeds limit {}", 
                    projected_position, request.symbol, self.config.max_single_position_size),
                current_value: projected_position.abs(),
                limit_value: self.config.max_single_position_size,
                suggested_action: "Reduce order size".to_string(),
            }));
        }
        
        // 检查头寸权重限制
        let position_weight = projected_position / context.total_equity;
        if position_weight.abs() > self.config.max_single_position_weight {
            return Ok(Some(RiskViolation {
                rule_type: RiskRuleType::ConcentrationLimit,
                severity: RiskSeverity::Medium,
                description: format!("Position weight {:.2}% for {} exceeds limit {:.2}%", 
                    position_weight * Decimal::from(100), request.symbol, 
                    self.config.max_single_position_weight * Decimal::from(100)),
                current_value: position_weight.abs(),
                limit_value: self.config.max_single_position_weight,
                suggested_action: "Reduce position relative to portfolio size".to_string(),
            }));
        }
        
        Ok(None)
    }
    
    /// 检查行业暴露限制
    async fn check_sector_limits(&self, request: &PretradeRiskRequest) -> Result<Option<RiskViolation>> {
        let tracker = self.exposure_tracker.read().await;
        
        // 假设我们能从某处获取symbol的行业信息
        let symbol_sector = self.get_symbol_sector(&request.symbol).await?;
        
        if let Some(sector) = symbol_sector {
            if let Some(sector_exposure) = tracker.sector_exposures.get(&sector) {
                if sector_exposure.weight > self.config.max_sector_exposure {
                    return Ok(Some(RiskViolation {
                        rule_type: RiskRuleType::ConcentrationLimit,
                        severity: RiskSeverity::Medium,
                        description: format!("Sector exposure {} for {} exceeds limit {}", 
                            sector_exposure.weight, sector, self.config.max_sector_exposure),
                        current_value: sector_exposure.weight,
                        limit_value: self.config.max_sector_exposure,
                        suggested_action: format!("Reduce exposure to {} sector", sector),
                    }));
                }
                
                // 检查特定行业限制
                if let Some(&sector_limit) = self.config.sector_limits.get(&sector) {
                    if sector_exposure.weight > sector_limit {
                        return Ok(Some(RiskViolation {
                            rule_type: RiskRuleType::ConcentrationLimit,
                            severity: RiskSeverity::High,
                            description: format!("Specific sector limit exceeded for {}", sector),
                            current_value: sector_exposure.weight,
                            limit_value: sector_limit,
                            suggested_action: format!("Comply with {} sector specific limit", sector),
                        }));
                    }
                }
            }
        }
        
        Ok(None)
    }
    
    /// 检查地域暴露限制
    async fn check_geographic_limits(&self, request: &PretradeRiskRequest) -> Result<Option<RiskViolation>> {
        let tracker = self.exposure_tracker.read().await;
        
        let symbol_country = self.get_symbol_country(&request.symbol).await?;
        
        if let Some(country) = symbol_country {
            if let Some(country_exposure) = tracker.country_exposures.get(&country) {
                if country_exposure.weight > self.config.max_country_exposure {
                    return Ok(Some(RiskViolation {
                        rule_type: RiskRuleType::ConcentrationLimit,
                        severity: RiskSeverity::Medium,
                        description: format!("Country exposure {} for {} exceeds limit {}", 
                            country_exposure.weight, country, self.config.max_country_exposure),
                        current_value: country_exposure.weight,
                        limit_value: self.config.max_country_exposure,
                        suggested_action: format!("Reduce exposure to {}", country),
                    }));
                }
            }
        }
        
        Ok(None)
    }
    
    /// 检查货币暴露限制
    async fn check_currency_limits(&self, request: &PretradeRiskRequest) -> Result<Option<RiskViolation>> {
        let tracker = self.exposure_tracker.read().await;
        
        let symbol_currency = self.get_symbol_currency(&request.symbol).await?;
        
        if let Some(currency_exposure) = tracker.currency_exposures.get(&symbol_currency) {
            let total_weight = currency_exposure.total_exposure / 
                tracker.current_exposures.values().map(|e| e.market_value).sum::<Decimal>();
                
            if total_weight > self.config.max_currency_exposure {
                return Ok(Some(RiskViolation {
                    rule_type: RiskRuleType::ConcentrationLimit,
                    severity: RiskSeverity::Medium,
                    description: format!("Currency exposure {} for {} exceeds limit {}", 
                        total_weight, symbol_currency, self.config.max_currency_exposure),
                    current_value: total_weight,
                    limit_value: self.config.max_currency_exposure,
                    suggested_action: format!("Hedge or reduce {} exposure", symbol_currency),
                }));
            }
        }
        
        Ok(None)
    }
    
    /// 检查杠杆限制
    async fn check_leverage_limits(&self, context: &RiskContext) -> Result<Option<RiskViolation>> {
        if context.account_state.leverage_ratio > self.config.max_leverage_ratio {
            return Ok(Some(RiskViolation {
                rule_type: RiskRuleType::LeverageLimit,
                severity: RiskSeverity::High,
                description: format!("Leverage ratio {} exceeds limit {}", 
                    context.account_state.leverage_ratio, self.config.max_leverage_ratio),
                current_value: context.account_state.leverage_ratio,
                limit_value: self.config.max_leverage_ratio,
                suggested_action: "Reduce leverage by closing positions".to_string(),
            }));
        }
        
        Ok(None)
    }
    
    /// 检查相关性限制
    async fn check_correlation_limits(&self, request: &PretradeRiskRequest) -> Result<Option<RiskViolation>> {
        let tracker = self.exposure_tracker.read().await;
        
        // 计算与现有头寸的相关性暴露
        let mut correlated_exposure = Decimal::ZERO;
        
        for (symbol, exposure) in &tracker.current_exposures {
            if let Some(symbol_correlations) = tracker.correlation_matrix.get(symbol) {
                if let Some(correlation) = symbol_correlations.get(&request.symbol) {
                    if correlation.abs() >= self.config.correlation_threshold {
                        correlated_exposure += exposure.market_value;
                    }
                }
            }
        }
        
        let total_equity = tracker.current_exposures.values()
            .map(|e| e.market_value)
            .sum::<Decimal>();
        let correlated_weight = if total_equity > Decimal::ZERO {
            correlated_exposure / total_equity
        } else {
            Decimal::ZERO
        };
        
        if correlated_weight > self.config.max_correlated_exposure {
            return Ok(Some(RiskViolation {
                rule_type: RiskRuleType::CorrelationExposure,
                severity: RiskSeverity::Medium,
                description: format!("Correlated exposure {:.2}% exceeds limit {:.2}%", 
                    correlated_weight * Decimal::from(100),
                    self.config.max_correlated_exposure * Decimal::from(100)),
                current_value: correlated_weight,
                limit_value: self.config.max_correlated_exposure,
                suggested_action: "Diversify positions to reduce correlation".to_string(),
            }));
        }
        
        Ok(None)
    }
    
    /// 检查流动性限制
    async fn check_liquidity_limits(&self, request: &PretradeRiskRequest) -> Result<Option<RiskViolation>> {
        let tracker = self.exposure_tracker.read().await;
        
        if let Some(liquidity) = tracker.liquidity_scores.get(&request.symbol) {
            if liquidity.score < self.config.min_liquidity_score {
                return Ok(Some(RiskViolation {
                    rule_type: RiskRuleType::LiquidityLimit,
                    severity: RiskSeverity::Medium,
                    description: format!("Liquidity score {} for {} below minimum {}", 
                        liquidity.score, request.symbol, self.config.min_liquidity_score),
                    current_value: liquidity.score,
                    limit_value: self.config.min_liquidity_score,
                    suggested_action: "Trade only highly liquid instruments".to_string(),
                }));
            }
        }
        
        Ok(None)
    }
    
    /// 检查基于时间的限制
    async fn check_time_based_limits(&self, request: &PretradeRiskRequest) -> Result<Option<RiskViolation>> {
        let now = request.timestamp;
        let hour = now.hour();
        let weekday = now.weekday();
        
        // 检查是否是交易时间外
        let is_overnight = hour < 9 || hour > 17;  // 简化的判断
        let is_weekend = weekday == chrono::Weekday::Sat || weekday == chrono::Weekday::Sun;
        
        if is_weekend {
            if let Some(&limit) = self.config.weekend_position_limits.get(&request.symbol) {
                let order_value = request.quantity * request.price.unwrap_or(Decimal::ZERO);
                if order_value > limit {
                    return Ok(Some(RiskViolation {
                        rule_type: RiskRuleType::PositionLimit,
                        severity: RiskSeverity::Medium,
                        description: "Weekend position limit exceeded".to_string(),
                        current_value: order_value,
                        limit_value: limit,
                        suggested_action: "Reduce weekend position sizes".to_string(),
                    }));
                }
            }
        } else if is_overnight {
            if let Some(&limit) = self.config.overnight_position_limits.get(&request.symbol) {
                let order_value = request.quantity * request.price.unwrap_or(Decimal::ZERO);
                if order_value > limit {
                    return Ok(Some(RiskViolation {
                        rule_type: RiskRuleType::PositionLimit,
                        severity: RiskSeverity::Low,
                        description: "Overnight position limit exceeded".to_string(),
                        current_value: order_value,
                        limit_value: limit,
                        suggested_action: "Reduce overnight exposure".to_string(),
                    }));
                }
            }
        }
        
        Ok(None)
    }
    
    /// 更新暴露跟踪
    async fn update_exposure_tracking(&self, request: &PretradeRiskRequest, context: &RiskContext) -> Result<()> {
        let mut tracker = self.exposure_tracker.write().await;
        
        // 更新当前暴露
        for (symbol, position) in &context.current_positions {
            tracker.current_exposures.insert(symbol.clone(), AssetExposure {
                symbol: symbol.clone(),
                position_size: position.quantity,
                market_value: position.market_value,
                weight: position.weight,
                unrealized_pnl: position.unrealized_pnl,
                beta: None,
                sector: self.get_symbol_sector(symbol).await.ok().flatten(),
                country: self.get_symbol_country(symbol).await.ok().flatten(),
                currency: self.get_symbol_currency(symbol).await.unwrap_or_else(|_| "USD".to_string()),
                last_updated: Utc::now(),
            });
        }
        
        // 重新计算总体暴露
        tracker.total_long_exposure = tracker.current_exposures.values()
            .filter(|e| e.market_value > Decimal::ZERO)
            .map(|e| e.market_value)
            .sum();
            
        tracker.total_short_exposure = tracker.current_exposures.values()
            .filter(|e| e.market_value < Decimal::ZERO)
            .map(|e| e.market_value.abs())
            .sum();
            
        tracker.net_exposure = tracker.total_long_exposure - tracker.total_short_exposure;
        tracker.gross_exposure = tracker.total_long_exposure + tracker.total_short_exposure;
        
        tracker.current_leverage = context.account_state.leverage_ratio;
        tracker.last_updated = Utc::now();
        
        Ok(())
    }
    
    /// 生成预警
    async fn generate_warnings(&self) -> Result<Vec<RiskWarning>> {
        let tracker = self.exposure_tracker.read().await;
        let mut warnings = Vec::new();
        
        // 检查接近限制的暴露
        let gross_utilization = tracker.gross_exposure / self.config.max_gross_exposure;
        if gross_utilization >= self.config.warning_threshold_percent {
            warnings.push(RiskWarning {
                rule_type: RiskRuleType::ConcentrationLimit,
                description: format!("Gross exposure approaching limit: {:.1}%", 
                    gross_utilization * Decimal::from(100)),
                threshold_breach_percent: gross_utilization * Decimal::from(100),
                recommendation: "Monitor position sizes closely".to_string(),
            });
        }
        
        Ok(warnings)
    }
    
    /// 生成建议调整
    async fn generate_suggested_adjustments(&self, violations: &[RiskViolation]) -> Result<Vec<String>> {
        let mut suggestions = Vec::new();
        
        for violation in violations {
            match violation.rule_type {
                RiskRuleType::PositionLimit => {
                    suggestions.push("Consider splitting large orders".to_string());
                    suggestions.push("Review position sizing strategy".to_string());
                }
                RiskRuleType::ConcentrationLimit => {
                    suggestions.push("Diversify holdings across sectors/regions".to_string());
                    suggestions.push("Implement dynamic position sizing".to_string());
                }
                RiskRuleType::LeverageLimit => {
                    suggestions.push("Reduce margin usage".to_string());
                    suggestions.push("Close low-conviction positions".to_string());
                }
                RiskRuleType::CorrelationExposure => {
                    suggestions.push("Add uncorrelated assets".to_string());
                    suggestions.push("Review correlation metrics".to_string());
                }
                _ => {}
            }
        }
        
        suggestions.dedup();
        Ok(suggestions)
    }
    
    // 辅助方法（在实际系统中应该从外部数据源获取）
    async fn get_symbol_sector(&self, symbol: &str) -> Result<Option<String>> {
        // 简化实现 - 实际应该从数据源获取
        let sectors = [
            ("AAPL", "Technology"),
            ("GOOGL", "Technology"),
            ("MSFT", "Technology"),
            ("JPM", "Financial"),
            ("BAC", "Financial"),
            ("XOM", "Energy"),
            ("CVX", "Energy"),
        ];
        
        Ok(sectors.iter()
            .find(|(s, _)| *s == symbol)
            .map(|(_, sector)| sector.to_string()))
    }
    
    async fn get_symbol_country(&self, symbol: &str) -> Result<Option<String>> {
        // 简化实现
        Ok(Some("US".to_string()))
    }
    
    async fn get_symbol_currency(&self, symbol: &str) -> Result<String> {
        // 简化实现
        Ok("USD".to_string())
    }
}

impl ExposureTracker {
    pub fn new() -> Result<Self> {
        Ok(Self {
            current_exposures: HashMap::new(),
            sector_exposures: HashMap::new(),
            country_exposures: HashMap::new(),
            currency_exposures: HashMap::new(),
            total_long_exposure: Decimal::ZERO,
            total_short_exposure: Decimal::ZERO,
            net_exposure: Decimal::ZERO,
            gross_exposure: Decimal::ZERO,
            current_leverage: Decimal::ZERO,
            correlation_matrix: HashMap::new(),
            correlation_last_updated: Utc::now(),
            liquidity_scores: HashMap::new(),
            exposure_history: std::collections::VecDeque::new(),
            last_updated: Utc::now(),
        })
    }
}

impl RiskRule for ExposureLimitsRule {
    fn rule_type(&self) -> RiskRuleType {
        RiskRuleType::ConcentrationLimit
    }
    
    fn check(&self, request: &PretradeRiskRequest, context: &RiskContext) -> Result<RiskCheckResult> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            self.check_exposure_limits(request, context).await
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

impl Default for ExposureLimitsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            priority: 3,
            max_total_exposure: Decimal::from(10000000),     // $10M
            max_net_exposure: Decimal::from(5000000),        // $5M  
            max_gross_exposure: Decimal::from(15000000),     // $15M
            max_single_position_size: Decimal::from(1000000), // $1M
            max_single_position_weight: Decimal::from_parts(5, 0, 0, false, 2), // 5%
            max_position_concentration: Decimal::from_parts(20, 0, 0, false, 2), // 20%
            enable_sector_limits: true,
            max_sector_exposure: Decimal::from_parts(25, 0, 0, false, 2), // 25%
            max_sector_concentration: Decimal::from_parts(30, 0, 0, false, 2), // 30%
            sector_limits: HashMap::new(),
            enable_geographic_limits: true,
            max_country_exposure: Decimal::from_parts(50, 0, 0, false, 2), // 50%
            max_region_exposure: Decimal::from_parts(70, 0, 0, false, 2), // 70%
            country_limits: HashMap::new(),
            enable_currency_limits: true,
            max_currency_exposure: Decimal::from_parts(60, 0, 0, false, 2), // 60%
            currency_limits: HashMap::new(),
            max_leverage_ratio: Decimal::from(5), // 5x
            leverage_by_asset_class: HashMap::new(),
            enable_correlation_limits: true,
            max_correlated_exposure: Decimal::from_parts(40, 0, 0, false, 2), // 40%
            correlation_threshold: Decimal::from_parts(70, 0, 0, false, 2), // 0.7
            correlation_window_days: 30,
            enable_liquidity_limits: true,
            min_liquidity_score: Decimal::from(30), // Minimum 30/100
            max_illiquid_exposure: Decimal::from_parts(10, 0, 0, false, 2), // 10%
            liquidity_adjustment_factor: Decimal::from_parts(80, 0, 0, false, 2), // 0.8
            enable_time_limits: true,
            intraday_position_limits: HashMap::new(),
            overnight_position_limits: HashMap::new(),
            weekend_position_limits: HashMap::new(),
            warning_threshold_percent: Decimal::from_parts(85, 0, 0, false, 2), // 85%
            critical_threshold_percent: Decimal::from_parts(95, 0, 0, false, 2), // 95%
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_single_position_limit() {
        let config = ExposureLimitsConfig {
            max_single_position_size: Decimal::from(500000), // $500K limit
            ..Default::default()
        };
        
        let rule = ExposureLimitsRule::new(config).unwrap();
        
        let request = PretradeRiskRequest {
            order_id: "test_order".to_string(),
            strategy_id: "test_strategy".to_string(),
            account_id: "test_account".to_string(),
            symbol: "AAPL".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(10000), // Large order
            price: Some(Decimal::from(150)),
            order_type: OrderType::Limit,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };
        
        let context = create_test_context();
        let result = rule.check(&request, &context).unwrap();
        
        // Should violate position limit (10,000 * $150 = $1.5M > $500K)
        assert!(!result.passed);
    }
    
    #[test]
    fn test_exposure_config_default() {
        let config = ExposureLimitsConfig::default();
        assert_eq!(config.max_leverage_ratio, Decimal::from(5));
        assert_eq!(config.max_single_position_weight, Decimal::from_parts(5, 0, 0, false, 2));
        assert!(config.enable_sector_limits);
    }
}

fn create_test_context() -> RiskContext {
    let mut positions = HashMap::new();
    positions.insert("AAPL".to_string(), Position {
        symbol: "AAPL".to_string(),
        quantity: Decimal::from(1000),
        average_price: Decimal::from(150),
        market_value: Decimal::from(150000),
        unrealized_pnl: Decimal::from(5000),
        weight: Decimal::from_parts(15, 0, 0, false, 2), // 15%
    });
    
    RiskContext {
        current_positions: positions,
        daily_pnl: Decimal::from(2000),
        total_equity: Decimal::from(1000000),
        max_drawdown: Decimal::from_parts(5, 0, 0, false, 2), // 5%
        correlation_matrix: HashMap::new(),
        market_conditions: MarketConditions {
            timestamp: Utc::now(),
            volatility_index: Decimal::from_parts(20, 0, 0, false, 2),
            market_stress_indicator: Decimal::from_parts(15, 0, 0, false, 2),
            liquidity_conditions: LiquidityConditions {
                bid_ask_spread_percentile: Decimal::from_parts(10, 0, 0, false, 2),
                depth_ratio: Decimal::from_parts(85, 0, 0, false, 2),
                market_impact_estimate: Decimal::from_parts(1, 0, 0, false, 2),
            },
        },
        account_state: AccountState {
            account_id: "test_account".to_string(),
            total_equity: Decimal::from(1000000),
            available_margin: Decimal::from(500000),
            used_margin: Decimal::from(500000),
            leverage_ratio: Decimal::from(2),
            daily_pnl: Decimal::from(2000),
            max_drawdown_today: Decimal::from_parts(5, 0, 0, false, 2),
            order_count_today: 25,
            volume_traded_today: Decimal::from(500000),
        },
    }
}