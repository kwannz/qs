use anyhow::Result;
use rust_decimal::prelude::ToPrimitive;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Risk severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Risk categories
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskCategory {
    Market,
    Credit,
    Operational,
    Liquidity,
    Regulatory,
    Concentration,
    Technology,
}

/// Real-time risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub timestamp: DateTime<Utc>,
    pub portfolio_value: Decimal,
    pub total_exposure: Decimal,
    pub max_drawdown: Decimal,
    pub var_95: Decimal,  // Value at Risk 95%
    pub var_99: Decimal,  // Value at Risk 99%
    pub expected_shortfall: Decimal,
    pub leverage_ratio: Decimal,
    pub concentration_risk: Decimal,
    pub liquidity_ratio: Decimal,
    pub positions_by_asset: HashMap<String, PositionRisk>,
    pub overall_risk_score: RiskScore,
}

/// Position-level risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionRisk {
    pub asset: String,
    pub position_size: Decimal,
    pub market_value: Decimal,
    pub unrealized_pnl: Decimal,
    pub delta: Decimal,
    pub gamma: Decimal,
    pub vega: Decimal,
    pub theta: Decimal,
    pub var_contribution: Decimal,
    pub risk_weight: Decimal,
}

/// Overall risk scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskScore {
    pub overall_score: u8,        // 0-100
    pub market_risk_score: u8,
    pub credit_risk_score: u8,
    pub operational_risk_score: u8,
    pub liquidity_risk_score: u8,
    pub severity: RiskSeverity,
    pub alerts: Vec<RiskAlert>,
}

/// Risk alerts and warnings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAlert {
    pub id: Uuid,
    pub category: RiskCategory,
    pub severity: RiskSeverity,
    pub title: String,
    pub description: String,
    pub threshold_breached: Decimal,
    pub current_value: Decimal,
    pub recommended_action: String,
    pub timestamp: DateTime<Utc>,
}

/// Risk limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    pub max_portfolio_var: Decimal,
    pub max_position_size: Decimal,
    pub max_leverage: Decimal,
    pub max_concentration: Decimal,
    pub min_liquidity_ratio: Decimal,
    pub stop_loss_threshold: Decimal,
    pub daily_loss_limit: Decimal,
    pub position_limits_by_asset: HashMap<String, Decimal>,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_portfolio_var: Decimal::from_f32_retain(0.02).unwrap(), // 2%
            max_position_size: Decimal::from(1000000), // $1M
            max_leverage: Decimal::from(10),
            max_concentration: Decimal::from_f32_retain(0.15).unwrap(), // 15%
            min_liquidity_ratio: Decimal::from_f32_retain(0.1).unwrap(), // 10%
            stop_loss_threshold: Decimal::from_f32_retain(-0.05).unwrap(), // -5%
            daily_loss_limit: Decimal::from(100000), // $100K
            position_limits_by_asset: HashMap::new(),
        }
    }
}

/// Risk calculation utilities
pub struct RiskCalculator;

impl RiskCalculator {
    /// Calculate Value at Risk using historical simulation
    pub fn calculate_var(returns: &[Decimal], confidence_level: f64) -> Result<Decimal> {
        if returns.is_empty() {
            return Ok(Decimal::ZERO);
        }
        
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort();
        
        let index = ((1.0 - confidence_level) * sorted_returns.len() as f64) as usize;
        let var = sorted_returns.get(index).unwrap_or(&Decimal::ZERO);
        
        Ok(-*var) // VaR is typically positive
    }
    
    /// Calculate Expected Shortfall (Conditional VaR)
    pub fn calculate_expected_shortfall(returns: &[Decimal], confidence_level: f64) -> Result<Decimal> {
        if returns.is_empty() {
            return Ok(Decimal::ZERO);
        }
        
        let var = Self::calculate_var(returns, confidence_level)?;
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort();
        
        let cutoff_index = ((1.0 - confidence_level) * sorted_returns.len() as f64) as usize;
        
        let tail_returns: Vec<_> = sorted_returns.iter().take(cutoff_index + 1).collect();
        if tail_returns.is_empty() {
            return Ok(var);
        }
        
        let sum: Decimal = tail_returns.iter().map(|&x| *x).sum();
        let expected_shortfall = -sum / Decimal::from(tail_returns.len());
        
        Ok(expected_shortfall)
    }
    
    /// Calculate maximum drawdown
    pub fn calculate_max_drawdown(portfolio_values: &[Decimal]) -> Decimal {
        if portfolio_values.len() < 2 {
            return Decimal::ZERO;
        }
        
        let mut max_value = portfolio_values[0];
        let mut max_drawdown = Decimal::ZERO;
        
        for &value in portfolio_values.iter().skip(1) {
            if value > max_value {
                max_value = value;
            }
            
            let drawdown = (max_value - value) / max_value;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
        
        max_drawdown
    }
    
    /// Calculate concentration risk (Herfindahl-Hirschman Index)
    pub fn calculate_concentration_risk(position_weights: &[Decimal]) -> Decimal {
        position_weights.iter()
            .map(|w| w * w)
            .sum()
    }
    
    /// Calculate overall risk score
    pub fn calculate_risk_score(metrics: &RiskMetrics, limits: &RiskLimits) -> RiskScore {
        let mut alerts = Vec::new();
        
        // Market risk scoring
        let var_ratio = metrics.var_95 / limits.max_portfolio_var;
        let market_score = (var_ratio * Decimal::from(100)).min(Decimal::from(100)).max(Decimal::ZERO);
        
        // Leverage risk
        let leverage_ratio = metrics.leverage_ratio / limits.max_leverage;
        if leverage_ratio > Decimal::ONE {
            alerts.push(RiskAlert {
                id: Uuid::new_v4(),
                category: RiskCategory::Market,
                severity: if leverage_ratio > Decimal::from_f32_retain(1.5).unwrap() { 
                    RiskSeverity::Critical 
                } else { 
                    RiskSeverity::High 
                },
                title: "Leverage Limit Exceeded".to_string(),
                description: format!("Current leverage {} exceeds limit {}", 
                    metrics.leverage_ratio, limits.max_leverage),
                threshold_breached: limits.max_leverage,
                current_value: metrics.leverage_ratio,
                recommended_action: "Reduce position sizes to lower leverage".to_string(),
                timestamp: Utc::now(),
            });
        }
        
        // Concentration risk
        let concentration_ratio = metrics.concentration_risk / limits.max_concentration;
        if concentration_ratio > Decimal::ONE {
            alerts.push(RiskAlert {
                id: Uuid::new_v4(),
                category: RiskCategory::Concentration,
                severity: RiskSeverity::Medium,
                title: "High Concentration Risk".to_string(),
                description: "Portfolio is highly concentrated in few assets".to_string(),
                threshold_breached: limits.max_concentration,
                current_value: metrics.concentration_risk,
                recommended_action: "Diversify portfolio across more assets".to_string(),
                timestamp: Utc::now(),
            });
        }
        
        // Liquidity risk
        if metrics.liquidity_ratio < limits.min_liquidity_ratio {
            alerts.push(RiskAlert {
                id: Uuid::new_v4(),
                category: RiskCategory::Liquidity,
                severity: RiskSeverity::High,
                title: "Low Liquidity Ratio".to_string(),
                description: "Portfolio liquidity below minimum threshold".to_string(),
                threshold_breached: limits.min_liquidity_ratio,
                current_value: metrics.liquidity_ratio,
                recommended_action: "Increase cash reserves or reduce illiquid positions".to_string(),
                timestamp: Utc::now(),
            });
        }
        
        let market_score_u8 = market_score.to_u8().unwrap_or(0);
        let overall_score = market_score_u8;
        
        let severity = match overall_score {
            0..=25 => RiskSeverity::Low,
            26..=50 => RiskSeverity::Medium,
            51..=75 => RiskSeverity::High,
            _ => RiskSeverity::Critical,
        };
        
        RiskScore {
            overall_score,
            market_risk_score: market_score_u8,
            credit_risk_score: 20, // Placeholder
            operational_risk_score: 10, // Placeholder
            liquidity_risk_score: if metrics.liquidity_ratio < limits.min_liquidity_ratio { 80 } else { 20 },
            severity,
            alerts,
        }
    }
}

/// Historical risk metrics for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskHistory {
    pub date: DateTime<Utc>,
    pub daily_var: Decimal,
    pub realized_volatility: Decimal,
    pub portfolio_return: Decimal,
    pub max_drawdown: Decimal,
    pub risk_score: u8,
}