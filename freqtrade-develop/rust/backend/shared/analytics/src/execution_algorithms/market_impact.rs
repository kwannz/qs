//! Almgren-Chriss Market Impact Model Implementation
//!
//! Provides realistic market impact modeling for execution algorithms including:
//! - Permanent impact (price discovery and adverse selection)
//! - Temporary impact (liquidity consumption and recovery)
//! - Multi-factor attribution and regime-dependent adjustments
//! - Dynamic calibration from real-time market data

use anyhow::{Result, Context};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use super::{MarketConditions, OrderSide, ParentOrder};

/// Almgren-Chriss market impact model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlmgrenChrissConfig {
    /// Permanent impact coefficient (γ)
    pub gamma: f64,
    /// Temporary impact coefficient (η) 
    pub eta: f64,
    /// Volatility scaling factor
    pub volatility_scaling: f64,
    /// Volume scaling exponent (typically 0.5 for permanent, 0.6 for temporary)
    pub permanent_exponent: f64,
    pub temporary_exponent: f64,
    /// Minimum impact threshold (basis points)
    pub min_impact_bps: f64,
    /// Maximum impact threshold (basis points)
    pub max_impact_bps: f64,
    /// Calibration lookback period
    pub calibration_window_days: u32,
    /// Enable regime-dependent adjustments
    pub enable_regime_adjustments: bool,
    /// Enable cross-asset impact modeling
    pub enable_cross_impact: bool,
}

impl Default for AlmgrenChrissConfig {
    fn default() -> Self {
        Self {
            gamma: 0.314,                // Empirical estimate for permanent impact
            eta: 0.142,                  // Empirical estimate for temporary impact
            volatility_scaling: 1.0,
            permanent_exponent: 0.5,     // Square-root law for permanent impact
            temporary_exponent: 0.6,     // Empirically observed for temporary impact
            min_impact_bps: 0.1,
            max_impact_bps: 500.0,
            calibration_window_days: 30,
            enable_regime_adjustments: true,
            enable_cross_impact: true,
        }
    }
}

/// Market impact calculator using Almgren-Chriss model
#[derive(Debug)]
pub struct AlmgrenChrissModel {
    config: AlmgrenChrissConfig,
    calibration_data: HashMap<String, MarketImpactCalibration>,
    regime_detector: RegimeDetector,
    cross_impact_matrix: HashMap<String, HashMap<String, f64>>,
}

/// Market impact calibration data for a symbol
#[derive(Debug, Clone)]
struct MarketImpactCalibration {
    symbol: String,
    permanent_coefficient: f64,
    temporary_coefficient: f64,
    volatility_adjustment: f64,
    last_updated: DateTime<Utc>,
    sample_count: u32,
    r_squared: f64,
    confidence_interval: (f64, f64),
}

/// Market regime detector
#[derive(Debug)]
struct RegimeDetector {
    current_regime: MarketRegime,
    regime_adjustments: HashMap<MarketRegime, RegimeAdjustment>,
    volatility_threshold_low: f64,
    volatility_threshold_high: f64,
    liquidity_threshold_low: f64,
    liquidity_threshold_high: f64,
}

/// Market regime classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    Normal,
    HighVolatility,
    LowLiquidity,
    Stressed,
    PreOpen,
    PostClose,
    NewsEvent,
}

/// Regime-specific adjustments
#[derive(Debug, Clone)]
struct RegimeAdjustment {
    permanent_multiplier: f64,
    temporary_multiplier: f64,
    decay_rate_adjustment: f64,
    min_participation_rate: f64,
}

/// Market impact calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketImpactResult {
    /// Total market impact in basis points
    pub total_impact_bps: f64,
    /// Permanent impact component
    pub permanent_impact_bps: f64,
    /// Temporary impact component  
    pub temporary_impact_bps: f64,
    /// Cross-asset impact component
    pub cross_impact_bps: f64,
    /// Regime adjustment applied
    pub regime_adjustment_bps: f64,
    /// Confidence interval for the estimate
    pub confidence_interval: (f64, f64),
    /// Market regime during calculation
    pub market_regime: MarketRegime,
    /// Impact timeline for gradual execution
    pub impact_timeline: Vec<ImpactTimePoint>,
    /// Model diagnostics
    pub diagnostics: ImpactModelDiagnostics,
}

/// Impact at a specific time point during execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactTimePoint {
    pub time_offset_minutes: f64,
    pub cumulative_quantity: f64,
    pub instantaneous_permanent_impact_bps: f64,
    pub instantaneous_temporary_impact_bps: f64,
    pub cumulative_permanent_impact_bps: f64,
    pub temporary_impact_with_decay_bps: f64,
    pub expected_recovery_time_minutes: f64,
}

/// Model diagnostics and quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactModelDiagnostics {
    pub model_confidence: f64,
    pub calibration_quality: f64,
    pub regime_detection_confidence: f64,
    pub cross_impact_significance: f64,
    pub volatility_adjustment_factor: f64,
    pub liquidity_adjustment_factor: f64,
}

impl AlmgrenChrissModel {
    /// Create a new Almgren-Chriss market impact model
    pub fn new(config: AlmgrenChrissConfig) -> Self {
        Self {
            config,
            calibration_data: HashMap::new(),
            regime_detector: RegimeDetector::new(),
            cross_impact_matrix: HashMap::new(),
        }
    }

    /// Calculate market impact for a given order and market conditions
    pub fn calculate_impact(
        &self,
        parent_order: &ParentOrder,
        market_conditions: &MarketConditions,
        execution_strategy: &ImpactExecutionStrategy,
    ) -> Result<MarketImpactResult> {
        info!("Calculating market impact for order {} ({})", 
              parent_order.id, parent_order.symbol);

        // Detect current market regime
        let current_regime = self.regime_detector.detect_regime(market_conditions)?;
        
        // Get calibration data for the symbol
        let calibration = self.get_or_create_calibration(&parent_order.symbol, market_conditions)?;
        
        // Calculate base impact components
        let permanent_impact = self.calculate_permanent_impact(
            parent_order,
            market_conditions,
            &calibration,
            &current_regime,
        )?;
        
        let temporary_impact = self.calculate_temporary_impact(
            parent_order,
            market_conditions,
            &calibration,
            execution_strategy,
            &current_regime,
        )?;
        
        // Calculate cross-asset impact if enabled
        let cross_impact = if self.config.enable_cross_impact {
            self.calculate_cross_impact(parent_order, market_conditions)?
        } else {
            0.0
        };
        
        // Apply regime adjustments
        let regime_adjustment = self.apply_regime_adjustments(
            permanent_impact + temporary_impact,
            &current_regime,
        )?;
        
        let total_impact = permanent_impact + temporary_impact + cross_impact + regime_adjustment;
        
        // Generate impact timeline for execution planning
        let impact_timeline = self.generate_impact_timeline(
            parent_order,
            market_conditions,
            execution_strategy,
            &calibration,
            &current_regime,
        )?;
        
        // Calculate confidence interval
        let confidence_interval = self.calculate_confidence_interval(
            total_impact,
            &calibration,
            market_conditions,
        )?;
        
        // Generate diagnostics
        let diagnostics = self.generate_diagnostics(&calibration, market_conditions, &current_regime)?;
        
        let result = MarketImpactResult {
            total_impact_bps: total_impact.clamp(self.config.min_impact_bps, self.config.max_impact_bps),
            permanent_impact_bps: permanent_impact,
            temporary_impact_bps: temporary_impact,
            cross_impact_bps: cross_impact,
            regime_adjustment_bps: regime_adjustment,
            confidence_interval,
            market_regime: current_regime,
            impact_timeline,
            diagnostics,
        };

        debug!("Market impact calculation completed: {:.2} bps total ({:.2} permanent, {:.2} temporary)",
               result.total_impact_bps, result.permanent_impact_bps, result.temporary_impact_bps);

        Ok(result)
    }

    /// Calculate permanent impact using Almgren-Chriss model
    /// Formula: γ * σ * (V / VD)^α where α is typically 0.5
    fn calculate_permanent_impact(
        &self,
        parent_order: &ParentOrder,
        market_conditions: &MarketConditions,
        calibration: &MarketImpactCalibration,
        regime: &MarketRegime,
    ) -> Result<f64> {
        let volume_ratio = parent_order.total_quantity / market_conditions.average_daily_volume;
        let volatility = market_conditions.realized_volatility;
        
        let base_impact = calibration.permanent_coefficient 
            * volatility 
            * volume_ratio.powf(self.config.permanent_exponent);
        
        // Convert to basis points
        let impact_bps = base_impact * 10000.0;
        
        // Apply volatility scaling
        let volatility_adjusted = impact_bps * (1.0 + calibration.volatility_adjustment);
        
        // Apply regime-specific adjustments
        let regime_multiplier = self.regime_detector.get_permanent_multiplier(regime);
        
        Ok(volatility_adjusted * regime_multiplier)
    }

    /// Calculate temporary impact with decay modeling
    /// Formula: η * σ * (V / VD)^β * f(t) where β is typically 0.6 and f(t) is decay function
    fn calculate_temporary_impact(
        &self,
        parent_order: &ParentOrder,
        market_conditions: &MarketConditions,
        calibration: &MarketImpactCalibration,
        execution_strategy: &ImpactExecutionStrategy,
        regime: &MarketRegime,
    ) -> Result<f64> {
        let volume_ratio = parent_order.total_quantity / market_conditions.average_daily_volume;
        let volatility = market_conditions.realized_volatility;
        
        // Base temporary impact
        let base_impact = calibration.temporary_coefficient 
            * volatility 
            * volume_ratio.powf(self.config.temporary_exponent);
        
        // Adjust for execution speed (faster execution = higher temporary impact)
        let time_factor = self.calculate_time_factor(parent_order.time_horizon, execution_strategy);
        
        // Apply participation rate adjustment
        let participation_factor = self.calculate_participation_factor(
            execution_strategy.target_participation_rate,
            market_conditions,
        )?;
        
        let impact_bps = base_impact * time_factor * participation_factor * 10000.0;
        
        // Apply regime-specific adjustments
        let regime_multiplier = self.regime_detector.get_temporary_multiplier(regime);
        
        Ok(impact_bps * regime_multiplier)
    }

    /// Calculate cross-asset impact from correlated instruments
    fn calculate_cross_impact(
        &self,
        parent_order: &ParentOrder,
        market_conditions: &MarketConditions,
    ) -> Result<f64> {
        // Simplified cross-impact calculation
        // In production, this would use correlation matrices and spillover effects
        
        if let Some(correlations) = self.cross_impact_matrix.get(&parent_order.symbol) {
            let mut cross_impact = 0.0;
            
            for (correlated_symbol, correlation) in correlations {
                if correlation.abs() > 0.3 { // Only consider significant correlations
                    let spillover_factor = correlation * 0.1; // Simplified spillover
                    cross_impact += spillover_factor * market_conditions.price_momentum;
                }
            }
            
            Ok(cross_impact.abs())
        } else {
            Ok(0.0)
        }
    }

    /// Apply regime-specific adjustments to base impact
    fn apply_regime_adjustments(&self, base_impact: f64, regime: &MarketRegime) -> Result<f64> {
        let adjustment = match regime {
            MarketRegime::HighVolatility => base_impact * 0.3, // 30% increase in high vol
            MarketRegime::LowLiquidity => base_impact * 0.5,   // 50% increase in low liquidity
            MarketRegime::Stressed => base_impact * 0.8,       // 80% increase in stress
            MarketRegime::PreOpen | MarketRegime::PostClose => base_impact * 0.2, // 20% increase
            MarketRegime::NewsEvent => base_impact * 0.4,      // 40% increase
            MarketRegime::Normal => 0.0,                       // No adjustment
        };
        
        Ok(adjustment)
    }

    /// Generate impact timeline for execution planning
    fn generate_impact_timeline(
        &self,
        parent_order: &ParentOrder,
        market_conditions: &MarketConditions,
        execution_strategy: &ImpactExecutionStrategy,
        calibration: &MarketImpactCalibration,
        regime: &MarketRegime,
    ) -> Result<Vec<ImpactTimePoint>> {
        let mut timeline = Vec::new();
        let total_minutes = parent_order.time_horizon as f64 / 60.0;
        let num_points = (total_minutes / 5.0).ceil() as usize; // 5-minute intervals
        
        let mut cumulative_quantity = 0.0;
        let quantity_per_interval = parent_order.total_quantity / num_points as f64;
        
        for i in 0..num_points {
            let time_offset = (i as f64) * 5.0; // 5 minutes per interval
            cumulative_quantity += quantity_per_interval;
            
            // Calculate instantaneous impacts
            let volume_ratio = quantity_per_interval / market_conditions.average_daily_volume;
            let volatility = market_conditions.realized_volatility;
            
            let instant_permanent = calibration.permanent_coefficient 
                * volatility 
                * volume_ratio.powf(self.config.permanent_exponent) 
                * 10000.0;
                
            let instant_temporary = calibration.temporary_coefficient 
                * volatility 
                * volume_ratio.powf(self.config.temporary_exponent) 
                * 10000.0;
            
            // Calculate cumulative permanent impact (doesn't decay)
            let cumulative_permanent = instant_permanent * (i + 1) as f64;
            
            // Calculate temporary impact with decay
            let decay_rate = 0.1; // 10% per interval
            let temporary_with_decay = instant_temporary * (-decay_rate * time_offset).exp();
            
            // Estimate recovery time
            let recovery_time = (-decay_rate.ln()) / decay_rate; // Time to 1/e of original impact
            
            timeline.push(ImpactTimePoint {
                time_offset_minutes: time_offset,
                cumulative_quantity,
                instantaneous_permanent_impact_bps: instant_permanent,
                instantaneous_temporary_impact_bps: instant_temporary,
                cumulative_permanent_impact_bps: cumulative_permanent,
                temporary_impact_with_decay_bps: temporary_with_decay,
                expected_recovery_time_minutes: recovery_time,
            });
        }
        
        Ok(timeline)
    }

    /// Calculate time factor for execution speed adjustment
    fn calculate_time_factor(&self, time_horizon_seconds: i64, strategy: &ImpactExecutionStrategy) -> f64 {
        let time_hours = time_horizon_seconds as f64 / 3600.0;
        
        // Inverse relationship: shorter time = higher impact
        match strategy {
            ImpactExecutionStrategy::Aggressive => (1.0 / time_hours).min(5.0), // Cap at 5x
            ImpactExecutionStrategy::Moderate => (2.0 / time_hours).min(3.0),   // Cap at 3x
            ImpactExecutionStrategy::Passive => (4.0 / time_hours).min(2.0),    // Cap at 2x
        }
    }

    /// Calculate participation rate factor
    fn calculate_participation_factor(
        &self,
        participation_rate: f64,
        market_conditions: &MarketConditions,
    ) -> Result<f64> {
        // Higher participation rate = higher temporary impact
        let base_factor = participation_rate.powf(0.7); // Empirical exponent
        
        // Adjust for current market volume
        let volume_adjustment = if market_conditions.current_volume > 0.0 {
            (market_conditions.average_daily_volume / market_conditions.current_volume).sqrt()
        } else {
            2.0 // Penalty for no volume
        };
        
        Ok(base_factor * volume_adjustment)
    }

    /// Get or create calibration data for a symbol
    fn get_or_create_calibration(
        &self,
        symbol: &str,
        market_conditions: &MarketConditions,
    ) -> Result<MarketImpactCalibration> {
        if let Some(calibration) = self.calibration_data.get(symbol) {
            // Check if calibration is still fresh
            let age = Utc::now() - calibration.last_updated;
            if age.num_days() < self.config.calibration_window_days as i64 {
                return Ok(calibration.clone());
            }
        }
        
        // Create default calibration (in production, this would use historical data)
        Ok(MarketImpactCalibration {
            symbol: symbol.to_string(),
            permanent_coefficient: self.config.gamma,
            temporary_coefficient: self.config.eta,
            volatility_adjustment: market_conditions.realized_volatility / 0.02 - 1.0, // Normalized to 2% baseline
            last_updated: Utc::now(),
            sample_count: 100, // Simulated sample size
            r_squared: 0.75,   // Simulated fit quality
            confidence_interval: (0.8, 1.2), // Simulated confidence
        })
    }

    /// Calculate confidence interval for impact estimate
    fn calculate_confidence_interval(
        &self,
        total_impact: f64,
        calibration: &MarketImpactCalibration,
        _market_conditions: &MarketConditions,
    ) -> Result<(f64, f64)> {
        let uncertainty_factor = 1.0 - calibration.r_squared;
        let error_margin = total_impact * uncertainty_factor * 0.5;
        
        Ok((total_impact - error_margin, total_impact + error_margin))
    }

    /// Generate model diagnostics
    fn generate_diagnostics(
        &self,
        calibration: &MarketImpactCalibration,
        market_conditions: &MarketConditions,
        regime: &MarketRegime,
    ) -> Result<ImpactModelDiagnostics> {
        Ok(ImpactModelDiagnostics {
            model_confidence: calibration.r_squared,
            calibration_quality: (calibration.sample_count as f64 / 1000.0).min(1.0),
            regime_detection_confidence: 0.8, // Simulated
            cross_impact_significance: 0.3,   // Simulated
            volatility_adjustment_factor: 1.0 + calibration.volatility_adjustment,
            liquidity_adjustment_factor: (market_conditions.market_depth.total_bid_volume 
                + market_conditions.market_depth.total_ask_volume) / 1000.0, // Normalized
        })
    }

    /// Update calibration with new execution data (for online learning)
    pub fn update_calibration(
        &mut self,
        symbol: &str,
        actual_impact: f64,
        predicted_impact: f64,
        market_conditions: &MarketConditions,
    ) -> Result<()> {
        // Simple exponential weighted moving average update
        let alpha = 0.1; // Learning rate
        
        if let Some(calibration) = self.calibration_data.get_mut(symbol) {
            let prediction_error = actual_impact - predicted_impact;
            
            // Update coefficients based on prediction error
            calibration.permanent_coefficient *= 1.0 + alpha * prediction_error / predicted_impact;
            calibration.temporary_coefficient *= 1.0 + alpha * prediction_error / predicted_impact;
            
            // Update volatility adjustment
            let volatility_error = market_conditions.realized_volatility - 0.02;
            calibration.volatility_adjustment = alpha * volatility_error + (1.0 - alpha) * calibration.volatility_adjustment;
            
            calibration.last_updated = Utc::now();
            calibration.sample_count += 1;
            
            info!("Updated calibration for {} - error: {:.2} bps", symbol, prediction_error);
        }
        
        Ok(())
    }
}

/// Execution strategy for impact calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactExecutionStrategy {
    Aggressive,
    Moderate, 
    Passive,
}

impl ImpactExecutionStrategy {
    pub fn target_participation_rate(&self) -> f64 {
        match self {
            ImpactExecutionStrategy::Aggressive => 0.3, // 30%
            ImpactExecutionStrategy::Moderate => 0.15,  // 15%
            ImpactExecutionStrategy::Passive => 0.05,   // 5%
        }
    }
}

impl RegimeDetector {
    fn new() -> Self {
        let mut regime_adjustments = HashMap::new();
        
        regime_adjustments.insert(MarketRegime::Normal, RegimeAdjustment {
            permanent_multiplier: 1.0,
            temporary_multiplier: 1.0,
            decay_rate_adjustment: 1.0,
            min_participation_rate: 0.01,
        });
        
        regime_adjustments.insert(MarketRegime::HighVolatility, RegimeAdjustment {
            permanent_multiplier: 1.3,
            temporary_multiplier: 1.5,
            decay_rate_adjustment: 0.7, // Slower decay
            min_participation_rate: 0.005, // Lower participation
        });
        
        regime_adjustments.insert(MarketRegime::LowLiquidity, RegimeAdjustment {
            permanent_multiplier: 1.5,
            temporary_multiplier: 2.0,
            decay_rate_adjustment: 0.5, // Much slower decay
            min_participation_rate: 0.002, // Much lower participation
        });
        
        // Add other regime adjustments...
        
        Self {
            current_regime: MarketRegime::Normal,
            regime_adjustments,
            volatility_threshold_low: 0.01,   // 1% daily volatility
            volatility_threshold_high: 0.04,  // 4% daily volatility
            liquidity_threshold_low: 0.5,     // 50% of normal volume
            liquidity_threshold_high: 2.0,    // 200% of normal volume
        }
    }
    
    fn detect_regime(&self, market_conditions: &MarketConditions) -> Result<MarketRegime> {
        // Simple regime detection based on volatility and liquidity
        if market_conditions.realized_volatility > self.volatility_threshold_high {
            return Ok(MarketRegime::HighVolatility);
        }
        
        let volume_ratio = market_conditions.current_volume / market_conditions.average_daily_volume;
        if volume_ratio < self.liquidity_threshold_low {
            return Ok(MarketRegime::LowLiquidity);
        }
        
        // Check for pre/post market conditions
        if market_conditions.is_auction_period {
            return Ok(MarketRegime::PreOpen);
        }
        
        // Default to normal regime
        Ok(MarketRegime::Normal)
    }
    
    fn get_permanent_multiplier(&self, regime: &MarketRegime) -> f64 {
        self.regime_adjustments
            .get(regime)
            .map(|adj| adj.permanent_multiplier)
            .unwrap_or(1.0)
    }
    
    fn get_temporary_multiplier(&self, regime: &MarketRegime) -> f64 {
        self.regime_adjustments
            .get(regime)
            .map(|adj| adj.temporary_multiplier)
            .unwrap_or(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_market_conditions() -> MarketConditions {
        MarketConditions {
            symbol: "BTCUSDT".to_string(),
            timestamp: Utc::now(),
            mid_price: 50000.0,
            bid_price: 49995.0,
            ask_price: 50005.0,
            spread_bps: 2.0,
            tick_size: 0.01,
            bid_size: 1.0,
            ask_size: 1.0,
            market_depth: super::super::MarketDepth {
                bids: vec![],
                asks: vec![],
                total_bid_volume: 10.0,
                total_ask_volume: 10.0,
            },
            average_daily_volume: 1000.0,
            current_volume: 100.0,
            volume_profile: vec![],
            realized_volatility: 0.02,
            implied_volatility: 0.025,
            price_momentum: 0.001,
            short_term_trend: 0.0005,
            order_book_imbalance: 0.0,
            queue_position_estimate: 0.5,
            toxic_flow_indicator: 0.1,
            informed_trading_probability: 0.2,
            time_to_close: 7200, // 2 hours
            intraday_period: super::super::IntradayPeriod::MorningSession,
            is_auction_period: false,
            trading_session: super::super::TradingSession::NewYork,
        }
    }

    fn create_test_parent_order() -> ParentOrder {
        ParentOrder {
            id: "test_order_1".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            total_quantity: 10.0, // 1% of daily volume
            order_type: super::super::OrderType::Market,
            time_horizon: 3600, // 1 hour
            urgency: 0.5,
            limit_price: Some(50100.0),
            arrival_price: 50000.0,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_permanent_impact_calculation() {
        let model = AlmgrenChrissModel::new(AlmgrenChrissConfig::default());
        let parent_order = create_test_parent_order();
        let market_conditions = create_test_market_conditions();
        let execution_strategy = ImpactExecutionStrategy::Moderate;
        
        let result = model.calculate_impact(&parent_order, &market_conditions, &execution_strategy);
        
        assert!(result.is_ok());
        let impact = result.unwrap();
        
        // Verify impact is reasonable for 1% of daily volume
        assert!(impact.total_impact_bps > 0.0);
        assert!(impact.total_impact_bps < 100.0); // Should be less than 100 bps
        assert!(impact.permanent_impact_bps > 0.0);
        assert!(impact.temporary_impact_bps > 0.0);
    }

    #[test]
    fn test_regime_detection() {
        let model = AlmgrenChrissModel::new(AlmgrenChrissConfig::default());
        
        // Test high volatility regime
        let mut market_conditions = create_test_market_conditions();
        market_conditions.realized_volatility = 0.05; // 5% volatility
        
        let regime = model.regime_detector.detect_regime(&market_conditions);
        assert!(regime.is_ok());
        assert_eq!(regime.unwrap(), MarketRegime::HighVolatility);
    }

    #[test]
    fn test_impact_timeline_generation() {
        let model = AlmgrenChrissModel::new(AlmgrenChrissConfig::default());
        let parent_order = create_test_parent_order();
        let market_conditions = create_test_market_conditions();
        let execution_strategy = ImpactExecutionStrategy::Moderate;
        
        let result = model.calculate_impact(&parent_order, &market_conditions, &execution_strategy);
        assert!(result.is_ok());
        
        let impact = result.unwrap();
        assert!(!impact.impact_timeline.is_empty());
        
        // Verify timeline is monotonically increasing in cumulative quantity
        for i in 1..impact.impact_timeline.len() {
            assert!(impact.impact_timeline[i].cumulative_quantity >= 
                   impact.impact_timeline[i-1].cumulative_quantity);
        }
    }
}