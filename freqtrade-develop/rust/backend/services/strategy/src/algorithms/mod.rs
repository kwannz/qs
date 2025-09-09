use anyhow::Result;
use chrono::Utc;
use rust_decimal::Decimal;
use std::collections::HashMap;
use uuid::Uuid;

use crate::models::*;
use crate::services::market_data_client::MarketData;

#[derive(Debug)]
pub struct SignalBuilder {
    pub strategy_id: Uuid,
    pub symbol: String,
    pub exchange: String,
    pub signal_type: SignalType,
    pub action: SignalAction,
    pub strength: f64,
    pub confidence: f64,
    pub price: Decimal,
    pub quantity: Option<Decimal>,
    pub reason: String,
    pub factors: HashMap<String, f64>,
}

impl SignalBuilder {
    pub fn new(strategy_id: Uuid, symbol: String, exchange: String) -> Self {
        Self {
            strategy_id,
            symbol,
            exchange,
            signal_type: SignalType::Entry,
            action: SignalAction::Buy,
            strength: 0.0,
            confidence: 0.0,
            price: Decimal::ZERO,
            quantity: None,
            reason: String::new(),
            factors: HashMap::new(),
        }
    }

    pub fn signal_type(mut self, signal_type: SignalType) -> Self {
        self.signal_type = signal_type;
        self
    }

    pub fn action(mut self, action: SignalAction) -> Self {
        self.action = action;
        self
    }

    pub fn strength(mut self, strength: f64) -> Self {
        self.strength = strength;
        self
    }

    pub fn confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    pub fn price(mut self, price: Decimal) -> Self {
        self.price = price;
        self
    }

    pub fn quantity(mut self, quantity: Option<Decimal>) -> Self {
        self.quantity = quantity;
        self
    }

    pub fn reason(mut self, reason: String) -> Self {
        self.reason = reason;
        self
    }

    pub fn factors(mut self, factors: HashMap<String, f64>) -> Self {
        self.factors = factors;
        self
    }

    pub fn build(self) -> Signal {
        Signal {
            id: Uuid::new_v4(),
            strategy_id: self.strategy_id,
            symbol: self.symbol,
            exchange: self.exchange,
            signal_type: self.signal_type,
            action: self.action,
            strength: self.strength,
            confidence: self.confidence,
            price: self.price,
            quantity: self.quantity,
            reason: self.reason,
            factors: self.factors,
            created_at: Utc::now(),
            expires_at: Some(Utc::now() + chrono::Duration::minutes(15)), // 15-minute expiry
            executed: false,
            executed_at: None,
        }
    }
}
// Remove unused Ohlcv import

pub mod momentum;
pub mod mean_reversion;
pub mod arbitrage;
pub mod grid_trading;
pub mod dca;
pub mod pairs_trading;
pub mod ml_prediction;
pub mod factor_based;

pub use momentum::*;
pub use mean_reversion::*;
pub use arbitrage::*;
pub use grid_trading::*;
pub use dca::*;
pub use pairs_trading::*;
pub use ml_prediction::*;
pub use factor_based::*;

// Main algorithm execution interface
#[allow(dead_code)]
pub struct AlgorithmEngine {
    pub momentum: MomentumAlgorithm,
    pub mean_reversion: MeanReversionAlgorithm,
    pub arbitrage: ArbitrageAlgorithm,
    pub grid_trading: GridTradingAlgorithm,
    pub dca: DCAAlgorithm,
    pub pairs_trading: PairsTradingAlgorithm,
    pub ml_prediction: MLPredictionAlgorithm,
    pub factor_based: FactorBasedAlgorithm,
}

#[allow(dead_code)]
impl AlgorithmEngine {
    pub fn new() -> Self {
        Self {
            momentum: MomentumAlgorithm::new(),
            mean_reversion: MeanReversionAlgorithm::new(),
            arbitrage: ArbitrageAlgorithm::new(),
            grid_trading: GridTradingAlgorithm::new(),
            dca: DCAAlgorithm::new(),
            pairs_trading: PairsTradingAlgorithm::new(),
            ml_prediction: MLPredictionAlgorithm::new(),
            factor_based: FactorBasedAlgorithm::new(),
        }
    }

    // Execute algorithm based on strategy type
    pub async fn execute_algorithm(
        &self,
        strategy: &Strategy,
        market_data: &HashMap<String, MarketData>,
        indicators: &HashMap<String, HashMap<String, Indicator>>,
        factors: &HashMap<String, f64>,
    ) -> Result<Vec<Signal>> {
        match strategy.strategy_type {
            StrategyType::Momentum => {
                self.momentum.generate_signals(strategy, market_data, indicators).await
            }
            StrategyType::MeanReversion => {
                self.mean_reversion.generate_signals(strategy, market_data, indicators).await
            }
            StrategyType::Arbitrage => {
                self.arbitrage.generate_signals(strategy, market_data).await
            }
            StrategyType::GridTrading => {
                self.grid_trading.generate_signals(strategy, market_data).await
            }
            StrategyType::Dca => {
                self.dca.generate_signals(strategy, market_data).await
            }
            StrategyType::PairsTrading => {
                self.pairs_trading.generate_signals(strategy, market_data, indicators).await
            }
            StrategyType::MLPrediction => {
                self.ml_prediction.generate_signals(strategy, market_data, indicators, factors).await
            }
            StrategyType::FactorBased => {
                self.factor_based.generate_signals(strategy, market_data, factors).await
            }
            StrategyType::Custom => {
                // For custom strategies, use a combination of approaches based on parameters
                self.execute_custom_strategy(strategy, market_data, indicators, factors).await
            }
        }
    }

    // Execute custom strategy (hybrid approach)
    async fn execute_custom_strategy(
        &self,
        strategy: &Strategy,
        market_data: &HashMap<String, MarketData>,
        indicators: &HashMap<String, HashMap<String, Indicator>>,
        factors: &HashMap<String, f64>,
    ) -> Result<Vec<Signal>> {
        let mut signals = Vec::new();

        // Check custom parameters to determine which algorithms to use
        let custom_params = &strategy.parameters.custom_params;

        // Use momentum if momentum parameters are present
        if custom_params.contains_key("momentum_enabled") 
            && custom_params["momentum_enabled"].as_bool().unwrap_or(false) {
            let momentum_signals = self.momentum.generate_signals(strategy, market_data, indicators).await?;
            signals.extend(momentum_signals);
        }

        // Use mean reversion if parameters are present
        if custom_params.contains_key("mean_reversion_enabled")
            && custom_params["mean_reversion_enabled"].as_bool().unwrap_or(false) {
            let mr_signals = self.mean_reversion.generate_signals(strategy, market_data, indicators).await?;
            signals.extend(mr_signals);
        }

        // Use factor-based if factor weights are present
        if !factors.is_empty() && custom_params.contains_key("factor_based_enabled")
            && custom_params["factor_based_enabled"].as_bool().unwrap_or(false) {
            let factor_signals = self.factor_based.generate_signals(strategy, market_data, factors).await?;
            signals.extend(factor_signals);
        }

        // Use grid trading if grid parameters are present
        if strategy.parameters.grid_size.is_some() && strategy.parameters.grid_spacing_pct.is_some() {
            let grid_signals = self.grid_trading.generate_signals(strategy, market_data).await?;
            signals.extend(grid_signals);
        }

        // Use DCA if DCA parameters are present
        if strategy.parameters.dca_interval.is_some() && strategy.parameters.dca_amount.is_some() {
            let dca_signals = self.dca.generate_signals(strategy, market_data).await?;
            signals.extend(dca_signals);
        }

        // If no specific algorithms are enabled, use a default hybrid approach
        if signals.is_empty() {
            // Default to momentum + mean reversion combination
            let momentum_signals = self.momentum.generate_signals(strategy, market_data, indicators).await?;
            let mr_signals = self.mean_reversion.generate_signals(strategy, market_data, indicators).await?;
            
            signals.extend(momentum_signals);
            signals.extend(mr_signals);
            
            // Combine and filter conflicting signals
            signals = self.filter_conflicting_signals(signals);
        }

        Ok(signals)
    }

    // Filter out conflicting signals (e.g., buy and sell for same symbol at same time)
    fn filter_conflicting_signals(&self, mut signals: Vec<Signal>) -> Vec<Signal> {
        // Sort by symbol and timestamp
        signals.sort_by(|a, b| {
            a.symbol.cmp(&b.symbol)
                .then_with(|| a.created_at.cmp(&b.created_at))
        });

        let mut filtered_signals = Vec::new();
        let mut i = 0;

        while i < signals.len() {
            let current = &signals[i];
            let mut conflicting_signals = vec![current.clone()];
            
            // Find all signals for the same symbol within a short time window (1 minute)
            let mut j = i + 1;
            while j < signals.len() 
                && signals[j].symbol == current.symbol 
                && signals[j].created_at.signed_duration_since(current.created_at).num_minutes().abs() <= 1 {
                conflicting_signals.push(signals[j].clone());
                j += 1;
            }

            if conflicting_signals.len() == 1 {
                // No conflicts, add the signal
                filtered_signals.push(current.clone());
            } else {
                // Resolve conflicts by choosing the signal with highest confidence
                let Some(best_signal) = conflicting_signals
                    .into_iter()
                    .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal)) else {
                    continue; // Skip if no best signal found
                };
                
                filtered_signals.push(best_signal);
            }

            i = j;
        }

        filtered_signals
    }

    // Validate signal against risk parameters
    pub fn validate_signal(&self, signal: &Signal, strategy: &Strategy) -> bool {
        // Check confidence threshold
        if signal.confidence < strategy.parameters.custom_params
            .get("min_signal_confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) {
            return false;
        }

        // Check strength threshold
        if signal.strength < strategy.parameters.custom_params
            .get("min_signal_strength")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.3) {
            return false;
        }

        // Check position size limits
        if let Some(quantity) = signal.quantity {
            if quantity > strategy.parameters.max_position_size {
                return false;
            }
        }

        // Check risk tolerance
        let signal_risk = signal.strength * (1.0 - signal.confidence);
        if signal_risk > strategy.parameters.risk_tolerance {
            return false;
        }

        true
    }

    // Calculate position size based on strategy parameters and market conditions
    pub fn calculate_position_size(
        &self,
        strategy: &Strategy,
        signal: &Signal,
        current_portfolio_value: Decimal,
    ) -> Decimal {
        let risk_per_trade = strategy.parameters.risk_tolerance;
        let max_position_size = strategy.parameters.max_position_size;
        
        // Calculate position size based on Kelly Criterion approximation
        let win_probability = signal.confidence;
        let risk_reward_ratio = strategy.parameters.take_profit_pct / strategy.parameters.stop_loss_pct;
        
        // Kelly fraction = (bp - q) / b
        // where b = risk/reward ratio, p = win probability, q = loss probability
        let kelly_fraction = (win_probability * risk_reward_ratio - (1.0 - win_probability)) / risk_reward_ratio;
        let kelly_fraction = kelly_fraction.clamp(0.0, 0.25); // Cap at 25% of portfolio
        
        // Calculate position size
        let portfolio_risk = current_portfolio_value * Decimal::from_f64_retain(risk_per_trade).unwrap_or_default();
        let kelly_position_size = current_portfolio_value * Decimal::from_f64_retain(kelly_fraction).unwrap_or_default();
        
        // Use the smaller of Kelly size and risk-based size
        let position_size = portfolio_risk.min(kelly_position_size);
        
        // Cap at maximum position size
        position_size.min(max_position_size)
    }
}

impl Default for AlgorithmEngine {
    fn default() -> Self {
        Self::new()
    }
}

// Common helper functions for all algorithms

#[allow(dead_code)]
pub fn calculate_price_change_percent(current: f64, previous: f64) -> f64 {
    if previous != 0.0 {
        ((current - previous) / previous) * 100.0
    } else {
        0.0
    }
}

#[allow(dead_code)]
pub fn calculate_volatility(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }
    
    let mean = prices.iter().sum::<f64>() / prices.len() as f64;
    let variance = prices.iter()
        .map(|price| (price - mean).powi(2))
        .sum::<f64>() / prices.len() as f64;
    
    variance.sqrt()
}

#[allow(dead_code)]
pub fn calculate_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    
    let n = x.len() as f64;
    let sum_x = x.iter().sum::<f64>();
    let sum_y = y.iter().sum::<f64>();
    let sum_xy = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum::<f64>();
    let sum_x2 = x.iter().map(|xi| xi * xi).sum::<f64>();
    let sum_y2 = y.iter().map(|yi| yi * yi).sum::<f64>();
    
    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
    
    if denominator != 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

#[allow(dead_code)]
pub fn create_signal(strategy_id: Uuid, symbol: String, exchange: String) -> SignalBuilder {
    SignalBuilder::new(strategy_id, symbol, exchange)
}