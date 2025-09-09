use anyhow::Result;
use rust_decimal::prelude::ToPrimitive;
use std::collections::HashMap;

use crate::models::*;
use crate::services::market_data_client::MarketData;
use super::create_signal;
// Remove unused imports

#[allow(dead_code)]
pub struct MomentumAlgorithm;

#[allow(dead_code)]
impl MomentumAlgorithm {
    pub fn new() -> Self {
        Self
    }

    pub async fn generate_signals(
        &self,
        strategy: &Strategy,
        market_data: &HashMap<String, MarketData>,
        indicators: &HashMap<String, HashMap<String, Indicator>>,
    ) -> Result<Vec<Signal>> {
        let mut signals = Vec::new();

        for symbol in &strategy.symbols {
            if let Some(data) = market_data.get(symbol) {
                if let Some(symbol_indicators) = indicators.get(symbol) {
                    // Generate momentum-based signals
                    if let Some(signal) = self.analyze_momentum_indicators(
                        strategy,
                        symbol,
                        data,
                        symbol_indicators,
                    ).await? {
                        signals.push(signal);
                    }
                }
            }
        }

        Ok(signals)
    }

    async fn analyze_momentum_indicators(
        &self,
        strategy: &Strategy,
        symbol: &str,
        data: &MarketData,
        indicators: &HashMap<String, Indicator>,
    ) -> Result<Option<Signal>> {
        let mut signal_strength = 0.0;
        let mut confidence_factors = Vec::new();
        let mut reason_parts = Vec::new();
        let mut factors = HashMap::new();

        // RSI Analysis
        if let Some(rsi) = indicators.get("rsi") {
            let rsi_signal = self.analyze_rsi_momentum(rsi);
            signal_strength += rsi_signal.strength * 0.3; // 30% weight
            confidence_factors.push(rsi_signal.confidence);
            reason_parts.push(rsi_signal.reason);
            factors.insert("rsi_momentum".to_string(), rsi_signal.strength);
        }

        // MACD Analysis
        if let (Some(macd_line), Some(macd_signal)) = 
           (indicators.get("macd_line"), indicators.get("macd_signal")) {
            let macd_signal_analysis = self.analyze_macd_momentum(macd_line, macd_signal);
            signal_strength += macd_signal_analysis.strength * 0.25; // 25% weight
            confidence_factors.push(macd_signal_analysis.confidence);
            reason_parts.push(macd_signal_analysis.reason);
            factors.insert("macd_momentum".to_string(), macd_signal_analysis.strength);
        }

        // Moving Average Analysis
        if let (Some(ema_12), Some(ema_26)) = (indicators.get("ema_12"), indicators.get("ema_26")) {
            let ma_signal = self.analyze_moving_average_momentum(data, ema_12, ema_26);
            signal_strength += ma_signal.strength * 0.25; // 25% weight
            confidence_factors.push(ma_signal.confidence);
            reason_parts.push(ma_signal.reason);
            factors.insert("ma_momentum".to_string(), ma_signal.strength);
        }

        // Price momentum analysis
        let price_momentum = self.analyze_price_momentum(data);
        signal_strength += price_momentum.strength * 0.2; // 20% weight
        confidence_factors.push(price_momentum.confidence);
        reason_parts.push(price_momentum.reason);
        factors.insert("price_momentum".to_string(), price_momentum.strength);

        // Calculate overall confidence
        let confidence = if !confidence_factors.is_empty() {
            confidence_factors.iter().sum::<f64>() / confidence_factors.len() as f64
        } else {
            0.0
        };

        // Determine signal action
        let (action, signal_type) = if signal_strength > 0.3 {
            (SignalAction::Buy, SignalType::Entry)
        } else if signal_strength < -0.3 {
            (SignalAction::Sell, SignalType::Entry)
        } else {
            return Ok(None); // No clear signal
        };

        // Apply momentum-specific filters
        if !self.momentum_filter_passed(signal_strength, confidence, strategy) {
            return Ok(None);
        }

        let signal = create_signal(
            strategy.id,
            symbol.to_string(),
            data.exchange.clone()
        )
        .signal_type(signal_type)
        .action(action)
        .strength(signal_strength.abs())
        .confidence(confidence)
        .price(data.price)
        .quantity(None) // Position size will be calculated later
        .reason(reason_parts.join("; "))
        .factors(factors)
        .build();

        Ok(Some(signal))
    }

    fn analyze_rsi_momentum(&self, rsi: &Indicator) -> MomentumSignalAnalysis {
        let rsi_value = rsi.value;
        let rsi_change = rsi.change;

        let (strength, confidence, reason) = if rsi_value > 50.0 && rsi_change > 0.0 {
            // Rising RSI above 50 - bullish momentum
            let strength = ((rsi_value - 50.0) / 50.0) * 0.8; // Scale to 0-0.8
            let confidence = if rsi_value > 60.0 { 0.8 } else { 0.6 };
            (strength, confidence, format!("RSI bullish momentum: {rsi_value:.1} (+{rsi_change:.1})"))
        } else if rsi_value < 50.0 && rsi_change < 0.0 {
            // Falling RSI below 50 - bearish momentum
            let strength = -((50.0 - rsi_value) / 50.0) * 0.8;
            let confidence = if rsi_value < 40.0 { 0.8 } else { 0.6 };
            (strength, confidence, format!("RSI bearish momentum: {rsi_value:.1} ({rsi_change:.1})"))
        } else {
            // Neutral or conflicting signals
            (0.0, 0.3, format!("RSI neutral: {rsi_value:.1}"))
        };

        MomentumSignalAnalysis {
            strength,
            confidence,
            reason,
        }
    }

    fn analyze_macd_momentum(&self, macd_line: &Indicator, macd_signal: &Indicator) -> MomentumSignalAnalysis {
        let macd_value = macd_line.value;
        let signal_value = macd_signal.value;
        let macd_change = macd_line.change;

        let (strength, confidence, reason) = if macd_value > signal_value && macd_change > 0.0 {
            // MACD above signal line and rising - bullish momentum
            let spread = macd_value - signal_value;
            let strength = (spread * 100.0).clamp(0.0, 1.0); // Scale appropriately
            let confidence = if spread > 0.01 { 0.8 } else { 0.6 };
            (strength, confidence, format!("MACD bullish crossover: {macd_value:.4} > {signal_value:.4}"))
        } else if macd_value < signal_value && macd_change < 0.0 {
            // MACD below signal line and falling - bearish momentum
            let spread = signal_value - macd_value;
            let strength = -(spread * 100.0).clamp(0.0, 1.0);
            let confidence = if spread > 0.01 { 0.8 } else { 0.6 };
            (strength, confidence, format!("MACD bearish crossover: {macd_value:.4} < {signal_value:.4}"))
        } else {
            (0.0, 0.4, format!("MACD neutral: {macd_value:.4}"))
        };

        MomentumSignalAnalysis {
            strength,
            confidence,
            reason,
        }
    }

    fn analyze_moving_average_momentum(
        &self,
        data: &MarketData,
        ema_12: &Indicator,
        ema_26: &Indicator,
    ) -> MomentumSignalAnalysis {
        let current_price = data.price.to_f64().unwrap_or(0.0);
        let ema_12_value = ema_12.value;
        let ema_26_value = ema_26.value;

        let (strength, confidence, reason) = if ema_12_value > ema_26_value && current_price > ema_12_value {
            // Golden cross with price above EMAs - strong bullish momentum
            let ema_spread_pct = ((ema_12_value - ema_26_value) / ema_26_value) * 100.0;
            let price_above_pct = ((current_price - ema_12_value) / ema_12_value) * 100.0;
            let strength = (ema_spread_pct * 0.5 + price_above_pct * 0.5).clamp(0.0, 1.0);
            let confidence = if ema_spread_pct > 1.0 && price_above_pct > 0.5 { 0.9 } else { 0.7 };
            (strength, confidence, format!("Bullish EMA alignment: Price {current_price:.2} > EMA12 {ema_12_value:.2} > EMA26 {ema_26_value:.2}"))
        } else if ema_12_value < ema_26_value && current_price < ema_12_value {
            // Death cross with price below EMAs - strong bearish momentum
            let ema_spread_pct = ((ema_26_value - ema_12_value) / ema_26_value) * 100.0;
            let price_below_pct = ((ema_12_value - current_price) / ema_12_value) * 100.0;
            let strength = -(ema_spread_pct * 0.5 + price_below_pct * 0.5).clamp(0.0, 1.0);
            let confidence = if ema_spread_pct > 1.0 && price_below_pct > 0.5 { 0.9 } else { 0.7 };
            (strength, confidence, format!("Bearish EMA alignment: Price {current_price:.2} < EMA12 {ema_12_value:.2} < EMA26 {ema_26_value:.2}"))
        } else {
            (0.0, 0.5, "EMA neutral alignment".to_string())
        };

        MomentumSignalAnalysis {
            strength,
            confidence,
            reason,
        }
    }

    fn analyze_price_momentum(&self, data: &MarketData) -> MomentumSignalAnalysis {
        let _current_price = data.price.to_f64().unwrap_or(0.0);
        let change_24h_pct = data.change_24h_pct.unwrap_or(0.0);

        let (strength, confidence, reason) = if change_24h_pct > 2.0 {
            // Strong positive price momentum
            let strength = (change_24h_pct / 20.0).min(1.0); // Scale to max 1.0
            let confidence = if change_24h_pct > 5.0 { 0.8 } else { 0.6 };
            (strength, confidence, format!("Strong price momentum: +{change_24h_pct:.1}% in 24h"))
        } else if change_24h_pct < -2.0 {
            // Strong negative price momentum
            let strength = -(change_24h_pct.abs() / 20.0).min(1.0);
            let confidence = if change_24h_pct < -5.0 { 0.8 } else { 0.6 };
            (strength, confidence, format!("Strong negative momentum: {change_24h_pct:.1}% in 24h"))
        } else {
            (0.0, 0.4, format!("Weak price momentum: {change_24h_pct:.1}%"))
        };

        MomentumSignalAnalysis {
            strength,
            confidence,
            reason,
        }
    }

    fn momentum_filter_passed(&self, signal_strength: f64, confidence: f64, strategy: &Strategy) -> bool {
        // Check minimum momentum threshold
        let min_momentum = strategy.parameters.custom_params
            .get("min_momentum_strength")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.3);

        if signal_strength.abs() < min_momentum {
            return false;
        }

        // Check confidence threshold
        let min_confidence = strategy.parameters.custom_params
            .get("min_momentum_confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.6);

        if confidence < min_confidence {
            return false;
        }

        // Check lookback period alignment (if specified)
        if let Some(lookback) = strategy.parameters.momentum_lookback {
            // In a real implementation, we would check if momentum persists over the lookback period
            // For now, we just ensure it's a reasonable value
            if !(1..=100).contains(&lookback) {
                return false;
            }
        }

        true
    }
}

impl Default for MomentumAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MomentumSignalAnalysis {
    strength: f64,
    confidence: f64,
    reason: String,
}