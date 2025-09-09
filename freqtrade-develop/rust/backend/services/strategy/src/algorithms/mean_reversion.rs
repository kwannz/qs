use anyhow::Result;
use rust_decimal::prelude::ToPrimitive;
use std::collections::HashMap;

use crate::models::*;
use crate::services::market_data_client::MarketData;
use super::create_signal;
// Remove unused imports

#[allow(dead_code)]
pub struct MeanReversionAlgorithm;

#[allow(dead_code)]
impl MeanReversionAlgorithm {
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
                    // Generate mean reversion signals
                    if let Some(signal) = self.analyze_mean_reversion_indicators(
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

    async fn analyze_mean_reversion_indicators(
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

        // RSI Oversold/Overbought Analysis
        if let Some(rsi) = indicators.get("rsi") {
            let rsi_signal = self.analyze_rsi_mean_reversion(rsi, strategy);
            signal_strength += rsi_signal.strength * 0.4; // 40% weight - high importance for mean reversion
            confidence_factors.push(rsi_signal.confidence);
            reason_parts.push(rsi_signal.reason);
            factors.insert("rsi_mean_reversion".to_string(), rsi_signal.strength);
        }

        // Bollinger Bands Analysis
        if let (Some(bb_upper), Some(bb_lower)) = 
           (indicators.get("bb_upper"), indicators.get("bb_lower")) {
            let bb_signal = self.analyze_bollinger_bands_mean_reversion(data, bb_upper, bb_lower);
            signal_strength += bb_signal.strength * 0.3; // 30% weight
            confidence_factors.push(bb_signal.confidence);
            reason_parts.push(bb_signal.reason);
            factors.insert("bb_mean_reversion".to_string(), bb_signal.strength);
        }

        // Stochastic Oscillator Analysis
        if let (Some(stoch_k), Some(stoch_d)) = (indicators.get("stoch_k"), indicators.get("stoch_d")) {
            let stoch_signal = self.analyze_stochastic_mean_reversion(stoch_k, stoch_d);
            signal_strength += stoch_signal.strength * 0.2; // 20% weight
            confidence_factors.push(stoch_signal.confidence);
            reason_parts.push(stoch_signal.reason);
            factors.insert("stoch_mean_reversion".to_string(), stoch_signal.strength);
        }

        // Price deviation from moving averages
        if let Some(sma_20) = indicators.get("sma_20") {
            let price_deviation_signal = self.analyze_price_deviation_from_mean(data, sma_20);
            signal_strength += price_deviation_signal.strength * 0.1; // 10% weight
            confidence_factors.push(price_deviation_signal.confidence);
            reason_parts.push(price_deviation_signal.reason);
            factors.insert("price_deviation".to_string(), price_deviation_signal.strength);
        }

        // Calculate overall confidence
        let confidence = if !confidence_factors.is_empty() {
            confidence_factors.iter().sum::<f64>() / confidence_factors.len() as f64
        } else {
            0.0
        };

        // Determine signal action (mean reversion logic - buy when oversold, sell when overbought)
        let (action, signal_type) = if signal_strength > 0.3 {
            (SignalAction::Buy, SignalType::Entry) // Oversold condition - buy
        } else if signal_strength < -0.3 {
            (SignalAction::Sell, SignalType::Entry) // Overbought condition - sell
        } else {
            return Ok(None); // No clear mean reversion signal
        };

        // Apply mean reversion specific filters
        if !self.mean_reversion_filter_passed(signal_strength, confidence, strategy, data) {
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

    fn analyze_rsi_mean_reversion(&self, rsi: &Indicator, strategy: &Strategy) -> MeanReversionSignalAnalysis {
        let rsi_value = rsi.value;
        let rsi_change = rsi.change;
        
        // Get custom thresholds from strategy parameters
        let oversold_threshold = strategy.parameters.rsi_oversold.unwrap_or(30.0);
        let overbought_threshold = strategy.parameters.rsi_overbought.unwrap_or(70.0);

        let (strength, confidence, reason) = if rsi_value <= oversold_threshold {
            // Oversold condition - expect mean reversion upward
            let oversold_intensity = (oversold_threshold - rsi_value) / oversold_threshold;
            let strength = oversold_intensity.min(1.0) * 0.9; // Positive strength for buy signal
            let confidence = if rsi_value < 20.0 { 0.9 } else { 0.7 };
            (strength, confidence, format!("RSI oversold mean reversion: {rsi_value:.1} (threshold: {oversold_threshold:.1})"))
        } else if rsi_value >= overbought_threshold {
            // Overbought condition - expect mean reversion downward
            let overbought_intensity = (rsi_value - overbought_threshold) / (100.0 - overbought_threshold);
            let strength = -(overbought_intensity.min(1.0) * 0.9); // Negative strength for sell signal
            let confidence = if rsi_value > 80.0 { 0.9 } else { 0.7 };
            (strength, confidence, format!("RSI overbought mean reversion: {rsi_value:.1} (threshold: {overbought_threshold:.1})"))
        } else if rsi_value > oversold_threshold + 5.0 && rsi_value < oversold_threshold + 15.0 && rsi_change > 1.0 {
            // Recently recovering from oversold - potential continuation
            let recovery_strength = rsi_change / 10.0;
            (recovery_strength * 0.5, 0.6, format!("RSI recovering from oversold: {rsi_value:.1} (+{rsi_change:.1})"))
        } else if rsi_value < overbought_threshold - 5.0 && rsi_value > overbought_threshold - 15.0 && rsi_change < -1.0 {
            // Recently declining from overbought - potential continuation
            let decline_strength = rsi_change.abs() / 10.0;
            (-decline_strength * 0.5, 0.6, format!("RSI declining from overbought: {rsi_value:.1} ({rsi_change:.1})"))
        } else {
            (0.0, 0.3, format!("RSI in neutral zone: {rsi_value:.1}"))
        };

        MeanReversionSignalAnalysis {
            strength,
            confidence,
            reason,
        }
    }

    fn analyze_bollinger_bands_mean_reversion(
        &self,
        data: &MarketData,
        bb_upper: &Indicator,
        bb_lower: &Indicator,
    ) -> MeanReversionSignalAnalysis {
        let current_price = data.price.to_f64().unwrap_or(0.0);
        let upper_band = bb_upper.value;
        let lower_band = bb_lower.value;
        let band_width = upper_band - lower_band;

        let (strength, confidence, reason) = if current_price <= lower_band {
            // Price at or below lower band - oversold, expect reversion up
            let deviation_pct = ((lower_band - current_price) / band_width) * 100.0;
            let strength = (deviation_pct / 10.0).min(1.0); // Scale to max 1.0
            let confidence = if deviation_pct > 5.0 { 0.8 } else { 0.6 };
            (strength, confidence, format!("Price below BB lower: {current_price:.2} < {lower_band:.2} ({deviation_pct:.1}% deviation)"))
        } else if current_price >= upper_band {
            // Price at or above upper band - overbought, expect reversion down
            let deviation_pct = ((current_price - upper_band) / band_width) * 100.0;
            let strength = -(deviation_pct / 10.0).min(1.0);
            let confidence = if deviation_pct > 5.0 { 0.8 } else { 0.6 };
            (strength, confidence, format!("Price above BB upper: {current_price:.2} > {upper_band:.2} ({deviation_pct:.1}% deviation)"))
        } else {
            // Price within bands
            let _middle = (upper_band + lower_band) / 2.0;
            let position_in_band = (current_price - lower_band) / band_width;
            
            if position_in_band < 0.2 {
                // Close to lower band
                let strength = (0.2 - position_in_band) * 2.0; // Scale to 0-0.4
                (strength, 0.5, format!("Price near BB lower: {:.1}% of band width", position_in_band * 100.0))
            } else if position_in_band > 0.8 {
                // Close to upper band
                let strength = -((position_in_band - 0.8) * 2.0);
                (strength, 0.5, format!("Price near BB upper: {:.1}% of band width", position_in_band * 100.0))
            } else {
                (0.0, 0.3, "Price in BB middle range".to_string())
            }
        };

        MeanReversionSignalAnalysis {
            strength,
            confidence,
            reason,
        }
    }

    fn analyze_stochastic_mean_reversion(
        &self,
        stoch_k: &Indicator,
        stoch_d: &Indicator,
    ) -> MeanReversionSignalAnalysis {
        let k_value = stoch_k.value;
        let d_value = stoch_d.value;

        let (strength, confidence, reason) = if k_value <= 20.0 && d_value <= 20.0 {
            // Both %K and %D in oversold territory - strong mean reversion signal
            let oversold_intensity = (20.0 - k_value.min(d_value)) / 20.0;
            let strength = oversold_intensity * 0.8;
            let confidence = 0.8;
            (strength, confidence, format!("Stochastic oversold: K={k_value:.1}, D={d_value:.1}"))
        } else if k_value >= 80.0 && d_value >= 80.0 {
            // Both %K and %D in overbought territory
            let overbought_intensity = (k_value.max(d_value) - 80.0) / 20.0;
            let strength = -overbought_intensity * 0.8;
            let confidence = 0.8;
            (strength, confidence, format!("Stochastic overbought: K={k_value:.1}, D={d_value:.1}"))
        } else if k_value < 30.0 && k_value > d_value && (k_value - d_value) > 2.0 {
            // %K crossing above %D in oversold area - bullish divergence
            let crossover_strength = (k_value - d_value) / 10.0;
            (crossover_strength * 0.6, 0.7, format!("Stochastic bullish crossover: K={k_value:.1} > D={d_value:.1}"))
        } else if k_value > 70.0 && k_value < d_value && (d_value - k_value) > 2.0 {
            // %K crossing below %D in overbought area - bearish divergence
            let crossover_strength = (d_value - k_value) / 10.0;
            (-crossover_strength * 0.6, 0.7, format!("Stochastic bearish crossover: K={k_value:.1} < D={d_value:.1}"))
        } else {
            (0.0, 0.4, format!("Stochastic neutral: K={k_value:.1}, D={d_value:.1}"))
        };

        MeanReversionSignalAnalysis {
            strength,
            confidence,
            reason,
        }
    }

    fn analyze_price_deviation_from_mean(&self, data: &MarketData, sma: &Indicator) -> MeanReversionSignalAnalysis {
        let current_price = data.price.to_f64().unwrap_or(0.0);
        let sma_value = sma.value;
        let deviation_pct = ((current_price - sma_value) / sma_value) * 100.0;

        let (strength, confidence, reason) = if deviation_pct < -3.0 {
            // Price significantly below SMA - potential mean reversion up
            let strength = (deviation_pct.abs() / 10.0).min(0.6); // Cap at 0.6
            let confidence = if deviation_pct < -5.0 { 0.6 } else { 0.4 };
            (strength, confidence, format!("Price below SMA: {deviation_pct:.1}% deviation"))
        } else if deviation_pct > 3.0 {
            // Price significantly above SMA - potential mean reversion down
            let strength = -(deviation_pct / 10.0).min(0.6);
            let confidence = if deviation_pct > 5.0 { 0.6 } else { 0.4 };
            (strength, confidence, format!("Price above SMA: {deviation_pct:.1}% deviation"))
        } else {
            (0.0, 0.2, format!("Price near SMA: {deviation_pct:.1}% deviation"))
        };

        MeanReversionSignalAnalysis {
            strength,
            confidence,
            reason,
        }
    }

    fn mean_reversion_filter_passed(
        &self, 
        signal_strength: f64, 
        confidence: f64, 
        strategy: &Strategy, 
        data: &MarketData
    ) -> bool {
        // Check minimum signal strength
        let min_strength = strategy.parameters.custom_params
            .get("min_mean_reversion_strength")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.3);

        if signal_strength.abs() < min_strength {
            return false;
        }

        // Check confidence threshold
        let min_confidence = strategy.parameters.custom_params
            .get("min_mean_reversion_confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.6);

        if confidence < min_confidence {
            return false;
        }

        // Volume filter - mean reversion works better with adequate volume
        let _min_volume_ratio = strategy.parameters.custom_params
            .get("min_volume_ratio")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        // Simple volume check (in real implementation, we'd compare to average volume)
        let current_volume = data.volume.to_f64().unwrap_or(0.0);
        if current_volume < 1000.0 { // Minimum absolute volume threshold
            return false;
        }

        // Volatility filter - avoid mean reversion in extremely volatile conditions
        let max_24h_change = strategy.parameters.custom_params
            .get("max_24h_change_for_mean_reversion")
            .and_then(|v| v.as_f64())
            .unwrap_or(15.0);

        if let Some(change_24h_pct) = data.change_24h_pct {
            if change_24h_pct.abs() > max_24h_change {
                return false; // Too volatile for mean reversion
            }
        }

        true
    }
}

impl Default for MeanReversionAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MeanReversionSignalAnalysis {
    strength: f64,
    confidence: f64,
    reason: String,
}