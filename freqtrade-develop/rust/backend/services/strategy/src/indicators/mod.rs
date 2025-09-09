use anyhow::Result;
use chrono::Utc;
use std::collections::HashMap;

use crate::models::{Indicator, IndicatorSignal};
use crate::services::market_data_client::Ohlcv;

pub mod technical;
pub mod volatility;
pub mod momentum;
pub mod trend;

pub use technical::*;
pub use volatility::*;
pub use momentum::*;
pub use trend::*;

// Main indicator calculation interface
// Development placeholder
#[allow(dead_code)]
pub struct IndicatorCalculator {
    pub technical: TechnicalIndicators,
    pub volatility: VolatilityIndicators,
    pub momentum: MomentumIndicators,
    pub trend: TrendIndicators,
}

#[allow(dead_code)]
impl IndicatorCalculator {
    pub fn new() -> Self {
        Self {
            technical: TechnicalIndicators::new(),
            volatility: VolatilityIndicators::new(),
            momentum: MomentumIndicators::new(),
            trend: TrendIndicators::new(),
        }
    }

    // Calculate all indicators for a symbol
    #[allow(dead_code)]
    pub async fn calculate_all_indicators(
        &self,
        symbol: &str,
        timeframe: &str,
        data: &[Ohlcv],
    ) -> Result<HashMap<String, Indicator>> {
        let mut indicators = HashMap::new();

        if data.len() < 20 {
            return Ok(indicators); // Need at least 20 periods for most indicators
        }

        // Technical indicators
        if let Ok(sma_20) = self.technical.sma(data, 20) {
            indicators.insert("sma_20".to_string(), self.create_indicator(
                "sma_20", symbol, timeframe, sma_20, data
            ));
        }

        if let Ok(sma_50) = self.technical.sma(data, 50) {
            indicators.insert("sma_50".to_string(), self.create_indicator(
                "sma_50", symbol, timeframe, sma_50, data
            ));
        }

        if let Ok(ema_12) = self.technical.ema(data, 12) {
            indicators.insert("ema_12".to_string(), self.create_indicator(
                "ema_12", symbol, timeframe, ema_12, data
            ));
        }

        if let Ok(ema_26) = self.technical.ema(data, 26) {
            indicators.insert("ema_26".to_string(), self.create_indicator(
                "ema_26", symbol, timeframe, ema_26, data
            ));
        }

        // Volatility indicators
        if let Ok(bb_upper) = self.volatility.bollinger_bands_upper(data, 20, 2.0) {
            indicators.insert("bb_upper".to_string(), self.create_indicator(
                "bb_upper", symbol, timeframe, bb_upper, data
            ));
        }

        if let Ok(bb_lower) = self.volatility.bollinger_bands_lower(data, 20, 2.0) {
            indicators.insert("bb_lower".to_string(), self.create_indicator(
                "bb_lower", symbol, timeframe, bb_lower, data
            ));
        }

        if let Ok(atr) = self.volatility.atr(data, 14) {
            indicators.insert("atr".to_string(), self.create_indicator(
                "atr", symbol, timeframe, atr, data
            ));
        }

        // Momentum indicators
        if let Ok(rsi) = self.momentum.rsi(data, 14) {
            indicators.insert("rsi".to_string(), self.create_indicator(
                "rsi", symbol, timeframe, rsi, data
            ));
        }

        if let Ok(macd_line) = self.momentum.macd_line(data, 12, 26) {
            indicators.insert("macd_line".to_string(), self.create_indicator(
                "macd_line", symbol, timeframe, macd_line, data
            ));
        }

        if let Ok(macd_signal) = self.momentum.macd_signal(data, 12, 26, 9) {
            indicators.insert("macd_signal".to_string(), self.create_indicator(
                "macd_signal", symbol, timeframe, macd_signal, data
            ));
        }

        if let Ok(stoch_k) = self.momentum.stochastic_k(data, 14) {
            indicators.insert("stoch_k".to_string(), self.create_indicator(
                "stoch_k", symbol, timeframe, stoch_k, data
            ));
        }

        if let Ok(stoch_d) = self.momentum.stochastic_d(data, 14, 3) {
            indicators.insert("stoch_d".to_string(), self.create_indicator(
                "stoch_d", symbol, timeframe, stoch_d, data
            ));
        }

        // Trend indicators
        if let Ok(adx) = self.trend.adx(data, 14) {
            indicators.insert("adx".to_string(), self.create_indicator(
                "adx", symbol, timeframe, adx, data
            ));
        }

        if let Ok(cci) = self.trend.cci(data, 20) {
            indicators.insert("cci".to_string(), self.create_indicator(
                "cci", symbol, timeframe, cci, data
            ));
        }

        Ok(indicators)
    }

    // Helper method to create Indicator struct
    #[allow(dead_code)]
    fn create_indicator(
        &self,
        name: &str,
        symbol: &str,
        timeframe: &str,
        values: Vec<f64>,
        _data: &[Ohlcv],
    ) -> Indicator {
        let current_value = values.last().copied().unwrap_or(0.0);
        let previous_value = if values.len() > 1 {
            values[values.len() - 2]
        } else {
            current_value
        };
        
        let change = current_value - previous_value;
        let change_pct = if previous_value != 0.0 {
            (change / previous_value) * 100.0
        } else {
            0.0
        };

        let signal = self.determine_signal(name, current_value, &values);
        let confidence = self.calculate_confidence(name, current_value, &values);

        let mut parameters = HashMap::new();
        match name {
            "sma_20" | "sma_50" => {
                parameters.insert("period".to_string(), name.split('_').nth(1).unwrap_or("20").parse().unwrap_or(20.0));
            }
            "ema_12" | "ema_26" => {
                parameters.insert("period".to_string(), name.split('_').nth(1).unwrap_or("12").parse().unwrap_or(12.0));
            }
            "bb_upper" | "bb_lower" => {
                parameters.insert("period".to_string(), 20.0);
                parameters.insert("std_dev".to_string(), 2.0);
            }
            "rsi" => {
                parameters.insert("period".to_string(), 14.0);
                parameters.insert("overbought".to_string(), 70.0);
                parameters.insert("oversold".to_string(), 30.0);
            }
            "macd_line" => {
                parameters.insert("fast_period".to_string(), 12.0);
                parameters.insert("slow_period".to_string(), 26.0);
            }
            "macd_signal" => {
                parameters.insert("fast_period".to_string(), 12.0);
                parameters.insert("slow_period".to_string(), 26.0);
                parameters.insert("signal_period".to_string(), 9.0);
            }
            "stoch_k" | "stoch_d" => {
                parameters.insert("k_period".to_string(), 14.0);
                parameters.insert("d_period".to_string(), 3.0);
            }
            "atr" => {
                parameters.insert("period".to_string(), 14.0);
            }
            "adx" => {
                parameters.insert("period".to_string(), 14.0);
            }
            "cci" => {
                parameters.insert("period".to_string(), 20.0);
            }
            _ => {}
        }

        Indicator {
            name: name.to_string(),
            symbol: symbol.to_string(),
            timeframe: timeframe.to_string(),
            value: current_value,
            previous_value,
            change,
            change_pct,
            signal,
            confidence,
            calculated_at: Utc::now(),
            parameters,
        }
    }

    // Determine signal based on indicator type and value
    #[allow(dead_code)]
    fn determine_signal(&self, name: &str, current_value: f64, values: &[f64]) -> IndicatorSignal {
        match name {
            "rsi" => {
                if current_value > 70.0 {
                    IndicatorSignal::StrongSell
                } else if current_value > 60.0 {
                    IndicatorSignal::Sell
                } else if current_value < 30.0 {
                    IndicatorSignal::StrongBuy
                } else if current_value < 40.0 {
                    IndicatorSignal::Buy
                } else {
                    IndicatorSignal::Neutral
                }
            }
            "stoch_k" | "stoch_d" => {
                if current_value > 80.0 {
                    IndicatorSignal::Sell
                } else if current_value < 20.0 {
                    IndicatorSignal::Buy
                } else {
                    IndicatorSignal::Neutral
                }
            }
            "macd_line" => {
                if values.len() >= 2 {
                    let prev_value = values[values.len() - 2];
                    if current_value > 0.0 && prev_value <= 0.0 {
                        IndicatorSignal::Buy
                    } else if current_value < 0.0 && prev_value >= 0.0 {
                        IndicatorSignal::Sell
                    } else {
                        IndicatorSignal::Neutral
                    }
                } else {
                    IndicatorSignal::Neutral
                }
            }
            "adx" => {
                if current_value > 25.0 {
                    // Strong trend, but direction needs to be determined from +DI/-DI
                    IndicatorSignal::Neutral // Simplified
                } else {
                    IndicatorSignal::Neutral
                }
            }
            "cci" => {
                if current_value > 100.0 {
                    IndicatorSignal::Sell
                } else if current_value < -100.0 {
                    IndicatorSignal::Buy
                } else {
                    IndicatorSignal::Neutral
                }
            }
            _ => IndicatorSignal::Neutral,
        }
    }

    // Calculate confidence based on indicator strength
    #[allow(dead_code)]
    fn calculate_confidence(&self, name: &str, current_value: f64, values: &[f64]) -> f64 {
        match name {
            "rsi" => {
                // Higher confidence when RSI is in extreme zones
                if !(30.0..=70.0).contains(&current_value) {
                    0.8
                } else if !(40.0..=60.0).contains(&current_value) {
                    0.6
                } else {
                    0.3
                }
            }
            "stoch_k" | "stoch_d" => {
                if !(20.0..=80.0).contains(&current_value) {
                    0.7
                } else {
                    0.4
                }
            }
            "macd_line" => {
                // Higher confidence for stronger crossovers
                let abs_value = current_value.abs();
                (abs_value * 100.0).clamp(0.3, 0.9)
            }
            "adx" => {
                // Higher confidence for stronger trends
                if current_value > 40.0 {
                    0.9
                } else if current_value > 25.0 {
                    0.7
                } else {
                    0.3
                }
            }
            "cci" => {
                let abs_value = current_value.abs();
                if abs_value > 200.0 {
                    0.8
                } else if abs_value > 100.0 {
                    0.6
                } else {
                    0.3
                }
            }
            "atr" => {
                // ATR doesn't give directional signals, so confidence is about data quality
                if values.len() > 14 {
                    0.7
                } else {
                    0.4
                }
            }
            _ => 0.5, // Default confidence for other indicators
        }
    }
}

impl Default for IndicatorCalculator {
    fn default() -> Self {
        Self::new()
    }
}