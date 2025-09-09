use anyhow::Result;
use rust_decimal::prelude::ToPrimitive;
// Remove unused import
use crate::services::market_data_client::Ohlcv;

#[allow(dead_code)]
pub struct MomentumIndicators;

#[allow(dead_code)]
impl MomentumIndicators {
    pub fn new() -> Self {
        Self
    }

    // Relative Strength Index
    pub fn rsi(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        if data.len() < period + 1 {
            return Err(anyhow::anyhow!("Insufficient data for RSI calculation"));
        }

        let mut gains = Vec::new();
        let mut losses = Vec::new();

        // Calculate price changes
        for i in 1..data.len() {
            let change = data[i].close.to_f64().unwrap_or(0.0) - data[i - 1].close.to_f64().unwrap_or(0.0);
            if change >= 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }

        let mut result = Vec::new();

        // Calculate initial averages using SMA
        if gains.len() >= period {
            let mut avg_gain: f64 = gains[0..period].iter().sum::<f64>() / period as f64;
            let mut avg_loss: f64 = losses[0..period].iter().sum::<f64>() / period as f64;

            // Calculate RSI for first period
            let rs = if avg_loss != 0.0 { avg_gain / avg_loss } else { f64::INFINITY };
            let rsi = 100.0 - (100.0 / (1.0 + rs));
            result.push(rsi);

            // Calculate subsequent RSI values using EMA
            for i in period..gains.len() {
                avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
                avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

                let rs = if avg_loss != 0.0 { avg_gain / avg_loss } else { f64::INFINITY };
                let rsi = 100.0 - (100.0 / (1.0 + rs));
                result.push(rsi);
            }
        }

        Ok(result)
    }

    // MACD Line (12-period EMA - 26-period EMA)
    pub fn macd_line(&self, data: &[Ohlcv], fast_period: usize, slow_period: usize) -> Result<Vec<f64>> {
        let fast_ema = self.ema(data, fast_period)?;
        let slow_ema = self.ema(data, slow_period)?;

        if fast_ema.len() != slow_ema.len() {
            return Err(anyhow::anyhow!("EMA lengths don't match"));
        }

        let macd_line: Vec<f64> = fast_ema
            .into_iter()
            .zip(slow_ema)
            .map(|(fast, slow)| fast - slow)
            .collect();

        Ok(macd_line)
    }

    // MACD Signal Line (9-period EMA of MACD Line)
    pub fn macd_signal(&self, data: &[Ohlcv], fast_period: usize, slow_period: usize, signal_period: usize) -> Result<Vec<f64>> {
        let macd_line = self.macd_line(data, fast_period, slow_period)?;
        
        // Create dummy Ohlcv data for signal EMA calculation
        let macd_data: Vec<Ohlcv> = macd_line
            .into_iter()
            .map(|value| Ohlcv {
                symbol: "MACD".to_string(),
                timestamp: chrono::Utc::now(),
                open: rust_decimal::Decimal::from_f64_retain(value).unwrap_or_default(),
                high: rust_decimal::Decimal::from_f64_retain(value).unwrap_or_default(),
                low: rust_decimal::Decimal::from_f64_retain(value).unwrap_or_default(),
                close: rust_decimal::Decimal::from_f64_retain(value).unwrap_or_default(),
                volume: rust_decimal::Decimal::ZERO,
            })
            .collect();

        self.ema(&macd_data, signal_period)
    }

    // MACD Histogram (MACD Line - Signal Line)
    pub fn macd_histogram(&self, data: &[Ohlcv], fast_period: usize, slow_period: usize, signal_period: usize) -> Result<Vec<f64>> {
        let macd_line = self.macd_line(data, fast_period, slow_period)?;
        let signal_line = self.macd_signal(data, fast_period, slow_period, signal_period)?;

        if macd_line.len() != signal_line.len() {
            return Err(anyhow::anyhow!("MACD and Signal line lengths don't match"));
        }

        let histogram: Vec<f64> = macd_line
            .into_iter()
            .zip(signal_line)
            .map(|(macd, signal)| macd - signal)
            .collect();

        Ok(histogram)
    }

    // Stochastic %K
    pub fn stochastic_k(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        if data.len() < period {
            return Err(anyhow::anyhow!("Insufficient data for Stochastic %K calculation"));
        }

        let mut result = Vec::new();

        for i in (period - 1)..data.len() {
            let window = &data[(i + 1 - period)..=i];
            let current_close = data[i].close.to_f64().unwrap_or(0.0);
            
            let lowest_low = window
                .iter()
                .map(|candle| candle.low.to_f64().unwrap_or(f64::MAX))
                .fold(f64::MAX, f64::min);
                
            let highest_high = window
                .iter()
                .map(|candle| candle.high.to_f64().unwrap_or(f64::MIN))
                .fold(f64::MIN, f64::max);

            let k_percent = if highest_high != lowest_low {
                ((current_close - lowest_low) / (highest_high - lowest_low)) * 100.0
            } else {
                50.0
            };

            result.push(k_percent);
        }

        Ok(result)
    }

    // Stochastic %D (3-period SMA of %K)
    pub fn stochastic_d(&self, data: &[Ohlcv], k_period: usize, d_period: usize) -> Result<Vec<f64>> {
        let k_values = self.stochastic_k(data, k_period)?;

        if k_values.len() < d_period {
            return Err(anyhow::anyhow!("Insufficient %K values for %D calculation"));
        }

        let mut result = Vec::new();

        for i in (d_period - 1)..k_values.len() {
            let sum: f64 = k_values[(i + 1 - d_period)..=i].iter().sum();
            result.push(sum / d_period as f64);
        }

        Ok(result)
    }

    // Williams %R
    pub fn williams_r(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        if data.len() < period {
            return Err(anyhow::anyhow!("Insufficient data for Williams %R calculation"));
        }

        let mut result = Vec::new();

        for i in (period - 1)..data.len() {
            let window = &data[(i + 1 - period)..=i];
            let current_close = data[i].close.to_f64().unwrap_or(0.0);
            
            let lowest_low = window
                .iter()
                .map(|candle| candle.low.to_f64().unwrap_or(f64::MAX))
                .fold(f64::MAX, f64::min);
                
            let highest_high = window
                .iter()
                .map(|candle| candle.high.to_f64().unwrap_or(f64::MIN))
                .fold(f64::MIN, f64::max);

            let williams_r = if highest_high != lowest_low {
                ((highest_high - current_close) / (highest_high - lowest_low)) * -100.0
            } else {
                -50.0
            };

            result.push(williams_r);
        }

        Ok(result)
    }

    // Ultimate Oscillator
    pub fn ultimate_oscillator(&self, data: &[Ohlcv], period1: usize, period2: usize, period3: usize) -> Result<Vec<f64>> {
        if data.len() < period3.max(period2.max(period1)) + 1 {
            return Err(anyhow::anyhow!("Insufficient data for Ultimate Oscillator calculation"));
        }

        let mut buying_pressure = Vec::new();
        let mut true_range = Vec::new();

        // Calculate Buying Pressure and True Range
        for i in 0..data.len() {
            let high = data[i].high.to_f64().unwrap_or(0.0);
            let low = data[i].low.to_f64().unwrap_or(0.0);
            let close = data[i].close.to_f64().unwrap_or(0.0);

            let bp = close - low.min(if i > 0 { data[i - 1].close.to_f64().unwrap_or(0.0) } else { low });
            buying_pressure.push(bp);

            let tr = if i == 0 {
                high - low
            } else {
                let prev_close = data[i - 1].close.to_f64().unwrap_or(0.0);
                (high - low)
                    .max((high - prev_close).abs())
                    .max((low - prev_close).abs())
            };
            true_range.push(tr);
        }

        let mut result = Vec::new();
        let max_period = period3.max(period2.max(period1));

        for i in (max_period - 1)..data.len() {
            let bp1: f64 = buying_pressure[(i + 1 - period1)..=i].iter().sum();
            let tr1: f64 = true_range[(i + 1 - period1)..=i].iter().sum();
            let avg1 = if tr1 != 0.0 { bp1 / tr1 } else { 0.0 };

            let bp2: f64 = buying_pressure[(i + 1 - period2)..=i].iter().sum();
            let tr2: f64 = true_range[(i + 1 - period2)..=i].iter().sum();
            let avg2 = if tr2 != 0.0 { bp2 / tr2 } else { 0.0 };

            let bp3: f64 = buying_pressure[(i + 1 - period3)..=i].iter().sum();
            let tr3: f64 = true_range[(i + 1 - period3)..=i].iter().sum();
            let avg3 = if tr3 != 0.0 { bp3 / tr3 } else { 0.0 };

            let uo = 100.0 * ((4.0 * avg1) + (2.0 * avg2) + avg3) / 7.0;
            result.push(uo);
        }

        Ok(result)
    }

    // Helper method to calculate EMA (same as in technical.rs)
    fn ema(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        if data.len() < period {
            return Err(anyhow::anyhow!("Insufficient data for EMA calculation"));
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut result = Vec::new();

        // Start with SMA for the first value
        let first_sma: f64 = data[0..period]
            .iter()
            .map(|candle| candle.close.to_f64().unwrap_or(0.0))
            .sum::<f64>() / period as f64;
        
        result.push(first_sma);

        // Calculate EMA for remaining values
        for item in data.iter().skip(period) {
            let close_price = item.close.to_f64().unwrap_or(0.0);
            let Some(last_value) = result.last() else {
                return Err(anyhow::anyhow!("Missing previous EMA value"));
            };
            let ema = (close_price * multiplier) + (last_value * (1.0 - multiplier));
            result.push(ema);
        }

        Ok(result)
    }
}

impl Default for MomentumIndicators {
    fn default() -> Self {
        Self::new()
    }
}