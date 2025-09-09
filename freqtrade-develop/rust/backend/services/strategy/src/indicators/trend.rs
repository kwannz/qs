use anyhow::Result;
use rust_decimal::prelude::ToPrimitive;
use crate::services::market_data_client::Ohlcv;

#[allow(dead_code)]
pub struct TrendIndicators;

#[allow(dead_code)]
impl TrendIndicators {
    pub fn new() -> Self {
        Self
    }

    // Average Directional Index (ADX)
    pub fn adx(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        if data.len() < period * 2 {
            return Err(anyhow::anyhow!("Insufficient data for ADX calculation"));
        }

        let (plus_di, minus_di) = self.calculate_directional_indicators(data, period)?;
        
        let mut dx_values = Vec::new();
        
        // Calculate DX (Directional Movement Index)
        for i in 0..plus_di.len() {
            let plus_di_val = plus_di[i];
            let minus_di_val = minus_di[i];
            
            let dx = if plus_di_val + minus_di_val != 0.0 {
                ((plus_di_val - minus_di_val).abs() / (plus_di_val + minus_di_val)) * 100.0
            } else {
                0.0
            };
            
            dx_values.push(dx);
        }

        // Calculate ADX as smoothed average of DX
        let mut adx_values = Vec::new();
        
        if dx_values.len() >= period {
            // First ADX is SMA of DX
            let first_adx: f64 = dx_values[0..period].iter().sum::<f64>() / period as f64;
            adx_values.push(first_adx);
            
            // Subsequent ADX values use Wilder's smoothing
            for &dx_value in dx_values.iter().skip(period) {
                let Some(last_adx) = adx_values.last() else {
                    return Err(anyhow::anyhow!("Missing previous ADX value"));
                };
                let adx = (last_adx * (period - 1) as f64 + dx_value) / period as f64;
                adx_values.push(adx);
            }
        }

        Ok(adx_values)
    }

    // Plus Directional Indicator (+DI)
    pub fn plus_di(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        let (plus_di, _) = self.calculate_directional_indicators(data, period)?;
        Ok(plus_di)
    }

    // Minus Directional Indicator (-DI)
    pub fn minus_di(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        let (_, minus_di) = self.calculate_directional_indicators(data, period)?;
        Ok(minus_di)
    }

    // Commodity Channel Index (CCI)
    pub fn cci(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        if data.len() < period {
            return Err(anyhow::anyhow!("Insufficient data for CCI calculation"));
        }

        let mut result = Vec::new();

        for i in (period - 1)..data.len() {
            // Calculate typical prices for the period
            let typical_prices: Vec<f64> = data[(i + 1 - period)..=i]
                .iter()
                .map(|candle| {
                    let high = candle.high.to_f64().unwrap_or(0.0);
                    let low = candle.low.to_f64().unwrap_or(0.0);
                    let close = candle.close.to_f64().unwrap_or(0.0);
                    (high + low + close) / 3.0
                })
                .collect();

            let Some(current_tp) = typical_prices.last() else {
                return Err(anyhow::anyhow!("No typical price data available"));
            };
            let sma_tp = typical_prices.iter().sum::<f64>() / period as f64;

            // Calculate mean deviation
            let mean_deviation = typical_prices
                .iter()
                .map(|tp| (tp - sma_tp).abs())
                .sum::<f64>() / period as f64;

            let cci = if mean_deviation != 0.0 {
                (current_tp - sma_tp) / (0.015 * mean_deviation)
            } else {
                0.0
            };

            result.push(cci);
        }

        Ok(result)
    }

    // Aroon Up
    pub fn aroon_up(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        if data.len() < period {
            return Err(anyhow::anyhow!("Insufficient data for Aroon Up calculation"));
        }

        let mut result = Vec::new();

        for i in (period - 1)..data.len() {
            let window = &data[(i + 1 - period)..=i];
            
            // Find the position of the highest high within the period
            let mut highest_high_idx = 0;
            let mut highest_high = window[0].high.to_f64().unwrap_or(f64::MIN);
            
            for (j, candle) in window.iter().enumerate() {
                let high = candle.high.to_f64().unwrap_or(f64::MIN);
                if high > highest_high {
                    highest_high = high;
                    highest_high_idx = j;
                }
            }

            // Calculate periods since highest high
            let periods_since_high = (period - 1) - highest_high_idx;
            let aroon_up = ((period - periods_since_high) as f64 / period as f64) * 100.0;
            
            result.push(aroon_up);
        }

        Ok(result)
    }

    // Aroon Down
    pub fn aroon_down(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        if data.len() < period {
            return Err(anyhow::anyhow!("Insufficient data for Aroon Down calculation"));
        }

        let mut result = Vec::new();

        for i in (period - 1)..data.len() {
            let window = &data[(i + 1 - period)..=i];
            
            // Find the position of the lowest low within the period
            let mut lowest_low_idx = 0;
            let mut lowest_low = window[0].low.to_f64().unwrap_or(f64::MAX);
            
            for (j, candle) in window.iter().enumerate() {
                let low = candle.low.to_f64().unwrap_or(f64::MAX);
                if low < lowest_low {
                    lowest_low = low;
                    lowest_low_idx = j;
                }
            }

            // Calculate periods since lowest low
            let periods_since_low = (period - 1) - lowest_low_idx;
            let aroon_down = ((period - periods_since_low) as f64 / period as f64) * 100.0;
            
            result.push(aroon_down);
        }

        Ok(result)
    }

    // Aroon Oscillator (Aroon Up - Aroon Down)
    pub fn aroon_oscillator(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        let aroon_up = self.aroon_up(data, period)?;
        let aroon_down = self.aroon_down(data, period)?;

        let oscillator: Vec<f64> = aroon_up
            .into_iter()
            .zip(aroon_down)
            .map(|(up, down)| up - down)
            .collect();

        Ok(oscillator)
    }

    // Parabolic SAR
    pub fn parabolic_sar(&self, data: &[Ohlcv], acceleration: f64, max_acceleration: f64) -> Result<Vec<f64>> {
        if data.len() < 2 {
            return Err(anyhow::anyhow!("Insufficient data for Parabolic SAR calculation"));
        }

        let mut result = Vec::new();
        let mut sar = data[0].low.to_f64().unwrap_or(0.0); // Start with first low
        let mut extreme_point = data[0].high.to_f64().unwrap_or(0.0);
        let mut acceleration_factor = acceleration;
        let mut is_long = true; // Start with long position

        result.push(sar);

        for i in 1..data.len() {
            let high = data[i].high.to_f64().unwrap_or(0.0);
            let low = data[i].low.to_f64().unwrap_or(0.0);

            if is_long {
                // Long position
                sar = sar + acceleration_factor * (extreme_point - sar);
                
                // Check for trend reversal
                if low <= sar {
                    is_long = false;
                    sar = extreme_point;
                    extreme_point = low;
                    acceleration_factor = acceleration;
                } else {
                    // Update extreme point and acceleration factor
                    if high > extreme_point {
                        extreme_point = high;
                        acceleration_factor = (acceleration_factor + acceleration).min(max_acceleration);
                    }
                    
                    // SAR cannot be above the previous two lows
                    if i >= 2 {
                        let prev_low = data[i - 1].low.to_f64().unwrap_or(0.0);
                        let prev_prev_low = data[i - 2].low.to_f64().unwrap_or(0.0);
                        sar = sar.min(prev_low).min(prev_prev_low);
                    }
                }
            } else {
                // Short position
                sar = sar + acceleration_factor * (extreme_point - sar);
                
                // Check for trend reversal
                if high >= sar {
                    is_long = true;
                    sar = extreme_point;
                    extreme_point = high;
                    acceleration_factor = acceleration;
                } else {
                    // Update extreme point and acceleration factor
                    if low < extreme_point {
                        extreme_point = low;
                        acceleration_factor = (acceleration_factor + acceleration).min(max_acceleration);
                    }
                    
                    // SAR cannot be below the previous two highs
                    if i >= 2 {
                        let prev_high = data[i - 1].high.to_f64().unwrap_or(0.0);
                        let prev_prev_high = data[i - 2].high.to_f64().unwrap_or(0.0);
                        sar = sar.max(prev_high).max(prev_prev_high);
                    }
                }
            }

            result.push(sar);
        }

        Ok(result)
    }

    // Ichimoku Kijun-sen (Base Line)
    pub fn ichimoku_kijun(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        if data.len() < period {
            return Err(anyhow::anyhow!("Insufficient data for Ichimoku Kijun-sen calculation"));
        }

        let mut result = Vec::new();

        for i in (period - 1)..data.len() {
            let window = &data[(i + 1 - period)..=i];
            
            let highest_high = window
                .iter()
                .map(|candle| candle.high.to_f64().unwrap_or(f64::MIN))
                .fold(f64::MIN, f64::max);
                
            let lowest_low = window
                .iter()
                .map(|candle| candle.low.to_f64().unwrap_or(f64::MAX))
                .fold(f64::MAX, f64::min);

            let kijun = (highest_high + lowest_low) / 2.0;
            result.push(kijun);
        }

        Ok(result)
    }

    // Ichimoku Tenkan-sen (Conversion Line) 
    pub fn ichimoku_tenkan(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        // Same calculation as Kijun-sen but different period (typically 9)
        self.ichimoku_kijun(data, period)
    }

    // Helper method to calculate +DI and -DI
    fn calculate_directional_indicators(&self, data: &[Ohlcv], period: usize) -> Result<(Vec<f64>, Vec<f64>)> {
        if data.len() < period + 1 {
            return Err(anyhow::anyhow!("Insufficient data for directional indicators calculation"));
        }

        let mut plus_dm = Vec::new();
        let mut minus_dm = Vec::new();
        let mut true_range = Vec::new();

        // Calculate directional movements and true ranges
        for i in 1..data.len() {
            let high = data[i].high.to_f64().unwrap_or(0.0);
            let low = data[i].low.to_f64().unwrap_or(0.0);
            let prev_high = data[i - 1].high.to_f64().unwrap_or(0.0);
            let prev_low = data[i - 1].low.to_f64().unwrap_or(0.0);
            let prev_close = data[i - 1].close.to_f64().unwrap_or(0.0);

            let up_move = high - prev_high;
            let down_move = prev_low - low;

            let plus_dm_val = if up_move > down_move && up_move > 0.0 { up_move } else { 0.0 };
            let minus_dm_val = if down_move > up_move && down_move > 0.0 { down_move } else { 0.0 };

            plus_dm.push(plus_dm_val);
            minus_dm.push(minus_dm_val);

            let tr = (high - low)
                .max((high - prev_close).abs())
                .max((low - prev_close).abs());
            true_range.push(tr);
        }

        // Calculate smoothed averages using Wilder's smoothing
        let mut plus_di = Vec::new();
        let mut minus_di = Vec::new();

        if plus_dm.len() >= period && minus_dm.len() >= period && true_range.len() >= period {
            // Initial values (SMA)
            let mut smoothed_plus_dm: f64 = plus_dm[0..period].iter().sum::<f64>() / period as f64;
            let mut smoothed_minus_dm: f64 = minus_dm[0..period].iter().sum::<f64>() / period as f64;
            let mut smoothed_tr: f64 = true_range[0..period].iter().sum::<f64>() / period as f64;

            // Calculate initial DI values
            let initial_plus_di = if smoothed_tr != 0.0 { (smoothed_plus_dm / smoothed_tr) * 100.0 } else { 0.0 };
            let initial_minus_di = if smoothed_tr != 0.0 { (smoothed_minus_dm / smoothed_tr) * 100.0 } else { 0.0 };
            
            plus_di.push(initial_plus_di);
            minus_di.push(initial_minus_di);

            // Calculate subsequent values using Wilder's smoothing
            for i in period..plus_dm.len() {
                smoothed_plus_dm = (smoothed_plus_dm * (period - 1) as f64 + plus_dm[i]) / period as f64;
                smoothed_minus_dm = (smoothed_minus_dm * (period - 1) as f64 + minus_dm[i]) / period as f64;
                smoothed_tr = (smoothed_tr * (period - 1) as f64 + true_range[i]) / period as f64;

                let plus_di_val = if smoothed_tr != 0.0 { (smoothed_plus_dm / smoothed_tr) * 100.0 } else { 0.0 };
                let minus_di_val = if smoothed_tr != 0.0 { (smoothed_minus_dm / smoothed_tr) * 100.0 } else { 0.0 };

                plus_di.push(plus_di_val);
                minus_di.push(minus_di_val);
            }
        }

        Ok((plus_di, minus_di))
    }
}

impl Default for TrendIndicators {
    fn default() -> Self {
        Self::new()
    }
}