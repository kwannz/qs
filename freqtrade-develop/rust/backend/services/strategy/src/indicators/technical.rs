use anyhow::Result;
// Remove unused import
use rust_decimal::prelude::ToPrimitive;

use crate::services::market_data_client::Ohlcv;

#[allow(dead_code)]
pub struct TechnicalIndicators;

#[allow(dead_code)]
impl TechnicalIndicators {
    pub fn new() -> Self {
        Self
    }

    // Simple Moving Average
    pub fn sma(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        if data.len() < period {
            return Err(anyhow::anyhow!("Insufficient data for SMA calculation"));
        }

        let mut result = Vec::new();
        
        for i in (period - 1)..data.len() {
            let sum: f64 = data[(i + 1 - period)..=i]
                .iter()
                .map(|candle| candle.close.to_f64().unwrap_or(0.0))
                .sum();
            result.push(sum / period as f64);
        }

        Ok(result)
    }

    // Exponential Moving Average
    pub fn ema(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
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

    // Weighted Moving Average
    pub fn wma(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        if data.len() < period {
            return Err(anyhow::anyhow!("Insufficient data for WMA calculation"));
        }

        let mut result = Vec::new();
        let weight_sum: f64 = (1..=period).sum::<usize>() as f64;

        for i in (period - 1)..data.len() {
            let weighted_sum: f64 = data[(i + 1 - period)..=i]
                .iter()
                .enumerate()
                .map(|(j, candle)| {
                    let price = candle.close.to_f64().unwrap_or(0.0);
                    let weight = (j + 1) as f64;
                    price * weight
                })
                .sum();
            
            result.push(weighted_sum / weight_sum);
        }

        Ok(result)
    }

    // Typical Price (HLC/3)
    pub fn typical_price(&self, data: &[Ohlcv]) -> Vec<f64> {
        data.iter()
            .map(|candle| {
                let high = candle.high.to_f64().unwrap_or(0.0);
                let low = candle.low.to_f64().unwrap_or(0.0);
                let close = candle.close.to_f64().unwrap_or(0.0);
                (high + low + close) / 3.0
            })
            .collect()
    }

    // True Range
    pub fn true_range(&self, data: &[Ohlcv]) -> Vec<f64> {
        let mut result = Vec::new();
        
        for i in 0..data.len() {
            let high = data[i].high.to_f64().unwrap_or(0.0);
            let low = data[i].low.to_f64().unwrap_or(0.0);
            let _close = data[i].close.to_f64().unwrap_or(0.0);
            
            let tr = if i == 0 {
                high - low
            } else {
                let prev_close = data[i - 1].close.to_f64().unwrap_or(0.0);
                (high - low)
                    .max((high - prev_close).abs())
                    .max((low - prev_close).abs())
            };
            
            result.push(tr);
        }
        
        result
    }

    // Price Rate of Change
    pub fn roc(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        if data.len() < period + 1 {
            return Err(anyhow::anyhow!("Insufficient data for ROC calculation"));
        }

        let mut result = Vec::new();
        
        for i in period..data.len() {
            let current_price = data[i].close.to_f64().unwrap_or(0.0);
            let past_price = data[i - period].close.to_f64().unwrap_or(0.0);
            
            let roc = if past_price != 0.0 {
                ((current_price - past_price) / past_price) * 100.0
            } else {
                0.0
            };
            
            result.push(roc);
        }

        Ok(result)
    }

    // On-Balance Volume
    pub fn obv(&self, data: &[Ohlcv]) -> Vec<f64> {
        let mut result = Vec::new();
        let mut obv = 0.0;
        
        for i in 0..data.len() {
            if i == 0 {
                obv = data[i].volume.to_f64().unwrap_or(0.0);
            } else {
                let current_close = data[i].close.to_f64().unwrap_or(0.0);
                let prev_close = data[i - 1].close.to_f64().unwrap_or(0.0);
                let volume = data[i].volume.to_f64().unwrap_or(0.0);
                
                if current_close > prev_close {
                    obv += volume;
                } else if current_close < prev_close {
                    obv -= volume;
                }
                // If prices are equal, OBV remains unchanged
            }
            
            result.push(obv);
        }
        
        result
    }

    // Accumulation/Distribution Line
    pub fn ad_line(&self, data: &[Ohlcv]) -> Vec<f64> {
        let mut result = Vec::new();
        let mut ad_line = 0.0;
        
        for candle in data {
            let high = candle.high.to_f64().unwrap_or(0.0);
            let low = candle.low.to_f64().unwrap_or(0.0);
            let close = candle.close.to_f64().unwrap_or(0.0);
            let volume = candle.volume.to_f64().unwrap_or(0.0);
            
            let money_flow_multiplier = if high != low {
                ((close - low) - (high - close)) / (high - low)
            } else {
                0.0
            };
            
            let money_flow_volume = money_flow_multiplier * volume;
            ad_line += money_flow_volume;
            
            result.push(ad_line);
        }
        
        result
    }

    // Volume Weighted Average Price
    pub fn vwap(&self, data: &[Ohlcv]) -> Vec<f64> {
        let mut result = Vec::new();
        let mut cumulative_volume = 0.0;
        let mut cumulative_price_volume = 0.0;
        
        for candle in data {
            let typical_price = {
                let high = candle.high.to_f64().unwrap_or(0.0);
                let low = candle.low.to_f64().unwrap_or(0.0);
                let close = candle.close.to_f64().unwrap_or(0.0);
                (high + low + close) / 3.0
            };
            let volume = candle.volume.to_f64().unwrap_or(0.0);
            
            cumulative_price_volume += typical_price * volume;
            cumulative_volume += volume;
            
            let vwap = if cumulative_volume != 0.0 {
                cumulative_price_volume / cumulative_volume
            } else {
                typical_price
            };
            
            result.push(vwap);
        }
        
        result
    }

    // Money Flow Index components
    pub fn money_flow(&self, data: &[Ohlcv]) -> Vec<f64> {
        data.iter()
            .map(|candle| {
                let typical_price = {
                    let high = candle.high.to_f64().unwrap_or(0.0);
                    let low = candle.low.to_f64().unwrap_or(0.0);
                    let close = candle.close.to_f64().unwrap_or(0.0);
                    (high + low + close) / 3.0
                };
                let volume = candle.volume.to_f64().unwrap_or(0.0);
                typical_price * volume
            })
            .collect()
    }

    // Pivot Points (Standard)
    pub fn pivot_points(&self, data: &[Ohlcv]) -> Result<Vec<PivotPoint>> {
        if data.is_empty() {
            return Err(anyhow::anyhow!("No data provided for pivot point calculation"));
        }

        let mut result = Vec::new();
        
        for i in 1..data.len() {
            let prev_candle = &data[i - 1];
            let high = prev_candle.high.to_f64().unwrap_or(0.0);
            let low = prev_candle.low.to_f64().unwrap_or(0.0);
            let close = prev_candle.close.to_f64().unwrap_or(0.0);
            
            let pivot = (high + low + close) / 3.0;
            let r1 = 2.0 * pivot - low;
            let s1 = 2.0 * pivot - high;
            let r2 = pivot + (high - low);
            let s2 = pivot - (high - low);
            let r3 = high + 2.0 * (pivot - low);
            let s3 = low - 2.0 * (high - pivot);
            
            result.push(PivotPoint {
                pivot,
                r1,
                r2,
                r3,
                s1,
                s2,
                s3,
            });
        }
        
        Ok(result)
    }
}

impl Default for TechnicalIndicators {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PivotPoint {
    pub pivot: f64,
    pub r1: f64,
    pub r2: f64,
    pub r3: f64,
    pub s1: f64,
    pub s2: f64,
    pub s3: f64,
}