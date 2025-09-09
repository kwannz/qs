use anyhow::Result;
use rust_decimal::prelude::ToPrimitive;
use crate::services::market_data_client::Ohlcv;

#[allow(dead_code)]
pub struct VolatilityIndicators;

#[allow(dead_code)]
impl VolatilityIndicators {
    pub fn new() -> Self {
        Self
    }

    // Average True Range
    pub fn atr(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        if data.len() < period {
            return Err(anyhow::anyhow!("Insufficient data for ATR calculation"));
        }

        let true_ranges = self.calculate_true_range(data);
        
        let mut result = Vec::new();
        
        // First ATR value is SMA of true ranges
        let first_atr: f64 = true_ranges[0..period].iter().sum::<f64>() / period as f64;
        result.push(first_atr);

        // Subsequent ATR values use Wilder's smoothing
        for &tr_value in true_ranges.iter().skip(period) {
            let Some(last_atr) = result.last() else {
                return Err(anyhow::anyhow!("Missing previous ATR value"));
            };
            let atr = (last_atr * (period - 1) as f64 + tr_value) / period as f64;
            result.push(atr);
        }

        Ok(result)
    }

    // Bollinger Bands Upper Band
    pub fn bollinger_bands_upper(&self, data: &[Ohlcv], period: usize, std_dev: f64) -> Result<Vec<f64>> {
        let sma = self.sma(data, period)?;
        let std_devs = self.rolling_std_dev(data, period)?;

        let upper_bands: Vec<f64> = sma
            .into_iter()
            .zip(std_devs)
            .map(|(sma, std)| sma + (std_dev * std))
            .collect();

        Ok(upper_bands)
    }

    // Bollinger Bands Lower Band
    pub fn bollinger_bands_lower(&self, data: &[Ohlcv], period: usize, std_dev: f64) -> Result<Vec<f64>> {
        let sma = self.sma(data, period)?;
        let std_devs = self.rolling_std_dev(data, period)?;

        let lower_bands: Vec<f64> = sma
            .into_iter()
            .zip(std_devs)
            .map(|(sma, std)| sma - (std_dev * std))
            .collect();

        Ok(lower_bands)
    }

    // Bollinger Bands Middle Band (SMA)
    pub fn bollinger_bands_middle(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        self.sma(data, period)
    }

    // Bollinger Bands %B
    pub fn bollinger_percent_b(&self, data: &[Ohlcv], period: usize, std_dev: f64) -> Result<Vec<f64>> {
        let upper_bands = self.bollinger_bands_upper(data, period, std_dev)?;
        let lower_bands = self.bollinger_bands_lower(data, period, std_dev)?;
        
        let mut result = Vec::new();
        let data_offset = data.len() - upper_bands.len();

        for i in 0..upper_bands.len() {
            let close = data[data_offset + i].close.to_f64().unwrap_or(0.0);
            let upper = upper_bands[i];
            let lower = lower_bands[i];
            
            let percent_b = if upper != lower {
                (close - lower) / (upper - lower)
            } else {
                0.5
            };
            
            result.push(percent_b);
        }

        Ok(result)
    }

    // Bollinger Bands Width
    pub fn bollinger_bandwidth(&self, data: &[Ohlcv], period: usize, std_dev: f64) -> Result<Vec<f64>> {
        let upper_bands = self.bollinger_bands_upper(data, period, std_dev)?;
        let lower_bands = self.bollinger_bands_lower(data, period, std_dev)?;
        let middle_bands = self.bollinger_bands_middle(data, period)?;

        let bandwidths: Vec<f64> = upper_bands
            .into_iter()
            .zip(lower_bands)
            .zip(middle_bands)
            .map(|((upper, lower), middle)| {
                if middle != 0.0 {
                    (upper - lower) / middle
                } else {
                    0.0
                }
            })
            .collect();

        Ok(bandwidths)
    }

    // Keltner Channel Upper
    pub fn keltner_upper(&self, data: &[Ohlcv], period: usize, multiplier: f64) -> Result<Vec<f64>> {
        let ema = self.ema(data, period)?;
        let atr = self.atr(data, period)?;

        // Align lengths (ATR might be shorter than EMA)
        let offset = ema.len() - atr.len();
        let upper_bands: Vec<f64> = ema[offset..]
            .iter()
            .zip(atr.iter())
            .map(|(ema, atr)| ema + (multiplier * atr))
            .collect();

        Ok(upper_bands)
    }

    // Keltner Channel Lower
    pub fn keltner_lower(&self, data: &[Ohlcv], period: usize, multiplier: f64) -> Result<Vec<f64>> {
        let ema = self.ema(data, period)?;
        let atr = self.atr(data, period)?;

        // Align lengths
        let offset = ema.len() - atr.len();
        let lower_bands: Vec<f64> = ema[offset..]
            .iter()
            .zip(atr.iter())
            .map(|(ema, atr)| ema - (multiplier * atr))
            .collect();

        Ok(lower_bands)
    }

    // Donchian Channel Upper (Highest High)
    pub fn donchian_upper(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        if data.len() < period {
            return Err(anyhow::anyhow!("Insufficient data for Donchian Upper calculation"));
        }

        let mut result = Vec::new();

        for i in (period - 1)..data.len() {
            let highest_high = data[(i + 1 - period)..=i]
                .iter()
                .map(|candle| candle.high.to_f64().unwrap_or(f64::MIN))
                .fold(f64::MIN, f64::max);
            
            result.push(highest_high);
        }

        Ok(result)
    }

    // Donchian Channel Lower (Lowest Low)
    pub fn donchian_lower(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        if data.len() < period {
            return Err(anyhow::anyhow!("Insufficient data for Donchian Lower calculation"));
        }

        let mut result = Vec::new();

        for i in (period - 1)..data.len() {
            let lowest_low = data[(i + 1 - period)..=i]
                .iter()
                .map(|candle| candle.low.to_f64().unwrap_or(f64::MAX))
                .fold(f64::MAX, f64::min);
            
            result.push(lowest_low);
        }

        Ok(result)
    }

    // Donchian Channel Middle
    pub fn donchian_middle(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        let upper = self.donchian_upper(data, period)?;
        let lower = self.donchian_lower(data, period)?;

        let middle: Vec<f64> = upper
            .into_iter()
            .zip(lower)
            .map(|(u, l)| (u + l) / 2.0)
            .collect();

        Ok(middle)
    }

    // Chaikin Volatility
    pub fn chaikin_volatility(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        if data.len() < period + 1 {
            return Err(anyhow::anyhow!("Insufficient data for Chaikin Volatility calculation"));
        }

        // Calculate High-Low spread
        let hl_spread: Vec<f64> = data
            .iter()
            .map(|candle| {
                let high = candle.high.to_f64().unwrap_or(0.0);
                let low = candle.low.to_f64().unwrap_or(0.0);
                high - low
            })
            .collect();

        // Create dummy Ohlcv data for EMA calculation
        let spread_data: Vec<Ohlcv> = hl_spread
            .into_iter()
            .map(|value| Ohlcv {
                symbol: "SPREAD".to_string(),
                timestamp: chrono::Utc::now(),
                open: rust_decimal::Decimal::from_f64_retain(value).unwrap_or_default(),
                high: rust_decimal::Decimal::from_f64_retain(value).unwrap_or_default(),
                low: rust_decimal::Decimal::from_f64_retain(value).unwrap_or_default(),
                close: rust_decimal::Decimal::from_f64_retain(value).unwrap_or_default(),
                volume: rust_decimal::Decimal::ZERO,
            })
            .collect();

        let ema_spread = self.ema(&spread_data, period)?;

        // Calculate volatility as percentage change
        let mut result = Vec::new();
        for i in period..ema_spread.len() {
            let current = ema_spread[i];
            let previous = ema_spread[i - period];
            
            let volatility = if previous != 0.0 {
                ((current - previous) / previous) * 100.0
            } else {
                0.0
            };
            
            result.push(volatility);
        }

        Ok(result)
    }

    // Helper Methods

    fn calculate_true_range(&self, data: &[Ohlcv]) -> Vec<f64> {
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

    fn sma(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
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

    fn rolling_std_dev(&self, data: &[Ohlcv], period: usize) -> Result<Vec<f64>> {
        if data.len() < period {
            return Err(anyhow::anyhow!("Insufficient data for standard deviation calculation"));
        }

        let mut result = Vec::new();

        for i in (period - 1)..data.len() {
            let window: Vec<f64> = data[(i + 1 - period)..=i]
                .iter()
                .map(|candle| candle.close.to_f64().unwrap_or(0.0))
                .collect();

            let mean = window.iter().sum::<f64>() / period as f64;
            let variance = window
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>() / period as f64;
            
            result.push(variance.sqrt());
        }

        Ok(result)
    }
}

impl Default for VolatilityIndicators {
    fn default() -> Self {
        Self::new()
    }
}