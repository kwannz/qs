use anyhow::{Result, Context};
use async_trait::async_trait;
use chrono::{DateTime, Utc, Duration};
use ndarray::{Array1, Array2};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};

use super::models::{
    Factor, FactorResult, FactorTimeSeries, FactorDataPoint, FactorResultMetadata,
    FactorComputationContext, FactorError, FactorCategory, DataField, ComputationCost,
    FactorParameterValue, TimeSeriesStatistics,
};

/// Trait for factor calculation algorithms
#[async_trait]
pub trait FactorCalculator: Send + Sync {
    /// Calculate factor value for a single point in time
    async fn calculate_point(
        &self,
        context: &FactorComputationContext,
        timestamp: DateTime<Utc>,
    ) -> Result<FactorResult>;

    /// Calculate factor time series over a date range
    async fn calculate_time_series(
        &self,
        context: &FactorComputationContext,
    ) -> Result<FactorTimeSeries>;

    /// Get required data fields for this calculator
    fn required_data_fields(&self) -> Vec<DataField>;

    /// Get computational cost estimate
    fn computation_cost(&self) -> ComputationCost;

    /// Validate that we have sufficient data for calculation
    async fn validate_data(&self, context: &FactorComputationContext) -> Result<()>;
}

/// Market data provider trait
#[async_trait]
pub trait MarketDataProvider: Send + Sync {
    async fn get_price_data(
        &self,
        symbol: &str,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<Vec<(DateTime<Utc>, f64)>>;

    async fn get_ohlc_data(
        &self,
        symbol: &str,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<Vec<OhlcData>>;

    async fn get_volume_data(
        &self,
        symbol: &str,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<Vec<(DateTime<Utc>, f64)>>;
}

#[derive(Debug, Clone)]
pub struct OhlcData {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Main factor calculation engine
pub struct FactorCalculationEngine {
    calculators: HashMap<String, Box<dyn FactorCalculator>>,
    data_provider: Arc<dyn MarketDataProvider>,
    cache: Arc<RwLock<CalculationCache>>,
}

#[derive(Debug)]
struct CalculationCache {
    results: HashMap<String, (DateTime<Utc>, FactorResult)>,
    market_data: HashMap<String, (DateTime<Utc>, Vec<OhlcData>)>,
    max_cache_age: Duration,
}

impl FactorCalculationEngine {
    pub fn new(data_provider: Arc<dyn MarketDataProvider>) -> Self {
        let mut engine = Self {
            calculators: HashMap::new(),
            data_provider,
            cache: Arc::new(RwLock::new(CalculationCache {
                results: HashMap::new(),
                market_data: HashMap::new(),
                max_cache_age: Duration::hours(1),
            })),
        };

        // Register built-in calculators
        engine.register_default_calculators();
        engine
    }

    pub fn register_calculator(&mut self, factor_name: String, calculator: Box<dyn FactorCalculator>) {
        self.calculators.insert(factor_name, calculator);
    }

    fn register_default_calculators(&mut self) {
        self.register_calculator("SMA".to_string(), Box::new(SimpleMovingAverageCalculator));
        self.register_calculator("EMA".to_string(), Box::new(ExponentialMovingAverageCalculator));
        self.register_calculator("RSI".to_string(), Box::new(RSICalculator));
        self.register_calculator("MACD".to_string(), Box::new(MACDCalculator));
        self.register_calculator("BollingerBands".to_string(), Box::new(BollingerBandsCalculator));
        self.register_calculator("Momentum".to_string(), Box::new(MomentumCalculator));
        self.register_calculator("Volatility".to_string(), Box::new(VolatilityCalculator));
        self.register_calculator("VolumeWeightedAverage".to_string(), Box::new(VWAPCalculator));
        self.register_calculator("StochasticOscillator".to_string(), Box::new(StochasticCalculator));
        self.register_calculator("WilliamsR".to_string(), Box::new(WilliamsRCalculator));
    }

    pub async fn calculate_factor(
        &self,
        factor: &Factor,
        symbol: &str,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<FactorTimeSeries> {
        let calculator_name = self.extract_calculator_name(&factor.name)?;
        
        let calculator = self.calculators.get(&calculator_name)
            .ok_or_else(|| FactorError::UnknownFactor(calculator_name.clone()))?;

        // Build computation context
        let context = self.build_computation_context(factor, symbol, start_date, end_date).await?;

        // Validate data
        calculator.validate_data(&context).await?;

        // Calculate time series
        let time_series = calculator.calculate_time_series(&context).await?;

        debug!("Calculated factor {} for {} with {} data points", 
               factor.name, symbol, time_series.data_points.len());

        Ok(time_series)
    }

    pub async fn calculate_single_point(
        &self,
        factor: &Factor,
        symbol: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<FactorResult> {
        let calculator_name = self.extract_calculator_name(&factor.name)?;
        
        let calculator = self.calculators.get(&calculator_name)
            .ok_or_else(|| FactorError::UnknownFactor(calculator_name.clone()))?;

        // Build computation context for single point
        let lookback_start = timestamp - Duration::days(365); // Default lookback
        let context = self.build_computation_context(factor, symbol, lookback_start, timestamp).await?;

        // Calculate single point
        calculator.calculate_point(&context, timestamp).await
    }

    async fn build_computation_context(
        &self,
        factor: &Factor,
        symbol: &str,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<FactorComputationContext> {
        // Get market data with extended lookback for calculations
        let lookback_days = factor.parameters.window_size.unwrap_or(252) as i64;
        let extended_start = start_date - Duration::days(lookback_days);

        let ohlc_data = self.data_provider.get_ohlc_data(symbol, extended_start, end_date).await?;

        // Convert to required format
        let mut market_data = HashMap::new();
        
        let timestamps: Vec<f64> = ohlc_data.iter()
            .map(|d| d.timestamp.timestamp() as f64)
            .collect();
        let opens: Vec<f64> = ohlc_data.iter().map(|d| d.open).collect();
        let highs: Vec<f64> = ohlc_data.iter().map(|d| d.high).collect();
        let lows: Vec<f64> = ohlc_data.iter().map(|d| d.low).collect();
        let closes: Vec<f64> = ohlc_data.iter().map(|d| d.close).collect();
        let volumes: Vec<f64> = ohlc_data.iter().map(|d| d.volume).collect();

        market_data.insert("timestamp".to_string(), timestamps);
        market_data.insert("open".to_string(), opens);
        market_data.insert("high".to_string(), highs);
        market_data.insert("low".to_string(), lows);
        market_data.insert("close".to_string(), closes);
        market_data.insert("volume".to_string(), volumes);

        Ok(FactorComputationContext {
            factor: factor.clone(),
            symbol: symbol.to_string(),
            start_date,
            end_date,
            market_data,
            custom_data: HashMap::new(),
        })
    }

    fn extract_calculator_name(&self, factor_name: &str) -> Result<String> {
        // Extract base calculator name from factor name (e.g., "SMA_20" -> "SMA")
        let parts: Vec<&str> = factor_name.split('_').collect();
        if parts.is_empty() {
            return Err(FactorError::InvalidParameters("Invalid factor name".to_string()).into());
        }
        Ok(parts[0].to_string())
    }
}

// Helper functions for common calculations
pub struct MathUtils;

impl MathUtils {
    pub fn simple_moving_average(data: &[f64], window: usize) -> Vec<f64> {
        if window == 0 || data.len() < window {
            return vec![];
        }

        let mut result = Vec::new();
        for i in (window - 1)..data.len() {
            let sum: f64 = data[(i + 1 - window)..=i].iter().sum();
            result.push(sum / window as f64);
        }
        result
    }

    pub fn exponential_moving_average(data: &[f64], alpha: f64) -> Vec<f64> {
        if data.is_empty() {
            return vec![];
        }

        let mut result = Vec::with_capacity(data.len());
        result.push(data[0]);

        for i in 1..data.len() {
            let ema = alpha * data[i] + (1.0 - alpha) * result[i - 1];
            result.push(ema);
        }
        result
    }

    pub fn standard_deviation(data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (data.len() - 1) as f64;
        variance.sqrt()
    }

    pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let numerator: f64 = x.iter().zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let sum_sq_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();

        if sum_sq_x == 0.0 || sum_sq_y == 0.0 {
            return 0.0;
        }

        numerator / (sum_sq_x * sum_sq_y).sqrt()
    }
}

// Built-in factor calculators

/// Simple Moving Average Calculator
struct SimpleMovingAverageCalculator;

#[async_trait]
impl FactorCalculator for SimpleMovingAverageCalculator {
    async fn calculate_point(
        &self,
        context: &FactorComputationContext,
        timestamp: DateTime<Utc>,
    ) -> Result<FactorResult> {
        let window = context.factor.parameters.window_size.unwrap_or(20) as usize;
        let closes = context.market_data.get("close")
            .ok_or_else(|| FactorError::InsufficientData("No close price data".to_string()))?;

        let sma_values = MathUtils::simple_moving_average(closes, window);
        
        if sma_values.is_empty() {
            return Err(FactorError::InsufficientData("Insufficient data for SMA calculation".to_string()).into());
        }

        let value = *sma_values.last().unwrap();

        Ok(FactorResult {
            factor_id: context.factor.id.clone(),
            symbol: context.symbol.clone(),
            timestamp,
            value,
            percentile_rank: None,
            z_score: None,
            confidence: 0.95,
            metadata: FactorResultMetadata {
                computation_time_ms: 1.0,
                data_points_used: window as u32,
                data_quality_score: 1.0,
                cache_hit: false,
                warnings: vec![],
                debug_info: None,
            },
        })
    }

    async fn calculate_time_series(
        &self,
        context: &FactorComputationContext,
    ) -> Result<FactorTimeSeries> {
        let window = context.factor.parameters.window_size.unwrap_or(20) as usize;
        let closes = context.market_data.get("close")
            .ok_or_else(|| FactorError::InsufficientData("No close price data".to_string()))?;
        let timestamps = context.market_data.get("timestamp")
            .ok_or_else(|| FactorError::InsufficientData("No timestamp data".to_string()))?;

        let sma_values = MathUtils::simple_moving_average(closes, window);
        
        if sma_values.is_empty() {
            return Err(FactorError::InsufficientData("Insufficient data for SMA calculation".to_string()).into());
        }

        let mut data_points = Vec::new();
        for (i, &value) in sma_values.iter().enumerate() {
            let timestamp_index = i + window - 1;
            if timestamp_index < timestamps.len() {
                let timestamp = DateTime::from_timestamp(timestamps[timestamp_index] as i64, 0)
                    .unwrap_or(Utc::now());
                
                data_points.push(FactorDataPoint {
                    timestamp,
                    value,
                    volume: None,
                    quality_score: 1.0,
                });
            }
        }

        let statistics = TimeSeriesStatistics {
            count: data_points.len(),
            mean: sma_values.iter().sum::<f64>() / sma_values.len() as f64,
            median: {
                let mut sorted = sma_values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sorted[sorted.len() / 2]
            },
            std_dev: MathUtils::standard_deviation(&sma_values),
            min: sma_values.iter().cloned().fold(f64::INFINITY, f64::min),
            max: sma_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            skewness: 0.0, // Simplified
            kurtosis: 0.0, // Simplified
            first_timestamp: data_points.first().map(|p| p.timestamp).unwrap_or(Utc::now()),
            last_timestamp: data_points.last().map(|p| p.timestamp).unwrap_or(Utc::now()),
        };

        Ok(FactorTimeSeries {
            factor_id: context.factor.id.clone(),
            symbol: context.symbol.clone(),
            data_points,
            statistics,
        })
    }

    fn required_data_fields(&self) -> Vec<DataField> {
        vec![DataField::Close]
    }

    fn computation_cost(&self) -> ComputationCost {
        ComputationCost::Low
    }

    async fn validate_data(&self, context: &FactorComputationContext) -> Result<()> {
        let window = context.factor.parameters.window_size.unwrap_or(20) as usize;
        let closes = context.market_data.get("close")
            .ok_or_else(|| FactorError::InsufficientData("No close price data".to_string()))?;

        if closes.len() < window {
            return Err(FactorError::InsufficientData(
                format!("Need at least {} data points, got {}", window, closes.len())
            ).into());
        }

        Ok(())
    }
}

/// RSI Calculator
struct RSICalculator;

#[async_trait]
impl FactorCalculator for RSICalculator {
    async fn calculate_point(
        &self,
        context: &FactorComputationContext,
        timestamp: DateTime<Utc>,
    ) -> Result<FactorResult> {
        let window = context.factor.parameters.window_size.unwrap_or(14) as usize;
        let closes = context.market_data.get("close")
            .ok_or_else(|| FactorError::InsufficientData("No close price data".to_string()))?;

        let rsi_values = self.calculate_rsi(closes, window)?;
        
        if rsi_values.is_empty() {
            return Err(FactorError::InsufficientData("Insufficient data for RSI calculation".to_string()).into());
        }

        let value = *rsi_values.last().unwrap();

        Ok(FactorResult {
            factor_id: context.factor.id.clone(),
            symbol: context.symbol.clone(),
            timestamp,
            value,
            percentile_rank: None,
            z_score: None,
            confidence: 0.95,
            metadata: FactorResultMetadata {
                computation_time_ms: 2.0,
                data_points_used: window as u32,
                data_quality_score: 1.0,
                cache_hit: false,
                warnings: vec![],
                debug_info: None,
            },
        })
    }

    async fn calculate_time_series(
        &self,
        context: &FactorComputationContext,
    ) -> Result<FactorTimeSeries> {
        let window = context.factor.parameters.window_size.unwrap_or(14) as usize;
        let closes = context.market_data.get("close")
            .ok_or_else(|| FactorError::InsufficientData("No close price data".to_string()))?;
        let timestamps = context.market_data.get("timestamp")
            .ok_or_else(|| FactorError::InsufficientData("No timestamp data".to_string()))?;

        let rsi_values = self.calculate_rsi(closes, window)?;
        
        if rsi_values.is_empty() {
            return Err(FactorError::InsufficientData("Insufficient data for RSI calculation".to_string()).into());
        }

        let mut data_points = Vec::new();
        for (i, &value) in rsi_values.iter().enumerate() {
            let timestamp_index = i + window;
            if timestamp_index < timestamps.len() {
                let timestamp = DateTime::from_timestamp(timestamps[timestamp_index] as i64, 0)
                    .unwrap_or(Utc::now());
                
                data_points.push(FactorDataPoint {
                    timestamp,
                    value,
                    volume: None,
                    quality_score: 1.0,
                });
            }
        }

        let statistics = TimeSeriesStatistics {
            count: data_points.len(),
            mean: rsi_values.iter().sum::<f64>() / rsi_values.len() as f64,
            median: {
                let mut sorted = rsi_values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sorted[sorted.len() / 2]
            },
            std_dev: MathUtils::standard_deviation(&rsi_values),
            min: rsi_values.iter().cloned().fold(f64::INFINITY, f64::min),
            max: rsi_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            skewness: 0.0,
            kurtosis: 0.0,
            first_timestamp: data_points.first().map(|p| p.timestamp).unwrap_or(Utc::now()),
            last_timestamp: data_points.last().map(|p| p.timestamp).unwrap_or(Utc::now()),
        };

        Ok(FactorTimeSeries {
            factor_id: context.factor.id.clone(),
            symbol: context.symbol.clone(),
            data_points,
            statistics,
        })
    }

    fn required_data_fields(&self) -> Vec<DataField> {
        vec![DataField::Close]
    }

    fn computation_cost(&self) -> ComputationCost {
        ComputationCost::Medium
    }

    async fn validate_data(&self, context: &FactorComputationContext) -> Result<()> {
        let window = context.factor.parameters.window_size.unwrap_or(14) as usize;
        let closes = context.market_data.get("close")
            .ok_or_else(|| FactorError::InsufficientData("No close price data".to_string()))?;

        if closes.len() < window + 1 {
            return Err(FactorError::InsufficientData(
                format!("Need at least {} data points for RSI, got {}", window + 1, closes.len())
            ).into());
        }

        Ok(())
    }
}

impl RSICalculator {
    fn calculate_rsi(&self, prices: &[f64], window: usize) -> Result<Vec<f64>> {
        if prices.len() <= window {
            return Ok(vec![]);
        }

        // Calculate price changes
        let mut gains = Vec::new();
        let mut losses = Vec::new();

        for i in 1..prices.len() {
            let change = prices[i] - prices[i - 1];
            gains.push(if change > 0.0 { change } else { 0.0 });
            losses.push(if change < 0.0 { -change } else { 0.0 });
        }

        // Calculate average gains and losses
        let mut avg_gains = Vec::new();
        let mut avg_losses = Vec::new();
        let mut rsi_values = Vec::new();

        for i in (window - 1)..gains.len() {
            let avg_gain = if i == window - 1 {
                gains[0..window].iter().sum::<f64>() / window as f64
            } else {
                (avg_gains[i - window] * (window - 1) as f64 + gains[i]) / window as f64
            };

            let avg_loss = if i == window - 1 {
                losses[0..window].iter().sum::<f64>() / window as f64
            } else {
                (avg_losses[i - window] * (window - 1) as f64 + losses[i]) / window as f64
            };

            avg_gains.push(avg_gain);
            avg_losses.push(avg_loss);

            let rs = if avg_loss == 0.0 {
                100.0
            } else {
                avg_gain / avg_loss
            };

            let rsi = 100.0 - (100.0 / (1.0 + rs));
            rsi_values.push(rsi);
        }

        Ok(rsi_values)
    }
}

// Additional calculator implementations would follow similar patterns...

/// Exponential Moving Average Calculator
struct ExponentialMovingAverageCalculator;

#[async_trait]
impl FactorCalculator for ExponentialMovingAverageCalculator {
    async fn calculate_point(&self, context: &FactorComputationContext, timestamp: DateTime<Utc>) -> Result<FactorResult> {
        let span = context.factor.parameters.window_size.unwrap_or(20) as f64;
        let alpha = 2.0 / (span + 1.0);
        let closes = context.market_data.get("close")
            .ok_or_else(|| FactorError::InsufficientData("No close price data".to_string()))?;

        let ema_values = MathUtils::exponential_moving_average(closes, alpha);
        let value = *ema_values.last().unwrap_or(&0.0);

        Ok(FactorResult {
            factor_id: context.factor.id.clone(),
            symbol: context.symbol.clone(),
            timestamp,
            value,
            percentile_rank: None,
            z_score: None,
            confidence: 0.95,
            metadata: FactorResultMetadata {
                computation_time_ms: 1.5,
                data_points_used: closes.len() as u32,
                data_quality_score: 1.0,
                cache_hit: false,
                warnings: vec![],
                debug_info: None,
            },
        })
    }

    async fn calculate_time_series(&self, _context: &FactorComputationContext) -> Result<FactorTimeSeries> {
        // Implementation similar to SMA but using EMA calculation
        todo!("Implement EMA time series calculation")
    }

    fn required_data_fields(&self) -> Vec<DataField> {
        vec![DataField::Close]
    }

    fn computation_cost(&self) -> ComputationCost {
        ComputationCost::Low
    }

    async fn validate_data(&self, _context: &FactorComputationContext) -> Result<()> {
        // Basic validation
        Ok(())
    }
}

// Placeholder implementations for other calculators
struct MACDCalculator;
struct BollingerBandsCalculator;
struct MomentumCalculator;
struct VolatilityCalculator;
struct VWAPCalculator;
struct StochasticCalculator;
struct WilliamsRCalculator;

macro_rules! impl_placeholder_calculator {
    ($calculator:ident) => {
        #[async_trait]
        impl FactorCalculator for $calculator {
            async fn calculate_point(&self, _context: &FactorComputationContext, _timestamp: DateTime<Utc>) -> Result<FactorResult> {
                todo!("Implement {} calculator", stringify!($calculator))
            }

            async fn calculate_time_series(&self, _context: &FactorComputationContext) -> Result<FactorTimeSeries> {
                todo!("Implement {} time series calculation", stringify!($calculator))
            }

            fn required_data_fields(&self) -> Vec<DataField> {
                vec![DataField::Close]
            }

            fn computation_cost(&self) -> ComputationCost {
                ComputationCost::Medium
            }

            async fn validate_data(&self, _context: &FactorComputationContext) -> Result<()> {
                Ok(())
            }
        }
    };
}

impl_placeholder_calculator!(MACDCalculator);
impl_placeholder_calculator!(BollingerBandsCalculator);
impl_placeholder_calculator!(MomentumCalculator);
impl_placeholder_calculator!(VolatilityCalculator);
impl_placeholder_calculator!(VWAPCalculator);
impl_placeholder_calculator!(StochasticCalculator);
impl_placeholder_calculator!(WilliamsRCalculator);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factors::models::*;

    #[tokio::test]
    async fn test_math_utils() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let sma = MathUtils::simple_moving_average(&data, 3);
        assert_eq!(sma.len(), 3);
        assert_eq!(sma[0], 2.0); // (1+2+3)/3
        assert_eq!(sma[1], 3.0); // (2+3+4)/3
        assert_eq!(sma[2], 4.0); // (3+4+5)/3
        
        let ema = MathUtils::exponential_moving_average(&data, 0.5);
        assert_eq!(ema.len(), 5);
        assert_eq!(ema[0], 1.0);
        
        let std_dev = MathUtils::standard_deviation(&data);
        assert!((std_dev - 1.5811388300841898).abs() < 1e-10);
        
        let corr = MathUtils::correlation(&data, &data);
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rsi_calculation() {
        let calculator = RSICalculator;
        let prices = vec![44.0, 44.25, 44.5, 43.75, 44.5, 44.0, 44.25, 44.75, 44.5, 44.0, 44.25];
        
        let rsi = calculator.calculate_rsi(&prices, 5).unwrap();
        assert!(!rsi.is_empty());
        
        // RSI should be between 0 and 100
        for &value in &rsi {
            assert!(value >= 0.0 && value <= 100.0);
        }
    }
}