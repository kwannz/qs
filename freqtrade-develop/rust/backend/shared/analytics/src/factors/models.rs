use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap};
use uuid::Uuid;

/// Core factor data structure
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Factor {
    pub id: String,
    pub name: String,
    pub category: FactorCategory,
    pub description: Option<String>,
    pub formula: Option<String>,
    pub parameters: FactorParameters,
    pub metadata: FactorMetadata,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Factor categories for organization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FactorCategory {
    Technical,
    Fundamental,
    Sentiment,
    Macro,
    Alternative,
    Risk,
    Momentum,
    Mean_reversion,
    Volatility,
    Volume,
    Custom(String),
}

impl std::fmt::Display for FactorCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FactorCategory::Technical => write!(f, "technical"),
            FactorCategory::Fundamental => write!(f, "fundamental"),
            FactorCategory::Sentiment => write!(f, "sentiment"),
            FactorCategory::Macro => write!(f, "macro"),
            FactorCategory::Alternative => write!(f, "alternative"),
            FactorCategory::Risk => write!(f, "risk"),
            FactorCategory::Momentum => write!(f, "momentum"),
            FactorCategory::Mean_reversion => write!(f, "mean_reversion"),
            FactorCategory::Volatility => write!(f, "volatility"),
            FactorCategory::Volume => write!(f, "volume"),
            FactorCategory::Custom(s) => write!(f, "custom_{}", s),
        }
    }
}

/// Factor computation parameters
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FactorParameters {
    pub window_size: Option<u64>,
    pub lookback_period: Option<Duration>,
    pub smoothing_factor: Option<f64>,
    pub custom_params: HashMap<String, FactorParameterValue>,
}

impl Default for FactorParameters {
    fn default() -> Self {
        Self {
            window_size: Some(20),
            lookback_period: Some(Duration::days(30)),
            smoothing_factor: Some(0.1),
            custom_params: HashMap::new(),
        }
    }
}

/// Flexible parameter value types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FactorParameterValue {
    Number(f64),
    Integer(i64),
    Text(String),
    Boolean(bool),
    Array(Vec<f64>),
    Duration(Duration),
}

/// Factor metadata and performance metrics
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FactorMetadata {
    pub version: u32,
    pub computation_cost: ComputationCost,
    pub dependencies: Vec<String>,
    pub data_requirements: DataRequirements,
    pub performance_metrics: Option<FactorPerformanceMetrics>,
    pub tags: Vec<String>,
    pub author: Option<String>,
}

impl Default for FactorMetadata {
    fn default() -> Self {
        Self {
            version: 1,
            computation_cost: ComputationCost::Low,
            dependencies: Vec::new(),
            data_requirements: DataRequirements::default(),
            performance_metrics: None,
            tags: Vec::new(),
            author: None,
        }
    }
}

/// Computation cost classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComputationCost {
    Low,    // Simple calculations, O(n)
    Medium, // Moderate complexity, O(n log n)
    High,   // Complex calculations, O(n²) or higher
    Critical, // Extremely expensive, requires special handling
}

/// Data requirements for factor computation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataRequirements {
    pub required_fields: Vec<DataField>,
    pub optional_fields: Vec<DataField>,
    pub minimum_history_days: u32,
    pub update_frequency: UpdateFrequency,
    pub data_sources: Vec<String>,
}

impl Default for DataRequirements {
    fn default() -> Self {
        Self {
            required_fields: vec![DataField::Price],
            optional_fields: Vec::new(),
            minimum_history_days: 30,
            update_frequency: UpdateFrequency::Daily,
            data_sources: Vec::new(),
        }
    }
}

/// Data fields required for factor computation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DataField {
    Price,
    Volume,
    Open,
    High,
    Low,
    Close,
    AdjustedClose,
    Dividend,
    Split,
    MarketCap,
    BookValue,
    Revenue,
    Earnings,
    Volatility,
    OpenInterest,
    Custom(String),
}

/// Factor update frequency
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum UpdateFrequency {
    RealTime,
    Intraday(Duration),
    Daily,
    Weekly,
    Monthly,
    OnDemand,
}

/// Comprehensive factor performance metrics
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FactorPerformanceMetrics {
    // Information coefficient metrics
    pub ic: f64,
    pub rank_ic: f64,
    pub ic_information_ratio: f64,
    pub ic_skewness: f64,
    pub ic_kurtosis: f64,
    
    // Statistical significance
    pub t_statistic: f64,
    pub p_value: f64,
    pub confidence_interval: (f64, f64),
    
    // Return metrics
    pub annualized_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub max_drawdown: f64,
    pub var_95: f64,
    pub cvar_95: f64,
    
    // Factor characteristics
    pub turnover: f64,
    pub decay_rate: f64,
    pub autocorrelation: f64,
    pub momentum_strength: f64,
    pub mean_reversion_strength: f64,
    
    // Cross-sectional analysis
    pub cross_sectional_dispersion: f64,
    pub factor_exposure_coverage: f64,
    pub neutrality_test_p_value: f64,
    
    // Regime analysis
    pub regime_stability: f64,
    pub bull_market_ic: f64,
    pub bear_market_ic: f64,
    pub high_vol_ic: f64,
    pub low_vol_ic: f64,
    
    // Timing and execution
    pub average_computation_time_ms: f64,
    pub data_freshness_score: f64,
    pub signal_strength: f64,
    
    pub last_calculated: DateTime<Utc>,
    pub calculation_window: Duration,
}

/// Factor computation result with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorResult {
    pub factor_id: String,
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub percentile_rank: Option<f64>,
    pub z_score: Option<f64>,
    pub confidence: f64,
    pub metadata: FactorResultMetadata,
}

/// Metadata for factor computation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorResultMetadata {
    pub computation_time_ms: f64,
    pub data_points_used: u32,
    pub data_quality_score: f64,
    pub cache_hit: bool,
    pub warnings: Vec<String>,
    pub debug_info: Option<HashMap<String, String>>,
}

/// Time series of factor values for a symbol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorTimeSeries {
    pub factor_id: String,
    pub symbol: String,
    pub data_points: Vec<FactorDataPoint>,
    pub statistics: TimeSeriesStatistics,
}

/// Individual data point in factor time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorDataPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub volume: Option<f64>,
    pub quality_score: f64,
}

/// Statistical summary of factor time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesStatistics {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub first_timestamp: DateTime<Utc>,
    pub last_timestamp: DateTime<Utc>,
}

/// Factor universe - collection of factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorUniverse {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub factors: Vec<Factor>,
    pub correlation_matrix: Option<FactorCorrelationMatrix>,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

/// Correlation matrix for factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorCorrelationMatrix {
    pub factor_ids: Vec<String>,
    pub correlations: Vec<Vec<f64>>,
    pub calculation_date: DateTime<Utc>,
    pub window_days: u32,
}

/// Factor screening criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorScreeningCriteria {
    pub min_ic: Option<f64>,
    pub max_ic: Option<f64>,
    pub min_ic_ir: Option<f64>,
    pub max_turnover: Option<f64>,
    pub min_sharpe: Option<f64>,
    pub max_correlation: Option<f64>,
    pub categories: Option<Vec<FactorCategory>>,
    pub tags: Option<Vec<String>>,
    pub min_data_coverage: Option<f64>,
    pub max_computation_cost: Option<ComputationCost>,
}

/// Batch computation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorBatchRequest {
    pub id: String,
    pub factor_ids: Vec<String>,
    pub symbols: Vec<String>,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub frequency: UpdateFrequency,
    pub priority: BatchPriority,
    pub callback_url: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// Priority levels for batch processing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum BatchPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// Batch computation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorBatchStatus {
    pub request_id: String,
    pub status: BatchStatusType,
    pub progress_percentage: f32,
    pub started_at: Option<DateTime<Utc>>,
    pub estimated_completion: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub results_ready: bool,
    pub error_message: Option<String>,
    pub processed_factors: u32,
    pub total_factors: u32,
}

/// Batch processing status types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BatchStatusType {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
    PartiallyCompleted,
}

/// Factor computation context
#[derive(Debug, Clone)]
pub struct FactorComputationContext {
    pub factor: Factor,
    pub symbol: String,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub market_data: HashMap<String, Vec<f64>>,
    pub custom_data: HashMap<String, HashMap<String, f64>>,
}

/// Error types specific to factor operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactorError {
    InvalidParameters(String),
    InsufficientData(String),
    ComputationFailed(String),
    DataQualityIssue(String),
    DependencyMissing(String),
    CacheError(String),
    DatabaseError(String),
    NetworkError(String),
    TimeoutError(String),
    UnknownFactor(String),
}

impl std::fmt::Display for FactorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FactorError::InvalidParameters(msg) => write!(f, "Invalid parameters: {}", msg),
            FactorError::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
            FactorError::ComputationFailed(msg) => write!(f, "Computation failed: {}", msg),
            FactorError::DataQualityIssue(msg) => write!(f, "Data quality issue: {}", msg),
            FactorError::DependencyMissing(msg) => write!(f, "Dependency missing: {}", msg),
            FactorError::CacheError(msg) => write!(f, "Cache error: {}", msg),
            FactorError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            FactorError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            FactorError::TimeoutError(msg) => write!(f, "Timeout error: {}", msg),
            FactorError::UnknownFactor(msg) => write!(f, "Unknown factor: {}", msg),
        }
    }
}

impl std::error::Error for FactorError {}

// Helper functions for creating common factors
impl Factor {
    /// Create a new factor with default settings
    pub fn new(name: String, category: FactorCategory) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            category,
            description: None,
            formula: None,
            parameters: FactorParameters::default(),
            metadata: FactorMetadata::default(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    /// Create a simple moving average factor
    pub fn simple_moving_average(window: u64) -> Self {
        let mut factor = Self::new(
            format!("SMA_{}", window),
            FactorCategory::Technical,
        );
        factor.description = Some(format!("Simple Moving Average with {} period window", window));
        factor.parameters.window_size = Some(window);
        factor.formula = Some(format!("SUM(CLOSE, {}) / {}", window, window));
        factor
    }

    /// Create a RSI factor
    pub fn rsi(window: u64) -> Self {
        let mut factor = Self::new(
            format!("RSI_{}", window),
            FactorCategory::Momentum,
        );
        factor.description = Some(format!("Relative Strength Index with {} period window", window));
        factor.parameters.window_size = Some(window);
        factor.formula = Some("100 - (100 / (1 + RS))".to_string());
        factor.metadata.computation_cost = ComputationCost::Medium;
        factor
    }

    /// Create a Bollinger Bands factor
    pub fn bollinger_bands(window: u64, std_dev: f64) -> Self {
        let mut factor = Self::new(
            format!("BB_{}_{}", window, std_dev),
            FactorCategory::Technical,
        );
        factor.description = Some(format!("Bollinger Bands with {} period window and {} standard deviations", window, std_dev));
        factor.parameters.window_size = Some(window);
        factor.parameters.custom_params.insert(
            "std_multiplier".to_string(),
            FactorParameterValue::Number(std_dev),
        );
        factor.formula = Some("(CLOSE - SMA) / (STD * multiplier)".to_string());
        factor
    }
}

impl FactorUniverse {
    /// Create a new factor universe
    pub fn new(name: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            description: None,
            factors: Vec::new(),
            correlation_matrix: None,
            created_at: Utc::now(),
            last_updated: Utc::now(),
        }
    }

    /// Add a factor to the universe
    pub fn add_factor(&mut self, factor: Factor) {
        self.factors.push(factor);
        self.last_updated = Utc::now();
    }

    /// Get factors by category
    pub fn get_factors_by_category(&self, category: &FactorCategory) -> Vec<&Factor> {
        self.factors.iter()
            .filter(|f| &f.category == category)
            .collect()
    }

    /// Filter factors by screening criteria
    pub fn screen_factors(&self, criteria: &FactorScreeningCriteria) -> Vec<&Factor> {
        self.factors.iter()
            .filter(|factor| {
                // Check category filter
                if let Some(ref categories) = criteria.categories {
                    if !categories.contains(&factor.category) {
                        return false;
                    }
                }

                // Check computation cost
                if let Some(ref max_cost) = criteria.max_computation_cost {
                    if factor.metadata.computation_cost > *max_cost {
                        return false;
                    }
                }

                // Check performance metrics if available
                if let Some(ref metrics) = factor.metadata.performance_metrics {
                    if let Some(min_ic) = criteria.min_ic {
                        if metrics.ic < min_ic {
                            return false;
                        }
                    }
                    if let Some(max_ic) = criteria.max_ic {
                        if metrics.ic > max_ic {
                            return false;
                        }
                    }
                    if let Some(min_ic_ir) = criteria.min_ic_ir {
                        if metrics.ic_information_ratio < min_ic_ir {
                            return false;
                        }
                    }
                    if let Some(max_turnover) = criteria.max_turnover {
                        if metrics.turnover > max_turnover {
                            return false;
                        }
                    }
                    if let Some(min_sharpe) = criteria.min_sharpe {
                        if metrics.sharpe_ratio < min_sharpe {
                            return false;
                        }
                    }
                }

                true
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_creation() {
        let factor = Factor::simple_moving_average(20);
        assert_eq!(factor.name, "SMA_20");
        assert_eq!(factor.category, FactorCategory::Technical);
        assert_eq!(factor.parameters.window_size, Some(20));
    }

    #[test]
    fn test_factor_universe() {
        let mut universe = FactorUniverse::new("Test Universe".to_string());
        let sma = Factor::simple_moving_average(20);
        let rsi = Factor::rsi(14);
        
        universe.add_factor(sma);
        universe.add_factor(rsi);
        
        assert_eq!(universe.factors.len(), 2);
        
        let technical_factors = universe.get_factors_by_category(&FactorCategory::Technical);
        assert_eq!(technical_factors.len(), 1);
        
        let momentum_factors = universe.get_factors_by_category(&FactorCategory::Momentum);
        assert_eq!(momentum_factors.len(), 1);
    }

    #[test]
    fn test_factor_screening() {
        let mut universe = FactorUniverse::new("Test Universe".to_string());
        
        // Add factors with performance metrics
        let mut low_ic_factor = Factor::simple_moving_average(20);
        low_ic_factor.metadata.performance_metrics = Some(FactorPerformanceMetrics {
            ic: 0.02,
            rank_ic: 0.01,
            ic_information_ratio: 0.5,
            sharpe_ratio: 0.3,
            turnover: 0.8,
            ..Default::default()
        });
        
        let mut high_ic_factor = Factor::rsi(14);
        high_ic_factor.metadata.performance_metrics = Some(FactorPerformanceMetrics {
            ic: 0.08,
            rank_ic: 0.06,
            ic_information_ratio: 1.5,
            sharpe_ratio: 0.8,
            turnover: 0.6,
            ..Default::default()
        });
        
        universe.add_factor(low_ic_factor);
        universe.add_factor(high_ic_factor);
        
        let criteria = FactorScreeningCriteria {
            min_ic: Some(0.05),
            min_sharpe: Some(0.5),
            ..Default::default()
        };
        
        let filtered = universe.screen_factors(&criteria);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "RSI_14");
    }
}

// Default implementation for performance metrics
impl Default for FactorPerformanceMetrics {
    fn default() -> Self {
        Self {
            ic: 0.0,
            rank_ic: 0.0,
            ic_information_ratio: 0.0,
            ic_skewness: 0.0,
            ic_kurtosis: 0.0,
            t_statistic: 0.0,
            p_value: 1.0,
            confidence_interval: (0.0, 0.0),
            annualized_return: 0.0,
            volatility: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            calmar_ratio: 0.0,
            max_drawdown: 0.0,
            var_95: 0.0,
            cvar_95: 0.0,
            turnover: 0.0,
            decay_rate: 0.0,
            autocorrelation: 0.0,
            momentum_strength: 0.0,
            mean_reversion_strength: 0.0,
            cross_sectional_dispersion: 0.0,
            factor_exposure_coverage: 0.0,
            neutrality_test_p_value: 1.0,
            regime_stability: 0.0,
            bull_market_ic: 0.0,
            bear_market_ic: 0.0,
            high_vol_ic: 0.0,
            low_vol_ic: 0.0,
            average_computation_time_ms: 0.0,
            data_freshness_score: 0.0,
            signal_strength: 0.0,
            last_calculated: Utc::now(),
            calculation_window: Duration::days(252),
        }
    }
}

impl Default for FactorScreeningCriteria {
    fn default() -> Self {
        Self {
            min_ic: None,
            max_ic: None,
            min_ic_ir: None,
            max_turnover: None,
            min_sharpe: None,
            max_correlation: None,
            categories: None,
            tags: None,
            min_data_coverage: None,
            max_computation_cost: None,
        }
    }
}