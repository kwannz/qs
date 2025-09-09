use axum::{http::StatusCode, response::{IntoResponse, Response}, Json};
use serde_json::json;
use std::fmt;

#[allow(dead_code)]
#[derive(Debug)]
pub enum StrategyError {
    // Configuration Errors
    ConfigError(String),
    DatabaseError(String),
    
    // Strategy Management Errors
    StrategyNotFound(String),
    StrategyAlreadyExists(String),
    StrategyValidationError(String),
    StrategyExecutionError(String),
    
    // Market Data Errors
    MarketDataError(String),
    MarketDataUnavailable(String),
    InvalidSymbol(String),
    
    // Signal Generation Errors
    SignalGenerationError(String),
    InvalidStrategyType(String),
    InsufficientData(String),
    
    // Performance Tracking Errors
    PerformanceCalculationError(String),
    MetricsError(String),
    
    // Factor Analysis Errors
    FactorAnalysisError(String),
    IndicatorCalculationError(String),
    
    // Backtesting Errors
    BacktestError(String),
    InvalidBacktestParameters(String),
    
    // Resource Errors
    ResourceExhausted(String),
    ConcurrencyLimitExceeded(String),
    
    // External Service Errors
    ExternalServiceError(String),
    NetworkError(String),
    TimeoutError(String),
    
    // Validation Errors
    ValidationError(String),
    SerializationError(String),
    
    // Generic Errors
    InternalError(String),
    NotImplemented(String),
}

impl fmt::Display for StrategyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Configuration Errors
            StrategyError::ConfigError(msg) => write!(f, "Configuration error: {msg}"),
            StrategyError::DatabaseError(msg) => write!(f, "Database error: {msg}"),
            
            // Strategy Management Errors
            StrategyError::StrategyNotFound(msg) => write!(f, "Strategy not found: {msg}"),
            StrategyError::StrategyAlreadyExists(msg) => write!(f, "Strategy already exists: {msg}"),
            StrategyError::StrategyValidationError(msg) => write!(f, "Strategy validation error: {msg}"),
            StrategyError::StrategyExecutionError(msg) => write!(f, "Strategy execution error: {msg}"),
            
            // Market Data Errors
            StrategyError::MarketDataError(msg) => write!(f, "Market data error: {msg}"),
            StrategyError::MarketDataUnavailable(msg) => write!(f, "Market data unavailable: {msg}"),
            StrategyError::InvalidSymbol(msg) => write!(f, "Invalid symbol: {msg}"),
            
            // Signal Generation Errors
            StrategyError::SignalGenerationError(msg) => write!(f, "Signal generation error: {msg}"),
            StrategyError::InvalidStrategyType(msg) => write!(f, "Invalid strategy type: {msg}"),
            StrategyError::InsufficientData(msg) => write!(f, "Insufficient data: {msg}"),
            
            // Performance Tracking Errors
            StrategyError::PerformanceCalculationError(msg) => write!(f, "Performance calculation error: {msg}"),
            StrategyError::MetricsError(msg) => write!(f, "Metrics error: {msg}"),
            
            // Factor Analysis Errors
            StrategyError::FactorAnalysisError(msg) => write!(f, "Factor analysis error: {msg}"),
            StrategyError::IndicatorCalculationError(msg) => write!(f, "Indicator calculation error: {msg}"),
            
            // Backtesting Errors
            StrategyError::BacktestError(msg) => write!(f, "Backtest error: {msg}"),
            StrategyError::InvalidBacktestParameters(msg) => write!(f, "Invalid backtest parameters: {msg}"),
            
            // Resource Errors
            StrategyError::ResourceExhausted(msg) => write!(f, "Resource exhausted: {msg}"),
            StrategyError::ConcurrencyLimitExceeded(msg) => write!(f, "Concurrency limit exceeded: {msg}"),
            
            // External Service Errors
            StrategyError::ExternalServiceError(msg) => write!(f, "External service error: {msg}"),
            StrategyError::NetworkError(msg) => write!(f, "Network error: {msg}"),
            StrategyError::TimeoutError(msg) => write!(f, "Timeout error: {msg}"),
            
            // Validation Errors
            StrategyError::ValidationError(msg) => write!(f, "Validation error: {msg}"),
            StrategyError::SerializationError(msg) => write!(f, "Serialization error: {msg}"),
            
            // Generic Errors
            StrategyError::InternalError(msg) => write!(f, "Internal error: {msg}"),
            StrategyError::NotImplemented(msg) => write!(f, "Not implemented: {msg}"),
        }
    }
}

impl std::error::Error for StrategyError {}

impl IntoResponse for StrategyError {
    fn into_response(self) -> Response {
        let (status_code, error_type, message) = match &self {
            // Configuration Errors (500)
            StrategyError::ConfigError(msg) | StrategyError::DatabaseError(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "configuration_error", msg.as_str())
            }
            
            // Strategy Management Errors
            StrategyError::StrategyNotFound(msg) => {
                (StatusCode::NOT_FOUND, "strategy_not_found", msg.as_str())
            }
            StrategyError::StrategyAlreadyExists(msg) => {
                (StatusCode::CONFLICT, "strategy_already_exists", msg.as_str())
            }
            StrategyError::StrategyValidationError(msg) => {
                (StatusCode::BAD_REQUEST, "strategy_validation_error", msg.as_str())
            }
            StrategyError::StrategyExecutionError(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "strategy_execution_error", msg.as_str())
            }
            
            // Market Data Errors
            StrategyError::MarketDataError(msg) | StrategyError::MarketDataUnavailable(msg) => {
                (StatusCode::SERVICE_UNAVAILABLE, "market_data_error", msg.as_str())
            }
            StrategyError::InvalidSymbol(msg) => {
                (StatusCode::BAD_REQUEST, "invalid_symbol", msg.as_str())
            }
            
            // Signal Generation Errors
            StrategyError::SignalGenerationError(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "signal_generation_error", msg.as_str())
            }
            StrategyError::InvalidStrategyType(msg) => {
                (StatusCode::BAD_REQUEST, "invalid_strategy_type", msg.as_str())
            }
            StrategyError::InsufficientData(msg) => {
                (StatusCode::UNPROCESSABLE_ENTITY, "insufficient_data", msg.as_str())
            }
            
            // Performance Tracking Errors
            StrategyError::PerformanceCalculationError(msg) | StrategyError::MetricsError(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "performance_error", msg.as_str())
            }
            
            // Factor Analysis Errors
            StrategyError::FactorAnalysisError(msg) | StrategyError::IndicatorCalculationError(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "analysis_error", msg.as_str())
            }
            
            // Backtesting Errors
            StrategyError::BacktestError(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "backtest_error", msg.as_str())
            }
            StrategyError::InvalidBacktestParameters(msg) => {
                (StatusCode::BAD_REQUEST, "invalid_backtest_parameters", msg.as_str())
            }
            
            // Resource Errors
            StrategyError::ResourceExhausted(msg) | StrategyError::ConcurrencyLimitExceeded(msg) => {
                (StatusCode::TOO_MANY_REQUESTS, "resource_exhausted", msg.as_str())
            }
            
            // External Service Errors
            StrategyError::ExternalServiceError(msg) => {
                (StatusCode::BAD_GATEWAY, "external_service_error", msg.as_str())
            }
            StrategyError::NetworkError(msg) => {
                (StatusCode::SERVICE_UNAVAILABLE, "network_error", msg.as_str())
            }
            StrategyError::TimeoutError(msg) => {
                (StatusCode::REQUEST_TIMEOUT, "timeout_error", msg.as_str())
            }
            
            // Validation Errors
            StrategyError::ValidationError(msg) | StrategyError::SerializationError(msg) => {
                (StatusCode::BAD_REQUEST, "validation_error", msg.as_str())
            }
            
            // Generic Errors
            StrategyError::InternalError(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", msg.as_str())
            }
            StrategyError::NotImplemented(msg) => {
                (StatusCode::NOT_IMPLEMENTED, "not_implemented", msg.as_str())
            }
        };

        let body = Json(json!({
            "error": error_type,
            "message": message,
            "status": status_code.as_u16()
        }));

        (status_code, body).into_response()
    }
}

// Conversion implementations for common error types

impl From<anyhow::Error> for StrategyError {
    fn from(err: anyhow::Error) -> Self {
        StrategyError::InternalError(err.to_string())
    }
}

impl From<serde_json::Error> for StrategyError {
    fn from(err: serde_json::Error) -> Self {
        StrategyError::SerializationError(err.to_string())
    }
}

impl From<reqwest::Error> for StrategyError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            StrategyError::TimeoutError(err.to_string())
        } else if err.is_connect() {
            StrategyError::NetworkError(err.to_string())
        } else {
            StrategyError::ExternalServiceError(err.to_string())
        }
    }
}

impl From<tokio::time::error::Elapsed> for StrategyError {
    fn from(err: tokio::time::error::Elapsed) -> Self {
        StrategyError::TimeoutError(err.to_string())
    }
}

// Result type alias for convenience
#[allow(dead_code)]
pub type StrategyResult<T> = Result<T, StrategyError>;

// Helper functions for common error scenarios

#[allow(dead_code)]
pub fn strategy_not_found(strategy_id: &str) -> StrategyError {
    StrategyError::StrategyNotFound(format!("Strategy with ID '{strategy_id}' not found"))
}

#[allow(dead_code)]
pub fn invalid_strategy_parameters(details: &str) -> StrategyError {
    StrategyError::StrategyValidationError(format!("Invalid strategy parameters: {details}"))
}

#[allow(dead_code)]
pub fn market_data_unavailable(symbol: &str) -> StrategyError {
    StrategyError::MarketDataUnavailable(format!("Market data unavailable for symbol: {symbol}"))
}

#[allow(dead_code)]
pub fn insufficient_historical_data(symbol: &str, required: usize, available: usize) -> StrategyError {
    StrategyError::InsufficientData(format!(
        "Insufficient historical data for {symbol}: required {required}, available {available}"
    ))
}

#[allow(dead_code)]
pub fn concurrency_limit_exceeded(current: usize, limit: usize) -> StrategyError {
    StrategyError::ConcurrencyLimitExceeded(format!(
        "Concurrency limit exceeded: {current} active strategies, limit is {limit}"
    ))
}

#[allow(dead_code)]
pub fn indicator_calculation_failed(indicator: &str, reason: &str) -> StrategyError {
    StrategyError::IndicatorCalculationError(format!(
        "Failed to calculate {indicator}: {reason}"
    ))
}

#[allow(dead_code)]
pub fn signal_generation_failed(strategy_type: &str, reason: &str) -> StrategyError {
    StrategyError::SignalGenerationError(format!(
        "Failed to generate signals for {strategy_type} strategy: {reason}"
    ))
}

// Error context helpers

#[allow(dead_code)]
pub trait StrategyErrorExt<T> {
    fn with_strategy_context(self, strategy_id: &str) -> StrategyResult<T>;
    fn with_symbol_context(self, symbol: &str) -> StrategyResult<T>;
    fn with_indicator_context(self, indicator: &str) -> StrategyResult<T>;
}

impl<T, E: Into<StrategyError>> StrategyErrorExt<T> for Result<T, E> {
    fn with_strategy_context(self, strategy_id: &str) -> StrategyResult<T> {
        self.map_err(|e| {
            let base_error = e.into();
            StrategyError::StrategyExecutionError(format!(
                "Strategy {strategy_id}: {base_error}"
            ))
        })
    }

    fn with_symbol_context(self, symbol: &str) -> StrategyResult<T> {
        self.map_err(|e| {
            let base_error = e.into();
            StrategyError::MarketDataError(format!(
                "Symbol {symbol}: {base_error}"
            ))
        })
    }

    fn with_indicator_context(self, indicator: &str) -> StrategyResult<T> {
        self.map_err(|e| {
            let base_error = e.into();
            StrategyError::IndicatorCalculationError(format!(
                "Indicator {indicator}: {base_error}"
            ))
        })
    }
}