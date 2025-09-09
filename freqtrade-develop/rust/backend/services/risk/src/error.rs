use axum::{
    http::StatusCode,
    response::{IntoResponse, Json},
};
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum RiskServiceError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Configuration error: {0}")]
    Config(#[from] config::ConfigError),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Validation error: {message}")]
    Validation { message: String },

    #[error("Risk limit exceeded: {message}")]
    RiskLimitExceeded { message: String },

    #[error("Insufficient margin: required {required}, available {available}")]
    InsufficientMargin { required: String, available: String },

    #[error("Position not found: {symbol}")]
    PositionNotFound { symbol: String },

    #[error("Invalid leverage: {leverage}, max allowed: {max_allowed}")]
    InvalidLeverage { leverage: String, max_allowed: String },

    #[error("Internal error: {0}")]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for RiskServiceError {
    fn into_response(self) -> axum::response::Response {
        let (status, error_code, message) = match &self {
            RiskServiceError::Database(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "DATABASE_ERROR",
                "Database operation failed",
            ),
            RiskServiceError::Config(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "CONFIG_ERROR",
                "Configuration error",
            ),
            RiskServiceError::Serialization(_) => (
                StatusCode::BAD_REQUEST,
                "SERIALIZATION_ERROR",
                "Invalid request format",
            ),
            RiskServiceError::Validation { message } => (
                StatusCode::BAD_REQUEST,
                "VALIDATION_ERROR",
                message.as_str(),
            ),
            RiskServiceError::RiskLimitExceeded { message } => (
                StatusCode::FORBIDDEN,
                "RISK_LIMIT_EXCEEDED",
                message.as_str(),
            ),
            RiskServiceError::InsufficientMargin { required: _required, available: _available } => (
                StatusCode::FORBIDDEN,
                "INSUFFICIENT_MARGIN",
                "Insufficient margin",
            ),
            RiskServiceError::PositionNotFound { symbol: _symbol } => (
                StatusCode::NOT_FOUND,
                "POSITION_NOT_FOUND",
                "Position not found",
            ),
            RiskServiceError::InvalidLeverage { leverage: _leverage, max_allowed: _max_allowed } => (
                StatusCode::BAD_REQUEST,
                "INVALID_LEVERAGE",
                "Invalid leverage",
            ),
            RiskServiceError::Internal(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL_ERROR",
                "Internal server error",
            ),
        };

        let error_response = json!({
            "error": {
                "code": error_code,
                "message": message,
                "timestamp": chrono::Utc::now().to_rfc3339(),
            }
        });

        (status, Json(error_response)).into_response()
    }
}

#[allow(dead_code)]
pub type RiskResult<T> = Result<T, RiskServiceError>;