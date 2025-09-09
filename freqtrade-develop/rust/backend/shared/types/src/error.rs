use thiserror::Error;

/// 平台错误类型
#[derive(Error, Debug)]
pub enum PlatformError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Authentication error: {0}")]
    Authentication(String),

    #[error("Authorization error: {0}")]
    Authorization(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Business logic error: {0}")]
    Business(String),

    #[error("External service error: {0}")]
    ExternalService(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Conflict: {0}")]
    Conflict(String),

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    #[error("Timeout: {0}")]
    Timeout(String),
}

/// 结果类型别名
pub type PlatformResult<T> = Result<T, PlatformError>;

/// 错误详情
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ErrorDetail {
    pub code: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
    pub trace_id: Option<String>,
}

impl ErrorDetail {
    pub fn new(code: &str, message: &str) -> Self {
        Self {
            code: code.to_string(),
            message: message.to_string(),
            details: None,
            trace_id: None,
        }
    }

    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }

    pub fn with_trace_id(mut self, trace_id: String) -> Self {
        self.trace_id = Some(trace_id);
        self
    }
}

impl From<PlatformError> for ErrorDetail {
    fn from(err: PlatformError) -> Self {
        let (code, message) = match &err {
            PlatformError::Config(_) => ("CONFIG_ERROR", err.to_string()),
            PlatformError::Database(_) => ("DATABASE_ERROR", err.to_string()),
            PlatformError::Network(_) => ("NETWORK_ERROR", err.to_string()),
            PlatformError::Authentication(_) => ("AUTH_ERROR", err.to_string()),
            PlatformError::Authorization(_) => ("AUTHORIZATION_ERROR", err.to_string()),
            PlatformError::Validation(_) => ("VALIDATION_ERROR", err.to_string()),
            PlatformError::Business(_) => ("BUSINESS_ERROR", err.to_string()),
            PlatformError::ExternalService(_) => ("EXTERNAL_SERVICE_ERROR", err.to_string()),
            PlatformError::Internal(_) => ("INTERNAL_ERROR", err.to_string()),
            PlatformError::NotFound(_) => ("NOT_FOUND", err.to_string()),
            PlatformError::Conflict(_) => ("CONFLICT", err.to_string()),
            PlatformError::RateLimit => ("RATE_LIMIT", "Rate limit exceeded".to_string()),
            PlatformError::ServiceUnavailable(_) => ("SERVICE_UNAVAILABLE", err.to_string()),
            PlatformError::Timeout(_) => ("TIMEOUT", err.to_string()),
        };

        ErrorDetail::new(code, &message)
    }
}