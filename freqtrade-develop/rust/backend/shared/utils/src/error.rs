use std::fmt;

/// Common error handling utilities
pub struct ErrorUtils;

impl ErrorUtils {
    /// Create a formatted error message with context
    pub fn format_error_with_context(error: &dyn std::error::Error, context: &str) -> String {
        format!("Error in {}: {}", context, error)
    }

    /// Chain error messages for better debugging
    pub fn chain_error_messages(error: &dyn std::error::Error) -> String {
        let mut messages = vec![error.to_string()];
        let mut source = error.source();
        
        while let Some(err) = source {
            messages.push(err.to_string());
            source = err.source();
        }
        
        messages.join(" -> ")
    }

    /// Log error with appropriate level based on error type
    pub fn log_error(error: &PlatformError) {
        match error.severity() {
            ErrorSeverity::Critical => tracing::error!("CRITICAL: {}", error),
            ErrorSeverity::High => tracing::error!("{}", error),
            ErrorSeverity::Medium => tracing::warn!("{}", error),
            ErrorSeverity::Low => tracing::info!("{}", error),
        }
    }
}

/// Platform-wide error types
#[derive(Debug)]
pub enum PlatformError {
    Configuration { message: String },
    Network { message: String, code: Option<u16> },
    Database { message: String, query: Option<String> },
    Authentication { message: String },
    Authorization { message: String, resource: String },
    Validation { message: String, field: String },
    Internal { message: String, source: Option<Box<dyn std::error::Error + Send + Sync>> },
}

impl PlatformError {
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            PlatformError::Configuration { .. } | PlatformError::Internal { .. } => ErrorSeverity::Critical,
            PlatformError::Network { code: Some(5..), .. } | PlatformError::Database { .. } => ErrorSeverity::High,
            PlatformError::Network { .. } | PlatformError::Authentication { .. } | PlatformError::Authorization { .. } => ErrorSeverity::Medium,
            PlatformError::Validation { .. } => ErrorSeverity::Low,
        }
    }

    /// Get error category for metrics and monitoring
    pub fn category(&self) -> &'static str {
        match self {
            PlatformError::Configuration { .. } => "configuration",
            PlatformError::Network { .. } => "network",
            PlatformError::Database { .. } => "database",
            PlatformError::Authentication { .. } => "auth",
            PlatformError::Authorization { .. } => "authz",
            PlatformError::Validation { .. } => "validation",
            PlatformError::Internal { .. } => "internal",
        }
    }
}

impl fmt::Display for PlatformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlatformError::Configuration { message } => {
                write!(f, "Configuration error: {}", message)
            }
            PlatformError::Network { message, code } => {
                if let Some(code) = code {
                    write!(f, "Network error ({}): {}", code, message)
                } else {
                    write!(f, "Network error: {}", message)
                }
            }
            PlatformError::Database { message, query } => {
                if let Some(query) = query {
                    write!(f, "Database error: {} (query: {})", message, query)
                } else {
                    write!(f, "Database error: {}", message)
                }
            }
            PlatformError::Authentication { message } => {
                write!(f, "Authentication error: {}", message)
            }
            PlatformError::Authorization { message, resource } => {
                write!(f, "Authorization error for '{}': {}", resource, message)
            }
            PlatformError::Validation { message, field } => {
                write!(f, "Validation error for '{}': {}", field, message)
            }
            PlatformError::Internal { message, source } => {
                if let Some(source) = source {
                    write!(f, "Internal error: {} (caused by: {})", message, source)
                } else {
                    write!(f, "Internal error: {}", message)
                }
            }
        }
    }
}

impl std::error::Error for PlatformError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PlatformError::Internal { source: Some(source), .. } => {
                Some(source.as_ref())
            }
            _ => None,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Critical,
    High,
    Medium,
    Low,
}

/// Result type alias for platform operations
pub type PlatformResult<T> = Result<T, PlatformError>;