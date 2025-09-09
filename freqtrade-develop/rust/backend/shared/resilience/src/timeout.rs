use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, warn};

/// Timeout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    pub default_timeout: Duration,
    pub operation_timeouts: std::collections::HashMap<String, Duration>,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_secs(30),
            operation_timeouts: std::collections::HashMap::new(),
        }
    }
}

/// Timeout manager for handling operation timeouts
pub struct TimeoutManager {
    config: TimeoutConfig,
}

impl TimeoutManager {
    pub fn new(config: TimeoutConfig) -> Self {
        Self { config }
    }

    /// Execute an operation with timeout
    pub async fn execute<F, T>(
        &self,
        operation_name: &str,
        future: F,
    ) -> Result<T, TimeoutError>
    where
        F: std::future::Future<Output = T>,
    {
        let timeout_duration = self.get_timeout_for_operation(operation_name);
        
        debug!(
            operation = operation_name,
            timeout = ?timeout_duration,
            "Executing operation with timeout"
        );

        match timeout(timeout_duration, future).await {
            Ok(result) => {
                debug!(
                    operation = operation_name,
                    timeout = ?timeout_duration,
                    "Operation completed within timeout"
                );
                Ok(result)
            }
            Err(_) => {
                warn!(
                    operation = operation_name,
                    timeout = ?timeout_duration,
                    "Operation timed out"
                );
                Err(TimeoutError {
                    operation: operation_name.to_string(),
                    timeout: timeout_duration,
                })
            }
        }
    }

    /// Get timeout for a specific operation
    fn get_timeout_for_operation(&self, operation_name: &str) -> Duration {
        self.config
            .operation_timeouts
            .get(operation_name)
            .copied()
            .unwrap_or(self.config.default_timeout)
    }

    /// Set timeout for a specific operation
    pub fn set_operation_timeout(&mut self, operation: String, timeout: Duration) {
        self.config.operation_timeouts.insert(operation, timeout);
    }

    /// Remove timeout for a specific operation (will use default)
    pub fn remove_operation_timeout(&mut self, operation: &str) {
        self.config.operation_timeouts.remove(operation);
    }
}

/// Timeout error
#[derive(Debug, thiserror::Error)]
#[error("Operation '{operation}' timed out after {timeout:?}")]
pub struct TimeoutError {
    pub operation: String,
    pub timeout: Duration,
}

/// Timeout wrapper for easy use
pub struct TimeoutWrapper {
    timeout_duration: Duration,
    operation_name: String,
}

impl TimeoutWrapper {
    pub fn new(operation_name: String, timeout_duration: Duration) -> Self {
        Self {
            timeout_duration,
            operation_name,
        }
    }

    pub async fn execute<F, T>(&self, future: F) -> Result<T, TimeoutError>
    where
        F: std::future::Future<Output = T>,
    {
        debug!(
            operation = %self.operation_name,
            timeout = ?self.timeout_duration,
            "Executing operation with timeout wrapper"
        );

        match timeout(self.timeout_duration, future).await {
            Ok(result) => Ok(result),
            Err(_) => {
                warn!(
                    operation = %self.operation_name,
                    timeout = ?self.timeout_duration,
                    "Operation timed out in wrapper"
                );
                Err(TimeoutError {
                    operation: self.operation_name.clone(),
                    timeout: self.timeout_duration,
                })
            }
        }
    }
}

/// Convenience functions for common timeout scenarios
pub mod presets {
    use super::*;

    /// Create a timeout manager with common presets
    pub fn create_default_timeout_manager() -> TimeoutManager {
        let mut config = TimeoutConfig::default();
        
        // Database operation timeouts
        config.operation_timeouts.insert("database_query".to_string(), Duration::from_secs(10));
        config.operation_timeouts.insert("database_transaction".to_string(), Duration::from_secs(30));
        config.operation_timeouts.insert("database_migration".to_string(), Duration::from_secs(300));
        
        // HTTP operation timeouts
        config.operation_timeouts.insert("http_request".to_string(), Duration::from_secs(30));
        config.operation_timeouts.insert("http_upload".to_string(), Duration::from_secs(120));
        config.operation_timeouts.insert("http_download".to_string(), Duration::from_secs(300));
        
        // Cache operation timeouts
        config.operation_timeouts.insert("cache_get".to_string(), Duration::from_secs(2));
        config.operation_timeouts.insert("cache_set".to_string(), Duration::from_secs(5));
        config.operation_timeouts.insert("cache_delete".to_string(), Duration::from_secs(5));
        
        // Trading operation timeouts
        config.operation_timeouts.insert("order_placement".to_string(), Duration::from_secs(10));
        config.operation_timeouts.insert("order_cancellation".to_string(), Duration::from_secs(5));
        config.operation_timeouts.insert("market_data_fetch".to_string(), Duration::from_secs(5));
        config.operation_timeouts.insert("strategy_execution".to_string(), Duration::from_secs(60));
        
        // Authentication timeouts
        config.operation_timeouts.insert("user_authentication".to_string(), Duration::from_secs(10));
        config.operation_timeouts.insert("token_validation".to_string(), Duration::from_secs(5));
        config.operation_timeouts.insert("session_creation".to_string(), Duration::from_secs(10));
        
        TimeoutManager::new(config)
    }

    /// Create timeout wrapper for database operations
    pub fn database_timeout(operation: &str) -> TimeoutWrapper {
        let timeout = match operation {
            "query" => Duration::from_secs(10),
            "transaction" => Duration::from_secs(30),
            "migration" => Duration::from_secs(300),
            _ => Duration::from_secs(15),
        };
        TimeoutWrapper::new(format!("database_{operation}"), timeout)
    }

    /// Create timeout wrapper for HTTP operations
    pub fn http_timeout(operation: &str) -> TimeoutWrapper {
        let timeout = match operation {
            "get" | "post" | "put" | "delete" => Duration::from_secs(30),
            "upload" => Duration::from_secs(120),
            "download" => Duration::from_secs(300),
            _ => Duration::from_secs(30),
        };
        TimeoutWrapper::new(format!("http_{operation}"), timeout)
    }

    /// Create timeout wrapper for trading operations
    pub fn trading_timeout(operation: &str) -> TimeoutWrapper {
        let timeout = match operation {
            "order_placement" => Duration::from_secs(10),
            "order_cancellation" => Duration::from_secs(5),
            "market_data" => Duration::from_secs(5),
            "strategy_execution" => Duration::from_secs(60),
            "risk_check" => Duration::from_secs(2),
            _ => Duration::from_secs(15),
        };
        TimeoutWrapper::new(format!("trading_{operation}"), timeout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_timeout_success() {
        let config = TimeoutConfig::default();
        let manager = TimeoutManager::new(config);

        let result = manager
            .execute("test_operation", async { 42 })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_timeout_failure() {
        let mut config = TimeoutConfig::default();
        config.operation_timeouts.insert(
            "slow_operation".to_string(),
            Duration::from_millis(10)
        );
        let manager = TimeoutManager::new(config);

        let result = manager
            .execute("slow_operation", async {
                sleep(Duration::from_millis(50)).await;
                42
            })
            .await;

        assert!(result.is_err());
        match result {
            Err(TimeoutError { operation, timeout }) => {
                assert_eq!(operation, "slow_operation");
                assert_eq!(timeout, Duration::from_millis(10));
            }
            Ok(_) => panic!("Expected timeout error, but operation succeeded"),
        }
    }

    #[tokio::test]
    async fn test_timeout_wrapper() {
        let wrapper = TimeoutWrapper::new(
            "test_operation".to_string(),
            Duration::from_millis(10)
        );

        // Should succeed
        let result = wrapper.execute(async { 42 }).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);

        // Should timeout
        let result = wrapper.execute(async {
            sleep(Duration::from_millis(50)).await;
            42
        }).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_default_timeout_manager() {
        let manager = presets::create_default_timeout_manager();
        
        // Should use specific timeout for database queries
        let result = manager
            .execute("database_query", async { "success" })
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_preset_wrappers() {
        let db_wrapper = presets::database_timeout("query");
        let result = db_wrapper.execute(async { "db_result" }).await;
        assert!(result.is_ok());

        let http_wrapper = presets::http_timeout("get");
        let result = http_wrapper.execute(async { "http_result" }).await;
        assert!(result.is_ok());

        let trading_wrapper = presets::trading_timeout("order_placement");
        let result = trading_wrapper.execute(async { "trading_result" }).await;
        assert!(result.is_ok());
    }
}