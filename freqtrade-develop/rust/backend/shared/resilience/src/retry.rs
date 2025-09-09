use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, warn};

/// Retry strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetryStrategy {
    /// Fixed interval between retries
    Fixed {
        interval: Duration,
    },
    /// Exponential backoff with jitter
    ExponentialBackoff {
        initial_interval: Duration,
        multiplier: f64,
        max_interval: Duration,
        jitter: bool,
    },
    /// Linear backoff
    LinearBackoff {
        initial_interval: Duration,
        increment: Duration,
        max_interval: Duration,
    },
}

impl Default for RetryStrategy {
    fn default() -> Self {
        Self::ExponentialBackoff {
            initial_interval: Duration::from_millis(100),
            multiplier: 2.0,
            max_interval: Duration::from_secs(30),
            jitter: true,
        }
    }
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub strategy: RetryStrategy,
    pub timeout_per_attempt: Option<Duration>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            strategy: RetryStrategy::default(),
            timeout_per_attempt: Some(Duration::from_secs(30)),
        }
    }
}

/// Retry policy trait for determining if an error should be retried
pub trait RetryPolicy<E>: Send + Sync {
    fn should_retry(&self, error: &E, attempt: u32) -> bool;
}

/// Default retry policy that retries all errors
pub struct AlwaysRetryPolicy;

impl<E> RetryPolicy<E> for AlwaysRetryPolicy {
    fn should_retry(&self, _error: &E, _attempt: u32) -> bool {
        true
    }
}

/// Retry policy that never retries
pub struct NeverRetryPolicy;

impl<E> RetryPolicy<E> for NeverRetryPolicy {
    fn should_retry(&self, _error: &E, _attempt: u32) -> bool {
        false
    }
}

/// HTTP-specific retry policy
pub struct HttpRetryPolicy {
    retryable_status_codes: Vec<u16>,
}

impl Default for HttpRetryPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpRetryPolicy {
    pub fn new() -> Self {
        Self {
            retryable_status_codes: vec![
                408, // Request Timeout
                429, // Too Many Requests
                500, // Internal Server Error
                502, // Bad Gateway
                503, // Service Unavailable
                504, // Gateway Timeout
            ],
        }
    }

    pub fn with_status_codes(status_codes: Vec<u16>) -> Self {
        Self {
            retryable_status_codes: status_codes,
        }
    }
}

impl RetryPolicy<reqwest::Error> for HttpRetryPolicy {
    fn should_retry(&self, error: &reqwest::Error, _attempt: u32) -> bool {
        if error.is_timeout() || error.is_connect() {
            return true;
        }

        if let Some(status) = error.status() {
            return self.retryable_status_codes.contains(&status.as_u16());
        }

        false
    }
}

/// Database-specific retry policy
pub struct DatabaseRetryPolicy;

impl RetryPolicy<sqlx::Error> for DatabaseRetryPolicy {
    fn should_retry(&self, error: &sqlx::Error, _attempt: u32) -> bool {
        match error {
            sqlx::Error::Database(_) => false, // Don't retry database-specific errors
            sqlx::Error::Io(_) => true, // Network errors
            sqlx::Error::PoolTimedOut => true,
            sqlx::Error::PoolClosed => true,
            _ => false,
        }
    }
}


/// Redis-specific retry policy  
pub struct RedisRetryPolicy;

impl RetryPolicy<redis::RedisError> for RedisRetryPolicy {
    fn should_retry(&self, error: &redis::RedisError, _attempt: u32) -> bool {
        match error.kind() {
            redis::ErrorKind::IoError => true,
            redis::ErrorKind::ResponseError => false, // Don't retry on response errors
            redis::ErrorKind::AuthenticationFailed => false, // Don't retry auth failures
            _ => true, // Default to retry for unknown errors including connection drops
        }
    }
}

/// Retry executor with configurable strategy and policy
pub struct RetryExecutor<P> {
    config: RetryConfig,
    policy: P,
    name: String,
}

impl<P> RetryExecutor<P> {
    pub fn new(name: String, config: RetryConfig, policy: P) -> Self {
        Self {
            config,
            policy,
            name,
        }
    }

    pub async fn execute<F, T, E>(&self, mut operation: F) -> Result<T, RetryError<E>>
    where
        F: FnMut() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T, E>> + Send>>,
        P: RetryPolicy<E>,
        E: std::fmt::Debug + Send,
    {
        let mut attempt = 1;

        loop {
            debug!(
                name = %self.name,
                attempt = attempt,
                max_attempts = self.config.max_attempts,
                "Executing operation"
            );

            let result = if let Some(timeout) = self.config.timeout_per_attempt {
                match tokio::time::timeout(timeout, operation()).await {
                    Ok(result) => result,
                    Err(_) => {
                        warn!(
                            name = %self.name,
                            attempt = attempt,
                            timeout = ?timeout,
                            "Operation timed out"
                        );
                        
                        if attempt >= self.config.max_attempts {
                            return Err(RetryError::Timeout {
                                attempts: attempt,
                                last_timeout: timeout,
                            });
                        }
                        
                        self.wait_before_retry(attempt).await;
                        attempt += 1;
                        continue;
                    }
                }
            } else {
                operation().await
            };

            match result {
                Ok(value) => {
                    if attempt > 1 {
                        debug!(
                            name = %self.name,
                            attempts = attempt,
                            "Operation succeeded after retries"
                        );
                    }
                    return Ok(value);
                }
                Err(error) => {
                    if attempt >= self.config.max_attempts {
                        warn!(
                            name = %self.name,
                            attempts = attempt,
                            error = ?error,
                            "All retry attempts exhausted"
                        );
                        return Err(RetryError::AllAttemptsFailed {
                            attempts: attempt,
                            last_error: error,
                        });
                    }

                    if !self.policy.should_retry(&error, attempt) {
                        warn!(
                            name = %self.name,
                            attempt = attempt,
                            error = ?error,
                            "Error is not retryable according to policy"
                        );
                        return Err(RetryError::NotRetryable {
                            attempts: attempt,
                            error,
                        });
                    }

                    debug!(
                        name = %self.name,
                        attempt = attempt,
                        error = ?error,
                        "Operation failed, will retry"
                    );

                    self.wait_before_retry(attempt).await;
                    attempt += 1;
                }
            }
        }
    }

    async fn wait_before_retry(&self, attempt: u32) {
        let delay = self.calculate_delay(attempt);
        debug!(
            name = %self.name,
            attempt = attempt,
            delay = ?delay,
            "Waiting before retry"
        );
        sleep(delay).await;
    }

    fn calculate_delay(&self, attempt: u32) -> Duration {
        match &self.config.strategy {
            RetryStrategy::Fixed { interval } => *interval,
            RetryStrategy::ExponentialBackoff {
                initial_interval,
                multiplier,
                max_interval,
                jitter,
            } => {
                let base_delay = initial_interval.as_millis() as f64
                    * multiplier.powi((attempt - 1) as i32);
                let delay = Duration::from_millis(base_delay as u64).min(*max_interval);

                if *jitter {
                    let jitter_factor = fastrand::f64() * 0.1; // 0-10% jitter
                    let jittered_delay = delay.as_millis() as f64 * (1.0 + jitter_factor);
                    Duration::from_millis(jittered_delay as u64)
                } else {
                    delay
                }
            }
            RetryStrategy::LinearBackoff {
                initial_interval,
                increment,
                max_interval,
            } => {
                let delay = *initial_interval + *increment * (attempt - 1);
                delay.min(*max_interval)
            }
        }
    }
}

/// Retry error types
#[derive(Debug, thiserror::Error)]
pub enum RetryError<E> {
    #[error("All {attempts} retry attempts failed, last error: {last_error:?}")]
    AllAttemptsFailed { attempts: u32, last_error: E },
    
    #[error("Error is not retryable according to policy after {attempts} attempts: {error:?}")]
    NotRetryable { attempts: u32, error: E },
    
    #[error("Operation timed out after {attempts} attempts, last timeout: {last_timeout:?}")]
    Timeout {
        attempts: u32,
        last_timeout: Duration,
    },
}

/// Convenience functions for common retry scenarios
pub mod presets {
    use super::*;

    /// Create a retry executor for HTTP requests
    pub fn http_retry(name: String) -> RetryExecutor<HttpRetryPolicy> {
        let config = RetryConfig {
            max_attempts: 3,
            strategy: RetryStrategy::ExponentialBackoff {
                initial_interval: Duration::from_millis(200),
                multiplier: 2.0,
                max_interval: Duration::from_secs(10),
                jitter: true,
            },
            timeout_per_attempt: Some(Duration::from_secs(30)),
        };
        RetryExecutor::new(name, config, HttpRetryPolicy::new())
    }

    /// Create a retry executor for database operations
    pub fn database_retry(name: String) -> RetryExecutor<DatabaseRetryPolicy> {
        let config = RetryConfig {
            max_attempts: 5,
            strategy: RetryStrategy::ExponentialBackoff {
                initial_interval: Duration::from_millis(100),
                multiplier: 1.5,
                max_interval: Duration::from_secs(5),
                jitter: true,
            },
            timeout_per_attempt: Some(Duration::from_secs(10)),
        };
        RetryExecutor::new(name, config, DatabaseRetryPolicy)
    }

    /// Create a retry executor for Redis operations
    pub fn redis_retry(name: String) -> RetryExecutor<RedisRetryPolicy> {
        let config = RetryConfig {
            max_attempts: 3,
            strategy: RetryStrategy::ExponentialBackoff {
                initial_interval: Duration::from_millis(50),
                multiplier: 2.0,
                max_interval: Duration::from_secs(2),
                jitter: true,
            },
            timeout_per_attempt: Some(Duration::from_secs(5)),
        };
        RetryExecutor::new(name, config, RedisRetryPolicy)
    }

    /// Create a retry executor that always retries
    pub fn always_retry(name: String, config: RetryConfig) -> RetryExecutor<AlwaysRetryPolicy> {
        RetryExecutor::new(name, config, AlwaysRetryPolicy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_retry_success_on_first_attempt() {
        let executor = RetryExecutor::new(
            "test".to_string(),
            RetryConfig::default(),
            AlwaysRetryPolicy,
        );

        let result = executor
            .execute(|| Box::pin(async { Ok::<i32, &str>(42) }))
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_retry_success_after_failures() {
        let attempt_count = Arc::new(AtomicU32::new(0));
        let attempt_count_clone = attempt_count.clone();

        let executor = RetryExecutor::new(
            "test".to_string(),
            RetryConfig::default(),
            AlwaysRetryPolicy,
        );

        let result = executor
            .execute(move || {
                let count = attempt_count_clone.clone();
                Box::pin(async move {
                    let current_attempt = count.fetch_add(1, Ordering::SeqCst) + 1;
                    if current_attempt < 3 {
                        Err("Temporary failure")
                    } else {
                        Ok(42)
                    }
                })
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_all_attempts_failed() {
        let executor = RetryExecutor::new(
            "test".to_string(),
            RetryConfig {
                max_attempts: 2,
                ..Default::default()
            },
            AlwaysRetryPolicy,
        );

        let result = executor
            .execute(|| Box::pin(async { Err::<i32, &str>("Persistent failure") }))
            .await;

        assert!(result.is_err());
        match result {
            Err(RetryError::AllAttemptsFailed { attempts, .. }) => {
                assert_eq!(attempts, 2);
            }
            _ => panic!("Expected AllAttemptsFailed error"),
        }
    }

    #[tokio::test]
    async fn test_retry_not_retryable() {
        let executor = RetryExecutor::new(
            "test".to_string(),
            RetryConfig::default(),
            NeverRetryPolicy,
        );

        let result = executor
            .execute(|| Box::pin(async { Err::<i32, &str>("Not retryable") }))
            .await;

        assert!(result.is_err());
        match result {
            Err(RetryError::NotRetryable { attempts, .. }) => {
                assert_eq!(attempts, 1);
            }
            _ => panic!("Expected NotRetryable error"),
        }
    }

    #[test]
    fn test_delay_calculation() {
        let config = RetryConfig {
            max_attempts: 5,
            strategy: RetryStrategy::ExponentialBackoff {
                initial_interval: Duration::from_millis(100),
                multiplier: 2.0,
                max_interval: Duration::from_secs(5),
                jitter: false,
            },
            timeout_per_attempt: None,
        };

        let executor = RetryExecutor::new("test".to_string(), config, AlwaysRetryPolicy);

        assert_eq!(executor.calculate_delay(1), Duration::from_millis(100));
        assert_eq!(executor.calculate_delay(2), Duration::from_millis(200));
        assert_eq!(executor.calculate_delay(3), Duration::from_millis(400));
        assert_eq!(executor.calculate_delay(4), Duration::from_millis(800));
        assert_eq!(executor.calculate_delay(10), Duration::from_secs(5)); // Max interval
    }

    #[test]
    fn test_fixed_delay() {
        let config = RetryConfig {
            max_attempts: 3,
            strategy: RetryStrategy::Fixed {
                interval: Duration::from_millis(500),
            },
            timeout_per_attempt: None,
        };

        let executor = RetryExecutor::new("test".to_string(), config, AlwaysRetryPolicy);

        assert_eq!(executor.calculate_delay(1), Duration::from_millis(500));
        assert_eq!(executor.calculate_delay(2), Duration::from_millis(500));
        assert_eq!(executor.calculate_delay(3), Duration::from_millis(500));
    }

    #[test]
    fn test_linear_backoff() {
        let config = RetryConfig {
            max_attempts: 5,
            strategy: RetryStrategy::LinearBackoff {
                initial_interval: Duration::from_millis(100),
                increment: Duration::from_millis(50),
                max_interval: Duration::from_millis(300),
            },
            timeout_per_attempt: None,
        };

        let executor = RetryExecutor::new("test".to_string(), config, AlwaysRetryPolicy);

        assert_eq!(executor.calculate_delay(1), Duration::from_millis(100));
        assert_eq!(executor.calculate_delay(2), Duration::from_millis(150));
        assert_eq!(executor.calculate_delay(3), Duration::from_millis(200));
        assert_eq!(executor.calculate_delay(4), Duration::from_millis(250));
        assert_eq!(executor.calculate_delay(5), Duration::from_millis(300)); // Max interval
        assert_eq!(executor.calculate_delay(10), Duration::from_millis(300)); // Max interval
    }
}