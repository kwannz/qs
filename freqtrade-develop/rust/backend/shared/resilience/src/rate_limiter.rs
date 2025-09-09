use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Rate limiting algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateLimitAlgorithm {
    /// Token bucket algorithm
    TokenBucket {
        capacity: u64,
        refill_rate: u64, // tokens per second
    },
    /// Fixed window counter
    FixedWindow {
        limit: u64,
        window_size: Duration,
    },
    /// Sliding window log
    SlidingWindow {
        limit: u64,
        window_size: Duration,
    },
    /// Leaky bucket algorithm
    LeakyBucket {
        capacity: u64,
        leak_rate: u64, // requests per second
    },
}

impl Default for RateLimitAlgorithm {
    fn default() -> Self {
        Self::TokenBucket {
            capacity: 100,
            refill_rate: 10,
        }
    }
}

/// Rate limiter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimiterConfig {
    pub algorithm: RateLimitAlgorithm,
    pub burst_allowed: bool,
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            algorithm: RateLimitAlgorithm::default(),
            burst_allowed: true,
        }
    }
}

/// Token bucket rate limiter state
#[derive(Debug)]
struct TokenBucketState {
    tokens: f64,
    last_refill: Instant,
    capacity: u64,
    refill_rate: u64,
}

impl TokenBucketState {
    fn new(capacity: u64, refill_rate: u64) -> Self {
        Self {
            tokens: capacity as f64,
            last_refill: Instant::now(),
            capacity,
            refill_rate,
        }
    }

    fn try_consume(&mut self, tokens_requested: u64) -> bool {
        self.refill_tokens();
        
        if self.tokens >= tokens_requested as f64 {
            self.tokens -= tokens_requested as f64;
            true
        } else {
            false
        }
    }

    fn refill_tokens(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        
        let tokens_to_add = elapsed * self.refill_rate as f64;
        self.tokens = (self.tokens + tokens_to_add).min(self.capacity as f64);
        self.last_refill = now;
    }
}

/// Fixed window rate limiter state
#[derive(Debug)]
struct FixedWindowState {
    count: u64,
    window_start: Instant,
    limit: u64,
    window_size: Duration,
}

impl FixedWindowState {
    fn new(limit: u64, window_size: Duration) -> Self {
        Self {
            count: 0,
            window_start: Instant::now(),
            limit,
            window_size,
        }
    }

    fn try_consume(&mut self) -> bool {
        let now = Instant::now();
        
        // Check if we need to reset the window
        if now.duration_since(self.window_start) >= self.window_size {
            self.count = 0;
            self.window_start = now;
        }

        if self.count < self.limit {
            self.count += 1;
            true
        } else {
            false
        }
    }
}

/// Sliding window rate limiter state
#[derive(Debug)]
struct SlidingWindowState {
    requests: Vec<Instant>,
    limit: u64,
    window_size: Duration,
}

impl SlidingWindowState {
    fn new(limit: u64, window_size: Duration) -> Self {
        Self {
            requests: Vec::new(),
            limit,
            window_size,
        }
    }

    fn try_consume(&mut self) -> bool {
        let now = Instant::now();
        let cutoff = now - self.window_size;
        
        // Remove old requests outside the window
        self.requests.retain(|&request_time| request_time > cutoff);

        if self.requests.len() < self.limit as usize {
            self.requests.push(now);
            true
        } else {
            false
        }
    }
}

/// Leaky bucket rate limiter state
#[derive(Debug)]
struct LeakyBucketState {
    queue_size: u64,
    last_leak: Instant,
    capacity: u64,
    leak_rate: u64,
}

impl LeakyBucketState {
    fn new(capacity: u64, leak_rate: u64) -> Self {
        Self {
            queue_size: 0,
            last_leak: Instant::now(),
            capacity,
            leak_rate,
        }
    }

    fn try_consume(&mut self) -> bool {
        self.leak();
        
        if self.queue_size < self.capacity {
            self.queue_size += 1;
            true
        } else {
            false
        }
    }

    fn leak(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_leak).as_secs_f64();
        
        let leaked = (elapsed * self.leak_rate as f64) as u64;
        self.queue_size = self.queue_size.saturating_sub(leaked);
        self.last_leak = now;
    }
}

/// Rate limiter state enum
#[derive(Debug)]
enum RateLimiterState {
    TokenBucket(TokenBucketState),
    FixedWindow(FixedWindowState),
    SlidingWindow(SlidingWindowState),
    LeakyBucket(LeakyBucketState),
}

/// Rate limiter implementation
pub struct RateLimiter {
    name: String,
    state: Arc<RwLock<RateLimiterState>>,
    config: RateLimiterConfig,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(name: String, config: RateLimiterConfig) -> Self {
        let state = match &config.algorithm {
            RateLimitAlgorithm::TokenBucket { capacity, refill_rate } => {
                RateLimiterState::TokenBucket(TokenBucketState::new(*capacity, *refill_rate))
            }
            RateLimitAlgorithm::FixedWindow { limit, window_size } => {
                RateLimiterState::FixedWindow(FixedWindowState::new(*limit, *window_size))
            }
            RateLimitAlgorithm::SlidingWindow { limit, window_size } => {
                RateLimiterState::SlidingWindow(SlidingWindowState::new(*limit, *window_size))
            }
            RateLimitAlgorithm::LeakyBucket { capacity, leak_rate } => {
                RateLimiterState::LeakyBucket(LeakyBucketState::new(*capacity, *leak_rate))
            }
        };

        Self {
            name,
            state: Arc::new(RwLock::new(state)),
            config,
        }
    }

    /// Try to acquire permission for a request
    pub async fn try_acquire(&self) -> bool {
        self.try_acquire_tokens(1).await
    }

    /// Try to acquire multiple tokens
    pub async fn try_acquire_tokens(&self, tokens: u64) -> bool {
        let mut state = self.state.write().await;
        let allowed = match &mut *state {
            RateLimiterState::TokenBucket(bucket) => bucket.try_consume(tokens),
            RateLimiterState::FixedWindow(window) => {
                if tokens == 1 {
                    window.try_consume()
                } else {
                    false // Fixed window doesn't support multi-token consumption
                }
            }
            RateLimiterState::SlidingWindow(window) => {
                if tokens == 1 {
                    window.try_consume()
                } else {
                    false // Sliding window doesn't support multi-token consumption
                }
            }
            RateLimiterState::LeakyBucket(bucket) => {
                if tokens == 1 {
                    bucket.try_consume()
                } else {
                    false // Leaky bucket doesn't support multi-token consumption
                }
            }
        };

        if allowed {
            debug!(
                rate_limiter = %self.name,
                tokens = tokens,
                "Request allowed by rate limiter"
            );
        } else {
            warn!(
                rate_limiter = %self.name,
                tokens = tokens,
                "Request rejected by rate limiter"
            );
        }

        allowed
    }

    /// Execute a function with rate limiting
    pub async fn execute<F, T>(&self, f: F) -> Result<T, RateLimitError>
    where
        F: std::future::Future<Output = T>,
    {
        if self.try_acquire().await {
            Ok(f.await)
        } else {
            Err(RateLimitError {
                limiter_name: self.name.clone(),
                algorithm: format!("{:?}", self.config.algorithm),
            })
        }
    }

    /// Get rate limiter name
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Rate limit error
#[derive(Debug, thiserror::Error)]
#[error("Rate limit exceeded for limiter '{limiter_name}' using algorithm '{algorithm}'")]
pub struct RateLimitError {
    pub limiter_name: String,
    pub algorithm: String,
}

/// Rate limiter registry for managing multiple rate limiters
pub struct RateLimiterRegistry {
    limiters: Arc<RwLock<HashMap<String, Arc<RateLimiter>>>>,
}

impl RateLimiterRegistry {
    pub fn new() -> Self {
        Self {
            limiters: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a new rate limiter
    pub async fn register(&self, name: String, config: RateLimiterConfig) -> Arc<RateLimiter> {
        let limiter = Arc::new(RateLimiter::new(name.clone(), config));
        let mut limiters = self.limiters.write().await;
        limiters.insert(name, limiter.clone());
        limiter
    }

    /// Get a rate limiter by name
    pub async fn get(&self, name: &str) -> Option<Arc<RateLimiter>> {
        let limiters = self.limiters.read().await;
        limiters.get(name).cloned()
    }

    /// Try to acquire from a specific rate limiter
    pub async fn try_acquire(&self, limiter_name: &str) -> Result<bool, String> {
        if let Some(limiter) = self.get(limiter_name).await {
            Ok(limiter.try_acquire().await)
        } else {
            Err(format!("Rate limiter '{limiter_name}' not found"))
        }
    }
}

impl Default for RateLimiterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for common rate limiting scenarios
pub mod presets {
    use super::*;

    /// API rate limiter (100 requests per minute)
    pub fn api_rate_limiter(name: String) -> RateLimiter {
        let config = RateLimiterConfig {
            algorithm: RateLimitAlgorithm::TokenBucket {
                capacity: 100,
                refill_rate: 100, // 100 tokens per 60 seconds = ~1.67 per second
            },
            burst_allowed: true,
        };
        RateLimiter::new(name, config)
    }

    /// Database connection rate limiter
    pub fn database_rate_limiter(name: String) -> RateLimiter {
        let config = RateLimiterConfig {
            algorithm: RateLimitAlgorithm::TokenBucket {
                capacity: 50,
                refill_rate: 10,
            },
            burst_allowed: false,
        };
        RateLimiter::new(name, config)
    }

    /// Trading order rate limiter (strict)
    pub fn trading_rate_limiter(name: String) -> RateLimiter {
        let config = RateLimiterConfig {
            algorithm: RateLimitAlgorithm::LeakyBucket {
                capacity: 10,
                leak_rate: 5, // 5 orders per second max
            },
            burst_allowed: false,
        };
        RateLimiter::new(name, config)
    }

    /// Authentication rate limiter (prevent brute force)
    pub fn auth_rate_limiter(name: String) -> RateLimiter {
        let config = RateLimiterConfig {
            algorithm: RateLimitAlgorithm::SlidingWindow {
                limit: 5, // 5 attempts per window
                window_size: Duration::from_secs(300), // 5 minutes
            },
            burst_allowed: false,
        };
        RateLimiter::new(name, config)
    }

    /// General purpose rate limiter
    pub fn general_rate_limiter(name: String, requests_per_second: u64) -> RateLimiter {
        let config = RateLimiterConfig {
            algorithm: RateLimitAlgorithm::TokenBucket {
                capacity: requests_per_second * 2, // Allow some burst
                refill_rate: requests_per_second,
            },
            burst_allowed: true,
        };
        RateLimiter::new(name, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_token_bucket_rate_limiter() {
        let config = RateLimiterConfig {
            algorithm: RateLimitAlgorithm::TokenBucket {
                capacity: 5,
                refill_rate: 1, // 1 token per second
            },
            burst_allowed: true,
        };
        let limiter = RateLimiter::new("test".to_string(), config);

        // Should allow initial burst
        for _ in 0..5 {
            assert!(limiter.try_acquire().await);
        }

        // Should reject when bucket is empty
        assert!(!limiter.try_acquire().await);

        // Wait for refill and try again
        sleep(Duration::from_secs(2)).await;
        assert!(limiter.try_acquire().await);
    }

    #[tokio::test]
    async fn test_fixed_window_rate_limiter() {
        let config = RateLimiterConfig {
            algorithm: RateLimitAlgorithm::FixedWindow {
                limit: 3,
                window_size: Duration::from_secs(1),
            },
            burst_allowed: true,
        };
        let limiter = RateLimiter::new("test".to_string(), config);

        // Should allow up to limit
        for _ in 0..3 {
            assert!(limiter.try_acquire().await);
        }

        // Should reject when limit reached
        assert!(!limiter.try_acquire().await);

        // Wait for window to reset
        sleep(Duration::from_secs(2)).await;
        assert!(limiter.try_acquire().await);
    }

    #[tokio::test]
    async fn test_sliding_window_rate_limiter() {
        let config = RateLimiterConfig {
            algorithm: RateLimitAlgorithm::SlidingWindow {
                limit: 2,
                window_size: Duration::from_millis(100),
            },
            burst_allowed: true,
        };
        let limiter = RateLimiter::new("test".to_string(), config);

        // Should allow up to limit
        assert!(limiter.try_acquire().await);
        assert!(limiter.try_acquire().await);

        // Should reject when limit reached
        assert!(!limiter.try_acquire().await);

        // Wait for window to slide
        sleep(Duration::from_millis(150)).await;
        assert!(limiter.try_acquire().await);
    }

    #[tokio::test]
    async fn test_rate_limiter_registry() {
        let registry = RateLimiterRegistry::new();
        
        let config = RateLimiterConfig::default();
        let _limiter = registry.register("test-limiter".to_string(), config).await;

        let retrieved = registry.get("test-limiter").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name(), "test-limiter");

        let acquire_result = registry.try_acquire("test-limiter").await;
        assert!(acquire_result.is_ok());
        assert!(acquire_result.unwrap());
    }

    #[tokio::test]
    async fn test_rate_limiter_execute() {
        let config = RateLimiterConfig {
            algorithm: RateLimitAlgorithm::TokenBucket {
                capacity: 1,
                refill_rate: 1,
            },
            burst_allowed: true,
        };
        let limiter = RateLimiter::new("test".to_string(), config);

        // Should succeed
        let result = limiter.execute(async { 42 }).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);

        // Should fail when rate limited
        let result = limiter.execute(async { 43 }).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_preset_rate_limiters() {
        let api_limiter = presets::api_rate_limiter("api".to_string());
        assert!(api_limiter.try_acquire().await);

        let db_limiter = presets::database_rate_limiter("db".to_string());
        assert!(db_limiter.try_acquire().await);

        let trading_limiter = presets::trading_rate_limiter("trading".to_string());
        assert!(trading_limiter.try_acquire().await);

        let auth_limiter = presets::auth_rate_limiter("auth".to_string());
        assert!(auth_limiter.try_acquire().await);
    }
}