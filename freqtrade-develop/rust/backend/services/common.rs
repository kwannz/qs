//! 通用服务组件
//! 
//! 提供所有服务共享的基础组件和工具

use crate::services::{ServiceResult, ServiceError, RequestMetadata, ResponseMetadata};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tracing::{info, warn, error, debug};

/// 连接池配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolConfig {
    /// 最大连接数
    pub max_connections: u32,
    /// 最小连接数
    pub min_connections: u32,
    /// 连接超时时间（秒）
    pub connection_timeout_secs: u64,
    /// 空闲超时时间（秒）
    pub idle_timeout_secs: u64,
    /// 最大等待时间（秒）
    pub max_wait_secs: u64,
}

impl Default for ConnectionPoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 100,
            min_connections: 10,
            connection_timeout_secs: 30,
            idle_timeout_secs: 300,
            max_wait_secs: 10,
        }
    }
}

/// 通用连接池
pub struct ConnectionPool<T> {
    config: ConnectionPoolConfig,
    semaphore: Arc<Semaphore>,
    connections: Arc<RwLock<Vec<PooledConnection<T>>>>,
    factory: Arc<dyn ConnectionFactory<T>>,
}

/// 池化连接
pub struct PooledConnection<T> {
    connection: T,
    created_at: Instant,
    last_used: Instant,
    usage_count: u64,
}

/// 连接工厂trait
#[async_trait]
pub trait ConnectionFactory<T>: Send + Sync {
    async fn create_connection(&self) -> ServiceResult<T>;
    async fn validate_connection(&self, connection: &T) -> bool;
    async fn close_connection(&self, connection: T) -> ServiceResult<()>;
}

impl<T: Send + Sync> ConnectionPool<T> {
    pub fn new(config: ConnectionPoolConfig, factory: Arc<dyn ConnectionFactory<T>>) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(config.max_connections as usize)),
            config,
            connections: Arc::new(RwLock::new(Vec::new())),
            factory,
        }
    }
    
    /// 获取连接
    pub async fn acquire(&self) -> ServiceResult<PooledConnection<T>> {
        // 等待信号量
        let _permit = self.semaphore
            .acquire()
            .await
            .map_err(|_| ServiceError::ResourceExhausted {
                resource: "connection_pool".to_string(),
            })?;
        
        // 尝试复用现有连接
        {
            let mut connections = self.connections.write().await;
            if let Some(mut conn) = connections.pop() {
                // 验证连接是否仍然有效
                if self.factory.validate_connection(&conn.connection).await {
                    conn.last_used = Instant::now();
                    conn.usage_count += 1;
                    return Ok(conn);
                } else {
                    // 关闭无效连接
                    let _ = self.factory.close_connection(conn.connection).await;
                }
            }
        }
        
        // 创建新连接
        let connection = self.factory.create_connection().await?;
        let now = Instant::now();
        
        Ok(PooledConnection {
            connection,
            created_at: now,
            last_used: now,
            usage_count: 1,
        })
    }
    
    /// 归还连接
    pub async fn release(&self, mut pooled_conn: PooledConnection<T>) {
        pooled_conn.last_used = Instant::now();
        
        // 检查连接是否应该被保留
        let should_keep = pooled_conn.created_at.elapsed() 
            < Duration::from_secs(self.config.idle_timeout_secs);
        
        if should_keep && self.factory.validate_connection(&pooled_conn.connection).await {
            let mut connections = self.connections.write().await;
            connections.push(pooled_conn);
        } else {
            // 关闭连接
            let _ = self.factory.close_connection(pooled_conn.connection).await;
        }
    }
    
    /// 清理过期连接
    pub async fn cleanup_expired(&self) {
        let mut connections = self.connections.write().await;
        let cutoff = Instant::now() - Duration::from_secs(self.config.idle_timeout_secs);
        
        connections.retain(|conn| {
            if conn.last_used < cutoff {
                // 在后台关闭连接（不等待）
                let factory = self.factory.clone();
                let connection = conn.connection;
                tokio::spawn(async move {
                    let _ = factory.close_connection(connection).await;
                });
                false
            } else {
                true
            }
        });
    }
    
    /// 获取池状态
    pub async fn get_stats(&self) -> ConnectionPoolStats {
        let connections = self.connections.read().await;
        ConnectionPoolStats {
            active_connections: connections.len() as u32,
            max_connections: self.config.max_connections,
            available_permits: self.semaphore.available_permits() as u32,
        }
    }
}

/// 连接池统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolStats {
    pub active_connections: u32,
    pub max_connections: u32,
    pub available_permits: u32,
}

/// 重试策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// 最大重试次数
    pub max_retries: u32,
    /// 基础延迟时间（毫秒）
    pub base_delay_ms: u64,
    /// 最大延迟时间（毫秒）
    pub max_delay_ms: u64,
    /// 指数退避倍数
    pub backoff_multiplier: f64,
    /// 抖动因子
    pub jitter_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 100,
            max_delay_ms: 5000,
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
        }
    }
}

/// 重试执行器
pub struct RetryExecutor {
    config: RetryConfig,
}

impl RetryExecutor {
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }
    
    /// 执行带重试的操作
    pub async fn execute<F, Fut, T>(&self, operation: F) -> ServiceResult<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = ServiceResult<T>>,
    {
        let mut last_error = None;
        
        for attempt in 0..=self.config.max_retries {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    last_error = Some(error);
                    
                    // 如果不是最后一次尝试，等待后重试
                    if attempt < self.config.max_retries {
                        let delay = self.calculate_delay(attempt);
                        tokio::time::sleep(Duration::from_millis(delay)).await;
                        debug!("重试操作，第{}次尝试", attempt + 2);
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or(ServiceError::InternalError {
            message: "重试执行失败".to_string(),
        }))
    }
    
    /// 计算延迟时间
    fn calculate_delay(&self, attempt: u32) -> u64 {
        let base_delay = self.config.base_delay_ms as f64;
        let exponential_delay = base_delay * self.config.backoff_multiplier.powi(attempt as i32);
        
        // 添加抖动
        let jitter = exponential_delay * self.config.jitter_factor * (rand::random::<f64>() - 0.5);
        let delay_with_jitter = exponential_delay + jitter;
        
        // 限制在最大延迟范围内
        (delay_with_jitter as u64).min(self.config.max_delay_ms)
    }
}

/// 熔断器状态
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitBreakerState {
    /// 关闭状态（正常）
    Closed,
    /// 打开状态（熔断）
    Open,
    /// 半开状态（试探）
    HalfOpen,
}

/// 熔断器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// 失败阈值
    pub failure_threshold: u32,
    /// 成功阈值（半开状态下）
    pub success_threshold: u32,
    /// 超时时间（毫秒）
    pub timeout_ms: u64,
    /// 恢复时间（秒）
    pub recovery_timeout_secs: u64,
    /// 最小请求数（统计窗口内）
    pub minimum_requests: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout_ms: 5000,
            recovery_timeout_secs: 60,
            minimum_requests: 10,
        }
    }
}

/// 熔断器
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Arc<RwLock<CircuitBreakerState>>,
    stats: Arc<RwLock<CircuitBreakerStats>>,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
}

/// 熔断器统计信息
#[derive(Debug, Clone, Default)]
struct CircuitBreakerStats {
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    consecutive_failures: u32,
    consecutive_successes: u32,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(CircuitBreakerState::Closed)),
            stats: Arc::new(RwLock::new(CircuitBreakerStats::default())),
            last_failure_time: Arc::new(RwLock::new(None)),
        }
    }
    
    /// 执行受熔断器保护的操作
    pub async fn execute<F, Fut, T>(&self, operation: F) -> ServiceResult<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = ServiceResult<T>>,
    {
        // 检查熔断器状态
        self.check_state().await;
        
        let current_state = *self.state.read().await;
        
        match current_state {
            CircuitBreakerState::Open => {
                return Err(ServiceError::ServiceUnavailable {
                    service: "circuit_breaker_open".to_string(),
                });
            }
            CircuitBreakerState::Closed | CircuitBreakerState::HalfOpen => {
                // 执行操作
                let result = tokio::time::timeout(
                    Duration::from_millis(self.config.timeout_ms),
                    operation()
                ).await;
                
                match result {
                    Ok(Ok(value)) => {
                        self.record_success().await;
                        Ok(value)
                    }
                    Ok(Err(error)) => {
                        self.record_failure().await;
                        Err(error)
                    }
                    Err(_) => {
                        self.record_failure().await;
                        Err(ServiceError::RequestTimeout { 
                            timeout_ms: self.config.timeout_ms 
                        })
                    }
                }
            }
        }
    }
    
    /// 检查并更新熔断器状态
    async fn check_state(&self) {
        let mut state = self.state.write().await;
        let stats = self.stats.read().await;
        
        match *state {
            CircuitBreakerState::Closed => {
                if stats.consecutive_failures >= self.config.failure_threshold 
                   && stats.total_requests >= self.config.minimum_requests as u64 {
                    *state = CircuitBreakerState::Open;
                    *self.last_failure_time.write().await = Some(Instant::now());
                    warn!("熔断器打开，连续失败次数: {}", stats.consecutive_failures);
                }
            }
            CircuitBreakerState::Open => {
                if let Some(last_failure) = *self.last_failure_time.read().await {
                    if last_failure.elapsed() >= Duration::from_secs(self.config.recovery_timeout_secs) {
                        *state = CircuitBreakerState::HalfOpen;
                        info!("熔断器进入半开状态");
                    }
                }
            }
            CircuitBreakerState::HalfOpen => {
                if stats.consecutive_successes >= self.config.success_threshold {
                    *state = CircuitBreakerState::Closed;
                    info!("熔断器关闭，连续成功次数: {}", stats.consecutive_successes);
                } else if stats.consecutive_failures > 0 {
                    *state = CircuitBreakerState::Open;
                    *self.last_failure_time.write().await = Some(Instant::now());
                    warn!("熔断器重新打开");
                }
            }
        }
    }
    
    /// 记录成功
    async fn record_success(&self) {
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
        stats.successful_requests += 1;
        stats.consecutive_successes += 1;
        stats.consecutive_failures = 0;
    }
    
    /// 记录失败
    async fn record_failure(&self) {
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
        stats.failed_requests += 1;
        stats.consecutive_failures += 1;
        stats.consecutive_successes = 0;
    }
    
    /// 获取熔断器状态
    pub async fn get_state(&self) -> CircuitBreakerState {
        *self.state.read().await
    }
    
    /// 获取统计信息
    pub async fn get_stats(&self) -> (CircuitBreakerState, u64, u64, u64) {
        let state = *self.state.read().await;
        let stats = self.stats.read().await;
        (state, stats.total_requests, stats.successful_requests, stats.failed_requests)
    }
}

/// 请求限流器
pub struct RateLimiter {
    /// 令牌桶容量
    capacity: u32,
    /// 令牌生成速率（每秒）
    rate: u32,
    /// 当前令牌数
    tokens: Arc<RwLock<u32>>,
    /// 上次更新时间
    last_update: Arc<RwLock<Instant>>,
}

impl RateLimiter {
    pub fn new(capacity: u32, rate: u32) -> Self {
        Self {
            capacity,
            rate,
            tokens: Arc::new(RwLock::new(capacity)),
            last_update: Arc::new(RwLock::new(Instant::now())),
        }
    }
    
    /// 尝试获取令牌
    pub async fn try_acquire(&self, tokens_needed: u32) -> bool {
        self.refill_tokens().await;
        
        let mut tokens = self.tokens.write().await;
        if *tokens >= tokens_needed {
            *tokens -= tokens_needed;
            true
        } else {
            false
        }
    }
    
    /// 重新填充令牌
    async fn refill_tokens(&self) {
        let now = Instant::now();
        let mut last_update = self.last_update.write().await;
        let time_passed = now.duration_since(*last_update);
        
        if time_passed >= Duration::from_millis(100) { // 最小更新间隔100ms
            let new_tokens = (time_passed.as_secs_f64() * self.rate as f64) as u32;
            
            if new_tokens > 0 {
                let mut tokens = self.tokens.write().await;
                *tokens = (*tokens + new_tokens).min(self.capacity);
                *last_update = now;
            }
        }
    }
    
    /// 获取当前令牌数
    pub async fn available_tokens(&self) -> u32 {
        self.refill_tokens().await;
        *self.tokens.read().await
    }
}

/// 服务度量收集器
#[derive(Debug, Clone, Default)]
pub struct ServiceMetrics {
    /// 请求总数
    pub total_requests: u64,
    /// 成功请求数
    pub successful_requests: u64,
    /// 失败请求数
    pub failed_requests: u64,
    /// 平均响应时间（微秒）
    pub avg_response_time_us: f64,
    /// 最大响应时间（微秒）
    pub max_response_time_us: u64,
    /// 最小响应时间（微秒）
    pub min_response_time_us: u64,
    /// 活跃连接数
    pub active_connections: u32,
    /// 上次更新时间
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl ServiceMetrics {
    pub fn new() -> Self {
        Self {
            min_response_time_us: u64::MAX,
            last_updated: chrono::Utc::now(),
            ..Default::default()
        }
    }
    
    /// 记录请求
    pub fn record_request(&mut self, response_time_us: u64, success: bool) {
        self.total_requests += 1;
        
        if success {
            self.successful_requests += 1;
        } else {
            self.failed_requests += 1;
        }
        
        // 更新响应时间统计
        self.max_response_time_us = self.max_response_time_us.max(response_time_us);
        if self.min_response_time_us == u64::MAX {
            self.min_response_time_us = response_time_us;
        } else {
            self.min_response_time_us = self.min_response_time_us.min(response_time_us);
        }
        
        // 更新平均响应时间
        self.avg_response_time_us = (self.avg_response_time_us * (self.total_requests - 1) as f64 + 
                                    response_time_us as f64) / self.total_requests as f64;
        
        self.last_updated = chrono::Utc::now();
    }
    
    /// 获取成功率
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.successful_requests as f64 / self.total_requests as f64
        }
    }
    
    /// 获取错误率
    pub fn error_rate(&self) -> f64 {
        1.0 - self.success_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test]
    async fn test_rate_limiter() {
        let limiter = RateLimiter::new(10, 5);
        
        // 应该能获取到令牌
        assert!(limiter.try_acquire(5).await);
        assert!(limiter.try_acquire(5).await);
        
        // 应该没有足够的令牌
        assert!(!limiter.try_acquire(1).await);
        
        // 等待一段时间后应该能重新获取
        tokio::time::sleep(Duration::from_millis(1200)).await;
        assert!(limiter.try_acquire(5).await);
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout_ms: 100,
            recovery_timeout_secs: 1,
            minimum_requests: 1,
        };
        
        let breaker = CircuitBreaker::new(config);
        let failure_count = Arc::new(AtomicU32::new(0));
        
        // 执行会失败的操作
        let failure_count_clone = failure_count.clone();
        for _ in 0..3 {
            let _ = breaker.execute(|| {
                let count = failure_count_clone.fetch_add(1, Ordering::SeqCst);
                async move {
                    if count < 2 {
                        Err(ServiceError::InternalError { 
                            message: "test failure".to_string() 
                        })
                    } else {
                        Ok(())
                    }
                }
            }).await;
        }
        
        // 熔断器应该是打开状态
        assert_eq!(breaker.get_state().await, CircuitBreakerState::Open);
    }

    #[tokio::test]
    async fn test_retry_executor() {
        let config = RetryConfig {
            max_retries: 3,
            base_delay_ms: 10,
            max_delay_ms: 1000,
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
        };
        
        let executor = RetryExecutor::new(config);
        let attempt_count = Arc::new(AtomicU32::new(0));
        
        let attempt_count_clone = attempt_count.clone();
        let result = executor.execute(|| {
            let count = attempt_count_clone.fetch_add(1, Ordering::SeqCst);
            async move {
                if count < 2 {
                    Err(ServiceError::InternalError { 
                        message: "retry test".to_string() 
                    })
                } else {
                    Ok("success")
                }
            }
        }).await;
        
        assert!(result.is_ok());
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_service_metrics() {
        let mut metrics = ServiceMetrics::new();
        
        // 记录一些请求
        metrics.record_request(1000, true);
        metrics.record_request(2000, true);
        metrics.record_request(1500, false);
        
        assert_eq!(metrics.total_requests, 3);
        assert_eq!(metrics.successful_requests, 2);
        assert_eq!(metrics.failed_requests, 1);
        assert_eq!(metrics.success_rate(), 2.0 / 3.0);
        assert_eq!(metrics.avg_response_time_us, 1500.0);
        assert_eq!(metrics.max_response_time_us, 2000);
        assert_eq!(metrics.min_response_time_us, 1000);
    }
}