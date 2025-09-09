use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::time::interval;
use uuid::Uuid;

/// Health status enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Degraded => write!(f, "degraded"),
            HealthStatus::Unhealthy => write!(f, "unhealthy"),
            HealthStatus::Unknown => write!(f, "unknown"),
        }
    }
}

/// Detailed health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub component: String,
    pub status: HealthStatus,
    pub message: Option<String>,
    pub last_checked: u64, // Unix timestamp
    pub response_time_ms: u64,
    pub metadata: HashMap<String, String>,
    pub check_id: String,
}

impl HealthCheckResult {
    pub fn healthy(component: &str) -> Self {
        Self {
            component: component.to_string(),
            status: HealthStatus::Healthy,
            message: None,
            last_checked: SystemTime::now().duration_since(UNIX_EPOCH)
                .unwrap_or_default().as_secs(),
            response_time_ms: 0,
            metadata: HashMap::new(),
            check_id: Uuid::new_v4().to_string(),
        }
    }
    
    pub fn unhealthy(component: &str, message: &str) -> Self {
        Self {
            component: component.to_string(),
            status: HealthStatus::Unhealthy,
            message: Some(message.to_string()),
            last_checked: SystemTime::now().duration_since(UNIX_EPOCH)
                .unwrap_or_default().as_secs(),
            response_time_ms: 0,
            metadata: HashMap::new(),
            check_id: Uuid::new_v4().to_string(),
        }
    }
    
    pub fn with_response_time(mut self, duration: Duration) -> Self {
        self.response_time_ms = duration.as_millis() as u64;
        self
    }
    
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Health check trait for components
#[async_trait]
pub trait HealthChecker: Send + Sync {
    async fn check_health(&self) -> HealthCheckResult;
    fn component_name(&self) -> &str;
    fn check_interval(&self) -> Duration {
        Duration::from_secs(30)
    }
}

/// Database health checker
pub struct DatabaseHealthChecker {
    component_name: String,
    connection_string: String,
    timeout: Duration,
}

impl DatabaseHealthChecker {
    pub fn new(component_name: String, connection_string: String) -> Self {
        Self {
            component_name,
            connection_string,
            timeout: Duration::from_secs(5),
        }
    }
    
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[async_trait]
impl HealthChecker for DatabaseHealthChecker {
    async fn check_health(&self) -> HealthCheckResult {
        let start = Instant::now();
        
        // Simple connection test - in production, use actual database connections
        let result = tokio::time::timeout(
            self.timeout,
            async {
                // Simulate database health check
                tokio::time::sleep(Duration::from_millis(10)).await;
                Ok::<(), String>(())
            }
        ).await;
        
        let duration = start.elapsed();
        
        match result {
            Ok(Ok(())) => {
                HealthCheckResult::healthy(&self.component_name)
                    .with_response_time(duration)
                    .with_metadata("connection_string", &self.connection_string)
                    .with_metadata("timeout_ms", &self.timeout.as_millis().to_string())
            }
            Ok(Err(e)) => {
                HealthCheckResult::unhealthy(&self.component_name, &e)
                    .with_response_time(duration)
            }
            Err(_) => {
                HealthCheckResult::unhealthy(&self.component_name, "Timeout")
                    .with_response_time(duration)
            }
        }
    }
    
    fn component_name(&self) -> &str {
        &self.component_name
    }
}

/// Redis health checker
pub struct RedisHealthChecker {
    component_name: String,
    connection_string: String,
    timeout: Duration,
}

impl RedisHealthChecker {
    pub fn new(component_name: String, connection_string: String) -> Self {
        Self {
            component_name,
            connection_string,
            timeout: Duration::from_secs(3),
        }
    }
}

#[async_trait]
impl HealthChecker for RedisHealthChecker {
    async fn check_health(&self) -> HealthCheckResult {
        let start = Instant::now();
        
        let result = tokio::time::timeout(
            self.timeout,
            async {
                // Simulate Redis health check (PING command)
                tokio::time::sleep(Duration::from_millis(5)).await;
                Ok::<(), String>(())
            }
        ).await;
        
        let duration = start.elapsed();
        
        match result {
            Ok(Ok(())) => {
                HealthCheckResult::healthy(&self.component_name)
                    .with_response_time(duration)
                    .with_metadata("connection_string", &self.connection_string)
            }
            Ok(Err(e)) => {
                HealthCheckResult::unhealthy(&self.component_name, &e)
                    .with_response_time(duration)
            }
            Err(_) => {
                HealthCheckResult::unhealthy(&self.component_name, "Timeout")
                    .with_response_time(duration)
            }
        }
    }
    
    fn component_name(&self) -> &str {
        &self.component_name
    }
}

/// HTTP endpoint health checker
pub struct HttpHealthChecker {
    component_name: String,
    endpoint: String,
    client: Client,
    expected_status: u16,
    timeout: Duration,
}

impl HttpHealthChecker {
    pub fn new(component_name: String, endpoint: String) -> Self {
        Self {
            component_name,
            endpoint,
            client: Client::new(),
            expected_status: 200,
            timeout: Duration::from_secs(10),
        }
    }
    
    pub fn with_expected_status(mut self, status: u16) -> Self {
        self.expected_status = status;
        self
    }
    
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[async_trait]
impl HealthChecker for HttpHealthChecker {
    async fn check_health(&self) -> HealthCheckResult {
        let start = Instant::now();
        
        let result = tokio::time::timeout(
            self.timeout,
            self.client.get(&self.endpoint).send()
        ).await;
        
        let duration = start.elapsed();
        
        match result {
            Ok(Ok(response)) => {
                let status = response.status().as_u16();
                if status == self.expected_status {
                    HealthCheckResult::healthy(&self.component_name)
                        .with_response_time(duration)
                        .with_metadata("endpoint", &self.endpoint)
                        .with_metadata("status_code", &status.to_string())
                } else {
                    HealthCheckResult::unhealthy(
                        &self.component_name,
                        &format!("Unexpected status code: {status}")
                    )
                    .with_response_time(duration)
                    .with_metadata("endpoint", &self.endpoint)
                    .with_metadata("status_code", &status.to_string())
                }
            }
            Ok(Err(e)) => {
                HealthCheckResult::unhealthy(&self.component_name, &e.to_string())
                    .with_response_time(duration)
            }
            Err(_) => {
                HealthCheckResult::unhealthy(&self.component_name, "Request timeout")
                    .with_response_time(duration)
            }
        }
    }
    
    fn component_name(&self) -> &str {
        &self.component_name
    }
}

/// Custom business logic health checker
pub struct BusinessHealthChecker {
    component_name: String,
    check_fn: Box<dyn Fn() -> Result<(), String> + Send + Sync>,
}

impl BusinessHealthChecker {
    pub fn new<F>(component_name: String, check_fn: F) -> Self
    where
        F: Fn() -> Result<(), String> + Send + Sync + 'static,
    {
        Self {
            component_name,
            check_fn: Box::new(check_fn),
        }
    }
}

#[async_trait]
impl HealthChecker for BusinessHealthChecker {
    async fn check_health(&self) -> HealthCheckResult {
        let start = Instant::now();
        
        match (self.check_fn)() {
            Ok(()) => {
                HealthCheckResult::healthy(&self.component_name)
                    .with_response_time(start.elapsed())
            }
            Err(e) => {
                HealthCheckResult::unhealthy(&self.component_name, &e)
                    .with_response_time(start.elapsed())
            }
        }
    }
    
    fn component_name(&self) -> &str {
        &self.component_name
    }
}

/// System health summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthSummary {
    pub overall_status: HealthStatus,
    pub timestamp: u64,
    pub total_components: usize,
    pub healthy_components: usize,
    pub degraded_components: usize,
    pub unhealthy_components: usize,
    pub unknown_components: usize,
    pub components: Vec<HealthCheckResult>,
    pub uptime_seconds: u64,
    pub version: String,
}

/// Health monitoring manager
pub struct HealthMonitor {
    checkers: Arc<RwLock<Vec<Arc<dyn HealthChecker>>>>,
    results: Arc<RwLock<HashMap<String, HealthCheckResult>>>,
    start_time: Instant,
    version: String,
}

impl HealthMonitor {
    pub fn new(version: String) -> Self {
        Self {
            checkers: Arc::new(RwLock::new(Vec::new())),
            results: Arc::new(RwLock::new(HashMap::new())),
            start_time: Instant::now(),
            version,
        }
    }
    
    /// Register a health checker
    pub async fn register_checker<T>(&self, checker: T)
    where
        T: HealthChecker + 'static,
    {
        let mut checkers = self.checkers.write().await;
        checkers.push(Arc::new(checker));
    }
    
    /// Start periodic health checks
    pub async fn start_monitoring(&self) {
        let checkers = self.checkers.clone();
        let results = self.results.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                let checkers_guard = checkers.read().await;
                let mut check_futures = Vec::new();
                
                for checker in checkers_guard.iter() {
                    let checker_clone = checker.clone();
                    check_futures.push(async move {
                        let result = checker_clone.check_health().await;
                        (checker_clone.component_name().to_string(), result)
                    });
                }
                
                drop(checkers_guard);
                
                // Run all health checks concurrently
                let check_results = futures::future::join_all(check_futures).await;
                
                // Update results
                let mut results_guard = results.write().await;
                for (component_name, result) in check_results {
                    results_guard.insert(component_name, result);
                }
            }
        });
    }
    
    /// Run all health checks once
    pub async fn check_all_health(&self) -> SystemHealthSummary {
        let checkers = self.checkers.read().await;
        let mut check_futures = Vec::new();
        
        for checker in checkers.iter() {
            let checker_clone = checker.clone();
            check_futures.push(async move {
                checker_clone.check_health().await
            });
        }
        
        drop(checkers);
        
        // Run all health checks concurrently
        let results = futures::future::join_all(check_futures).await;
        
        self.create_health_summary(results).await
    }
    
    /// Get current health summary from cached results
    pub async fn get_health_summary(&self) -> SystemHealthSummary {
        let results_guard = self.results.read().await;
        let results: Vec<HealthCheckResult> = results_guard.values().cloned().collect();
        drop(results_guard);
        
        self.create_health_summary(results).await
    }
    
    async fn create_health_summary(&self, results: Vec<HealthCheckResult>) -> SystemHealthSummary {
        let total_components = results.len();
        let mut healthy_components = 0;
        let mut degraded_components = 0;
        let mut unhealthy_components = 0;
        let mut unknown_components = 0;
        
        for result in &results {
            match result.status {
                HealthStatus::Healthy => healthy_components += 1,
                HealthStatus::Degraded => degraded_components += 1,
                HealthStatus::Unhealthy => unhealthy_components += 1,
                HealthStatus::Unknown => unknown_components += 1,
            }
        }
        
        // Determine overall status
        let overall_status = if unhealthy_components > 0 {
            HealthStatus::Unhealthy
        } else if degraded_components > 0 {
            HealthStatus::Degraded
        } else if unknown_components > 0 {
            HealthStatus::Unknown
        } else {
            HealthStatus::Healthy
        };
        
        SystemHealthSummary {
            overall_status,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)
                .unwrap_or_default().as_secs(),
            total_components,
            healthy_components,
            degraded_components,
            unhealthy_components,
            unknown_components,
            components: results,
            uptime_seconds: self.start_time.elapsed().as_secs(),
            version: self.version.clone(),
        }
    }
    
    /// Check if system is ready (all critical components healthy)
    pub async fn is_ready(&self, critical_components: &[&str]) -> bool {
        let results = self.results.read().await;
        
        for component in critical_components {
            if let Some(result) = results.get(*component) {
                if result.status != HealthStatus::Healthy {
                    return false;
                }
            } else {
                return false; // Component not found
            }
        }
        
        true
    }
    
    /// Check if system is alive (basic functionality)
    pub async fn is_alive(&self) -> bool {
        // Simple liveness check - system is alive if monitor is running
        true
    }
}

/// Global health monitor instance - thread-safe singleton
use std::sync::OnceLock;
static HEALTH_MONITOR: OnceLock<HealthMonitor> = OnceLock::new();

/// Initialize global health monitor
pub fn init_health_monitor(version: String) -> &'static HealthMonitor {
    let monitor = HealthMonitor::new(version);
    HEALTH_MONITOR.get_or_init(|| monitor)
}

/// Get global health monitor instance (thread-safe)
pub fn get_health_monitor() -> Option<&'static HealthMonitor> {
    HEALTH_MONITOR.get()
}

/// Convenience functions for common health checkers
pub mod presets {
    use super::*;
    
    /// Create PostgreSQL health checker
    pub fn postgres_health_checker(connection_string: &str) -> DatabaseHealthChecker {
        DatabaseHealthChecker::new(
            "postgresql".to_string(),
            connection_string.to_string(),
        )
        .with_timeout(Duration::from_secs(5))
    }
    
    /// Create Redis health checker
    pub fn redis_health_checker(connection_string: &str) -> RedisHealthChecker {
        RedisHealthChecker::new(
            "redis".to_string(),
            connection_string.to_string(),
        )
    }
    
    /// Create HTTP service health checker
    pub fn http_service_checker(service_name: &str, endpoint: &str) -> HttpHealthChecker {
        HttpHealthChecker::new(
            service_name.to_string(),
            endpoint.to_string(),
        )
        .with_timeout(Duration::from_secs(10))
    }
    
    /// Create trading system health checker
    pub fn trading_system_checker() -> BusinessHealthChecker {
        BusinessHealthChecker::new(
            "trading-system".to_string(),
            || {
                // Add business logic checks here
                // For example: check if strategies are running, positions are within limits, etc.
                Ok(())
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_health_monitor() {
        let monitor = HealthMonitor::new("1.0.0".to_string());
        
        // Register a simple health checker
        let checker = BusinessHealthChecker::new(
            "test-component".to_string(),
            || Ok(())
        );
        monitor.register_checker(checker).await;
        
        // Run health check
        let summary = monitor.check_all_health().await;
        assert_eq!(summary.total_components, 1);
        assert_eq!(summary.healthy_components, 1);
        assert_eq!(summary.overall_status, HealthStatus::Healthy);
    }
    
    #[tokio::test]
    async fn test_health_checkers() {
        // Test database health checker
        let db_checker = DatabaseHealthChecker::new(
            "test-db".to_string(),
            "postgresql://localhost/test".to_string(),
        );
        let result = db_checker.check_health().await;
        assert_eq!(result.component, "test-db");
        
        // Test Redis health checker
        let redis_checker = RedisHealthChecker::new(
            "test-redis".to_string(),
            "redis://localhost:6379".to_string(),
        );
        let result = redis_checker.check_health().await;
        assert_eq!(result.component, "test-redis");
    }
    
    #[test]
    fn test_health_status() {
        assert_eq!(HealthStatus::Healthy.to_string(), "healthy");
        assert_eq!(HealthStatus::Unhealthy.to_string(), "unhealthy");
        assert_eq!(HealthStatus::Degraded.to_string(), "degraded");
        assert_eq!(HealthStatus::Unknown.to_string(), "unknown");
    }
}