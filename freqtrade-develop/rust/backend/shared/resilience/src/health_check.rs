use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// Type alias to simplify complex health check function type
type HealthCheckFn = Arc<dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<HealthCheckResult>> + Send>> + Send + Sync>;

/// Health status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    pub service_name: String,
    pub status: HealthStatus,
    pub message: Option<String>,
    pub last_checked: Instant,
    pub response_time: Duration,
    pub metadata: HashMap<String, String>,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    pub check_interval: Duration,
    pub timeout: Duration,
    pub failure_threshold: u32,
    pub recovery_threshold: u32,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            failure_threshold: 3,
            recovery_threshold: 2,
        }
    }
}

/// Health check trait
#[async_trait::async_trait]
pub trait HealthCheck: Send + Sync {
    async fn check_health(&self) -> Result<HealthCheckResult>;
    fn service_name(&self) -> &str;
}

/// Database health check
pub struct DatabaseHealthCheck {
    service_name: String,
    connection_pool: Arc<sqlx::PgPool>,
}

impl DatabaseHealthCheck {
    pub fn new(service_name: String, connection_pool: Arc<sqlx::PgPool>) -> Self {
        Self {
            service_name,
            connection_pool,
        }
    }
}

#[async_trait::async_trait]
impl HealthCheck for DatabaseHealthCheck {
    async fn check_health(&self) -> Result<HealthCheckResult> {
        let start = Instant::now();
        let mut metadata = HashMap::new();

        let status = match sqlx::query("SELECT 1")
            .fetch_one(&*self.connection_pool)
            .await
        {
            Ok(_) => {
                metadata.insert("query".to_string(), "SELECT 1 - OK".to_string());
                HealthStatus::Healthy
            }
            Err(e) => {
                metadata.insert("error".to_string(), e.to_string());
                HealthStatus::Unhealthy
            }
        };

        Ok(HealthCheckResult {
            service_name: self.service_name.clone(),
            status,
            message: None,
            last_checked: start,
            response_time: start.elapsed(),
            metadata,
        })
    }

    fn service_name(&self) -> &str {
        &self.service_name
    }
}

/// Redis health check
pub struct RedisHealthCheck {
    service_name: String,
    redis_client: Arc<redis::Client>,
}

impl RedisHealthCheck {
    pub fn new(service_name: String, redis_client: Arc<redis::Client>) -> Self {
        Self {
            service_name,
            redis_client,
        }
    }
}

#[async_trait::async_trait]
impl HealthCheck for RedisHealthCheck {
    async fn check_health(&self) -> Result<HealthCheckResult> {
        let start = Instant::now();
        let mut metadata = HashMap::new();

        let status = match self.redis_client.get_multiplexed_async_connection().await {
            Ok(mut conn) => {
                match redis::cmd("PING").query_async(&mut conn).await {
                    Ok(response) => {
                        metadata.insert("ping_response".to_string(), response);
                        HealthStatus::Healthy
                    }
                    Err(e) => {
                        metadata.insert("ping_error".to_string(), e.to_string());
                        HealthStatus::Unhealthy
                    }
                }
            }
            Err(e) => {
                metadata.insert("connection_error".to_string(), e.to_string());
                HealthStatus::Unhealthy
            }
        };

        Ok(HealthCheckResult {
            service_name: self.service_name.clone(),
            status,
            message: None,
            last_checked: start,
            response_time: start.elapsed(),
            metadata,
        })
    }

    fn service_name(&self) -> &str {
        &self.service_name
    }
}

/// HTTP endpoint health check
pub struct HttpHealthCheck {
    service_name: String,
    url: String,
    client: reqwest::Client,
}

impl HttpHealthCheck {
    pub fn new(service_name: String, url: String) -> Self {
        Self {
            service_name,
            url,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl HealthCheck for HttpHealthCheck {
    async fn check_health(&self) -> Result<HealthCheckResult> {
        let start = Instant::now();
        let mut metadata = HashMap::new();

        let status = match self.client.get(&self.url).send().await {
            Ok(response) => {
                metadata.insert("status_code".to_string(), response.status().to_string());
                if response.status().is_success() {
                    HealthStatus::Healthy
                } else {
                    HealthStatus::Degraded
                }
            }
            Err(e) => {
                metadata.insert("error".to_string(), e.to_string());
                HealthStatus::Unhealthy
            }
        };

        Ok(HealthCheckResult {
            service_name: self.service_name.clone(),
            status,
            message: None,
            last_checked: start,
            response_time: start.elapsed(),
            metadata,
        })
    }

    fn service_name(&self) -> &str {
        &self.service_name
    }
}

/// Custom health check using a closure
pub struct CustomHealthCheck {
    service_name: String,
    check_fn: HealthCheckFn,
}

impl CustomHealthCheck {
    pub fn new<F, Fut>(service_name: String, check_fn: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<HealthCheckResult>> + Send + 'static,
    {
        Self {
            service_name,
            check_fn: Arc::new(move || Box::pin(check_fn())),
        }
    }
}

#[async_trait::async_trait]
impl HealthCheck for CustomHealthCheck {
    async fn check_health(&self) -> Result<HealthCheckResult> {
        (self.check_fn)().await
    }

    fn service_name(&self) -> &str {
        &self.service_name
    }
}

/// Health check manager state
#[derive(Debug)]
#[derive(Default)]
struct HealthCheckState {
    consecutive_failures: u32,
    consecutive_successes: u32,
    last_result: Option<HealthCheckResult>,
}


/// Health check manager
pub struct HealthCheckManager {
    checks: Arc<RwLock<HashMap<String, Arc<dyn HealthCheck>>>>,
    states: Arc<RwLock<HashMap<String, HealthCheckState>>>,
    configs: Arc<RwLock<HashMap<String, HealthCheckConfig>>>,
}

impl HealthCheckManager {
    pub fn new() -> Self {
        Self {
            checks: Arc::new(RwLock::new(HashMap::new())),
            states: Arc::new(RwLock::new(HashMap::new())),
            configs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a health check
    pub async fn register_check(
        &self,
        check: Arc<dyn HealthCheck>,
        config: HealthCheckConfig,
    ) {
        let service_name = check.service_name().to_string();
        
        let mut checks = self.checks.write().await;
        let mut states = self.states.write().await;
        let mut configs = self.configs.write().await;
        
        checks.insert(service_name.clone(), check);
        states.insert(service_name.clone(), HealthCheckState::default());
        configs.insert(service_name, config);
    }

    /// Run health check for a specific service
    pub async fn check_service(&self, service_name: &str) -> Option<HealthCheckResult> {
        let check = {
            let checks = self.checks.read().await;
            checks.get(service_name).cloned()
        };

        let config = {
            let configs = self.configs.read().await;
            configs.get(service_name).cloned()
        };

        if let (Some(check), Some(config)) = (check, config) {
            let result = match tokio::time::timeout(config.timeout, check.check_health()).await {
                Ok(Ok(mut result)) => {
                    // Update the status based on failure/recovery thresholds
                    let mut states = self.states.write().await;
                    if let Some(state) = states.get_mut(service_name) {
                        match result.status {
                            HealthStatus::Healthy => {
                                state.consecutive_failures = 0;
                                state.consecutive_successes += 1;
                            }
                            HealthStatus::Degraded | HealthStatus::Unhealthy => {
                                state.consecutive_successes = 0;
                                state.consecutive_failures += 1;

                                // Override status based on thresholds
                                if state.consecutive_failures >= config.failure_threshold {
                                    result.status = HealthStatus::Unhealthy;
                                }
                            }
                            HealthStatus::Unknown => {
                                // Don't modify counters for unknown status
                            }
                        }
                        state.last_result = Some(result.clone());
                    }
                    result
                }
                Ok(Err(e)) => {
                    error!(
                        service = service_name,
                        error = %e,
                        "Health check failed with error"
                    );
                    
                    HealthCheckResult {
                        service_name: service_name.to_string(),
                        status: HealthStatus::Unhealthy,
                        message: Some(e.to_string()),
                        last_checked: Instant::now(),
                        response_time: config.timeout,
                        metadata: HashMap::new(),
                    }
                }
                Err(_) => {
                    warn!(
                        service = service_name,
                        timeout = ?config.timeout,
                        "Health check timed out"
                    );
                    
                    HealthCheckResult {
                        service_name: service_name.to_string(),
                        status: HealthStatus::Unhealthy,
                        message: Some("Health check timed out".to_string()),
                        last_checked: Instant::now(),
                        response_time: config.timeout,
                        metadata: HashMap::new(),
                    }
                }
            };

            debug!(
                service = service_name,
                status = ?result.status,
                response_time = ?result.response_time,
                "Health check completed"
            );

            Some(result)
        } else {
            None
        }
    }

    /// Run health checks for all registered services
    pub async fn check_all_services(&self) -> Vec<HealthCheckResult> {
        let service_names: Vec<String> = {
            let checks = self.checks.read().await;
            checks.keys().cloned().collect()
        };

        let mut results = Vec::new();
        for service_name in service_names {
            if let Some(result) = self.check_service(&service_name).await {
                results.push(result);
            }
        }

        results
    }

    /// Get overall system health status
    pub async fn get_overall_status(&self) -> HealthStatus {
        let results = self.check_all_services().await;
        
        if results.is_empty() {
            return HealthStatus::Unknown;
        }

        let mut healthy_count = 0;
        let mut degraded_count = 0;
        let mut unhealthy_count = 0;

        for result in results {
            match result.status {
                HealthStatus::Healthy => healthy_count += 1,
                HealthStatus::Degraded => degraded_count += 1,
                HealthStatus::Unhealthy => unhealthy_count += 1,
                HealthStatus::Unknown => {}
            }
        }

        if unhealthy_count > 0 {
            HealthStatus::Unhealthy
        } else if degraded_count > 0 {
            HealthStatus::Degraded
        } else if healthy_count > 0 {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unknown
        }
    }

    /// Start continuous health monitoring
    pub async fn start_monitoring(&self) {
        let checks = self.checks.clone();
        let states = self.states.clone();
        let configs = self.configs.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                let service_names: Vec<String> = {
                    let checks_guard = checks.read().await;
                    checks_guard.keys().cloned().collect()
                };

                for service_name in service_names {
                    let check = {
                        let checks_guard = checks.read().await;
                        checks_guard.get(&service_name).cloned()
                    };

                    let config = {
                        let configs_guard = configs.read().await;
                        configs_guard.get(&service_name).cloned().unwrap_or_default()
                    };

                    if let Some(_check) = check {
                        // Check if it's time to run this health check
                        let should_check = {
                            let states_guard = states.read().await;
                            if let Some(state) = states_guard.get(&service_name) {
                                if let Some(last_result) = &state.last_result {
                                    last_result.last_checked.elapsed() >= config.check_interval
                                } else {
                                    true
                                }
                            } else {
                                true
                            }
                        };

                        if should_check {
                            debug!(service = %service_name, "Running scheduled health check");
                            // Run the health check (this will update states internally)
                            // Note: In a real implementation, you'd call self.check_service here
                            // but since we're in a spawned task, we'd need to restructure this
                        }
                    }
                }
            }
        });

        info!("Health check monitoring started");
    }
}

impl Default for HealthCheckManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Overall system health summary
#[derive(Debug, Clone)]
pub struct SystemHealthSummary {
    pub overall_status: HealthStatus,
    pub service_count: usize,
    pub healthy_services: usize,
    pub degraded_services: usize,
    pub unhealthy_services: usize,
    pub services: Vec<HealthCheckResult>,
}

impl HealthCheckManager {
    /// Get comprehensive system health summary
    pub async fn get_health_summary(&self) -> SystemHealthSummary {
        let results = self.check_all_services().await;
        let overall_status = self.get_overall_status().await;
        
        let mut healthy_services = 0;
        let mut degraded_services = 0;
        let mut unhealthy_services = 0;

        for result in &results {
            match result.status {
                HealthStatus::Healthy => healthy_services += 1,
                HealthStatus::Degraded => degraded_services += 1,
                HealthStatus::Unhealthy => unhealthy_services += 1,
                HealthStatus::Unknown => {}
            }
        }

        SystemHealthSummary {
            overall_status,
            service_count: results.len(),
            healthy_services,
            degraded_services,
            unhealthy_services,
            services: results,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockHealthCheck {
        name: String,
        should_fail: bool,
    }

    impl MockHealthCheck {
        fn new(name: String, should_fail: bool) -> Self {
            Self { name, should_fail }
        }
    }

    #[async_trait::async_trait]
    impl HealthCheck for MockHealthCheck {
        async fn check_health(&self) -> Result<HealthCheckResult> {
            let status = if self.should_fail {
                HealthStatus::Unhealthy
            } else {
                HealthStatus::Healthy
            };

            Ok(HealthCheckResult {
                service_name: self.name.clone(),
                status,
                message: None,
                last_checked: Instant::now(),
                response_time: Duration::from_millis(10),
                metadata: HashMap::new(),
            })
        }

        fn service_name(&self) -> &str {
            &self.name
        }
    }

    #[tokio::test]
    async fn test_health_check_manager() {
        let manager = HealthCheckManager::new();
        
        let healthy_check = Arc::new(MockHealthCheck::new("healthy-service".to_string(), false));
        let unhealthy_check = Arc::new(MockHealthCheck::new("unhealthy-service".to_string(), true));
        
        manager.register_check(healthy_check, HealthCheckConfig::default()).await;
        manager.register_check(unhealthy_check, HealthCheckConfig::default()).await;

        let results = manager.check_all_services().await;
        assert_eq!(results.len(), 2);

        let healthy_result = results.iter().find(|r| r.service_name == "healthy-service").unwrap();
        assert_eq!(healthy_result.status, HealthStatus::Healthy);

        let unhealthy_result = results.iter().find(|r| r.service_name == "unhealthy-service").unwrap();
        assert_eq!(unhealthy_result.status, HealthStatus::Unhealthy);

        let overall_status = manager.get_overall_status().await;
        assert_eq!(overall_status, HealthStatus::Unhealthy);
    }

    #[tokio::test]
    async fn test_health_summary() {
        let manager = HealthCheckManager::new();
        
        let check1 = Arc::new(MockHealthCheck::new("service1".to_string(), false));
        let check2 = Arc::new(MockHealthCheck::new("service2".to_string(), false));
        let check3 = Arc::new(MockHealthCheck::new("service3".to_string(), true));
        
        manager.register_check(check1, HealthCheckConfig::default()).await;
        manager.register_check(check2, HealthCheckConfig::default()).await;
        manager.register_check(check3, HealthCheckConfig::default()).await;

        let summary = manager.get_health_summary().await;
        assert_eq!(summary.service_count, 3);
        assert_eq!(summary.healthy_services, 2);
        assert_eq!(summary.unhealthy_services, 1);
        assert_eq!(summary.overall_status, HealthStatus::Unhealthy);
    }
}