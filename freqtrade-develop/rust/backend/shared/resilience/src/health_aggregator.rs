use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn, error};

#[derive(Clone, Debug)]
pub struct ServiceConfig {
    pub name: String,
    pub url: String,
    pub health_endpoint: String,
    pub check_interval: Duration,
    pub timeout: Duration,
    pub max_retries: u32,
}

#[derive(Clone, Debug)]
pub struct ServiceHealth {
    pub service: String,
    pub status: String,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub response_time_ms: Option<u64>,
    pub error_count: u32,
    pub consecutive_failures: u32,
    pub last_error: Option<String>,
}

pub struct HealthAggregator {
    services: HashMap<String, ServiceConfig>,
    health_status: Arc<RwLock<HashMap<String, ServiceHealth>>>,
    error_counts: Arc<RwLock<HashMap<String, u32>>>,
    client: reqwest::Client,
}

impl Default for HealthAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthAggregator {
    pub fn new() -> Self {
        Self {
            services: HashMap::new(),
            health_status: Arc::new(RwLock::new(HashMap::new())),
            error_counts: Arc::new(RwLock::new(HashMap::new())),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap(),
        }
    }
    
    pub fn add_service(&mut self, config: ServiceConfig) {
        self.services.insert(config.name.clone(), config);
    }
    
    pub fn with_default_services() -> Self {
        let mut aggregator = Self::new();
        
        // 添加默认服务配置
        aggregator.add_service(ServiceConfig {
            name: "gateway".to_string(),
            url: "http://localhost:8080".to_string(),
            health_endpoint: "/health".to_string(),
            check_interval: Duration::from_secs(30),
            timeout: Duration::from_secs(2),
            max_retries: 3,
        });
        
        aggregator.add_service(ServiceConfig {
            name: "trading".to_string(),
            url: "http://localhost:50051".to_string(),
            health_endpoint: "/health".to_string(),
            check_interval: Duration::from_secs(30),
            timeout: Duration::from_secs(2),
            max_retries: 3,
        });
        
        aggregator.add_service(ServiceConfig {
            name: "market".to_string(),
            url: "http://localhost:50052".to_string(),
            health_endpoint: "/health".to_string(),
            check_interval: Duration::from_secs(30),
            timeout: Duration::from_secs(2),
            max_retries: 3,
        });
        
        aggregator.add_service(ServiceConfig {
            name: "analytics".to_string(),
            url: "http://localhost:8090".to_string(),
            health_endpoint: "/health".to_string(),
            check_interval: Duration::from_secs(30),
            timeout: Duration::from_secs(2),
            max_retries: 3,
        });
        
        aggregator.add_service(ServiceConfig {
            name: "admin".to_string(),
            url: "http://localhost:8084".to_string(),
            health_endpoint: "/health".to_string(),
            check_interval: Duration::from_secs(30),
            timeout: Duration::from_secs(2),
            max_retries: 3,
        });
        
        aggregator.add_service(ServiceConfig {
            name: "monitoring".to_string(),
            url: "http://localhost:9010".to_string(),
            health_endpoint: "/health".to_string(),
            check_interval: Duration::from_secs(30),
            timeout: Duration::from_secs(2),
            max_retries: 3,
        });
        
        aggregator
    }
    
    pub async fn start_monitoring(&self) {
        let services = self.services.clone();
        let health_status = self.health_status.clone();
        let error_counts = self.error_counts.clone();
        let client = self.client.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(15));
            
            loop {
                interval.tick().await;
                
                for (service_name, config) in &services {
                    let start = Instant::now();
                    let health_url = format!("{}{}", config.url, config.health_endpoint);
                    
                    match client
                        .get(&health_url)
                        .timeout(config.timeout)
                        .send()
                        .await
                    {
                        Ok(response) => {
                            let response_time = start.elapsed().as_millis() as u64;
                            let is_healthy = response.status().is_success();
                            
                            let health = ServiceHealth {
                                service: service_name.clone(),
                                status: if is_healthy { "healthy".to_string() } else { "unhealthy".to_string() },
                                last_check: chrono::Utc::now(),
                                response_time_ms: Some(response_time),
                                error_count: if is_healthy {
                                    0
                                } else {
                                    error_counts.read().await.get(service_name).copied().unwrap_or(0) + 1
                                },
                                consecutive_failures: 0,
                                last_error: None,
                            };
                            
                            if is_healthy {
                                info!("Service {} is healthy ({}ms)", service_name, response_time);
                                // 重置错误计数
                                error_counts.write().await.insert(service_name.clone(), 0);
                            } else {
                                warn!("Service {} returned unhealthy status", service_name);
                                let mut counts = error_counts.write().await;
                                let current_count = counts.get(service_name).copied().unwrap_or(0);
                                counts.insert(service_name.clone(), current_count + 1);
                            }
                            
                            health_status.write().await.insert(service_name.clone(), health);
                        }
                        Err(err) => {
                            error!("Failed to check service {}: {}", service_name, err);
                            
                            // 增加错误计数
                            let mut counts = error_counts.write().await;
                            let current_count = counts.get(service_name).copied().unwrap_or(0);
                            counts.insert(service_name.clone(), current_count + 1);
                            
                            let health = ServiceHealth {
                                service: service_name.clone(),
                                status: "unhealthy".to_string(),
                                last_check: chrono::Utc::now(),
                                response_time_ms: None,
                                error_count: current_count + 1,
                                consecutive_failures: current_count + 1,
                                last_error: Some(err.to_string()),
                            };
                            
                            health_status.write().await.insert(service_name.clone(), health);
                        }
                    }
                }
                
                // 记录总体状态
                let status = health_status.read().await;
                let healthy_count = status.values().filter(|h| h.status == "healthy").count();
                let total_count = status.len();
                
                if total_count > 0 {
                    info!("Health check completed: {}/{} services healthy", healthy_count, total_count);
                }
            }
        });
    }
    
    pub async fn get_service_health(&self, service_name: &str) -> Option<ServiceHealth> {
        self.health_status.read().await.get(service_name).cloned()
    }
    
    pub async fn get_all_health(&self) -> HashMap<String, ServiceHealth> {
        self.health_status.read().await.clone()
    }
    
    pub async fn get_error_count(&self, service_name: &str) -> u32 {
        self.error_counts.read().await.get(service_name).copied().unwrap_or(0)
    }
    
    pub async fn reset_error_count(&self, service_name: &str) {
        self.error_counts.write().await.insert(service_name.to_string(), 0);
    }
    
    pub async fn get_overall_health(&self) -> (String, u32, u32) {
        let status = self.health_status.read().await;
        let healthy_count = status.values().filter(|h| h.status == "healthy").count() as u32;
        let total_count = status.len() as u32;
        
        let overall_status = if total_count == 0 {
            "unknown".to_string()
        } else if healthy_count == total_count {
            "healthy".to_string()
        } else if healthy_count > total_count / 2 {
            "degraded".to_string()
        } else {
            "unhealthy".to_string()
        };
        
        (overall_status, healthy_count, total_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::Duration;
    
    #[tokio::test]
    async fn test_health_aggregator_creation() {
        let aggregator = HealthAggregator::new();
        assert_eq!(aggregator.services.len(), 0);
    }
    
    #[tokio::test]
    async fn test_add_service() {
        let mut aggregator = HealthAggregator::new();
        let config = ServiceConfig {
            name: "test".to_string(),
            url: "http://localhost:8080".to_string(),
            health_endpoint: "/health".to_string(),
            check_interval: Duration::from_secs(30),
            timeout: Duration::from_secs(2),
            max_retries: 3,
        };
        
        aggregator.add_service(config);
        assert_eq!(aggregator.services.len(), 1);
        assert!(aggregator.services.contains_key("test"));
    }
    
    #[tokio::test]
    async fn test_default_services() {
        let aggregator = HealthAggregator::with_default_services();
        assert!(!aggregator.services.is_empty());
        assert!(aggregator.services.contains_key("gateway"));
        assert!(aggregator.services.contains_key("trading"));
        assert!(aggregator.services.contains_key("market"));
    }
}