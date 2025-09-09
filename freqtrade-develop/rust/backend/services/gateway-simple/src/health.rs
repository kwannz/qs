//! Health Check for upstream services

use anyhow::Result;
use reqwest::Client;
use std::{collections::HashMap, time::Duration};

#[derive(Debug, Clone)]
pub struct HealthChecker {
    services: HashMap<String, String>,
}

impl HealthChecker {
    pub fn new(config: crate::config::Config) -> Self {
        // 仅纳入当前已实现并部署的服务，避免健康检查中的假阴性
        let mut services = HashMap::new();
        services.insert("risk".to_string(), config.upstream_services.risk);
        services.insert("strategy".to_string(), config.upstream_services.strategy);
        Self { services }
    }

    pub async fn check_all(&self) -> Result<HashMap<String, bool>> {
        Ok(self.check_all_services().await)
    }

    pub async fn check_all_services(&self) -> HashMap<String, bool> {
        let client = Client::builder()
            .timeout(Duration::from_secs(3))
            .build()
            .unwrap();

        let mut results = HashMap::new();
        for (name, base) in &self.services {
            let url = format!("{}/health", base.trim_end_matches('/'));
            let ok = match client.get(&url).send().await {
                Ok(resp) => resp.status().is_success(),
                Err(_) => false,
            };
            results.insert(name.clone(), ok);
        }
        results
    }

    pub async fn is_healthy(&self) -> bool {
        self.check_all_services().await.values().all(|v| *v)
    }

    pub async fn start_monitoring(&self) {
        // No background monitoring in simple gateway (yet)
    }
}
