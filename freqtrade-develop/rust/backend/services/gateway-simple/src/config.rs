//! Configuration for Gateway Simple Service

use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub upstream_services: UpstreamServices,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpstreamServices {
    pub market_data: String,
    pub trading: String,
    pub risk: String,
    pub strategy: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            upstream_services: UpstreamServices {
                market_data: "http://localhost:8081".to_string(),
                trading: "http://localhost:8082".to_string(),
                risk: "http://localhost:8083".to_string(),
                strategy: "http://localhost:8084".to_string(),
            },
        }
    }
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let mut cfg = Self::default();

        if let Ok(host) = env::var("GATEWAY_HOST") {
            if !host.trim().is_empty() {
                cfg.host = host;
            }
        }
        if let Ok(port) = env::var("GATEWAY_PORT") {
            if let Ok(p) = port.parse::<u16>() { cfg.port = p; }
        }

        if let Ok(url) = env::var("MARKET_DATA_SERVICE_URL") {
            if !url.trim().is_empty() { cfg.upstream_services.market_data = url; }
        }
        if let Ok(url) = env::var("TRADING_SERVICE_URL") {
            if !url.trim().is_empty() { cfg.upstream_services.trading = url; }
        }
        if let Ok(url) = env::var("RISK_SERVICE_URL") {
            if !url.trim().is_empty() { cfg.upstream_services.risk = url; }
        }
        if let Ok(url) = env::var("STRATEGY_SERVICE_URL") {
            if !url.trim().is_empty() { cfg.upstream_services.strategy = url; }
        }

        Ok(cfg)
    }
}
