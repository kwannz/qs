// Sprint 1: Unified client helpers for v1 services

use anyhow::Result;
use tonic::transport::Channel;
use std::time::Duration;

pub use crate::execution::{
    ExecutionServiceClient,
    PlaceOrderRequest, PlaceOrderResponse,
    OrderRequest, OrderStatus,
    HealthCheckRequest, HealthCheckResponse,
};

pub use crate::backtest::{
    BacktestServiceClient,
    BacktestRequest, BacktestResult,
};

pub use crate::factor::{
    FactorServiceClient,
    FactorBatchRequest, FactorBatchResult,
};

pub use crate::risk::{
    RiskServiceClient,
    PretradeRequest, PretradeDecision,
};

pub use crate::common::{JobAck, JobId, Status};

/// Sprint 1: Unified trading client for all services
#[derive(Clone)]
pub struct TradingClient {
    pub execution: Option<ExecutionServiceClient<Channel>>,
    pub backtest: Option<BacktestServiceClient<Channel>>,
    pub factor: Option<FactorServiceClient<Channel>>,
    pub risk: Option<RiskServiceClient<Channel>>,
}

impl TradingClient {
    /// Create a new trading client with service endpoints
    pub async fn connect(config: ClientConfig) -> Result<Self> {
        let execution = if let Some(endpoint) = config.execution_endpoint {
            Some(ExecutionServiceClient::connect(endpoint).await?)
        } else {
            None
        };

        let backtest = if let Some(endpoint) = config.backtest_endpoint {
            Some(BacktestServiceClient::connect(endpoint).await?)
        } else {
            None
        };

        let factor = if let Some(endpoint) = config.factor_endpoint {
            Some(FactorServiceClient::connect(endpoint).await?)
        } else {
            None
        };

        let risk = if let Some(endpoint) = config.risk_endpoint {
            Some(RiskServiceClient::connect(endpoint).await?)
        } else {
            None
        };

        Ok(Self {
            execution,
            backtest,
            factor,
            risk,
        })
    }

    /// Sprint 1 Place Order (matches Sprint 1 specification exactly)
    pub async fn place_order(&mut self, request: PlaceOrderRequest) -> Result<PlaceOrderResponse> {
        if let Some(ref mut client) = self.execution {
            let response = client.place_order(request).await?.into_inner();
            Ok(response)
        } else {
            Err(anyhow::anyhow!("Execution service not available"))
        }
    }

    /// Health check for all connected services
    pub async fn health_check(&mut self) -> Result<ServiceHealthStatus> {
        let mut health_status = ServiceHealthStatus::default();

        if let Some(ref mut client) = self.execution {
            match client.health_check(HealthCheckRequest {}).await {
                Ok(response) => {
                    let status = response.into_inner();
                    health_status.execution = Some(status.status);
                }
                Err(e) => {
                    health_status.execution = Some(format!("Error: {}", e));
                }
            }
        }

        // Add other service health checks as needed
        Ok(health_status)
    }
}

/// Client configuration
#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub execution_endpoint: Option<String>,
    pub backtest_endpoint: Option<String>,
    pub factor_endpoint: Option<String>,
    pub risk_endpoint: Option<String>,
    pub timeout: Option<Duration>,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            execution_endpoint: Some("http://localhost:9001".to_string()),
            backtest_endpoint: Some("http://localhost:9002".to_string()),
            factor_endpoint: Some("http://localhost:9003".to_string()),
            risk_endpoint: Some("http://localhost:9004".to_string()),
            timeout: Some(Duration::from_secs(30)),
        }
    }
}

/// Service health status
#[derive(Debug, Default)]
pub struct ServiceHealthStatus {
    pub execution: Option<String>,
    pub backtest: Option<String>,
    pub factor: Option<String>,
    pub risk: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_config_default() {
        let config = ClientConfig::default();
        assert!(config.execution_endpoint.is_some());
        assert!(config.timeout.is_some());
    }
}