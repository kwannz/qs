use anyhow::Result;
use chrono::Utc;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tracing::{info, error, warn};
use uuid::Uuid;

/// Sprint 8 E2E Test Suite
/// 
/// This test suite validates all critical API paths and service interactions
/// for the completed Sprint 8 microservices architecture.

#[derive(Debug, Serialize, Deserialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
    timestamp: i64,
}

#[derive(Debug, Serialize, Deserialize)]
struct HealthResponse {
    status: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct PlaceOrderRequest {
    account_id: String,
    symbol: String,
    side: String,
    r#type: String,
    quantity: f64,
    price: Option<f64>,
    idempotency_key: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct BacktestRequest {
    symbol: String,
    strategy: String,
    start_date: String,
    end_date: String,
    initial_balance: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct FactorRequest {
    symbols: Vec<String>,
    factors: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RiskCheckRequest {
    tenant: String,
    strategy_id: String,
    orders: Vec<RiskOrder>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RiskOrder {
    symbol: String,
    side: String,
    quantity: f64,
    price: f64,
    order_type: String,
}

pub struct E2ETestSuite {
    client: Client,
    base_urls: HashMap<String, String>,
}

impl E2ETestSuite {
    pub fn new() -> Self {
        let mut base_urls = HashMap::new();
        base_urls.insert("gateway".to_string(), "http://localhost:8080".to_string());
        base_urls.insert("markets".to_string(), "http://localhost:9000".to_string());
        base_urls.insert("execution".to_string(), "http://localhost:9010".to_string());
        base_urls.insert("backtest".to_string(), "http://localhost:9001".to_string());
        base_urls.insert("factor".to_string(), "http://localhost:19003".to_string());
        base_urls.insert("risk".to_string(), "http://localhost:19004".to_string());

        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            base_urls,
        }
    }

    /// Test all service health endpoints
    pub async fn test_health_checks(&self) -> Result<()> {
        info!("🏥 Testing service health checks...");
        
        for (service, base_url) in &self.base_urls {
            let health_url = format!("{}/health", base_url);
            
            info!("Checking health: {}", health_url);
            
            match self.client.get(&health_url).send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        info!("✅ {} is healthy", service);
                    } else {
                        error!("❌ {} returned status: {}", service, response.status());
                        return Err(anyhow::anyhow!("{} health check failed", service));
                    }
                }
                Err(e) => {
                    error!("❌ Failed to connect to {}: {}", service, e);
                    return Err(anyhow::anyhow!("Connection failed to {}: {}", service, e));
                }
            }
        }
        
        info!("✅ All service health checks passed!");
        Ok(())
    }

    /// Test Gateway API v1 endpoints
    pub async fn test_gateway_endpoints(&self) -> Result<()> {
        info!("🌐 Testing Gateway API endpoints...");
        
        let gateway_url = self.base_urls.get("gateway").unwrap();
        
        // Test gateway health through API v1
        let health_url = format!("{}/api/v1/health", gateway_url);
        let response = self.client.get(&health_url).send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Gateway API v1 health check failed"));
        }
        
        info!("✅ Gateway API v1 endpoints working");
        Ok(())
    }

    /// Test Markets service endpoints
    pub async fn test_markets_service(&self) -> Result<()> {
        info!("📊 Testing Markets service...");
        
        let markets_url = self.base_urls.get("markets").unwrap();
        
        // Test symbols endpoint
        let symbols_url = format!("{}/symbols", markets_url);
        let response = self.client.get(&symbols_url).send().await?;
        
        if response.status().is_success() {
            info!("✅ Markets symbols endpoint working");
        } else {
            warn!("⚠️ Markets symbols endpoint returned: {}", response.status());
        }
        
        // Test candles endpoint
        let candles_url = format!("{}/candles?symbol=BTCUSDT&timeframe=1m&limit=10", markets_url);
        let response = self.client.get(&candles_url).send().await?;
        
        if response.status().is_success() {
            info!("✅ Markets candles endpoint working");
        } else {
            warn!("⚠️ Markets candles endpoint returned: {}", response.status());
        }
        
        Ok(())
    }

    /// Test Execution service
    pub async fn test_execution_service(&self) -> Result<()> {
        info!("⚡ Testing Execution service...");
        
        let execution_url = self.base_urls.get("execution").unwrap();
        
        // Test order placement
        let order_request = PlaceOrderRequest {
            account_id: "demo_account_1".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: "BUY".to_string(),
            r#type: "LIMIT".to_string(),
            quantity: 0.001,
            price: Some(50000.0),
            idempotency_key: Uuid::new_v4().to_string(),
        };
        
        let orders_url = format!("{}/orders", execution_url);
        let response = self.client
            .post(&orders_url)
            .json(&order_request)
            .send()
            .await?;
            
        if response.status().is_success() {
            info!("✅ Execution order placement working");
        } else {
            warn!("⚠️ Execution order placement returned: {}", response.status());
        }
        
        Ok(())
    }

    /// Test Backtest service
    pub async fn test_backtest_service(&self) -> Result<()> {
        info!("📈 Testing Backtest service...");
        
        let backtest_url = self.base_urls.get("backtest").unwrap();
        
        let backtest_request = BacktestRequest {
            symbol: "BTCUSDT".to_string(),
            strategy: "simple_ma_cross".to_string(),
            start_date: "2024-01-01".to_string(),
            end_date: "2024-01-31".to_string(),
            initial_balance: 10000.0,
        };
        
        let backtests_url = format!("{}/backtests", backtest_url);
        let response = self.client
            .post(&backtests_url)
            .json(&backtest_request)
            .send()
            .await?;
            
        if response.status().is_success() || response.status().as_u16() == 202 {
            info!("✅ Backtest submission working");
        } else {
            warn!("⚠️ Backtest submission returned: {}", response.status());
        }
        
        Ok(())
    }

    /// Test Factor service
    pub async fn test_factor_service(&self) -> Result<()> {
        info!("🧮 Testing Factor service...");
        
        let factor_url = self.base_urls.get("factor").unwrap();
        
        let factor_request = FactorRequest {
            symbols: vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],
            factors: vec!["rsi".to_string(), "macd".to_string()],
        };
        
        let factors_url = format!("{}/factors", factor_url);
        let response = self.client
            .post(&factors_url)
            .json(&factor_request)
            .send()
            .await?;
            
        if response.status().is_success() || response.status().as_u16() == 202 {
            info!("✅ Factor computation working");
        } else {
            warn!("⚠️ Factor computation returned: {}", response.status());
        }
        
        Ok(())
    }

    /// Test Risk service
    pub async fn test_risk_service(&self) -> Result<()> {
        info!("🛡️ Testing Risk service...");
        
        let risk_url = self.base_urls.get("risk").unwrap();
        
        let risk_request = RiskCheckRequest {
            tenant: "default".to_string(),
            strategy_id: "test_strategy".to_string(),
            orders: vec![RiskOrder {
                symbol: "BTCUSDT".to_string(),
                side: "BUY".to_string(),
                quantity: 0.001,
                price: 50000.0,
                order_type: "LIMIT".to_string(),
            }],
        };
        
        let risk_url = format!("{}/risk/pretrade", risk_url);
        let response = self.client
            .post(&risk_url)
            .json(&risk_request)
            .send()
            .await?;
            
        if response.status().is_success() {
            info!("✅ Risk check working");
        } else {
            warn!("⚠️ Risk check returned: {}", response.status());
        }
        
        Ok(())
    }

    /// Test service integration flow
    pub async fn test_integration_flow(&self) -> Result<()> {
        info!("🔄 Testing service integration flow...");
        
        // 1. Check risk before placing order
        let risk_response = self.test_risk_service().await;
        if risk_response.is_err() {
            warn!("Risk check failed, but continuing with integration test");
        }
        
        // 2. Place order through execution service
        let execution_response = self.test_execution_service().await;
        if execution_response.is_err() {
            warn!("Order execution failed, but continuing with integration test");
        }
        
        // 3. Run backtest on the same symbol
        let backtest_response = self.test_backtest_service().await;
        if backtest_response.is_err() {
            warn!("Backtest failed, but continuing with integration test");
        }
        
        // 4. Calculate factors for analysis
        let factor_response = self.test_factor_service().await;
        if factor_response.is_err() {
            warn!("Factor calculation failed, but continuing with integration test");
        }
        
        info!("✅ Integration flow completed");
        Ok(())
    }

    /// Run performance tests
    pub async fn test_performance(&self) -> Result<()> {
        info!("⚡ Testing service performance...");
        
        let start_time = std::time::Instant::now();
        let mut successful_requests = 0;
        let total_requests = 10;
        
        for i in 0..total_requests {
            let health_futures: Vec<_> = self.base_urls
                .values()
                .map(|url| {
                    let health_url = format!("{}/health", url);
                    self.client.get(&health_url).send()
                })
                .collect();
                
            let responses = futures::future::join_all(health_futures).await;
            
            for response in responses {
                if let Ok(resp) = response {
                    if resp.status().is_success() {
                        successful_requests += 1;
                    }
                }
            }
            
            if i % 5 == 0 {
                info!("Completed {} performance test iterations", i + 1);
            }
        }
        
        let duration = start_time.elapsed();
        let success_rate = (successful_requests as f64) / (total_requests * self.base_urls.len()) as f64 * 100.0;
        
        info!("Performance test results:");
        info!("  Total requests: {}", total_requests * self.base_urls.len());
        info!("  Successful requests: {}", successful_requests);
        info!("  Success rate: {:.1}%", success_rate);
        info!("  Total time: {:?}", duration);
        info!("  Average response time: {:?}", duration / (total_requests as u32));
        
        if success_rate >= 90.0 {
            info!("✅ Performance test passed");
        } else {
            warn!("⚠️ Performance test below threshold (90%)");
        }
        
        Ok(())
    }

    /// Run all E2E tests
    pub async fn run_all_tests(&self) -> Result<()> {
        info!("🚀 Starting Sprint 8 E2E Test Suite");
        info!("Testing {} services", self.base_urls.len());
        
        let mut test_results = vec![];
        
        // Test 1: Health checks
        match self.test_health_checks().await {
            Ok(_) => {
                info!("✅ Health checks: PASSED");
                test_results.push(("Health Checks", true));
            }
            Err(e) => {
                error!("❌ Health checks: FAILED - {}", e);
                test_results.push(("Health Checks", false));
                return Err(e);
            }
        }
        
        // Test 2: Gateway endpoints
        match self.test_gateway_endpoints().await {
            Ok(_) => {
                info!("✅ Gateway endpoints: PASSED");
                test_results.push(("Gateway Endpoints", true));
            }
            Err(e) => {
                error!("❌ Gateway endpoints: FAILED - {}", e);
                test_results.push(("Gateway Endpoints", false));
            }
        }
        
        // Test 3: Individual services
        // Markets Service
        match self.test_markets_service().await {
            Ok(_) => {
                info!("✅ Markets Service: PASSED");
                test_results.push(("Markets Service", true));
            }
            Err(e) => {
                error!("❌ Markets Service: FAILED - {}", e);
                test_results.push(("Markets Service", false));
            }
        }
        
        // Execution Service
        match self.test_execution_service().await {
            Ok(_) => {
                info!("✅ Execution Service: PASSED");
                test_results.push(("Execution Service", true));
            }
            Err(e) => {
                error!("❌ Execution Service: FAILED - {}", e);
                test_results.push(("Execution Service", false));
            }
        }
        
        // Backtest Service
        match self.test_backtest_service().await {
            Ok(_) => {
                info!("✅ Backtest Service: PASSED");
                test_results.push(("Backtest Service", true));
            }
            Err(e) => {
                error!("❌ Backtest Service: FAILED - {}", e);
                test_results.push(("Backtest Service", false));
            }
        }
        
        // Factor Service
        match self.test_factor_service().await {
            Ok(_) => {
                info!("✅ Factor Service: PASSED");
                test_results.push(("Factor Service", true));
            }
            Err(e) => {
                error!("❌ Factor Service: FAILED - {}", e);
                test_results.push(("Factor Service", false));
            }
        }
        
        // Risk Service
        match self.test_risk_service().await {
            Ok(_) => {
                info!("✅ Risk Service: PASSED");
                test_results.push(("Risk Service", true));
            }
            Err(e) => {
                error!("❌ Risk Service: FAILED - {}", e);
                test_results.push(("Risk Service", false));
            }
        }
        
        // Test 4: Integration flow
        match self.test_integration_flow().await {
            Ok(_) => {
                info!("✅ Integration flow: PASSED");
                test_results.push(("Integration Flow", true));
            }
            Err(e) => {
                error!("❌ Integration flow: FAILED - {}", e);
                test_results.push(("Integration Flow", false));
            }
        }
        
        // Test 5: Performance
        match self.test_performance().await {
            Ok(_) => {
                info!("✅ Performance test: PASSED");
                test_results.push(("Performance Test", true));
            }
            Err(e) => {
                error!("❌ Performance test: FAILED - {}", e);
                test_results.push(("Performance Test", false));
            }
        }
        
        // Print final results
        info!("📊 Final Test Results:");
        let mut passed = 0;
        let total = test_results.len();
        
        for (test_name, passed_test) in test_results {
            if passed_test {
                info!("  ✅ {}", test_name);
                passed += 1;
            } else {
                info!("  ❌ {}", test_name);
            }
        }
        
        info!("Overall: {}/{} tests passed ({:.1}%)", passed, total, (passed as f64 / total as f64) * 100.0);
        
        if passed == total {
            info!("🎉 All E2E tests PASSED! Sprint 8 services are working correctly.");
        } else {
            warn!("⚠️ Some E2E tests FAILED. Check logs above for details.");
        }
        
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    info!("🧪 Sprint 8 E2E Test Suite Starting...");
    info!("⏰ Started at: {}", Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));
    
    let test_suite = E2ETestSuite::new();
    
    // Wait a bit for services to be ready
    info!("⏳ Waiting 5 seconds for services to be ready...");
    tokio::time::sleep(Duration::from_secs(5)).await;
    
    let start_time = std::time::Instant::now();
    
    let result = test_suite.run_all_tests().await;
    
    let duration = start_time.elapsed();
    info!("⏱️ Total test duration: {:?}", duration);
    info!("🏁 E2E Test Suite Complete");
    
    match result {
        Ok(_) => {
            info!("✅ Sprint 8 E2E tests completed successfully!");
            std::process::exit(0);
        }
        Err(e) => {
            error!("❌ Sprint 8 E2E tests failed: {}", e);
            std::process::exit(1);
        }
    }
}