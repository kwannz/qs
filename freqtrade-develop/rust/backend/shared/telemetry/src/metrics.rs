use anyhow::Result;
use metrics::{counter, gauge, histogram, register_counter, register_gauge, register_histogram};
use prometheus::{Encoder, Registry, TextEncoder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub enabled: bool,
    pub export_interval: Duration,
    pub prometheus_endpoint: String,
    pub namespace: String,
    pub labels: HashMap<String, String>,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        let mut labels = HashMap::new();
        labels.insert("service".to_string(), "crypto-quant-platform".to_string());
        labels.insert("version".to_string(), "1.0.0".to_string());
        
        Self {
            enabled: true,
            export_interval: Duration::from_secs(30),
            prometheus_endpoint: "0.0.0.0:9090".to_string(),
            namespace: "crypto_quant".to_string(),
            labels,
        }
    }
}

/// Business metrics collector
pub struct BusinessMetrics {
    // Trading metrics
    pub trades_total: Arc<RwLock<u64>>,
    pub trades_successful: Arc<RwLock<u64>>,
    pub trades_failed: Arc<RwLock<u64>>,
    pub trading_volume_total: Arc<RwLock<f64>>,
    pub trading_pnl_total: Arc<RwLock<f64>>,
    
    // Order metrics
    pub orders_placed: Arc<RwLock<u64>>,
    pub orders_filled: Arc<RwLock<u64>>,
    pub orders_cancelled: Arc<RwLock<u64>>,
    pub order_fill_time: Arc<RwLock<Vec<Duration>>>,
    
    // Strategy metrics
    pub active_strategies: Arc<RwLock<u64>>,
    pub strategy_signals_generated: Arc<RwLock<u64>>,
    pub strategy_pnl: Arc<RwLock<HashMap<String, f64>>>,
    
    // Risk metrics
    pub risk_violations: Arc<RwLock<u64>>,
    pub portfolio_value: Arc<RwLock<f64>>,
    pub max_drawdown: Arc<RwLock<f64>>,
    pub var_95: Arc<RwLock<f64>>,
    
    // System metrics
    pub api_requests_total: Arc<RwLock<u64>>,
    pub api_request_duration: Arc<RwLock<Vec<Duration>>>,
    pub database_connections_active: Arc<RwLock<u64>>,
    pub redis_connections_active: Arc<RwLock<u64>>,
    
    // Error metrics
    pub errors_total: Arc<RwLock<u64>>,
    pub errors_by_type: Arc<RwLock<HashMap<String, u64>>>,
    
    config: MetricsConfig,
}

impl BusinessMetrics {
    pub fn new(config: MetricsConfig) -> Self {
        Self {
            trades_total: Arc::new(RwLock::new(0)),
            trades_successful: Arc::new(RwLock::new(0)),
            trades_failed: Arc::new(RwLock::new(0)),
            trading_volume_total: Arc::new(RwLock::new(0.0)),
            trading_pnl_total: Arc::new(RwLock::new(0.0)),
            
            orders_placed: Arc::new(RwLock::new(0)),
            orders_filled: Arc::new(RwLock::new(0)),
            orders_cancelled: Arc::new(RwLock::new(0)),
            order_fill_time: Arc::new(RwLock::new(Vec::new())),
            
            active_strategies: Arc::new(RwLock::new(0)),
            strategy_signals_generated: Arc::new(RwLock::new(0)),
            strategy_pnl: Arc::new(RwLock::new(HashMap::new())),
            
            risk_violations: Arc::new(RwLock::new(0)),
            portfolio_value: Arc::new(RwLock::new(0.0)),
            max_drawdown: Arc::new(RwLock::new(0.0)),
            var_95: Arc::new(RwLock::new(0.0)),
            
            api_requests_total: Arc::new(RwLock::new(0)),
            api_request_duration: Arc::new(RwLock::new(Vec::new())),
            database_connections_active: Arc::new(RwLock::new(0)),
            redis_connections_active: Arc::new(RwLock::new(0)),
            
            errors_total: Arc::new(RwLock::new(0)),
            errors_by_type: Arc::new(RwLock::new(HashMap::new())),
            
            config,
        }
    }
    
    /// Record a successful trade
    pub async fn record_successful_trade(&self, volume: f64, pnl: f64) {
        if !self.config.enabled {
            return;
        }
        
        *self.trades_total.write().await += 1;
        *self.trades_successful.write().await += 1;
        *self.trading_volume_total.write().await += volume;
        *self.trading_pnl_total.write().await += pnl;
        
        // Record Prometheus metrics
        counter!("crypto_quant_trades_total", "status" => "success").increment(1);
        histogram!("crypto_quant_trading_volume", "status" => "success").record(volume);
        histogram!("crypto_quant_trading_pnl", "status" => "success").record(pnl);
    }
    
    /// Record a failed trade
    pub async fn record_failed_trade(&self, error_type: &str) {
        if !self.config.enabled {
            return;
        }
        
        *self.trades_total.write().await += 1;
        *self.trades_failed.write().await += 1;
        
        let mut errors_by_type = self.errors_by_type.write().await;
        *errors_by_type.entry(error_type.to_string()).or_insert(0) += 1;
        
        // Record Prometheus metrics
        counter!("crypto_quant_trades_total", "status" => "failed", "error_type" => error_type).increment(1);
    }
    
    /// Record order placement
    pub async fn record_order_placed(&self, symbol: &str, side: &str, order_type: &str) {
        if !self.config.enabled {
            return;
        }
        
        *self.orders_placed.write().await += 1;
        
        counter!(
            "crypto_quant_orders_total",
            "action" => "placed",
            "symbol" => symbol,
            "side" => side,
            "order_type" => order_type
        ).increment(1);
    }
    
    /// Record order fill
    pub async fn record_order_filled(&self, symbol: &str, side: &str, fill_time: Duration) {
        if !self.config.enabled {
            return;
        }
        
        *self.orders_filled.write().await += 1;
        self.order_fill_time.write().await.push(fill_time);
        
        counter!(
            "crypto_quant_orders_total",
            "action" => "filled",
            "symbol" => symbol,
            "side" => side
        ).increment(1);
        
        histogram!("crypto_quant_order_fill_duration_seconds").record(fill_time.as_secs_f64());
    }
    
    /// Record strategy signal
    pub async fn record_strategy_signal(&self, strategy_id: &str, signal_type: &str) {
        if !self.config.enabled {
            return;
        }
        
        *self.strategy_signals_generated.write().await += 1;
        
        counter!(
            "crypto_quant_strategy_signals_total",
            "strategy_id" => strategy_id,
            "signal_type" => signal_type
        ).increment(1);
    }
    
    /// Update strategy PnL
    pub async fn update_strategy_pnl(&self, strategy_id: &str, pnl: f64) {
        if !self.config.enabled {
            return;
        }
        
        let mut strategy_pnl = self.strategy_pnl.write().await;
        strategy_pnl.insert(strategy_id.to_string(), pnl);
        
        gauge!("crypto_quant_strategy_pnl", "strategy_id" => strategy_id).set(pnl);
    }
    
    /// Record risk violation
    pub async fn record_risk_violation(&self, risk_type: &str, severity: &str) {
        if !self.config.enabled {
            return;
        }
        
        *self.risk_violations.write().await += 1;
        
        counter!(
            "crypto_quant_risk_violations_total",
            "risk_type" => risk_type,
            "severity" => severity
        ).increment(1);
    }
    
    /// Update portfolio metrics
    pub async fn update_portfolio_metrics(&self, portfolio_value: f64, max_drawdown: f64, var_95: f64) {
        if !self.config.enabled {
            return;
        }
        
        *self.portfolio_value.write().await = portfolio_value;
        *self.max_drawdown.write().await = max_drawdown;
        *self.var_95.write().await = var_95;
        
        gauge!("crypto_quant_portfolio_value").set(portfolio_value);
        gauge!("crypto_quant_portfolio_max_drawdown").set(max_drawdown);
        gauge!("crypto_quant_portfolio_var_95").set(var_95);
    }
    
    /// Record API request
    pub async fn record_api_request(&self, method: &str, endpoint: &str, status: u16, duration: Duration) {
        if !self.config.enabled {
            return;
        }
        
        *self.api_requests_total.write().await += 1;
        self.api_request_duration.write().await.push(duration);
        
        counter!(
            "crypto_quant_http_requests_total",
            "method" => method,
            "endpoint" => endpoint,
            "status" => status.to_string().as_str()
        ).increment(1);
        
        histogram!(
            "crypto_quant_http_request_duration_seconds",
            "method" => method,
            "endpoint" => endpoint
        ).record(duration.as_secs_f64());
    }
    
    /// Update system resource metrics
    pub async fn update_system_metrics(&self, db_connections: u64, redis_connections: u64) {
        if !self.config.enabled {
            return;
        }
        
        *self.database_connections_active.write().await = db_connections;
        *self.redis_connections_active.write().await = redis_connections;
        
        gauge!("crypto_quant_database_connections_active").set(db_connections as f64);
        gauge!("crypto_quant_redis_connections_active").set(redis_connections as f64);
    }
    
    /// Get metrics summary for health checks
    pub async fn get_metrics_summary(&self) -> MetricsSummary {
        MetricsSummary {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)
                .unwrap_or_default().as_secs(),
            trades_total: *self.trades_total.read().await,
            trades_successful: *self.trades_successful.read().await,
            trades_failed: *self.trades_failed.read().await,
            trading_volume_total: *self.trading_volume_total.read().await,
            trading_pnl_total: *self.trading_pnl_total.read().await,
            orders_placed: *self.orders_placed.read().await,
            orders_filled: *self.orders_filled.read().await,
            active_strategies: *self.active_strategies.read().await,
            risk_violations: *self.risk_violations.read().await,
            portfolio_value: *self.portfolio_value.read().await,
            api_requests_total: *self.api_requests_total.read().await,
            errors_total: *self.errors_total.read().await,
        }
    }
}

/// Metrics summary for monitoring
#[derive(Debug, Clone, Serialize)]
pub struct MetricsSummary {
    pub timestamp: u64,
    pub trades_total: u64,
    pub trades_successful: u64,
    pub trades_failed: u64,
    pub trading_volume_total: f64,
    pub trading_pnl_total: f64,
    pub orders_placed: u64,
    pub orders_filled: u64,
    pub active_strategies: u64,
    pub risk_violations: u64,
    pub portfolio_value: f64,
    pub api_requests_total: u64,
    pub errors_total: u64,
}

/// Global metrics instance
static mut BUSINESS_METRICS: Option<BusinessMetrics> = None;
static INIT: std::sync::Once = std::sync::Once::new();

/// Initialize metrics system
pub async fn init_metrics() -> Result<()> {
    let config = MetricsConfig::default();
    init_metrics_with_config(config).await
}

/// Initialize metrics with custom configuration
pub async fn init_metrics_with_config(config: MetricsConfig) -> Result<()> {
    if !config.enabled {
        tracing::info!("Metrics collection disabled");
        return Ok();
    }
    
    // Initialize Prometheus metrics
    metrics_prometheus::install();
    
    // Register standard metrics
    register_counter!("crypto_quant_trades_total", "Total number of trades executed");
    register_histogram!("crypto_quant_trading_volume", "Trading volume per trade");
    register_histogram!("crypto_quant_trading_pnl", "Profit and loss per trade");
    
    register_counter!("crypto_quant_orders_total", "Total number of orders");
    register_histogram!("crypto_quant_order_fill_duration_seconds", "Order fill duration");
    
    register_counter!("crypto_quant_strategy_signals_total", "Strategy signals generated");
    register_gauge!("crypto_quant_strategy_pnl", "Strategy profit and loss");
    
    register_counter!("crypto_quant_risk_violations_total", "Risk violations count");
    register_gauge!("crypto_quant_portfolio_value", "Total portfolio value");
    register_gauge!("crypto_quant_portfolio_max_drawdown", "Maximum portfolio drawdown");
    register_gauge!("crypto_quant_portfolio_var_95", "Portfolio Value at Risk (95%)");
    
    register_counter!("crypto_quant_http_requests_total", "HTTP requests count");
    register_histogram!("crypto_quant_http_request_duration_seconds", "HTTP request duration");
    
    register_gauge!("crypto_quant_database_connections_active", "Active database connections");
    register_gauge!("crypto_quant_redis_connections_active", "Active Redis connections");
    
    // Initialize global business metrics instance
    unsafe {
        INIT.call_once(|| {
            BUSINESS_METRICS = Some(BusinessMetrics::new(config.clone()));
        });
    }
    
    tracing::info!(
        prometheus_endpoint = %config.prometheus_endpoint,
        namespace = %config.namespace,
        "Metrics collection initialized"
    );
    
    Ok(())
}

/// Get global business metrics instance
pub fn get_business_metrics() -> &'static BusinessMetrics {
    unsafe {
        BUSINESS_METRICS.as_ref().expect("Business metrics not initialized")
    }
}

/// Export Prometheus metrics
pub fn export_prometheus_metrics() -> Result<String> {
    let registry = prometheus::default_registry();
    let encoder = TextEncoder::new();
    let metric_families = registry.gather();
    
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)?;
    
    Ok(String::from_utf8(buffer)?)
}

/// Metrics middleware for timing operations
pub struct MetricsMiddleware;

impl MetricsMiddleware {
    /// Time an operation and record metrics
    pub async fn time_operation<F, T, E>(
        operation_name: &str,
        operation: F,
    ) -> Result<T, E>
    where
        F: std::future::Future<Output = Result<T, E>>,
    {
        let start = Instant::now();
        let result = operation.await;
        let duration = start.elapsed();
        
        let status = if result.is_ok() { "success" } else { "failure" };
        
        histogram!(
            "crypto_quant_operation_duration_seconds",
            "operation" => operation_name,
            "status" => status
        ).record(duration.as_secs_f64());
        
        counter!(
            "crypto_quant_operations_total",
            "operation" => operation_name,
            "status" => status
        ).increment(1);
        
        result
    }
}

/// Convenience macros for metrics recording
#[macro_export]
macro_rules! record_trade {
    ($success:expr, $volume:expr, $pnl:expr, $error_type:expr) => {
        if let Ok(metrics) = $crate::metrics::get_business_metrics() {
            if $success {
                metrics.record_successful_trade($volume, $pnl).await;
            } else {
                metrics.record_failed_trade($error_type).await;
            }
        }
    };
}

#[macro_export]
macro_rules! record_api_request {
    ($method:expr, $endpoint:expr, $status:expr, $duration:expr) => {
        if let Ok(metrics) = $crate::metrics::get_business_metrics() {
            metrics.record_api_request($method, $endpoint, $status, $duration).await;
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[tokio::test]
    async fn test_metrics_initialization() {
        let config = MetricsConfig {
            enabled: true,
            ..Default::default()
        };
        
        let result = init_metrics_with_config(config).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_business_metrics() {
        let config = MetricsConfig::default();
        let metrics = BusinessMetrics::new(config);
        
        // Test trade recording
        metrics.record_successful_trade(100.0, 5.0).await;
        assert_eq!(*metrics.trades_total.read().await, 1);
        assert_eq!(*metrics.trades_successful.read().await, 1);
        
        metrics.record_failed_trade("network_error").await;
        assert_eq!(*metrics.trades_total.read().await, 2);
        assert_eq!(*metrics.trades_failed.read().await, 1);
        
        // Test order recording
        metrics.record_order_placed("BTCUSDT", "buy", "market").await;
        assert_eq!(*metrics.orders_placed.read().await, 1);
        
        metrics.record_order_filled("BTCUSDT", "buy", Duration::from_millis(50)).await;
        assert_eq!(*metrics.orders_filled.read().await, 1);
        
        // Test metrics summary
        let summary = metrics.get_metrics_summary().await;
        assert_eq!(summary.trades_total, 2);
        assert_eq!(summary.trades_successful, 1);
        assert_eq!(summary.trades_failed, 1);
    }
    
    #[test]
    fn test_prometheus_export() {
        let result = export_prometheus_metrics();
        assert!(result.is_ok());
    }
}