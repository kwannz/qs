// Simplified metrics implementation
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Simple metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleMetricsConfig {
    pub enabled: bool,
    pub namespace: String,
}

impl Default for SimpleMetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            namespace: "crypto_quant".to_string(),
        }
    }
}

/// Simple business metrics collector using atomic counters
pub struct SimpleBusinessMetrics {
    // Trading metrics
    pub trades_total: AtomicU64,
    pub trades_successful: AtomicU64,
    pub trades_failed: AtomicU64,
    
    // Order metrics
    pub orders_placed: AtomicU64,
    pub orders_filled: AtomicU64,
    pub orders_cancelled: AtomicU64,
    
    // API metrics
    pub api_requests_total: AtomicU64,
    
    // Error metrics
    pub errors_total: AtomicU64,
    
    config: SimpleMetricsConfig,
}

impl SimpleBusinessMetrics {
    pub fn new(config: SimpleMetricsConfig) -> Self {
        Self {
            trades_total: AtomicU64::new(0),
            trades_successful: AtomicU64::new(0),
            trades_failed: AtomicU64::new(0),
            orders_placed: AtomicU64::new(0),
            orders_filled: AtomicU64::new(0),
            orders_cancelled: AtomicU64::new(0),
            api_requests_total: AtomicU64::new(0),
            errors_total: AtomicU64::new(0),
            config,
        }
    }
    
    /// Record a successful trade
    pub async fn record_successful_trade(&self, _volume: f64, _pnl: f64) {
        if !self.config.enabled {
            return;
        }
        
        self.trades_total.fetch_add(1, Ordering::Relaxed);
        self.trades_successful.fetch_add(1, Ordering::Relaxed);
        
        tracing::info!("Trade recorded successfully");
    }
    
    /// Record a failed trade
    pub async fn record_failed_trade(&self, _error_type: &str) {
        if !self.config.enabled {
            return;
        }
        
        self.trades_total.fetch_add(1, Ordering::Relaxed);
        self.trades_failed.fetch_add(1, Ordering::Relaxed);
        
        tracing::warn!("Failed trade recorded");
    }
    
    /// Record order placement
    pub async fn record_order_placed(&self, _symbol: &str, _side: &str, _order_type: &str) {
        if !self.config.enabled {
            return;
        }
        
        self.orders_placed.fetch_add(1, Ordering::Relaxed);
        tracing::debug!("Order placement recorded");
    }
    
    /// Record order fill
    pub async fn record_order_filled(&self, _symbol: &str, _side: &str, _fill_time: Duration) {
        if !self.config.enabled {
            return;
        }
        
        self.orders_filled.fetch_add(1, Ordering::Relaxed);
        tracing::debug!("Order fill recorded");
    }
    
    /// Record API request
    pub async fn record_api_request(&self, _method: &str, _endpoint: &str, _status: u16, _duration: Duration) {
        if !self.config.enabled {
            return;
        }
        
        self.api_requests_total.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get metrics summary
    pub async fn get_metrics_summary(&self) -> SimpleMetricsSummary {
        SimpleMetricsSummary {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)
                .unwrap_or_default().as_secs(),
            trades_total: self.trades_total.load(Ordering::Relaxed),
            trades_successful: self.trades_successful.load(Ordering::Relaxed),
            trades_failed: self.trades_failed.load(Ordering::Relaxed),
            orders_placed: self.orders_placed.load(Ordering::Relaxed),
            orders_filled: self.orders_filled.load(Ordering::Relaxed),
            orders_cancelled: self.orders_cancelled.load(Ordering::Relaxed),
            api_requests_total: self.api_requests_total.load(Ordering::Relaxed),
            errors_total: self.errors_total.load(Ordering::Relaxed),
        }
    }
}

/// Simple metrics summary
#[derive(Debug, Clone, Serialize)]
pub struct SimpleMetricsSummary {
    pub timestamp: u64,
    pub trades_total: u64,
    pub trades_successful: u64,
    pub trades_failed: u64,
    pub orders_placed: u64,
    pub orders_filled: u64,
    pub orders_cancelled: u64,
    pub api_requests_total: u64,
    pub errors_total: u64,
}

/// Global simple metrics instance - thread-safe singleton
use std::sync::OnceLock;
static SIMPLE_BUSINESS_METRICS: OnceLock<SimpleBusinessMetrics> = OnceLock::new();

/// Initialize simple metrics system
pub async fn init_simple_metrics() -> Result<()> {
    let config = SimpleMetricsConfig::default();
    init_simple_metrics_with_config(config).await
}

/// Initialize simple metrics with custom configuration
pub async fn init_simple_metrics_with_config(config: SimpleMetricsConfig) -> Result<()> {
    if !config.enabled {
        tracing::info!("Simple metrics collection disabled");
        return Ok(());
    }
    
    // Initialize global business metrics instance
    let metrics = SimpleBusinessMetrics::new(config.clone());
    SIMPLE_BUSINESS_METRICS.set(metrics).map_err(|_| 
        anyhow::anyhow!("Simple metrics already initialized"))?;
    
    tracing::info!(
        namespace = %config.namespace,
        "Simple metrics collection initialized"
    );
    
    Ok(())
}

/// Get global simple business metrics instance (thread-safe)
pub fn get_simple_business_metrics() -> Option<&'static SimpleBusinessMetrics> {
    SIMPLE_BUSINESS_METRICS.get()
}

/// Export simple metrics as JSON
pub fn export_simple_metrics() -> Result<String> {
    if let Some(metrics) = get_simple_business_metrics() {
        let runtime = tokio::runtime::Handle::current();
        let summary = runtime.block_on(metrics.get_metrics_summary());
        Ok(serde_json::to_string_pretty(&summary)?)
    } else {
        Ok("{}".to_string())
    }
}

/// Simple metrics middleware for timing operations
pub struct SimpleMetricsMiddleware;

impl SimpleMetricsMiddleware {
    /// Time an operation and record metrics
    pub async fn time_operation<F, T, E>(
        operation_name: &str,
        operation: F,
    ) -> Result<T, E>
    where
        F: std::future::Future<Output = Result<T, E>>,
    {
        let start = std::time::Instant::now();
        let result = operation.await;
        let duration = start.elapsed();
        
        tracing::info!(
            operation = operation_name,
            duration_ms = duration.as_millis(),
            success = result.is_ok(),
            "Operation completed"
        );
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_simple_metrics_initialization() {
        let config = SimpleMetricsConfig {
            enabled: true,
            ..Default::default()
        };
        
        let result = init_simple_metrics_with_config(config).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_simple_business_metrics() {
        let config = SimpleMetricsConfig::default();
        let metrics = SimpleBusinessMetrics::new(config);
        
        // Test trade recording
        metrics.record_successful_trade(100.0, 5.0).await;
        assert_eq!(metrics.trades_total.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.trades_successful.load(Ordering::Relaxed), 1);
        
        metrics.record_failed_trade("network_error").await;
        assert_eq!(metrics.trades_total.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.trades_failed.load(Ordering::Relaxed), 1);
        
        // Test order recording
        metrics.record_order_placed("BTCUSDT", "buy", "market").await;
        assert_eq!(metrics.orders_placed.load(Ordering::Relaxed), 1);
        
        // Test metrics summary
        let summary = metrics.get_metrics_summary().await;
        assert_eq!(summary.trades_total, 2);
        assert_eq!(summary.trades_successful, 1);
        assert_eq!(summary.trades_failed, 1);
    }
    
    #[test]
    fn test_simple_metrics_export() {
        // This test will only work if metrics are initialized
        let result = export_simple_metrics();
        assert!(result.is_ok());
    }
}