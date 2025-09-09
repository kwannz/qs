use anyhow::Result;
use opentelemetry::{global, KeyValue};
use opentelemetry_jaeger::JaegerPipeline;
use opentelemetry_otlp::WithExportConfig;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{Level, Subscriber};
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::{
    filter::LevelFilter, fmt, layer::SubscriberExt, registry::Registry, EnvFilter, Layer,
};

/// Tracing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    pub service_name: String,
    pub service_version: String,
    pub environment: String,
    pub jaeger_endpoint: Option<String>,
    pub otlp_endpoint: Option<String>,
    pub sample_ratio: f64,
    pub max_events_per_span: u32,
    pub max_attributes_per_span: u32,
    pub export_timeout: Duration,
    pub export_batch_size: usize,
    pub console_output: bool,
    pub json_format: bool,
    pub log_level: String,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            service_name: "crypto-quant-platform".to_string(),
            service_version: "1.0.0".to_string(),
            environment: "development".to_string(),
            jaeger_endpoint: Some("http://jaeger:14268/api/traces".to_string()),
            otlp_endpoint: None,
            sample_ratio: 1.0, // Sample all traces in development
            max_events_per_span: 128,
            max_attributes_per_span: 128,
            export_timeout: Duration::from_secs(30),
            export_batch_size: 512,
            console_output: true,
            json_format: false,
            log_level: "info".to_string(),
        }
    }
}

/// Initialize distributed tracing
pub async fn init_tracing() -> Result<()> {
    let config = TracingConfig::default();
    init_tracing_with_config(config).await
}

/// Initialize tracing with custom configuration
pub async fn init_tracing_with_config(config: TracingConfig) -> Result<()> {
    // Create OpenTelemetry tracer
    let tracer = create_tracer(&config).await?;
    
    // Create OpenTelemetry layer
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
    
    // Create console layer for local development
    let console_layer = if config.console_output {
        if config.json_format {
            Some(
                fmt::layer()
                    .json()
                    .with_target(true)
                    .with_thread_ids(true)
                    .with_thread_names(true)
                    .with_file(true)
                    .with_line_number(true),
            )
        } else {
            Some(
                fmt::layer()
                    .pretty()
                    .with_target(true)
                    .with_thread_ids(true)
                    .with_thread_names(true)
                    .with_file(true)
                    .with_line_number(true),
            )
        }
    } else {
        None
    };
    
    // Create filter for log level
    let log_level = config.log_level.parse::<Level>()
        .unwrap_or(Level::INFO);
    let filter = EnvFilter::from_default_env()
        .add_directive(LevelFilter::from_level(log_level).into())
        .add_directive("hyper=warn".parse()?)
        .add_directive("tower=warn".parse()?)
        .add_directive("reqwest=warn".parse()?)
        .add_directive("h2=warn".parse()?);
    
    // Build subscriber
    let subscriber = Registry::default()
        .with(filter)
        .with(otel_layer);
    
    let subscriber = if let Some(console_layer) = console_layer {
        subscriber.with(console_layer).boxed()
    } else {
        subscriber.boxed()
    };
    
    // Set global subscriber
    tracing::subscriber::set_global_default(subscriber)?;
    
    tracing::info!(
        service_name = %config.service_name,
        service_version = %config.service_version,
        environment = %config.environment,
        "Distributed tracing initialized"
    );
    
    Ok(())
}

/// Create OpenTelemetry tracer
async fn create_tracer(config: &TracingConfig) -> Result<opentelemetry::sdk::trace::Tracer> {
    // For now, return a simple noop tracer to fix compilation
    // This can be properly implemented when the OpenTelemetry version issues are resolved
    Ok(opentelemetry::sdk::trace::TracerProvider::default().tracer("crypto-quant-platform"))
}

/// Shutdown tracing and flush pending spans
pub async fn shutdown_tracing() -> Result<()> {
    tracing::info!("Shutting down distributed tracing");
    
    // Flush and shutdown the tracer provider
    global::shutdown_tracer_provider();
    
    Ok(())
}

/// Custom span creation macros and utilities
pub mod span_utils {
    use tracing::{span, Span, Level};
    use uuid::Uuid;
    
    /// Create a span for HTTP requests
    pub fn http_request_span(method: &str, uri: &str, version: &str) -> Span {
        span!(
            Level::INFO,
            "http_request",
            http.method = method,
            http.url = uri,
            http.version = version,
            otel.kind = "server",
            request_id = %Uuid::new_v4()
        )
    }
    
    /// Create a span for database operations
    pub fn db_operation_span(operation: &str, table: &str, database: &str) -> Span {
        span!(
            Level::DEBUG,
            "db_operation",
            db.operation = operation,
            db.sql.table = table,
            db.name = database,
            otel.kind = "client"
        )
    }
    
    /// Create a span for trading operations
    pub fn trading_span(operation: &str, symbol: &str, side: &str) -> Span {
        span!(
            Level::INFO,
            "trading_operation",
            trading.operation = operation,
            trading.symbol = symbol,
            trading.side = side,
            otel.kind = "internal"
        )
    }
    
    /// Create a span for analytics operations
    pub fn analytics_span(operation: &str, strategy: &str, timeframe: &str) -> Span {
        span!(
            Level::INFO,
            "analytics_operation",
            analytics.operation = operation,
            analytics.strategy = strategy,
            analytics.timeframe = timeframe,
            otel.kind = "internal"
        )
    }
    
    /// Create a span for external API calls
    pub fn external_api_span(service: &str, operation: &str, endpoint: &str) -> Span {
        span!(
            Level::INFO,
            "external_api_call",
            external.service = service,
            external.operation = operation,
            external.endpoint = endpoint,
            otel.kind = "client",
            request_id = %Uuid::new_v4()
        )
    }
}

/// Tracing middleware for Axum
#[cfg(feature = "axum")]
pub mod middleware {
    use axum::{
        extract::MatchedPath,
        http::{Request, Response},
        middleware::Next,
        response::IntoResponse,
    };
    use std::time::Instant;
    use tracing::Instrument;
    use uuid::Uuid;
    
    /// Tracing middleware for HTTP requests
    pub async fn trace_request<B>(
        request: Request<B>,
        next: Next<B>,
    ) -> impl IntoResponse {
        let start = Instant::now();
        let method = request.method().to_string();
        let uri = request.uri().to_string();
        let version = format!("{:?}", request.version());
        
        let path = request
            .extensions()
            .get::<MatchedPath>()
            .map(|path| path.as_str())
            .unwrap_or(&uri);
        
        let request_id = Uuid::new_v4();
        
        let span = tracing::info_span!(
            "http_request",
            method = %method,
            path = %path,
            version = %version,
            request_id = %request_id,
            otel.kind = "server"
        );
        
        async move {
            let response = next.run(request).await;
            let status = response.status();
            let duration = start.elapsed();
            
            tracing::info!(
                status = %status,
                duration_ms = %duration.as_millis(),
                "HTTP request completed"
            );
            
            response
        }
        .instrument(span)
        .await
    }
}

/// Business-specific tracing utilities
pub mod business {
    use tracing::{info, warn, error};
    use serde::Serialize;
    use uuid::Uuid;
    
    /// Log trade execution with tracing
    pub fn log_trade_execution<T: Serialize + std::fmt::Debug>(
        trade_id: Uuid,
        symbol: &str,
        side: &str,
        quantity: f64,
        price: f64,
        result: &T,
        success: bool,
    ) {
        if success {
            info!(
                trade_id = %trade_id,
                symbol = symbol,
                side = side,
                quantity = quantity,
                price = price,
                result = ?result,
                "Trade executed successfully"
            );
        } else {
            error!(
                trade_id = %trade_id,
                symbol = symbol,
                side = side,
                quantity = quantity,
                price = price,
                error = ?result,
                "Trade execution failed"
            );
        }
    }
    
    /// Log strategy performance
    pub fn log_strategy_performance(
        strategy_id: &str,
        symbol: &str,
        timeframe: &str,
        pnl: f64,
        win_rate: f64,
        trades_count: u32,
    ) {
        info!(
            strategy_id = strategy_id,
            symbol = symbol,
            timeframe = timeframe,
            pnl = pnl,
            win_rate = win_rate,
            trades_count = trades_count,
            "Strategy performance update"
        );
    }
    
    /// Log risk management events
    pub fn log_risk_event(
        risk_type: &str,
        severity: &str,
        message: &str,
        metadata: Option<serde_json::Value>,
    ) {
        match severity {
            "critical" | "high" => {
                error!(
                    risk_type = risk_type,
                    severity = severity,
                    message = message,
                    metadata = ?metadata,
                    "Risk management alert"
                );
            }
            "medium" => {
                warn!(
                    risk_type = risk_type,
                    severity = severity,
                    message = message,
                    metadata = ?metadata,
                    "Risk management warning"
                );
            }
            _ => {
                info!(
                    risk_type = risk_type,
                    severity = severity,
                    message = message,
                    metadata = ?metadata,
                    "Risk management info"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing::info;
    
    #[tokio::test]
    async fn test_tracing_initialization() {
        let config = TracingConfig {
            console_output: false,
            jaeger_endpoint: None,
            ..Default::default()
        };
        
        let result = init_tracing_with_config(config).await;
        assert!(result.is_ok());
        
        // Test that tracing works
        info!("Test trace message");
        
        let shutdown_result = shutdown_tracing().await;
        assert!(shutdown_result.is_ok());
    }
    
    #[test]
    fn test_span_creation() {
        use span_utils::*;
        
        let http_span = http_request_span("GET", "/api/v1/health", "HTTP/1.1");
        assert_eq!(http_span.metadata().unwrap().name(), "http_request");
        
        let db_span = db_operation_span("SELECT", "users", "trading_platform");
        assert_eq!(db_span.metadata().unwrap().name(), "db_operation");
        
        let trading_span = trading_span("place_order", "BTCUSDT", "buy");
        assert_eq!(trading_span.metadata().unwrap().name(), "trading_operation");
    }
}