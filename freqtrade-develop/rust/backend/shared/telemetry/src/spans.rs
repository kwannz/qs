use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{Level, Span};
use uuid::Uuid;

/// Span context for distributed tracing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanContext {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub baggage: HashMap<String, String>,
}

impl SpanContext {
    pub fn new() -> Self {
        Self {
            trace_id: Uuid::new_v4().to_string(),
            span_id: Uuid::new_v4().to_string(),
            parent_span_id: None,
            baggage: HashMap::new(),
        }
    }
    
    pub fn child_of(parent: &SpanContext) -> Self {
        Self {
            trace_id: parent.trace_id.clone(),
            span_id: Uuid::new_v4().to_string(),
            parent_span_id: Some(parent.span_id.clone()),
            baggage: parent.baggage.clone(),
        }
    }
    
    pub fn with_baggage(mut self, key: String, value: String) -> Self {
        self.baggage.insert(key, value);
        self
    }
}

impl Default for SpanContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Span builder for creating business-specific spans
pub struct SpanBuilder {
    name: String,
    level: Level,
    fields: HashMap<String, String>,
}

impl SpanBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            level: Level::INFO,
            fields: HashMap::new(),
        }
    }
    
    pub fn with_level(mut self, level: Level) -> Self {
        self.level = level;
        self
    }
    
    pub fn with_field(mut self, key: &str, value: &str) -> Self {
        self.fields.insert(key.to_string(), value.to_string());
        self
    }
    
    pub fn build(self) -> Span {
        use tracing::Level;
        match self.level {
            Level::ERROR => tracing::error_span!("business_operation", operation = %self.name),
            Level::WARN => tracing::warn_span!("business_operation", operation = %self.name),
            Level::INFO => tracing::info_span!("business_operation", operation = %self.name),
            Level::DEBUG => tracing::debug_span!("business_operation", operation = %self.name),
            Level::TRACE => tracing::trace_span!("business_operation", operation = %self.name),
        }
    }
}

/// Trading operation span utilities
pub mod trading {
    use super::*;
    use tracing::{info_span, debug_span};
    
    /// Create span for order placement
    pub fn order_placement_span(
        order_id: &str,
        symbol: &str,
        side: &str,
        order_type: &str,
        quantity: f64,
        price: Option<f64>,
    ) -> Span {
        info_span!(
            "place_order",
            order_id = order_id,
            symbol = symbol,
            side = side,
            order_type = order_type,
            quantity = quantity,
            price = price,
            otel.kind = "internal",
            operation = "place_order"
        )
    }
    
    /// Create span for order execution
    pub fn order_execution_span(
        order_id: &str,
        execution_id: &str,
        symbol: &str,
        executed_quantity: f64,
        execution_price: f64,
    ) -> Span {
        info_span!(
            "execute_order",
            order_id = order_id,
            execution_id = execution_id,
            symbol = symbol,
            executed_quantity = executed_quantity,
            execution_price = execution_price,
            otel.kind = "internal",
            operation = "execute_order"
        )
    }
    
    /// Create span for portfolio operations
    pub fn portfolio_operation_span(
        operation: &str,
        portfolio_id: &str,
        account_id: &str,
    ) -> Span {
        info_span!(
            "portfolio_operation",
            operation = operation,
            portfolio_id = portfolio_id,
            account_id = account_id,
            otel.kind = "internal"
        )
    }
    
    /// Create span for risk checks
    pub fn risk_check_span(
        check_type: &str,
        symbol: &str,
        account_id: &str,
        risk_level: &str,
    ) -> Span {
        debug_span!(
            "risk_check",
            check_type = check_type,
            symbol = symbol,
            account_id = account_id,
            risk_level = risk_level,
            otel.kind = "internal",
            operation = "risk_check"
        )
    }
    
    /// Create span for position management
    pub fn position_management_span(
        operation: &str,
        symbol: &str,
        position_id: &str,
        current_quantity: f64,
    ) -> Span {
        debug_span!(
            "position_management",
            operation = operation,
            symbol = symbol,
            position_id = position_id,
            current_quantity = current_quantity,
            otel.kind = "internal"
        )
    }
}

/// Analytics operation span utilities
pub mod analytics {
    use super::*;
    use tracing::{info_span, debug_span};
    
    /// Create span for strategy execution
    pub fn strategy_execution_span(
        strategy_id: &str,
        strategy_type: &str,
        symbols: &[String],
        timeframe: &str,
    ) -> Span {
        info_span!(
            "execute_strategy",
            strategy_id = strategy_id,
            strategy_type = strategy_type,
            symbols = ?symbols,
            timeframe = timeframe,
            otel.kind = "internal",
            operation = "execute_strategy"
        )
    }
    
    /// Create span for factor calculation
    pub fn factor_calculation_span(
        factor_name: &str,
        symbol: &str,
        timeframe: &str,
        calculation_method: &str,
    ) -> Span {
        debug_span!(
            "calculate_factor",
            factor_name = factor_name,
            symbol = symbol,
            timeframe = timeframe,
            calculation_method = calculation_method,
            otel.kind = "internal",
            operation = "calculate_factor"
        )
    }
    
    /// Create span for signal generation
    pub fn signal_generation_span(
        signal_type: &str,
        strategy_id: &str,
        symbol: &str,
        confidence: f64,
    ) -> Span {
        info_span!(
            "generate_signal",
            signal_type = signal_type,
            strategy_id = strategy_id,
            symbol = symbol,
            confidence = confidence,
            otel.kind = "internal",
            operation = "generate_signal"
        )
    }
    
    /// Create span for backtesting
    pub fn backtest_span(
        strategy_id: &str,
        start_date: &str,
        end_date: &str,
        symbols: &[String],
    ) -> Span {
        info_span!(
            "backtest_strategy",
            strategy_id = strategy_id,
            start_date = start_date,
            end_date = end_date,
            symbols = ?symbols,
            otel.kind = "internal",
            operation = "backtest_strategy"
        )
    }
}

/// Data processing span utilities
pub mod data {
    use super::*;
    use tracing::{debug_span, trace_span};
    
    /// Create span for market data processing
    pub fn market_data_processing_span(
        data_type: &str,
        symbol: &str,
        timestamp: u64,
        source: &str,
    ) -> Span {
        trace_span!(
            "process_market_data",
            data_type = data_type,
            symbol = symbol,
            timestamp = timestamp,
            source = source,
            otel.kind = "internal",
            operation = "process_market_data"
        )
    }
    
    /// Create span for data normalization
    pub fn data_normalization_span(
        data_type: &str,
        source_format: &str,
        target_format: &str,
        record_count: u64,
    ) -> Span {
        debug_span!(
            "normalize_data",
            data_type = data_type,
            source_format = source_format,
            target_format = target_format,
            record_count = record_count,
            otel.kind = "internal",
            operation = "normalize_data"
        )
    }
    
    /// Create span for data validation
    pub fn data_validation_span(
        validation_type: &str,
        data_source: &str,
        record_count: u64,
    ) -> Span {
        debug_span!(
            "validate_data",
            validation_type = validation_type,
            data_source = data_source,
            record_count = record_count,
            otel.kind = "internal",
            operation = "validate_data"
        )
    }
}

/// External service span utilities
pub mod external {
    use super::*;
    use tracing::info_span;
    
    /// Create span for exchange API calls
    pub fn exchange_api_span(
        exchange: &str,
        operation: &str,
        endpoint: &str,
        method: &str,
    ) -> Span {
        info_span!(
            "exchange_api_call",
            exchange = exchange,
            operation = operation,
            endpoint = endpoint,
            method = method,
            otel.kind = "client",
            service_name = exchange,
            http_method = method,
            http_url = endpoint
        )
    }
    
    /// Create span for database operations
    pub fn database_operation_span(
        operation: &str,
        table: &str,
        database: &str,
        query_type: &str,
    ) -> Span {
        info_span!(
            "database_operation",
            operation = operation,
            table = table,
            database = database,
            query_type = query_type,
            otel.kind = "client",
            db_system = "postgresql",
            db_name = database,
            db_sql_table = table,
            db_operation = operation
        )
    }
    
    /// Create span for cache operations
    pub fn cache_operation_span(
        operation: &str,
        cache_key: &str,
        cache_type: &str,
        ttl_seconds: Option<u64>,
    ) -> Span {
        info_span!(
            "cache_operation",
            operation = operation,
            cache_key = cache_key,
            cache_type = cache_type,
            ttl_seconds = ttl_seconds,
            otel.kind = "client",
            cache_system = cache_type
        )
    }
}

/// Business event tracking
pub mod events {
    use super::*;
    use tracing::{info, warn, error};
    use serde_json::Value;
    
    /// Track business event with structured data
    pub fn track_business_event(
        event_type: &str,
        event_category: &str,
        event_data: Value,
        user_id: Option<&str>,
        session_id: Option<&str>,
    ) {
        info!(
            event_type = event_type,
            event_category = event_category,
            event_data = ?event_data,
            user_id = user_id,
            session_id = session_id,
            timestamp = SystemTime::now().duration_since(UNIX_EPOCH)
                .unwrap_or_default().as_secs(),
            "Business event tracked"
        );
    }
    
    /// Track performance metrics
    pub fn track_performance_metric(
        metric_name: &str,
        value: f64,
        unit: &str,
        tags: HashMap<String, String>,
    ) {
        info!(
            metric_name = metric_name,
            value = value,
            unit = unit,
            tags = ?tags,
            timestamp = SystemTime::now().duration_since(UNIX_EPOCH)
                .unwrap_or_default().as_secs(),
            "Performance metric tracked"
        );
    }
    
    /// Track user action
    pub fn track_user_action(
        user_id: &str,
        action: &str,
        resource: &str,
        success: bool,
        metadata: Option<Value>,
    ) {
        let log_level = if success { "info" } else { "warn" };
        
        match log_level {
            "info" => info!(
                user_id = user_id,
                action = action,
                resource = resource,
                success = success,
                metadata = ?metadata,
                "User action tracked"
            ),
            _ => warn!(
                user_id = user_id,
                action = action,
                resource = resource,
                success = success,
                metadata = ?metadata,
                "User action failed"
            ),
        }
    }
    
    /// Track system error
    pub fn track_system_error(
        error_type: &str,
        error_message: &str,
        component: &str,
        severity: &str,
        metadata: Option<Value>,
    ) {
        match severity {
            "critical" | "high" => error!(
                error_type = error_type,
                error_message = error_message,
                component = component,
                severity = severity,
                metadata = ?metadata,
                "System error tracked"
            ),
            "medium" => warn!(
                error_type = error_type,
                error_message = error_message,
                component = component,
                severity = severity,
                metadata = ?metadata,
                "System warning tracked"
            ),
            _ => info!(
                error_type = error_type,
                error_message = error_message,
                component = component,
                severity = severity,
                metadata = ?metadata,
                "System info tracked"
            ),
        }
    }
}

/// Span timing utilities
pub struct SpanTimer {
    start_time: Instant,
    span_name: String,
}

impl SpanTimer {
    pub fn new(span_name: String) -> Self {
        Self {
            start_time: Instant::now(),
            span_name,
        }
    }
    
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    pub fn finish(self) -> Duration {
        let duration = self.elapsed();
        tracing::info!(
            span_name = %self.span_name,
            duration_ms = duration.as_millis(),
            "Span completed"
        );
        duration
    }
}

/// Macro for easy span timing
#[macro_export]
macro_rules! time_span {
    ($name:expr, $block:expr) => {{
        let _timer = $crate::spans::SpanTimer::new($name.to_string());
        let result = $block;
        _timer.finish();
        result
    }};
}

/// Macro for creating business spans with context
#[macro_export]
macro_rules! business_span {
    ($span_type:expr, $operation:expr, { $($key:expr => $value:expr),* }) => {
        tracing::info_span!(
            $span_type,
            operation = $operation,
            otel.kind = "internal",
            $(
                $key = $value,
            )*
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing::Instrument;
    
    #[test]
    fn test_span_context() {
        let parent = SpanContext::new();
        let child = SpanContext::child_of(&parent);
        
        assert_eq!(child.trace_id, parent.trace_id);
        assert_ne!(child.span_id, parent.span_id);
        assert_eq!(child.parent_span_id, Some(parent.span_id));
    }
    
    #[test]
    fn test_span_builder() {
        let _span = SpanBuilder::new("test_operation")
            .with_level(Level::DEBUG)
            .with_field("test_key", "test_value")
            .build();
        
        // Just check that the span creation doesn't panic
        // The actual functionality is tested in the span_utilities test
    }
    
    #[tokio::test]
    async fn test_span_utilities() {
        // Test trading span
        let order_span = trading::order_placement_span(
            "order-123",
            "BTCUSDT",
            "buy",
            "market",
            1.0,
            Some(50000.0),
        );
        
        async {
            tracing::info!("Processing order");
        }.instrument(order_span).await;
        
        // Test analytics span
        let strategy_span = analytics::strategy_execution_span(
            "momentum-strategy-1",
            "momentum",
            &["BTCUSDT".to_string(), "ETHUSDT".to_string()],
            "1h",
        );
        
        async {
            tracing::info!("Executing strategy");
        }.instrument(strategy_span).await;
    }
    
    #[test]
    fn test_span_timer() {
        let timer = SpanTimer::new("test_operation".to_string());
        std::thread::sleep(std::time::Duration::from_millis(10));
        let duration = timer.finish();
        
        assert!(duration.as_millis() >= 10);
    }
}