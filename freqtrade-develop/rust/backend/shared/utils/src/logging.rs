use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::{Layer, Registry};

pub struct StructuredLogger {
    service_name: String,
    version: String,
    environment: String,
}

impl StructuredLogger {
    pub fn new(service_name: String, version: String, environment: String) -> Self {
        Self {
            service_name,
            version,
            environment,
        }
    }

    pub fn init(self) -> Result<(), Box<dyn std::error::Error>> {
        let logger_layer = StructuredLoggerLayer::new(self);
        
        let subscriber = Registry::default()
            .with(logger_layer)
            .with(tracing_subscriber::fmt::layer().json().flatten_event(true));

        tracing::subscriber::set_global_default(subscriber)?;
        Ok(())
    }

    pub fn format_log(&self, level: &str, message: &str, fields: Option<HashMap<String, Value>>) -> Value {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut log_entry = json!({
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "service": self.service_name,
            "version": self.version,
            "environment": self.environment,
            "host": gethostname::gethostname().to_string_lossy(),
        });

        if let Some(fields) = fields {
            if let Value::Object(ref mut map) = log_entry {
                for (key, value) in fields {
                    map.insert(key, value);
                }
            }
        }

        log_entry
    }
}

pub struct StructuredLoggerLayer {
    logger: StructuredLogger,
}

impl StructuredLoggerLayer {
    pub fn new(logger: StructuredLogger) -> Self {
        Self { logger }
    }
}

impl<S> Layer<S> for StructuredLoggerLayer
where
    S: Subscriber,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let metadata = event.metadata();
        let mut fields = HashMap::new();
        let mut message = String::new();

        let mut visitor = FieldVisitor {
            fields: &mut fields,
            message: &mut message,
        };

        event.record(&mut visitor);

        if message.is_empty() {
            message = "No message".to_string();
        }

        // Add tracing metadata
        fields.insert("target".to_string(), json!(metadata.target()));
        fields.insert("file".to_string(), json!(metadata.file().unwrap_or("unknown")));
        fields.insert("line".to_string(), json!(metadata.line().unwrap_or(0)));

        let log_entry = self.logger.format_log(
            &metadata.level().to_string(),
            &message,
            Some(fields),
        );

        println!("{}", log_entry);
    }
}

struct FieldVisitor<'a> {
    fields: &'a mut HashMap<String, Value>,
    message: &'a mut String,
}

impl tracing::field::Visit for FieldVisitor<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        let field_name = field.name();
        let field_value = format!("{:?}", value);

        if field_name == "message" {
            *self.message = field_value.trim_matches('"').to_string();
        } else {
            self.fields.insert(field_name.to_string(), json!(field_value));
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        let field_name = field.name();

        if field_name == "message" {
            *self.message = value.to_string();
        } else {
            self.fields.insert(field_name.to_string(), json!(value));
        }
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        let field_name = field.name();
        self.fields.insert(field_name.to_string(), json!(value));
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        let field_name = field.name();
        self.fields.insert(field_name.to_string(), json!(value));
    }

    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        let field_name = field.name();
        self.fields.insert(field_name.to_string(), json!(value));
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        let field_name = field.name();
        self.fields.insert(field_name.to_string(), json!(value));
    }
}

// Audit logging for trading operations
pub struct AuditLogger {
    logger: StructuredLogger,
}

impl AuditLogger {
    pub fn new(service_name: String, version: String, environment: String) -> Self {
        Self {
            logger: StructuredLogger::new(service_name, version, environment),
        }
    }

    pub fn log_trade_signal(&self, signal_id: &str, symbol: &str, signal_type: &str, strength: f64) {
        let fields = HashMap::from([
            ("event_type".to_string(), json!("trade_signal")),
            ("signal_id".to_string(), json!(signal_id)),
            ("symbol".to_string(), json!(symbol)),
            ("signal_type".to_string(), json!(signal_type)),
            ("strength".to_string(), json!(strength)),
        ]);

        let log_entry = self.logger.format_log(
            "INFO",
            &format!("Trade signal generated: {} {} {}", symbol, signal_type, strength),
            Some(fields),
        );

        println!("{}", log_entry);
    }

    pub fn log_order_placed(&self, order_id: &str, symbol: &str, side: &str, quantity: f64, price: Option<f64>) {
        let mut fields = HashMap::from([
            ("event_type".to_string(), json!("order_placed")),
            ("order_id".to_string(), json!(order_id)),
            ("symbol".to_string(), json!(symbol)),
            ("side".to_string(), json!(side)),
            ("quantity".to_string(), json!(quantity)),
        ]);

        if let Some(price) = price {
            fields.insert("price".to_string(), json!(price));
        }

        let log_entry = self.logger.format_log(
            "INFO",
            &format!("Order placed: {} {} {} @ {:?}", symbol, side, quantity, price),
            Some(fields),
        );

        println!("{}", log_entry);
    }

    pub fn log_order_executed(&self, order_id: &str, execution_id: &str, symbol: &str, quantity: f64, price: f64) {
        let fields = HashMap::from([
            ("event_type".to_string(), json!("order_executed")),
            ("order_id".to_string(), json!(order_id)),
            ("execution_id".to_string(), json!(execution_id)),
            ("symbol".to_string(), json!(symbol)),
            ("quantity".to_string(), json!(quantity)),
            ("price".to_string(), json!(price)),
        ]);

        let log_entry = self.logger.format_log(
            "INFO",
            &format!("Order executed: {} {} @ {}", symbol, quantity, price),
            Some(fields),
        );

        println!("{}", log_entry);
    }

    pub fn log_risk_event(&self, event_id: &str, event_type: &str, severity: &str, description: &str) {
        let fields = HashMap::from([
            ("event_type".to_string(), json!("risk_event")),
            ("event_id".to_string(), json!(event_id)),
            ("risk_event_type".to_string(), json!(event_type)),
            ("severity".to_string(), json!(severity)),
            ("description".to_string(), json!(description)),
        ]);

        let log_entry = self.logger.format_log(
            "WARN",
            &format!("Risk event: {} - {}", event_type, description),
            Some(fields),
        );

        println!("{}", log_entry);
    }

    pub fn log_system_error(&self, component: &str, error: &str, context: Option<HashMap<String, Value>>) {
        let mut fields = HashMap::from([
            ("event_type".to_string(), json!("system_error")),
            ("component".to_string(), json!(component)),
            ("error".to_string(), json!(error)),
        ]);

        if let Some(context) = context {
            fields.extend(context);
        }

        let log_entry = self.logger.format_log(
            "ERROR",
            &format!("System error in {}: {}", component, error),
            Some(fields),
        );

        println!("{}", log_entry);
    }
}

// Log aggregation utilities
pub struct LogAggregator {
    _service_name: String,
    buffer: Vec<Value>,
    max_buffer_size: usize,
}

impl LogAggregator {
    pub fn new(service_name: String, max_buffer_size: usize) -> Self {
        Self {
            _service_name: service_name,
            buffer: Vec::new(),
            max_buffer_size,
        }
    }

    pub fn add_log(&mut self, log_entry: Value) {
        self.buffer.push(log_entry);

        if self.buffer.len() >= self.max_buffer_size {
            self.flush();
        }
    }

    pub fn flush(&mut self) {
        if !self.buffer.is_empty() {
            // In production, this would send logs to a centralized logging system
            // like ELK stack, Fluentd, or cloud logging services
            for log in &self.buffer {
                println!("{}", log);
            }
            self.buffer.clear();
        }
    }

    pub fn get_logs(&self) -> &[Value] {
        &self.buffer
    }
}

// Convenience macros for structured logging
#[macro_export]
macro_rules! log_info {
    ($message:expr) => {
        tracing::info!($message);
    };
    ($message:expr, $($field:tt)*) => {
        tracing::info!($message, $($field)*);
    };
}

#[macro_export]
macro_rules! log_warn {
    ($message:expr) => {
        tracing::warn!($message);
    };
    ($message:expr, $($field:tt)*) => {
        tracing::warn!($message, $($field)*);
    };
}

#[macro_export]
macro_rules! log_error {
    ($message:expr) => {
        tracing::error!($message);
    };
    ($message:expr, $($field:tt)*) => {
        tracing::error!($message, $($field)*);
    };
}