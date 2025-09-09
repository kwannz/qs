use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct MetricValue {
    pub value: f64,
    pub timestamp: u64,
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

#[derive(Debug)]
pub struct Metric {
    pub name: String,
    pub metric_type: MetricType,
    pub help: String,
    pub values: Vec<MetricValue>,
}

pub struct MetricsCollector {
    metrics: Arc<RwLock<HashMap<String, Metric>>>,
    service_name: String,
}

impl MetricsCollector {
    pub fn new(service_name: String) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            service_name,
        }
    }

    pub async fn increment_counter(&self, name: &str, labels: HashMap<String, String>) {
        self.increment_counter_by(name, 1.0, labels).await;
    }

    pub async fn increment_counter_by(&self, name: &str, value: f64, labels: HashMap<String, String>) {
        let mut metrics = self.metrics.write().await;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let metric = metrics.entry(name.to_string()).or_insert_with(|| Metric {
            name: name.to_string(),
            metric_type: MetricType::Counter,
            help: format!("Counter metric: {}", name),
            values: Vec::new(),
        });

        // For counters, we add to existing value or create new
        if let Some(existing) = metric.values.iter_mut().find(|v| v.labels == labels) {
            existing.value += value;
            existing.timestamp = timestamp;
        } else {
            metric.values.push(MetricValue {
                value,
                timestamp,
                labels,
            });
        }
    }

    pub async fn set_gauge(&self, name: &str, value: f64, labels: HashMap<String, String>) {
        let mut metrics = self.metrics.write().await;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let metric = metrics.entry(name.to_string()).or_insert_with(|| Metric {
            name: name.to_string(),
            metric_type: MetricType::Gauge,
            help: format!("Gauge metric: {}", name),
            values: Vec::new(),
        });

        // For gauges, we set the value directly
        if let Some(existing) = metric.values.iter_mut().find(|v| v.labels == labels) {
            existing.value = value;
            existing.timestamp = timestamp;
        } else {
            metric.values.push(MetricValue {
                value,
                timestamp,
                labels,
            });
        }
    }

    pub async fn record_histogram(&self, name: &str, value: f64, labels: HashMap<String, String>) {
        let mut metrics = self.metrics.write().await;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let metric = metrics.entry(name.to_string()).or_insert_with(|| Metric {
            name: name.to_string(),
            metric_type: MetricType::Histogram,
            help: format!("Histogram metric: {}", name),
            values: Vec::new(),
        });

        metric.values.push(MetricValue {
            value,
            timestamp,
            labels,
        });

        // Keep only recent values for histograms (last 1000 values)
        if metric.values.len() > 1000 {
            metric.values.drain(0..(metric.values.len() - 1000));
        }
    }

    pub async fn get_metrics_text(&self) -> String {
        let metrics = self.metrics.read().await;
        let mut output = String::new();

        for (_, metric) in metrics.iter() {
            use std::fmt::Write;
            let _ = writeln!(output, "# HELP {} {}", metric.name, metric.help);
            output.push_str(&format!("# TYPE {} {:?}\n", metric.name, metric.metric_type).to_lowercase());

            for value in &metric.values {
                let labels_str = if value.labels.is_empty() {
                    String::new()
                } else {
                    let labels: Vec<String> = value.labels
                        .iter()
                        .map(|(k, v)| format!("{}=\"{}\"", k, v))
                        .collect();
                    format!("{{{}}}", labels.join(","))
                };

                let _ = writeln!(output, "{}{} {} {}", metric.name, labels_str, value.value, value.timestamp);
            }
        }

        output
    }

    pub async fn get_metrics_json(&self) -> Value {
        let metrics = self.metrics.read().await;
        let mut json_metrics = serde_json::Map::new();

        for (name, metric) in metrics.iter() {
            let mut metric_json = serde_json::Map::new();
            metric_json.insert("type".to_string(), serde_json::Value::String(format!("{:?}", metric.metric_type)));
            metric_json.insert("help".to_string(), serde_json::Value::String(metric.help.clone()));

            let values: Vec<Value> = metric.values.iter().map(|v| {
                let mut value_json = serde_json::Map::new();
                value_json.insert("value".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(v.value).unwrap()));
                value_json.insert("timestamp".to_string(), serde_json::Value::Number(v.timestamp.into()));
                value_json.insert("labels".to_string(), serde_json::to_value(&v.labels).unwrap());
                Value::Object(value_json)
            }).collect();

            metric_json.insert("values".to_string(), Value::Array(values));
            json_metrics.insert(name.clone(), Value::Object(metric_json));
        }

        json_metrics.insert("service".to_string(), Value::String(self.service_name.clone()));
        Value::Object(json_metrics)
    }
}

// Performance monitoring
pub struct PerformanceMonitor {
    start_time: Instant,
    collector: Arc<MetricsCollector>,
    operation: String,
    labels: HashMap<String, String>,
}

impl PerformanceMonitor {
    pub fn new(collector: Arc<MetricsCollector>, operation: String, labels: HashMap<String, String>) -> Self {
        Self {
            start_time: Instant::now(),
            collector,
            operation,
            labels,
        }
    }

    pub async fn finish(self) {
        let duration = self.start_time.elapsed();
        // Convert to f64 safely using saturating conversion to avoid precision loss
        #[allow(clippy::cast_precision_loss)]
        let duration_ms = duration.as_millis()
            .min(u128::from(u64::MAX))
            .min(1u128 << 52) // Cap at 2^52 to maintain f64 precision
            as f64;

        // Record histogram for operation duration
        self.collector
            .record_histogram(
                &format!("{}_duration_ms", self.operation),
                duration_ms,
                self.labels.clone(),
            )
            .await;

        // Increment operation counter
        self.collector
            .increment_counter(&format!("{}_total", self.operation), self.labels)
            .await;
    }
}

// Health check system
#[derive(Debug, Clone)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Debug, Clone)]
pub struct HealthCheck {
    pub name: String,
    pub status: HealthStatus,
    pub last_check: u64,
    pub message: String,
}

pub struct HealthMonitor {
    checks: Arc<RwLock<HashMap<String, HealthCheck>>>,
}

impl HealthMonitor {
    pub fn new() -> Self {
        Self {
            checks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register_check(&self, name: String, status: HealthStatus, message: String) {
        let mut checks = self.checks.write().await;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        checks.insert(name.clone(), HealthCheck {
            name,
            status,
            last_check: timestamp,
            message,
        });
    }

    pub async fn get_overall_health(&self) -> HealthStatus {
        let checks = self.checks.read().await;
        
        if checks.is_empty() {
            return HealthStatus::Healthy;
        }

        let mut has_unhealthy = false;
        let mut has_degraded = false;

        for check in checks.values() {
            match check.status {
                HealthStatus::Unhealthy => has_unhealthy = true,
                HealthStatus::Degraded => has_degraded = true,
                HealthStatus::Healthy => {}
            }
        }

        if has_unhealthy {
            HealthStatus::Unhealthy
        } else if has_degraded {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    }

    pub async fn get_health_report(&self) -> Value {
        let checks = self.checks.read().await;
        let overall = self.get_overall_health().await;

        let checks_json: Vec<Value> = checks.values().map(|check| {
            serde_json::json!({
                "name": check.name,
                "status": format!("{:?}", check.status),
                "last_check": check.last_check,
                "message": check.message
            })
        }).collect();

        serde_json::json!({
            "status": format!("{:?}", overall),
            "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            "checks": checks_json
        })
    }
}

// Convenience macros for metrics
#[macro_export]
macro_rules! monitor_performance {
    ($collector:expr, $operation:expr) => {
        monitor_performance!($collector, $operation, std::collections::HashMap::new())
    };
    ($collector:expr, $operation:expr, $labels:expr) => {
        $crate::monitoring::PerformanceMonitor::new($collector, $operation.to_string(), $labels)
    };
}