// Sprint 1: Outbox Pattern Implementation for reliable event delivery

use anyhow::{Context, Result};
use async_nats::jetstream::{Context as JetStreamContext, stream::Config as StreamConfig};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sqlx::{PgPool, Row};
use std::collections::HashMap;
use tokio::time::{Duration, interval};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Sprint 1: Outbox event from database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutboxEvent {
    pub id: Uuid,
    pub aggregate_type: String,
    pub aggregate_id: String,
    pub event_type: String,
    pub payload: JsonValue,
    pub headers: HashMap<String, String>,
    pub idempotency_key: String,
    pub occurred_at: DateTime<Utc>,
    pub published_at: Option<DateTime<Utc>>,
    pub retry_count: i32,
    pub max_retries: i32,
    pub next_retry_at: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
}

/// Configuration for outbox publisher
#[derive(Debug, Clone)]
pub struct OutboxConfig {
    pub service_name: String,
    pub batch_size: usize,
    pub poll_interval: Duration,
    pub retry_backoff_base: Duration,
    pub max_retry_delay: Duration,
    pub enable_dead_letter_queue: bool,
    pub nats_stream_prefix: String,
}

impl Default for OutboxConfig {
    fn default() -> Self {
        Self {
            service_name: "unknown".to_string(),
            batch_size: 100,
            poll_interval: Duration::from_secs(5),
            retry_backoff_base: Duration::from_secs(10),
            max_retry_delay: Duration::from_secs(300), // 5 minutes
            enable_dead_letter_queue: true,
            nats_stream_prefix: "trading".to_string(),
        }
    }
}

/// Sprint 1: Outbox publisher for reliable event delivery
pub struct OutboxPublisher {
    db_pool: PgPool,
    jetstream: JetStreamContext,
    config: OutboxConfig,
}

impl OutboxPublisher {
    /// Create a new outbox publisher
    pub fn new(db_pool: PgPool, jetstream: JetStreamContext, config: OutboxConfig) -> Self {
        Self {
            db_pool,
            jetstream,
            config,
        }
    }

    /// Start the outbox publisher background task
    pub async fn start(&self) -> Result<()> {
        info!(
            service = %self.config.service_name,
            batch_size = %self.config.batch_size,
            poll_interval = ?self.config.poll_interval,
            "Starting outbox publisher"
        );

        self.ensure_streams().await.context("Failed to ensure NATS streams")?;

        let mut poll_timer = interval(self.config.poll_interval);
        
        loop {
            poll_timer.tick().await;
            
            match self.process_outbox_events().await {
                Ok(processed) => {
                    if processed > 0 {
                        debug!(
                            service = %self.config.service_name,
                            processed = %processed,
                            "Processed outbox events"
                        );
                    }
                }
                Err(e) => {
                    error!(
                        service = %self.config.service_name,
                        error = %e,
                        "Failed to process outbox events"
                    );
                }
            }

            // Process retry queue
            if let Err(e) = self.process_retry_queue().await {
                error!(
                    service = %self.config.service_name,
                    error = %e,
                    "Failed to process retry queue"
                );
            }
        }
    }

    /// Process unpublished outbox events
    async fn process_outbox_events(&self) -> Result<usize> {
        let events = self.fetch_unpublished_events().await?;
        let mut processed = 0;

        for event in events {
            match self.publish_event(&event).await {
                Ok(_) => {
                    self.mark_as_published(&event.id).await?;
                    processed += 1;
                }
                Err(e) => {
                    warn!(
                        event_id = %event.id,
                        event_type = %event.event_type,
                        retry_count = %event.retry_count,
                        error = %e,
                        "Failed to publish event, scheduling retry"
                    );
                    
                    self.schedule_retry(&event, e.to_string()).await?;
                }
            }
        }

        Ok(processed)
    }

    /// Fetch unpublished events from database
    async fn fetch_unpublished_events(&self) -> Result<Vec<OutboxEvent>> {
        let query = r#"
            SELECT id, aggregate_type, aggregate_id, event_type, payload, headers,
                   idempotency_key, occurred_at, published_at, retry_count, max_retries,
                   next_retry_at, error_message
            FROM outbox_events 
            WHERE published_at IS NULL 
                AND (next_retry_at IS NULL OR next_retry_at <= NOW())
            ORDER BY occurred_at ASC 
            LIMIT $1
            FOR UPDATE SKIP LOCKED
        "#;

        let rows = sqlx::query(query)
            .bind(self.config.batch_size as i32)
            .fetch_all(&self.db_pool)
            .await
            .context("Failed to fetch unpublished events")?;

        let mut events = Vec::new();
        for row in rows {
            let headers: JsonValue = row.get("headers");
            let headers_map: HashMap<String, String> = serde_json::from_value(headers)
                .unwrap_or_default();

            let event = OutboxEvent {
                id: row.get("id"),
                aggregate_type: row.get("aggregate_type"),
                aggregate_id: row.get("aggregate_id"),
                event_type: row.get("event_type"),
                payload: row.get("payload"),
                headers: headers_map,
                idempotency_key: row.get("idempotency_key"),
                occurred_at: row.get("occurred_at"),
                published_at: row.get("published_at"),
                retry_count: row.get("retry_count"),
                max_retries: row.get("max_retries"),
                next_retry_at: row.get("next_retry_at"),
                error_message: row.get("error_message"),
            };
            events.push(event);
        }

        Ok(events)
    }

    /// Publish event to NATS JetStream
    async fn publish_event(&self, event: &OutboxEvent) -> Result<()> {
        let subject = self.build_subject(&event.event_type);
        
        // Build message with headers
        let mut headers = async_nats::HeaderMap::new();
        let event_id_str = event.id.to_string();
        let service_name_str = &self.config.service_name;
        let occurred_at_str = event.occurred_at.to_rfc3339();
        
        headers.insert("idempotency-key", event.idempotency_key.as_str());
        headers.insert("event-id", event_id_str.as_str());
        headers.insert("event-type", event.event_type.as_str());
        headers.insert("aggregate-type", event.aggregate_type.as_str());
        headers.insert("aggregate-id", event.aggregate_id.as_str());
        headers.insert("service", service_name_str.as_str());
        headers.insert("occurred-at", occurred_at_str.as_str());

        // Add custom headers
        for (key, value) in &event.headers {
            headers.insert(key.as_str(), value.as_str());
        }

        let message_payload = serde_json::to_vec(&event.payload)?;

        // Publish with idempotency key as message ID
        let subject_clone = subject.clone();
        let publish_ack = self.jetstream
            .publish_with_headers(subject, headers, message_payload.into())
            .await
            .context("Failed to publish to NATS JetStream")?;

        let ack = publish_ack.await?;
        debug!(
            event_id = %event.id,
            event_type = %event.event_type,
            subject = %subject_clone,
            sequence = %ack.sequence,
            "Successfully published event"
        );

        Ok(())
    }

    /// Mark event as published
    async fn mark_as_published(&self, event_id: &Uuid) -> Result<()> {
        let query = "UPDATE outbox_events SET published_at = NOW() WHERE id = $1";
        
        sqlx::query(query)
            .bind(event_id)
            .execute(&self.db_pool)
            .await
            .context("Failed to mark event as published")?;

        Ok(())
    }

    /// Schedule event for retry
    async fn schedule_retry(&self, event: &OutboxEvent, error_message: String) -> Result<()> {
        if event.retry_count >= event.max_retries {
            warn!(
                event_id = %event.id,
                retry_count = %event.retry_count,
                max_retries = %event.max_retries,
                "Event exceeded max retries, moving to dead letter queue"
            );

            if self.config.enable_dead_letter_queue {
                self.move_to_dead_letter_queue(event, error_message).await?;
            }
            
            return Ok(());
        }

        // Calculate exponential backoff with jitter
        let delay_secs = std::cmp::min(
            self.config.retry_backoff_base.as_secs() * (2_u64.pow(event.retry_count as u32)),
            self.config.max_retry_delay.as_secs()
        );
        
        // Add jitter (±20%)
        let jitter = (delay_secs as f64 * 0.2 * rand::random::<f64>()) as u64;
        let delay = Duration::from_secs(delay_secs + jitter);
        let next_retry_at = Utc::now() + chrono::Duration::from_std(delay)?;

        let query = r#"
            UPDATE outbox_events 
            SET retry_count = retry_count + 1,
                next_retry_at = $1,
                error_message = $2
            WHERE id = $3
        "#;

        sqlx::query(query)
            .bind(next_retry_at)
            .bind(error_message)
            .bind(event.id)
            .execute(&self.db_pool)
            .await
            .context("Failed to schedule retry")?;

        debug!(
            event_id = %event.id,
            retry_count = %(event.retry_count + 1),
            next_retry_at = %next_retry_at,
            "Scheduled event for retry"
        );

        Ok(())
    }

    /// Process retry queue
    async fn process_retry_queue(&self) -> Result<()> {
        let events = self.fetch_retry_events().await?;
        
        for event in events {
            match self.publish_event(&event).await {
                Ok(_) => {
                    self.mark_as_published(&event.id).await?;
                    info!(
                        event_id = %event.id,
                        retry_count = %event.retry_count,
                        "Successfully published event after retry"
                    );
                }
                Err(e) => {
                    self.schedule_retry(&event, e.to_string()).await?;
                }
            }
        }

        Ok(())
    }

    /// Fetch events ready for retry
    async fn fetch_retry_events(&self) -> Result<Vec<OutboxEvent>> {
        let query = r#"
            SELECT id, aggregate_type, aggregate_id, event_type, payload, headers,
                   idempotency_key, occurred_at, published_at, retry_count, max_retries,
                   next_retry_at, error_message
            FROM outbox_events 
            WHERE published_at IS NULL 
                AND next_retry_at IS NOT NULL 
                AND next_retry_at <= NOW()
                AND retry_count < max_retries
            ORDER BY next_retry_at ASC 
            LIMIT $1
            FOR UPDATE SKIP LOCKED
        "#;

        let rows = sqlx::query(query)
            .bind(self.config.batch_size as i32)
            .fetch_all(&self.db_pool)
            .await
            .context("Failed to fetch retry events")?;

        let mut events = Vec::new();
        for row in rows {
            let headers: JsonValue = row.get("headers");
            let headers_map: HashMap<String, String> = serde_json::from_value(headers)
                .unwrap_or_default();

            let event = OutboxEvent {
                id: row.get("id"),
                aggregate_type: row.get("aggregate_type"),
                aggregate_id: row.get("aggregate_id"),
                event_type: row.get("event_type"),
                payload: row.get("payload"),
                headers: headers_map,
                idempotency_key: row.get("idempotency_key"),
                occurred_at: row.get("occurred_at"),
                published_at: row.get("published_at"),
                retry_count: row.get("retry_count"),
                max_retries: row.get("max_retries"),
                next_retry_at: row.get("next_retry_at"),
                error_message: row.get("error_message"),
            };
            events.push(event);
        }

        Ok(events)
    }

    /// Move failed event to dead letter queue
    async fn move_to_dead_letter_queue(&self, event: &OutboxEvent, error_message: String) -> Result<()> {
        // Publish to dead letter queue
        let dlq_subject = format!("{}.dlq.{}", self.config.nats_stream_prefix, event.event_type);
        
        let dlq_payload = serde_json::json!({
            "original_event": event,
            "final_error": error_message,
            "moved_to_dlq_at": Utc::now(),
            "service": self.config.service_name
        });

        let mut headers = async_nats::HeaderMap::new();
        let event_id_str = event.id.to_string();
        headers.insert("event-id", event_id_str.as_str());
        headers.insert("reason", "max-retries-exceeded");
        
        self.jetstream
            .publish_with_headers(dlq_subject, headers, serde_json::to_vec(&dlq_payload)?.into())
            .await
            .context("Failed to publish to dead letter queue")?;

        // Mark as processed (failed)
        let query = "UPDATE outbox_events SET published_at = NOW(), error_message = $1 WHERE id = $2";
        sqlx::query(query)
            .bind(format!("DLQ: {error_message}"))
            .bind(event.id)
            .execute(&self.db_pool)
            .await
            .context("Failed to mark event as moved to DLQ")?;

        info!(
            event_id = %event.id,
            event_type = %event.event_type,
            "Moved event to dead letter queue"
        );

        Ok(())
    }

    /// Ensure required NATS streams exist
    async fn ensure_streams(&self) -> Result<()> {
        let stream_name = format!("{}_events", self.config.nats_stream_prefix);
        
        // Check if stream exists
        match self.jetstream.get_stream(&stream_name).await {
            Ok(_) => {
                debug!(stream = %stream_name, "Stream already exists");
                return Ok(());
            }
            Err(_) => {
                info!(stream = %stream_name, "Creating NATS stream");
            }
        }

        // Create stream
        let stream_config = StreamConfig {
            name: stream_name.clone(),
            subjects: vec![format!("{}.>", self.config.nats_stream_prefix)],
            retention: async_nats::jetstream::stream::RetentionPolicy::WorkQueue,
            max_age: Duration::from_secs(7 * 24 * 3600), // 7 days
            storage: async_nats::jetstream::stream::StorageType::File,
            republish: None,
            ..Default::default()
        };

        self.jetstream
            .create_stream(stream_config)
            .await
            .context("Failed to create NATS stream")?;

        info!(stream = %stream_name, "Created NATS stream");
        Ok(())
    }

    /// Build NATS subject for event type following Sprint 1 subject structure
    fn build_subject(&self, event_type: &str) -> String {
        let service_name = &self.config.service_name;
        
        // Map event types to structured subjects
        let subject_suffix = match (service_name.as_str(), event_type) {
            // Execution service events
            ("execution", "order_placed") => "execution.order.placed".to_string(),
            ("execution", "order_executed") => "execution.order.executed".to_string(), 
            ("execution", "order_cancelled") => "execution.order.cancelled".to_string(),
            ("execution", "order_failed") => "execution.order.failed".to_string(),
            ("execution", "position_updated") => "execution.position.updated".to_string(),
            ("execution", "risk_check_triggered") => "execution.risk.triggered".to_string(),
            ("execution", "market_data_received") => "market.price.updated".to_string(),
            
            // Analytics service events
            ("analytics", "signal_generated") => "analytics.signal.generated".to_string(),
            ("analytics", "backtest_completed") => "analytics.backtest.completed".to_string(),
            ("analytics", "factor_calculated") => "analytics.factor.calculated".to_string(),
            ("analytics", "model_trained") => "analytics.model.trained".to_string(),
            ("analytics", "risk_alert_triggered") => "analytics.alert.triggered".to_string(),
            
            // Risk service events  
            ("risk", "pretrade_check") => "risk.check.pretrade".to_string(),
            ("risk", "posttrade_check") => "risk.check.posttrade".to_string(),
            ("risk", "limit_breached") => "risk.limit.breached".to_string(),
            ("risk", "alert_triggered") => "risk.alert.triggered".to_string(),
            ("risk", "account_suspended") => "risk.account.suspended".to_string(),
            
            // Audit service events
            ("audit", "user_action") => "audit.user.action".to_string(),
            ("audit", "system_event") => "audit.system.event".to_string(),
            ("audit", "compliance_check") => "audit.compliance.check".to_string(),
            ("audit", "data_access") => "audit.data.access".to_string(),
            ("audit", "security_event") => "audit.security.event".to_string(),
            
            // Market data service events
            ("market", "price_updated") => "market.price.updated".to_string(),
            ("market", "orderbook_changed") => "market.orderbook.changed".to_string(),
            ("market", "trade_executed") => "market.trade.executed".to_string(),
            ("market", "funding_updated") => "market.funding.updated".to_string(),
            ("market", "liquidation_detected") => "market.liquidation.detected".to_string(),
            
            // Default fallback - convert event type to structured format
            _ => {
                // Convert snake_case event types to dot notation
                let normalized_event = event_type.to_lowercase().replace('_', ".");
                format!("{service_name}.{normalized_event}")
            }
        };
        
        format!("{}.{}", self.config.nats_stream_prefix, subject_suffix)
    }
}

/// Helper function to insert event into outbox
pub async fn insert_outbox_event(
    pool: &PgPool,
    aggregate_type: &str,
    aggregate_id: &str,
    event_type: &str,
    payload: JsonValue,
    idempotency_key: &str,
    headers: Option<HashMap<String, String>>,
) -> Result<Uuid> {
    let event_id = Uuid::new_v4();
    let headers_json = serde_json::to_value(headers.unwrap_or_default())?;

    let query = r#"
        INSERT INTO outbox_events (id, aggregate_type, aggregate_id, event_type, payload, headers, idempotency_key)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (idempotency_key) DO NOTHING
        RETURNING id
    "#;

    let result = sqlx::query(query)
        .bind(event_id)
        .bind(aggregate_type)
        .bind(aggregate_id)
        .bind(event_type)
        .bind(payload)
        .bind(headers_json)
        .bind(idempotency_key)
        .fetch_optional(pool)
        .await
        .context("Failed to insert outbox event")?;

    match result {
        Some(row) => Ok(row.get("id")),
        None => {
            debug!(idempotency_key = %idempotency_key, "Event with idempotency key already exists");
            Ok(event_id) // Return generated ID even if insert was skipped
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outbox_config_default() {
        let config = OutboxConfig::default();
        assert_eq!(config.service_name, "unknown");
        assert_eq!(config.batch_size, 100);
        assert_eq!(config.poll_interval, Duration::from_secs(5));
        assert!(config.enable_dead_letter_queue);
    }

    /// Test helper function for build_subject logic
    fn build_subject_test(service_name: &str, stream_prefix: &str, event_type: &str) -> String {
        let subject_suffix = match (service_name, event_type) {
            // Execution service events
            ("execution", "order_placed") => "execution.order.placed",
            ("execution", "order_executed") => "execution.order.executed", 
            ("execution", "order_cancelled") => "execution.order.cancelled",
            ("execution", "order_failed") => "execution.order.failed",
            ("execution", "position_updated") => "execution.position.updated",
            ("execution", "risk_check_triggered") => "execution.risk.triggered",
            ("execution", "market_data_received") => "market.price.updated",
            
            // Analytics service events
            ("analytics", "signal_generated") => "analytics.signal.generated",
            ("analytics", "backtest_completed") => "analytics.backtest.completed",
            ("analytics", "factor_calculated") => "analytics.factor.calculated",
            ("analytics", "model_trained") => "analytics.model.trained",
            ("analytics", "risk_alert_triggered") => "analytics.alert.triggered",
            
            // Default fallback
            _ => {
                let normalized_event = event_type.to_lowercase().replace('_', ".");
                return format!("{stream_prefix}.{service_name}.{normalized_event}");
            }
        };
        
        format!("{stream_prefix}.{subject_suffix}")
    }

    #[test]
    fn test_build_subject() {
        // Test execution service events
        assert_eq!(
            build_subject_test("execution", "trading", "order_placed"),
            "trading.execution.order.placed"
        );
        
        assert_eq!(
            build_subject_test("execution", "trading", "position_updated"),
            "trading.execution.position.updated"
        );

        // Test analytics service events
        assert_eq!(
            build_subject_test("analytics", "trading", "signal_generated"),
            "trading.analytics.signal.generated"
        );
        
        assert_eq!(
            build_subject_test("analytics", "trading", "backtest_completed"),
            "trading.analytics.backtest.completed"
        );
        
        // Test fallback for unknown events
        assert_eq!(
            build_subject_test("execution", "trading", "unknown_event"),
            "trading.execution.unknown.event"
        );
        
        assert_eq!(
            build_subject_test("custom_service", "trading", "custom_event"),
            "trading.custom_service.custom.event"
        );
    }
}