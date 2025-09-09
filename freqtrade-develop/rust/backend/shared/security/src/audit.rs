use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::mpsc;
use uuid::Uuid;

/// Audit event types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuditEventType {
    // Authentication events
    LoginSuccess,
    LoginFailure,
    Logout,
    TokenGenerated,
    TokenRevoked,
    ApiKeyCreated,
    ApiKeyRevoked,

    // Authorization events
    PermissionGranted,
    PermissionDenied,
    RoleAssigned,
    RoleRevoked,

    // Data access events
    DataRead,
    DataWrite,
    DataDelete,
    DataExport,
    ConfigurationChanged,

    // Trading events
    OrderPlaced,
    OrderCancelled,
    OrderExecuted,
    PositionOpened,
    PositionClosed,
    StrategyDeployed,
    StrategyModified,

    // Risk management events
    RiskLimitTriggered,
    EmergencyStopActivated,
    RiskOverride,

    // System events
    SystemStartup,
    SystemShutdown,
    ServiceStarted,
    ServiceStopped,
    ConfigurationLoaded,

    // Security events
    SecurityViolation,
    EncryptionKeyRotated,
    CertificateRenewed,
    SecurityScanCompleted,

    // Administrative events
    UserCreated,
    UserModified,
    UserDeleted,
    AdminAction,
}

/// Audit event severity levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuditSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Comprehensive audit event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub id: Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub event_type: AuditEventType,
    pub severity: AuditSeverity,
    
    // Actor information
    pub user_id: Option<Uuid>,
    pub username: Option<String>,
    pub session_id: Option<String>,
    pub api_key_id: Option<String>,
    pub client_ip: Option<String>,
    pub user_agent: Option<String>,
    
    // Resource information
    pub resource_type: Option<String>,
    pub resource_id: Option<String>,
    pub resource_name: Option<String>,
    
    // Event details
    pub description: String,
    pub details: HashMap<String, serde_json::Value>,
    
    // Context information
    pub service_name: String,
    pub component: String,
    pub correlation_id: Option<String>,
    
    // Compliance fields
    pub compliance_tags: Vec<String>,
    pub retention_period_days: Option<u32>,
    
    // Security fields
    pub risk_score: Option<u32>, // 0-100
    pub requires_review: bool,
    pub data_classification: Option<String>, // public, internal, confidential, restricted
    
    // Performance fields
    pub duration_ms: Option<u64>,
    pub bytes_processed: Option<u64>,
    
    // Result information
    pub success: bool,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
}

impl AuditEvent {
    /// Create a new audit event builder
    pub fn builder() -> AuditEventBuilder {
        AuditEventBuilder::new()
    }

    /// Create a login success event
    pub fn login_success(
        user_id: Uuid,
        username: &str,
        session_id: &str,
        client_ip: Option<String>,
    ) -> Self {
        Self::builder()
            .event_type(AuditEventType::LoginSuccess)
            .severity(AuditSeverity::Info)
            .user_id(user_id)
            .username(username)
            .session_id(session_id)
            .client_ip(client_ip)
            .description(format!("User {} logged in successfully", username))
            .success(true)
            .build()
    }

    /// Create a login failure event
    pub fn login_failure(
        username: &str,
        reason: &str,
        client_ip: Option<String>,
    ) -> Self {
        Self::builder()
            .event_type(AuditEventType::LoginFailure)
            .severity(AuditSeverity::Warning)
            .username(username)
            .client_ip(client_ip)
            .description(format!("Login failed for user {}: {}", username, reason))
            .success(false)
            .error_message(reason)
            .risk_score(30) // Failed logins are moderately risky
            .requires_review(true)
            .build()
    }

    /// Create a data access event
    pub fn data_access(
        user_id: Uuid,
        resource_type: &str,
        resource_id: &str,
        operation: &str,
        success: bool,
    ) -> Self {
        let event_type = match operation {
            "write" => AuditEventType::DataWrite,
            "delete" => AuditEventType::DataDelete,
            "export" => AuditEventType::DataExport,
            _ => AuditEventType::DataRead, // Default for "read" and unknown operations
        };

        Self::builder()
            .event_type(event_type)
            .severity(if success { AuditSeverity::Info } else { AuditSeverity::Warning })
            .user_id(user_id)
            .resource_type(resource_type)
            .resource_id(resource_id)
            .description(format!("Data {} operation on {} {}", operation, resource_type, resource_id))
            .success(success)
            .compliance_tags(vec!["data-access".to_string()])
            .build()
    }

    /// Create a trading event
    pub fn trading_event(
        user_id: Uuid,
        event_type: AuditEventType,
        order_id: &str,
        symbol: &str,
        details: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self::builder()
            .event_type(event_type)
            .severity(AuditSeverity::Info)
            .user_id(user_id)
            .resource_type("order")
            .resource_id(order_id)
            .description(format!("Trading operation for {}", symbol))
            .details(details)
            .success(true)
            .compliance_tags(vec!["trading".to_string(), "financial".to_string()])
            .retention_period_days(2555) // 7 years for financial records
            .build()
    }

    /// Create a security violation event
    pub fn security_violation(
        user_id: Option<Uuid>,
        violation_type: &str,
        description: &str,
        client_ip: Option<String>,
    ) -> Self {
        Self::builder()
            .event_type(AuditEventType::SecurityViolation)
            .severity(AuditSeverity::Critical)
            .user_id_opt(user_id)
            .client_ip(client_ip)
            .description(description.to_string())
            .risk_score(90) // Security violations are high risk
            .requires_review(true)
            .compliance_tags(vec!["security".to_string()])
            .detail("violation_type", violation_type)
            .success(false)
            .build()
    }
}

/// Builder pattern for creating audit events
pub struct AuditEventBuilder {
    event: AuditEvent,
}

impl AuditEventBuilder {
    pub fn new() -> Self {
        Self {
            event: AuditEvent {
                id: Uuid::new_v4(),
                timestamp: chrono::Utc::now(),
                event_type: AuditEventType::SystemStartup,
                severity: AuditSeverity::Info,
                user_id: None,
                username: None,
                session_id: None,
                api_key_id: None,
                client_ip: None,
                user_agent: None,
                resource_type: None,
                resource_id: None,
                resource_name: None,
                description: String::new(),
                details: HashMap::new(),
                service_name: "platform".to_string(),
                component: "unknown".to_string(),
                correlation_id: None,
                compliance_tags: Vec::new(),
                retention_period_days: None,
                risk_score: None,
                requires_review: false,
                data_classification: None,
                duration_ms: None,
                bytes_processed: None,
                success: true,
                error_code: None,
                error_message: None,
            },
        }
    }

    #[must_use]
    pub fn event_type(mut self, event_type: AuditEventType) -> Self {
        self.event.event_type = event_type;
        self
    }

    #[must_use]
    pub fn severity(mut self, severity: AuditSeverity) -> Self {
        self.event.severity = severity;
        self
    }

    #[must_use]
    pub fn user_id(mut self, user_id: Uuid) -> Self {
        self.event.user_id = Some(user_id);
        self
    }

    #[must_use]
    pub fn user_id_opt(mut self, user_id: Option<Uuid>) -> Self {
        self.event.user_id = user_id;
        self
    }

    #[must_use]
    pub fn username(mut self, username: &str) -> Self {
        self.event.username = Some(username.to_string());
        self
    }

    #[must_use]
    pub fn session_id(mut self, session_id: &str) -> Self {
        self.event.session_id = Some(session_id.to_string());
        self
    }

    #[must_use]
    pub fn client_ip(mut self, client_ip: Option<String>) -> Self {
        self.event.client_ip = client_ip;
        self
    }

    #[must_use]
    pub fn resource_type(mut self, resource_type: &str) -> Self {
        self.event.resource_type = Some(resource_type.to_string());
        self
    }

    #[must_use]
    pub fn resource_id(mut self, resource_id: &str) -> Self {
        self.event.resource_id = Some(resource_id.to_string());
        self
    }

    #[must_use]
    pub fn description(mut self, description: String) -> Self {
        self.event.description = description;
        self
    }

    #[must_use]
    pub fn detail<T: Serialize>(mut self, key: &str, value: T) -> Self {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.event.details.insert(key.to_string(), json_value);
        }
        self
    }

    #[must_use]
    pub fn details(mut self, details: HashMap<String, serde_json::Value>) -> Self {
        self.event.details = details;
        self
    }

    #[must_use]
    pub fn service_name(mut self, service_name: &str) -> Self {
        self.event.service_name = service_name.to_string();
        self
    }

    #[must_use]
    pub fn component(mut self, component: &str) -> Self {
        self.event.component = component.to_string();
        self
    }

    #[must_use]
    pub fn compliance_tags(mut self, tags: Vec<String>) -> Self {
        self.event.compliance_tags = tags;
        self
    }

    #[must_use]
    pub fn risk_score(mut self, score: u32) -> Self {
        self.event.risk_score = Some(score.min(100));
        self
    }

    #[must_use]
    pub fn requires_review(mut self, requires_review: bool) -> Self {
        self.event.requires_review = requires_review;
        self
    }

    #[must_use]
    pub fn retention_period_days(mut self, days: u32) -> Self {
        self.event.retention_period_days = Some(days);
        self
    }

    #[must_use]
    pub fn success(mut self, success: bool) -> Self {
        self.event.success = success;
        self
    }

    #[must_use]
    pub fn error_message(mut self, message: &str) -> Self {
        self.event.error_message = Some(message.to_string());
        self
    }

    #[must_use]
    pub fn duration_ms(mut self, duration: u64) -> Self {
        self.event.duration_ms = Some(duration);
        self
    }

    pub fn build(self) -> AuditEvent {
        self.event
    }
}

impl Default for AuditEventBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Audit logger interface
#[async_trait::async_trait]
pub trait AuditLogger: Send + Sync {
    async fn log_event(&self, event: AuditEvent) -> Result<()>;
    async fn query_events(
        &self,
        filter: AuditEventFilter,
    ) -> Result<Vec<AuditEvent>>;
}

/// Audit event filter for queries
#[derive(Debug, Clone, Default)]
pub struct AuditEventFilter {
    pub user_id: Option<Uuid>,
    pub event_types: Vec<AuditEventType>,
    pub severity: Option<AuditSeverity>,
    pub start_time: Option<chrono::DateTime<chrono::Utc>>,
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    pub resource_type: Option<String>,
    pub success_only: Option<bool>,
    pub requires_review: Option<bool>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// In-memory audit logger for development/testing
pub struct InMemoryAuditLogger {
    events: tokio::sync::RwLock<Vec<AuditEvent>>,
}

impl InMemoryAuditLogger {
    pub fn new() -> Self {
        Self {
            events: tokio::sync::RwLock::new(Vec::new()),
        }
    }
}

#[async_trait::async_trait]
impl AuditLogger for InMemoryAuditLogger {
    async fn log_event(&self, event: AuditEvent) -> Result<()> {
        let mut events = self.events.write().await;
        events.push(event);
        Ok(())
    }

    async fn query_events(&self, filter: AuditEventFilter) -> Result<Vec<AuditEvent>> {
        let events = self.events.read().await;
        let mut filtered_events: Vec<AuditEvent> = events
            .iter()
            .filter(|event| {
                // Apply filters
                if let Some(user_id) = filter.user_id {
                    if event.user_id != Some(user_id) {
                        return false;
                    }
                }

                if !filter.event_types.is_empty()
                    && !filter.event_types.contains(&event.event_type) {
                        return false;
                    }

                if let Some(severity) = &filter.severity {
                    if &event.severity != severity {
                        return false;
                    }
                }

                if let Some(start_time) = filter.start_time {
                    if event.timestamp < start_time {
                        return false;
                    }
                }

                if let Some(end_time) = filter.end_time {
                    if event.timestamp > end_time {
                        return false;
                    }
                }

                if let Some(resource_type) = &filter.resource_type {
                    if event.resource_type.as_ref() != Some(resource_type) {
                        return false;
                    }
                }

                if let Some(success_only) = filter.success_only {
                    if event.success != success_only {
                        return false;
                    }
                }

                if let Some(requires_review) = filter.requires_review {
                    if event.requires_review != requires_review {
                        return false;
                    }
                }

                true
            })
            .cloned()
            .collect();

        // Sort by timestamp (newest first)
        filtered_events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        // Apply offset and limit
        if let Some(offset) = filter.offset {
            if offset >= filtered_events.len() {
                return Ok(Vec::new());
            }
            filtered_events = filtered_events.into_iter().skip(offset).collect();
        }

        if let Some(limit) = filter.limit {
            filtered_events.truncate(limit);
        }

        Ok(filtered_events)
    }
}

/// Async audit event processor
pub struct AuditProcessor {
    logger: Box<dyn AuditLogger>,
    event_receiver: mpsc::UnboundedReceiver<AuditEvent>,
    event_sender: mpsc::UnboundedSender<AuditEvent>,
}

impl AuditProcessor {
    /// Create new audit processor
    pub fn new(logger: Box<dyn AuditLogger>) -> Self {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();

        Self {
            logger,
            event_receiver,
            event_sender,
        }
    }

    /// Get audit event sender for other components
    pub fn get_sender(&self) -> mpsc::UnboundedSender<AuditEvent> {
        self.event_sender.clone()
    }

    /// Start processing audit events
    pub async fn start_processing(mut self) {
        tracing::info!("Starting audit event processor");

        while let Some(event) = self.event_receiver.recv().await {
            if let Err(e) = self.logger.log_event(event.clone()).await {
                tracing::error!(
                    event_id = %event.id,
                    error = %e,
                    "Failed to log audit event"
                );
            }

            // Log high-risk events to tracing as well
            if let Some(risk_score) = event.risk_score {
                if risk_score >= 70 {
                    tracing::warn!(
                        event_id = %event.id,
                        event_type = ?event.event_type,
                        risk_score = risk_score,
                        user_id = ?event.user_id,
                        "High-risk audit event detected"
                    );
                }
            }

            // Log critical events
            if event.severity == AuditSeverity::Critical {
                tracing::error!(
                    event_id = %event.id,
                    event_type = ?event.event_type,
                    description = %event.description,
                    "Critical audit event"
                );
            }
        }

        tracing::info!("Audit event processor stopped");
    }
}

/// Convenient audit logging functions
pub struct AuditLog {
    sender: mpsc::UnboundedSender<AuditEvent>,
}

impl AuditLog {
    pub fn new(sender: mpsc::UnboundedSender<AuditEvent>) -> Self {
        Self { sender }
    }

    /// Log an audit event
    pub fn log(&self, event: AuditEvent) {
        if let Err(e) = self.sender.send(event) {
            tracing::error!(error = %e, "Failed to send audit event");
        }
    }

    /// Log user login success
    pub fn login_success(&self, user_id: Uuid, username: &str, session_id: &str, client_ip: Option<String>) {
        self.log(AuditEvent::login_success(user_id, username, session_id, client_ip));
    }

    /// Log user login failure
    pub fn login_failure(&self, username: &str, reason: &str, client_ip: Option<String>) {
        self.log(AuditEvent::login_failure(username, reason, client_ip));
    }

    /// Log data access
    pub fn data_access(&self, user_id: Uuid, resource_type: &str, resource_id: &str, operation: &str, success: bool) {
        self.log(AuditEvent::data_access(user_id, resource_type, resource_id, operation, success));
    }

    /// Log security violation
    pub fn security_violation(&self, user_id: Option<Uuid>, violation_type: &str, description: &str, client_ip: Option<String>) {
        self.log(AuditEvent::security_violation(user_id, violation_type, description, client_ip));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_event_builder() {
        let user_id = Uuid::new_v4();
        let event = AuditEvent::builder()
            .event_type(AuditEventType::LoginSuccess)
            .user_id(user_id)
            .username("testuser")
            .description("Test login".to_string())
            .success(true)
            .build();

        assert_eq!(event.event_type, AuditEventType::LoginSuccess);
        assert_eq!(event.user_id, Some(user_id));
        assert_eq!(event.username, Some("testuser".to_string()));
        assert!(event.success);
    }

    #[tokio::test]
    async fn test_in_memory_audit_logger() {
        let logger = InMemoryAuditLogger::new();
        let user_id = Uuid::new_v4();
        
        let event = AuditEvent::login_success(
            user_id,
            "testuser",
            "session123",
            Some("192.168.1.1".to_string()),
        );

        // Log event
        logger.log_event(event.clone()).await.unwrap();

        // Query events
        let filter = AuditEventFilter {
            user_id: Some(user_id),
            ..Default::default()
        };

        let events = logger.query_events(filter).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].user_id, Some(user_id));
        assert_eq!(events[0].event_type, AuditEventType::LoginSuccess);
    }

    #[test]
    fn test_audit_event_factory_methods() {
        let user_id = Uuid::new_v4();
        
        // Test login success
        let login_event = AuditEvent::login_success(
            user_id,
            "testuser",
            "session123",
            Some("192.168.1.1".to_string()),
        );
        assert_eq!(login_event.event_type, AuditEventType::LoginSuccess);
        assert!(login_event.success);

        // Test login failure
        let failure_event = AuditEvent::login_failure(
            "baduser",
            "Invalid credentials",
            Some("192.168.1.1".to_string()),
        );
        assert_eq!(failure_event.event_type, AuditEventType::LoginFailure);
        assert!(!failure_event.success);
        assert!(failure_event.requires_review);

        // Test security violation
        let violation_event = AuditEvent::security_violation(
            Some(user_id),
            "Brute force attempt",
            "Multiple failed login attempts",
            Some("192.168.1.1".to_string()),
        );
        assert_eq!(violation_event.event_type, AuditEventType::SecurityViolation);
        assert_eq!(violation_event.severity, AuditSeverity::Critical);
        assert!(violation_event.requires_review);
    }
}