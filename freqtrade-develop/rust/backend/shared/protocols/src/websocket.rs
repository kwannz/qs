/// WebSocket protocol definitions and utilities
use serde::{Deserialize, Serialize};
use std::fmt;

/// WebSocket connection states
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WebSocketState {
    Connecting,
    Connected,
    Reconnecting,
    Disconnected,
    Error,
}

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebSocketMessageType {
    Subscription,
    Unsubscription,
    Data,
    Heartbeat,
    Error,
    Authentication,
}

/// Generic WebSocket message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketMessage {
    pub message_type: WebSocketMessageType,
    pub channel: Option<String>,
    pub symbol: Option<String>,
    pub data: serde_json::Value,
    pub timestamp: Option<u64>,
    pub id: Option<String>,
}

/// WebSocket 订阅请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketSubscriptionRequest {
    pub channel: String,
    pub symbol: String,
    pub parameters: Option<serde_json::Value>,
}

/// WebSocket error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WebSocketError {
    ConnectionFailed(String),
    AuthenticationFailed(String),
    SubscriptionFailed(String),
    MessageParsingFailed(String),
    RateLimitExceeded(String),
    UnknownError(String),
}

impl fmt::Display for WebSocketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WebSocketError::ConnectionFailed(msg) => write!(f, "Connection failed: {}", msg),
            WebSocketError::AuthenticationFailed(msg) => write!(f, "Authentication failed: {}", msg),
            WebSocketError::SubscriptionFailed(msg) => write!(f, "Subscription failed: {}", msg),
            WebSocketError::MessageParsingFailed(msg) => write!(f, "Message parsing failed: {}", msg),
            WebSocketError::RateLimitExceeded(msg) => write!(f, "Rate limit exceeded: {}", msg),
            WebSocketError::UnknownError(msg) => write!(f, "Unknown error: {}", msg),
        }
    }
}

impl std::error::Error for WebSocketError {}

/// WebSocket configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketConfig {
    pub url: String,
    pub heartbeat_interval: Option<u64>,
    pub reconnect_attempts: Option<u32>,
    pub reconnect_delay: Option<u64>,
    pub authentication: Option<serde_json::Value>,
}

/// Backoff strategy for reconnection
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WebSocketBackoffStrategy {
    Fixed(u64),
    Linear(u64),
    Exponential { base: u64, max: u64 },
}

impl WebSocketBackoffStrategy {
    pub fn next_delay(&self, attempt: u32) -> u64 {
        match self {
            WebSocketBackoffStrategy::Fixed(delay) => *delay,
            WebSocketBackoffStrategy::Linear(step) => step * (u64::from(attempt) + 1),
            WebSocketBackoffStrategy::Exponential { base, max } => {
                let exponential_delay = base * (2_u64.pow(attempt));
                exponential_delay.min(*max)
            }
        }
    }
}

/// WebSocket connection metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct WebSocketMetrics {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub connection_attempts: u32,
    pub reconnection_count: u32,
    pub last_heartbeat: Option<u64>,
    pub uptime_seconds: u64,
}