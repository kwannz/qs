use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use platform_types::HealthStatus;

/// 消息ID类型
pub type MessageId = Uuid;

/// 通用消息包装器
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message<T> {
    pub id: MessageId,
    pub timestamp: DateTime<Utc>,
    pub message_type: String,
    pub source: String,
    pub destination: Option<String>,
    pub correlation_id: Option<MessageId>,
    pub payload: T,
    pub metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
}

impl<T> Message<T> {
    pub fn new(message_type: &str, source: &str, payload: T) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            message_type: message_type.to_string(),
            source: source.to_string(),
            destination: None,
            correlation_id: None,
            payload,
            metadata: None,
        }
    }

    #[must_use]
    pub fn with_destination(mut self, destination: &str) -> Self {
        self.destination = Some(destination.to_string());
        self
    }

    #[must_use]
    pub fn with_correlation_id(mut self, correlation_id: MessageId) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    pub fn reply<R>(&self, message_type: &str, payload: R) -> Message<R> {
        Message {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            message_type: message_type.to_string(),
            source: self.destination.clone().unwrap_or_else(|| "unknown".to_string()),
            destination: Some(self.source.clone()),
            correlation_id: Some(self.id),
            payload,
            metadata: None,
        }
    }
}

/// 消息类型枚举
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MessageType {
    // 系统消息
    Heartbeat,
    SystemStatus,
    ConfigUpdate,
    ServiceDiscovery,
    HealthCheck,
    
    // 市场数据消息
    MarketData,
    Kline,
    Tick,
    OrderBook,
    Ticker,
    
    // 交易执行消息
    OrderRequest,
    OrderResponse,
    ExecutionReport,
    PositionUpdate,
    AccountUpdate,
    
    // 策略消息
    StrategySignal,
    StrategyStatus,
    StrategyConfig,
    BacktestRequest,
    BacktestResult,
    
    // 分析消息
    FactorData,
    AnalysisResult,
    OptimizationResult,
    
    // 错误和响应
    Error,
    Response,
    Acknowledgment,
}

/// 风险管理事件类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RiskEventType {
    OrderRejected,
    PositionLimitExceeded,
    DrawdownLimitExceeded,
    VolumeAnomalyDetected,
    PriceAnomalyDetected,
    SystemOverloaded,
    ConnectionLost,
}

/// 风险严重程度
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RiskSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// 系统状态消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatusMessage {
    pub service_name: String,
    pub status: HealthStatus,
    pub uptime_seconds: u64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub active_connections: u32,
    pub processed_messages: u64,
    pub error_count: u32,
    pub last_error: Option<String>,
}

/// 心跳消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatMessage {
    pub service_name: String,
    pub timestamp: DateTime<Utc>,
    pub sequence: u64,
    pub data: Option<serde_json::Value>,
}

/// 配置更新消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigUpdateMessage {
    pub config_key: String,
    pub old_value: Option<serde_json::Value>,
    pub new_value: serde_json::Value,
    pub updated_by: String,
    pub reason: Option<String>,
}

/// 错误消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMessage {
    pub error_code: String,
    pub error_type: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
    pub service: String,
    pub stack_trace: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// 响应消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMessage<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<ErrorMessage>,
    pub processing_time_ms: Option<u64>,
}

impl<T> ResponseMessage<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            processing_time_ms: None,
        }
    }

    pub fn error(error: ErrorMessage) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error),
            processing_time_ms: None,
        }
    }

    #[must_use]
    pub fn with_processing_time(mut self, processing_time_ms: u64) -> Self {
        self.processing_time_ms = Some(processing_time_ms);
        self
    }
}

/// 确认消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcknowledgmentMessage {
    pub message_id: MessageId,
    pub status: AckStatus,
    pub timestamp: DateTime<Utc>,
    pub processing_node: String,
}

/// 确认状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AckStatus {
    Received,
    Processing,
    Completed,
    Failed,
    Rejected,
}

/// 消息优先级
#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum MessagePriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// 消息路由信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageRoute {
    pub path: Vec<String>,
    pub timestamp: DateTime<Utc>,
    pub latency_ms: Option<u64>,
}

/// 批量消息包装器
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchMessage<T> {
    pub batch_id: MessageId,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub messages: Vec<Message<T>>,
    pub total_count: usize,
    pub batch_index: usize,
    pub is_last_batch: bool,
}

/// 订阅消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionMessage {
    pub subscription_id: MessageId,
    pub topics: Vec<String>,
    pub filters: Option<std::collections::HashMap<String, serde_json::Value>>,
    pub subscriber: String,
    pub action: SubscriptionAction,
}

/// 订阅操作
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SubscriptionAction {
    Subscribe,
    Unsubscribe,
    Update,
}

/// 广播消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BroadcastMessage<T> {
    pub broadcast_id: MessageId,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub topics: Vec<String>,
    pub payload: T,
    pub ttl_seconds: Option<u64>,
}

/// 请求-响应模式消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMessage<T> {
    pub request_id: MessageId,
    pub timestamp: DateTime<Utc>,
    pub requester: String,
    pub timeout_ms: Option<u64>,
    pub payload: T,
}

/// 消息验证结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageValidation {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// 消息统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageStats {
    pub total_sent: u64,
    pub total_received: u64,
    pub total_failed: u64,
    pub average_latency_ms: f64,
    pub messages_per_second: f64,
    pub error_rate: f64,
    pub by_type: std::collections::HashMap<String, u64>,
    pub by_source: std::collections::HashMap<String, u64>,
    pub timestamp: DateTime<Utc>,
}

/// 消息处理器接口
#[async_trait::async_trait]
pub trait MessageHandler<T> {
    async fn handle(&self, message: Message<T>) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

/// 消息路由器接口
#[async_trait::async_trait]
pub trait MessageRouter {
    async fn route(&self, message: &Message<serde_json::Value>) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>>;
    async fn register_route(&self, pattern: &str, destination: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    async fn unregister_route(&self, pattern: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

/// 消息序列化器接口
pub trait MessageSerializer {
    fn serialize<T: Serialize>(&self, message: &Message<T>) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>>;
    fn deserialize<T: for<'a> Deserialize<'a>>(&self, data: &[u8]) -> Result<Message<T>, Box<dyn std::error::Error + Send + Sync>>;
}