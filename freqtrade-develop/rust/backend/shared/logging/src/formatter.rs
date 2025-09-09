//! 结构化日志格式定义

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// 日志级别
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl From<tracing::Level> for LogLevel {
    fn from(level: tracing::Level) -> Self {
        match level {
            tracing::Level::TRACE => LogLevel::Trace,
            tracing::Level::DEBUG => LogLevel::Debug,
            tracing::Level::INFO => LogLevel::Info,
            tracing::Level::WARN => LogLevel::Warn,
            tracing::Level::ERROR => LogLevel::Error,
        }
    }
}

/// 标准日志条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// 时间戳
    pub timestamp: DateTime<Utc>,
    
    /// 日志级别
    pub level: LogLevel,
    
    /// 服务名称
    pub service: String,
    
    /// 追踪ID
    pub trace_id: String,
    
    /// 跨度ID
    pub span_id: String,
    
    /// 用户ID (可选)
    pub user_id: Option<String>,
    
    /// 操作/事件名称
    pub action: String,
    
    /// 详细信息
    pub details: Value,
    
    /// 执行时间 (毫秒)
    pub duration_ms: Option<u64>,
    
    /// 错误信息 (可选)
    pub error: Option<String>,
    
    /// 请求ID (可选)
    pub request_id: Option<String>,
    
    /// IP地址 (可选)
    pub ip_address: Option<String>,
    
    /// User-Agent (可选)
    pub user_agent: Option<String>,
}

impl LogEntry {
    /// 创建新的日志条目
    pub fn new(service: String, action: String) -> Self {
        Self {
            timestamp: Utc::now(),
            level: LogLevel::Info,
            service,
            trace_id: Uuid::new_v4().to_string(),
            span_id: Uuid::new_v4().to_string(),
            user_id: None,
            action,
            details: Value::Null,
            duration_ms: None,
            error: None,
            request_id: None,
            ip_address: None,
            user_agent: None,
        }
    }

    /// 设置日志级别
    pub fn with_level(mut self, level: LogLevel) -> Self {
        self.level = level;
        self
    }

    /// 设置追踪ID
    pub fn with_trace_id(mut self, trace_id: String) -> Self {
        self.trace_id = trace_id;
        self
    }

    /// 设置用户ID
    pub fn with_user_id(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }

    /// 设置详细信息
    pub fn with_details(mut self, details: Value) -> Self {
        self.details = details;
        self
    }

    /// 设置执行时间
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    /// 设置错误信息
    pub fn with_error(mut self, error: String) -> Self {
        self.error = Some(error);
        self.level = LogLevel::Error;
        self
    }

    /// 设置请求信息
    pub fn with_request_info(mut self, request_id: String, ip_address: String, user_agent: String) -> Self {
        self.request_id = Some(request_id);
        self.ip_address = Some(ip_address);
        self.user_agent = Some(user_agent);
        self
    }

    /// 转换为JSON字符串
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// 从JSON字符串解析
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// 交易操作日志
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingLogEntry {
    /// 基础日志信息
    #[serde(flatten)]
    pub base: LogEntry,
    
    /// 订单ID
    pub order_id: String,
    
    /// 交易对
    pub symbol: String,
    
    /// 交易方向 (BUY/SELL)
    pub side: String,
    
    /// 订单类型 (MARKET/LIMIT)
    pub order_type: String,
    
    /// 数量
    pub quantity: f64,
    
    /// 价格 (可选)
    pub price: Option<f64>,
    
    /// 订单状态
    pub status: String,
}

impl TradingLogEntry {
    /// 创建交易日志
    pub fn new(action: String, order_id: String, symbol: String) -> Self {
        Self {
            base: LogEntry::new("trading".to_string(), action),
            order_id,
            symbol,
            side: String::new(),
            order_type: String::new(),
            quantity: 0.0,
            price: None,
            status: String::new(),
        }
    }
}

/// 市场数据日志
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketLogEntry {
    /// 基础日志信息
    #[serde(flatten)]
    pub base: LogEntry,
    
    /// 数据源
    pub source: String,
    
    /// 交易对列表
    pub symbols: Vec<String>,
    
    /// 数据类型 (orderbook, trades, klines)
    pub data_type: String,
    
    /// 数据量
    pub count: u64,
}

impl MarketLogEntry {
    /// 创建市场数据日志
    pub fn new(action: String, source: String, data_type: String) -> Self {
        Self {
            base: LogEntry::new("market".to_string(), action),
            source,
            symbols: Vec::new(),
            data_type,
            count: 0,
        }
    }
}

/// HTTP请求日志
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpLogEntry {
    /// 基础日志信息
    #[serde(flatten)]
    pub base: LogEntry,
    
    /// HTTP方法
    pub method: String,
    
    /// 请求路径
    pub path: String,
    
    /// 查询参数
    pub query: Option<String>,
    
    /// 响应状态码
    pub status_code: u16,
    
    /// 响应大小 (字节)
    pub response_size: Option<u64>,
    
    /// 客户端IP
    pub client_ip: String,
}

impl HttpLogEntry {
    /// 创建HTTP请求日志
    pub fn new(method: String, path: String, client_ip: String) -> Self {
        Self {
            base: LogEntry::new("gateway".to_string(), "http_request".to_string()),
            method,
            path,
            query: None,
            status_code: 0,
            response_size: None,
            client_ip,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_entry_creation() {
        let entry = LogEntry::new("test-service".to_string(), "test-action".to_string());
        assert_eq!(entry.service, "test-service");
        assert_eq!(entry.action, "test-action");
        assert!(!entry.trace_id.is_empty());
    }

    #[test]
    fn test_log_entry_json_serialization() {
        let entry = LogEntry::new("test-service".to_string(), "test-action".to_string())
            .with_level(LogLevel::Info)
            .with_details(serde_json::json!({"test": "value"}));
        
        let json = entry.to_json().unwrap();
        assert!(json.contains("test-service"));
        assert!(json.contains("test-action"));
        
        let parsed = LogEntry::from_json(&json).unwrap();
        assert_eq!(parsed.service, entry.service);
        assert_eq!(parsed.action, entry.action);
    }

    #[test]
    fn test_trading_log_entry() {
        let entry = TradingLogEntry::new(
            "order_created".to_string(),
            "order-123".to_string(),
            "BTCUSDT".to_string()
        );
        
        assert_eq!(entry.base.service, "trading");
        assert_eq!(entry.base.action, "order_created");
        assert_eq!(entry.order_id, "order-123");
        assert_eq!(entry.symbol, "BTCUSDT");
    }
}