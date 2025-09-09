//! 审计日志模块 - 满足金融合规要求

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use uuid::Uuid;
use anyhow::Result;

#[cfg(feature = "audit-db")]
use sqlx::PgPool;

/// 审计日志条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// 唯一ID
    pub id: String,
    
    /// 时间戳
    pub timestamp: DateTime<Utc>,
    
    /// 追踪ID
    pub trace_id: String,
    
    /// 用户ID (可选)
    pub user_id: Option<String>,
    
    /// 操作类型
    pub action: String,
    
    /// 资源标识
    pub resource: Option<String>,
    
    /// 操作结果 (SUCCESS, FAILURE, PARTIAL)
    pub result: String,
    
    /// 详细信息
    pub details: Value,
    
    /// IP地址
    pub ip_address: Option<String>,
    
    /// User-Agent
    pub user_agent: Option<String>,
    
    /// 服务名称
    pub service: String,
    
    /// 严重级别 (LOW, MEDIUM, HIGH, CRITICAL)
    pub severity: String,
}

impl AuditEntry {
    /// 创建新的审计条目
    pub fn new(action: String, service: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            trace_id: Uuid::new_v4().to_string(),
            user_id: None,
            action,
            resource: None,
            result: "SUCCESS".to_string(),
            details: Value::Null,
            ip_address: None,
            user_agent: None,
            service,
            severity: "MEDIUM".to_string(),
        }
    }

    /// 设置用户信息
    pub fn with_user(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }

    /// 设置追踪ID
    pub fn with_trace_id(mut self, trace_id: String) -> Self {
        self.trace_id = trace_id;
        self
    }

    /// 设置资源
    pub fn with_resource(mut self, resource: String) -> Self {
        self.resource = Some(resource);
        self
    }

    /// 设置操作结果
    pub fn with_result(mut self, result: String) -> Self {
        self.result = result;
        self
    }

    /// 设置详细信息
    pub fn with_details(mut self, details: Value) -> Self {
        self.details = details;
        self
    }

    /// 设置请求信息
    pub fn with_request_info(mut self, ip_address: String, user_agent: String) -> Self {
        self.ip_address = Some(ip_address);
        self.user_agent = Some(user_agent);
        self
    }

    /// 设置严重级别
    pub fn with_severity(mut self, severity: String) -> Self {
        self.severity = severity;
        self
    }

    /// 标记为失败
    pub fn as_failure(mut self, error: &str) -> Self {
        self.result = "FAILURE".to_string();
        self.details = serde_json::json!({
            "error": error,
            "original_details": self.details
        });
        self.severity = "HIGH".to_string();
        self
    }

    /// 是否为敏感操作
    pub fn is_sensitive(&self) -> bool {
        matches!(self.action.as_str(), 
            "order_create" | "order_cancel" | "withdraw" | "deposit" | 
            "api_key_create" | "api_key_delete" | "password_change" |
            "permission_grant" | "permission_revoke"
        )
    }
}

/// 交易审计条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingAuditEntry {
    #[serde(flatten)]
    pub base: AuditEntry,
    
    /// 订单ID
    pub order_id: String,
    
    /// 交易对
    pub symbol: String,
    
    /// 交易方向
    pub side: String,
    
    /// 数量
    pub quantity: f64,
    
    /// 价格 (可选)
    pub price: Option<f64>,
    
    /// 订单状态
    pub status: String,
    
    /// 成交金额
    pub filled_amount: Option<f64>,
}

impl TradingAuditEntry {
    /// 创建交易审计条目
    pub fn new(action: String, order_id: String, symbol: String) -> Self {
        Self {
            base: AuditEntry::new(action, "trading".to_string())
                .with_severity("HIGH".to_string()),
            order_id,
            symbol,
            side: String::new(),
            quantity: 0.0,
            price: None,
            status: String::new(),
            filled_amount: None,
        }
    }

    /// 设置交易参数
    pub fn with_trade_params(mut self, side: String, quantity: f64, price: Option<f64>) -> Self {
        self.side = side;
        self.quantity = quantity;
        self.price = price;
        self
    }

    /// 设置订单状态
    pub fn with_status(mut self, status: String) -> Self {
        self.status = status;
        self
    }

    /// 设置成交金额
    pub fn with_filled_amount(mut self, amount: f64) -> Self {
        self.filled_amount = Some(amount);
        self
    }
}

impl From<TradingAuditEntry> for AuditEntry {
    fn from(trading: TradingAuditEntry) -> Self {
        let mut base = trading.base;
        base.details = serde_json::json!({
            "order_id": trading.order_id,
            "symbol": trading.symbol,
            "side": trading.side,
            "quantity": trading.quantity,
            "price": trading.price,
            "status": trading.status,
            "filled_amount": trading.filled_amount,
        });
        base
    }
}

/// 审计日志记录器
pub struct AuditLogger {
    /// 异步发送通道
    tx: mpsc::UnboundedSender<AuditEntry>,
    
    /// 服务名称
    service_name: String,
}

impl AuditLogger {
    /// 创建新的审计记录器
    pub fn new(service_name: String) -> (Self, AuditLoggerHandle) {
        let (tx, rx) = mpsc::unbounded_channel();
        
        let logger = Self {
            tx,
            service_name: service_name.clone(),
        };

        let handle = AuditLoggerHandle {
            rx,
            service_name,
            #[cfg(feature = "audit-db")]
            pool: None,
        };

        (logger, handle)
    }

    /// 记录审计日志
    pub fn log(&self, entry: AuditEntry) -> Result<()> {
        self.tx.send(entry)
            .map_err(|e| anyhow::anyhow!("发送审计日志失败: {}", e))
    }

    /// 记录交易审计
    pub fn log_trading(&self, entry: TradingAuditEntry) -> Result<()> {
        self.log(entry.into())
    }

    /// 记录API调用审计
    pub fn log_api_call(&self, method: &str, path: &str, status_code: u16, user_id: Option<String>) -> Result<()> {
        let result = if status_code >= 400 { "FAILURE" } else { "SUCCESS" };
        let severity = if status_code >= 500 { "HIGH" } else { "MEDIUM" };

        let entry = AuditEntry::new("api_call".to_string(), self.service_name.clone())
            .with_result(result.to_string())
            .with_severity(severity.to_string())
            .with_details(serde_json::json!({
                "method": method,
                "path": path,
                "status_code": status_code,
            }));

        let entry = if let Some(uid) = user_id {
            entry.with_user(uid)
        } else {
            entry
        };

        self.log(entry)
    }

    /// 记录敏感操作审计
    pub fn log_sensitive_operation(&self, action: &str, resource: &str, user_id: Option<String>, details: Value) -> Result<()> {
        let entry = AuditEntry::new(action.to_string(), self.service_name.clone())
            .with_resource(resource.to_string())
            .with_details(details)
            .with_severity("CRITICAL".to_string());

        let entry = if let Some(uid) = user_id {
            entry.with_user(uid)
        } else {
            entry
        };

        self.log(entry)
    }
}

/// 审计日志处理句柄
pub struct AuditLoggerHandle {
    rx: mpsc::UnboundedReceiver<AuditEntry>,
    service_name: String,
    
    #[cfg(feature = "audit-db")]
    pool: Option<Arc<PgPool>>,
}

impl AuditLoggerHandle {
    /// 设置数据库连接池
    #[cfg(feature = "audit-db")]
    pub fn with_database(mut self, pool: Arc<PgPool>) -> Self {
        self.pool = Some(pool);
        self
    }

    /// 启动审计日志处理器
    pub async fn run(mut self) {
        tracing::info!(service = %self.service_name, "启动审计日志处理器");

        while let Some(entry) = self.rx.recv().await {
            if let Err(e) = self.process_entry(entry).await {
                tracing::error!(error = %e, "处理审计日志失败");
            }
        }

        tracing::info!("审计日志处理器已停止");
    }

    /// 处理审计条目
    async fn process_entry(&self, entry: AuditEntry) -> Result<()> {
        // 记录到tracing
        if entry.is_sensitive() {
            tracing::warn!(
                audit_id = %entry.id,
                action = %entry.action,
                user_id = ?entry.user_id,
                resource = ?entry.resource,
                result = %entry.result,
                severity = %entry.severity,
                "敏感操作审计"
            );
        } else {
            tracing::info!(
                audit_id = %entry.id,
                action = %entry.action,
                user_id = ?entry.user_id,
                resource = ?entry.resource,
                result = %entry.result,
                "操作审计"
            );
        }

        // 存储到数据库 (如果启用)
        #[cfg(feature = "audit-db")]
        if let Some(pool) = &self.pool {
            self.store_to_database(&entry, pool).await?;
        }

        Ok(())
    }

    /// 存储到数据库
    #[cfg(feature = "audit-db")]
    async fn store_to_database(&self, entry: &AuditEntry, pool: &PgPool) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO audit_logs (
                id, timestamp, trace_id, user_id, action, resource, result, 
                details, ip_address, user_agent, service, severity
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            "#,
            entry.id,
            entry.timestamp,
            entry.trace_id,
            entry.user_id,
            entry.action,
            entry.resource,
            entry.result,
            entry.details,
            entry.ip_address,
            entry.user_agent,
            entry.service,
            entry.severity
        )
        .execute(pool)
        .await
        .map_err(|e| anyhow::anyhow!("存储审计日志到数据库失败: {}", e))?;

        Ok(())
    }
}

/// 审计宏 - 简化审计日志记录
#[macro_export]
macro_rules! audit {
    ($logger:expr, $action:expr, $($key:ident = $value:expr),*) => {
        {
            let entry = $crate::audit::AuditEntry::new(
                $action.to_string(),
                "gateway".to_string()
            )
            $(.with_details(serde_json::json!({
                stringify!($key): $value,
            })))*;
            
            if let Err(e) = $logger.log(entry) {
                tracing::error!(error = %e, "审计日志记录失败");
            }
        }
    };
}

/// 交易审计宏
#[macro_export]  
macro_rules! audit_trading {
    ($logger:expr, $action:expr, order_id = $order_id:expr, symbol = $symbol:expr, $($key:ident = $value:expr),*) => {
        {
            let mut entry = $crate::audit::TradingAuditEntry::new(
                $action.to_string(),
                $order_id.to_string(),
                $symbol.to_string()
            );
            
            $(
                match stringify!($key) {
                    "side" => entry.side = $value.to_string(),
                    "quantity" => entry.quantity = $value,
                    "price" => entry.price = Some($value),
                    "status" => entry.status = $value.to_string(),
                    _ => {}
                }
            )*
            
            if let Err(e) = $logger.log_trading(entry) {
                tracing::error!(error = %e, "交易审计日志记录失败");
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_entry_creation() {
        let entry = AuditEntry::new("test_action".to_string(), "test_service".to_string());
        assert_eq!(entry.action, "test_action");
        assert_eq!(entry.service, "test_service");
        assert_eq!(entry.result, "SUCCESS");
    }

    #[test]
    fn test_trading_audit_entry() {
        let entry = TradingAuditEntry::new(
            "order_create".to_string(),
            "order-123".to_string(),
            "BTCUSDT".to_string()
        );
        
        assert_eq!(entry.base.action, "order_create");
        assert_eq!(entry.order_id, "order-123");
        assert_eq!(entry.symbol, "BTCUSDT");
        assert_eq!(entry.base.service, "trading");
    }

    #[test]
    fn test_sensitive_operation_detection() {
        let entry = AuditEntry::new("order_create".to_string(), "trading".to_string());
        assert!(entry.is_sensitive());
        
        let entry = AuditEntry::new("health_check".to_string(), "gateway".to_string());
        assert!(!entry.is_sensitive());
    }

    #[tokio::test]
    async fn test_audit_logger() {
        let (logger, mut handle) = AuditLogger::new("test-service".to_string());
        
        // 发送测试日志
        let entry = AuditEntry::new("test_action".to_string(), "test_service".to_string());
        logger.log(entry).unwrap();
        
        // 测试接收
        tokio::spawn(async move {
            if let Some(entry) = handle.rx.recv().await {
                assert_eq!(entry.action, "test_action");
            }
        });
    }
}