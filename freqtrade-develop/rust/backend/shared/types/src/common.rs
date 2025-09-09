use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// 通用ID类型
pub type Id = Uuid;

/// 时间戳类型
pub type Timestamp = DateTime<Utc>;

/// 通用响应包装器
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub message: Option<String>,
    pub timestamp: Timestamp,
    pub request_id: Option<Id>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            message: None,
            timestamp: Utc::now(),
            request_id: None,
        }
    }

    pub fn error(error: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error),
            message: None,
            timestamp: Utc::now(),
            request_id: None,
        }
    }

    pub fn with_request_id(mut self, request_id: Id) -> Self {
        self.request_id = Some(request_id);
        self
    }

    pub fn with_message(mut self, message: String) -> Self {
        self.message = Some(message);
        self
    }
}

/// 分页请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginationParams {
    pub page: Option<u32>,
    pub limit: Option<u32>,
    pub offset: Option<u32>,
}

impl Default for PaginationParams {
    fn default() -> Self {
        Self {
            page: Some(1),
            limit: Some(20),
            offset: None,
        }
    }
}

impl PaginationParams {
    pub fn get_offset(&self) -> u32 {
        if let Some(offset) = self.offset {
            offset
        } else {
            let page = self.page.unwrap_or(1).max(1);
            let limit = self.limit.unwrap_or(20);
            (page - 1) * limit
        }
    }

    pub fn get_limit(&self) -> u32 {
        self.limit.unwrap_or(20).min(100) // 最大100条记录
    }
}

/// 分页响应数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginatedResponse<T> {
    pub items: Vec<T>,
    pub total: u64,
    pub page: u32,
    pub limit: u32,
    pub total_pages: u32,
    pub has_next: bool,
    pub has_prev: bool,
}

impl<T> PaginatedResponse<T> {
    pub fn new(items: Vec<T>, total: u64, page: u32, limit: u32) -> Self {
        let total_pages = ((total as f64) / (limit as f64)).ceil() as u32;
        let has_next = page < total_pages;
        let has_prev = page > 1;

        Self {
            items,
            total,
            page,
            limit,
            total_pages,
            has_next,
            has_prev,
        }
    }
}

/// 服务健康状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Degraded => write!(f, "degraded"),
            HealthStatus::Unhealthy => write!(f, "unhealthy"),
            HealthStatus::Unknown => write!(f, "unknown"),
        }
    }
}

/// 服务信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub status: HealthStatus,
    pub url: String,
    pub last_check: Option<Timestamp>,
    pub uptime: Option<u64>, // seconds
}

/// 用户信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: Id,
    pub username: String,
    pub email: String,
    pub role: UserRole,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    pub last_login: Option<Timestamp>,
    pub is_active: bool,
}

/// 用户角色
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UserRole {
    Admin,
    Trader,
    Analyst,
    Viewer,
}

impl fmt::Display for UserRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UserRole::Admin => write!(f, "admin"),
            UserRole::Trader => write!(f, "trader"),
            UserRole::Analyst => write!(f, "analyst"),
            UserRole::Viewer => write!(f, "viewer"),
        }
    }
}

/// 审计日志条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLog {
    pub id: Id,
    pub user_id: Option<Id>,
    pub action: String,
    pub resource: String,
    pub resource_id: Option<String>,
    pub details: Option<serde_json::Value>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub timestamp: Timestamp,
    pub success: bool,
}

/// 配置信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigInfo {
    pub key: String,
    pub value: String,
    pub category: String,
    pub description: Option<String>,
    pub is_sensitive: bool,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

/// 系统指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub network_in: u64,
    pub network_out: u64,
    pub active_connections: u32,
    pub timestamp: Timestamp,
}

/// 服务指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMetrics {
    pub service_id: String,
    pub requests_per_second: f64,
    pub average_response_time: f64,
    pub error_rate: f64,
    pub success_rate: f64,
    pub active_connections: u32,
    pub timestamp: Timestamp,
}

/// 持仓方向 - 统一定义避免模块间重复
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PositionSide {
    Long,
    Short,
    Both, // 对于支持双向持仓的交易所
    Flat, // 无持仓状态
}