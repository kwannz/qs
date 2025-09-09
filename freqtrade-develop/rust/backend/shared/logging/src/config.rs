//! 日志配置管理

use serde::{Deserialize, Serialize};
use std::env;

/// 日志配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogConfig {
    /// 服务名称
    pub service_name: String,
    
    /// 服务版本
    pub version: String,
    
    /// 日志级别 (trace, debug, info, warn, error)
    pub level: String,
    
    /// 输出格式 (json, pretty, compact)
    pub output: String,
    
    /// 是否启用文件输出
    pub file_enabled: bool,
    
    /// 日志目录
    pub log_dir: String,
    
    /// 轮转策略 (daily, hourly, never)
    pub rotation: String,
    
    /// 是否启用审计日志
    pub audit_enabled: bool,
    
    /// 是否启用性能监控
    pub metrics_enabled: bool,
    
    /// 追踪采样率 (0.0-1.0)
    pub trace_sample_rate: f64,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            service_name: "unknown".to_string(),
            version: "0.1.0".to_string(),
            level: "info".to_string(),
            output: "compact".to_string(),
            file_enabled: false,
            log_dir: "./logs".to_string(),
            rotation: "daily".to_string(),
            audit_enabled: false,
            metrics_enabled: false,
            trace_sample_rate: 0.1,
        }
    }
}

impl LogConfig {
    /// 为特定服务创建默认配置 (开发环境)
    pub fn default_for_service(service_name: &str) -> Self {
        Self {
            service_name: service_name.to_string(),
            version: env::var("SERVICE_VERSION").unwrap_or_else(|_| "0.1.0".to_string()),
            level: env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
            output: env::var("LOG_OUTPUT").unwrap_or_else(|_| "pretty".to_string()),
            file_enabled: false,
            ..Default::default()
        }
    }

    /// 为特定服务创建生产级配置
    pub fn production_for_service(service_name: &str, log_dir: &str) -> Self {
        Self {
            service_name: service_name.to_string(),
            version: env::var("SERVICE_VERSION").unwrap_or_else(|_| "0.1.0".to_string()),
            level: env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
            output: env::var("LOG_OUTPUT").unwrap_or_else(|_| "json".to_string()),
            file_enabled: true,
            log_dir: log_dir.to_string(),
            rotation: env::var("LOG_ROTATION").unwrap_or_else(|_| "daily".to_string()),
            audit_enabled: env::var("AUDIT_ENABLED").map(|v| v == "true").unwrap_or(true),
            metrics_enabled: env::var("METRICS_ENABLED").map(|v| v == "true").unwrap_or(true),
            trace_sample_rate: env::var("TRACE_SAMPLE_RATE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.1),
        }
    }

    /// 从环境变量加载配置
    pub fn from_env() -> Self {
        Self {
            service_name: env::var("SERVICE_NAME").unwrap_or_else(|_| "unknown".to_string()),
            version: env::var("SERVICE_VERSION").unwrap_or_else(|_| "0.1.0".to_string()),
            level: env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
            output: env::var("LOG_OUTPUT").unwrap_or_else(|_| "json".to_string()),
            file_enabled: env::var("LOG_FILE_ENABLED").map(|v| v == "true").unwrap_or(true),
            log_dir: env::var("LOG_DIR").unwrap_or_else(|_| "./logs".to_string()),
            rotation: env::var("LOG_ROTATION").unwrap_or_else(|_| "daily".to_string()),
            audit_enabled: env::var("AUDIT_ENABLED").map(|v| v == "true").unwrap_or(false),
            metrics_enabled: env::var("METRICS_ENABLED").map(|v| v == "true").unwrap_or(false),
            trace_sample_rate: env::var("TRACE_SAMPLE_RATE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.1),
        }
    }

    /// 验证配置有效性
    pub fn validate(&self) -> Result<(), String> {
        // 验证日志级别
        match self.level.as_str() {
            "trace" | "debug" | "info" | "warn" | "error" => {},
            _ => return Err(format!("无效的日志级别: {}", self.level)),
        }

        // 验证输出格式
        match self.output.as_str() {
            "json" | "pretty" | "compact" => {},
            _ => return Err(format!("无效的输出格式: {}", self.output)),
        }

        // 验证轮转策略
        match self.rotation.as_str() {
            "daily" | "hourly" | "never" => {},
            _ => return Err(format!("无效的轮转策略: {}", self.rotation)),
        }

        // 验证采样率
        if self.trace_sample_rate < 0.0 || self.trace_sample_rate > 1.0 {
            return Err(format!("追踪采样率必须在0.0-1.0之间: {}", self.trace_sample_rate));
        }

        Ok(())
    }

    /// 更新日志级别 (运行时调整)
    pub fn update_level(&mut self, new_level: String) -> Result<(), String> {
        match new_level.as_str() {
            "trace" | "debug" | "info" | "warn" | "error" => {
                self.level = new_level;
                Ok(())
            },
            _ => Err(format!("无效的日志级别: {new_level}")),
        }
    }
}