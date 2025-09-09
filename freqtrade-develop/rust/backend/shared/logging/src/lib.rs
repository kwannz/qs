//! # 统一日志系统 - 生产级可观测性
//! 
//! 提供结构化日志、分布式追踪、性能监控、审计日志等功能
//! 遵循 Rust-first、MVP 原则，支持渐进增强

pub mod config;
pub mod formatter;
pub mod middleware;
pub mod audit;
pub mod metrics;
pub mod trace;
pub mod storage;

use std::sync::Once;
use anyhow::Result;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Registry};
use tracing_appender::non_blocking::WorkerGuard;

pub use config::LogConfig;
pub use formatter::{LogEntry, LogLevel};
pub use middleware::LoggingLayer;
pub use audit::{AuditLogger, AuditEntry};
pub use metrics::MetricsCollector;

/// 全局日志初始化
static INIT: Once = Once::new();

/// 日志系统
pub struct LogSystem {
    config: LogConfig,
    _guard: Option<WorkerGuard>,
}

impl LogSystem {
    /// 创建新的日志系统
    pub fn new(config: LogConfig) -> Result<Self> {
        Ok(Self {
            config,
            _guard: None,
        })
    }

    /// 初始化日志系统
    pub fn init(&mut self) -> Result<()> {
        INIT.call_once(|| {
            self.setup_tracing().expect("Failed to initialize logging");
        });
        Ok(())
    }

    /// 设置追踪订阅器
    fn setup_tracing(&mut self) -> Result<()> {
        let env_filter = EnvFilter::try_from_default_env()
            .or_else(|_| EnvFilter::try_new(&self.config.level))
            .unwrap();

        let subscriber = Registry::default().with(env_filter);

        match self.config.output.as_str() {
            "json" => {
                if self.config.file_enabled {
                    // JSON文件输出
                    let file_appender = self.create_file_appender()?;
                    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
                    self._guard = Some(guard);

                    subscriber
                        .with(
                            tracing_subscriber::fmt::layer()
                                .json()
                                .with_writer(non_blocking)
                                .with_current_span(true)
                                .with_span_list(true)
                        )
                        .with(
                            tracing_subscriber::fmt::layer()
                                .json()
                                .with_writer(std::io::stdout)
                        )
                        .init();
                } else {
                    // 仅控制台JSON输出
                    subscriber
                        .with(
                            tracing_subscriber::fmt::layer()
                                .json()
                                .with_current_span(true)
                                .with_span_list(true)
                        )
                        .init();
                }
            }
            "pretty" => {
                // 美化输出 (开发环境)
                subscriber
                    .with(
                        tracing_subscriber::fmt::layer()
                            .pretty()
                            .with_thread_ids(true)
                            .with_thread_names(true)
                    )
                    .init();
            }
            _ => {
                // 紧凑输出 (默认)
                subscriber
                    .with(
                        tracing_subscriber::fmt::layer()
                            .compact()
                    )
                    .init();
            }
        }

        tracing::info!(
            service = %self.config.service_name,
            version = %self.config.version,
            "日志系统初始化完成"
        );

        Ok(())
    }

    /// 创建文件输出器
    fn create_file_appender(&self) -> Result<Box<dyn std::io::Write + Send + 'static>> {
        use tracing_appender::rolling::{RollingFileAppender, Rotation};

        let rotation = match self.config.rotation.as_str() {
            "hourly" => Rotation::HOURLY,
            "daily" => Rotation::DAILY,
            "never" => Rotation::NEVER,
            _ => Rotation::DAILY,
        };

        let appender = RollingFileAppender::new(
            rotation,
            &self.config.log_dir,
            format!("{}.log", self.config.service_name)
        );

        Ok(Box::new(appender))
    }
}

/// 快速初始化日志系统 (MVP模式)
pub fn init_simple(service_name: &str) -> Result<LogSystem> {
    let config = LogConfig::default_for_service(service_name);
    let mut log_system = LogSystem::new(config)?;
    log_system.init()?;
    Ok(log_system)
}

/// 快速初始化生产级日志系统
pub fn init_production(service_name: &str, log_dir: &str) -> Result<LogSystem> {
    let config = LogConfig::production_for_service(service_name, log_dir);
    let mut log_system = LogSystem::new(config)?;
    log_system.init()?;
    Ok(log_system)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_system_creation() {
        let config = LogConfig::default_for_service("test-service");
        let log_system = LogSystem::new(config);
        assert!(log_system.is_ok());
    }
}