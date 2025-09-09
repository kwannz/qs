use anyhow::{Context, Result};
use config::{Config, Environment, File};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// 平台配置结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformConfig {
    pub platform: PlatformInfo,
    pub services: ServicesConfig,
    pub database: DatabaseConfig,
    pub message_queue: MessageQueueConfig,
    pub exchanges: HashMap<String, ExchangeConfig>,
    pub risk_management: RiskManagementConfig,
    pub logging: LoggingConfig,
    pub security: SecurityConfig,
    pub environment: EnvironmentConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInfo {
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub environment: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicesConfig {
    pub gateway: ServiceConfig,
    pub analytics: AnalyticsServicesConfig,
    pub execution: ExecutionServicesConfig,
    pub monitoring: MonitoringServicesConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    pub host: String,
    pub port: u16,
    pub enabled: Option<bool>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsServicesConfig {
    pub rust_engine: ServiceConfig,
    pub python_api: ServiceConfig,
    pub frontend: ServiceConfig,
    pub websocket: Option<ServiceConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionServicesConfig {
    pub main_engine: ServiceConfig,
    pub http_api: ServiceConfig,
    pub websocket: ServiceConfig,
    pub grpc: ServiceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringServicesConfig {
    pub prometheus: ServiceConfig,
    pub grafana: ServiceConfig,
    pub health_check: Option<ServiceConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub postgresql: PostgresConfig,
    pub redis: RedisConfig,
    pub mongodb: Option<MongoConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostgresConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub username: String,
    pub password: String,
    pub max_connections: Option<u32>,
    pub connection_timeout: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    pub host: String,
    pub port: u16,
    pub database: Option<u32>,
    pub pool_size: Option<u32>,
    pub timeout: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MongoConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub timeout: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageQueueConfig {
    pub redis_url: String,
    pub channels: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeConfig {
    pub name: String,
    pub enabled: bool,
    pub sandbox: bool,
    pub api_key: String,
    pub secret_key: String,
    pub passphrase: Option<String>,
    pub rest_base_url: String,
    pub websocket_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskManagementConfig {
    pub enabled: bool,
    pub max_daily_loss_ratio: f64,
    pub max_position_ratio: f64,
    pub max_single_order_ratio: f64,
    pub max_leverage: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub console_output: bool,
    pub file_output: bool,
    pub file_path: Option<String>,
    pub file_rotation: Option<String>,
    pub json_format: Option<bool>,
    pub verbose_logging: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub enable_authentication: bool,
    pub enable_authorization: Option<bool>,
    pub jwt_secret: String,
    pub cors_allowed_origins: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    pub name: String,
    pub debug: bool,
}

/// 配置管理器
pub struct ConfigManager {
    config: PlatformConfig,
}

impl ConfigManager {
    /// 从文件和环境变量加载配置
    pub fn new(config_path: Option<&str>) -> Result<Self> {
        let mut builder = Config::builder();

        // 加载基础配置文件
        if let Some(path) = config_path {
            builder = builder.add_source(File::with_name(path));
        } else {
            // 默认配置路径
            builder = builder
                .add_source(File::with_name("shared/config/platform").required(false))
                .add_source(File::with_name("config/platform").required(false));
        }

        // 根据环境加载特定配置
        let env = std::env::var("PLATFORM_ENV").unwrap_or_else(|_| "development".to_string());
        
        builder = builder
            .add_source(File::with_name(&format!("shared/config/{}", env)).required(false))
            .add_source(File::with_name(&format!("config/{}", env)).required(false));

        // 加载环境变量
        builder = builder.add_source(
            Environment::with_prefix("PLATFORM")
                .try_parsing(true)
                .separator("_")
                .list_separator(","),
        );

        let config = builder
            .build()
            .context("Failed to build configuration")?;

        let platform_config: PlatformConfig = config
            .try_deserialize()
            .context("Failed to deserialize configuration")?;

        Ok(ConfigManager {
            config: platform_config,
        })
    }

    /// 获取完整配置
    pub fn get_config(&self) -> &PlatformConfig {
        &self.config
    }

    /// 获取服务配置
    pub fn get_service_config(&self, service: &str) -> Option<&ServiceConfig> {
        match service {
            "gateway" => Some(&self.config.services.gateway),
            "analytics.rust_engine" => Some(&self.config.services.analytics.rust_engine),
            "analytics.python_api" => Some(&self.config.services.analytics.python_api),
            "analytics.frontend" => Some(&self.config.services.analytics.frontend),
            "execution.main_engine" => Some(&self.config.services.execution.main_engine),
            "execution.http_api" => Some(&self.config.services.execution.http_api),
            "execution.websocket" => Some(&self.config.services.execution.websocket),
            "execution.grpc" => Some(&self.config.services.execution.grpc),
            "monitoring.prometheus" => Some(&self.config.services.monitoring.prometheus),
            "monitoring.grafana" => Some(&self.config.services.monitoring.grafana),
            _ => None,
        }
    }

    /// 获取服务URL
    pub fn get_service_url(&self, service: &str) -> Option<String> {
        self.get_service_config(service).map(|config| {
            format!("http://{}:{}", config.host, config.port)
        })
    }

    /// 获取数据库连接字符串
    pub fn get_database_url(&self) -> String {
        format!(
            "postgresql://{}:{}@{}:{}/{}",
            self.config.database.postgresql.username,
            self.config.database.postgresql.password,
            self.config.database.postgresql.host,
            self.config.database.postgresql.port,
            self.config.database.postgresql.database
        )
    }

    /// 获取Redis连接字符串
    pub fn get_redis_url(&self) -> String {
        let db = self.config.database.redis.database.unwrap_or(0);
        format!(
            "redis://{}:{}/{}",
            self.config.database.redis.host, self.config.database.redis.port, db
        )
    }

    /// 验证配置
    pub fn validate(&self) -> Result<()> {
        // 验证必需的服务配置
        if self.config.services.gateway.port == 0 {
            return Err(anyhow::anyhow!("Gateway port must be configured"));
        }

        // 验证端口冲突
        let mut ports = std::collections::HashSet::new();
        
        let all_services = vec![
            &self.config.services.gateway,
            &self.config.services.analytics.rust_engine,
            &self.config.services.analytics.python_api,
            &self.config.services.analytics.frontend,
            &self.config.services.execution.main_engine,
            &self.config.services.execution.http_api,
            &self.config.services.execution.websocket,
            &self.config.services.execution.grpc,
            &self.config.services.monitoring.prometheus,
            &self.config.services.monitoring.grafana,
        ];

        for service in all_services {
            if !ports.insert(service.port) {
                return Err(anyhow::anyhow!("Port {} is used by multiple services", service.port));
            }
        }

        // 验证数据库配置
        if self.config.database.postgresql.host.is_empty() {
            return Err(anyhow::anyhow!("PostgreSQL host must be configured"));
        }

        Ok(())
    }

    /// 热重载配置
    pub fn reload(&mut self, config_path: Option<&str>) -> Result<()> {
        let new_manager = ConfigManager::new(config_path)?;
        new_manager.validate()?;
        self.config = new_manager.config;
        Ok(())
    }
}

impl fmt::Display for PlatformConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} v{} ({})",
            self.platform.name, self.platform.version, self.platform.environment
        )
    }
}

impl Default for PlatformConfig {
    fn default() -> Self {
        Self {
            platform: PlatformInfo {
                name: "crypto-quant-platform".to_string(),
                version: "2.0.0".to_string(),
                description: Some("End-to-end crypto quantitative trading platform".to_string()),
                environment: "development".to_string(),
            },
            services: ServicesConfig {
                gateway: ServiceConfig {
                    host: "0.0.0.0".to_string(),
                    port: 8080,
                    enabled: Some(true),
                    description: Some("Unified API Gateway".to_string()),
                },
                analytics: AnalyticsServicesConfig {
                    rust_engine: ServiceConfig {
                        host: "0.0.0.0".to_string(),
                        port: 8000,
                        enabled: Some(true),
                        description: Some("Analytics Rust Engine".to_string()),
                    },
                    python_api: ServiceConfig {
                        host: "0.0.0.0".to_string(),
                        port: 8001,
                        enabled: Some(true),
                        description: Some("Analytics Python API".to_string()),
                    },
                    frontend: ServiceConfig {
                        host: "0.0.0.0".to_string(),
                        port: 8002,
                        enabled: Some(true),
                        description: Some("Analytics Frontend Dashboard".to_string()),
                    },
                    websocket: Some(ServiceConfig {
                        host: "0.0.0.0".to_string(),
                        port: 8003,
                        enabled: Some(true),
                        description: Some("Analytics WebSocket Service".to_string()),
                    }),
                },
                execution: ExecutionServicesConfig {
                    main_engine: ServiceConfig {
                        host: "0.0.0.0".to_string(),
                        port: 8010,
                        enabled: Some(true),
                        description: Some("Main Trading Engine".to_string()),
                    },
                    http_api: ServiceConfig {
                        host: "0.0.0.0".to_string(),
                        port: 8011,
                        enabled: Some(true),
                        description: Some("Trading HTTP API".to_string()),
                    },
                    websocket: ServiceConfig {
                        host: "0.0.0.0".to_string(),
                        port: 8012,
                        enabled: Some(true),
                        description: Some("Trading WebSocket Service".to_string()),
                    },
                    grpc: ServiceConfig {
                        host: "0.0.0.0".to_string(),
                        port: 8013,
                        enabled: Some(true),
                        description: Some("Trading gRPC Service".to_string()),
                    },
                },
                monitoring: MonitoringServicesConfig {
                    prometheus: ServiceConfig {
                        host: "0.0.0.0".to_string(),
                        port: 9090,
                        enabled: Some(true),
                        description: Some("Prometheus Metrics".to_string()),
                    },
                    grafana: ServiceConfig {
                        host: "0.0.0.0".to_string(),
                        port: 9091,
                        enabled: Some(true),
                        description: Some("Grafana Dashboard".to_string()),
                    },
                    health_check: Some(ServiceConfig {
                        host: "0.0.0.0".to_string(),
                        port: 9092,
                        enabled: Some(true),
                        description: Some("Health Check Service".to_string()),
                    }),
                },
            },
            database: DatabaseConfig {
                postgresql: PostgresConfig {
                    host: "localhost".to_string(),
                    port: 5432,
                    database: "crypto_quant_platform".to_string(),
                    username: "trading_user".to_string(),
                    password: "trading_pass".to_string(),
                    max_connections: Some(20),
                    connection_timeout: Some(30),
                },
                redis: RedisConfig {
                    host: "localhost".to_string(),
                    port: 6379,
                    database: Some(0),
                    pool_size: Some(20),
                    timeout: Some(5),
                },
                mongodb: Some(MongoConfig {
                    host: "localhost".to_string(),
                    port: 27017,
                    database: "crypto_analytics".to_string(),
                    timeout: Some(5000),
                }),
            },
            message_queue: MessageQueueConfig {
                redis_url: "redis://localhost:6379/1".to_string(),
                channels: vec!["strategy_signals".to_string(), "market_data".to_string(), "execution_results".to_string()],
            },
            exchanges: std::collections::HashMap::new(),
            risk_management: RiskManagementConfig {
                enabled: true,
                max_daily_loss_ratio: 0.05,
                max_position_ratio: 0.8,
                max_single_order_ratio: 0.1,
                max_leverage: Some(3.0),
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                console_output: true,
                file_output: true,
                file_path: Some("logs/platform.log".to_string()),
                file_rotation: Some("daily".to_string()),
                json_format: Some(true),
                verbose_logging: Some(false),
            },
            security: SecurityConfig {
                enable_authentication: true,
                enable_authorization: Some(true),
                jwt_secret: "platform_jwt_secret_change_in_production".to_string(),
                cors_allowed_origins: vec!["http://localhost:3000".to_string(), "http://localhost:8002".to_string()],
            },
            environment: EnvironmentConfig {
                name: "development".to_string(),
                debug: true,
            },
        }
    }
}

/// 全局配置实例
static CONFIG_MANAGER: Lazy<std::sync::RwLock<Option<ConfigManager>>> =
    Lazy::new(|| std::sync::RwLock::new(None));

/// 初始化全局配置
pub fn init_config(config_path: Option<&str>) -> Result<()> {
    let manager = ConfigManager::new(config_path)?;
    manager.validate()?;
    
    let mut global_config = CONFIG_MANAGER.write().unwrap();
    *global_config = Some(manager);
    
    Ok(())
}

/// 获取全局配置
pub fn get_config() -> Result<PlatformConfig> {
    let config_lock = CONFIG_MANAGER.read().unwrap();
    match config_lock.as_ref() {
        Some(manager) => Ok(manager.get_config().clone()),
        None => Err(anyhow::anyhow!("Configuration not initialized")),
    }
}

/// 获取服务URL
pub fn get_service_url(service: &str) -> Result<String> {
    let config_lock = CONFIG_MANAGER.read().unwrap();
    match config_lock.as_ref() {
        Some(manager) => manager
            .get_service_url(service)
            .ok_or_else(|| anyhow::anyhow!("Service {} not found", service)),
        None => Err(anyhow::anyhow!("Configuration not initialized")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_parsing() {
        // Test individual components instead of full config
        let platform_info = PlatformInfo {
            name: "test-platform".to_string(),
            version: "1.0.0".to_string(),
            description: Some("Test platform".to_string()),
            environment: "test".to_string(),
        };
        
        let service_config = ServiceConfig {
            host: "127.0.0.1".to_string(),
            port: 8080,
            enabled: Some(true),
            description: Some("Test service".to_string()),
        };

        assert_eq!(platform_info.name, "test-platform");
        assert_eq!(platform_info.version, "1.0.0");
        assert_eq!(service_config.port, 8080);
        assert_eq!(service_config.host, "127.0.0.1");
    }

    #[test]
    fn test_service_url_generation() {
        // 这里可以添加更多测试
    }
}