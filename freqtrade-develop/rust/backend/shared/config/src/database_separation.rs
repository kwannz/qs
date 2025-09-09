use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 分离的数据库配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparatedDatabaseConfig {
    /// 分析数据库 - 市场数据、指标、信号
    pub analytics: DatabaseInstanceConfig,
    /// 执行数据库 - 订单、交易、持仓
    pub execution: DatabaseInstanceConfig,
    /// 网关数据库 - 会话、认证、路由
    pub gateway: DatabaseInstanceConfig,
    /// 审计数据库 - 日志、合规、监管
    pub audit: DatabaseInstanceConfig,
    /// 共享数据库 - 配置、用户、权限
    pub shared: DatabaseInstanceConfig,
    /// Redis 分片配置
    pub redis_clusters: HashMap<String, RedisClusterConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseInstanceConfig {
    /// 主库配置
    pub primary: PostgresInstanceConfig,
    /// 只读副本配置 (可选)
    pub read_replicas: Vec<PostgresInstanceConfig>,
    /// 数据库特定配置
    pub settings: DatabaseSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostgresInstanceConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub username: String,
    pub password: String,
    pub ssl_mode: String,
    pub max_connections: u32,
    pub connection_timeout: u64,
    pub statement_timeout: Option<u64>,
    pub idle_timeout: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseSettings {
    /// 是否启用只读副本
    pub enable_read_replicas: bool,
    /// 读写分离策略
    pub read_write_split: ReadWriteSplitStrategy,
    /// 连接池设置
    pub pool_settings: PoolSettings,
    /// 备份配置
    pub backup_config: Option<BackupConfig>,
    /// 监控配置
    pub monitoring: MonitoringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReadWriteSplitStrategy {
    /// 所有读操作到副本
    ReadReplicasOnly,
    /// 负载均衡读操作
    LoadBalanced { ratio: f32 },
    /// 按查询类型分离
    QueryTypeBased { analytics_to_replica: bool },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolSettings {
    pub min_connections: u32,
    pub max_connections: u32,
    pub acquire_timeout: u64,
    pub max_lifetime: u64,
    pub idle_timeout: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    pub enabled: bool,
    pub schedule: String, // Cron format
    pub retention_days: u32,
    pub backup_location: String,
    pub compression_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_metrics: bool,
    pub slow_query_threshold: u64, // milliseconds
    pub connection_pool_metrics: bool,
    pub query_metrics: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisClusterConfig {
    pub nodes: Vec<RedisNodeConfig>,
    pub cluster_mode: bool,
    pub password: Option<String>,
    pub database: u32,
    pub pool_size: u32,
    pub timeout: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisNodeConfig {
    pub host: String,
    pub port: u16,
    pub role: RedisRole, // Master/Slave
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RedisRole {
    Master,
    Slave,
}

impl SeparatedDatabaseConfig {
    /// 获取服务对应的数据库配置
    pub fn get_database_for_service(&self, service_name: &str) -> Result<&DatabaseInstanceConfig> {
        match service_name {
            "analytics-api" | "analytics-worker" => Ok(&self.analytics),
            "execution-api" | "execution-engine" => Ok(&self.execution),
            "api-gateway" | "gateway" => Ok(&self.gateway),
            "audit-service" => Ok(&self.audit),
            _ => Ok(&self.shared),
        }
    }

    /// 获取 Redis 集群配置
    pub fn get_redis_cluster(&self, cluster_name: &str) -> Option<&RedisClusterConfig> {
        self.redis_clusters.get(cluster_name)
    }

    /// 验证配置有效性
    pub fn validate(&self) -> Result<()> {
        // 验证数据库连接配置
        self.validate_database_config(&self.analytics, "analytics")?;
        self.validate_database_config(&self.execution, "execution")?;
        self.validate_database_config(&self.gateway, "gateway")?;
        self.validate_database_config(&self.audit, "audit")?;
        self.validate_database_config(&self.shared, "shared")?;

        // 验证 Redis 配置
        for (name, cluster) in &self.redis_clusters {
            self.validate_redis_cluster(cluster, name)?;
        }

        Ok(())
    }

    fn validate_database_config(&self, config: &DatabaseInstanceConfig, name: &str) -> Result<()> {
        if config.primary.host.is_empty() {
            return Err(anyhow::anyhow!("{} database host cannot be empty", name));
        }
        
        if config.primary.database.is_empty() {
            return Err(anyhow::anyhow!("{} database name cannot be empty", name));
        }

        if config.primary.max_connections == 0 {
            return Err(anyhow::anyhow!("{} max_connections must be greater than 0", name));
        }

        Ok(())
    }

    fn validate_redis_cluster(&self, config: &RedisClusterConfig, name: &str) -> Result<()> {
        if config.nodes.is_empty() {
            return Err(anyhow::anyhow!("Redis cluster {} must have at least one node", name));
        }

        if config.cluster_mode {
            if config.nodes.len() < 3 {
                return Err(anyhow::anyhow!("Redis cluster {} requires at least 3 nodes", name));
            }
        }

        Ok(())
    }
}

/// 数据库连接管理器
pub struct DatabaseConnectionManager {
    config: SeparatedDatabaseConfig,
    connection_pools: HashMap<String, DatabasePool>,
}

#[derive(Debug)]
pub struct DatabasePool {
    primary_pool: sqlx::PgPool,
    read_replica_pools: Vec<sqlx::PgPool>,
    settings: DatabaseSettings,
}

impl DatabaseConnectionManager {
    pub fn new(config: SeparatedDatabaseConfig) -> Self {
        Self {
            config,
            connection_pools: HashMap::new(),
        }
    }

    /// 初始化所有数据库连接池
    pub async fn initialize(&mut self) -> Result<()> {
        // 初始化分析数据库连接
        let analytics_pool = self.create_database_pool(&self.config.analytics).await?;
        self.connection_pools.insert("analytics".to_string(), analytics_pool);

        // 初始化执行数据库连接
        let execution_pool = self.create_database_pool(&self.config.execution).await?;
        self.connection_pools.insert("execution".to_string(), execution_pool);

        // 初始化网关数据库连接
        let gateway_pool = self.create_database_pool(&self.config.gateway).await?;
        self.connection_pools.insert("gateway".to_string(), gateway_pool);

        // 初始化审计数据库连接
        let audit_pool = self.create_database_pool(&self.config.audit).await?;
        self.connection_pools.insert("audit".to_string(), audit_pool);

        // 初始化共享数据库连接
        let shared_pool = self.create_database_pool(&self.config.shared).await?;
        self.connection_pools.insert("shared".to_string(), shared_pool);

        Ok(())
    }

    async fn create_database_pool(&self, config: &DatabaseInstanceConfig) -> Result<DatabasePool> {
        // 创建主库连接池
        let primary_pool = self.create_postgres_pool(&config.primary).await?;

        // 创建只读副本连接池
        let mut read_replica_pools = Vec::new();
        if config.settings.enable_read_replicas {
            for replica_config in &config.read_replicas {
                let replica_pool = self.create_postgres_pool(replica_config).await?;
                read_replica_pools.push(replica_pool);
            }
        }

        Ok(DatabasePool {
            primary_pool,
            read_replica_pools,
            settings: config.settings.clone(),
        })
    }

    async fn create_postgres_pool(&self, config: &PostgresInstanceConfig) -> Result<sqlx::PgPool> {
        let database_url = format!(
            "postgresql://{}:{}@{}:{}/{}?sslmode={}",
            config.username,
            config.password,
            config.host,
            config.port,
            config.database,
            config.ssl_mode
        );

        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(config.max_connections)
            .acquire_timeout(std::time::Duration::from_secs(config.connection_timeout))
            .connect(&database_url)
            .await?;

        Ok(pool)
    }

    /// 获取指定服务的数据库连接池
    pub fn get_pool(&self, service_name: &str) -> Result<&DatabasePool> {
        let pool_name = match service_name {
            "analytics-api" | "analytics-worker" => "analytics",
            "execution-api" | "execution-engine" => "execution", 
            "api-gateway" | "gateway" => "gateway",
            "audit-service" => "audit",
            _ => "shared",
        };

        self.connection_pools
            .get(pool_name)
            .ok_or_else(|| anyhow::anyhow!("Database pool not found for service: {}", service_name))
    }

    /// 获取读连接池 (读写分离)
    pub fn get_read_pool(&self, service_name: &str) -> Result<&sqlx::PgPool> {
        let pool = self.get_pool(service_name)?;
        
        // 根据策略选择连接池
        match pool.settings.read_write_split {
            ReadWriteSplitStrategy::ReadReplicasOnly => {
                if !pool.read_replica_pools.is_empty() {
                    // 简单轮询选择副本
                    let index = fastrand::usize(0..pool.read_replica_pools.len());
                    Ok(&pool.read_replica_pools[index])
                } else {
                    Ok(&pool.primary_pool)
                }
            }
            ReadWriteSplitStrategy::LoadBalanced { ratio } => {
                if !pool.read_replica_pools.is_empty() && fastrand::f32() < ratio {
                    let index = fastrand::usize(0..pool.read_replica_pools.len());
                    Ok(&pool.read_replica_pools[index])
                } else {
                    Ok(&pool.primary_pool)
                }
            }
            ReadWriteSplitStrategy::QueryTypeBased { .. } => {
                // 这里可以基于查询类型进一步优化
                Ok(&pool.primary_pool)
            }
        }
    }

    /// 获取写连接池
    pub fn get_write_pool(&self, service_name: &str) -> Result<&sqlx::PgPool> {
        let pool = self.get_pool(service_name)?;
        Ok(&pool.primary_pool)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_database_mapping() {
        let config = create_test_config();
        
        assert!(config.get_database_for_service("analytics-api").is_ok());
        assert!(config.get_database_for_service("execution-api").is_ok());
        assert!(config.get_database_for_service("api-gateway").is_ok());
        assert!(config.get_database_for_service("unknown-service").is_ok()); // fallback to shared
    }

    #[test]
    fn test_config_validation() {
        let config = create_test_config();
        assert!(config.validate().is_ok());
    }

    fn create_test_config() -> SeparatedDatabaseConfig {
        let postgres_config = PostgresInstanceConfig {
            host: "localhost".to_string(),
            port: 5432,
            database: "test_db".to_string(),
            username: "test_user".to_string(),
            password: "test_pass".to_string(),
            ssl_mode: "prefer".to_string(),
            max_connections: 10,
            connection_timeout: 30,
            statement_timeout: Some(30000),
            idle_timeout: Some(600),
        };

        let db_settings = DatabaseSettings {
            enable_read_replicas: false,
            read_write_split: ReadWriteSplitStrategy::ReadReplicasOnly,
            pool_settings: PoolSettings {
                min_connections: 1,
                max_connections: 10,
                acquire_timeout: 30,
                max_lifetime: 1800,
                idle_timeout: 600,
            },
            backup_config: None,
            monitoring: MonitoringConfig {
                enable_metrics: true,
                slow_query_threshold: 1000,
                connection_pool_metrics: true,
                query_metrics: false,
            },
        };

        let db_instance = DatabaseInstanceConfig {
            primary: postgres_config.clone(),
            read_replicas: vec![],
            settings: db_settings,
        };

        SeparatedDatabaseConfig {
            analytics: db_instance.clone(),
            execution: db_instance.clone(),
            gateway: db_instance.clone(),
            audit: db_instance.clone(),
            shared: db_instance,
            redis_clusters: HashMap::new(),
        }
    }
}