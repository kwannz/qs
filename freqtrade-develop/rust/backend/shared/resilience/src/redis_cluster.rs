use anyhow::Result;
use redis::{cluster::ClusterClient, AsyncCommands};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, error, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisClusterConfig {
    pub nodes: Vec<String>,
    pub password: Option<String>,
    pub connection_timeout: Duration,
    pub response_timeout: Duration,
    pub max_retries: u32,
    pub retry_delay: Duration,
    pub readonly_replicas: bool,
    pub cluster_name: String,
}

impl Default for RedisClusterConfig {
    fn default() -> Self {
        Self {
            nodes: vec![
                "redis://redis-node-1:7001".to_string(),
                "redis://redis-node-2:7002".to_string(),
                "redis://redis-node-3:7003".to_string(),
            ],
            password: Some("trading_redis_pass".to_string()),
            connection_timeout: Duration::from_secs(5),
            response_timeout: Duration::from_secs(3),
            max_retries: 3,
            retry_delay: Duration::from_millis(100),
            readonly_replicas: true,
            cluster_name: "crypto-quant-cluster".to_string(),
        }
    }
}

pub struct RedisClusterManager {
    client: ClusterClient,
    config: RedisClusterConfig,
}

impl RedisClusterManager {
    pub fn new(config: RedisClusterConfig) -> Result<Self> {
        let mut client_builder = ClusterClient::builder(config.nodes.clone());
        
        if let Some(ref password) = config.password {
            client_builder = client_builder.password(password.clone());
        }
        
        client_builder = client_builder
            .connection_timeout(config.connection_timeout)
            .response_timeout(config.response_timeout)
            .read_from_replicas();

        let client = client_builder.build()?;

        Ok(Self { client, config })
    }

    pub async fn get_connection(&self) -> Result<redis::cluster_async::ClusterConnection> {
        let connection = timeout(
            self.config.connection_timeout,
            self.client.get_async_connection(),
        )
        .await??;

        Ok(connection)
    }

    pub async fn set_with_expiry(
        &self,
        key: &str,
        value: &str,
        expiry: Duration,
    ) -> Result<bool> {
        let mut retries = 0;
        
        while retries < self.config.max_retries {
            match self.try_set_with_expiry(key, value, expiry).await {
                Ok(result) => {
                    debug!(
                        key = key,
                        expiry_secs = expiry.as_secs(),
                        "Successfully set key in Redis cluster"
                    );
                    return Ok(result);
                }
                Err(e) => {
                    retries += 1;
                    warn!(
                        key = key,
                        retry = retries,
                        max_retries = self.config.max_retries,
                        error = %e,
                        "Failed to set key in Redis cluster, retrying..."
                    );
                    
                    if retries < self.config.max_retries {
                        tokio::time::sleep(self.config.retry_delay).await;
                    }
                }
            }
        }
        
        error!(
            key = key,
            max_retries = self.config.max_retries,
            "Failed to set key after all retries"
        );
        
        Err(anyhow::anyhow!(
            "Failed to set key '{}' after {} retries",
            key,
            self.config.max_retries
        ))
    }

    async fn try_set_with_expiry(
        &self,
        key: &str,
        value: &str,
        expiry: Duration,
    ) -> Result<bool> {
        let mut conn = self.get_connection().await?;
        
        let result: String = timeout(
            self.config.response_timeout,
            conn.set_ex(key, value, expiry.as_secs()),
        )
        .await??;

        Ok(result == "OK")
    }

    pub async fn get(&self, key: &str) -> Result<Option<String>> {
        let mut retries = 0;
        
        while retries < self.config.max_retries {
            match self.try_get(key).await {
                Ok(result) => {
                    debug!(key = key, found = result.is_some(), "Retrieved key from Redis cluster");
                    return Ok(result);
                }
                Err(e) => {
                    retries += 1;
                    warn!(
                        key = key,
                        retry = retries,
                        max_retries = self.config.max_retries,
                        error = %e,
                        "Failed to get key from Redis cluster, retrying..."
                    );
                    
                    if retries < self.config.max_retries {
                        tokio::time::sleep(self.config.retry_delay).await;
                    }
                }
            }
        }
        
        error!(
            key = key,
            max_retries = self.config.max_retries,
            "Failed to get key after all retries"
        );
        
        Err(anyhow::anyhow!(
            "Failed to get key '{}' after {} retries",
            key,
            self.config.max_retries
        ))
    }

    async fn try_get(&self, key: &str) -> Result<Option<String>> {
        let mut conn = self.get_connection().await?;
        
        let result: Option<String> = timeout(
            self.config.response_timeout,
            conn.get(key),
        )
        .await??;

        Ok(result)
    }

    pub async fn delete(&self, key: &str) -> Result<bool> {
        let mut retries = 0;
        
        while retries < self.config.max_retries {
            match self.try_delete(key).await {
                Ok(result) => {
                    debug!(key = key, deleted = result, "Deleted key from Redis cluster");
                    return Ok(result);
                }
                Err(e) => {
                    retries += 1;
                    warn!(
                        key = key,
                        retry = retries,
                        max_retries = self.config.max_retries,
                        error = %e,
                        "Failed to delete key from Redis cluster, retrying..."
                    );
                    
                    if retries < self.config.max_retries {
                        tokio::time::sleep(self.config.retry_delay).await;
                    }
                }
            }
        }
        
        Err(anyhow::anyhow!(
            "Failed to delete key '{}' after {} retries",
            key,
            self.config.max_retries
        ))
    }

    async fn try_delete(&self, key: &str) -> Result<bool> {
        let mut conn = self.get_connection().await?;
        
        let result: u32 = timeout(
            self.config.response_timeout,
            conn.del(key),
        )
        .await??;

        Ok(result > 0)
    }

    pub async fn exists(&self, key: &str) -> Result<bool> {
        let mut conn = self.get_connection().await?;
        
        let result: bool = timeout(
            self.config.response_timeout,
            conn.exists(key),
        )
        .await??;

        Ok(result)
    }

    pub async fn increment(&self, key: &str) -> Result<i64> {
        let mut conn = self.get_connection().await?;
        
        let result: i64 = timeout(
            self.config.response_timeout,
            conn.incr(key, 1),
        )
        .await??;

        Ok(result)
    }

    pub async fn set_hash_field(
        &self,
        hash_key: &str,
        field: &str,
        value: &str,
    ) -> Result<bool> {
        let mut conn = self.get_connection().await?;
        
        let result: u32 = timeout(
            self.config.response_timeout,
            conn.hset(hash_key, field, value),
        )
        .await??;

        Ok(result > 0)
    }

    pub async fn get_hash_field(&self, hash_key: &str, field: &str) -> Result<Option<String>> {
        let mut conn = self.get_connection().await?;
        
        let result: Option<String> = timeout(
            self.config.response_timeout,
            conn.hget(hash_key, field),
        )
        .await??;

        Ok(result)
    }

    pub async fn get_all_hash_fields(
        &self,
        hash_key: &str,
    ) -> Result<std::collections::HashMap<String, String>> {
        let mut conn = self.get_connection().await?;
        
        let result: std::collections::HashMap<String, String> = timeout(
            self.config.response_timeout,
            conn.hgetall(hash_key),
        )
        .await??;

        Ok(result)
    }

    pub async fn push_to_list(&self, list_key: &str, value: &str) -> Result<u32> {
        let mut conn = self.get_connection().await?;
        
        let result: u32 = timeout(
            self.config.response_timeout,
            conn.lpush(list_key, value),
        )
        .await??;

        Ok(result)
    }

    pub async fn pop_from_list(&self, list_key: &str) -> Result<Option<String>> {
        let mut conn = self.get_connection().await?;
        
        let result: Option<String> = timeout(
            self.config.response_timeout,
            conn.rpop(list_key, None),
        )
        .await??;

        Ok(result)
    }

    pub async fn get_cluster_info(&self) -> Result<String> {
        let mut conn = self.get_connection().await?;
        
        let result: String = timeout(
            self.config.response_timeout,
            redis::cmd("CLUSTER").arg("INFO").query_async(&mut conn),
        )
        .await??;

        Ok(result)
    }

    pub async fn get_cluster_nodes(&self) -> Result<String> {
        let mut conn = self.get_connection().await?;
        
        let result: String = timeout(
            self.config.response_timeout,
            redis::cmd("CLUSTER").arg("NODES").query_async(&mut conn),
        )
        .await??;

        Ok(result)
    }

    pub async fn health_check(&self) -> Result<bool> {
        match timeout(
            self.config.response_timeout,
            self.client.get_async_connection(),
        )
        .await
        {
            Ok(Ok(mut conn)) => {
                match timeout(
                    Duration::from_secs(1),
                    redis::cmd("PING").query_async::<String>(&mut conn),
                )
                .await
                {
                    Ok(Ok(response)) => Ok(response == "PONG"),
                    Ok(Err(_)) => Ok(false),
                    Err(_) => Ok(false),
                }
            }
            Ok(Err(_)) => Ok(false),
            Err(_) => Ok(false),
        }
    }
}

// Session management using Redis cluster
pub struct RedisSessionStore {
    redis: RedisClusterManager,
    session_prefix: String,
    default_expiry: Duration,
}

impl RedisSessionStore {
    pub fn new(redis: RedisClusterManager) -> Self {
        Self {
            redis,
            session_prefix: "session:".to_string(),
            default_expiry: Duration::from_secs(3600), // 1 hour
        }
    }

    pub fn with_prefix_and_expiry(
        redis: RedisClusterManager,
        prefix: String,
        expiry: Duration,
    ) -> Self {
        Self {
            redis,
            session_prefix: prefix,
            default_expiry: expiry,
        }
    }

    pub async fn create_session(
        &self,
        session_id: &str,
        session_data: &str,
    ) -> Result<bool> {
        let key = format!("{}{}", self.session_prefix, session_id);
        self.redis.set_with_expiry(&key, session_data, self.default_expiry).await
    }

    pub async fn get_session(&self, session_id: &str) -> Result<Option<String>> {
        let key = format!("{}{}", self.session_prefix, session_id);
        self.redis.get(&key).await
    }

    pub async fn update_session(
        &self,
        session_id: &str,
        session_data: &str,
    ) -> Result<bool> {
        let key = format!("{}{}", self.session_prefix, session_id);
        self.redis.set_with_expiry(&key, session_data, self.default_expiry).await
    }

    pub async fn delete_session(&self, session_id: &str) -> Result<bool> {
        let key = format!("{}{}", self.session_prefix, session_id);
        self.redis.delete(&key).await
    }

    pub async fn extend_session(&self, session_id: &str) -> Result<bool> {
        let key = format!("{}{}", self.session_prefix, session_id);
        
        if let Some(session_data) = self.redis.get(&key).await? {
            self.redis.set_with_expiry(&key, &session_data, self.default_expiry).await
        } else {
            Ok(false)
        }
    }

    pub async fn session_exists(&self, session_id: &str) -> Result<bool> {
        let key = format!("{}{}", self.session_prefix, session_id);
        self.redis.exists(&key).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_redis_cluster_config() {
        let config = RedisClusterConfig::default();
        assert_eq!(config.nodes.len(), 3);
        assert!(config.password.is_some());
        assert_eq!(config.cluster_name, "crypto-quant-cluster");
    }

    #[tokio::test]
    async fn test_session_store() {
        let config = RedisClusterConfig {
            nodes: vec!["redis://localhost:7001".to_string()],
            password: None,
            ..Default::default()
        };

        if let Ok(redis) = RedisClusterManager::new(config) {
            let store = RedisSessionStore::new(redis);
            let session_id = "test-session-123";
            let session_data = "test-data";

            // This test will only pass if Redis cluster is running
            if store.redis.health_check().await.unwrap_or(false) {
                let result = store.create_session(session_id, session_data).await;
                assert!(result.is_ok());

                let retrieved = store.get_session(session_id).await;
                assert!(retrieved.is_ok());
                assert_eq!(retrieved.unwrap(), Some(session_data.to_string()));

                let deleted = store.delete_session(session_id).await;
                assert!(deleted.is_ok());
            }
        }
    }
}