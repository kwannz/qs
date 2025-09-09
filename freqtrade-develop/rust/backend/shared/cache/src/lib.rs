use anyhow::Result;
use redis::Client;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc, time::Duration as StdDuration};
use thiserror::Error;
use tracing::{debug, error, warn};

pub mod distributed;
pub mod local;
pub mod layered;
pub mod metrics;

#[derive(Error, Debug)]
pub enum CacheError {
    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Key not found: {0}")]
    KeyNotFound(String),
    #[error("Cache operation timeout")]
    Timeout,
    #[error("Invalid TTL: {0}")]
    InvalidTtl(String),
}

pub type CacheResult<T> = Result<T, CacheError>;

/// Cache trait for different caching strategies
#[async_trait::async_trait]
pub trait CacheProvider: Send + Sync {
    /// Get a value from cache
    async fn get<T>(&self, key: &str) -> CacheResult<Option<T>>
    where
        T: DeserializeOwned + Send + Serialize + Sync + Clone;

    /// Set a value in cache with TTL
    async fn set<T>(&self, key: &str, value: &T, ttl: Option<StdDuration>) -> CacheResult<()>
    where
        T: Serialize + Send + Sync;

    /// Delete a key from cache
    async fn delete(&self, key: &str) -> CacheResult<bool>;

    /// Check if key exists
    async fn exists(&self, key: &str) -> CacheResult<bool>;

    /// Set TTL for existing key
    async fn expire(&self, key: &str, ttl: StdDuration) -> CacheResult<bool>;

    /// Get multiple keys at once
    async fn get_multi<T>(&self, keys: &[&str]) -> CacheResult<HashMap<String, T>>
    where
        T: DeserializeOwned + Send + Serialize + Sync + Clone;

    /// Set multiple keys at once
    async fn set_multi<T>(&self, items: &HashMap<&str, T>, ttl: Option<StdDuration>) -> CacheResult<()>
    where
        T: Serialize + Send + Sync;

    /// Flush all cache entries
    async fn flush(&self) -> CacheResult<()>;
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub redis_url: String,
    pub max_connections: u32,
    pub connection_timeout: StdDuration,
    pub local_cache_size: u64,
    pub local_cache_ttl: StdDuration,
    pub enable_compression: bool,
    pub compression_threshold: usize,
    pub key_prefix: Option<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            redis_url: "redis://localhost:6379".to_string(),
            max_connections: 10,
            connection_timeout: StdDuration::from_secs(5),
            local_cache_size: 1000,
            local_cache_ttl: StdDuration::from_secs(300), // 5 minutes
            enable_compression: true,
            compression_threshold: 1024, // 1KB
            key_prefix: None,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub total_operations: u64,
    pub errors: u64,
    pub evictions: u64,
    pub memory_usage: u64,
}

/// Cache provider types
pub enum CacheProviderType {
    Redis(distributed::RedisCache),
    Memory(local::MemoryCache),
    Layered(Box<layered::LayeredCache>),
    Metrics(Box<metrics::MetricsCache>),
}

#[async_trait::async_trait]
impl CacheProvider for CacheProviderType {
    async fn get<T>(&self, key: &str) -> CacheResult<Option<T>>
    where
        T: DeserializeOwned + Send + Serialize + Sync + Clone,
    {
        match self {
            CacheProviderType::Redis(cache) => cache.get(key).await,
            CacheProviderType::Memory(cache) => cache.get(key).await,
            CacheProviderType::Layered(cache) => cache.get(key).await,
            CacheProviderType::Metrics(cache) => cache.get(key).await,
        }
    }

    async fn set<T>(&self, key: &str, value: &T, ttl: Option<StdDuration>) -> CacheResult<()>
    where
        T: Serialize + Send + Sync,
    {
        match self {
            CacheProviderType::Redis(cache) => cache.set(key, value, ttl).await,
            CacheProviderType::Memory(cache) => cache.set(key, value, ttl).await,
            CacheProviderType::Layered(cache) => cache.set(key, value, ttl).await,
            CacheProviderType::Metrics(cache) => cache.set(key, value, ttl).await,
        }
    }

    async fn delete(&self, key: &str) -> CacheResult<bool> {
        match self {
            CacheProviderType::Redis(cache) => cache.delete(key).await,
            CacheProviderType::Memory(cache) => cache.delete(key).await,
            CacheProviderType::Layered(cache) => cache.delete(key).await,
            CacheProviderType::Metrics(cache) => cache.delete(key).await,
        }
    }

    async fn exists(&self, key: &str) -> CacheResult<bool> {
        match self {
            CacheProviderType::Redis(cache) => cache.exists(key).await,
            CacheProviderType::Memory(cache) => cache.exists(key).await,
            CacheProviderType::Layered(cache) => cache.exists(key).await,
            CacheProviderType::Metrics(cache) => cache.exists(key).await,
        }
    }

    async fn expire(&self, key: &str, ttl: StdDuration) -> CacheResult<bool> {
        match self {
            CacheProviderType::Redis(cache) => cache.expire(key, ttl).await,
            CacheProviderType::Memory(cache) => cache.expire(key, ttl).await,
            CacheProviderType::Layered(cache) => cache.expire(key, ttl).await,
            CacheProviderType::Metrics(cache) => cache.expire(key, ttl).await,
        }
    }

    async fn get_multi<T>(&self, keys: &[&str]) -> CacheResult<HashMap<String, T>>
    where
        T: DeserializeOwned + Send + Serialize + Sync + Clone,
    {
        match self {
            CacheProviderType::Redis(cache) => cache.get_multi(keys).await,
            CacheProviderType::Memory(cache) => cache.get_multi(keys).await,
            CacheProviderType::Layered(cache) => cache.get_multi(keys).await,
            CacheProviderType::Metrics(cache) => cache.get_multi(keys).await,
        }
    }

    async fn set_multi<T>(&self, items: &HashMap<&str, T>, ttl: Option<StdDuration>) -> CacheResult<()>
    where
        T: Serialize + Send + Sync,
    {
        match self {
            CacheProviderType::Redis(cache) => cache.set_multi(items, ttl).await,
            CacheProviderType::Memory(cache) => cache.set_multi(items, ttl).await,
            CacheProviderType::Layered(cache) => cache.set_multi(items, ttl).await,
            CacheProviderType::Metrics(cache) => cache.set_multi(items, ttl).await,
        }
    }

    async fn flush(&self) -> CacheResult<()> {
        match self {
            CacheProviderType::Redis(cache) => cache.flush().await,
            CacheProviderType::Memory(cache) => cache.flush().await,
            CacheProviderType::Layered(cache) => cache.flush().await,
            CacheProviderType::Metrics(cache) => cache.flush().await,
        }
    }
}

/// Cache manager that coordinates different cache layers
pub struct CacheManager {
    distributed: Option<CacheProviderType>,
    local: Option<CacheProviderType>,
    config: CacheConfig,
    stats: Arc<tokio::sync::Mutex<CacheStats>>,
}

impl CacheManager {
    /// Create a new cache manager with Redis distributed cache
    pub async fn new_with_redis(config: CacheConfig) -> Result<Self> {
        let redis_client = Client::open(config.redis_url.as_str())?;
        let distributed = CacheProviderType::Redis(distributed::RedisCache::new(redis_client, config.clone()).await?);
        
        let local = if config.local_cache_size > 0 {
            Some(CacheProviderType::Memory(local::MemoryCache::new(config.clone())))
        } else {
            None
        };

        Ok(Self {
            distributed: Some(distributed),
            local,
            config,
            stats: Arc::new(tokio::sync::Mutex::new(CacheStats::default())),
        })
    }

    /// Create a new cache manager with only local cache
    pub fn new_local_only(config: CacheConfig) -> Self {
        let local = CacheProviderType::Memory(local::MemoryCache::new(config.clone()));
        
        Self {
            distributed: None,
            local: Some(local),
            config,
            stats: Arc::new(tokio::sync::Mutex::new(CacheStats::default())),
        }
    }

    /// Get value from cache (tries local first, then distributed)
    pub async fn get<T>(&self, key: &str) -> CacheResult<Option<T>>
    where
        T: DeserializeOwned + Serialize + Send + Sync + Clone,
    {
        let full_key = self.build_key(key);
        
        // Try local cache first
        if let Some(ref local) = self.local {
            match local.get(&full_key).await {
                Ok(Some(value)) => {
                    self.record_hit().await;
                    debug!("Cache hit (local): {}", key);
                    return Ok(Some(value));
                }
                Ok(None) => {
                    debug!("Cache miss (local): {}", key);
                }
                Err(e) => {
                    warn!("Local cache error for key {}: {}", key, e);
                }
            }
        }

        // Try distributed cache
        if let Some(ref distributed) = self.distributed {
            match distributed.get(&full_key).await {
                Ok(Some(value)) => {
                    self.record_hit().await;
                    debug!("Cache hit (distributed): {}", key);
                    
                    // Populate local cache
                    if let Some(ref local) = self.local {
                        if let Err(e) = local.set(&full_key, &value, Some(self.config.local_cache_ttl)).await {
                            warn!("Failed to populate local cache for key {}: {}", key, e);
                        }
                    }
                    
                    return Ok(Some(value));
                }
                Ok(None) => {
                    debug!("Cache miss (distributed): {}", key);
                    self.record_miss().await;
                    return Ok(None);
                }
                Err(e) => {
                    error!("Distributed cache error for key {}: {}", key, e);
                    self.record_error().await;
                }
            }
        }

        self.record_miss().await;
        Ok(None)
    }

    /// Set value in cache (writes to both local and distributed)
    pub async fn set<T>(&self, key: &str, value: &T, ttl: Option<StdDuration>) -> CacheResult<()>
    where
        T: Serialize + Send + Sync + Clone,
    {
        let full_key = self.build_key(key);
        let mut errors = Vec::new();

        // Write to local cache
        if let Some(ref local) = self.local {
            if let Err(e) = local.set(&full_key, value, ttl.or(Some(self.config.local_cache_ttl))).await {
                warn!("Failed to set local cache for key {}: {}", key, e);
                errors.push(e);
            }
        }

        // Write to distributed cache
        if let Some(ref distributed) = self.distributed {
            if let Err(e) = distributed.set(&full_key, value, ttl).await {
                error!("Failed to set distributed cache for key {}: {}", key, e);
                self.record_error().await;
                errors.push(e);
            }
        }

        if errors.is_empty() {
            debug!("Cache set: {}", key);
            Ok(())
        } else {
            // If distributed cache failed but local succeeded, still return error
            // but the operation partially succeeded
            Err(errors.into_iter().next().unwrap())
        }
    }

    /// Delete key from all cache layers
    pub async fn delete(&self, key: &str) -> CacheResult<bool> {
        let full_key = self.build_key(key);
        let mut deleted = false;

        if let Some(ref local) = self.local {
            if let Ok(result) = local.delete(&full_key).await {
                deleted |= result;
            }
        }

        if let Some(ref distributed) = self.distributed {
            if let Ok(result) = distributed.delete(&full_key).await {
                deleted |= result;
            }
        }

        debug!("Cache delete: {} (deleted: {})", key, deleted);
        Ok(deleted)
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        let stats = self.stats.lock().await;
        let mut result = stats.clone();
        
        if result.total_operations > 0 {
            result.hit_rate = (result.hits as f64) / (result.total_operations as f64) * 100.0;
        }
        
        result
    }

    /// Reset cache statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.lock().await;
        *stats = CacheStats::default();
    }

    /// Build full cache key with prefix
    fn build_key(&self, key: &str) -> String {
        match &self.config.key_prefix {
            Some(prefix) => format!("{prefix}:{key}"),
            None => key.to_string(),
        }
    }

    async fn record_hit(&self) {
        let mut stats = self.stats.lock().await;
        stats.hits += 1;
        stats.total_operations += 1;
    }

    async fn record_miss(&self) {
        let mut stats = self.stats.lock().await;
        stats.misses += 1;
        stats.total_operations += 1;
    }

    async fn record_error(&self) {
        let mut stats = self.stats.lock().await;
        stats.errors += 1;
        stats.total_operations += 1;
    }
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            hits: 0,
            misses: 0,
            hit_rate: 0.0,
            total_operations: 0,
            errors: 0,
            evictions: 0,
            memory_usage: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_local_only_cache() {
        let config = CacheConfig {
            local_cache_size: 100,
            ..Default::default()
        };
        
        let cache = CacheManager::new_local_only(config);
        
        // Test set and get
        let key = "test_key";
        let value = "test_value".to_string();
        
        assert!(cache.set(key, &value, None).await.is_ok());
        
        let retrieved: Option<String> = cache.get(key).await.unwrap();
        assert_eq!(retrieved, Some(value));
        
        // Test cache miss
        let missing: Option<String> = cache.get("nonexistent").await.unwrap();
        assert_eq!(missing, None);
        
        // Test delete
        assert!(cache.delete(key).await.unwrap());
        let deleted: Option<String> = cache.get(key).await.unwrap();
        assert_eq!(deleted, None);
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let config = CacheConfig {
            local_cache_size: 100,
            ..Default::default()
        };
        
        let cache = CacheManager::new_local_only(config);
        
        // Generate some cache activity
        cache.set("key1", &"value1".to_string(), None).await.unwrap();
        let _: Option<String> = cache.get("key1").await.unwrap(); // hit
        let _: Option<String> = cache.get("key2").await.unwrap(); // miss
        
        let stats = cache.get_stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.total_operations, 2);
        assert!((stats.hit_rate - 50.0).abs() < 0.01);
    }
}