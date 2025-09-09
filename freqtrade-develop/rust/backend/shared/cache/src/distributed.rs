use crate::{CacheError, CacheProvider, CacheResult, CacheConfig};
use async_trait::async_trait;
use redis::{AsyncCommands, Client, cmd};
use serde::{de::DeserializeOwned, Serialize};
use std::{collections::HashMap, sync::Arc, time::Duration as StdDuration};
use tracing::{debug, error, info, warn};

/// Redis-based distributed cache implementation
pub struct RedisCache {
    client: Arc<Client>,
    config: CacheConfig,
    connection_manager: redis::aio::ConnectionManager,
}

impl RedisCache {
    /// Create a new Redis cache instance with connection pooling
    pub async fn new(client: Client, config: CacheConfig) -> CacheResult<Self> {
        info!("Initializing Redis cache with URL: {}", config.redis_url);
        
        // Create connection manager for connection pooling
        let connection_manager = redis::aio::ConnectionManager::new(client.clone())
            .await
            .map_err(CacheError::Redis)?;
        
        let cache = Self {
            client: Arc::new(client),
            config,
            connection_manager,
        };
        
        // Test connection
        cache.test_connection().await?;
        info!("Redis cache initialized successfully");
        
        Ok(cache)
    }
    
    /// Test Redis connection
    async fn test_connection(&self) -> CacheResult<()> {
        let mut conn = self.client.get_multiplexed_async_connection().await.map_err(CacheError::Redis)?;
        let _: String = cmd("PING").query_async(&mut conn).await.map_err(CacheError::Redis)?;
        debug!("Redis connection test successful");
        Ok(())
    }
    
    /// Serialize value to bytes with optional compression
    fn serialize_value<T>(&self, value: &T) -> CacheResult<Vec<u8>>
    where
        T: Serialize,
    {
        let serialized = serde_json::to_vec(value)?;
        
        if self.config.enable_compression && serialized.len() > self.config.compression_threshold {
            // Use simple compression - in production you might want to use something like lz4 or zstd
            let compressed = self.compress_data(&serialized);
            debug!("Compressed data from {} to {} bytes", serialized.len(), compressed.len());
            Ok(compressed)
        } else {
            Ok(serialized)
        }
    }
    
    /// Deserialize value from bytes with optional decompression
    fn deserialize_value<T>(&self, data: Vec<u8>) -> CacheResult<T>
    where
        T: DeserializeOwned,
    {
        let decompressed = if self.config.enable_compression && self.is_compressed(&data) {
            self.decompress_data(&data)
        } else {
            data
        };
        
        serde_json::from_slice(&decompressed).map_err(CacheError::Serialization)
    }
    
    /// Simple compression using flate2 (in production, consider more efficient algorithms)
    fn compress_data(&self, data: &[u8]) -> Vec<u8> {
        use std::io::Write;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }
    
    /// Simple decompression
    fn decompress_data(&self, data: &[u8]) -> Vec<u8> {
        use std::io::Read;
        let mut decoder = flate2::read::GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).unwrap();
        decompressed
    }
    
    /// Check if data is compressed (simple heuristic)
    fn is_compressed(&self, data: &[u8]) -> bool {
        data.len() > 2 && data[0] == 0x1f && data[1] == 0x8b
    }
    
    /// Convert standard duration to seconds for Redis TTL
    fn duration_to_seconds(duration: StdDuration) -> u64 {
        duration.as_secs()
    }
}

#[async_trait]
impl CacheProvider for RedisCache {
    async fn get<T>(&self, key: &str) -> CacheResult<Option<T>>
    where
        T: DeserializeOwned + Send + Serialize + Sync + Clone,
    {
        let mut conn = self.client.get_multiplexed_async_connection().await.map_err(CacheError::Redis)?;
        
        match conn.get::<_, Vec<u8>>(key).await {
            Ok(data) => {
                debug!("Retrieved key '{}' from Redis ({} bytes)", key, data.len());
                match self.deserialize_value(data) {
                    Ok(value) => Ok(Some(value)),
                    Err(e) => {
                        error!("Failed to deserialize value for key '{}': {}", key, e);
                        Err(e)
                    }
                }
            }
            Err(e) if e.kind() == redis::ErrorKind::TypeError => {
                debug!("Key '{}' not found in Redis", key);
                Ok(None)
            }
            Err(e) => {
                error!("Redis get error for key '{}': {}", key, e);
                Err(CacheError::Redis(e))
            }
        }
    }

    async fn set<T>(&self, key: &str, value: &T, ttl: Option<StdDuration>) -> CacheResult<()>
    where
        T: Serialize + Send + Sync,
    {
        let mut conn = self.client.get_multiplexed_async_connection().await.map_err(CacheError::Redis)?;
        let serialized = self.serialize_value(value)?;
        
        match ttl {
            Some(duration) => {
                let seconds = Self::duration_to_seconds(duration);
                if seconds == 0 {
                    return Err(CacheError::InvalidTtl("TTL must be greater than 0".to_string()));
                }
                conn.set_ex::<_, _, ()>(key, serialized, seconds).await.map_err(CacheError::Redis)?;
                debug!("Set key '{}' in Redis with TTL {} seconds", key, seconds);
            }
            None => {
                conn.set::<_, _, ()>(key, serialized).await.map_err(CacheError::Redis)?;
                debug!("Set key '{}' in Redis without TTL", key);
            }
        }
        
        Ok(())
    }

    async fn delete(&self, key: &str) -> CacheResult<bool> {
        let mut conn = self.client.get_multiplexed_async_connection().await.map_err(CacheError::Redis)?;
        
        match conn.del::<_, i32>(key).await {
            Ok(deleted_count) => {
                let deleted = deleted_count > 0;
                debug!("Delete key '{}' from Redis: {}", key, deleted);
                Ok(deleted)
            }
            Err(e) => {
                error!("Redis delete error for key '{}': {}", key, e);
                Err(CacheError::Redis(e))
            }
        }
    }

    async fn exists(&self, key: &str) -> CacheResult<bool> {
        let mut conn = self.client.get_multiplexed_async_connection().await.map_err(CacheError::Redis)?;
        
        match conn.exists::<_, i32>(key).await {
            Ok(exists_count) => {
                let exists = exists_count > 0;
                debug!("Key '{}' exists in Redis: {}", key, exists);
                Ok(exists)
            }
            Err(e) => {
                error!("Redis exists error for key '{}': {}", key, e);
                Err(CacheError::Redis(e))
            }
        }
    }

    async fn expire(&self, key: &str, ttl: StdDuration) -> CacheResult<bool> {
        let mut conn = self.client.get_multiplexed_async_connection().await.map_err(CacheError::Redis)?;
        let seconds = Self::duration_to_seconds(ttl);
        
        if seconds == 0 {
            return Err(CacheError::InvalidTtl("TTL must be greater than 0".to_string()));
        }
        
        match conn.expire::<_, i32>(key, seconds as i64).await {
            Ok(result) => {
                let success = result == 1;
                debug!("Set TTL for key '{}' to {} seconds: {}", key, seconds, success);
                Ok(success)
            }
            Err(e) => {
                error!("Redis expire error for key '{}': {}", key, e);
                Err(CacheError::Redis(e))
            }
        }
    }

    async fn get_multi<T>(&self, keys: &[&str]) -> CacheResult<HashMap<String, T>>
    where
        T: DeserializeOwned + Send + Serialize + Sync + Clone,
    {
        if keys.is_empty() {
            return Ok(HashMap::new());
        }
        
        let mut conn = self.client.get_multiplexed_async_connection().await.map_err(CacheError::Redis)?;
        let mut result = HashMap::new();
        
        match conn.mget::<_, Vec<Option<Vec<u8>>>>(keys).await {
            Ok(values) => {
                for (i, value) in values.into_iter().enumerate() {
                    if let Some(data) = value {
                        let key = keys[i].to_string();
                        match self.deserialize_value(data) {
                            Ok(deserialized) => {
                                result.insert(key, deserialized);
                            }
                            Err(e) => {
                                warn!("Failed to deserialize value for key '{}': {}", keys[i], e);
                            }
                        }
                    }
                }
                debug!("Retrieved {} out of {} keys from Redis", result.len(), keys.len());
                Ok(result)
            }
            Err(e) => {
                error!("Redis mget error: {}", e);
                Err(CacheError::Redis(e))
            }
        }
    }

    async fn set_multi<T>(&self, items: &HashMap<&str, T>, ttl: Option<StdDuration>) -> CacheResult<()>
    where
        T: Serialize + Send + Sync,
    {
        if items.is_empty() {
            return Ok(());
        }
        
        let mut conn = self.client.get_multiplexed_async_connection().await.map_err(CacheError::Redis)?;
        let mut pipe = redis::pipe();
        
        for (key, value) in items {
            let serialized = self.serialize_value(value)?;
            match ttl {
                Some(duration) => {
                    let seconds = Self::duration_to_seconds(duration);
                    if seconds == 0 {
                        return Err(CacheError::InvalidTtl("TTL must be greater than 0".to_string()));
                    }
                    pipe.set_ex(*key, serialized, seconds);
                }
                None => {
                    pipe.set(*key, serialized);
                }
            }
        }
        
        match pipe.query_async(&mut conn).await {
            Ok(()) => {
                debug!("Set {} keys in Redis with pipeline", items.len());
                Ok(())
            }
            Err(e) => {
                error!("Redis pipeline set error: {}", e);
                Err(CacheError::Redis(e))
            }
        }
    }

    async fn flush(&self) -> CacheResult<()> {
        let mut conn = self.client.get_multiplexed_async_connection().await.map_err(CacheError::Redis)?;
        
        match cmd("FLUSHDB").query_async(&mut conn).await {
            Ok(()) => {
                warn!("Flushed all keys from Redis database");
                Ok(())
            }
            Err(e) => {
                error!("Redis flush error: {}", e);
                Err(CacheError::Redis(e))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    async fn create_test_cache() -> RedisCache {
        let config = CacheConfig {
            redis_url: "redis://127.0.0.1:6379".to_string(),
            enable_compression: true,
            compression_threshold: 10,
            ..Default::default()
        };
        
        let client = Client::open(config.redis_url.as_str()).unwrap();
        RedisCache::new(client, config).await.unwrap()
    }

    #[tokio::test]
    #[ignore] // Requires Redis server
    async fn test_redis_set_get() {
        let cache = create_test_cache().await;
        let key = "test:set_get";
        let value = "test_value".to_string();
        
        // Clean up first
        let _ = cache.delete(key).await;
        
        // Test set and get
        cache.set(key, &value, None).await.unwrap();
        let result: Option<String> = cache.get(key).await.unwrap();
        assert_eq!(result, Some(value));
        
        // Clean up
        cache.delete(key).await.unwrap();
    }

    #[tokio::test]
    #[ignore] // Requires Redis server
    async fn test_redis_ttl() {
        let cache = create_test_cache().await;
        let key = "test:ttl";
        let value = "test_value".to_string();
        
        // Clean up first
        let _ = cache.delete(key).await;
        
        // Set with short TTL
        cache.set(key, &value, Some(Duration::from_secs(1))).await.unwrap();
        
        // Should exist immediately
        assert!(cache.exists(key).await.unwrap());
        
        // Wait for expiration
        sleep(Duration::from_secs(2)).await;
        
        // Should not exist after expiration
        let result: Option<String> = cache.get(key).await.unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    #[ignore] // Requires Redis server
    async fn test_redis_multi_operations() {
        let cache = create_test_cache().await;
        
        let mut items = HashMap::new();
        items.insert("test:multi1", "value1".to_string());
        items.insert("test:multi2", "value2".to_string());
        items.insert("test:multi3", "value3".to_string());
        
        // Clean up first
        for key in items.keys() {
            let _ = cache.delete(key).await;
        }
        
        // Test multi set
        cache.set_multi(&items, None).await.unwrap();
        
        // Test multi get
        let keys: Vec<&str> = items.keys().copied().collect();
        let results: HashMap<String, String> = cache.get_multi(&keys).await.unwrap();
        
        assert_eq!(results.len(), 3);
        assert_eq!(results.get("test:multi1"), Some(&"value1".to_string()));
        assert_eq!(results.get("test:multi2"), Some(&"value2".to_string()));
        assert_eq!(results.get("test:multi3"), Some(&"value3".to_string()));
        
        // Clean up
        for key in items.keys() {
            cache.delete(key).await.unwrap();
        }
    }

    #[tokio::test]
    #[ignore] // Requires Redis server
    async fn test_redis_compression() {
        let cache = create_test_cache().await;
        let key = "test:compression";
        
        // Create a large value that should be compressed
        let large_value = "x".repeat(100);
        
        // Clean up first
        let _ = cache.delete(key).await;
        
        cache.set(key, &large_value, None).await.unwrap();
        let result: Option<String> = cache.get(key).await.unwrap();
        assert_eq!(result, Some(large_value));
        
        // Clean up
        cache.delete(key).await.unwrap();
    }
}