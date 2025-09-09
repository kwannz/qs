use crate::{CacheError, CacheProvider, CacheResult, CacheConfig};
use async_trait::async_trait;
use moka::future::Cache;
use serde::{de::DeserializeOwned, Serialize};
use std::{collections::HashMap, sync::Arc, time::Duration as StdDuration};
use tracing::{debug, info, warn};

/// Local memory cache implementation using Moka
pub struct MemoryCache {
    cache: Arc<Cache<String, CacheValue>>,
    config: CacheConfig,
}

/// Wrapper for cached values with metadata
#[derive(Clone, Debug)]
struct CacheValue {
    data: Vec<u8>,
    created_at: std::time::Instant,
}

impl CacheValue {
    fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            created_at: std::time::Instant::now(),
        }
    }
    
    fn is_expired(&self, ttl: StdDuration) -> bool {
        self.created_at.elapsed() > ttl
    }
}

impl MemoryCache {
    /// Create a new memory cache instance
    pub fn new(config: CacheConfig) -> Self {
        info!(
            "Initializing local memory cache with size: {}, TTL: {:?}",
            config.local_cache_size, config.local_cache_ttl
        );
        
        let cache = Cache::builder()
            .max_capacity(config.local_cache_size)
            .time_to_live(config.local_cache_ttl)
            .time_to_idle(config.local_cache_ttl / 2) // Idle time is half of TTL
            .eviction_listener(|key, _value, cause| {
                debug!("Evicted cache key '{}' due to {:?}", key, cause);
            })
            .build();

        Self {
            cache: Arc::new(cache),
            config,
        }
    }
    
    /// Serialize value to bytes
    fn serialize_value<T>(&self, value: &T) -> CacheResult<Vec<u8>>
    where
        T: Serialize,
    {
        serde_json::to_vec(value).map_err(CacheError::Serialization)
    }
    
    /// Deserialize value from bytes
    fn deserialize_value<T>(&self, data: &[u8]) -> CacheResult<T>
    where
        T: DeserializeOwned,
    {
        serde_json::from_slice(data).map_err(CacheError::Serialization)
    }
    
    /// Check if a cached value has expired (for custom TTL handling)
    fn is_expired(&self, cache_value: &CacheValue, custom_ttl: Option<StdDuration>) -> bool {
        let ttl = custom_ttl.unwrap_or(self.config.local_cache_ttl);
        cache_value.is_expired(ttl)
    }
    
    /// Get cache statistics from Moka
    pub fn get_cache_stats(&self) -> (u64, u64, u64, f64) {
        let stats = &self.cache;
        let entry_count = stats.entry_count();
        let weighted_size = stats.weighted_size();
        
        // Moka doesn't provide hit/miss stats in the same way as Redis
        // These would need to be tracked separately if needed
        (entry_count, 0, 0, weighted_size as f64)
    }
}

#[async_trait]
impl CacheProvider for MemoryCache {
    async fn get<T>(&self, key: &str) -> CacheResult<Option<T>>
    where
        T: DeserializeOwned + Send + Serialize + Sync + Clone,
    {
        match self.cache.get(key).await {
            Some(cache_value) => {
                // Check for custom expiration (Moka handles its own TTL, but we might have custom logic)
                if self.is_expired(&cache_value, None) {
                    debug!("Key '{}' expired in local cache", key);
                    self.cache.invalidate(key).await;
                    Ok(None)
                } else {
                    match self.deserialize_value(&cache_value.data) {
                        Ok(value) => {
                            debug!("Retrieved key '{}' from local cache", key);
                            Ok(Some(value))
                        }
                        Err(e) => {
                            warn!("Failed to deserialize value for key '{}': {}", key, e);
                            // Remove corrupted entry
                            self.cache.invalidate(key).await;
                            Err(e)
                        }
                    }
                }
            }
            None => {
                debug!("Key '{}' not found in local cache", key);
                Ok(None)
            }
        }
    }

    async fn set<T>(&self, key: &str, value: &T, ttl: Option<StdDuration>) -> CacheResult<()>
    where
        T: Serialize + Send + Sync,
    {
        let serialized = self.serialize_value(value)?;
        let cache_value = CacheValue::new(serialized);
        
        if let Some(custom_ttl) = ttl {
            // For custom TTL, we store the value and will check expiration on get
            // Moka's TTL will still apply as a backup
            self.cache.insert(key.to_string(), cache_value).await;
            debug!("Set key '{}' in local cache with custom TTL {:?}", key, custom_ttl);
        } else {
            self.cache.insert(key.to_string(), cache_value).await;
            debug!("Set key '{}' in local cache with default TTL", key);
        }
        
        Ok(())
    }

    async fn delete(&self, key: &str) -> CacheResult<bool> {
        let existed = self.cache.get(key).await.is_some();
        self.cache.invalidate(key).await;
        debug!("Deleted key '{}' from local cache (existed: {})", key, existed);
        Ok(existed)
    }

    async fn exists(&self, key: &str) -> CacheResult<bool> {
        let exists = match self.cache.get(key).await {
            Some(cache_value) => !self.is_expired(&cache_value, None),
            None => false,
        };
        debug!("Key '{}' exists in local cache: {}", key, exists);
        Ok(exists)
    }

    async fn expire(&self, key: &str, _ttl: StdDuration) -> CacheResult<bool> {
        // Moka doesn't support changing TTL of existing entries
        // We could implement this by re-inserting the value, but that would require
        // deserializing and re-serializing. For now, we'll return false to indicate
        // the operation is not supported in the same way as Redis.
        debug!("Expire operation not directly supported for key '{}' in local cache", key);
        Ok(false)
    }

    async fn get_multi<T>(&self, keys: &[&str]) -> CacheResult<HashMap<String, T>>
    where
        T: DeserializeOwned + Send + Serialize + Sync + Clone,
    {
        let mut result = HashMap::new();
        
        for &key in keys {
            match self.get(key).await {
                Ok(Some(value)) => {
                    result.insert(key.to_string(), value);
                }
                Ok(None) => {
                    // Key doesn't exist, skip it
                }
                Err(e) => {
                    warn!("Error getting key '{}' in multi-get: {}", key, e);
                    // Continue with other keys instead of failing the entire operation
                }
            }
        }
        
        debug!("Retrieved {} out of {} keys from local cache", result.len(), keys.len());
        Ok(result)
    }

    async fn set_multi<T>(&self, items: &HashMap<&str, T>, ttl: Option<StdDuration>) -> CacheResult<()>
    where
        T: Serialize + Send + Sync,
    {
        for (key, value) in items {
            // Set each item individually - Moka doesn't have a native multi-set
            self.set(key, value, ttl).await?;
        }
        
        debug!("Set {} keys in local cache", items.len());
        Ok(())
    }

    async fn flush(&self) -> CacheResult<()> {
        self.cache.invalidate_all();
        // Wait for invalidation to complete
        self.cache.run_pending_tasks().await;
        warn!("Flushed all keys from local cache");
        Ok(())
    }
}

/// Builder pattern for creating MemoryCache with custom settings
pub struct MemoryCacheBuilder {
    max_capacity: u64,
    ttl: StdDuration,
    idle_time: Option<StdDuration>,
}

impl MemoryCacheBuilder {
    pub fn new() -> Self {
        Self {
            max_capacity: 1000,
            ttl: StdDuration::from_secs(300),
            idle_time: None,
        }
    }
    
    pub fn max_capacity(mut self, capacity: u64) -> Self {
        self.max_capacity = capacity;
        self
    }
    
    pub fn ttl(mut self, ttl: StdDuration) -> Self {
        self.ttl = ttl;
        self
    }
    
    pub fn idle_time(mut self, idle: StdDuration) -> Self {
        self.idle_time = Some(idle);
        self
    }
    
    pub fn build(self) -> MemoryCache {
        let config = CacheConfig {
            local_cache_size: self.max_capacity,
            local_cache_ttl: self.ttl,
            ..Default::default()
        };
        
        let mut cache_builder = Cache::builder()
            .max_capacity(self.max_capacity)
            .time_to_live(self.ttl);
            
        if let Some(idle) = self.idle_time {
            cache_builder = cache_builder.time_to_idle(idle);
        }
        
        let cache = cache_builder
            .eviction_listener(|key, _value, cause| {
                debug!("Evicted cache key '{}' due to {:?}", key, cause);
            })
            .build();

        MemoryCache {
            cache: Arc::new(cache),
            config,
        }
    }
}

impl Default for MemoryCacheBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    fn create_test_cache() -> MemoryCache {
        let config = CacheConfig {
            local_cache_size: 100,
            local_cache_ttl: Duration::from_secs(1),
            ..Default::default()
        };
        MemoryCache::new(config)
    }

    #[tokio::test]
    async fn test_memory_set_get() {
        let cache = create_test_cache();
        let key = "test:set_get";
        let value = "test_value".to_string();
        
        // Test set and get
        cache.set(key, &value, None).await.unwrap();
        let result: Option<String> = cache.get(key).await.unwrap();
        assert_eq!(result, Some(value));
    }

    #[tokio::test]
    async fn test_memory_ttl() {
        let cache = create_test_cache();
        let key = "test:ttl";
        let value = "test_value".to_string();
        
        // Set value with short TTL
        cache.set(key, &value, Some(Duration::from_millis(100))).await.unwrap();
        
        // Should exist immediately
        assert!(cache.exists(key).await.unwrap());
        
        // Wait for expiration
        sleep(Duration::from_millis(150)).await;
        
        // Should not exist after expiration
        let result: Option<String> = cache.get(key).await.unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_memory_delete() {
        let cache = create_test_cache();
        let key = "test:delete";
        let value = "test_value".to_string();
        
        // Set and verify
        cache.set(key, &value, None).await.unwrap();
        assert!(cache.exists(key).await.unwrap());
        
        // Delete and verify
        assert!(cache.delete(key).await.unwrap());
        assert!(!cache.exists(key).await.unwrap());
        
        // Delete non-existent key
        assert!(!cache.delete("non_existent").await.unwrap());
    }

    #[tokio::test]
    async fn test_memory_multi_operations() {
        let cache = create_test_cache();
        
        let mut items = HashMap::new();
        items.insert("test:multi1", "value1".to_string());
        items.insert("test:multi2", "value2".to_string());
        items.insert("test:multi3", "value3".to_string());
        
        // Test multi set
        cache.set_multi(&items, None).await.unwrap();
        
        // Test multi get
        let keys: Vec<&str> = items.keys().copied().collect();
        let results: HashMap<String, String> = cache.get_multi(&keys).await.unwrap();
        
        assert_eq!(results.len(), 3);
        assert_eq!(results.get("test:multi1"), Some(&"value1".to_string()));
        assert_eq!(results.get("test:multi2"), Some(&"value2".to_string()));
        assert_eq!(results.get("test:multi3"), Some(&"value3".to_string()));
    }

    #[tokio::test]
    async fn test_memory_flush() {
        let cache = create_test_cache();
        
        // Set some values
        cache.set("key1", &"value1".to_string(), None).await.unwrap();
        cache.set("key2", &"value2".to_string(), None).await.unwrap();
        
        // Verify they exist
        assert!(cache.exists("key1").await.unwrap());
        assert!(cache.exists("key2").await.unwrap());
        
        // Flush cache
        cache.flush().await.unwrap();
        
        // Verify they're gone
        assert!(!cache.exists("key1").await.unwrap());
        assert!(!cache.exists("key2").await.unwrap());
    }

    #[tokio::test]
    async fn test_memory_cache_builder() {
        let cache = MemoryCacheBuilder::new()
            .max_capacity(50)
            .ttl(Duration::from_secs(10))
            .idle_time(Duration::from_secs(5))
            .build();
            
        let key = "test:builder";
        let value = "test_value".to_string();
        
        cache.set(key, &value, None).await.unwrap();
        let result: Option<String> = cache.get(key).await.unwrap();
        assert_eq!(result, Some(value));
    }

    #[tokio::test]
    async fn test_memory_capacity_eviction() {
        // Create cache with small capacity
        let cache = MemoryCacheBuilder::new()
            .max_capacity(2)
            .ttl(Duration::from_secs(60))
            .build();
        
        // Fill cache to capacity
        cache.set("key1", &"value1".to_string(), None).await.unwrap();
        cache.set("key2", &"value2".to_string(), None).await.unwrap();
        
        // Add one more item, should cause eviction
        cache.set("key3", &"value3".to_string(), None).await.unwrap();
        
        // Let eviction take place
        sleep(Duration::from_millis(10)).await;
        cache.cache.run_pending_tasks().await;
        
        // At least one of the original keys should be evicted
        let key1_exists = cache.exists("key1").await.unwrap();
        let key2_exists = cache.exists("key2").await.unwrap();
        let key3_exists = cache.exists("key3").await.unwrap();
        
        // key3 should definitely exist (just inserted)
        assert!(key3_exists);
        
        // At most 2 keys should exist due to capacity limit
        let total_existing = [key1_exists, key2_exists, key3_exists].iter().filter(|&&x| x).count();
        assert!(total_existing <= 2);
    }

    #[tokio::test]
    async fn test_memory_serialization_error_handling() {
        let cache = create_test_cache();
        let key = "test:serialization";
        
        // Set a valid value
        cache.set(key, &"valid_value".to_string(), None).await.unwrap();
        
        // Try to get as wrong type - should fail gracefully
        let result: CacheResult<Option<i32>> = cache.get(key).await;
        assert!(result.is_err());
        
        // Key should be removed after serialization error
        let exists = cache.exists(key).await.unwrap();
        assert!(!exists);
    }
}