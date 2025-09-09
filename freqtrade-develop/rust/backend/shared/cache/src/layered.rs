use crate::{CacheError, CacheProvider, CacheResult, CacheConfig, CacheProviderType, distributed::RedisCache, local::MemoryCache};
use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize};
use std::{collections::HashMap, time::Duration as StdDuration};
use tracing::{debug, error, info, warn};

/// Layered cache strategy combining multiple cache tiers
/// Implements the "cache-aside" pattern with configurable fallback behavior
pub struct LayeredCache {
    /// L1 cache (fastest, smallest) - typically local memory
    l1_cache: Option<CacheProviderType>,
    /// L2 cache (distributed) - typically Redis
    l2_cache: Option<CacheProviderType>,
    /// L3 cache (optional, persistent) - could be database or file system
    l3_cache: Option<CacheProviderType>,
    config: CacheConfig,
    strategy: LayeredStrategy,
}

/// Strategy for layered cache behavior
#[derive(Debug, Clone)]
#[derive(Default)]
pub enum LayeredStrategy {
    /// Write to all layers, read from fastest first (default)
    #[default]
    WriteThrough,
    /// Write only to L1, propagate to other layers asynchronously
    WriteBack,
    /// Write to L1 and L2, skip L3 for writes
    WriteBehind,
    /// Custom strategy with per-operation configuration
    Custom(LayeredConfig),
}

/// Configuration for custom layered strategies
#[derive(Debug, Clone)]
pub struct LayeredConfig {
    pub read_through: bool,
    pub write_through: bool,
    pub populate_lower_on_hit: bool,
    pub evict_from_all_on_delete: bool,
    pub l1_ttl_factor: f32, // TTL multiplier for L1 cache
    pub l2_ttl_factor: f32, // TTL multiplier for L2 cache
}

impl Default for LayeredConfig {
    fn default() -> Self {
        Self {
            read_through: true,
            write_through: true,
            populate_lower_on_hit: true,
            evict_from_all_on_delete: true,
            l1_ttl_factor: 0.5, // L1 cache has shorter TTL
            l2_ttl_factor: 1.0, // L2 cache uses full TTL
        }
    }
}


impl LayeredCache {
    /// Create a new layered cache with L1 (local) and L2 (distributed) tiers
    pub async fn new_two_tier(
        local_config: CacheConfig,
        redis_config: CacheConfig,
        strategy: LayeredStrategy,
    ) -> CacheResult<Self> {
        info!("Initializing two-tier layered cache");
        
        // Create L1 cache (local memory)
        let l1_cache = CacheProviderType::Memory(MemoryCache::new(local_config.clone()));
        
        // Create L2 cache (Redis)
        let redis_client = redis::Client::open(redis_config.redis_url.as_str())
            .map_err(CacheError::Redis)?;
        let l2_cache = CacheProviderType::Redis(RedisCache::new(redis_client, redis_config.clone()).await?);
        
        Ok(Self {
            l1_cache: Some(l1_cache),
            l2_cache: Some(l2_cache),
            l3_cache: None,
            config: local_config, // Use local config as primary
            strategy,
        })
    }
    
    /// Create a layered cache with only L1 (local) tier
    pub fn new_single_tier(config: CacheConfig, strategy: LayeredStrategy) -> Self {
        info!("Initializing single-tier layered cache");
        
        let l1_cache = CacheProviderType::Memory(MemoryCache::new(config.clone()));
        
        Self {
            l1_cache: Some(l1_cache),
            l2_cache: None,
            l3_cache: None,
            config,
            strategy,
        }
    }
    
    /// Add an L3 cache tier
    pub fn with_l3_cache(mut self, l3_cache: CacheProviderType) -> Self {
        self.l3_cache = Some(l3_cache);
        self
    }
    
    /// Get the effective configuration for the layered strategy
    fn get_layered_config(&self) -> LayeredConfig {
        match &self.strategy {
            LayeredStrategy::WriteThrough => LayeredConfig {
                read_through: true,
                write_through: true,
                populate_lower_on_hit: true,
                evict_from_all_on_delete: true,
                l1_ttl_factor: 0.5,
                l2_ttl_factor: 1.0,
            },
            LayeredStrategy::WriteBack => LayeredConfig {
                read_through: true,
                write_through: false,
                populate_lower_on_hit: false,
                evict_from_all_on_delete: false,
                l1_ttl_factor: 1.0,
                l2_ttl_factor: 1.2,
            },
            LayeredStrategy::WriteBehind => LayeredConfig {
                read_through: true,
                write_through: true,
                populate_lower_on_hit: true,
                evict_from_all_on_delete: true,
                l1_ttl_factor: 0.3,
                l2_ttl_factor: 1.0,
            },
            LayeredStrategy::Custom(config) => config.clone(),
        }
    }
    
    /// Calculate TTL for a specific cache tier
    fn calculate_ttl(&self, base_ttl: Option<StdDuration>, tier: u8) -> Option<StdDuration> {
        let config = self.get_layered_config();
        
        base_ttl.map(|ttl| {
            let factor = match tier {
                1 => config.l1_ttl_factor,
                2 => config.l2_ttl_factor,
                _ => 1.0,
            };
            
            let new_secs = (ttl.as_secs_f32() * factor) as u64;
            StdDuration::from_secs(new_secs.max(1)) // Ensure at least 1 second
        })
    }
    
    /// Populate lower cache tiers when a value is found in a higher tier
    async fn populate_lower_tiers<T>(
        &self,
        key: &str,
        value: &T,
        found_in_tier: u8,
        ttl: Option<StdDuration>,
    ) where
        T: Serialize + Send + Sync + Clone,
    {
        let config = self.get_layered_config();
        if !config.populate_lower_on_hit {
            return;
        }
        
        // Populate L1 if value was found in L2 or L3
        if found_in_tier > 1 {
            if let Some(ref l1) = self.l1_cache {
                let l1_ttl = self.calculate_ttl(ttl, 1);
                if let Err(e) = l1.set(key, value, l1_ttl).await {
                    warn!("Failed to populate L1 cache for key '{}': {}", key, e);
                }
            }
        }
        
        // Populate L2 if value was found in L3
        if found_in_tier > 2 {
            if let Some(ref l2) = self.l2_cache {
                let l2_ttl = self.calculate_ttl(ttl, 2);
                if let Err(e) = l2.set(key, value, l2_ttl).await {
                    warn!("Failed to populate L2 cache for key '{}': {}", key, e);
                }
            }
        }
    }
    
    /// Write value to appropriate cache tiers based on strategy
    async fn write_to_tiers<T>(&self, key: &str, value: &T, ttl: Option<StdDuration>) -> CacheResult<()>
    where
        T: Serialize + Send + Sync,
    {
        let config = self.get_layered_config();
        let mut errors = Vec::new();
        
        // Write to L1 cache
        if let Some(ref l1) = self.l1_cache {
            let l1_ttl = self.calculate_ttl(ttl, 1);
            if let Err(e) = l1.set(key, value, l1_ttl).await {
                warn!("Failed to write to L1 cache for key '{}': {}", key, e);
                errors.push(e);
            }
        }
        
        // Write to L2 cache (if write-through)
        if config.write_through {
            if let Some(ref l2) = self.l2_cache {
                let l2_ttl = self.calculate_ttl(ttl, 2);
                if let Err(e) = l2.set(key, value, l2_ttl).await {
                    error!("Failed to write to L2 cache for key '{}': {}", key, e);
                    errors.push(e);
                }
            }
            
            // Write to L3 cache
            if let Some(ref l3) = self.l3_cache {
                if let Err(e) = l3.set(key, value, ttl).await {
                    error!("Failed to write to L3 cache for key '{}': {}", key, e);
                    errors.push(e);
                }
            }
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            // Return the first error, but log all of them
            Err(errors.into_iter().next().unwrap())
        }
    }
    
    /// Delete from appropriate cache tiers based on strategy
    async fn delete_from_tiers(&self, key: &str) -> CacheResult<bool> {
        let config = self.get_layered_config();
        let mut any_deleted = false;
        
        if config.evict_from_all_on_delete {
            // Delete from all tiers
            if let Some(ref l1) = self.l1_cache {
                if let Ok(deleted) = l1.delete(key).await {
                    any_deleted |= deleted;
                }
            }
            
            if let Some(ref l2) = self.l2_cache {
                if let Ok(deleted) = l2.delete(key).await {
                    any_deleted |= deleted;
                }
            }
            
            if let Some(ref l3) = self.l3_cache {
                if let Ok(deleted) = l3.delete(key).await {
                    any_deleted |= deleted;
                }
            }
        } else {
            // Delete only from L1
            if let Some(ref l1) = self.l1_cache {
                if let Ok(deleted) = l1.delete(key).await {
                    any_deleted = deleted;
                }
            }
        }
        
        Ok(any_deleted)
    }
}

#[async_trait]
impl CacheProvider for LayeredCache {
    async fn get<T>(&self, key: &str) -> CacheResult<Option<T>>
    where
        T: DeserializeOwned + Send + Serialize + Sync + Clone,
    {
        let config = self.get_layered_config();
        
        // Try L1 cache first
        if let Some(ref l1) = self.l1_cache {
            match l1.get(key).await {
                Ok(Some(value)) => {
                    debug!("Cache hit in L1 for key '{}'", key);
                    return Ok(Some(value));
                }
                Ok(None) => {
                    debug!("Cache miss in L1 for key '{}'", key);
                }
                Err(e) => {
                    warn!("L1 cache error for key '{}': {}", key, e);
                }
            }
        }
        
        // Try L2 cache if read-through is enabled
        if config.read_through {
            if let Some(ref l2) = self.l2_cache {
                match l2.get(key).await {
                    Ok(Some(value)) => {
                        debug!("Cache hit in L2 for key '{}'", key);
                        
                        // Populate L1 cache
                        self.populate_lower_tiers(key, &value, 2, Some(self.config.local_cache_ttl)).await;
                        
                        return Ok(Some(value));
                    }
                    Ok(None) => {
                        debug!("Cache miss in L2 for key '{}'", key);
                    }
                    Err(e) => {
                        warn!("L2 cache error for key '{}': {}", key, e);
                    }
                }
            }
            
            // Try L3 cache
            if let Some(ref l3) = self.l3_cache {
                match l3.get(key).await {
                    Ok(Some(value)) => {
                        debug!("Cache hit in L3 for key '{}'", key);
                        
                        // Populate lower tiers
                        self.populate_lower_tiers(key, &value, 3, Some(self.config.local_cache_ttl)).await;
                        
                        return Ok(Some(value));
                    }
                    Ok(None) => {
                        debug!("Cache miss in L3 for key '{}'", key);
                    }
                    Err(e) => {
                        warn!("L3 cache error for key '{}': {}", key, e);
                    }
                }
            }
        }
        
        debug!("Complete cache miss for key '{}'", key);
        Ok(None)
    }

    async fn set<T>(&self, key: &str, value: &T, ttl: Option<StdDuration>) -> CacheResult<()>
    where
        T: Serialize + Send + Sync,
    {
        debug!("Setting key '{}' in layered cache with strategy {:?}", key, self.strategy);
        self.write_to_tiers(key, value, ttl).await
    }

    async fn delete(&self, key: &str) -> CacheResult<bool> {
        debug!("Deleting key '{}' from layered cache", key);
        self.delete_from_tiers(key).await
    }

    async fn exists(&self, key: &str) -> CacheResult<bool> {
        // Check existence in order of cache tiers
        if let Some(ref l1) = self.l1_cache {
            if l1.exists(key).await.unwrap_or(false) {
                return Ok(true);
            }
        }
        
        if let Some(ref l2) = self.l2_cache {
            if l2.exists(key).await.unwrap_or(false) {
                return Ok(true);
            }
        }
        
        if let Some(ref l3) = self.l3_cache {
            if l3.exists(key).await.unwrap_or(false) {
                return Ok(true);
            }
        }
        
        Ok(false)
    }

    async fn expire(&self, key: &str, ttl: StdDuration) -> CacheResult<bool> {
        let mut any_success = false;
        
        // Set expiration on all available tiers
        if let Some(ref l1) = self.l1_cache {
            let l1_ttl = self.calculate_ttl(Some(ttl), 1).unwrap_or(ttl);
            if l1.expire(key, l1_ttl).await.unwrap_or(false) {
                any_success = true;
            }
        }
        
        if let Some(ref l2) = self.l2_cache {
            let l2_ttl = self.calculate_ttl(Some(ttl), 2).unwrap_or(ttl);
            if l2.expire(key, l2_ttl).await.unwrap_or(false) {
                any_success = true;
            }
        }
        
        if let Some(ref l3) = self.l3_cache {
            if l3.expire(key, ttl).await.unwrap_or(false) {
                any_success = true;
            }
        }
        
        Ok(any_success)
    }

    async fn get_multi<T>(&self, keys: &[&str]) -> CacheResult<HashMap<String, T>>
    where
        T: DeserializeOwned + Send + Serialize + Sync + Clone,
    {
        let mut result = HashMap::new();
        let mut remaining_keys = keys.to_vec();
        
        // Try L1 cache first
        if let Some(ref l1) = self.l1_cache {
            if let Ok(l1_results) = l1.get_multi(&remaining_keys).await {
                result.extend(l1_results);
                remaining_keys.retain(|&key| !result.contains_key(key));
            }
        }
        
        // Try L2 cache for remaining keys
        if !remaining_keys.is_empty() {
            if let Some(ref l2) = self.l2_cache {
                if let Ok(l2_results) = l2.get_multi(&remaining_keys).await {
                    // Populate L1 cache with L2 results
                    for (key, value) in &l2_results {
                        if let Some(ref l1) = self.l1_cache {
                            let l1_ttl = self.calculate_ttl(Some(self.config.local_cache_ttl), 1);
                            let _ = l1.set(key, value, l1_ttl).await;
                        }
                    }
                    
                    result.extend(l2_results);
                    remaining_keys.retain(|&key| !result.contains_key(key));
                }
            }
        }
        
        // Try L3 cache for remaining keys
        if !remaining_keys.is_empty() {
            if let Some(ref l3) = self.l3_cache {
                if let Ok(l3_results) = l3.get_multi(&remaining_keys).await {
                    // Populate lower tiers with L3 results
                    for (key, value) in &l3_results {
                        self.populate_lower_tiers(key, value, 3, Some(self.config.local_cache_ttl)).await;
                    }
                    
                    result.extend(l3_results);
                }
            }
        }
        
        debug!("Multi-get retrieved {} out of {} keys from layered cache", result.len(), keys.len());
        Ok(result)
    }

    async fn set_multi<T>(&self, items: &HashMap<&str, T>, ttl: Option<StdDuration>) -> CacheResult<()>
    where
        T: Serialize + Send + Sync,
    {
        let mut errors = Vec::new();
        
        for (key, value) in items {
            if let Err(e) = self.set(key, value, ttl).await {
                errors.push(e);
            }
        }
        
        if errors.is_empty() {
            debug!("Multi-set completed for {} keys in layered cache", items.len());
            Ok(())
        } else {
            error!("Multi-set had {} errors in layered cache", errors.len());
            Err(errors.into_iter().next().unwrap())
        }
    }

    async fn flush(&self) -> CacheResult<()> {
        let mut errors = Vec::new();
        
        if let Some(ref l1) = self.l1_cache {
            if let Err(e) = l1.flush().await {
                errors.push(e);
            }
        }
        
        if let Some(ref l2) = self.l2_cache {
            if let Err(e) = l2.flush().await {
                errors.push(e);
            }
        }
        
        if let Some(ref l3) = self.l3_cache {
            if let Err(e) = l3.flush().await {
                errors.push(e);
            }
        }
        
        if errors.is_empty() {
            warn!("Flushed all tiers of layered cache");
            Ok(())
        } else {
            error!("Flush had {} errors in layered cache", errors.len());
            Err(errors.into_iter().next().unwrap())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CacheConfig;
    use tokio::time::{sleep, Duration};

    fn create_test_config() -> CacheConfig {
        CacheConfig {
            local_cache_size: 100,
            local_cache_ttl: Duration::from_secs(10),
            redis_url: "redis://127.0.0.1:6379".to_string(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_single_tier_layered_cache() {
        let config = create_test_config();
        let cache = LayeredCache::new_single_tier(config, LayeredStrategy::WriteThrough);
        
        let key = "test:single_tier";
        let value = "test_value".to_string();
        
        // Test set and get
        cache.set(key, &value, None).await.unwrap();
        let result: Option<String> = cache.get(key).await.unwrap();
        assert_eq!(result, Some(value));
        
        // Test delete
        assert!(cache.delete(key).await.unwrap());
        let result: Option<String> = cache.get(key).await.unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    #[ignore] // Requires Redis server
    async fn test_two_tier_layered_cache() {
        let local_config = create_test_config();
        let redis_config = create_test_config();
        
        let cache = LayeredCache::new_two_tier(
            local_config,
            redis_config,
            LayeredStrategy::WriteThrough,
        ).await.unwrap();
        
        let key = "test:two_tier";
        let value = "test_value".to_string();
        
        // Clean up first
        let _ = cache.delete(key).await;
        
        // Test set (should write to both tiers)
        cache.set(key, &value, None).await.unwrap();
        
        // Test get (should hit L1 first)
        let result: Option<String> = cache.get(key).await.unwrap();
        assert_eq!(result, Some(value));
        
        // Clean up
        cache.delete(key).await.unwrap();
    }

    #[tokio::test]
    async fn test_layered_strategy_write_back() {
        let config = create_test_config();
        let cache = LayeredCache::new_single_tier(config, LayeredStrategy::WriteBack);
        
        let key = "test:write_back";
        let value = "test_value".to_string();
        
        cache.set(key, &value, None).await.unwrap();
        let result: Option<String> = cache.get(key).await.unwrap();
        assert_eq!(result, Some(value));
    }

    #[tokio::test]
    async fn test_layered_ttl_calculation() {
        let config = create_test_config();
        let cache = LayeredCache::new_single_tier(config, LayeredStrategy::WriteThrough);
        
        let base_ttl = Some(Duration::from_secs(100));
        
        // Test TTL calculation for different tiers
        let l1_ttl = cache.calculate_ttl(base_ttl, 1);
        let l2_ttl = cache.calculate_ttl(base_ttl, 2);
        
        assert!(l1_ttl.is_some());
        assert!(l2_ttl.is_some());
        
        // L1 should have shorter TTL (factor 0.5)
        assert!(l1_ttl.unwrap() < base_ttl.unwrap());
        // L2 should have same TTL (factor 1.0)
        assert_eq!(l2_ttl.unwrap(), base_ttl.unwrap());
    }

    #[tokio::test]
    async fn test_layered_custom_strategy() {
        let config = create_test_config();
        let custom_config = LayeredConfig {
            l1_ttl_factor: 0.2,
            l2_ttl_factor: 0.8,
            populate_lower_on_hit: false,
            ..Default::default()
        };
        
        let cache = LayeredCache::new_single_tier(
            config, 
            LayeredStrategy::Custom(custom_config)
        );
        
        let base_ttl = Some(Duration::from_secs(100));
        let l1_ttl = cache.calculate_ttl(base_ttl, 1);
        let l2_ttl = cache.calculate_ttl(base_ttl, 2);
        
        assert_eq!(l1_ttl.unwrap(), Duration::from_secs(20)); // 100 * 0.2
        assert_eq!(l2_ttl.unwrap(), Duration::from_secs(80)); // 100 * 0.8
    }

    #[tokio::test]
    async fn test_layered_multi_operations() {
        let config = create_test_config();
        let cache = LayeredCache::new_single_tier(config, LayeredStrategy::WriteThrough);
        
        let mut items = HashMap::new();
        items.insert("test:layered1", "value1".to_string());
        items.insert("test:layered2", "value2".to_string());
        items.insert("test:layered3", "value3".to_string());
        
        // Test multi set
        cache.set_multi(&items, None).await.unwrap();
        
        // Test multi get
        let keys: Vec<&str> = items.keys().copied().collect();
        let results: HashMap<String, String> = cache.get_multi(&keys).await.unwrap();
        
        assert_eq!(results.len(), 3);
        assert_eq!(results.get("test:layered1"), Some(&"value1".to_string()));
        assert_eq!(results.get("test:layered2"), Some(&"value2".to_string()));
        assert_eq!(results.get("test:layered3"), Some(&"value3".to_string()));
    }

    #[tokio::test]
    async fn test_layered_exists() {
        let config = create_test_config();
        let cache = LayeredCache::new_single_tier(config, LayeredStrategy::WriteThrough);
        
        let key = "test:exists";
        let value = "test_value".to_string();
        
        // Should not exist initially
        assert!(!cache.exists(key).await.unwrap());
        
        // Set value
        cache.set(key, &value, None).await.unwrap();
        
        // Should exist now
        assert!(cache.exists(key).await.unwrap());
        
        // Delete and verify
        cache.delete(key).await.unwrap();
        assert!(!cache.exists(key).await.unwrap());
    }

    #[tokio::test]
    async fn test_layered_flush() {
        let config = create_test_config();
        let cache = LayeredCache::new_single_tier(config, LayeredStrategy::WriteThrough);
        
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
}