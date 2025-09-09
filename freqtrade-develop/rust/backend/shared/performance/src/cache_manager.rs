use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use moka::future::{Cache, CacheBuilder};
use platform_config::PlatformConfig;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, OnceLock};
use tracing::{info, warn};
use std::collections::HashMap;

/// Cache configuration for different cache tiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub l1_cache_size: u64,           // In-memory cache entries
    pub l2_cache_size: u64,           // Redis cache entries
    pub default_ttl_seconds: u64,    // Default time-to-live
    pub hot_data_ttl_seconds: u64,   // Hot data TTL
    pub warm_data_ttl_seconds: u64,  // Warm data TTL
    pub cold_data_ttl_seconds: u64,  // Cold data TTL
    pub cache_hit_ratio_target: f64, // Target cache hit ratio
    pub eviction_policy: EvictionPolicy,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_cache_size: 10000,        // 10k entries
            l2_cache_size: 100000,       // 100k entries
            default_ttl_seconds: 300,    // 5 minutes
            hot_data_ttl_seconds: 60,    // 1 minute
            warm_data_ttl_seconds: 900,  // 15 minutes
            cold_data_ttl_seconds: 3600, // 1 hour
            cache_hit_ratio_target: 0.85, // 85% hit ratio target
            eviction_policy: EvictionPolicy::LRU,
        }
    }
}

/// Cache eviction policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,  // Least Recently Used
    LFU,  // Least Frequently Used
    FIFO, // First In First Out
    TTL,  // Time-based eviction only
}

/// Multi-tier cache manager for optimal performance
pub struct CacheManager {
    config: CacheConfig,
    l1_cache: Cache<String, CacheEntry>, // In-memory L1 cache
    redis_pool: Option<redis::aio::MultiplexedConnection>, // L2 Redis cache
    hit_counts: HashMap<String, u64>,
    miss_counts: HashMap<String, u64>,
    cache_stats: CacheStats,
}

/// Cache entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub data: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u64,
    pub ttl: Duration,
    pub cache_tier: CacheTier,
}

/// Cache tier levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheTier {
    L1Memory,   // Fastest, in-memory
    L2Redis,    // Medium speed, Redis
    L3Storage,  // Slowest, persistent storage
}

/// Cache performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub l1_hits: u64,
    pub l1_misses: u64,
    pub l2_hits: u64,
    pub l2_misses: u64,
    pub total_gets: u64,
    pub total_sets: u64,
    pub evictions: u64,
    pub hit_ratio: f64,
    pub average_response_time_ms: f64,
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            l1_hits: 0,
            l1_misses: 0,
            l2_hits: 0,
            l2_misses: 0,
            total_gets: 0,
            total_sets: 0,
            evictions: 0,
            hit_ratio: 0.0,
            average_response_time_ms: 0.0,
        }
    }
}

impl CacheManager {
    pub async fn new(config: CacheConfig) -> Result<Self> {
        // Initialize L1 in-memory cache
        let l1_cache = CacheBuilder::new(config.l1_cache_size)
            .time_to_live(std::time::Duration::from_secs(config.default_ttl_seconds))
            .build();
        
        // Initialize Redis connection for L2 cache
        let redis_pool = Self::init_redis_connection().await.ok();
        
        Ok(Self {
            config,
            l1_cache,
            redis_pool,
            hit_counts: HashMap::new(),
            miss_counts: HashMap::new(),
            cache_stats: CacheStats::default(),
        })
    }
    
    /// Initialize Redis connection for L2 caching
    async fn init_redis_connection() -> Result<redis::aio::MultiplexedConnection> {
        let redis_url = std::env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://localhost:6379".to_string());
        
        let client = redis::Client::open(redis_url)?;
        let conn = client.get_multiplexed_async_connection().await?;
        
        info!("Redis L2 cache connection established");
        Ok(conn)
    }
    
    /// Get value from cache with automatic tier fallback
    pub async fn get<T>(&mut self, key: &str) -> Option<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        let start_time = std::time::Instant::now();
        self.cache_stats.total_gets += 1;
        
        // Try L1 cache first
        if let Some(entry) = self.l1_cache.get(key).await {
            self.cache_stats.l1_hits += 1;
            self.update_hit_count(key);
            
            // Update access metadata
            let mut updated_entry = entry;
            updated_entry.last_accessed = Utc::now();
            updated_entry.access_count += 1;
            self.l1_cache.insert(key.to_string(), updated_entry.clone()).await;
            
            // Deserialize data
            if let Ok(data) = serde_json::from_value(updated_entry.data) {
                self.update_response_time(start_time);
                return Some(data);
            }
        }
        
        // Try L2 Redis cache
        self.cache_stats.l1_misses += 1;
        if let Some(ref mut redis_conn) = self.redis_pool {
            if let Ok(Some(data_str)) = redis::cmd("GET")
                .arg(key)
                .query_async::<Option<String>>(redis_conn)
                .await
            {
                self.cache_stats.l2_hits += 1;
                self.update_hit_count(key);
                
                // Deserialize and promote to L1 cache
                if let Ok(entry) = serde_json::from_str::<CacheEntry>(&data_str) {
                    // Promote to L1 for faster future access
                    self.l1_cache.insert(key.to_string(), entry.clone()).await;
                    
                    if let Ok(data) = serde_json::from_value(entry.data) {
                        self.update_response_time(start_time);
                        return Some(data);
                    }
                }
            }
        }
        
        // Cache miss on all tiers
        self.cache_stats.l2_misses += 1;
        self.update_miss_count(key);
        self.update_response_time(start_time);
        self.update_hit_ratio();
        
        None
    }
    
    /// Set value in cache with intelligent tier placement
    pub async fn set<T>(&mut self, key: &str, value: T, ttl: Option<Duration>)
    where
        T: Serialize,
    {
        self.cache_stats.total_sets += 1;
        
        let ttl = ttl.unwrap_or_else(|| Duration::seconds(self.config.default_ttl_seconds as i64));
        
        let entry = CacheEntry {
            data: serde_json::to_value(value).unwrap_or_default(),
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 1,
            ttl,
            cache_tier: CacheTier::L1Memory,
        };
        
        // Store in L1 cache
        self.l1_cache.insert(key.to_string(), entry.clone()).await;
        
        // Store in L2 Redis cache for persistence
        if let Some(ref mut redis_conn) = self.redis_pool {
            let serialized = serde_json::to_string(&entry).unwrap_or_default();
            let ttl_seconds = ttl.num_seconds() as u64;
            
            let _: Result<(), redis::RedisError> = redis::cmd("SETEX")
                .arg(key)
                .arg(ttl_seconds)
                .arg(serialized)
                .query_async(redis_conn)
                .await;
        }
    }
    
    /// Smart cache warming based on access patterns
    pub async fn warm_cache(&mut self, keys: Vec<String>) {
        info!("Starting cache warming for {} keys", keys.len());
        
        for key in keys {
            // Check if key should be warmed based on access frequency
            let access_count = self.hit_counts.get(&key).unwrap_or(&0);
            
            if *access_count > 5 {  // Frequently accessed keys
                // Pre-load into L1 cache if not already present
                if !self.l1_cache.contains_key(&key) {
                    // Try loading from L2 cache
                    if let Some(cached_value) = self.get::<serde_json::Value>(&key).await {
                        self.set(&key, cached_value, None).await;
                    }
                }
            }
        }
        
        info!("Cache warming completed");
    }
    
    /// Evict cache entries based on configured policy
    pub async fn evict_expired(&mut self) {
        match self.config.eviction_policy {
            EvictionPolicy::LRU => {
                // Moka cache handles LRU automatically
                info!("LRU eviction handled automatically");
            }
            EvictionPolicy::TTL => {
                // TTL-based eviction is also automatic
                info!("TTL eviction handled automatically");
            }
            _ => {
                warn!("Eviction policy {:?} not yet implemented", self.config.eviction_policy);
            }
        }
    }
    
    /// Get cache performance statistics
    pub fn get_stats(&self) -> CacheStats {
        self.cache_stats.clone()
    }
    
    /// Optimize cache based on access patterns
    pub async fn optimize_cache(&mut self) {
        let hit_ratio = self.cache_stats.hit_ratio;
        
        if hit_ratio < self.config.cache_hit_ratio_target {
            warn!(
                "Cache hit ratio ({:.2}%) below target ({:.2}%), optimizing...",
                hit_ratio * 100.0,
                self.config.cache_hit_ratio_target * 100.0
            );
            
            // Identify hot keys for promotion
            let mut hot_keys: Vec<_> = self.hit_counts.iter().collect();
            hot_keys.sort_by(|a, b| b.1.cmp(a.1));
            
            // Warm cache with top 100 hot keys
            let keys_to_warm: Vec<String> = hot_keys
                .into_iter()
                .take(100)
                .map(|(k, _)| k.clone())
                .collect();
            
            self.warm_cache(keys_to_warm).await;
        }
    }
    
    /// Clear all caches
    pub async fn clear_all(&mut self) {
        self.l1_cache.invalidate_all();
        
        if let Some(ref mut redis_conn) = self.redis_pool {
            let _: Result<(), redis::RedisError> = redis::cmd("FLUSHDB")
                .query_async(redis_conn)
                .await;
        }
        
        self.hit_counts.clear();
        self.miss_counts.clear();
        self.cache_stats = CacheStats::default();
        
        info!("All caches cleared");
    }
    
    /// Update hit count for analytics
    fn update_hit_count(&mut self, key: &str) {
        *self.hit_counts.entry(key.to_string()).or_insert(0) += 1;
    }
    
    /// Update miss count for analytics
    fn update_miss_count(&mut self, key: &str) {
        *self.miss_counts.entry(key.to_string()).or_insert(0) += 1;
    }
    
    /// Update cache hit ratio
    fn update_hit_ratio(&mut self) {
        let total_hits = self.cache_stats.l1_hits + self.cache_stats.l2_hits;
        if self.cache_stats.total_gets > 0 {
            self.cache_stats.hit_ratio = total_hits as f64 / self.cache_stats.total_gets as f64;
        }
    }
    
    /// Update average response time
    fn update_response_time(&mut self, start_time: std::time::Instant) {
        let response_time_ms = start_time.elapsed().as_millis() as f64;
        
        // Running average calculation
        let current_avg = self.cache_stats.average_response_time_ms;
        let count = self.cache_stats.total_gets as f64;
        
        self.cache_stats.average_response_time_ms = 
            ((current_avg * (count - 1.0)) + response_time_ms) / count;
    }
}

/// Global cache manager instance
static CACHE_MANAGER: OnceLock<Arc<tokio::sync::Mutex<CacheManager>>> = OnceLock::new();

/// Initialize global cache manager
pub async fn init_cache_manager(_config: &PlatformConfig) -> Result<()> {
    let cache_config = CacheConfig::default();
    let manager = Arc::new(tokio::sync::Mutex::new(CacheManager::new(cache_config).await?));
    
    CACHE_MANAGER.set(manager)
        .map_err(|_| anyhow::anyhow!("Cache manager already initialized"))?;
    
    info!("Cache manager initialized with L1 and L2 tiers");
    Ok(())
}

/// Get global cache manager instance
pub async fn get_cache_manager() -> Option<Arc<tokio::sync::Mutex<CacheManager>>> {
    CACHE_MANAGER.get().cloned()
}

/// Shutdown cache manager
pub async fn shutdown_cache_manager() -> Result<()> {
    if let Some(manager) = get_cache_manager().await {
        let mut cache = manager.lock().await;
        cache.clear_all().await;
    }
    
    info!("Cache manager shutdown completed");
    Ok(())
}