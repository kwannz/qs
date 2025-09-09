use anyhow::{Result, Context};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::{HashMap, BTreeMap};
use std::hash::{Hash, Hasher, DefaultHasher};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// 两级缓存管理器
/// Level 1: 内存缓存 (快速访问)
/// Level 2: 持久化缓存 (Redis/文件系统)
#[derive(Debug)]
pub struct CacheManager {
    l1_cache: Arc<dyn CacheBackend + Send + Sync>,
    l2_cache: Option<Arc<dyn CacheBackend + Send + Sync>>,
    config: CacheConfig,
    stats: Arc<RwLock<CacheStats>>,
}

/// 缓存配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub l1_max_entries: usize,          // L1缓存最大条目数
    pub l1_ttl_default: Duration,       // L1默认TTL
    pub l2_enabled: bool,               // 启用L2缓存
    pub l2_ttl_default: Duration,       // L2默认TTL
    pub compression_enabled: bool,       // 启用压缩
    pub compression_threshold: usize,    // 压缩阈值(字节)
    pub eviction_policy: EvictionPolicy, // 淘汰策略
    pub prefetch_enabled: bool,         // 启用预取
    pub write_through: bool,            // 写透策略
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_max_entries: 10000,
            l1_ttl_default: Duration::from_secs(300), // 5分钟
            l2_enabled: true,
            l2_ttl_default: Duration::from_secs(3600), // 1小时
            compression_enabled: true,
            compression_threshold: 1024, // 1KB
            eviction_policy: EvictionPolicy::LRU,
            prefetch_enabled: true,
            write_through: true,
        }
    }
}

/// 淘汰策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,    // 最近最少使用
    LFU,    // 最不频繁使用
    FIFO,   // 先进先出
    TTL,    // 基于TTL
}

/// 缓存统计
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    pub l1_hits: u64,
    pub l1_misses: u64,
    pub l2_hits: u64,
    pub l2_misses: u64,
    pub l1_evictions: u64,
    pub l2_evictions: u64,
    pub total_gets: u64,
    pub total_sets: u64,
    pub hit_rate: f64,
    pub l1_hit_rate: f64,
    pub l2_hit_rate: f64,
    pub avg_get_time_ms: f64,
    pub avg_set_time_ms: f64,
    pub memory_usage_bytes: u64,
    pub compression_ratio: f64,
}

/// 缓存条目
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    value: T,
    created_at: Instant,
    last_accessed: Instant,
    access_count: u64,
    ttl: Option<Duration>,
    compressed: bool,
    size_bytes: usize,
}

impl<T> CacheEntry<T> {
    fn new(value: T, ttl: Option<Duration>, size_bytes: usize) -> Self {
        let now = Instant::now();
        Self {
            value,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            ttl,
            compressed: false,
            size_bytes,
        }
    }
    
    fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            self.created_at.elapsed() > ttl
        } else {
            false
        }
    }
    
    fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

/// 缓存后端特征
pub trait CacheBackend: std::fmt::Debug {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    fn set(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> Result<()>;
    fn delete(&self, key: &str) -> Result<()>;
    fn clear(&self) -> Result<()>;
    fn size(&self) -> usize;
    fn keys(&self) -> Vec<String>;
}

/// 内存缓存后端
#[derive(Debug)]
pub struct MemoryCacheBackend {
    storage: Arc<RwLock<HashMap<String, CacheEntry<Vec<u8>>>>>,
    max_entries: usize,
}

impl MemoryCacheBackend {
    pub fn new(max_entries: usize) -> Self {
        Self {
            storage: Arc::new(RwLock::new(HashMap::new())),
            max_entries,
        }
    }
    
    async fn evict_if_needed(&self) -> Result<()> {
        let mut storage = self.storage.write().await;
        
        if storage.len() >= self.max_entries {
            // 简单的LRU淘汰
            let mut entries: Vec<_> = storage.iter().map(|(k, v)| (k.clone(), v.last_accessed)).collect();
            entries.sort_by_key(|(_, last_accessed)| *last_accessed);
            
            let to_remove = entries.len() - self.max_entries + 1;
            for (key, _) in entries.iter().take(to_remove) {
                storage.remove(key);
            }
        }
        
        Ok(())
    }
}

impl CacheBackend for MemoryCacheBackend {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            let mut storage = self.storage.write().await;
            
            if let Some(entry) = storage.get_mut(key) {
                if entry.is_expired() {
                    storage.remove(key);
                    Ok(None)
                } else {
                    entry.touch();
                    Ok(Some(entry.value.clone()))
                }
            } else {
                Ok(None)
            }
        })
    }
    
    fn set(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> Result<()> {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            self.evict_if_needed().await?;
            
            let mut storage = self.storage.write().await;
            let entry = CacheEntry::new(value.to_vec(), ttl, value.len());
            storage.insert(key.to_string(), entry);
            Ok(())
        })
    }
    
    fn delete(&self, key: &str) -> Result<()> {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            let mut storage = self.storage.write().await;
            storage.remove(key);
            Ok(())
        })
    }
    
    fn clear(&self) -> Result<()> {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            let mut storage = self.storage.write().await;
            storage.clear();
            Ok(())
        })
    }
    
    fn size(&self) -> usize {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            let storage = self.storage.read().await;
            storage.len()
        })
    }
    
    fn keys(&self) -> Vec<String> {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            let storage = self.storage.read().await;
            storage.keys().cloned().collect()
        })
    }
}

impl CacheManager {
    pub fn new(config: CacheConfig) -> Self {
        let l1_cache = Arc::new(MemoryCacheBackend::new(config.l1_max_entries));
        
        Self {
            l1_cache,
            l2_cache: None,
            config,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }
    
    pub fn with_l2_cache(mut self, l2_cache: Arc<dyn CacheBackend + Send + Sync>) -> Self {
        self.l2_cache = Some(l2_cache);
        self
    }
    
    /// 获取缓存值
    pub async fn get<T>(&self, key: &str) -> Result<Option<T>>
    where
        T: DeserializeOwned,
    {
        let start_time = Instant::now();
        
        // 尝试L1缓存
        if let Some(data) = self.l1_cache.get(key)? {
            let value: T = self.deserialize(&data)?;
            
            // 更新统计
            {
                let mut stats = self.stats.write().await;
                stats.l1_hits += 1;
                stats.total_gets += 1;
                stats.avg_get_time_ms = self.update_avg_time(stats.avg_get_time_ms, stats.total_gets, start_time.elapsed());
                self.update_hit_rates(&mut stats);
            }
            
            debug!("L1 cache hit for key: {}", key);
            return Ok(Some(value));
        }
        
        // L1缓存未命中，尝试L2缓存
        if let Some(l2_cache) = &self.l2_cache {
            if let Some(data) = l2_cache.get(key)? {
                let value: T = self.deserialize(&data)?;
                
                // 回写到L1缓存
                if self.config.write_through {
                    let _ = self.l1_cache.set(key, &data, Some(self.config.l1_ttl_default));
                }
                
                // 更新统计
                {
                    let mut stats = self.stats.write().await;
                    stats.l1_misses += 1;
                    stats.l2_hits += 1;
                    stats.total_gets += 1;
                    stats.avg_get_time_ms = self.update_avg_time(stats.avg_get_time_ms, stats.total_gets, start_time.elapsed());
                    self.update_hit_rates(&mut stats);
                }
                
                debug!("L2 cache hit for key: {}", key);
                return Ok(Some(value));
            }
        }
        
        // 缓存完全未命中
        {
            let mut stats = self.stats.write().await;
            stats.l1_misses += 1;
            stats.l2_misses += 1;
            stats.total_gets += 1;
            stats.avg_get_time_ms = self.update_avg_time(stats.avg_get_time_ms, stats.total_gets, start_time.elapsed());
            self.update_hit_rates(&mut stats);
        }
        
        debug!("Cache miss for key: {}", key);
        Ok(None)
    }
    
    /// 设置缓存值
    pub async fn set<T>(&self, key: &str, value: &T, ttl: Option<Duration>) -> Result<()>
    where
        T: Serialize,
    {
        let start_time = Instant::now();
        let data = self.serialize(value)?;
        
        // 写入L1缓存
        let l1_ttl = ttl.unwrap_or(self.config.l1_ttl_default);
        self.l1_cache.set(key, &data, Some(l1_ttl))?;
        
        // 写入L2缓存
        if let Some(l2_cache) = &self.l2_cache {
            let l2_ttl = ttl.unwrap_or(self.config.l2_ttl_default);
            l2_cache.set(key, &data, Some(l2_ttl))?;
        }
        
        // 更新统计
        {
            let mut stats = self.stats.write().await;
            stats.total_sets += 1;
            stats.avg_set_time_ms = self.update_avg_time(stats.avg_set_time_ms, stats.total_sets, start_time.elapsed());
        }
        
        debug!("Cached value for key: {}", key);
        Ok(())
    }
    
    /// 删除缓存值
    pub async fn delete(&self, key: &str) -> Result<()> {
        self.l1_cache.delete(key)?;
        
        if let Some(l2_cache) = &self.l2_cache {
            l2_cache.delete(key)?;
        }
        
        Ok(())
    }
    
    /// 清空缓存
    pub async fn clear(&self) -> Result<()> {
        self.l1_cache.clear()?;
        
        if let Some(l2_cache) = &self.l2_cache {
            l2_cache.clear()?;
        }
        
        Ok(())
    }
    
    /// 批量获取
    pub async fn mget<T>(&self, keys: &[String]) -> Result<HashMap<String, T>>
    where
        T: DeserializeOwned,
    {
        let mut results = HashMap::new();
        
        for key in keys {
            if let Some(value) = self.get(key).await? {
                results.insert(key.clone(), value);
            }
        }
        
        Ok(results)
    }
    
    /// 批量设置
    pub async fn mset<T>(&self, values: HashMap<String, T>, ttl: Option<Duration>) -> Result<()>
    where
        T: Serialize,
    {
        for (key, value) in values {
            self.set(&key, &value, ttl).await?;
        }
        
        Ok(())
    }
    
    /// 预取相关键
    pub async fn prefetch(&self, pattern: &str, limit: Option<usize>) -> Result<usize> {
        if !self.config.prefetch_enabled {
            return Ok(0);
        }
        
        let keys = self.l1_cache.keys();
        let matching_keys: Vec<String> = keys.into_iter()
            .filter(|key| key.contains(pattern))
            .take(limit.unwrap_or(100))
            .collect();
        
        // 简单的预取逻辑 - 确保这些键在L1缓存中
        let mut prefetched = 0;
        for key in matching_keys {
            if self.l1_cache.get(&key)?.is_some() {
                prefetched += 1;
            }
        }
        
        Ok(prefetched)
    }
    
    /// 获取缓存统计
    pub async fn get_stats(&self) -> CacheStats {
        let stats = self.stats.read().await;
        stats.clone()
    }
    
    /// 获取缓存信息
    pub async fn get_cache_info(&self) -> CacheInfo {
        let l1_size = self.l1_cache.size();
        let l2_size = self.l2_cache.as_ref().map(|cache| cache.size()).unwrap_or(0);
        
        CacheInfo {
            l1_entries: l1_size,
            l2_entries: l2_size,
            total_entries: l1_size + l2_size,
            l1_max_entries: self.config.l1_max_entries,
            l1_usage_percent: (l1_size as f64 / self.config.l1_max_entries as f64) * 100.0,
            compression_enabled: self.config.compression_enabled,
            eviction_policy: self.config.eviction_policy.clone(),
        }
    }
    
    /// 序列化数据
    fn serialize<T>(&self, value: &T) -> Result<Vec<u8>>
    where
        T: Serialize,
    {
        let data = bincode::serialize(value)?;
        
        // 压缩大数据
        if self.config.compression_enabled && data.len() > self.config.compression_threshold {
            self.compress(&data)
        } else {
            Ok(data)
        }
    }
    
    /// 反序列化数据
    fn deserialize<T>(&self, data: &[u8]) -> Result<T>
    where
        T: DeserializeOwned,
    {
        // 检查是否需要解压缩
        let actual_data = if self.is_compressed(data) {
            self.decompress(data)?
        } else {
            data.to_vec()
        };
        
        bincode::deserialize(&actual_data).map_err(Into::into)
    }
    
    /// 压缩数据
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::Compression;
        use flate2::write::GzEncoder;
        use std::io::Write;
        
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)?;
        let compressed = encoder.finish()?;
        
        // 添加压缩标记
        let mut result = vec![1u8]; // 压缩标记
        result.extend(compressed);
        
        Ok(result)
    }
    
    /// 解压缩数据
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::GzDecoder;
        use std::io::Read;
        
        if data.is_empty() || data[0] != 1u8 {
            return Ok(data.to_vec());
        }
        
        let compressed_data = &data[1..]; // 跳过压缩标记
        let mut decoder = GzDecoder::new(compressed_data);
        let mut result = Vec::new();
        decoder.read_to_end(&mut result)?;
        
        Ok(result)
    }
    
    /// 检查是否压缩
    fn is_compressed(&self, data: &[u8]) -> bool {
        !data.is_empty() && data[0] == 1u8
    }
    
    /// 更新平均时间
    fn update_avg_time(&self, current_avg: f64, total_count: u64, elapsed: Duration) -> f64 {
        let new_time_ms = elapsed.as_secs_f64() * 1000.0;
        if total_count == 1 {
            new_time_ms
        } else {
            (current_avg * (total_count - 1) as f64 + new_time_ms) / total_count as f64
        }
    }
    
    /// 更新命中率
    fn update_hit_rates(&self, stats: &mut CacheStats) {
        let total_gets = stats.total_gets;
        if total_gets > 0 {
            stats.l1_hit_rate = stats.l1_hits as f64 / total_gets as f64;
            stats.l2_hit_rate = stats.l2_hits as f64 / total_gets as f64;
            stats.hit_rate = (stats.l1_hits + stats.l2_hits) as f64 / total_gets as f64;
        }
    }
}

/// 内存缓存管理器（简化版本，用于测试）
#[derive(Debug)]
pub struct MemoryCacheManager {
    backend: Arc<MemoryCacheBackend>,
}

impl MemoryCacheManager {
    pub fn new(max_entries: usize) -> Self {
        Self {
            backend: Arc::new(MemoryCacheBackend::new(max_entries)),
        }
    }
    
    pub async fn get<T>(&self, key: &str) -> Result<Option<T>>
    where
        T: DeserializeOwned,
    {
        if let Some(data) = self.backend.get(key)? {
            let value: T = bincode::deserialize(&data)?;
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }
    
    pub async fn set<T>(&self, key: &str, value: &T, ttl: Option<Duration>) -> Result<()>
    where
        T: Serialize,
    {
        let data = bincode::serialize(value)?;
        self.backend.set(key, &data, ttl)
    }
}

/// 缓存信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheInfo {
    pub l1_entries: usize,
    pub l2_entries: usize,
    pub total_entries: usize,
    pub l1_max_entries: usize,
    pub l1_usage_percent: f64,
    pub compression_enabled: bool,
    pub eviction_policy: EvictionPolicy,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_cache_basic() {
        let cache = MemoryCacheManager::new(100);
        
        // 测试设置和获取
        let key = "test_key";
        let value = "test_value".to_string();
        
        cache.set(key, &value, None).await.unwrap();
        let retrieved: Option<String> = cache.get(key).await.unwrap();
        
        assert_eq!(retrieved, Some(value));
    }
    
    #[tokio::test]
    async fn test_two_level_cache() {
        let config = CacheConfig::default();
        let cache = CacheManager::new(config);
        
        let key = "test_key";
        let value = vec![1, 2, 3, 4, 5];
        
        cache.set(key, &value, None).await.unwrap();
        let retrieved: Option<Vec<i32>> = cache.get(key).await.unwrap();
        
        assert_eq!(retrieved, Some(value));
    }
    
    #[tokio::test]
    async fn test_cache_stats() {
        let config = CacheConfig::default();
        let cache = CacheManager::new(config);
        
        // 缓存未命中
        let _: Option<String> = cache.get("missing_key").await.unwrap();
        
        // 缓存命中
        cache.set("existing_key", &"value", None).await.unwrap();
        let _: Option<String> = cache.get("existing_key").await.unwrap();
        
        let stats = cache.get_stats().await;
        assert!(stats.total_gets >= 2);
        assert!(stats.total_sets >= 1);
        assert!(stats.l1_hits >= 1);
        assert!(stats.l1_misses >= 1);
    }
    
    #[tokio::test]
    async fn test_batch_operations() {
        let cache = MemoryCacheManager::new(100);
        
        let mut values = HashMap::new();
        values.insert("key1".to_string(), "value1".to_string());
        values.insert("key2".to_string(), "value2".to_string());
        
        // 批量设置
        for (key, value) in &values {
            cache.set(key, value, None).await.unwrap();
        }
        
        // 批量获取
        for (key, expected_value) in &values {
            let retrieved: Option<String> = cache.get(key).await.unwrap();
            assert_eq!(retrieved, Some(expected_value.clone()));
        }
    }
}