//! 高性能分布式缓存系统
//! 支持Redis Cluster、一致性哈希等

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use chrono::{DateTime, Utc, Duration};
use anyhow::{Result, Context};
use async_trait::async_trait;

/// 缓存键
pub type CacheKey = String;

/// 缓存值
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheValue {
    pub data: Vec<u8>,
    pub expiry: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub access_count: u64,
    pub last_access: DateTime<Utc>,
}

impl CacheValue {
    pub fn new(data: Vec<u8>, ttl: Option<Duration>) -> Self {
        let now = Utc::now();
        Self {
            data,
            expiry: ttl.map(|duration| now + duration),
            created_at: now,
            access_count: 1,
            last_access: now,
        }
    }

    pub fn is_expired(&self) -> bool {
        if let Some(expiry) = self.expiry {
            Utc::now() > expiry
        } else {
            false
        }
    }

    pub fn access(&mut self) {
        self.access_count += 1;
        self.last_access = Utc::now();
    }
}

/// 缓存统计信息
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub sets: u64,
    pub deletes: u64,
    pub evictions: u64,
    pub memory_usage_bytes: u64,
    pub key_count: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f64 / (self.hits + self.misses) as f64
        }
    }

    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }
}

/// 缓存接口
#[async_trait]
pub trait Cache: Send + Sync {
    async fn get(&self, key: &CacheKey) -> Result<Option<CacheValue>>;
    async fn set(&self, key: CacheKey, value: CacheValue) -> Result<()>;
    async fn delete(&self, key: &CacheKey) -> Result<bool>;
    async fn exists(&self, key: &CacheKey) -> Result<bool>;
    async fn ttl(&self, key: &CacheKey) -> Result<Option<Duration>>;
    async fn expire(&self, key: &CacheKey, ttl: Duration) -> Result<bool>;
    async fn clear(&self) -> Result<()>;
    async fn stats(&self) -> Result<CacheStats>;
    
    // 批量操作
    async fn mget(&self, keys: &[CacheKey]) -> Result<Vec<Option<CacheValue>>>;
    async fn mset(&self, items: Vec<(CacheKey, CacheValue)>) -> Result<()>;
    async fn mdel(&self, keys: &[CacheKey]) -> Result<u64>;
    
    // 模式匹配
    async fn keys(&self, pattern: &str) -> Result<Vec<CacheKey>>;
    async fn scan(&self, cursor: u64, pattern: Option<&str>, count: Option<usize>) -> Result<(u64, Vec<CacheKey>)>;
}

/// 内存缓存实现
pub struct MemoryCache {
    data: Arc<RwLock<HashMap<CacheKey, CacheValue>>>,
    stats: Arc<RwLock<CacheStats>>,
    max_size: usize,
    max_memory: u64,
}

impl MemoryCache {
    pub fn new(max_size: usize, max_memory: u64) -> Self {
        let cache = Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CacheStats::default())),
            max_size,
            max_memory,
        };
        
        // 启动清理任务
        cache.start_cleanup_task();
        cache
    }

    fn start_cleanup_task(&self) {
        let data = Arc::clone(&self.data);
        let stats = Arc::clone(&self.stats);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                let now = Utc::now();
                let mut data_guard = data.write().await;
                let mut stats_guard = stats.write().await;
                
                // 移除过期键
                let expired_keys: Vec<_> = data_guard
                    .iter()
                    .filter(|(_, value)| value.is_expired())
                    .map(|(key, _)| key.clone())
                    .collect();
                
                for key in expired_keys {
                    data_guard.remove(&key);
                    stats_guard.evictions += 1;
                }
                
                // 更新统计
                stats_guard.key_count = data_guard.len() as u64;
                stats_guard.memory_usage_bytes = Self::calculate_memory_usage(&data_guard);
            }
        });
    }

    fn calculate_memory_usage(data: &HashMap<CacheKey, CacheValue>) -> u64 {
        data.iter()
            .map(|(key, value)| key.len() + value.data.len())
            .sum::<usize>() as u64
    }

    async fn evict_if_needed(&self) -> Result<()> {
        let mut data_guard = self.data.write().await;
        let mut stats_guard = self.stats.write().await;
        
        // 检查大小限制
        while data_guard.len() > self.max_size {
            if let Some((key_to_evict, _)) = data_guard
                .iter()
                .min_by_key(|(_, value)| value.last_access)
                .map(|(k, v)| (k.clone(), v.clone()))
            {
                data_guard.remove(&key_to_evict);
                stats_guard.evictions += 1;
            } else {
                break;
            }
        }
        
        // 检查内存限制
        let memory_usage = Self::calculate_memory_usage(&data_guard);
        while memory_usage > self.max_memory && !data_guard.is_empty() {
            if let Some((key_to_evict, _)) = data_guard
                .iter()
                .min_by_key(|(_, value)| value.last_access)
                .map(|(k, v)| (k.clone(), v.clone()))
            {
                data_guard.remove(&key_to_evict);
                stats_guard.evictions += 1;
            } else {
                break;
            }
        }
        
        Ok(())
    }
}

#[async_trait]
impl Cache for MemoryCache {
    async fn get(&self, key: &CacheKey) -> Result<Option<CacheValue>> {
        let mut data_guard = self.data.write().await;
        let mut stats_guard = self.stats.write().await;

        if let Some(mut value) = data_guard.get_mut(key) {
            if value.is_expired() {
                data_guard.remove(key);
                stats_guard.misses += 1;
                return Ok(None);
            }
            
            value.access();
            stats_guard.hits += 1;
            Ok(Some(value.clone()))
        } else {
            stats_guard.misses += 1;
            Ok(None)
        }
    }

    async fn set(&self, key: CacheKey, value: CacheValue) -> Result<()> {
        {
            let mut data_guard = self.data.write().await;
            let mut stats_guard = self.stats.write().await;
            
            data_guard.insert(key, value);
            stats_guard.sets += 1;
        }
        
        self.evict_if_needed().await?;
        Ok(())
    }

    async fn delete(&self, key: &CacheKey) -> Result<bool> {
        let mut data_guard = self.data.write().await;
        let mut stats_guard = self.stats.write().await;
        
        let existed = data_guard.remove(key).is_some();
        if existed {
            stats_guard.deletes += 1;
        }
        
        Ok(existed)
    }

    async fn exists(&self, key: &CacheKey) -> Result<bool> {
        let data_guard = self.data.read().await;
        
        if let Some(value) = data_guard.get(key) {
            Ok(!value.is_expired())
        } else {
            Ok(false)
        }
    }

    async fn ttl(&self, key: &CacheKey) -> Result<Option<Duration>> {
        let data_guard = self.data.read().await;
        
        if let Some(value) = data_guard.get(key) {
            if let Some(expiry) = value.expiry {
                let remaining = expiry - Utc::now();
                if remaining > Duration::zero() {
                    Ok(Some(remaining))
                } else {
                    Ok(Some(Duration::zero()))
                }
            } else {
                Ok(None) // 没有过期时间
            }
        } else {
            Err(anyhow::anyhow!("Key not found"))
        }
    }

    async fn expire(&self, key: &CacheKey, ttl: Duration) -> Result<bool> {
        let mut data_guard = self.data.write().await;
        
        if let Some(value) = data_guard.get_mut(key) {
            value.expiry = Some(Utc::now() + ttl);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn clear(&self) -> Result<()> {
        let mut data_guard = self.data.write().await;
        let mut stats_guard = self.stats.write().await;
        
        data_guard.clear();
        *stats_guard = CacheStats::default();
        
        Ok(())
    }

    async fn stats(&self) -> Result<CacheStats> {
        let stats_guard = self.stats.read().await;
        Ok(stats_guard.clone())
    }

    async fn mget(&self, keys: &[CacheKey]) -> Result<Vec<Option<CacheValue>>> {
        let mut results = Vec::with_capacity(keys.len());
        
        for key in keys {
            results.push(self.get(key).await?);
        }
        
        Ok(results)
    }

    async fn mset(&self, items: Vec<(CacheKey, CacheValue)>) -> Result<()> {
        for (key, value) in items {
            self.set(key, value).await?;
        }
        Ok(())
    }

    async fn mdel(&self, keys: &[CacheKey]) -> Result<u64> {
        let mut deleted_count = 0;
        
        for key in keys {
            if self.delete(key).await? {
                deleted_count += 1;
            }
        }
        
        Ok(deleted_count)
    }

    async fn keys(&self, pattern: &str) -> Result<Vec<CacheKey>> {
        let data_guard = self.data.read().await;
        let regex = globset::Glob::new(pattern)
            .context("Invalid pattern")?
            .compile_matcher();

        let matching_keys: Vec<_> = data_guard
            .keys()
            .filter(|key| regex.is_match(key))
            .cloned()
            .collect();

        Ok(matching_keys)
    }

    async fn scan(&self, cursor: u64, pattern: Option<&str>, count: Option<usize>) -> Result<(u64, Vec<CacheKey>)> {
        let data_guard = self.data.read().await;
        let all_keys: Vec<_> = data_guard.keys().cloned().collect();
        
        let start = cursor as usize;
        let batch_size = count.unwrap_or(10);
        let end = std::cmp::min(start + batch_size, all_keys.len());
        
        let mut batch_keys = Vec::new();
        for i in start..end {
            let key = &all_keys[i];
            
            // 应用模式过滤
            if let Some(pattern_str) = pattern {
                let regex = globset::Glob::new(pattern_str)
                    .context("Invalid pattern")?
                    .compile_matcher();
                
                if regex.is_match(key) {
                    batch_keys.push(key.clone());
                }
            } else {
                batch_keys.push(key.clone());
            }
        }
        
        let next_cursor = if end >= all_keys.len() { 0 } else { end as u64 };
        Ok((next_cursor, batch_keys))
    }
}

/// Redis缓存实现
pub struct RedisCache {
    client: redis::Client,
    connection_pool: deadpool_redis::Pool,
    prefix: String,
}

impl RedisCache {
    pub async fn new(
        connection_string: &str,
        pool_size: usize,
        prefix: String,
    ) -> Result<Self> {
        let client = redis::Client::open(connection_string)
            .context("Failed to create Redis client")?;

        let config = deadpool_redis::Config::from_url(connection_string);
        let pool = config.create_pool(Some(deadpool_redis::Runtime::Tokio1))
            .context("Failed to create connection pool")?;

        Ok(Self {
            client,
            connection_pool: pool,
            prefix,
        })
    }

    fn make_key(&self, key: &CacheKey) -> String {
        if self.prefix.is_empty() {
            key.clone()
        } else {
            format!("{}:{}", self.prefix, key)
        }
    }
}

#[async_trait]
impl Cache for RedisCache {
    async fn get(&self, key: &CacheKey) -> Result<Option<CacheValue>> {
        let mut conn = self.connection_pool.get().await
            .context("Failed to get Redis connection")?;
        
        let redis_key = self.make_key(key);
        let data: Option<Vec<u8>> = redis::cmd("GET")
            .arg(&redis_key)
            .query_async(&mut *conn)
            .await
            .context("Failed to execute GET command")?;

        if let Some(serialized_data) = data {
            let value: CacheValue = bincode::deserialize(&serialized_data)
                .context("Failed to deserialize cache value")?;
            
            if value.is_expired() {
                // 删除过期的键
                let _: () = redis::cmd("DEL")
                    .arg(&redis_key)
                    .query_async(&mut *conn)
                    .await
                    .context("Failed to delete expired key")?;
                
                Ok(None)
            } else {
                Ok(Some(value))
            }
        } else {
            Ok(None)
        }
    }

    async fn set(&self, key: CacheKey, value: CacheValue) -> Result<()> {
        let mut conn = self.connection_pool.get().await
            .context("Failed to get Redis connection")?;
        
        let redis_key = self.make_key(&key);
        let serialized_value = bincode::serialize(&value)
            .context("Failed to serialize cache value")?;

        if let Some(expiry) = value.expiry {
            let ttl_seconds = (expiry - Utc::now()).num_seconds().max(1);
            
            let _: () = redis::cmd("SETEX")
                .arg(&redis_key)
                .arg(ttl_seconds)
                .arg(&serialized_value)
                .query_async(&mut *conn)
                .await
                .context("Failed to execute SETEX command")?;
        } else {
            let _: () = redis::cmd("SET")
                .arg(&redis_key)
                .arg(&serialized_value)
                .query_async(&mut *conn)
                .await
                .context("Failed to execute SET command")?;
        }

        Ok(())
    }

    async fn delete(&self, key: &CacheKey) -> Result<bool> {
        let mut conn = self.connection_pool.get().await
            .context("Failed to get Redis connection")?;
        
        let redis_key = self.make_key(key);
        let deleted_count: u64 = redis::cmd("DEL")
            .arg(&redis_key)
            .query_async(&mut *conn)
            .await
            .context("Failed to execute DEL command")?;

        Ok(deleted_count > 0)
    }

    async fn exists(&self, key: &CacheKey) -> Result<bool> {
        let mut conn = self.connection_pool.get().await
            .context("Failed to get Redis connection")?;
        
        let redis_key = self.make_key(key);
        let exists: bool = redis::cmd("EXISTS")
            .arg(&redis_key)
            .query_async(&mut *conn)
            .await
            .context("Failed to execute EXISTS command")?;

        Ok(exists)
    }

    async fn ttl(&self, key: &CacheKey) -> Result<Option<Duration>> {
        let mut conn = self.connection_pool.get().await
            .context("Failed to get Redis connection")?;
        
        let redis_key = self.make_key(key);
        let ttl_seconds: i64 = redis::cmd("TTL")
            .arg(&redis_key)
            .query_async(&mut *conn)
            .await
            .context("Failed to execute TTL command")?;

        match ttl_seconds {
            -2 => Err(anyhow::anyhow!("Key does not exist")),
            -1 => Ok(None), // 没有过期时间
            seconds => Ok(Some(Duration::seconds(seconds))),
        }
    }

    async fn expire(&self, key: &CacheKey, ttl: Duration) -> Result<bool> {
        let mut conn = self.connection_pool.get().await
            .context("Failed to get Redis connection")?;
        
        let redis_key = self.make_key(key);
        let ttl_seconds = ttl.num_seconds().max(1);
        
        let success: bool = redis::cmd("EXPIRE")
            .arg(&redis_key)
            .arg(ttl_seconds)
            .query_async(&mut *conn)
            .await
            .context("Failed to execute EXPIRE command")?;

        Ok(success)
    }

    async fn clear(&self) -> Result<()> {
        let mut conn = self.connection_pool.get().await
            .context("Failed to get Redis connection")?;
        
        if self.prefix.is_empty() {
            // 清空整个数据库（危险操作）
            let _: () = redis::cmd("FLUSHDB")
                .query_async(&mut *conn)
                .await
                .context("Failed to execute FLUSHDB command")?;
        } else {
            // 只删除带前缀的键
            let pattern = format!("{}:*", self.prefix);
            let keys: Vec<String> = redis::cmd("KEYS")
                .arg(&pattern)
                .query_async(&mut *conn)
                .await
                .context("Failed to get keys with prefix")?;
            
            if !keys.is_empty() {
                let _: () = redis::cmd("DEL")
                    .arg(&keys)
                    .query_async(&mut *conn)
                    .await
                    .context("Failed to delete prefixed keys")?;
            }
        }

        Ok(())
    }

    async fn stats(&self) -> Result<CacheStats> {
        let mut conn = self.connection_pool.get().await
            .context("Failed to get Redis connection")?;
        
        let info: String = redis::cmd("INFO")
            .arg("memory")
            .query_async(&mut *conn)
            .await
            .context("Failed to get Redis INFO")?;

        // 解析Redis INFO输出（简化实现）
        let memory_usage = Self::parse_redis_memory(&info);
        
        Ok(CacheStats {
            hits: 0, // 需要从Redis INFO中解析
            misses: 0, // 需要从Redis INFO中解析  
            sets: 0,
            deletes: 0,
            evictions: 0,
            memory_usage_bytes: memory_usage,
            key_count: 0, // 需要使用DBSIZE命令获取
        })
    }

    async fn mget(&self, keys: &[CacheKey]) -> Result<Vec<Option<CacheValue>>> {
        let mut conn = self.connection_pool.get().await
            .context("Failed to get Redis connection")?;
        
        let redis_keys: Vec<String> = keys.iter().map(|k| self.make_key(k)).collect();
        let values: Vec<Option<Vec<u8>>> = redis::cmd("MGET")
            .arg(&redis_keys)
            .query_async(&mut *conn)
            .await
            .context("Failed to execute MGET command")?;

        let mut results = Vec::with_capacity(values.len());
        for value_data in values {
            if let Some(data) = value_data {
                match bincode::deserialize::<CacheValue>(&data) {
                    Ok(value) => {
                        if value.is_expired() {
                            results.push(None);
                        } else {
                            results.push(Some(value));
                        }
                    }
                    Err(_) => results.push(None),
                }
            } else {
                results.push(None);
            }
        }

        Ok(results)
    }

    async fn mset(&self, items: Vec<(CacheKey, CacheValue)>) -> Result<()> {
        let mut conn = self.connection_pool.get().await
            .context("Failed to get Redis connection")?;

        let mut pipe = redis::pipe();
        
        for (key, value) in items {
            let redis_key = self.make_key(&key);
            let serialized_value = bincode::serialize(&value)
                .context("Failed to serialize cache value")?;

            if let Some(expiry) = value.expiry {
                let ttl_seconds = (expiry - Utc::now()).num_seconds().max(1);
                pipe.cmd("SETEX").arg(&redis_key).arg(ttl_seconds).arg(&serialized_value);
            } else {
                pipe.cmd("SET").arg(&redis_key).arg(&serialized_value);
            }
        }

        pipe.query_async(&mut *conn)
            .await
            .context("Failed to execute pipeline")?;

        Ok(())
    }

    async fn mdel(&self, keys: &[CacheKey]) -> Result<u64> {
        let mut conn = self.connection_pool.get().await
            .context("Failed to get Redis connection")?;
        
        let redis_keys: Vec<String> = keys.iter().map(|k| self.make_key(k)).collect();
        let deleted_count: u64 = redis::cmd("DEL")
            .arg(&redis_keys)
            .query_async(&mut *conn)
            .await
            .context("Failed to execute DEL command")?;

        Ok(deleted_count)
    }

    async fn keys(&self, pattern: &str) -> Result<Vec<CacheKey>> {
        let mut conn = self.connection_pool.get().await
            .context("Failed to get Redis connection")?;
        
        let search_pattern = if self.prefix.is_empty() {
            pattern.to_string()
        } else {
            format!("{}:{}", self.prefix, pattern)
        };

        let keys: Vec<String> = redis::cmd("KEYS")
            .arg(&search_pattern)
            .query_async(&mut *conn)
            .await
            .context("Failed to execute KEYS command")?;

        // 移除前缀
        let result_keys = if self.prefix.is_empty() {
            keys
        } else {
            keys.into_iter()
                .filter_map(|key| key.strip_prefix(&format!("{}:", self.prefix)).map(|s| s.to_string()))
                .collect()
        };

        Ok(result_keys)
    }

    async fn scan(&self, cursor: u64, pattern: Option<&str>, count: Option<usize>) -> Result<(u64, Vec<CacheKey>)> {
        let mut conn = self.connection_pool.get().await
            .context("Failed to get Redis connection")?;
        
        let mut cmd = redis::cmd("SCAN");
        cmd.arg(cursor);
        
        if let Some(pattern_str) = pattern {
            let search_pattern = if self.prefix.is_empty() {
                pattern_str.to_string()
            } else {
                format!("{}:{}", self.prefix, pattern_str)
            };
            cmd.arg("MATCH").arg(&search_pattern);
        }
        
        if let Some(count_val) = count {
            cmd.arg("COUNT").arg(count_val);
        }

        let (next_cursor, keys): (u64, Vec<String>) = cmd
            .query_async(&mut *conn)
            .await
            .context("Failed to execute SCAN command")?;

        // 移除前缀
        let result_keys = if self.prefix.is_empty() {
            keys
        } else {
            keys.into_iter()
                .filter_map(|key| key.strip_prefix(&format!("{}:", self.prefix)).map(|s| s.to_string()))
                .collect()
        };

        Ok((next_cursor, result_keys))
    }
}

impl RedisCache {
    fn parse_redis_memory(info: &str) -> u64 {
        // 简化的Redis内存解析
        for line in info.lines() {
            if line.starts_with("used_memory:") {
                if let Some(value_str) = line.split(':').nth(1) {
                    return value_str.parse().unwrap_or(0);
                }
            }
        }
        0
    }
}

/// 类型化缓存包装器
pub struct TypedCache<T>
where
    T: Serialize + DeserializeOwned,
{
    cache: Arc<dyn Cache>,
    key_prefix: String,
    default_ttl: Option<Duration>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> TypedCache<T>
where
    T: Serialize + DeserializeOwned,
{
    pub fn new(cache: Arc<dyn Cache>, key_prefix: String, default_ttl: Option<Duration>) -> Self {
        Self {
            cache,
            key_prefix,
            default_ttl,
            _phantom: std::marker::PhantomData,
        }
    }

    fn make_key(&self, key: &str) -> String {
        if self.key_prefix.is_empty() {
            key.to_string()
        } else {
            format!("{}:{}", self.key_prefix, key)
        }
    }

    pub async fn get(&self, key: &str) -> Result<Option<T>> {
        let cache_key = self.make_key(key);
        
        if let Some(cache_value) = self.cache.get(&cache_key).await? {
            let obj: T = serde_json::from_slice(&cache_value.data)
                .context("Failed to deserialize cached object")?;
            Ok(Some(obj))
        } else {
            Ok(None)
        }
    }

    pub async fn set(&self, key: &str, value: &T) -> Result<()> {
        let cache_key = self.make_key(key);
        let serialized_data = serde_json::to_vec(value)
            .context("Failed to serialize object")?;
        
        let cache_value = CacheValue::new(serialized_data, self.default_ttl);
        self.cache.set(cache_key, cache_value).await
    }

    pub async fn set_with_ttl(&self, key: &str, value: &T, ttl: Duration) -> Result<()> {
        let cache_key = self.make_key(key);
        let serialized_data = serde_json::to_vec(value)
            .context("Failed to serialize object")?;
        
        let cache_value = CacheValue::new(serialized_data, Some(ttl));
        self.cache.set(cache_key, cache_value).await
    }

    pub async fn delete(&self, key: &str) -> Result<bool> {
        let cache_key = self.make_key(key);
        self.cache.delete(&cache_key).await
    }

    pub async fn exists(&self, key: &str) -> Result<bool> {
        let cache_key = self.make_key(key);
        self.cache.exists(&cache_key).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestData {
        pub id: u64,
        pub name: String,
        pub value: f64,
    }

    #[tokio::test]
    async fn test_memory_cache_basic_operations() {
        let cache = MemoryCache::new(1000, 1024 * 1024);
        
        let test_data = b"test value".to_vec();
        let cache_value = CacheValue::new(test_data.clone(), Some(Duration::seconds(60)));
        
        // 测试SET和GET
        cache.set("test_key".to_string(), cache_value).await.unwrap();
        let result = cache.get(&"test_key".to_string()).await.unwrap();
        
        assert!(result.is_some());
        assert_eq!(result.unwrap().data, test_data);
        
        // 测试EXISTS
        assert!(cache.exists(&"test_key".to_string()).await.unwrap());
        assert!(!cache.exists(&"nonexistent_key".to_string()).await.unwrap());
        
        // 测试DELETE
        assert!(cache.delete(&"test_key".to_string()).await.unwrap());
        assert!(!cache.delete(&"test_key".to_string()).await.unwrap());
        assert!(cache.get(&"test_key".to_string()).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_memory_cache_expiration() {
        let cache = MemoryCache::new(1000, 1024 * 1024);
        
        let test_data = b"test value".to_vec();
        let cache_value = CacheValue::new(test_data, Some(Duration::milliseconds(100)));
        
        cache.set("expire_key".to_string(), cache_value).await.unwrap();
        
        // 立即获取应该成功
        assert!(cache.get(&"expire_key".to_string()).await.unwrap().is_some());
        
        // 等待过期
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        
        // 过期后应该返回None
        assert!(cache.get(&"expire_key".to_string()).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_memory_cache_batch_operations() {
        let cache = MemoryCache::new(1000, 1024 * 1024);
        
        // 批量设置
        let items = vec![
            ("key1".to_string(), CacheValue::new(b"value1".to_vec(), None)),
            ("key2".to_string(), CacheValue::new(b"value2".to_vec(), None)),
            ("key3".to_string(), CacheValue::new(b"value3".to_vec(), None)),
        ];
        
        cache.mset(items).await.unwrap();
        
        // 批量获取
        let keys = vec!["key1".to_string(), "key2".to_string(), "key3".to_string(), "key4".to_string()];
        let results = cache.mget(&keys).await.unwrap();
        
        assert_eq!(results.len(), 4);
        assert!(results[0].is_some());
        assert!(results[1].is_some());
        assert!(results[2].is_some());
        assert!(results[3].is_none());
        
        // 批量删除
        let deleted_count = cache.mdel(&keys[0..2]).await.unwrap();
        assert_eq!(deleted_count, 2);
    }

    #[tokio::test]
    async fn test_memory_cache_pattern_matching() {
        let cache = MemoryCache::new(1000, 1024 * 1024);
        
        // 添加一些测试数据
        cache.set("user:1".to_string(), CacheValue::new(b"user1".to_vec(), None)).await.unwrap();
        cache.set("user:2".to_string(), CacheValue::new(b"user2".to_vec(), None)).await.unwrap();
        cache.set("post:1".to_string(), CacheValue::new(b"post1".to_vec(), None)).await.unwrap();
        
        // 测试模式匹配
        let user_keys = cache.keys("user:*").await.unwrap();
        assert_eq!(user_keys.len(), 2);
        assert!(user_keys.contains(&"user:1".to_string()));
        assert!(user_keys.contains(&"user:2".to_string()));
        
        // 测试扫描
        let (cursor, keys) = cache.scan(0, Some("user:*"), Some(10)).await.unwrap();
        assert_eq!(keys.len(), 2);
    }

    #[tokio::test]
    async fn test_typed_cache() {
        let memory_cache = Arc::new(MemoryCache::new(1000, 1024 * 1024));
        let typed_cache = TypedCache::<TestData>::new(
            memory_cache,
            "test".to_string(),
            Some(Duration::seconds(60)),
        );

        let test_data = TestData {
            id: 1,
            name: "Test".to_string(),
            value: 42.0,
        };

        // 测试类型化设置和获取
        typed_cache.set("data1", &test_data).await.unwrap();
        let retrieved = typed_cache.get("data1").await.unwrap();
        
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), test_data);
        
        // 测试不存在的键
        let missing = typed_cache.get("missing").await.unwrap();
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let cache = MemoryCache::new(1000, 1024 * 1024);
        
        // 执行一些操作
        cache.set("key1".to_string(), CacheValue::new(b"value1".to_vec(), None)).await.unwrap();
        let _ = cache.get(&"key1".to_string()).await;
        let _ = cache.get(&"missing_key".to_string()).await;
        
        let stats = cache.stats().await.unwrap();
        assert!(stats.hits > 0);
        assert!(stats.misses > 0);
        assert!(stats.sets > 0);
        assert!(stats.hit_rate() > 0.0 && stats.hit_rate() < 1.0);
    }
}

// 模拟Redis和连接池模块
#[allow(dead_code)]
mod redis {
    use anyhow::Result;
    
    pub struct Client;
    
    impl Client {
        pub fn open(_url: &str) -> Result<Self> {
            Ok(Self)
        }
    }
    
    pub fn cmd(command: &str) -> Cmd {
        Cmd { command: command.to_string() }
    }
    
    pub struct Cmd {
        command: String,
    }
    
    impl Cmd {
        pub fn arg<T>(self, _arg: T) -> Self { self }
        pub async fn query_async<T: Default>(&self, _conn: &mut deadpool_redis::Connection) -> Result<T> {
            Ok(T::default())
        }
    }
    
    pub fn pipe() -> Pipeline {
        Pipeline
    }
    
    pub struct Pipeline;
    
    impl Pipeline {
        pub fn cmd(&mut self, _command: &str) -> &mut Self { self }
        pub fn arg<T>(&mut self, _arg: T) -> &mut Self { self }
        pub async fn query_async(&mut self, _conn: &mut deadpool_redis::Connection) -> Result<()> {
            Ok(())
        }
    }
}

#[allow(dead_code)]
mod deadpool_redis {
    use anyhow::Result;
    
    pub struct Pool;
    pub struct Connection;
    pub struct Config;
    pub enum Runtime { Tokio1 }
    
    impl Pool {
        pub async fn get(&self) -> Result<Connection> {
            Ok(Connection)
        }
    }
    
    impl Config {
        pub fn from_url(_url: &str) -> Self { Self }
        pub fn create_pool(&self, _runtime: Option<Runtime>) -> Result<Pool> {
            Ok(Pool)
        }
    }
}

#[allow(dead_code)]
mod globset {
    use anyhow::Result;
    
    pub struct Glob;
    pub struct GlobMatcher;
    
    impl Glob {
        pub fn new(_pattern: &str) -> Result<Self> {
            Ok(Self)
        }
        
        pub fn compile_matcher(self) -> GlobMatcher {
            GlobMatcher
        }
    }
    
    impl GlobMatcher {
        pub fn is_match(&self, _text: &str) -> bool {
            true
        }
    }
}

#[allow(dead_code)]
mod bincode {
    use anyhow::Result;
    use serde::{Serialize, de::DeserializeOwned};
    
    pub fn serialize<T: Serialize>(_value: &T) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }
    
    pub fn deserialize<T: DeserializeOwned>(_data: &[u8]) -> Result<T> {
        Err(anyhow::anyhow!("Mock deserialization error"))
    }
}