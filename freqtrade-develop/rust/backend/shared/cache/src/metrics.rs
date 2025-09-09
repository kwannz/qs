use crate::{CacheProvider, CacheResult, CacheStats, CacheProviderType};
use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize, Deserialize};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration as StdDuration, Instant},
};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Comprehensive metrics tracking for cache operations
pub struct MetricsCache {
    /// Wrapped cache provider
    inner: CacheProviderType,
    /// Metrics collector
    metrics: Arc<CacheMetrics>,
    /// Configuration for metrics collection
    config: MetricsConfig,
}

/// Configuration for metrics collection
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Enable detailed per-key metrics (memory intensive)
    pub track_per_key_metrics: bool,
    /// Enable timing histograms for operations
    pub track_operation_timing: bool,
    /// Maximum number of keys to track individually
    pub max_tracked_keys: usize,
    /// Enable memory usage tracking
    pub track_memory_usage: bool,
    /// Histogram bucket boundaries for latency tracking (in microseconds)
    pub latency_buckets: Vec<u64>,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            track_per_key_metrics: false,
            track_operation_timing: true,
            max_tracked_keys: 1000,
            track_memory_usage: true,
            latency_buckets: vec![
                100,    // 0.1ms
                500,    // 0.5ms
                1000,   // 1ms
                5000,   // 5ms
                10000,  // 10ms
                50000,  // 50ms
                100000, // 100ms
                500000, // 500ms
                1000000, // 1s
            ],
        }
    }
}

/// Detailed metrics for cache operations
#[derive(Debug)]
pub struct CacheMetrics {
    // Basic counters
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub sets: AtomicU64,
    pub deletes: AtomicU64,
    pub errors: AtomicU64,
    pub evictions: AtomicU64,
    
    // Timing metrics (in microseconds)
    pub total_get_time: AtomicU64,
    pub total_set_time: AtomicU64,
    pub total_delete_time: AtomicU64,
    
    // Memory metrics
    pub estimated_memory_usage: AtomicU64,
    pub key_count: AtomicU64,
    
    // Per-operation metrics
    pub operation_latency_histogram: RwLock<HashMap<String, LatencyHistogram>>,
    
    // Per-key metrics (optional)
    pub key_metrics: RwLock<HashMap<String, KeyMetrics>>,
    
    // Cache effectiveness (using Mutex since AtomicF64 is not available)
    pub hit_rate: Mutex<f64>,
    pub miss_rate: Mutex<f64>,
}

/// Histogram for tracking operation latencies
#[derive(Debug, Clone)]
pub struct LatencyHistogram {
    pub buckets: HashMap<u64, u64>, // bucket_upper_bound -> count
    pub total_count: u64,
    pub sum: u64, // total time in microseconds
    pub min: u64,
    pub max: u64,
}

impl LatencyHistogram {
    fn new(bucket_bounds: &[u64]) -> Self {
        let mut buckets = HashMap::new();
        for &bound in bucket_bounds {
            buckets.insert(bound, 0);
        }
        
        Self {
            buckets,
            total_count: 0,
            sum: 0,
            min: u64::MAX,
            max: 0,
        }
    }
    
    fn record(&mut self, duration_micros: u64) {
        self.total_count += 1;
        self.sum += duration_micros;
        self.min = self.min.min(duration_micros);
        self.max = self.max.max(duration_micros);
        
        // Find appropriate bucket
        for (&bucket_bound, count) in &mut self.buckets {
            if duration_micros <= bucket_bound {
                *count += 1;
                break;
            }
        }
    }
    
    fn percentile(&self, p: f64) -> Option<u64> {
        if self.total_count == 0 {
            return None;
        }
        
        let target_count = (self.total_count as f64 * p) as u64;
        let mut cumulative = 0;
        
        let mut sorted_buckets: Vec<_> = self.buckets.iter().collect();
        sorted_buckets.sort_by_key(|(bound, _)| *bound);
        
        for (&bound, &count) in sorted_buckets {
            cumulative += count;
            if cumulative >= target_count {
                return Some(bound);
            }
        }
        
        None
    }
    
    fn average(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            self.sum as f64 / self.total_count as f64
        }
    }
}

/// Per-key metrics tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyMetrics {
    pub key: String,
    pub hit_count: u64,
    pub miss_count: u64,
    pub set_count: u64,
    pub delete_count: u64,
    pub last_accessed: Option<std::time::SystemTime>,
    pub estimated_size: u64,
    pub error_count: u64,
}

impl KeyMetrics {
    fn new(key: String) -> Self {
        Self {
            key,
            hit_count: 0,
            miss_count: 0,
            set_count: 0,
            delete_count: 0,
            last_accessed: None,
            estimated_size: 0,
            error_count: 0,
        }
    }
    
    fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            0.0
        } else {
            (self.hit_count as f64) / (total as f64) * 100.0
        }
    }
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            sets: AtomicU64::new(0),
            deletes: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            total_get_time: AtomicU64::new(0),
            total_set_time: AtomicU64::new(0),
            total_delete_time: AtomicU64::new(0),
            estimated_memory_usage: AtomicU64::new(0),
            key_count: AtomicU64::new(0),
            operation_latency_histogram: RwLock::new(HashMap::new()),
            key_metrics: RwLock::new(HashMap::new()),
            hit_rate: Mutex::new(0.0),
            miss_rate: Mutex::new(0.0),
        }
    }
}

impl CacheMetrics {
    /// Record a cache hit
    pub async fn record_hit(&self, key: &str, duration: StdDuration, config: &MetricsConfig) {
        self.hits.fetch_add(1, Ordering::Relaxed);
        
        let duration_micros = duration.as_micros() as u64;
        self.total_get_time.fetch_add(duration_micros, Ordering::Relaxed);
        
        if config.track_operation_timing {
            self.record_operation_latency("get", duration_micros).await;
        }
        
        if config.track_per_key_metrics {
            self.record_key_metric(key, "hit").await;
        }
        
        self.update_hit_rate().await;
    }
    
    /// Record a cache miss
    pub async fn record_miss(&self, key: &str, duration: StdDuration, config: &MetricsConfig) {
        self.misses.fetch_add(1, Ordering::Relaxed);
        
        let duration_micros = duration.as_micros() as u64;
        self.total_get_time.fetch_add(duration_micros, Ordering::Relaxed);
        
        if config.track_operation_timing {
            self.record_operation_latency("get", duration_micros).await;
        }
        
        if config.track_per_key_metrics {
            self.record_key_metric(key, "miss").await;
        }
        
        self.update_hit_rate().await;
    }
    
    /// Record a cache set operation
    pub async fn record_set(&self, key: &str, value_size: usize, duration: StdDuration, config: &MetricsConfig) {
        self.sets.fetch_add(1, Ordering::Relaxed);
        self.key_count.fetch_add(1, Ordering::Relaxed);
        
        if config.track_memory_usage {
            self.estimated_memory_usage.fetch_add(value_size as u64, Ordering::Relaxed);
        }
        
        let duration_micros = duration.as_micros() as u64;
        self.total_set_time.fetch_add(duration_micros, Ordering::Relaxed);
        
        if config.track_operation_timing {
            self.record_operation_latency("set", duration_micros).await;
        }
        
        if config.track_per_key_metrics {
            self.record_key_metric_with_size(key, "set", value_size).await;
        }
    }
    
    /// Record a cache delete operation
    pub async fn record_delete(&self, key: &str, duration: StdDuration, config: &MetricsConfig) {
        self.deletes.fetch_add(1, Ordering::Relaxed);
        
        let duration_micros = duration.as_micros() as u64;
        self.total_delete_time.fetch_add(duration_micros, Ordering::Relaxed);
        
        if config.track_operation_timing {
            self.record_operation_latency("delete", duration_micros).await;
        }
        
        if config.track_per_key_metrics {
            self.record_key_metric(key, "delete").await;
        }
    }
    
    /// Record an error
    pub async fn record_error(&self, key: &str, config: &MetricsConfig) {
        self.errors.fetch_add(1, Ordering::Relaxed);
        
        if config.track_per_key_metrics {
            self.record_key_error(key).await;
        }
    }
    
    /// Record an eviction
    pub fn record_eviction(&self, estimated_size: usize) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
        self.key_count.fetch_sub(1, Ordering::Relaxed);
        self.estimated_memory_usage.fetch_sub(estimated_size as u64, Ordering::Relaxed);
    }
    
    async fn record_operation_latency(&self, operation: &str, duration_micros: u64) {
        let mut histograms = self.operation_latency_histogram.write().await;
        let histogram = histograms.entry(operation.to_string())
            .or_insert_with(|| LatencyHistogram::new(&[
                100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000
            ]));
        histogram.record(duration_micros);
    }
    
    async fn record_key_metric(&self, key: &str, operation: &str) {
        let mut key_metrics = self.key_metrics.write().await;
        
        // Limit the number of tracked keys
        if key_metrics.len() >= 1000 && !key_metrics.contains_key(key) {
            return;
        }
        
        let metric = key_metrics.entry(key.to_string())
            .or_insert_with(|| KeyMetrics::new(key.to_string()));
        
        match operation {
            "hit" => {
                metric.hit_count += 1;
                metric.last_accessed = Some(std::time::SystemTime::now());
            }
            "miss" => {
                metric.miss_count += 1;
                metric.last_accessed = Some(std::time::SystemTime::now());
            }
            "set" => {
                metric.set_count += 1;
            }
            "delete" => {
                metric.delete_count += 1;
            }
            _ => {}
        }
    }
    
    async fn record_key_metric_with_size(&self, key: &str, operation: &str, size: usize) {
        let mut key_metrics = self.key_metrics.write().await;
        
        if key_metrics.len() >= 1000 && !key_metrics.contains_key(key) {
            return;
        }
        
        let metric = key_metrics.entry(key.to_string())
            .or_insert_with(|| KeyMetrics::new(key.to_string()));
        
        if operation == "set" {
            metric.set_count += 1;
            metric.estimated_size = size as u64;
        }
    }
    
    async fn record_key_error(&self, key: &str) {
        let mut key_metrics = self.key_metrics.write().await;
        
        if let Some(metric) = key_metrics.get_mut(key) {
            metric.error_count += 1;
        }
    }
    
    async fn update_hit_rate(&self) {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        
        if total > 0 {
            let hit_rate = (hits as f64) / (total as f64) * 100.0;
            let miss_rate = (misses as f64) / (total as f64) * 100.0;
            
            *self.hit_rate.lock().unwrap() = hit_rate;
            *self.miss_rate.lock().unwrap() = miss_rate;
        }
    }
    
    /// Get comprehensive cache statistics
    pub async fn get_stats(&self) -> DetailedCacheStats {
        let operation_stats = self.get_operation_stats().await;
        let key_stats = self.get_top_keys(10).await;
        
        DetailedCacheStats {
            basic: CacheStats {
                hits: self.hits.load(Ordering::Relaxed),
                misses: self.misses.load(Ordering::Relaxed),
                hit_rate: *self.hit_rate.lock().unwrap(),
                total_operations: self.hits.load(Ordering::Relaxed) + self.misses.load(Ordering::Relaxed),
                errors: self.errors.load(Ordering::Relaxed),
                evictions: self.evictions.load(Ordering::Relaxed),
                memory_usage: self.estimated_memory_usage.load(Ordering::Relaxed),
            },
            sets: self.sets.load(Ordering::Relaxed),
            deletes: self.deletes.load(Ordering::Relaxed),
            key_count: self.key_count.load(Ordering::Relaxed),
            operation_stats,
            top_keys: key_stats,
        }
    }
    
    async fn get_operation_stats(&self) -> HashMap<String, OperationStats> {
        let histograms = self.operation_latency_histogram.read().await;
        let mut stats = HashMap::new();
        
        for (operation, histogram) in histograms.iter() {
            stats.insert(operation.clone(), OperationStats {
                total_count: histogram.total_count,
                total_time_micros: histogram.sum,
                average_time_micros: histogram.average(),
                min_time_micros: if histogram.min == u64::MAX { 0 } else { histogram.min },
                max_time_micros: histogram.max,
                p50_micros: histogram.percentile(0.5),
                p95_micros: histogram.percentile(0.95),
                p99_micros: histogram.percentile(0.99),
            });
        }
        
        stats
    }
    
    async fn get_top_keys(&self, limit: usize) -> Vec<KeyMetrics> {
        let key_metrics = self.key_metrics.read().await;
        let mut keys: Vec<_> = key_metrics.values().cloned().collect();
        
        // Sort by total operations (hits + misses + sets + deletes)
        keys.sort_by(|a, b| {
            let a_total = a.hit_count + a.miss_count + a.set_count + a.delete_count;
            let b_total = b.hit_count + b.miss_count + b.set_count + b.delete_count;
            b_total.cmp(&a_total)
        });
        
        keys.truncate(limit);
        keys
    }
    
    /// Reset all metrics
    pub async fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.sets.store(0, Ordering::Relaxed);
        self.deletes.store(0, Ordering::Relaxed);
        self.errors.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.total_get_time.store(0, Ordering::Relaxed);
        self.total_set_time.store(0, Ordering::Relaxed);
        self.total_delete_time.store(0, Ordering::Relaxed);
        self.estimated_memory_usage.store(0, Ordering::Relaxed);
        self.key_count.store(0, Ordering::Relaxed);
        *self.hit_rate.lock().unwrap() = 0.0;
        *self.miss_rate.lock().unwrap() = 0.0;
        
        self.operation_latency_histogram.write().await.clear();
        self.key_metrics.write().await.clear();
    }
}

/// Detailed cache statistics including operation timings
#[derive(Debug, Serialize, Deserialize)]
pub struct DetailedCacheStats {
    pub basic: CacheStats,
    pub sets: u64,
    pub deletes: u64,
    pub key_count: u64,
    pub operation_stats: HashMap<String, OperationStats>,
    pub top_keys: Vec<KeyMetrics>,
}

/// Statistics for a specific operation
#[derive(Debug, Serialize, Deserialize)]
pub struct OperationStats {
    pub total_count: u64,
    pub total_time_micros: u64,
    pub average_time_micros: f64,
    pub min_time_micros: u64,
    pub max_time_micros: u64,
    pub p50_micros: Option<u64>,
    pub p95_micros: Option<u64>,
    pub p99_micros: Option<u64>,
}

impl MetricsCache {
    /// Wrap an existing cache provider with metrics collection
    pub fn wrap(inner: CacheProviderType, config: MetricsConfig) -> Self {
        info!("Wrapping cache with metrics collection");
        
        Self {
            inner,
            metrics: Arc::new(CacheMetrics::default()),
            config,
        }
    }
    
    /// Get detailed cache metrics
    pub async fn get_detailed_stats(&self) -> DetailedCacheStats {
        self.metrics.get_stats().await
    }
    
    /// Get the top N most accessed keys
    pub async fn get_top_keys(&self, limit: usize) -> Vec<KeyMetrics> {
        self.metrics.get_top_keys(limit).await
    }
    
    /// Reset all metrics
    pub async fn reset_metrics(&self) {
        self.metrics.reset().await;
    }
    
    /// Get operation latency statistics
    pub async fn get_operation_latencies(&self) -> HashMap<String, OperationStats> {
        self.metrics.get_operation_stats().await
    }
    
    /// Estimate memory usage per key
    pub async fn estimate_key_memory(&self, key: &str) -> Option<u64> {
        let key_metrics = self.metrics.key_metrics.read().await;
        key_metrics.get(key).map(|m| m.estimated_size)
    }
    
    /// Get cache efficiency metrics
    pub async fn get_efficiency_metrics(&self) -> CacheEfficiencyMetrics {
        let stats = self.get_detailed_stats().await;
        
        CacheEfficiencyMetrics {
            hit_rate: stats.basic.hit_rate,
            miss_rate: 100.0 - stats.basic.hit_rate,
            average_get_latency_micros: stats.operation_stats.get("get")
                .map(|s| s.average_time_micros)
                .unwrap_or(0.0),
            average_set_latency_micros: stats.operation_stats.get("set")
                .map(|s| s.average_time_micros)
                .unwrap_or(0.0),
            memory_efficiency: if stats.key_count > 0 {
                (stats.basic.memory_usage as f64) / (stats.key_count as f64)
            } else {
                0.0
            },
            error_rate: if stats.basic.total_operations > 0 {
                (stats.basic.errors as f64) / (stats.basic.total_operations as f64) * 100.0
            } else {
                0.0
            },
        }
    }
}

/// Cache efficiency metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct CacheEfficiencyMetrics {
    pub hit_rate: f64,
    pub miss_rate: f64,
    pub average_get_latency_micros: f64,
    pub average_set_latency_micros: f64,
    pub memory_efficiency: f64, // bytes per key
    pub error_rate: f64,
}

#[async_trait]
impl CacheProvider for MetricsCache {
    async fn get<T>(&self, key: &str) -> CacheResult<Option<T>>
    where
        T: DeserializeOwned + Send + Serialize + Sync + Clone,
    {
        let start = Instant::now();
        let result = self.inner.get(key).await;
        let duration = start.elapsed();
        
        match &result {
            Ok(Some(_)) => {
                self.metrics.record_hit(key, duration, &self.config).await;
                debug!("Cache hit for key '{}' (took {:?})", key, duration);
            }
            Ok(None) => {
                self.metrics.record_miss(key, duration, &self.config).await;
                debug!("Cache miss for key '{}' (took {:?})", key, duration);
            }
            Err(_) => {
                self.metrics.record_error(key, &self.config).await;
                warn!("Cache error for key '{}' (took {:?})", key, duration);
            }
        }
        
        result
    }

    async fn set<T>(&self, key: &str, value: &T, ttl: Option<StdDuration>) -> CacheResult<()>
    where
        T: Serialize + Send + Sync,
    {
        let start = Instant::now();
        
        // Estimate value size for memory tracking
        let value_size = if self.config.track_memory_usage {
            serde_json::to_vec(value)
                .map(|v| v.len())
                .unwrap_or(0)
        } else {
            0
        };
        
        let result = self.inner.set(key, value, ttl).await;
        let duration = start.elapsed();
        
        match &result {
            Ok(()) => {
                self.metrics.record_set(key, value_size, duration, &self.config).await;
                debug!("Cache set for key '{}' (took {:?}, {} bytes)", key, duration, value_size);
            }
            Err(_) => {
                self.metrics.record_error(key, &self.config).await;
                warn!("Cache set error for key '{}' (took {:?})", key, duration);
            }
        }
        
        result
    }

    async fn delete(&self, key: &str) -> CacheResult<bool> {
        let start = Instant::now();
        let result = self.inner.delete(key).await;
        let duration = start.elapsed();
        
        match &result {
            Ok(_) => {
                self.metrics.record_delete(key, duration, &self.config).await;
                debug!("Cache delete for key '{}' (took {:?})", key, duration);
            }
            Err(_) => {
                self.metrics.record_error(key, &self.config).await;
                warn!("Cache delete error for key '{}' (took {:?})", key, duration);
            }
        }
        
        result
    }

    async fn exists(&self, key: &str) -> CacheResult<bool> {
        self.inner.exists(key).await
    }

    async fn expire(&self, key: &str, ttl: StdDuration) -> CacheResult<bool> {
        self.inner.expire(key, ttl).await
    }

    async fn get_multi<T>(&self, keys: &[&str]) -> CacheResult<HashMap<String, T>>
    where
        T: DeserializeOwned + Send + Serialize + Sync + Clone,
    {
        let start = Instant::now();
        let result = self.inner.get_multi(keys).await;
        let duration = start.elapsed();
        
        if let Ok(ref values) = result {
            let hit_count = values.len();
            let miss_count = keys.len() - hit_count;
            
            // Record metrics for multi-get operation
            for _ in 0..hit_count {
                self.metrics.hits.fetch_add(1, Ordering::Relaxed);
            }
            for _ in 0..miss_count {
                self.metrics.misses.fetch_add(1, Ordering::Relaxed);
            }
            
            if self.config.track_operation_timing {
                let duration_micros = duration.as_micros() as u64;
                self.metrics.record_operation_latency("get_multi", duration_micros).await;
            }
        }
        
        result
    }

    async fn set_multi<T>(&self, items: &HashMap<&str, T>, ttl: Option<StdDuration>) -> CacheResult<()>
    where
        T: Serialize + Send + Sync,
    {
        let start = Instant::now();
        let result = self.inner.set_multi(items, ttl).await;
        let duration = start.elapsed();
        
        if result.is_ok() {
            self.metrics.sets.fetch_add(items.len() as u64, Ordering::Relaxed);
            
            if self.config.track_operation_timing {
                let duration_micros = duration.as_micros() as u64;
                self.metrics.record_operation_latency("set_multi", duration_micros).await;
            }
        }
        
        result
    }

    async fn flush(&self) -> CacheResult<()> {
        let result = self.inner.flush().await;
        
        if result.is_ok() {
            // Reset key count and memory usage after flush
            self.metrics.key_count.store(0, Ordering::Relaxed);
            self.metrics.estimated_memory_usage.store(0, Ordering::Relaxed);
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local::MemoryCache;
    use crate::CacheConfig;
    use tokio::time::{sleep, Duration};

    fn create_test_cache() -> MetricsCache {
        let config = CacheConfig {
            local_cache_size: 100,
            local_cache_ttl: Duration::from_secs(10),
            ..Default::default()
        };
        
        let memory_cache = CacheProviderType::Memory(MemoryCache::new(config));
        let metrics_config = MetricsConfig {
            track_per_key_metrics: true,
            track_operation_timing: true,
            ..Default::default()
        };
        
        MetricsCache::wrap(memory_cache, metrics_config)
    }

    #[tokio::test]
    async fn test_metrics_basic_operations() {
        let cache = create_test_cache();
        let key = "test:metrics";
        let value = "test_value".to_string();
        
        // Test set operation
        cache.set(key, &value, None).await.unwrap();
        
        // Test get operation (hit)
        let result: Option<String> = cache.get(key).await.unwrap();
        assert_eq!(result, Some(value));
        
        // Test get operation (miss)
        let _: Option<String> = cache.get("nonexistent").await.unwrap();
        
        // Test delete operation
        assert!(cache.delete(key).await.unwrap());
        
        // Check metrics
        let stats = cache.get_detailed_stats().await;
        assert_eq!(stats.basic.hits, 1);
        assert_eq!(stats.basic.misses, 1);
        assert_eq!(stats.sets, 1);
        assert_eq!(stats.deletes, 1);
        assert_eq!(stats.basic.hit_rate, 50.0);
    }

    #[tokio::test]
    async fn test_metrics_operation_timing() {
        let cache = create_test_cache();
        let key = "test:timing";
        let value = "test_value".to_string();
        
        // Perform operations
        cache.set(key, &value, None).await.unwrap();
        let _: Option<String> = cache.get(key).await.unwrap();
        
        // Check operation timing metrics
        let operation_stats = cache.get_operation_latencies().await;
        
        assert!(operation_stats.contains_key("get"));
        assert!(operation_stats.contains_key("set"));
        
        let get_stats = &operation_stats["get"];
        assert_eq!(get_stats.total_count, 1);
        assert!(get_stats.average_time_micros > 0.0);
    }

    #[tokio::test]
    async fn test_metrics_per_key_tracking() {
        let cache = create_test_cache();
        let key = "test:per_key";
        let value = "test_value".to_string();
        
        // Perform multiple operations on the same key
        cache.set(key, &value, None).await.unwrap();
        let _: Option<String> = cache.get(key).await.unwrap(); // hit
        let _: Option<String> = cache.get(key).await.unwrap(); // hit
        let _: Option<String> = cache.get("other_key").await.unwrap(); // miss
        
        // Check per-key metrics
        let top_keys = cache.get_top_keys(5).await;
        assert!(!top_keys.is_empty());
        
        let key_metric = top_keys.iter().find(|k| k.key == key).unwrap();
        assert_eq!(key_metric.hit_count, 2);
        assert_eq!(key_metric.set_count, 1);
        assert!(key_metric.hit_rate() > 0.0);
    }

    #[tokio::test]
    async fn test_metrics_efficiency() {
        let cache = create_test_cache();
        
        // Generate some cache activity
        cache.set("key1", &"value1".to_string(), None).await.unwrap();
        cache.set("key2", &"value2".to_string(), None).await.unwrap();
        let _: Option<String> = cache.get("key1").await.unwrap(); // hit
        let _: Option<String> = cache.get("key3").await.unwrap(); // miss
        
        // Check efficiency metrics
        let efficiency = cache.get_efficiency_metrics().await;
        
        assert_eq!(efficiency.hit_rate, 50.0);
        assert_eq!(efficiency.miss_rate, 50.0);
        assert!(efficiency.average_get_latency_micros > 0.0);
        assert!(efficiency.average_set_latency_micros > 0.0);
        assert_eq!(efficiency.error_rate, 0.0);
    }

    #[tokio::test]
    async fn test_metrics_multi_operations() {
        let cache = create_test_cache();
        
        let mut items = HashMap::new();
        items.insert("multi1", "value1".to_string());
        items.insert("multi2", "value2".to_string());
        
        // Test multi-set
        cache.set_multi(&items, None).await.unwrap();
        
        // Test multi-get
        let keys: Vec<&str> = items.keys().copied().collect();
        let results: HashMap<String, String> = cache.get_multi(&keys).await.unwrap();
        
        assert_eq!(results.len(), 2);
        
        // Check that multi-operations are tracked
        let stats = cache.get_detailed_stats().await;
        assert!(stats.sets >= 2);
        assert!(stats.basic.hits >= 2);
    }

    #[tokio::test]
    async fn test_metrics_reset() {
        let cache = create_test_cache();
        
        // Generate some activity
        cache.set("key", &"value".to_string(), None).await.unwrap();
        let _: Option<String> = cache.get("key").await.unwrap();
        
        // Verify metrics exist
        let stats_before = cache.get_detailed_stats().await;
        assert!(stats_before.basic.hits > 0);
        assert!(stats_before.sets > 0);
        
        // Reset metrics
        cache.reset_metrics().await;
        
        // Verify metrics are reset
        let stats_after = cache.get_detailed_stats().await;
        assert_eq!(stats_after.basic.hits, 0);
        assert_eq!(stats_after.sets, 0);
        assert_eq!(stats_after.basic.hit_rate, 0.0);
    }

    #[tokio::test]
    async fn test_latency_histogram() {
        let mut histogram = LatencyHistogram::new(&[100, 1000, 10000]);
        
        // Record some latencies
        histogram.record(50);   // Below first bucket
        histogram.record(500);  // In first bucket
        histogram.record(5000); // In second bucket
        histogram.record(15000); // In third bucket
        
        assert_eq!(histogram.total_count, 4);
        assert_eq!(histogram.min, 50);
        assert_eq!(histogram.max, 15000);
        assert!(histogram.average() > 0.0);
        
        // Test percentiles
        assert!(histogram.percentile(0.5).is_some());
        assert!(histogram.percentile(0.95).is_some());
    }

    #[tokio::test]
    async fn test_memory_usage_tracking() {
        let cache = create_test_cache();
        
        // Set a large value
        let large_value = "x".repeat(1000);
        cache.set("large_key", &large_value, None).await.unwrap();
        
        // Check memory tracking
        let stats = cache.get_detailed_stats().await;
        assert!(stats.basic.memory_usage > 0);
        
        // Estimate key memory
        let key_memory = cache.estimate_key_memory("large_key").await;
        assert!(key_memory.is_some());
        assert!(key_memory.unwrap() > 0);
    }
}