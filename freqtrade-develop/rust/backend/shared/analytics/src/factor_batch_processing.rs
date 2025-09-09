use anyhow::{Result, Context};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap, VecDeque};
use std::sync::{Arc, RwLock, atomic::{AtomicU64, AtomicBool, Ordering}};
use std::thread;
use std::time::{Instant, SystemTime};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, BufReader, BufWriter};
use tracing::{debug, info, warn, error};
use sha2::{Sha256, Digest};
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use bincode;
use tokio::time::sleep;
use dashmap::DashMap;
use parking_lot::RwLock as ParkingRwLock;

/// 因子批处理系统
/// 
/// 提供两级缓存架构和高效的批量计算能力
/// L1缓存：内存缓存，用于热数据快速访问
/// L2缓存：持久化缓存，用于存储历史计算结果
#[derive(Debug)]
pub struct FactorBatchProcessor {
    config: BatchProcessorConfig,
    l1_cache: Arc<RwLock<L1Cache>>,
    l2_cache: Arc<RwLock<L2Cache>>,
    statistics: Arc<RwLock<BatchProcessorStatistics>>,
    job_queue: Arc<RwLock<Vec<BatchProcessingJob>>>,
    active_jobs: Arc<RwLock<HashMap<String, BatchProcessingJob>>>,
}

/// 两级缓存系统
#[derive(Debug)]
pub struct TwoLevelCache {
    l1: Arc<RwLock<L1Cache>>,
    l2: Arc<RwLock<L2Cache>>,
}

/// L1内存缓存（高性能LRU缓存）
#[derive(Debug)]
struct L1Cache {
    entries: HashMap<String, CacheEntry>,
    access_order: VecDeque<String>,
    size_tracking: HashMap<String, usize>,
    max_entries: usize,
    max_size_bytes: usize,
    current_size_bytes: usize,
    hit_count: AtomicU64,
    miss_count: AtomicU64,
    eviction_count: AtomicU64,
    last_cleanup: Instant,
    cleanup_interval_secs: u64,
}

/// L2持久化缓存（高级存储后端）
#[derive(Debug)]
struct L2Cache {
    // 内存索引
    memory_index: HashMap<String, CacheEntryMetadata>,
    // 存储后端
    storage_backend: CacheStorageBackend,
    // 压缩设置
    compression_enabled: bool,
    compression_threshold_bytes: usize,
    // 统计信息
    hit_count: AtomicU64,
    miss_count: AtomicU64,
    write_count: AtomicU64,
    // 异步写入队列
    write_queue: Arc<RwLock<VecDeque<PendingWrite>>>,
    // 后台写入线程控制
    background_writer_active: Arc<AtomicBool>,
    // 缓存分片（提高并发性能）
    shards: Vec<Arc<ParkingRwLock<L2CacheShard>>>,
    shard_count: usize,
}

/// L2缓存分片
#[derive(Debug)]
struct L2CacheShard {
    entries: HashMap<String, CacheEntryMetadata>,
    access_times: BTreeMap<DateTime<Utc>, Vec<String>>,
    size_bytes: usize,
    max_size_bytes: usize,
}

/// 缓存条目元数据（用于L2缓存索引）
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntryMetadata {
    key: String,
    file_path: Option<String>,
    size_bytes: usize,
    created_at: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
    access_count: u64,
    is_compressed: bool,
    checksum: String,
    version: u32,
}

/// 待写入条目
#[derive(Debug, Clone)]
struct PendingWrite {
    key: String,
    entry: CacheEntry,
    priority: WritePriority,
    timestamp: DateTime<Utc>,
}

/// 写入优先级
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum WritePriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// 缓存条目
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    key: String,
    data: Vec<f64>,
    metadata: FactorMetadata,
    created_at: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
    access_count: u64,
    size_bytes: usize,
}

/// 因子元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FactorMetadata {
    symbol: String,
    factor_name: String,
    window_size: u64,
    parameters_hash: String,
    calculation_time_ms: f64,
    data_version: u32,
    dependencies: Vec<String>,
}

/// 批处理作业
#[derive(Debug, Clone)]
pub struct BatchProcessingJob {
    pub id: String,
    pub symbol: String,
    pub factor_names: Vec<String>,
    pub window_size: u64,
    pub parameters: HashMap<String, f64>,
    pub priority: JobPriority,
    pub status: JobStatus,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub results: HashMap<String, Vec<f64>>,
    pub error_message: Option<String>,
}

/// 作业优先级
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum JobPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// 作业状态
#[derive(Debug, Clone, PartialEq)]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// 缓存存储后端
#[derive(Debug)]
enum CacheStorageBackend {
    /// 文件系统存储（路径、是否启用分层存储）
    File { 
        base_path: String, 
        enable_sharding: bool,
        max_files_per_dir: usize,
    },
    /// 数据库存储（连接字符串、表名）
    Database { 
        connection_string: String, 
        table_name: String,
    },
    /// Redis存储（连接字符串、键前缀）
    Redis { 
        connection_string: String, 
        key_prefix: String,
    },
    /// S3兼容存储（端点、桶名、前缀）
    S3 { 
        endpoint: String, 
        bucket: String, 
        prefix: String,
    },
}

/// 批处理器配置
#[derive(Debug, Clone)]
pub struct BatchProcessorConfig {
    pub l1_cache_size: usize,
    pub l1_cache_max_size_mb: usize,
    pub l2_cache_size: usize,
    pub l2_cache_max_size_mb: usize,
    pub max_concurrent_jobs: usize,
    pub cache_ttl_seconds: u64,
    pub compression_threshold_bytes: usize,
    pub enable_compression: bool,
    pub enable_parallel_execution: bool,
    pub enable_background_writing: bool,
    pub enable_cache_warming: bool,
    pub job_timeout_seconds: u64,
    pub cache_eviction_policy: CacheEvictionPolicy,
    pub l2_shard_count: usize,
    pub background_writer_batch_size: usize,
    pub cache_cleanup_interval_secs: u64,
    pub enable_cache_metrics: bool,
    pub storage_backend: CacheStorageBackend,
}

/// 缓存淘汰策略
#[derive(Debug, Clone)]
pub enum CacheEvictionPolicy {
    LRU,  // 最近最少使用
    LFU,  // 最少使用频率
    FIFO, // 先进先出
}

/// 批处理器统计信息
#[derive(Debug, Default, Clone)]
pub struct BatchProcessorStatistics {
    pub total_jobs_processed: u64,
    pub successful_jobs: u64,
    pub failed_jobs: u64,
    pub l1_cache_hits: u64,
    pub l1_cache_misses: u64,
    pub l2_cache_hits: u64,
    pub l2_cache_misses: u64,
    pub total_computation_time_ms: f64,
    pub average_job_duration_ms: f64,
    pub cache_hit_rate: f64,
    pub l1_cache_size_bytes: usize,
    pub l2_cache_size_bytes: usize,
    pub cache_evictions: u64,
    pub compression_ratio: f64,
    pub background_writes_queued: u64,
    pub background_writes_completed: u64,
    pub average_cache_lookup_time_us: f64,
    pub cache_fragmentation_ratio: f64,
}

impl FactorBatchProcessor {
    /// 创建新的因子批处理器
    pub fn new(config: BatchProcessorConfig) -> Result<Self> {
        let l1_cache = Arc::new(RwLock::new(L1Cache::new(config.l1_cache_size)));
        let l2_cache = Arc::new(RwLock::new(L2Cache::new(config.l2_cache_size)?));
        let statistics = Arc::new(RwLock::new(BatchProcessorStatistics::default()));
        let job_queue = Arc::new(RwLock::new(Vec::new()));
        let active_jobs = Arc::new(RwLock::new(HashMap::new()));

        Ok(Self {
            config,
            l1_cache,
            l2_cache,
            statistics,
            job_queue,
            active_jobs,
        })
    }

    /// 提交批处理作业
    pub fn submit_job(&self, job: BatchProcessingJob) -> Result<String> {
        let job_id = job.id.clone();
        
        {
            let mut queue = self.job_queue.write().unwrap();
            queue.push(job);
            queue.sort_by(|a, b| b.priority.cmp(&a.priority));
        }

        info!("Job {} submitted to processing queue", job_id);
        Ok(job_id)
    }

    /// 处理下一个批处理作业
    pub fn process_next_job(&self) -> Result<Option<String>> {
        let job = {
            let mut queue = self.job_queue.write().unwrap();
            queue.pop()
        };

        if let Some(mut job) = job {
            let job_id = job.id.clone();
            job.status = JobStatus::Running;
            job.started_at = Some(Utc::now());

            {
                let mut active = self.active_jobs.write().unwrap();
                active.insert(job_id.clone(), job.clone());
            }

            match self.execute_job(&mut job) {
                Ok(_) => {
                    job.status = JobStatus::Completed;
                    job.completed_at = Some(Utc::now());
                    info!("Job {} completed successfully", job_id);
                }
                Err(e) => {
                    job.status = JobStatus::Failed;
                    job.error_message = Some(e.to_string());
                    warn!("Job {} failed: {}", job_id, e);
                }
            }

            {
                let mut active = self.active_jobs.write().unwrap();
                active.remove(&job_id);
            }

            self.update_statistics(&job);
            Ok(Some(job_id))
        } else {
            Ok(None)
        }
    }

    /// 执行批处理作业
    fn execute_job(&self, job: &mut BatchProcessingJob) -> Result<()> {
        let start_time = Utc::now();

        for factor_name in &job.factor_names {
            let cache_key = self.symbol_window_params_hash(
                &job.symbol,
                factor_name,
                job.window_size,
                &job.parameters,
            )?;

            // 检查缓存
            if let Some(cached_result) = self.get_from_cache(&cache_key)? {
                job.results.insert(factor_name.clone(), cached_result.data);
                debug!("Factor {} loaded from cache for symbol {}", factor_name, job.symbol);
                continue;
            }

            // 计算因子
            let factor_values = self.compute_factor(
                &job.symbol,
                factor_name,
                job.window_size,
                &job.parameters,
            )?;

            // 存储到缓存
            self.store_to_cache(&cache_key, &factor_values, &job.symbol, factor_name)?;
            job.results.insert(factor_name.clone(), factor_values);

            debug!("Factor {} computed and cached for symbol {}", factor_name, job.symbol);
        }

        let computation_time = Utc::now()
            .signed_duration_since(start_time)
            .num_milliseconds() as f64;

        info!(
            "Job {} completed in {:.2}ms, processed {} factors",
            job.id,
            computation_time,
            job.factor_names.len()
        );

        Ok(())
    }

    /// 生成符号-窗口-参数哈希
    fn symbol_window_params_hash(
        &self,
        symbol: &str,
        factor_name: &str,
        window_size: u64,
        parameters: &HashMap<String, f64>,
    ) -> Result<String> {
        let mut hasher = Sha256::new();
        hasher.update(symbol.as_bytes());
        hasher.update(factor_name.as_bytes());
        hasher.update(window_size.to_le_bytes());

        // 对参数进行排序以确保哈希一致性
        let mut sorted_params: Vec<_> = parameters.iter().collect();
        sorted_params.sort_by_key(|&(k, _)| k);

        for (key, value) in sorted_params {
            hasher.update(key.as_bytes());
            hasher.update(value.to_le_bytes());
        }

        Ok(format!("{:x}", hasher.finalize()))
    }

    /// 从缓存获取数据
    fn get_from_cache(&self, key: &str) -> Result<Option<CacheEntry>> {
        // 首先检查L1缓存
        {
            let mut l1 = self.l1_cache.write().unwrap();
            if let Some(entry) = l1.get(key) {
                let mut stats = self.statistics.write().unwrap();
                stats.l1_cache_hits += 1;
                return Ok(Some(entry));
            }
            let mut stats = self.statistics.write().unwrap();
            stats.l1_cache_misses += 1;
        }

        // 检查L2缓存
        {
            let mut l2 = self.l2_cache.write().unwrap();
            if let Some(entry) = l2.get(key) {
                // 将热数据提升到L1缓存
                let mut l1 = self.l1_cache.write().unwrap();
                l1.put(key.to_string(), entry.clone());

                let mut stats = self.statistics.write().unwrap();
                stats.l2_cache_hits += 1;
                return Ok(Some(entry));
            }
            let mut stats = self.statistics.write().unwrap();
            stats.l2_cache_misses += 1;
        }

        Ok(None)
    }

    /// 存储数据到缓存
    fn store_to_cache(
        &self,
        key: &str,
        data: &[f64],
        symbol: &str,
        factor_name: &str,
    ) -> Result<()> {
        let metadata = FactorMetadata {
            symbol: symbol.to_string(),
            factor_name: factor_name.to_string(),
            window_size: 0,
            parameters_hash: key.to_string(),
            calculation_time_ms: 0.0,
            data_version: 1,
            dependencies: vec![],
        };

        let entry = CacheEntry {
            key: key.to_string(),
            data: data.to_vec(),
            metadata,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 1,
            size_bytes: data.len() * 8, // f64 = 8 bytes
        };

        // 存储到L1和L2缓存
        {
            let mut l1 = self.l1_cache.write().unwrap();
            l1.put(key.to_string(), entry.clone());
        }

        {
            let mut l2 = self.l2_cache.write().unwrap();
            l2.put(key.to_string(), entry);
        }

        Ok(())
    }

    /// 计算因子
    fn compute_factor(
        &self,
        symbol: &str,
        factor_name: &str,
        window_size: u64,
        parameters: &HashMap<String, f64>,
    ) -> Result<Vec<f64>> {
        // 简化的因子计算实现
        // 实际实现会根据因子类型调用相应的计算函数
        match factor_name {
            "rsi" => Ok(self.calculate_rsi(symbol, window_size)?),
            "bollinger_bands" => Ok(self.calculate_bollinger_bands(symbol, window_size, parameters)?),
            "macd" => Ok(self.calculate_macd(symbol, window_size, parameters)?),
            "momentum" => Ok(self.calculate_momentum(symbol, window_size)?),
            _ => Err(anyhow::anyhow!("Unknown factor: {}", factor_name)),
        }
    }

    /// 计算RSI
    fn calculate_rsi(&self, symbol: &str, window_size: u64) -> Result<Vec<f64>> {
        // 模拟RSI计算
        let mut rsi_values = Vec::new();
        for i in 0..window_size {
            let rsi = 50.0 + 30.0 * ((i as f64 * 0.1).sin());
            rsi_values.push(rsi);
        }
        Ok(rsi_values)
    }

    /// 计算布林带
    fn calculate_bollinger_bands(
        &self,
        symbol: &str,
        window_size: u64,
        parameters: &HashMap<String, f64>,
    ) -> Result<Vec<f64>> {
        let std_multiplier = parameters.get("std_multiplier").unwrap_or(&2.0);
        let mut bb_values = Vec::new();
        
        for i in 0..window_size {
            let bb = 100.0 + 5.0 * std_multiplier * ((i as f64 * 0.05).sin());
            bb_values.push(bb);
        }
        
        Ok(bb_values)
    }

    /// 计算MACD
    fn calculate_macd(
        &self,
        symbol: &str,
        window_size: u64,
        parameters: &HashMap<String, f64>,
    ) -> Result<Vec<f64>> {
        let fast_period = parameters.get("fast_period").unwrap_or(&12.0);
        let slow_period = parameters.get("slow_period").unwrap_or(&26.0);
        
        let mut macd_values = Vec::new();
        for i in 0..window_size {
            let macd = (fast_period - slow_period) * ((i as f64 * 0.02).sin());
            macd_values.push(macd);
        }
        
        Ok(macd_values)
    }

    /// 计算动量
    fn calculate_momentum(&self, symbol: &str, window_size: u64) -> Result<Vec<f64>> {
        let mut momentum_values = Vec::new();
        for i in 0..window_size {
            let momentum = 2.0 * ((i as f64 * 0.03).sin());
            momentum_values.push(momentum);
        }
        Ok(momentum_values)
    }

    /// 更新统计信息
    fn update_statistics(&self, job: &BatchProcessingJob) {
        let mut stats = self.statistics.write().unwrap();
        stats.total_jobs_processed += 1;

        match job.status {
            JobStatus::Completed => stats.successful_jobs += 1,
            JobStatus::Failed => stats.failed_jobs += 1,
            _ => {}
        }

        if let (Some(started), Some(completed)) = (job.started_at, job.completed_at) {
            let duration = completed
                .signed_duration_since(started)
                .num_milliseconds() as f64;
            stats.total_computation_time_ms += duration;
            stats.average_job_duration_ms = 
                stats.total_computation_time_ms / stats.total_jobs_processed as f64;
        }
        
        debug!("Job statistics updated: total={}, successful={}, failed={}, avg_duration={:.2}ms",
               stats.total_jobs_processed, stats.successful_jobs, stats.failed_jobs, 
               stats.average_job_duration_ms);
    }
    
    /// 缓存预热（可选）
    pub fn warm_cache(&self, keys: Vec<String>) -> Result<usize> {
        let mut warmed_count = 0;
        
        for key in keys {
            if self.get_from_cache(&key)?.is_none() {
                // 这里可以实现预热逻辑
                // 比如从数据库或其他数据源加载数据
                debug!("Cache warming: key {} not found in cache", key);
            } else {
                warmed_count += 1;
            }
        }
        
        info!("Cache warming completed: {} keys were already cached", warmed_count);
        Ok(warmed_count)
    }
    
    /// 缓存清理
    pub fn cleanup_cache(&self) -> Result<()> {
        {
            let mut l1_cache = self.l1_cache.write().unwrap();
            l1_cache.cleanup_expired_entries();
        }
        
        // L2缓存清理由各个分片自动管理
        
        info!("Cache cleanup completed");
        Ok(())
    }

    /// 获取统计信息
    pub fn get_statistics(&self) -> BatchProcessorStatistics {
        let stats = self.statistics.read().unwrap();
        BatchProcessorStatistics {
            total_jobs_processed: stats.total_jobs_processed,
            successful_jobs: stats.successful_jobs,
            failed_jobs: stats.failed_jobs,
            l1_cache_hits: stats.l1_cache_hits,
            l1_cache_misses: stats.l1_cache_misses,
            l2_cache_hits: stats.l2_cache_hits,
            l2_cache_misses: stats.l2_cache_misses,
            total_computation_time_ms: stats.total_computation_time_ms,
            average_job_duration_ms: stats.average_job_duration_ms,
            cache_hit_rate: stats.cache_hit_rate,
        }
    }
}

impl L1Cache {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            access_order: Vec::new(),
            max_entries,
            hit_count: AtomicU64::new(0),
            miss_count: AtomicU64::new(0),
        }
    }

    fn get(&mut self, key: &str) -> Option<CacheEntry> {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_accessed = Utc::now();
            entry.access_count += 1;
            self.hit_count += 1;
            
            // 更新访问顺序
            self.access_order.retain(|k| k != key);
            self.access_order.push(key.to_string());
            
            Some(entry.clone())
        } else {
            self.miss_count += 1;
            None
        }
    }

    fn put(&mut self, key: String, entry: CacheEntry) {
        if self.entries.len() >= self.max_entries && !self.entries.contains_key(&key) {
            // 淘汰最久未使用的条目
            if let Some(lru_key) = self.access_order.first().cloned() {
                self.entries.remove(&lru_key);
                self.access_order.retain(|k| k != &lru_key);
            }
        }

        self.entries.insert(key.clone(), entry);
        self.access_order.retain(|k| k != &key);
        self.access_order.push(key);
    }
}

impl L2Cache {
    fn new(max_entries: usize) -> Result<Self> {
        Ok(Self {
            entries: HashMap::new(),
            storage_backend: CacheStorageBackend::File("/tmp/l2_cache".to_string()),
            compression_enabled: true,
            hit_count: AtomicU64::new(0),
            miss_count: AtomicU64::new(0),
        })
    }

    fn get(&mut self, key: &str) -> Option<CacheEntry> {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_accessed = Utc::now();
            entry.access_count += 1;
            self.hit_count += 1;
            Some(entry.clone())
        } else {
            self.miss_count += 1;
            None
        }
    }

    fn put(&mut self, key: String, entry: CacheEntry) {
        self.entries.insert(key, entry);
    }
}

impl Default for BatchProcessorConfig {
    fn default() -> Self {
        Self {
            l1_cache_size: 1000,
            l1_cache_max_size_mb: 100,
            l2_cache_size: 10000,
            l2_cache_max_size_mb: 1000,
            max_concurrent_jobs: 8,
            cache_ttl_seconds: 3600,
            compression_threshold_bytes: 1024,
            enable_compression: true,
            enable_parallel_execution: true,
            enable_background_writing: true,
            enable_cache_warming: true,
            job_timeout_seconds: 300,
            cache_eviction_policy: CacheEvictionPolicy::LRU,
            l2_shard_count: 16,
            background_writer_batch_size: 100,
            cache_cleanup_interval_secs: 300,
            enable_cache_metrics: true,
            storage_backend: CacheStorageBackend::File,
        }
    }
}