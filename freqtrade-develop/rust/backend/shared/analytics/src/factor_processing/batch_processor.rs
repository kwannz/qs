use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::hash::{Hash, Hasher, DefaultHasher};
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use tokio::time::{Duration, Instant};
use tracing::{debug, info, warn, error};
use futures::future::join_all;

use super::cache_manager::CacheManager;

/// 因子批处理配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessorConfig {
    pub max_concurrent_batches: usize,     // 最大并发批次
    pub batch_size: usize,                 // 批处理大小
    pub window_sizes: Vec<usize>,          // 窗口大小列表
    pub cache_ttl: Duration,               // 缓存TTL
    pub enable_incremental: bool,          // 启用增量计算
    pub prefetch_enabled: bool,            // 启用预取
    pub memory_limit_mb: usize,            // 内存限制(MB)
}

impl Default for BatchProcessorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_batches: 8,
            batch_size: 1000,
            window_sizes: vec![5, 10, 20, 50, 100],
            cache_ttl: Duration::from_secs(300), // 5分钟
            enable_incremental: true,
            prefetch_enabled: true,
            memory_limit_mb: 1024, // 1GB
        }
    }
}

/// 因子计算参数
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FactorParams {
    pub factor_name: String,
    pub window_size: usize,
    pub parameters: BTreeMap<String, serde_json::Value>,
    pub version: String,
}

impl FactorParams {
    pub fn cache_key(&self, symbol: &str) -> String {
        let mut hasher = DefaultHasher::new();
        symbol.hash(&mut hasher);
        self.hash(&mut hasher);
        format!("{}_{:x}", symbol, hasher.finish())
    }

    pub fn params_hash(&self) -> String {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

/// 因子计算输入数据
#[derive(Debug, Clone)]
pub struct FactorInput {
    pub symbol: String,
    pub timestamp: i64,
    pub price: f64,
    pub volume: f64,
    pub open: Option<f64>,
    pub high: Option<f64>,
    pub low: Option<f64>,
    pub close: Option<f64>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// 因子计算输出结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorOutput {
    pub symbol: String,
    pub timestamp: i64,
    pub factor_name: String,
    pub value: f64,
    pub confidence: f64,
    pub metadata: FactorMetadata,
}

/// 因子元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorMetadata {
    pub window_size: usize,
    pub sample_count: usize,
    pub last_update: i64,
    pub computation_time_ms: f64,
    pub cache_hit: bool,
    pub incremental_update: bool,
    pub quality_score: f64,
}

/// 批处理作业
#[derive(Debug, Clone)]
pub struct BatchJob {
    pub job_id: String,
    pub symbols: Vec<String>,
    pub factor_params: FactorParams,
    pub start_time: i64,
    pub end_time: i64,
    pub priority: JobPriority,
    pub callback: Option<String>, // 回调URL
}

/// 作业优先级
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum JobPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// 批处理状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BatchStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// 批处理结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    pub job_id: String,
    pub status: BatchStatus,
    pub start_time: i64,
    pub end_time: Option<i64>,
    pub total_symbols: usize,
    pub processed_symbols: usize,
    pub cache_hits: usize,
    pub computation_time_ms: f64,
    pub error_message: Option<String>,
    pub results: Vec<FactorOutput>,
}

/// 因子计算器特征
pub trait FactorCalculator: Send + Sync {
    fn calculate(&self, inputs: &[FactorInput], params: &FactorParams) -> Result<Vec<FactorOutput>>;
    fn supports_incremental(&self) -> bool;
    fn calculate_incremental(&self, previous: &FactorOutput, new_input: &FactorInput, params: &FactorParams) -> Result<FactorOutput>;
}

/// 因子批处理器
pub struct FactorBatchProcessor {
    config: BatchProcessorConfig,
    cache_manager: Arc<CacheManager>,
    calculators: Arc<RwLock<HashMap<String, Box<dyn FactorCalculator>>>>,
    
    // 作业管理
    job_queue: Arc<RwLock<VecDeque<BatchJob>>>,
    running_jobs: Arc<RwLock<HashMap<String, BatchResult>>>,
    completed_jobs: Arc<RwLock<HashMap<String, BatchResult>>>,
    
    // 并发控制
    semaphore: Arc<Semaphore>,
    
    // 统计信息
    stats: Arc<RwLock<BatchProcessorStats>>,
}

/// 批处理器统计
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchProcessorStats {
    pub total_jobs: u64,
    pub completed_jobs: u64,
    pub failed_jobs: u64,
    pub cancelled_jobs: u64,
    pub total_computation_time_ms: f64,
    pub average_batch_time_ms: f64,
    pub cache_hit_rate: f64,
    pub throughput_per_second: f64,
    pub memory_usage_mb: f64,
    pub last_update: i64,
}

impl FactorBatchProcessor {
    pub fn new(config: BatchProcessorConfig, cache_manager: Arc<CacheManager>) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_batches));
        
        Self {
            config,
            cache_manager,
            calculators: Arc::new(RwLock::new(HashMap::new())),
            job_queue: Arc::new(RwLock::new(VecDeque::new())),
            running_jobs: Arc::new(RwLock::new(HashMap::new())),
            completed_jobs: Arc::new(RwLock::new(HashMap::new())),
            semaphore,
            stats: Arc::new(RwLock::new(BatchProcessorStats::default())),
        }
    }

    /// 注册因子计算器
    pub async fn register_calculator(&mut self, name: String, calculator: Box<dyn FactorCalculator>) {
        self.calculators.write().await.insert(name.clone(), calculator);
        info!("Registered factor calculator: {}", name);
    }

    /// 启动批处理服务
    pub async fn start_service(&self) -> Result<()> {
        info!("Starting factor batch processor service");
        
        // 启动作业处理器
        self.start_job_processor().await?;
        
        // 启动统计收集器
        self.start_stats_collector().await?;
        
        // 启动内存监控
        self.start_memory_monitor().await?;
        
        Ok(())
    }

    /// 提交批处理作业
    pub async fn submit_job(&self, job: BatchJob) -> Result<String> {
        let job_id = job.job_id.clone();
        
        // 验证作业
        self.validate_job(&job).await?;
        
        // 添加到队列
        {
            let mut queue = self.job_queue.write().await;
            queue.push_back(job);
        }
        
        // 更新统计
        {
            let mut stats = self.stats.write().await;
            stats.total_jobs += 1;
        }
        
        info!("Submitted batch job: {}", job_id);
        Ok(job_id)
    }

    /// 获取作业状态
    pub async fn get_job_status(&self, job_id: &str) -> Result<BatchResult> {
        // 检查运行中的作业
        {
            let running = self.running_jobs.read().await;
            if let Some(result) = running.get(job_id) {
                return Ok(result.clone());
            }
        }
        
        // 检查已完成的作业
        {
            let completed = self.completed_jobs.read().await;
            if let Some(result) = completed.get(job_id) {
                return Ok(result.clone());
            }
        }
        
        // 检查队列中的作业
        {
            let queue = self.job_queue.read().await;
            for job in queue.iter() {
                if job.job_id == job_id {
                    return Ok(BatchResult {
                        job_id: job_id.to_string(),
                        status: BatchStatus::Queued,
                        start_time: 0,
                        end_time: None,
                        total_symbols: job.symbols.len(),
                        processed_symbols: 0,
                        cache_hits: 0,
                        computation_time_ms: 0.0,
                        error_message: None,
                        results: Vec::new(),
                    });
                }
            }
        }
        
        Err(anyhow::anyhow!("Job not found: {}", job_id))
    }

    /// 取消作业
    pub async fn cancel_job(&self, job_id: &str) -> Result<()> {
        // 从队列中移除
        {
            let mut queue = self.job_queue.write().await;
            queue.retain(|job| job.job_id != job_id);
        }
        
        // 标记运行中的作业为取消
        {
            let mut running = self.running_jobs.write().await;
            if let Some(result) = running.get_mut(job_id) {
                result.status = BatchStatus::Cancelled;
                result.end_time = Some(chrono::Utc::now().timestamp_millis());
            }
        }
        
        // 更新统计
        {
            let mut stats = self.stats.write().await;
            stats.cancelled_jobs += 1;
        }
        
        info!("Cancelled job: {}", job_id);
        Ok(())
    }

    /// 启动作业处理器
    async fn start_job_processor(&self) -> Result<()> {
        let job_queue = self.job_queue.clone();
        let running_jobs = self.running_jobs.clone();
        let completed_jobs = self.completed_jobs.clone();
        let cache_manager = self.cache_manager.clone();
        let calculators = Arc::clone(&self.calculators);
        let semaphore = self.semaphore.clone();
        let stats = self.stats.clone();
        let config = self.config.clone();
        
        tokio::spawn(async move {
            loop {
                // 获取下一个作业
                let next_job = {
                    let mut queue = job_queue.write().await;
                    queue.pop_front()
                };
                
                if let Some(job) = next_job {
                    let job_id = job.job_id.clone();
                    
                    // 获取信号量许可
                    let _permit = semaphore.acquire().await.unwrap();
                    
                    // 创建作业结果
                    let mut result = BatchResult {
                        job_id: job_id.clone(),
                        status: BatchStatus::Running,
                        start_time: chrono::Utc::now().timestamp_millis(),
                        end_time: None,
                        total_symbols: job.symbols.len(),
                        processed_symbols: 0,
                        cache_hits: 0,
                        computation_time_ms: 0.0,
                        error_message: None,
                        results: Vec::new(),
                    };
                    
                    // 添加到运行中的作业
                    {
                        let mut running = running_jobs.write().await;
                        running.insert(job_id.clone(), result.clone());
                    }
                    
                    // 执行作业
                    let job_clone = job.clone();
                    let cache_manager_clone = cache_manager.clone();
                    let calculators_clone = calculators.clone();
                    let stats_clone = stats.clone();
                    let config_clone = config.clone();
                    let running_jobs_clone = running_jobs.clone();
                    let completed_jobs_clone = completed_jobs.clone();
                    
                    tokio::spawn(async move {
                        let execution_result = Self::execute_job(
                            &job_clone,
                            &cache_manager_clone,
                            &calculators_clone,
                            &config_clone,
                            &running_jobs_clone,
                        ).await;
                        
                        // 更新结果
                        match execution_result {
                            Ok(job_result) => {
                                result.status = BatchStatus::Completed;
                                result.results = job_result.results;
                                result.processed_symbols = job_result.processed_symbols;
                                result.cache_hits = job_result.cache_hits;
                                result.computation_time_ms = job_result.computation_time_ms;
                                
                                // 更新统计
                                {
                                    let mut stats = stats_clone.write().await;
                                    stats.completed_jobs += 1;
                                    stats.total_computation_time_ms += result.computation_time_ms;
                                    stats.cache_hit_rate = (stats.cache_hit_rate * (stats.completed_jobs - 1) as f64 
                                        + job_result.cache_hits as f64 / job_result.processed_symbols.max(1) as f64) 
                                        / stats.completed_jobs as f64;
                                }
                            },
                            Err(e) => {
                                result.status = BatchStatus::Failed;
                                result.error_message = Some(e.to_string());
                                
                                // 更新统计
                                {
                                    let mut stats = stats_clone.write().await;
                                    stats.failed_jobs += 1;
                                }
                                
                                error!("Job {} failed: {}", job_id, e);
                            }
                        }
                        
                        result.end_time = Some(chrono::Utc::now().timestamp_millis());
                        
                        // 从运行中移除，添加到已完成
                        {
                            let mut running = running_jobs_clone.write().await;
                            running.remove(&job_id);
                        }
                        let status = result.status.clone();
                        {
                            let mut completed = completed_jobs_clone.write().await;
                            completed.insert(job_id.clone(), result);
                        }
                        
                        info!("Job {} completed with status: {:?}", job_id, status);
                    });
                } else {
                    // 没有作业，等待一段时间
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        });
        
        Ok(())
    }

    /// 执行批处理作业
    async fn execute_job(
        job: &BatchJob,
        cache_manager: &Arc<CacheManager>,
        calculators: &Arc<RwLock<HashMap<String, Box<dyn FactorCalculator>>>>,
        config: &BatchProcessorConfig,
        running_jobs: &Arc<RwLock<HashMap<String, BatchResult>>>,
    ) -> Result<BatchResult> {
        let start_time = Instant::now();
        let mut results = Vec::new();
        let mut processed_symbols = 0;
        let mut cache_hits = 0;
        
        // 获取计算器
        let calculators_read = calculators.read().await;
        let calculator = calculators_read.get(&job.factor_params.factor_name)
            .context("Factor calculator not found")?;
        
        // 按批次处理符号
        for symbol_batch in job.symbols.chunks(config.batch_size) {
            // 检查作业是否被取消
            {
                let running = running_jobs.read().await;
                if let Some(result) = running.get(&job.job_id) {
                    if result.status == BatchStatus::Cancelled {
                        return Err(anyhow::anyhow!("Job was cancelled"));
                    }
                }
            }
            
            let batch_results = Self::process_symbol_batch(
                symbol_batch,
                &job.factor_params,
                job.start_time,
                job.end_time,
                calculator.as_ref(),
                cache_manager,
            ).await?;
            
            // 统计缓存命中
            for result in &batch_results {
                if result.metadata.cache_hit {
                    cache_hits += 1;
                }
            }
            
            results.extend(batch_results);
            processed_symbols += symbol_batch.len();
            
            // 更新运行状态
            {
                let mut running = running_jobs.write().await;
                if let Some(result) = running.get_mut(&job.job_id) {
                    result.processed_symbols = processed_symbols;
                    result.cache_hits = cache_hits;
                    result.computation_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
                }
            }
            
            debug!("Processed batch of {} symbols for job {}", symbol_batch.len(), job.job_id);
        }
        
        let computation_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        
        Ok(BatchResult {
            job_id: job.job_id.clone(),
            status: BatchStatus::Completed,
            start_time: job.start_time,
            end_time: Some(chrono::Utc::now().timestamp_millis()),
            total_symbols: job.symbols.len(),
            processed_symbols,
            cache_hits,
            computation_time_ms,
            error_message: None,
            results,
        })
    }

    /// 处理符号批次
    async fn process_symbol_batch(
        symbols: &[String],
        factor_params: &FactorParams,
        start_time: i64,
        end_time: i64,
        calculator: &dyn FactorCalculator,
        cache_manager: &Arc<CacheManager>,
    ) -> Result<Vec<FactorOutput>> {
        let mut results = Vec::new();
        
        // 并行处理每个符号
        let futures = symbols.iter().map(|symbol| {
            Self::process_single_symbol(
                symbol.clone(),
                factor_params.clone(),
                start_time,
                end_time,
                calculator,
                cache_manager.clone(),
            )
        });
        
        let batch_results = join_all(futures).await;
        
        for result in batch_results {
            match result {
                Ok(factor_output) => results.extend(factor_output),
                Err(e) => warn!("Failed to process symbol: {}", e),
            }
        }
        
        Ok(results)
    }

    /// 处理单个符号
    async fn process_single_symbol(
        symbol: String,
        factor_params: FactorParams,
        start_time: i64,
        end_time: i64,
        calculator: &dyn FactorCalculator,
        cache_manager: Arc<CacheManager>,
    ) -> Result<Vec<FactorOutput>> {
        let cache_key = factor_params.cache_key(&symbol);
        
        // 检查缓存
        if let Ok(Some(cached_result)) = cache_manager.get::<Vec<FactorOutput>>(&cache_key).await {
            // 验证缓存数据是否仍然有效
            if let Some(first_result) = cached_result.first() {
                if first_result.timestamp >= start_time && first_result.timestamp <= end_time {
                    let mut updated_results = cached_result;
                    for result in &mut updated_results {
                        result.metadata.cache_hit = true;
                    }
                    return Ok(updated_results);
                }
            }
        }
        
        // 缓存未命中，获取原始数据并计算
        let factor_inputs = Self::fetch_market_data(&symbol, start_time, end_time).await?;
        
        if factor_inputs.is_empty() {
            return Ok(Vec::new());
        }
        
        let computation_start = Instant::now();
        let mut factor_outputs = calculator.calculate(&factor_inputs, &factor_params)?;
        let computation_time = computation_start.elapsed().as_secs_f64() * 1000.0;
        
        // 更新元数据
        for output in &mut factor_outputs {
            output.metadata.computation_time_ms = computation_time;
            output.metadata.cache_hit = false;
            output.metadata.last_update = chrono::Utc::now().timestamp_millis();
        }
        
        // 存储到缓存
        if let Err(e) = cache_manager.set(&cache_key, &factor_outputs, Some(Duration::from_secs(300))).await {
            warn!("Failed to cache factor results: {}", e);
        }
        
        Ok(factor_outputs)
    }

    /// 获取市场数据 (模拟实现)
    async fn fetch_market_data(symbol: &str, start_time: i64, end_time: i64) -> Result<Vec<FactorInput>> {
        // 实际实现中会从数据库或外部API获取数据
        // 这里提供模拟数据
        let mut inputs = Vec::new();
        let mut current_time = start_time;
        let interval = 60000; // 1分钟间隔
        
        while current_time <= end_time {
            inputs.push(FactorInput {
                symbol: symbol.to_string(),
                timestamp: current_time,
                price: 100.0 + (current_time as f64 / 1000000.0).sin() * 10.0,
                volume: 1000.0,
                open: None,
                high: None,
                low: None,
                close: None,
                metadata: HashMap::new(),
            });
            current_time += interval;
        }
        
        Ok(inputs)
    }

    /// 启动统计收集器
    async fn start_stats_collector(&self) -> Result<()> {
        let stats = self.stats.clone();
        let completed_jobs = self.completed_jobs.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60)); // 每分钟更新统计
            
            loop {
                interval.tick().await;
                
                let mut stats_guard = stats.write().await;
                let completed = completed_jobs.read().await;
                
                // 计算平均批次时间
                if stats_guard.completed_jobs > 0 {
                    stats_guard.average_batch_time_ms = stats_guard.total_computation_time_ms / stats_guard.completed_jobs as f64;
                }
                
                // 计算吞吐量
                let current_time = chrono::Utc::now().timestamp_millis();
                if stats_guard.last_update > 0 {
                    let time_diff = (current_time - stats_guard.last_update) as f64 / 1000.0; // 转换为秒
                    if time_diff > 0.0 {
                        let completed_in_period = completed.len() as f64;
                        stats_guard.throughput_per_second = completed_in_period / time_diff;
                    }
                }
                
                stats_guard.last_update = current_time;
                
                debug!("Updated batch processor stats: completed={}, average_time={:.2}ms, throughput={:.2}/s", 
                       stats_guard.completed_jobs, stats_guard.average_batch_time_ms, stats_guard.throughput_per_second);
            }
        });
        
        Ok(())
    }

    /// 启动内存监控
    async fn start_memory_monitor(&self) -> Result<()> {
        let stats = self.stats.clone();
        let config_memory_limit = self.config.memory_limit_mb;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                // 简化的内存使用估算
                let estimated_memory_mb = Self::estimate_memory_usage().await;
                
                {
                    let mut stats_guard = stats.write().await;
                    stats_guard.memory_usage_mb = estimated_memory_mb;
                }
                
                if estimated_memory_mb > config_memory_limit as f64 * 0.9 {
                    warn!("Memory usage approaching limit: {:.1}MB / {}MB", 
                          estimated_memory_mb, config_memory_limit);
                }
            }
        });
        
        Ok(())
    }

    /// 估算内存使用量
    async fn estimate_memory_usage() -> f64 {
        // 简化实现 - 实际中会使用系统调用获取真实内存使用
        // 这里返回一个模拟值
        128.0 // MB
    }

    /// 验证作业
    async fn validate_job(&self, job: &BatchJob) -> Result<()> {
        if job.symbols.is_empty() {
            return Err(anyhow::anyhow!("Job must contain at least one symbol"));
        }
        
        if job.start_time >= job.end_time {
            return Err(anyhow::anyhow!("Invalid time range"));
        }
        
        {
            let calculators_read = self.calculators.read().await;
            if !calculators_read.contains_key(&job.factor_params.factor_name) {
                return Err(anyhow::anyhow!("Unknown factor: {}", job.factor_params.factor_name));
            }
        }
        
        Ok(())
    }

    /// 获取统计信息
    pub async fn get_stats(&self) -> BatchProcessorStats {
        let stats = self.stats.read().await;
        stats.clone()
    }

    /// 获取队列状态
    pub async fn get_queue_status(&self) -> Result<QueueStatus> {
        let queue = self.job_queue.read().await;
        let running = self.running_jobs.read().await;
        let completed = self.completed_jobs.read().await;
        
        Ok(QueueStatus {
            queued_jobs: queue.len(),
            running_jobs: running.len(),
            completed_jobs: completed.len(),
            total_jobs: queue.len() + running.len() + completed.len(),
        })
    }
}

/// 队列状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStatus {
    pub queued_jobs: usize,
    pub running_jobs: usize,
    pub completed_jobs: usize,
    pub total_jobs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factor_processing::cache_manager::{MemoryCacheManager, CacheConfig};

    struct MockFactorCalculator;
    
    impl FactorCalculator for MockFactorCalculator {
        fn calculate(&self, inputs: &[FactorInput], _params: &FactorParams) -> Result<Vec<FactorOutput>> {
            Ok(inputs.iter().map(|input| FactorOutput {
                symbol: input.symbol.clone(),
                timestamp: input.timestamp,
                factor_name: "test_factor".to_string(),
                value: input.price * 0.1,
                confidence: 0.8,
                metadata: FactorMetadata {
                    window_size: 10,
                    sample_count: 1,
                    last_update: chrono::Utc::now().timestamp_millis(),
                    computation_time_ms: 1.0,
                    cache_hit: false,
                    incremental_update: false,
                    quality_score: 0.9,
                },
            }).collect())
        }
        
        fn supports_incremental(&self) -> bool {
            false
        }
        
        fn calculate_incremental(&self, _previous: &FactorOutput, _new_input: &FactorInput, _params: &FactorParams) -> Result<FactorOutput> {
            Err(anyhow::anyhow!("Incremental calculation not supported"))
        }
    }

    #[tokio::test]
    async fn test_batch_processor_creation() {
        let config = BatchProcessorConfig::default();
        let cache_config = CacheConfig::default();
        let cache_manager = Arc::new(CacheManager::new(cache_config));
        let processor = FactorBatchProcessor::new(config, cache_manager);
        
        assert_eq!(processor.calculators.read().await.len(), 0);
    }

    #[tokio::test]
    async fn test_job_submission() {
        let config = BatchProcessorConfig::default();
        let cache_config = CacheConfig::default();
        let cache_manager = Arc::new(CacheManager::new(cache_config));
        let mut processor = FactorBatchProcessor::new(config, cache_manager);
        
        // 注册计算器
        processor.register_calculator("test_factor".to_string(), Box::new(MockFactorCalculator));
        
        let job = BatchJob {
            job_id: "test_job_1".to_string(),
            symbols: vec!["BTCUSD".to_string()],
            factor_params: FactorParams {
                factor_name: "test_factor".to_string(),
                window_size: 10,
                parameters: BTreeMap::new(),
                version: "1.0".to_string(),
            },
            start_time: chrono::Utc::now().timestamp_millis() - 3600000, // 1小时前
            end_time: chrono::Utc::now().timestamp_millis(),
            priority: JobPriority::Normal,
            callback: None,
        };
        
        let result = processor.submit_job(job).await;
        assert!(result.is_ok());
    }
}