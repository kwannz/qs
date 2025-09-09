use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{Duration, Instant};
use tracing::{debug, info, warn};

use super::batch_processor::{FactorInput, FactorOutput, FactorParams, FactorMetadata};
use super::cache_manager::CacheManager;
use super::window_calculator::{RollingWindowCalculator, WindowFunction, WindowResult};

/// 增量更新器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalUpdaterConfig {
    pub max_symbols: usize,              // 最大支持的符号数
    pub update_batch_size: usize,        // 更新批次大小
    pub checkpoint_interval: Duration,    // 检查点间隔
    pub max_memory_mb: usize,            // 最大内存使用(MB)
    pub enable_compression: bool,         // 启用压缩存储
    pub parallel_updates: bool,          // 并行更新
    pub persistence_enabled: bool,       // 启用持久化
}

impl Default for IncrementalUpdaterConfig {
    fn default() -> Self {
        Self {
            max_symbols: 10000,
            update_batch_size: 100,
            checkpoint_interval: Duration::from_secs(300), // 5分钟
            max_memory_mb: 512,
            enable_compression: true,
            parallel_updates: true,
            persistence_enabled: true,
        }
    }
}

/// 增量状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalState {
    pub symbol: String,
    pub factor_name: String,
    pub window_size: usize,
    pub last_value: f64,
    pub last_timestamp: i64,
    pub accumulated_data: VecDeque<f64>,
    pub auxiliary_state: HashMap<String, f64>,
    pub update_count: u64,
    pub last_checkpoint: i64,
    pub version: u32,
}

impl IncrementalState {
    pub fn new(symbol: String, factor_name: String, window_size: usize) -> Self {
        Self {
            symbol,
            factor_name,
            window_size,
            last_value: 0.0,
            last_timestamp: 0,
            accumulated_data: VecDeque::with_capacity(window_size),
            auxiliary_state: HashMap::new(),
            update_count: 0,
            last_checkpoint: chrono::Utc::now().timestamp_millis(),
            version: 1,
        }
    }
    
    pub fn add_data_point(&mut self, value: f64, timestamp: i64) {
        if self.accumulated_data.len() >= self.window_size {
            self.accumulated_data.pop_front();
        }
        self.accumulated_data.push_back(value);
        self.last_value = value;
        self.last_timestamp = timestamp;
        self.update_count += 1;
    }
    
    pub fn is_stale(&self, ttl_ms: i64) -> bool {
        let current_time = chrono::Utc::now().timestamp_millis();
        current_time - self.last_timestamp > ttl_ms
    }
    
    pub fn needs_checkpoint(&self, interval: Duration) -> bool {
        let current_time = chrono::Utc::now().timestamp_millis();
        current_time - self.last_checkpoint > interval.as_millis() as i64
    }
    
    pub fn create_checkpoint(&mut self) {
        self.last_checkpoint = chrono::Utc::now().timestamp_millis();
        self.version += 1;
    }
}

/// 增量更新结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalUpdateResult {
    pub symbol: String,
    pub factor_name: String,
    pub old_value: f64,
    pub new_value: f64,
    pub timestamp: i64,
    pub update_type: UpdateType,
    pub computation_time_ms: f64,
    pub cache_updated: bool,
    pub checkpoint_created: bool,
}

/// 更新类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateType {
    Incremental,    // 增量更新
    Recalculate,    // 重新计算
    Checkpoint,     // 检查点更新
    Initialization, // 初始化
}

/// 增量更新器
#[derive(Debug)]
pub struct IncrementalUpdater {
    config: IncrementalUpdaterConfig,
    states: Arc<RwLock<HashMap<String, IncrementalState>>>, // state_key -> state
    cache_manager: Arc<CacheManager>,
    window_calculator: Arc<RollingWindowCalculator>,
    
    // 统计信息
    stats: Arc<RwLock<IncrementalStats>>,
    
    // 更新队列
    update_queue: Arc<RwLock<VecDeque<PendingUpdate>>>,
}

/// 待处理更新
#[derive(Debug, Clone)]
struct PendingUpdate {
    pub state_key: String,
    pub new_input: FactorInput,
    pub factor_params: FactorParams,
    pub priority: UpdatePriority,
    pub submitted_at: Instant,
}

/// 更新优先级
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum UpdatePriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// 增量更新统计
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IncrementalStats {
    pub total_updates: u64,
    pub incremental_updates: u64,
    pub full_recalculations: u64,
    pub checkpoints_created: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub avg_update_time_ms: f64,
    pub memory_usage_mb: f64,
    pub active_states: usize,
    pub stale_states_cleaned: u64,
    pub queue_size: usize,
    pub throughput_per_second: f64,
}

impl IncrementalUpdater {
    pub fn new(
        config: IncrementalUpdaterConfig,
        cache_manager: Arc<CacheManager>,
        window_calculator: Arc<RollingWindowCalculator>,
    ) -> Self {
        Self {
            config,
            states: Arc::new(RwLock::new(HashMap::new())),
            cache_manager,
            window_calculator,
            stats: Arc::new(RwLock::new(IncrementalStats::default())),
            update_queue: Arc::new(RwLock::new(VecDeque::new())),
        }
    }
    
    /// 启动增量更新服务
    pub async fn start_service(&self) -> Result<()> {
        info!("Starting incremental updater service");
        
        // 启动更新处理器
        self.start_update_processor().await?;
        
        // 启动检查点管理器
        self.start_checkpoint_manager().await?;
        
        // 启动状态清理器
        self.start_state_cleaner().await?;
        
        // 启动统计收集器
        self.start_stats_collector().await?;
        
        Ok(())
    }
    
    /// 提交增量更新请求
    pub async fn submit_update(
        &self,
        symbol: String,
        factor_params: FactorParams,
        new_input: FactorInput,
        priority: UpdatePriority,
    ) -> Result<String> {
        let state_key = self.generate_state_key(&symbol, &factor_params);
        
        let pending_update = PendingUpdate {
            state_key: state_key.clone(),
            new_input,
            factor_params,
            priority,
            submitted_at: Instant::now(),
        };
        
        // 添加到更新队列
        {
            let mut queue = self.update_queue.write().await;
            
            // 按优先级插入
            let insert_pos = queue.iter()
                .position(|update| update.priority < pending_update.priority)
                .unwrap_or(queue.len());
            
            queue.insert(insert_pos, pending_update);
        }
        
        // 更新统计
        {
            let mut stats = self.stats.write().await;
            stats.queue_size += 1;
        }
        
        debug!("Submitted incremental update for: {}", state_key);
        Ok(state_key)
    }
    
    /// 批量提交更新
    pub async fn submit_batch_updates(
        &self,
        updates: Vec<(String, FactorParams, FactorInput, UpdatePriority)>,
    ) -> Result<Vec<String>> {
        let mut state_keys = Vec::new();
        
        {
            let mut queue = self.update_queue.write().await;
            
            for (symbol, factor_params, new_input, priority) in updates {
                let state_key = self.generate_state_key(&symbol, &factor_params);
                
                let pending_update = PendingUpdate {
                    state_key: state_key.clone(),
                    new_input,
                    factor_params,
                    priority,
                    submitted_at: Instant::now(),
                };
                
                // 按优先级插入
                let insert_pos = queue.iter()
                    .position(|update| update.priority < pending_update.priority)
                    .unwrap_or(queue.len());
                
                queue.insert(insert_pos, pending_update);
                state_keys.push(state_key);
            }
        }
        
        // 更新统计
        {
            let mut stats = self.stats.write().await;
            stats.queue_size += state_keys.len();
        }
        
        Ok(state_keys)
    }
    
    /// 获取增量状态
    pub async fn get_state(&self, symbol: &str, factor_params: &FactorParams) -> Option<IncrementalState> {
        let state_key = self.generate_state_key(symbol, factor_params);
        let states = self.states.read().await;
        states.get(&state_key).cloned()
    }
    
    /// 重置状态（强制重新计算）
    pub async fn reset_state(&self, symbol: &str, factor_params: &FactorParams) -> Result<()> {
        let state_key = self.generate_state_key(symbol, factor_params);
        
        {
            let mut states = self.states.write().await;
            states.remove(&state_key);
        }
        
        // 清除缓存
        let cache_key = factor_params.cache_key(symbol);
        self.cache_manager.delete(&cache_key).await?;
        
        info!("Reset state for: {}", state_key);
        Ok(())
    }
    
    /// 启动更新处理器
    async fn start_update_processor(&self) -> Result<()> {
        let config = self.config.clone();
        let states = self.states.clone();
        let cache_manager = self.cache_manager.clone();
        let window_calculator = self.window_calculator.clone();
        let stats = self.stats.clone();
        let update_queue = self.update_queue.clone();
        
        tokio::spawn(async move {
            loop {
                // 处理更新队列
                let updates_batch = {
                    let mut queue = update_queue.write().await;
                    let batch_size = config.update_batch_size.min(queue.len());
                    let mut batch = Vec::new();
                    
                    for _ in 0..batch_size {
                        if let Some(update) = queue.pop_front() {
                            batch.push(update);
                        }
                    }
                    
                    batch
                };
                
                if updates_batch.is_empty() {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    continue;
                }
                
                // 处理批次
                if config.parallel_updates {
                    // 并行处理
                    let futures = updates_batch.into_iter().map(|update| {
                        Self::process_single_update(
                            update,
                            states.clone(),
                            cache_manager.clone(),
                            window_calculator.clone(),
                        )
                    });
                    
                    let results = futures::future::join_all(futures).await;
                    
                    // 更新统计
                    let mut successful_updates = 0;
                    for result in results {
                        match result {
                            Ok(_) => successful_updates += 1,
                            Err(e) => warn!("Update failed: {}", e),
                        }
                    }
                    
                    // 更新统计信息
                    {
                        let mut stats_guard = stats.write().await;
                        stats_guard.total_updates += successful_updates;
                        stats_guard.queue_size = stats_guard.queue_size.saturating_sub(successful_updates as usize);
                    }
                } else {
                    // 串行处理
                    for update in updates_batch {
                        match Self::process_single_update(
                            update,
                            states.clone(),
                            cache_manager.clone(),
                            window_calculator.clone(),
                        ).await {
                            Ok(_) => {
                                let mut stats_guard = stats.write().await;
                                stats_guard.total_updates += 1;
                                stats_guard.queue_size = stats_guard.queue_size.saturating_sub(1);
                            },
                            Err(e) => warn!("Update failed: {}", e),
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// 处理单个更新
    async fn process_single_update(
        update: PendingUpdate,
        states: Arc<RwLock<HashMap<String, IncrementalState>>>,
        cache_manager: Arc<CacheManager>,
        window_calculator: Arc<RollingWindowCalculator>,
    ) -> Result<IncrementalUpdateResult> {
        let start_time = Instant::now();
        
        // 获取或创建状态
        let mut state = {
            let states_read = states.read().await;
            states_read.get(&update.state_key).cloned()
                .unwrap_or_else(|| IncrementalState::new(
                    update.new_input.symbol.clone(),
                    update.factor_params.factor_name.clone(),
                    update.factor_params.window_size,
                ))
        };
        
        let old_value = state.last_value;
        
        // 添加新数据点
        state.add_data_point(update.new_input.price, update.new_input.timestamp);
        
        // 决定更新类型
        let update_type = if state.update_count == 1 {
            UpdateType::Initialization
        } else if state.accumulated_data.len() < state.window_size {
            UpdateType::Recalculate // 数据不足时重新计算
        } else {
            UpdateType::Incremental
        };
        
        // 计算新值
        let new_value = match update_type {
            UpdateType::Incremental => {
                // 尝试增量计算
                Self::calculate_incremental_value(&state, &update.factor_params)?
            },
            _ => {
                // 完整重新计算
                Self::calculate_full_value(&state, &update.factor_params, &window_calculator).await?
            }
        };
        
        state.last_value = new_value;
        
        // 检查是否需要创建检查点
        let checkpoint_created = if state.needs_checkpoint(Duration::from_secs(300)) {
            state.create_checkpoint();
            true
        } else {
            false
        };
        
        // 更新缓存
        let cache_key = update.factor_params.cache_key(&update.new_input.symbol);
        let factor_output = FactorOutput {
            symbol: update.new_input.symbol.clone(),
            timestamp: update.new_input.timestamp,
            factor_name: update.factor_params.factor_name.clone(),
            value: new_value,
            confidence: if state.accumulated_data.len() >= state.window_size { 1.0 } else { 0.8 },
            metadata: FactorMetadata {
                window_size: state.window_size,
                sample_count: state.accumulated_data.len(),
                last_update: chrono::Utc::now().timestamp_millis(),
                computation_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                cache_hit: false,
                incremental_update: matches!(update_type, UpdateType::Incremental),
                quality_score: 0.9,
            },
        };
        
        let cache_updated = cache_manager.set(&cache_key, &factor_output, Some(Duration::from_secs(300))).await.is_ok();
        
        // 保存状态
        {
            let mut states_write = states.write().await;
            states_write.insert(update.state_key.clone(), state);
        }
        
        Ok(IncrementalUpdateResult {
            symbol: update.new_input.symbol,
            factor_name: update.factor_params.factor_name,
            old_value,
            new_value,
            timestamp: update.new_input.timestamp,
            update_type,
            computation_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            cache_updated,
            checkpoint_created,
        })
    }
    
    /// 增量计算值
    fn calculate_incremental_value(state: &IncrementalState, factor_params: &FactorParams) -> Result<f64> {
        // 简化的增量计算实现
        match factor_params.factor_name.as_str() {
            "sma" => {
                // 简单移动平均的增量计算
                let sum: f64 = state.accumulated_data.iter().sum();
                Ok(sum / state.accumulated_data.len() as f64)
            },
            "ema" => {
                // 指数移动平均的增量计算
                let alpha = 2.0 / (state.window_size as f64 + 1.0);
                let new_price = state.accumulated_data.back().copied().unwrap_or(0.0);
                Ok(alpha * new_price + (1.0 - alpha) * state.last_value)
            },
            _ => {
                // 默认返回最新价格
                Ok(state.last_value)
            }
        }
    }
    
    /// 完整计算值
    async fn calculate_full_value(
        state: &IncrementalState,
        factor_params: &FactorParams,
        window_calculator: &Arc<RollingWindowCalculator>,
    ) -> Result<f64> {
        // 创建输入数据
        let inputs: Vec<FactorInput> = state.accumulated_data.iter().enumerate().map(|(i, &price)| {
            FactorInput {
                symbol: state.symbol.clone(),
                timestamp: state.last_timestamp - (state.accumulated_data.len() - i - 1) as i64 * 60000, // 假设1分钟间隔
                price,
                volume: 1000.0, // 默认成交量
                open: None,
                high: None,
                low: None,
                close: None,
                metadata: HashMap::new(),
            }
        }).collect();
        
        if inputs.is_empty() {
            return Ok(0.0);
        }
        
        // 根据因子类型计算
        match factor_params.factor_name.as_str() {
            "sma" => {
                let result = window_calculator.calculate_window(
                    &state.symbol,
                    WindowFunction::SMA,
                    state.window_size,
                    &inputs,
                    &factor_params.parameters,
                ).await?;
                Ok(result.value)
            },
            "ema" => {
                let result = window_calculator.calculate_window(
                    &state.symbol,
                    WindowFunction::EMA,
                    state.window_size,
                    &inputs,
                    &factor_params.parameters,
                ).await?;
                Ok(result.value)
            },
            "volatility" => {
                let result = window_calculator.calculate_window(
                    &state.symbol,
                    WindowFunction::Volatility,
                    state.window_size,
                    &inputs,
                    &factor_params.parameters,
                ).await?;
                Ok(result.value)
            },
            _ => {
                // 默认返回最新价格
                Ok(inputs.last().unwrap().price)
            }
        }
    }
    
    /// 启动检查点管理器
    async fn start_checkpoint_manager(&self) -> Result<()> {
        let config = self.config.clone();
        let states = self.states.clone();
        let stats = self.stats.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.checkpoint_interval);
            
            loop {
                interval.tick().await;
                
                let mut checkpoint_count = 0;
                {
                    let mut states_guard = states.write().await;
                    for state in states_guard.values_mut() {
                        if state.needs_checkpoint(config.checkpoint_interval) {
                            state.create_checkpoint();
                            checkpoint_count += 1;
                        }
                    }
                }
                
                if checkpoint_count > 0 {
                    let mut stats_guard = stats.write().await;
                    stats_guard.checkpoints_created += checkpoint_count;
                    debug!("Created {} checkpoints", checkpoint_count);
                }
            }
        });
        
        Ok(())
    }
    
    /// 启动状态清理器
    async fn start_state_cleaner(&self) -> Result<()> {
        let states = self.states.clone();
        let stats = self.stats.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(600)); // 10分钟清理一次
            
            loop {
                interval.tick().await;
                
                let ttl_ms = 24 * 3600 * 1000; // 24小时
                let mut cleaned_count = 0;
                
                {
                    let mut states_guard = states.write().await;
                    let initial_count = states_guard.len();
                    
                    states_guard.retain(|_, state| {
                        !state.is_stale(ttl_ms)
                    });
                    
                    cleaned_count = initial_count - states_guard.len();
                }
                
                if cleaned_count > 0 {
                    let mut stats_guard = stats.write().await;
                    stats_guard.stale_states_cleaned += cleaned_count as u64;
                    stats_guard.active_states = {
                        let states_read = states.read().await;
                        states_read.len()
                    };
                    
                    debug!("Cleaned {} stale states", cleaned_count);
                }
            }
        });
        
        Ok(())
    }
    
    /// 启动统计收集器
    async fn start_stats_collector(&self) -> Result<()> {
        let stats = self.stats.clone();
        let states = self.states.clone();
        let update_queue = self.update_queue.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                let mut stats_guard = stats.write().await;
                
                // 更新活跃状态数
                stats_guard.active_states = {
                    let states_read = states.read().await;
                    states_read.len()
                };
                
                // 更新队列大小
                stats_guard.queue_size = {
                    let queue = update_queue.read().await;
                    queue.len()
                };
                
                // 计算平均更新时间
                if stats_guard.total_updates > 0 {
                    // 这里应该有实际的时间测量逻辑
                    stats_guard.avg_update_time_ms = 5.0; // 简化设置
                }
                
                // 估算内存使用
                stats_guard.memory_usage_mb = (stats_guard.active_states * 1024) as f64 / 1024.0 / 1024.0;
                
                debug!("Incremental updater stats: active_states={}, queue_size={}, avg_time={:.2}ms",
                       stats_guard.active_states, stats_guard.queue_size, stats_guard.avg_update_time_ms);
            }
        });
        
        Ok(())
    }
    
    /// 生成状态键
    fn generate_state_key(&self, symbol: &str, factor_params: &FactorParams) -> String {
        format!("{}_{}_{}_{}", 
                symbol, 
                factor_params.factor_name, 
                factor_params.window_size,
                factor_params.params_hash())
    }
    
    /// 获取统计信息
    pub async fn get_stats(&self) -> IncrementalStats {
        let stats = self.stats.read().await;
        stats.clone()
    }
    
    /// 获取队列状态
    pub async fn get_queue_status(&self) -> QueueStatus {
        let queue = self.update_queue.read().await;
        
        let mut priority_counts = HashMap::new();
        priority_counts.insert(UpdatePriority::Low, 0);
        priority_counts.insert(UpdatePriority::Normal, 0);
        priority_counts.insert(UpdatePriority::High, 0);
        priority_counts.insert(UpdatePriority::Critical, 0);
        
        for update in queue.iter() {
            *priority_counts.get_mut(&update.priority).unwrap() += 1;
        }
        
        QueueStatus {
            total_size: queue.len(),
            priority_counts,
            oldest_waiting_time_ms: queue.front()
                .map(|update| update.submitted_at.elapsed().as_millis() as f64)
                .unwrap_or(0.0),
        }
    }
}

/// 队列状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStatus {
    pub total_size: usize,
    pub priority_counts: HashMap<UpdatePriority, usize>,
    pub oldest_waiting_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factor_processing::cache_manager::{MemoryCacheManager, CacheConfig, CacheManager};
    use crate::factor_processing::window_calculator::WindowCalculatorConfig;

    #[tokio::test]
    async fn test_incremental_updater_creation() {
        let config = IncrementalUpdaterConfig::default();
        let cache_config = CacheConfig::default();
        let cache_manager = Arc::new(CacheManager::new(cache_config));
        let window_calculator = Arc::new(RollingWindowCalculator::new(WindowCalculatorConfig::default()));
        
        let updater = IncrementalUpdater::new(config, cache_manager, window_calculator);
        
        let stats = updater.get_stats().await;
        assert_eq!(stats.active_states, 0);
        assert_eq!(stats.total_updates, 0);
    }

    #[tokio::test]
    async fn test_state_management() {
        let config = IncrementalUpdaterConfig::default();
        let cache_config = CacheConfig::default();
        let cache_manager = Arc::new(CacheManager::new(cache_config));
        let window_calculator = Arc::new(RollingWindowCalculator::new(WindowCalculatorConfig::default()));
        
        let updater = IncrementalUpdater::new(config, cache_manager, window_calculator);
        
        let factor_params = FactorParams {
            factor_name: "sma".to_string(),
            window_size: 10,
            parameters: BTreeMap::new(),
            version: "1.0".to_string(),
        };
        
        // 初始状态不存在
        assert!(updater.get_state("BTCUSD", &factor_params).await.is_none());
        
        // 提交更新
        let input = FactorInput {
            symbol: "BTCUSD".to_string(),
            timestamp: 1000,
            price: 100.0,
            volume: 1000.0,
            open: None,
            high: None,
            low: None,
            close: None,
            metadata: HashMap::new(),
        };
        
        let result = updater.submit_update("BTCUSD".to_string(), factor_params.clone(), input, UpdatePriority::Normal).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_queue_status() {
        let config = IncrementalUpdaterConfig::default();
        let cache_config = CacheConfig::default();
        let cache_manager = Arc::new(CacheManager::new(cache_config));
        let window_calculator = Arc::new(RollingWindowCalculator::new(WindowCalculatorConfig::default()));
        
        let updater = IncrementalUpdater::new(config, cache_manager, window_calculator);
        
        let initial_status = updater.get_queue_status().await;
        assert_eq!(initial_status.total_size, 0);
    }
}