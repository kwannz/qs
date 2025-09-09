use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use crate::{Symbol, Id};

/// 因子缓存键（AG3设计：symbol×window×params_hash）
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct FactorCacheKey {
    pub symbol: Symbol,
    pub window_size: u32,
    pub factor_type: String,
    pub params_hash: u64, // 参数的哈希值
    pub timeframe: String, // e.g., "1m", "5m", "1h", "1d"
}

impl FactorCacheKey {
    pub fn new(
        symbol: Symbol,
        window_size: u32,
        factor_type: &str,
        params: &HashMap<String, serde_json::Value>,
        timeframe: &str,
    ) -> Self {
        let params_hash = Self::hash_params(params);
        Self {
            symbol,
            window_size,
            factor_type: factor_type.to_string(),
            params_hash,
            timeframe: timeframe.to_string(),
        }
    }

    /// 计算参数哈希值
    fn hash_params(params: &HashMap<String, serde_json::Value>) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        
        // 按键排序以确保一致的哈希值
        let mut sorted_params: Vec<_> = params.iter().collect();
        sorted_params.sort_by_key(|&(k, _)| k);
        
        for (key, value) in sorted_params {
            key.hash(&mut hasher);
            // 简化值哈希处理
            value.to_string().hash(&mut hasher);
        }
        
        hasher.finish()
    }
}

/// 因子缓存值
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorCacheValue {
    pub cache_key: FactorCacheKey,
    pub factor_values: Vec<FactorPoint>,
    pub metadata: FactorMetadata,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub access_count: u32,
    pub last_access: DateTime<Utc>,
}

/// 因子数据点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorPoint {
    pub timestamp: DateTime<Utc>,
    pub value: Decimal,
    pub confidence: Option<Decimal>,
    pub volume: Option<u64>,
    pub quality_score: Option<Decimal>,
}

/// 因子元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorMetadata {
    pub calculation_time_ms: u64,
    pub data_points_count: u32,
    pub valid_data_ratio: Decimal, // 0.0-1.0
    pub outlier_count: u32,
    pub missing_data_count: u32,
    pub quality_metrics: FactorQualityMetrics,
    pub dependencies: Vec<String>, // 依赖的其他因子或数据源
}

/// 因子质量指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorQualityMetrics {
    pub stability_score: Decimal, // 0.0-1.0
    pub noise_ratio: Decimal,
    pub predictive_power: Option<Decimal>,
    pub correlation_with_returns: Option<Decimal>,
    pub information_ratio: Option<Decimal>,
}

/// 批处理请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchFactorRequest {
    pub request_id: Id,
    pub requests: Vec<FactorRequest>,
    pub priority: BatchPriority,
    pub max_parallel_jobs: Option<u32>,
    pub timeout_seconds: Option<u64>,
    pub cache_policy: CachePolicy,
}

/// 单个因子请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorRequest {
    pub request_id: Id,
    pub cache_key: FactorCacheKey,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub required_data_ratio: Option<Decimal>, // 最小数据完整性要求
    pub allow_stale_cache: bool,
    pub max_cache_age_seconds: Option<u64>,
}

/// 批处理优先级
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BatchPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// 缓存策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePolicy {
    pub use_memory_cache: bool,
    pub use_persistent_cache: bool,
    pub memory_ttl_seconds: u64,
    pub persistent_ttl_seconds: u64,
    pub max_memory_entries: u32,
    pub eviction_strategy: EvictionStrategy,
}

/// 驱逐策略
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvictionStrategy {
    LRU,    // 最近最少使用
    LFU,    // 最不频繁使用
    FIFO,   // 先进先出
    Random, // 随机
}

/// 批处理响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchFactorResponse {
    pub request_id: Id,
    pub responses: Vec<FactorResponse>,
    pub total_processing_time_ms: u64,
    pub cache_hit_rate: Decimal,
    pub success_count: u32,
    pub failure_count: u32,
    pub completed_at: DateTime<Utc>,
}

/// 单个因子响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorResponse {
    pub request_id: Id,
    pub cache_key: FactorCacheKey,
    pub result: FactorResult,
    pub cache_hit: bool,
    pub processing_time_ms: u64,
}

/// 因子计算结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactorResult {
    Success(FactorCacheValue),
    Error(FactorError),
    Partial(PartialFactorResult),
}

/// 部分因子结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialFactorResult {
    pub cache_value: FactorCacheValue,
    pub warnings: Vec<String>,
    pub missing_periods: Vec<(DateTime<Utc>, DateTime<Utc>)>,
}

/// 因子计算错误
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorError {
    pub error_type: FactorErrorType,
    pub message: String,
    pub retry_after_seconds: Option<u64>,
    pub is_recoverable: bool,
}

/// 因子错误类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FactorErrorType {
    InsufficientData,
    InvalidParameters,
    CalculationFailure,
    TimeoutError,
    DependencyError,
    CacheError,
}

/// 缓存统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    pub total_entries: u32,
    pub memory_entries: u32,
    pub persistent_entries: u32,
    pub total_size_bytes: u64,
    pub hit_rate: Decimal,
    pub miss_rate: Decimal,
    pub eviction_count: u32,
    pub average_access_time_ms: f64,
    pub last_cleanup: DateTime<Utc>,
}

/// 增量更新请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalUpdateRequest {
    pub cache_key: FactorCacheKey,
    pub new_data_start: DateTime<Utc>,
    pub new_data_end: DateTime<Utc>,
    pub merge_strategy: MergeStrategy,
    pub validate_consistency: bool,
}

/// 合并策略
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MergeStrategy {
    Append,    // 简单追加
    Merge,     // 智能合并，处理重叠
    Replace,   // 替换重叠部分
    Validate,  // 验证一致性后合并
}

/// 窗口复用配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowReuseConfig {
    pub enable_reuse: bool,
    pub reuse_threshold: Decimal, // 重叠度阈值
    pub max_extension_ratio: Decimal, // 最大扩展比例
    pub prefer_larger_window: bool,
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self {
            use_memory_cache: true,
            use_persistent_cache: true,
            memory_ttl_seconds: 3600, // 1小时
            persistent_ttl_seconds: 86400 * 7, // 7天
            max_memory_entries: 10000,
            eviction_strategy: EvictionStrategy::LRU,
        }
    }
}

impl Default for WindowReuseConfig {
    fn default() -> Self {
        Self {
            enable_reuse: true,
            reuse_threshold: Decimal::from_parts(8, 0, 0, false, 1), // 0.8
            max_extension_ratio: Decimal::from_parts(2, 0, 0, false, 1), // 2.0
            prefer_larger_window: true,
        }
    }
}