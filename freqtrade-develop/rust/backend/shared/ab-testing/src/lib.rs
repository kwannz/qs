#![allow(unused_imports, unused_variables, dead_code, unused_mut, deprecated, unreachable_code)]

//! A/B测试框架
//! 
//! 为交易模型提供安全的A/B测试和实验管理功能

pub mod experiment;
pub mod traffic_splitter;
pub mod metrics_collector;
pub mod statistical_analyzer;
pub mod experiment_manager;

pub use experiment::{Experiment, ExperimentConfig, ExperimentStatus};
pub use traffic_splitter::{TrafficSplitter, SplittingStrategy};
pub use metrics_collector::{MetricsCollector, ExperimentMetrics};
pub use statistical_analyzer::{StatisticalAnalyzer, StatisticalResult, SignificanceTest};
pub use experiment_manager::{ExperimentManager, ExperimentManagerConfig};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A/B测试结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestResult {
    pub experiment_id: uuid::Uuid,
    pub variant_results: HashMap<String, VariantResult>,
    pub statistical_significance: bool,
    pub confidence_level: f64,
    pub p_value: f64,
    pub effect_size: f64,
    pub recommendation: TestRecommendation,
}

/// 变体测试结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantResult {
    pub variant_name: String,
    pub sample_size: usize,
    pub conversion_rate: f64,
    pub mean_value: f64,
    pub std_deviation: f64,
    pub confidence_interval: (f64, f64),
    pub metrics: HashMap<String, f64>,
}

/// 测试建议
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestRecommendation {
    /// 继续测试，样本量不足
    ContinueTesting {
        required_sample_size: usize,
        estimated_days_remaining: u32,
    },
    /// 变体A获胜
    VariantAWins {
        confidence: f64,
        improvement: f64,
    },
    /// 变体B获胜
    VariantBWins {
        confidence: f64,
        improvement: f64,
    },
    /// 无统计显著性差异
    NoSignificantDifference {
        power: f64,
    },
    /// 测试失败或无效
    TestInvalid {
        reason: String,
    },
}

/// A/B测试错误类型
#[derive(Debug, thiserror::Error)]
pub enum ABTestError {
    #[error("实验不存在: {experiment_id}")]
    ExperimentNotFound { experiment_id: uuid::Uuid },
    
    #[error("实验配置无效: {reason}")]
    InvalidExperimentConfig { reason: String },
    
    #[error("流量分配失败: {reason}")]
    TrafficSplittingError { reason: String },
    
    #[error("统计分析失败: {reason}")]
    StatisticalAnalysisError { reason: String },
    
    #[error("数据库错误: {0}")]
    DatabaseError(#[from] sqlx::Error),
    
    #[error("序列化错误: {0}")]
    SerializationError(#[from] serde_json::Error),
}

pub type ABTestResult2<T> = std::result::Result<T, ABTestError>;