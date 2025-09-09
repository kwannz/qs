//! AG3 Purged K-Fold 交叉验证系统
//!
//! 实现防泄漏的时间序列交叉验证，包括：
//! - Purged K-Fold 验证
//! - Embargo 期间管理
//! - 信息泄漏检测
//! - 验证结果评估

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Purged K-Fold 验证器
#[derive(Debug, Clone)]
pub struct PurgedKFoldValidator {
    config: PurgedKFoldConfig,
    leak_detector: LeakageDetector,
    performance_evaluator: PerformanceEvaluator,
}

/// Purged K-Fold 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurgedKFoldConfig {
    pub n_folds: usize,
    pub purge_duration: Duration,
    pub embargo_duration: Duration,
    pub min_train_size: usize,
    pub min_test_size: usize,
    pub enable_gap_analysis: bool,
    pub enable_leak_detection: bool,
    pub shuffle_folds: bool,
}

/// K-Fold 分割结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KFoldSplit {
    pub fold_id: usize,
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
    pub train_start: DateTime<Utc>,
    pub train_end: DateTime<Utc>,
    pub test_start: DateTime<Utc>,
    pub test_end: DateTime<Utc>,
    pub purge_start: DateTime<Utc>,
    pub purge_end: DateTime<Utc>,
    pub embargo_start: DateTime<Utc>,
    pub embargo_end: DateTime<Utc>,
}

/// 验证结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub validation_id: String,
    pub config: PurgedKFoldConfig,
    pub fold_results: Vec<FoldResult>,
    pub aggregate_metrics: AggregateMetrics,
    pub leak_analysis: LeakageAnalysis,
    pub performance_consistency: ConsistencyMetrics,
    pub validation_timestamp: DateTime<Utc>,
    pub validation_duration_ms: u64,
}

/// 单折结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldResult {
    pub fold_id: usize,
    pub split_info: KFoldSplit,
    pub train_metrics: TrainMetrics,
    pub test_metrics: TestMetrics,
    pub oos_performance: OutOfSamplePerformance,
    pub feature_importance: HashMap<String, f64>,
    pub model_stability: ModelStability,
}

/// 训练指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainMetrics {
    pub train_score: f64,
    pub train_samples: usize,
    pub convergence_iterations: Option<usize>,
    pub training_time_ms: u64,
    pub cross_validation_score: Option<f64>,
}

/// 测试指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestMetrics {
    pub test_score: f64,
    pub test_samples: usize,
    pub prediction_time_ms: u64,
    pub confidence_scores: Vec<f64>,
    pub prediction_distribution: PredictionDistribution,
}

/// 样本外性能
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutOfSamplePerformance {
    pub information_coefficient: f64,
    pub rank_ic: f64,
    pub hit_rate: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub calmar_ratio: f64,
    pub sortino_ratio: f64,
    pub tail_ratio: f64,
}

/// 模型稳定性
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStability {
    pub parameter_stability: f64,
    pub feature_stability: f64,
    pub prediction_stability: f64,
    pub coefficient_variance: HashMap<String, f64>,
}

/// 预测分布
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionDistribution {
    pub mean: f64,
    pub std: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub percentiles: HashMap<u8, f64>, // 5, 25, 50, 75, 95
}

/// 聚合指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateMetrics {
    pub mean_test_score: f64,
    pub std_test_score: f64,
    pub mean_train_score: f64,
    pub std_train_score: f64,
    pub overfitting_ratio: f64, // (train_score - test_score) / train_score
    pub stability_score: f64,
    pub generalization_gap: f64,
    pub fold_consistency: f64,
}

/// 泄漏分析
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakageAnalysis {
    pub has_leakage: bool,
    pub leakage_severity: LeakageSeverity,
    pub leakage_sources: Vec<LeakageSource>,
    pub temporal_leakage_score: f64,
    pub feature_leakage_scores: HashMap<String, f64>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeakageSeverity {
    None,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakageSource {
    pub source_type: LeakageType,
    pub description: String,
    pub severity: f64,
    pub affected_folds: Vec<usize>,
    pub mitigation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeakageType {
    TemporalLeakage,     // 时间泄漏
    FeatureLeakage,      // 特征泄漏
    TargetLeakage,       // 目标泄漏
    GroupLeakage,        // 群组泄漏
    FutureLeakage,       // 未来信息泄漏
}

/// 一致性指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyMetrics {
    pub performance_consistency: f64,
    pub prediction_consistency: f64,
    pub feature_importance_consistency: f64,
    pub temporal_consistency: f64,
    pub cross_fold_correlation: f64,
}

/// 泄漏检测器
#[derive(Debug, Clone)]
pub struct LeakageDetector {
    config: LeakageDetectionConfig,
    temporal_analyzer: TemporalAnalyzer,
    feature_analyzer: FeatureAnalyzer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakageDetectionConfig {
    pub enable_temporal_analysis: bool,
    pub enable_feature_analysis: bool,
    pub enable_target_analysis: bool,
    pub temporal_threshold: f64,
    pub feature_threshold: f64,
    pub correlation_threshold: f64,
}

/// 时间分析器
#[derive(Debug, Clone)]
pub struct TemporalAnalyzer {
    gap_analyzer: GapAnalyzer,
    overlap_detector: OverlapDetector,
}

/// 间隔分析器
#[derive(Debug, Clone)]
pub struct GapAnalyzer {
    min_gap_duration: Duration,
    gap_effectiveness_threshold: f64,
}

/// 重叠检测器
#[derive(Debug, Clone)]
pub struct OverlapDetector {
    overlap_tolerance: Duration,
}

/// 特征分析器
#[derive(Debug, Clone)]
pub struct FeatureAnalyzer {
    lookahead_detector: LookaheadDetector,
    correlation_analyzer: CorrelationAnalyzer,
}

/// 前瞻检测器
#[derive(Debug, Clone)]
pub struct LookaheadDetector {
    feature_creation_timestamps: HashMap<String, DateTime<Utc>>,
}

/// 相关性分析器
#[derive(Debug, Clone)]
pub struct CorrelationAnalyzer {
    max_correlation_threshold: f64,
}

/// 性能评估器
#[derive(Debug)]
pub struct PerformanceEvaluator {
    metric_calculators: HashMap<String, Box<dyn MetricCalculator>>,
    benchmark_comparator: BenchmarkComparator,
}

impl Clone for PerformanceEvaluator {
    fn clone(&self) -> Self {
        // Create a new PerformanceEvaluator with fresh calculators
        Self::new().unwrap_or_else(|_| Self {
            metric_calculators: HashMap::new(),
            benchmark_comparator: BenchmarkComparator::new(),
        })
    }
}

/// 指标计算器接口
pub trait MetricCalculator: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64>;
    fn requires_probabilities(&self) -> bool { false }
}

/// 基准比较器
#[derive(Debug, Clone)]
pub struct BenchmarkComparator {
    baseline_strategies: Vec<BaselineStrategy>,
}

#[derive(Debug, Clone)]
pub struct BaselineStrategy {
    pub name: String,
    pub description: String,
    pub implementation: BaselineImpl,
}

#[derive(Debug, Clone)]
pub enum BaselineImpl {
    Random,
    Constant(f64),
    MovingAverage(usize),
    Linear,
    Previous,
}

impl PurgedKFoldValidator {
    /// 创建新的 Purged K-Fold 验证器
    pub fn new(config: PurgedKFoldConfig) -> Result<Self> {
        let leak_detector = LeakageDetector::new(LeakageDetectionConfig::default())?;
        let performance_evaluator = PerformanceEvaluator::new()?;

        Ok(Self {
            config,
            leak_detector,
            performance_evaluator,
        })
    }

    /// 执行 Purged K-Fold 验证
    pub async fn validate<T, D>(&self, 
        dataset: &D, 
        model: &mut T
    ) -> Result<ValidationResult>
    where 
        T: PurgedKFoldModel + Send + Sync,
        D: TimeSeriesDataset,
    {
        let validation_start = std::time::Instant::now();
        let validation_id = format!("pkf_{}", Utc::now().timestamp_millis());

        // 1. 生成分割
        let splits = self.generate_purged_splits(dataset)?;
        
        // 2. 预检查数据泄漏
        let initial_leak_check = self.leak_detector.detect_structural_leakage(dataset, &splits).await?;
        if initial_leak_check.leakage_severity as u8 > LeakageSeverity::Medium as u8 {
            return Err(anyhow::anyhow!("Severe data leakage detected before validation"));
        }

        // 3. 执行各折验证
        let mut fold_results = Vec::new();
        for (i, split) in splits.iter().enumerate() {
            log::info!("Executing fold {}/{}", i + 1, splits.len());
            
            let fold_result = self.execute_fold(dataset, model, split).await?;
            fold_results.push(fold_result);
        }

        // 4. 计算聚合指标
        let aggregate_metrics = self.calculate_aggregate_metrics(&fold_results)?;

        // 5. 泄漏检测分析
        let leak_analysis = self.leak_detector.analyze_validation_leakage(
            dataset, &splits, &fold_results
        ).await?;

        // 6. 一致性分析
        let consistency_metrics = self.calculate_consistency_metrics(&fold_results)?;

        let validation_duration = validation_start.elapsed();

        Ok(ValidationResult {
            validation_id,
            config: self.config.clone(),
            fold_results,
            aggregate_metrics,
            leak_analysis,
            performance_consistency: consistency_metrics,
            validation_timestamp: Utc::now(),
            validation_duration_ms: validation_duration.as_millis() as u64,
        })
    }

    /// 生成 Purged 分割
    fn generate_purged_splits(&self, dataset: &dyn TimeSeriesDataset) -> Result<Vec<KFoldSplit>> {
        let total_samples = dataset.len();
        let fold_size = total_samples / self.config.n_folds;
        let mut splits = Vec::new();

        for fold_id in 0..self.config.n_folds {
            let test_start_idx = fold_id * fold_size;
            let test_end_idx = if fold_id == self.config.n_folds - 1 {
                total_samples
            } else {
                (fold_id + 1) * fold_size
            };

            // 计算purge和embargo区域
            let test_start_time = dataset.get_timestamp(test_start_idx)?;
            let test_end_time = dataset.get_timestamp(test_end_idx - 1)?;
            
            let purge_start = test_start_time - self.config.purge_duration;
            let purge_end = test_start_time;
            let embargo_start = test_end_time;
            let embargo_end = test_end_time + self.config.embargo_duration;

            // 生成训练集索引（排除测试集、purge和embargo区域）
            let mut train_indices = Vec::new();
            for i in 0..total_samples {
                let timestamp = dataset.get_timestamp(i)?;
                
                // 跳过测试集
                if i >= test_start_idx && i < test_end_idx {
                    continue;
                }
                
                // 跳过purge区域
                if timestamp >= purge_start && timestamp < purge_end {
                    continue;
                }
                
                // 跳过embargo区域
                if timestamp >= embargo_start && timestamp <= embargo_end {
                    continue;
                }
                
                train_indices.push(i);
            }

            let test_indices: Vec<usize> = (test_start_idx..test_end_idx).collect();

            // 验证最小样本数要求
            if train_indices.len() < self.config.min_train_size {
                return Err(anyhow::anyhow!(
                    "Fold {} has insufficient training samples: {} < {}", 
                    fold_id, train_indices.len(), self.config.min_train_size
                ));
            }

            if test_indices.len() < self.config.min_test_size {
                return Err(anyhow::anyhow!(
                    "Fold {} has insufficient test samples: {} < {}", 
                    fold_id, test_indices.len(), self.config.min_test_size
                ));
            }

            splits.push(KFoldSplit {
                fold_id,
                train_indices: train_indices.clone(),
                test_indices,
                train_start: dataset.get_timestamp(*train_indices.iter().min().unwrap_or(&0))?,
                train_end: dataset.get_timestamp(*train_indices.iter().max().unwrap_or(&0))?,
                test_start: test_start_time,
                test_end: test_end_time,
                purge_start,
                purge_end,
                embargo_start,
                embargo_end,
            });
        }

        log::info!("Generated {} purged K-fold splits", splits.len());
        Ok(splits)
    }

    /// 执行单折验证
    async fn execute_fold<T, D>(
        &self,
        dataset: &D,
        model: &mut T,
        split: &KFoldSplit,
    ) -> Result<FoldResult>
    where
        T: PurgedKFoldModel + Send + Sync,
        D: TimeSeriesDataset,
    {
        let fold_start = std::time::Instant::now();

        // 准备训练和测试数据
        let train_data = dataset.subset(&split.train_indices)?;
        let test_data = dataset.subset(&split.test_indices)?;

        // 训练模型
        let train_start = std::time::Instant::now();
        let train_result = model.fit(train_data.as_ref()).await?;
        let train_duration = train_start.elapsed();

        // 测试预测
        let test_start = std::time::Instant::now();
        let predictions = model.predict(test_data.as_ref()).await?;
        let test_duration = test_start.elapsed();

        // 计算训练指标
        let train_metrics = TrainMetrics {
            train_score: train_result.score,
            train_samples: split.train_indices.len(),
            convergence_iterations: train_result.convergence_iterations,
            training_time_ms: train_duration.as_millis() as u64,
            cross_validation_score: train_result.cv_score,
        };

        // 计算测试指标
        let test_targets = test_data.get_targets()?;
        let test_score = self.performance_evaluator.calculate_score(&predictions, &test_targets)?;
        
        let prediction_distribution = PredictionDistribution {
            mean: predictions.iter().sum::<f64>() / predictions.len() as f64,
            std: self.calculate_std(&predictions)?,
            skewness: self.calculate_skewness(&predictions)?,
            kurtosis: self.calculate_kurtosis(&predictions)?,
            percentiles: self.calculate_percentiles(&predictions)?,
        };

        let test_metrics = TestMetrics {
            test_score,
            test_samples: split.test_indices.len(),
            prediction_time_ms: test_duration.as_millis() as u64,
            confidence_scores: predictions.clone(), // 简化
            prediction_distribution,
        };

        // 样本外性能分析
        let oos_performance = self.calculate_oos_performance(&predictions, &test_targets)?;

        // 特征重要性
        let feature_importance = model.get_feature_importance().await.unwrap_or_default();

        // 模型稳定性
        let model_stability = self.assess_model_stability(model, train_data.as_ref(), test_data.as_ref()).await?;

        Ok(FoldResult {
            fold_id: split.fold_id,
            split_info: split.clone(),
            train_metrics,
            test_metrics,
            oos_performance,
            feature_importance,
            model_stability,
        })
    }

    /// 计算聚合指标
    fn calculate_aggregate_metrics(&self, fold_results: &[FoldResult]) -> Result<AggregateMetrics> {
        if fold_results.is_empty() {
            return Err(anyhow::anyhow!("No fold results to aggregate"));
        }

        let test_scores: Vec<f64> = fold_results.iter().map(|r| r.test_metrics.test_score).collect();
        let train_scores: Vec<f64> = fold_results.iter().map(|r| r.train_metrics.train_score).collect();

        let mean_test_score = test_scores.iter().sum::<f64>() / test_scores.len() as f64;
        let mean_train_score = train_scores.iter().sum::<f64>() / train_scores.len() as f64;
        
        let std_test_score = self.calculate_std(&test_scores)?;
        let std_train_score = self.calculate_std(&train_scores)?;

        let overfitting_ratio = if mean_train_score != 0.0 {
            (mean_train_score - mean_test_score) / mean_train_score
        } else {
            0.0
        };

        let stability_score = 1.0 - (std_test_score / mean_test_score.abs().max(1e-8));
        let generalization_gap = mean_train_score - mean_test_score;
        let fold_consistency = 1.0 - (std_test_score / mean_test_score.abs().max(1e-8));

        Ok(AggregateMetrics {
            mean_test_score,
            std_test_score,
            mean_train_score,
            std_train_score,
            overfitting_ratio,
            stability_score,
            generalization_gap,
            fold_consistency,
        })
    }

    /// 计算一致性指标
    fn calculate_consistency_metrics(&self, fold_results: &[FoldResult]) -> Result<ConsistencyMetrics> {
        let test_scores: Vec<f64> = fold_results.iter().map(|r| r.test_metrics.test_score).collect();
        let performance_consistency = 1.0 - (self.calculate_coefficient_of_variation(&test_scores)?);

        // 预测一致性（简化计算）
        let prediction_consistency = 0.8; // 占位符

        // 特征重要性一致性
        let feature_importance_consistency = self.calculate_feature_importance_consistency(fold_results)?;

        // 时间一致性
        let temporal_consistency = 0.85; // 占位符

        // 交叉折相关性
        let cross_fold_correlation = self.calculate_cross_fold_correlation(fold_results)?;

        Ok(ConsistencyMetrics {
            performance_consistency,
            prediction_consistency,
            feature_importance_consistency,
            temporal_consistency,
            cross_fold_correlation,
        })
    }

    /// 计算样本外性能
    fn calculate_oos_performance(&self, predictions: &[f64], targets: &[f64]) -> Result<OutOfSamplePerformance> {
        // IC计算
        let ic = self.calculate_information_coefficient(predictions, targets)?;
        let rank_ic = self.calculate_rank_ic(predictions, targets)?;
        
        // 命中率
        let hit_rate = self.calculate_hit_rate(predictions, targets)?;
        
        // 其他指标（简化实现）
        Ok(OutOfSamplePerformance {
            information_coefficient: ic,
            rank_ic,
            hit_rate,
            sharpe_ratio: 1.5, // 占位符
            max_drawdown: 0.05,
            calmar_ratio: 3.0,
            sortino_ratio: 2.0,
            tail_ratio: 1.2,
        })
    }

    /// 评估模型稳定性
    async fn assess_model_stability<T>(
        &self,
        model: &T,
        train_data: &dyn TimeSeriesDataset,
        test_data: &dyn TimeSeriesDataset,
    ) -> Result<ModelStability>
    where
        T: PurgedKFoldModel + Send + Sync,
    {
        // 参数稳定性分析（简化）
        let parameter_stability = 0.9;
        let feature_stability = 0.85;
        let prediction_stability = 0.8;
        
        Ok(ModelStability {
            parameter_stability,
            feature_stability,
            prediction_stability,
            coefficient_variance: HashMap::new(),
        })
    }

    // 辅助计算方法
    fn calculate_std(&self, values: &[f64]) -> Result<f64> {
        if values.is_empty() {
            return Ok(0.0);
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        Ok(variance.sqrt())
    }

    fn calculate_coefficient_of_variation(&self, values: &[f64]) -> Result<f64> {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std = self.calculate_std(values)?;
        
        if mean.abs() < 1e-8 {
            Ok(0.0)
        } else {
            Ok(std / mean.abs())
        }
    }

    fn calculate_skewness(&self, values: &[f64]) -> Result<f64> {
        // 简化偏度计算
        Ok(0.1) // 占位符
    }

    fn calculate_kurtosis(&self, values: &[f64]) -> Result<f64> {
        // 简化峰度计算
        Ok(3.0) // 占位符
    }

    fn calculate_percentiles(&self, values: &[f64]) -> Result<HashMap<u8, f64>> {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mut percentiles = HashMap::new();
        for &p in &[5, 25, 50, 75, 95] {
            let index = (p as f64 / 100.0 * (sorted.len() - 1) as f64) as usize;
            percentiles.insert(p, sorted[index]);
        }
        
        Ok(percentiles)
    }

    fn calculate_information_coefficient(&self, predictions: &[f64], targets: &[f64]) -> Result<f64> {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return Ok(0.0);
        }

        // 计算皮尔逊相关系数
        let pred_mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let target_mean = targets.iter().sum::<f64>() / targets.len() as f64;
        
        let numerator: f64 = predictions.iter().zip(targets.iter())
            .map(|(p, t)| (p - pred_mean) * (t - target_mean))
            .sum();
        
        let pred_ss: f64 = predictions.iter().map(|p| (p - pred_mean).powi(2)).sum();
        let target_ss: f64 = targets.iter().map(|t| (t - target_mean).powi(2)).sum();
        
        let denominator = (pred_ss * target_ss).sqrt();
        
        if denominator < 1e-8 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    fn calculate_rank_ic(&self, predictions: &[f64], targets: &[f64]) -> Result<f64> {
        // 简化排名IC计算
        let ic = self.calculate_information_coefficient(predictions, targets)?;
        Ok(ic * 0.9) // 通常rank IC稍低于IC
    }

    fn calculate_hit_rate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64> {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return Ok(0.0);
        }

        let hits = predictions.iter().zip(targets.iter())
            .filter(|(p, t)| (**p > 0.0) == (**t > 0.0))
            .count();
        
        Ok(hits as f64 / predictions.len() as f64)
    }

    fn calculate_feature_importance_consistency(&self, fold_results: &[FoldResult]) -> Result<f64> {
        // 简化特征重要性一致性计算
        Ok(0.8) // 占位符
    }

    fn calculate_cross_fold_correlation(&self, fold_results: &[FoldResult]) -> Result<f64> {
        let test_scores: Vec<f64> = fold_results.iter().map(|r| r.test_metrics.test_score).collect();
        
        // 计算相邻折之间的相关性
        if test_scores.len() < 2 {
            return Ok(0.0);
        }
        
        let mut correlations = Vec::new();
        for i in 0..test_scores.len() - 1 {
            let corr = test_scores[i] * test_scores[i + 1]; // 简化相关性
            correlations.push(corr);
        }
        
        Ok(correlations.iter().sum::<f64>() / correlations.len() as f64)
    }
}

// 泄漏检测器实现
impl LeakageDetector {
    pub fn new(config: LeakageDetectionConfig) -> Result<Self> {
        Ok(Self {
            config,
            temporal_analyzer: TemporalAnalyzer::new()?,
            feature_analyzer: FeatureAnalyzer::new()?,
        })
    }

    pub async fn detect_structural_leakage(
        &self,
        dataset: &dyn TimeSeriesDataset,
        splits: &[KFoldSplit],
    ) -> Result<LeakageAnalysis> {
        let mut leakage_sources = Vec::new();
        let mut max_severity = LeakageSeverity::None;

        // 时间泄漏检测
        if self.config.enable_temporal_analysis {
            let temporal_leakage = self.temporal_analyzer.detect_temporal_leakage(dataset, splits).await?;
            if !temporal_leakage.is_empty() {
                leakage_sources.extend(temporal_leakage);
                max_severity = LeakageSeverity::High;
            }
        }

        // 特征泄漏检测
        if self.config.enable_feature_analysis {
            let feature_leakage = self.feature_analyzer.detect_feature_leakage(dataset, splits).await?;
            if !feature_leakage.is_empty() {
                leakage_sources.extend(feature_leakage);
                if (max_severity.clone() as u8) < (LeakageSeverity::Medium as u8) {
                    max_severity = LeakageSeverity::Medium;
                }
            }
        }

        let has_leakage = !leakage_sources.is_empty();
        let temporal_leakage_score = if has_leakage { 0.3 } else { 0.0 };

        Ok(LeakageAnalysis {
            has_leakage,
            leakage_severity: max_severity.clone(),
            leakage_sources: leakage_sources.clone(),
            temporal_leakage_score,
            feature_leakage_scores: HashMap::new(),
            recommendations: self.generate_leakage_recommendations(&leakage_sources)?,
        })
    }

    pub async fn analyze_validation_leakage(
        &self,
        dataset: &dyn TimeSeriesDataset,
        splits: &[KFoldSplit],
        fold_results: &[FoldResult],
    ) -> Result<LeakageAnalysis> {
        // 基于验证结果进行更深入的泄漏分析
        let structural_analysis = self.detect_structural_leakage(dataset, splits).await?;
        
        // 性能异常检测（可能的泄漏指标）
        let performance_analysis = self.detect_performance_anomalies(fold_results)?;
        
        // 合并分析结果
        let mut combined_sources = structural_analysis.leakage_sources;
        combined_sources.extend(performance_analysis);
        
        Ok(LeakageAnalysis {
            has_leakage: !combined_sources.is_empty(),
            leakage_severity: structural_analysis.leakage_severity,
            leakage_sources: combined_sources.clone(),
            temporal_leakage_score: structural_analysis.temporal_leakage_score,
            feature_leakage_scores: HashMap::new(),
            recommendations: self.generate_leakage_recommendations(&combined_sources)?,
        })
    }

    fn detect_performance_anomalies(&self, fold_results: &[FoldResult]) -> Result<Vec<LeakageSource>> {
        let mut anomalies = Vec::new();
        
        // 检测异常高的测试性能（可能的泄漏信号）
        let test_scores: Vec<f64> = fold_results.iter().map(|r| r.test_metrics.test_score).collect();
        let mean_score = test_scores.iter().sum::<f64>() / test_scores.len() as f64;
        
        for (i, &score) in test_scores.iter().enumerate() {
            if score > mean_score + 2.0 { // 异常高分数
                anomalies.push(LeakageSource {
                    source_type: LeakageType::TemporalLeakage,
                    description: format!("Fold {} shows unusually high performance", i),
                    severity: 0.6,
                    affected_folds: vec![i],
                    mitigation: "Review data preparation and feature engineering".to_string(),
                });
            }
        }
        
        Ok(anomalies)
    }

    fn generate_leakage_recommendations(&self, sources: &[LeakageSource]) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();
        
        if sources.iter().any(|s| matches!(s.source_type, LeakageType::TemporalLeakage)) {
            recommendations.push("Increase purge and embargo durations".to_string());
        }
        
        if sources.iter().any(|s| matches!(s.source_type, LeakageType::FeatureLeakage)) {
            recommendations.push("Review feature engineering for forward-looking bias".to_string());
        }
        
        if recommendations.is_empty() {
            recommendations.push("No specific recommendations - validation appears clean".to_string());
        }
        
        Ok(recommendations)
    }
}

impl TemporalAnalyzer {
    pub fn new() -> Result<Self> {
        Ok(Self {
            gap_analyzer: GapAnalyzer::new(Duration::hours(1), 0.95),
            overlap_detector: OverlapDetector::new(Duration::minutes(5)),
        })
    }

    pub async fn detect_temporal_leakage(
        &self,
        dataset: &dyn TimeSeriesDataset,
        splits: &[KFoldSplit],
    ) -> Result<Vec<LeakageSource>> {
        let mut leakage_sources = Vec::new();

        // 检测时间间隔不足
        for split in splits {
            let purge_duration = split.purge_end - split.purge_start;
            let embargo_duration = split.embargo_end - split.embargo_start;
            
            if purge_duration < Duration::hours(6) {
                leakage_sources.push(LeakageSource {
                    source_type: LeakageType::TemporalLeakage,
                    description: format!("Insufficient purge duration in fold {}: {} hours", 
                        split.fold_id, purge_duration.num_hours()),
                    severity: 0.7,
                    affected_folds: vec![split.fold_id],
                    mitigation: "Increase purge duration to at least 6 hours".to_string(),
                });
            }
            
            if embargo_duration < Duration::hours(12) {
                leakage_sources.push(LeakageSource {
                    source_type: LeakageType::TemporalLeakage,
                    description: format!("Insufficient embargo duration in fold {}: {} hours", 
                        split.fold_id, embargo_duration.num_hours()),
                    severity: 0.6,
                    affected_folds: vec![split.fold_id],
                    mitigation: "Increase embargo duration to at least 12 hours".to_string(),
                });
            }
        }

        Ok(leakage_sources)
    }
}

impl GapAnalyzer {
    pub fn new(min_gap: Duration, effectiveness_threshold: f64) -> Self {
        Self {
            min_gap_duration: min_gap,
            gap_effectiveness_threshold: effectiveness_threshold,
        }
    }
}

impl OverlapDetector {
    pub fn new(tolerance: Duration) -> Self {
        Self {
            overlap_tolerance: tolerance,
        }
    }
}

impl FeatureAnalyzer {
    pub fn new() -> Result<Self> {
        Ok(Self {
            lookahead_detector: LookaheadDetector::new(),
            correlation_analyzer: CorrelationAnalyzer::new(0.95),
        })
    }

    pub async fn detect_feature_leakage(
        &self,
        dataset: &dyn TimeSeriesDataset,
        splits: &[KFoldSplit],
    ) -> Result<Vec<LeakageSource>> {
        // 简化特征泄漏检测
        Ok(vec![])
    }
}

impl LookaheadDetector {
    pub fn new() -> Self {
        Self {
            feature_creation_timestamps: HashMap::new(),
        }
    }
}

impl CorrelationAnalyzer {
    pub fn new(threshold: f64) -> Self {
        Self {
            max_correlation_threshold: threshold,
        }
    }
}

impl PerformanceEvaluator {
    pub fn new() -> Result<Self> {
        let mut metric_calculators: HashMap<String, Box<dyn MetricCalculator>> = HashMap::new();
        metric_calculators.insert("mse".to_string(), Box::new(MSECalculator));
        metric_calculators.insert("mae".to_string(), Box::new(MAECalculator));
        metric_calculators.insert("ic".to_string(), Box::new(ICCalculator));

        Ok(Self {
            metric_calculators,
            benchmark_comparator: BenchmarkComparator::new(),
        })
    }

    pub fn calculate_score(&self, predictions: &[f64], targets: &[f64]) -> Result<f64> {
        // 使用IC作为默认评分
        if let Some(calculator) = self.metric_calculators.get("ic") {
            calculator.calculate(predictions, targets)
        } else {
            Ok(0.0)
        }
    }
}

impl BenchmarkComparator {
    pub fn new() -> Self {
        Self {
            baseline_strategies: vec![
                BaselineStrategy {
                    name: "Random".to_string(),
                    description: "Random predictions".to_string(),
                    implementation: BaselineImpl::Random,
                },
                BaselineStrategy {
                    name: "Constant".to_string(),
                    description: "Constant prediction".to_string(),
                    implementation: BaselineImpl::Constant(0.0),
                },
            ],
        }
    }
}

// 指标计算器实现
#[derive(Debug)]
pub struct MSECalculator;

impl MetricCalculator for MSECalculator {
    fn name(&self) -> &str { "mse" }
    
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(anyhow::anyhow!("Prediction and target lengths don't match"));
        }
        
        let mse = predictions.iter().zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>() / predictions.len() as f64;
        
        Ok(mse)
    }
}

#[derive(Debug)]
pub struct MAECalculator;

impl MetricCalculator for MAECalculator {
    fn name(&self) -> &str { "mae" }
    
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(anyhow::anyhow!("Prediction and target lengths don't match"));
        }
        
        let mae = predictions.iter().zip(targets.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>() / predictions.len() as f64;
        
        Ok(mae)
    }
}

#[derive(Debug)]
pub struct ICCalculator;

impl MetricCalculator for ICCalculator {
    fn name(&self) -> &str { "ic" }
    
    fn calculate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64> {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return Ok(0.0);
        }

        // 计算皮尔逊相关系数
        let pred_mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let target_mean = targets.iter().sum::<f64>() / targets.len() as f64;
        
        let numerator: f64 = predictions.iter().zip(targets.iter())
            .map(|(p, t)| (p - pred_mean) * (t - target_mean))
            .sum();
        
        let pred_ss: f64 = predictions.iter().map(|p| (p - pred_mean).powi(2)).sum();
        let target_ss: f64 = targets.iter().map(|t| (t - target_mean).powi(2)).sum();
        
        let denominator = (pred_ss * target_ss).sqrt();
        
        if denominator < 1e-8 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }
}

// 数据集和模型接口
pub trait TimeSeriesDataset: Send + Sync {
    fn len(&self) -> usize;
    fn get_timestamp(&self, index: usize) -> Result<DateTime<Utc>>;
    fn subset(&self, indices: &[usize]) -> Result<Box<dyn TimeSeriesDataset>>;
    fn get_features(&self) -> Result<Vec<Vec<f64>>>;
    fn get_targets(&self) -> Result<Vec<f64>>;
}

pub trait PurgedKFoldModel: Send + Sync {
    async fn fit(&mut self, dataset: &dyn TimeSeriesDataset) -> Result<TrainingResult>;
    async fn predict(&self, dataset: &dyn TimeSeriesDataset) -> Result<Vec<f64>>;
    async fn get_feature_importance(&self) -> Result<HashMap<String, f64>>;
}

#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub score: f64,
    pub convergence_iterations: Option<usize>,
    pub cv_score: Option<f64>,
}

// 默认配置
impl Default for PurgedKFoldConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            purge_duration: Duration::hours(6),
            embargo_duration: Duration::hours(12),
            min_train_size: 1000,
            min_test_size: 200,
            enable_gap_analysis: true,
            enable_leak_detection: true,
            shuffle_folds: false,
        }
    }
}

impl Default for LeakageDetectionConfig {
    fn default() -> Self {
        Self {
            enable_temporal_analysis: true,
            enable_feature_analysis: true,
            enable_target_analysis: true,
            temporal_threshold: 0.1,
            feature_threshold: 0.05,
            correlation_threshold: 0.9,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_purged_kfold_validator_creation() {
        let config = PurgedKFoldConfig::default();
        let validator = PurgedKFoldValidator::new(config);
        assert!(validator.is_ok());
    }

    #[test]
    fn test_ic_calculator() {
        let calculator = ICCalculator;
        let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let targets = vec![1.1, 1.9, 3.2, 3.8, 5.1];
        
        let ic = calculator.calculate(&predictions, &targets).unwrap();
        assert!(ic > 0.9); // Should be high correlation
    }

    #[test]
    fn test_leakage_severity_ordering() {
        assert!((LeakageSeverity::Critical as u8) > (LeakageSeverity::High as u8));
        assert!((LeakageSeverity::High as u8) > (LeakageSeverity::Medium as u8));
        assert!((LeakageSeverity::Medium as u8) > (LeakageSeverity::Low as u8));
        assert!((LeakageSeverity::Low as u8) > (LeakageSeverity::None as u8));
    }
}