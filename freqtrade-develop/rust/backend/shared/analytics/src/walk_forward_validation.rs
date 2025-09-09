//! AG3 Walk-Forward 验证系统
//!
//! 实现前进分析验证，用于时间序列模型的稳健性测试，包括：
//! - Walk-Forward 分割和验证
//! - 参数稳定性分析
//! - 效率比率计算
//! - 退化检测和适应性分析

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Walk-Forward 验证器
#[derive(Debug, Clone)]
pub struct WalkForwardValidator {
    config: WalkForwardConfig,
    parameter_analyzer: Arc<ParameterAnalyzer>,
    efficiency_calculator: Arc<EfficiencyCalculator>,
    degradation_detector: Arc<DegradationDetector>,
    adaptation_manager: Arc<AdaptationManager>,
}

/// Walk-Forward 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardConfig {
    pub initial_train_window: Duration,
    pub test_window: Duration,
    pub step_size: Duration,
    pub min_train_samples: usize,
    pub min_test_samples: usize,
    pub max_periods: Option<usize>,
    pub enable_parameter_optimization: bool,
    pub enable_efficiency_analysis: bool,
    pub enable_degradation_detection: bool,
    pub reoptimization_frequency: u32, // 每N个周期重新优化
    pub stability_threshold: f64,
    pub efficiency_threshold: f64,
}

/// Walk-Forward 结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardResult {
    pub validation_id: String,
    pub config: WalkForwardConfig,
    pub periods: Vec<WalkForwardPeriod>,
    pub aggregate_metrics: WalkForwardAggregateMetrics,
    pub parameter_stability: ParameterStabilityAnalysis,
    pub efficiency_analysis: EfficiencyAnalysis,
    pub degradation_analysis: DegradationAnalysis,
    pub adaptation_history: AdaptationHistory,
    pub validation_timestamp: DateTime<Utc>,
    pub total_duration_ms: u64,
}

/// Walk-Forward 周期
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardPeriod {
    pub period_id: u32,
    pub train_start: DateTime<Utc>,
    pub train_end: DateTime<Utc>,
    pub test_start: DateTime<Utc>,
    pub test_end: DateTime<Utc>,
    pub train_samples: usize,
    pub test_samples: usize,
    pub optimization_time_ms: u64,
    pub prediction_time_ms: u64,
    pub in_sample_metrics: InSampleMetrics,
    pub out_of_sample_metrics: OutOfSampleMetrics,
    pub optimized_parameters: HashMap<String, f64>,
    pub parameter_changes: HashMap<String, ParameterChange>,
    pub efficiency_ratio: f64,
    pub degradation_signals: Vec<DegradationSignal>,
    pub adaptations_applied: Vec<AdaptationAction>,
}

/// 样本内指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InSampleMetrics {
    pub return_pct: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub volatility: f64,
    pub hit_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
    pub information_coefficient: f64,
    pub turnover: f64,
}

/// 样本外指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutOfSampleMetrics {
    pub return_pct: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub volatility: f64,
    pub hit_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
    pub information_coefficient: f64,
    pub turnover: f64,
    pub slippage_cost: f64,
    pub transaction_cost: f64,
}

/// 参数变化
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterChange {
    pub parameter_name: String,
    pub old_value: f64,
    pub new_value: f64,
    pub change_magnitude: f64,
    pub change_significance: f64,
    pub reason: ParameterChangeReason,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterChangeReason {
    Optimization,
    DegradationResponse,
    MarketRegimeChange,
    PerformanceImprovement,
    RiskManagement,
    Manual,
}

/// 退化信号
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationSignal {
    pub signal_type: DegradationType,
    pub severity: DegradationSeverity,
    pub description: String,
    pub metric_value: f64,
    pub threshold: f64,
    pub detection_timestamp: DateTime<Utc>,
    pub recommended_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DegradationType {
    PerformanceDecline,
    ParameterInstability,
    EfficiencyDrop,
    OverfittingDetected,
    MarketRegimeShift,
    ModelStaleness,
    RiskIncrease,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DegradationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// 适应性行动
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationAction {
    pub action_type: AdaptationType,
    pub description: String,
    pub parameters_affected: Vec<String>,
    pub expected_impact: f64,
    pub success_probability: f64,
    pub implementation_timestamp: DateTime<Utc>,
    pub result: Option<AdaptationResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AdaptationType {
    ParameterReoptimization,
    ModelRetraining,
    FeatureSelection,
    RegimeAdaptation,
    RiskAdjustment,
    StrategyRotation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationResult {
    pub success: bool,
    pub performance_improvement: f64,
    pub stability_improvement: f64,
    pub side_effects: Vec<String>,
}

/// Walk-Forward 聚合指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardAggregateMetrics {
    pub total_periods: usize,
    pub successful_periods: usize,
    pub avg_efficiency_ratio: f64,
    pub std_efficiency_ratio: f64,
    pub cumulative_is_return: f64,
    pub cumulative_oos_return: f64,
    pub avg_is_sharpe: f64,
    pub avg_oos_sharpe: f64,
    pub parameter_stability_score: f64,
    pub degradation_frequency: f64,
    pub adaptation_success_rate: f64,
    pub overall_robustness_score: f64,
}

/// 参数稳定性分析
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterStabilityAnalysis {
    pub parameter_statistics: HashMap<String, ParameterStatistics>,
    pub stability_trends: HashMap<String, StabilityTrend>,
    pub correlation_matrix: HashMap<String, HashMap<String, f64>>,
    pub regime_sensitivity: HashMap<String, RegimeSensitivity>,
    pub overall_stability_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterStatistics {
    pub parameter_name: String,
    pub mean_value: f64,
    pub std_value: f64,
    pub min_value: f64,
    pub max_value: f64,
    pub coefficient_of_variation: f64,
    pub trend_slope: f64,
    pub change_frequency: f64,
    pub stability_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityTrend {
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub trend_significance: f64,
    pub recent_volatility: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeSensitivity {
    pub parameter_name: String,
    pub regime_correlations: HashMap<String, f64>,
    pub adaptation_frequency: f64,
    pub stability_across_regimes: f64,
}

/// 效率分析
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyAnalysis {
    pub period_efficiencies: Vec<PeriodEfficiency>,
    pub efficiency_trends: EfficiencyTrends,
    pub efficiency_factors: HashMap<String, f64>,
    pub optimization_effectiveness: OptimizationEffectiveness,
    pub cost_benefit_analysis: CostBenefitAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodEfficiency {
    pub period_id: u32,
    pub efficiency_ratio: f64,
    pub is_performance: f64,
    pub oos_performance: f64,
    pub optimization_cost: f64,
    pub net_efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyTrends {
    pub overall_trend: TrendDirection,
    pub trend_strength: f64,
    pub recent_efficiency: f64,
    pub efficiency_volatility: f64,
    pub declining_periods: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationEffectiveness {
    pub avg_improvement: f64,
    pub success_rate: f64,
    pub diminishing_returns_detected: bool,
    pub optimal_reopt_frequency: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBenefitAnalysis {
    pub total_optimization_cost: f64,
    pub total_performance_gain: f64,
    pub net_benefit: f64,
    pub roi: f64,
    pub break_even_periods: u32,
}

/// 退化分析
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationAnalysis {
    pub degradation_frequency: f64,
    pub degradation_patterns: Vec<DegradationPattern>,
    pub early_warning_indicators: Vec<EarlyWarningIndicator>,
    pub recovery_strategies: HashMap<DegradationType, RecoveryStrategy>,
    pub prevention_recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationPattern {
    pub pattern_type: String,
    pub frequency: f64,
    pub typical_duration_periods: u32,
    pub severity_distribution: HashMap<DegradationSeverity, f64>,
    pub precursors: Vec<String>,
    pub outcomes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyWarningIndicator {
    pub indicator_name: String,
    pub current_value: f64,
    pub warning_threshold: f64,
    pub critical_threshold: f64,
    pub trend_direction: TrendDirection,
    pub reliability_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryStrategy {
    pub strategy_name: String,
    pub actions: Vec<AdaptationType>,
    pub expected_recovery_time: u32,
    pub success_probability: f64,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

/// 适应性历史
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationHistory {
    pub total_adaptations: usize,
    pub adaptation_types_frequency: HashMap<AdaptationType, usize>,
    pub success_rate_by_type: HashMap<AdaptationType, f64>,
    pub learning_curve: Vec<LearningPoint>,
    pub best_practices: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningPoint {
    pub period: u32,
    pub cumulative_learning: f64,
    pub adaptation_effectiveness: f64,
    pub knowledge_gained: String,
}

/// 参数分析器
#[derive(Debug)]
pub struct ParameterAnalyzer {
    stability_calculator: StabilityCalculator,
    trend_analyzer: TrendAnalyzer,
    correlation_analyzer: CorrelationAnalyzer,
}

/// 稳定性计算器
#[derive(Debug)]
pub struct StabilityCalculator {
    window_size: usize,
    stability_threshold: f64,
}

/// 趋势分析器
#[derive(Debug)]
pub struct TrendAnalyzer {
    min_periods: usize,
    significance_threshold: f64,
}

/// 相关性分析器
#[derive(Debug)]
pub struct CorrelationAnalyzer {
    min_correlation: f64,
}

/// 效率计算器
#[derive(Debug)]
pub struct EfficiencyCalculator {
    efficiency_metrics: Vec<String>,
    cost_models: HashMap<String, Box<dyn CostModel>>,
}

/// 成本模型接口
pub trait CostModel: Send + Sync + std::fmt::Debug {
    fn calculate_cost(&self, period: &WalkForwardPeriod) -> Result<f64>;
}

/// 退化检测器
#[derive(Debug)]
pub struct DegradationDetector {
    detectors: HashMap<DegradationType, Box<dyn DegradationMetric>>,
    alert_thresholds: HashMap<DegradationType, f64>,
}

/// 退化指标接口
pub trait DegradationMetric: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;
    fn calculate(&self, periods: &[WalkForwardPeriod]) -> Result<f64>;
    fn get_threshold(&self) -> f64;
}

/// 适应性管理器
#[derive(Debug)]
pub struct AdaptationManager {
    adaptation_strategies: HashMap<AdaptationType, Box<dyn AdaptationStrategy>>,
    decision_engine: AdaptationDecisionEngine,
}

/// 适应性策略接口
pub trait AdaptationStrategy: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;
    fn can_handle(&self, degradation: &DegradationSignal) -> bool;
    fn create_action(&self, degradation: &DegradationSignal) -> Result<AdaptationAction>;
    fn estimate_success_probability(&self, action: &AdaptationAction) -> f64;
}

/// 适应性决策引擎
#[derive(Debug)]
pub struct AdaptationDecisionEngine {
    decision_rules: Vec<DecisionRule>,
    risk_tolerance: f64,
}

#[derive(Debug, Clone)]
pub struct DecisionRule {
    pub condition: String,
    pub action: AdaptationType,
    pub priority: u8,
    pub confidence: f64,
}

impl WalkForwardValidator {
    /// 创建新的 Walk-Forward 验证器
    pub fn new(config: WalkForwardConfig) -> Result<Self> {
        let parameter_analyzer = Arc::new(ParameterAnalyzer::new()?);
        let efficiency_calculator = Arc::new(EfficiencyCalculator::new()?);
        let degradation_detector = Arc::new(DegradationDetector::new()?);
        let adaptation_manager = Arc::new(AdaptationManager::new()?);

        Ok(Self {
            config,
            parameter_analyzer,
            efficiency_calculator,
            degradation_detector,
            adaptation_manager,
        })
    }

    /// 执行 Walk-Forward 验证
    pub async fn validate<T, D>(
        &self,
        dataset: &D,
        model_factory: &T,
    ) -> Result<WalkForwardResult>
    where
        T: WalkForwardModelFactory + Send + Sync,
        D: WalkForwardDataset + Send + Sync,
    {
        let validation_start = std::time::Instant::now();
        let validation_id = format!("wf_{}", Utc::now().timestamp_millis());

        // 1. 生成时间分割
        let time_splits = self.generate_time_splits(dataset)?;
        log::info!("Generated {} walk-forward periods", time_splits.len());

        // 2. 执行各周期验证
        let mut periods = Vec::new();
        let mut adaptation_history = Vec::new();

        for (i, split) in time_splits.iter().enumerate() {
            log::info!("Processing period {}/{}", i + 1, time_splits.len());

            let period_result = self.execute_period(
                dataset, 
                model_factory, 
                split, 
                i as u32,
                &periods, // 历史周期用于适应性分析
            ).await?;

            // 检测退化
            let degradation_signals = self.degradation_detector.detect_degradation(&periods, &period_result)?;
            
            // 应用适应性措施
            let adaptations = if !degradation_signals.is_empty() {
                self.adaptation_manager.create_adaptations(&degradation_signals)?
            } else {
                vec![]
            };

            let mut final_period = period_result;
            final_period.degradation_signals = degradation_signals;
            final_period.adaptations_applied = adaptations.clone();

            periods.push(final_period);
            adaptation_history.extend(adaptations);

            // 早期停止检查
            if let Some(max_periods) = self.config.max_periods {
                if periods.len() >= max_periods {
                    log::info!("Reached maximum periods limit: {}", max_periods);
                    break;
                }
            }
        }

        // 3. 计算聚合指标
        let aggregate_metrics = self.calculate_aggregate_metrics(&periods)?;

        // 4. 参数稳定性分析
        let parameter_stability = self.parameter_analyzer.analyze_stability(&periods).await?;

        // 5. 效率分析
        let efficiency_analysis = if self.config.enable_efficiency_analysis {
            self.efficiency_calculator.analyze_efficiency(&periods).await?
        } else {
            EfficiencyAnalysis::default()
        };

        // 6. 退化分析
        let degradation_analysis = if self.config.enable_degradation_detection {
            self.degradation_detector.analyze_degradation_patterns(&periods)?
        } else {
            DegradationAnalysis::default()
        };

        // 7. 适应性历史分析
        let adaptation_history_analysis = self.analyze_adaptation_history(&adaptation_history, &periods)?;

        let total_duration = validation_start.elapsed();

        Ok(WalkForwardResult {
            validation_id,
            config: self.config.clone(),
            periods,
            aggregate_metrics,
            parameter_stability,
            efficiency_analysis,
            degradation_analysis,
            adaptation_history: adaptation_history_analysis,
            validation_timestamp: Utc::now(),
            total_duration_ms: total_duration.as_millis() as u64,
        })
    }

    /// 生成时间分割
    fn generate_time_splits<D>(&self, dataset: &D) -> Result<Vec<TimeSplit>>
    where
        D: WalkForwardDataset + Send + Sync,
    {
        let dataset_start = dataset.start_time();
        let dataset_end = dataset.end_time();
        let mut splits = Vec::new();
        
        let mut current_start = dataset_start;
        let mut period_id = 0;

        while current_start + self.config.initial_train_window + self.config.test_window <= dataset_end {
            let train_start = current_start;
            let train_end = train_start + self.config.initial_train_window;
            let test_start = train_end;
            let test_end = test_start + self.config.test_window;

            // 验证数据充足性
            let train_samples = dataset.count_samples_in_range(train_start, train_end)?;
            let test_samples = dataset.count_samples_in_range(test_start, test_end)?;

            if train_samples < self.config.min_train_samples {
                log::warn!("Insufficient training samples in period {}: {} < {}", 
                    period_id, train_samples, self.config.min_train_samples);
                break;
            }

            if test_samples < self.config.min_test_samples {
                log::warn!("Insufficient test samples in period {}: {} < {}", 
                    period_id, test_samples, self.config.min_test_samples);
                break;
            }

            splits.push(TimeSplit {
                period_id,
                train_start,
                train_end,
                test_start,
                test_end,
                train_samples,
                test_samples,
            });

            current_start = current_start + self.config.step_size;
            period_id += 1;
        }

        if splits.is_empty() {
            return Err(anyhow::anyhow!("No valid time splits generated"));
        }

        log::info!("Generated {} time splits from {} to {}", 
            splits.len(), dataset_start, dataset_end);

        Ok(splits)
    }

    /// 执行单个周期
    async fn execute_period<T, D>(
        &self,
        dataset: &D,
        model_factory: &T,
        split: &TimeSplit,
        period_id: u32,
        historical_periods: &[WalkForwardPeriod],
    ) -> Result<WalkForwardPeriod>
    where
        T: WalkForwardModelFactory + Send + Sync,
        D: WalkForwardDataset + Send + Sync,
    {
        let period_start = std::time::Instant::now();

        // 准备训练和测试数据
        let train_data = dataset.get_data_in_range(split.train_start, split.train_end)?;
        let test_data = dataset.get_data_in_range(split.test_start, split.test_end)?;

        // 创建和优化模型
        let optimization_start = std::time::Instant::now();
        let mut model = model_factory.create_model()?;
        
        // 参数优化（如果启用）
        let optimized_parameters = if self.config.enable_parameter_optimization {
            if period_id % self.config.reoptimization_frequency == 0 || historical_periods.is_empty() {
                model.optimize_parameters(train_data.as_ref()).await?
            } else {
                // 使用上一周期的参数
                if let Some(last_period) = historical_periods.last() {
                    last_period.optimized_parameters.clone()
                } else {
                    HashMap::new()
                }
            }
        } else {
            HashMap::new()
        };

        model.set_parameters(&optimized_parameters)?;
        let optimization_duration = optimization_start.elapsed();

        // 训练模型
        let training_result = model.fit(train_data.as_ref()).await?;

        // 计算样本内指标
        let is_predictions = model.predict(train_data.as_ref()).await?;
        let is_targets = train_data.get_targets()?;
        let in_sample_metrics = self.calculate_in_sample_metrics(&is_predictions, &is_targets)?;

        // 样本外预测和评估
        let prediction_start = std::time::Instant::now();
        let oos_predictions = model.predict(test_data.as_ref()).await?;
        let oos_targets = test_data.get_targets()?;
        let prediction_duration = prediction_start.elapsed();

        let out_of_sample_metrics = self.calculate_out_of_sample_metrics(&oos_predictions, &oos_targets)?;

        // 计算效率比率
        let efficiency_ratio = if in_sample_metrics.return_pct != 0.0 {
            out_of_sample_metrics.return_pct / in_sample_metrics.return_pct
        } else {
            0.0
        };

        // 分析参数变化
        let parameter_changes = if let Some(last_period) = historical_periods.last() {
            self.analyze_parameter_changes(&last_period.optimized_parameters, &optimized_parameters)?
        } else {
            HashMap::new()
        };

        Ok(WalkForwardPeriod {
            period_id,
            train_start: split.train_start,
            train_end: split.train_end,
            test_start: split.test_start,
            test_end: split.test_end,
            train_samples: split.train_samples,
            test_samples: split.test_samples,
            optimization_time_ms: optimization_duration.as_millis() as u64,
            prediction_time_ms: prediction_duration.as_millis() as u64,
            in_sample_metrics,
            out_of_sample_metrics,
            optimized_parameters,
            parameter_changes,
            efficiency_ratio,
            degradation_signals: vec![], // 将在后续填充
            adaptations_applied: vec![], // 将在后续填充
        })
    }

    /// 计算样本内指标
    fn calculate_in_sample_metrics(&self, predictions: &[f64], targets: &[f64]) -> Result<InSampleMetrics> {
        let returns = self.calculate_returns(predictions, targets)?;
        
        Ok(InSampleMetrics {
            return_pct: returns.iter().sum::<f64>(),
            sharpe_ratio: self.calculate_sharpe_ratio(&returns)?,
            max_drawdown: self.calculate_max_drawdown(&returns)?,
            volatility: self.calculate_volatility(&returns)?,
            hit_rate: self.calculate_hit_rate(predictions, targets)?,
            avg_win: self.calculate_avg_win(&returns)?,
            avg_loss: self.calculate_avg_loss(&returns)?,
            profit_factor: self.calculate_profit_factor(&returns)?,
            information_coefficient: self.calculate_ic(predictions, targets)?,
            turnover: self.calculate_turnover(predictions)?,
        })
    }

    /// 计算样本外指标
    fn calculate_out_of_sample_metrics(&self, predictions: &[f64], targets: &[f64]) -> Result<OutOfSampleMetrics> {
        let returns = self.calculate_returns(predictions, targets)?;
        
        Ok(OutOfSampleMetrics {
            return_pct: returns.iter().sum::<f64>(),
            sharpe_ratio: self.calculate_sharpe_ratio(&returns)?,
            max_drawdown: self.calculate_max_drawdown(&returns)?,
            volatility: self.calculate_volatility(&returns)?,
            hit_rate: self.calculate_hit_rate(predictions, targets)?,
            avg_win: self.calculate_avg_win(&returns)?,
            avg_loss: self.calculate_avg_loss(&returns)?,
            profit_factor: self.calculate_profit_factor(&returns)?,
            information_coefficient: self.calculate_ic(predictions, targets)?,
            turnover: self.calculate_turnover(predictions)?,
            slippage_cost: self.estimate_slippage_cost(&returns)?,
            transaction_cost: self.estimate_transaction_cost(&returns)?,
        })
    }

    /// 分析参数变化
    fn analyze_parameter_changes(
        &self, 
        old_params: &HashMap<String, f64>, 
        new_params: &HashMap<String, f64>
    ) -> Result<HashMap<String, ParameterChange>> {
        let mut changes = HashMap::new();

        for (param_name, &new_value) in new_params {
            if let Some(&old_value) = old_params.get(param_name) {
                let change_magnitude = (new_value - old_value).abs();
                let change_significance = if old_value != 0.0 {
                    change_magnitude / old_value.abs()
                } else if new_value != 0.0 {
                    1.0
                } else {
                    0.0
                };

                if change_significance > 0.05 { // 5% threshold
                    changes.insert(param_name.clone(), ParameterChange {
                        parameter_name: param_name.clone(),
                        old_value,
                        new_value,
                        change_magnitude,
                        change_significance,
                        reason: ParameterChangeReason::Optimization,
                    });
                }
            }
        }

        Ok(changes)
    }

    /// 计算聚合指标
    fn calculate_aggregate_metrics(&self, periods: &[WalkForwardPeriod]) -> Result<WalkForwardAggregateMetrics> {
        if periods.is_empty() {
            return Err(anyhow::anyhow!("No periods to analyze"));
        }

        let total_periods = periods.len();
        let successful_periods = periods.iter()
            .filter(|p| p.out_of_sample_metrics.return_pct > 0.0)
            .count();

        let efficiency_ratios: Vec<f64> = periods.iter().map(|p| p.efficiency_ratio).collect();
        let avg_efficiency_ratio = efficiency_ratios.iter().sum::<f64>() / efficiency_ratios.len() as f64;
        let std_efficiency_ratio = self.calculate_std(&efficiency_ratios)?;

        let cumulative_is_return = periods.iter().map(|p| p.in_sample_metrics.return_pct).sum();
        let cumulative_oos_return = periods.iter().map(|p| p.out_of_sample_metrics.return_pct).sum();

        let is_sharpes: Vec<f64> = periods.iter().map(|p| p.in_sample_metrics.sharpe_ratio).collect();
        let oos_sharpes: Vec<f64> = periods.iter().map(|p| p.out_of_sample_metrics.sharpe_ratio).collect();
        
        let avg_is_sharpe = is_sharpes.iter().sum::<f64>() / is_sharpes.len() as f64;
        let avg_oos_sharpe = oos_sharpes.iter().sum::<f64>() / oos_sharpes.len() as f64;

        let parameter_stability_score = self.calculate_parameter_stability_score(periods)?;
        let degradation_frequency = periods.iter()
            .map(|p| p.degradation_signals.len() as f64)
            .sum::<f64>() / periods.len() as f64;

        let total_adaptations: usize = periods.iter().map(|p| p.adaptations_applied.len()).sum();
        let successful_adaptations = periods.iter()
            .flat_map(|p| &p.adaptations_applied)
            .filter(|a| a.result.as_ref().map_or(false, |r| r.success))
            .count();
        
        let adaptation_success_rate = if total_adaptations > 0 {
            successful_adaptations as f64 / total_adaptations as f64
        } else {
            1.0
        };

        let overall_robustness_score = self.calculate_robustness_score(periods)?;

        Ok(WalkForwardAggregateMetrics {
            total_periods,
            successful_periods,
            avg_efficiency_ratio,
            std_efficiency_ratio,
            cumulative_is_return,
            cumulative_oos_return,
            avg_is_sharpe,
            avg_oos_sharpe,
            parameter_stability_score,
            degradation_frequency,
            adaptation_success_rate,
            overall_robustness_score,
        })
    }

    /// 分析适应性历史
    fn analyze_adaptation_history(
        &self, 
        adaptations: &[AdaptationAction],
        periods: &[WalkForwardPeriod],
    ) -> Result<AdaptationHistory> {
        let total_adaptations = adaptations.len();
        
        let mut adaptation_types_frequency = HashMap::new();
        let mut success_rate_by_type = HashMap::new();
        
        for adaptation in adaptations {
            *adaptation_types_frequency.entry(adaptation.action_type.clone()).or_insert(0) += 1;
            
            let success_count = success_rate_by_type.entry(adaptation.action_type.clone()).or_insert((0, 0));
            success_count.1 += 1;
            if adaptation.result.as_ref().map_or(false, |r| r.success) {
                success_count.0 += 1;
            }
        }
        
        // 计算成功率
        let success_rates: HashMap<AdaptationType, f64> = success_rate_by_type.into_iter()
            .map(|(k, (success, total))| (k, if total > 0 { success as f64 / total as f64 } else { 0.0 }))
            .collect();

        // 生成学习曲线
        let learning_curve = self.generate_learning_curve(periods)?;

        // 提取最佳实践
        let best_practices = self.extract_best_practices(adaptations)?;

        Ok(AdaptationHistory {
            total_adaptations,
            adaptation_types_frequency,
            success_rate_by_type: success_rates,
            learning_curve,
            best_practices,
        })
    }

    // 辅助计算方法
    fn calculate_returns(&self, predictions: &[f64], targets: &[f64]) -> Result<Vec<f64>> {
        if predictions.len() != targets.len() {
            return Err(anyhow::anyhow!("Predictions and targets length mismatch"));
        }
        
        Ok(predictions.iter().zip(targets.iter())
            .map(|(pred, target)| pred * target) // 简化收益计算
            .collect())
    }

    fn calculate_sharpe_ratio(&self, returns: &[f64]) -> Result<f64> {
        if returns.is_empty() {
            return Ok(0.0);
        }
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let std_return = self.calculate_std(returns)?;
        
        if std_return > 0.0 {
            Ok(mean_return / std_return * (252.0_f64).sqrt()) // 年化
        } else {
            Ok(0.0)
        }
    }

    fn calculate_max_drawdown(&self, returns: &[f64]) -> Result<f64> {
        if returns.is_empty() {
            return Ok(0.0);
        }
        
        let mut cumulative = 0.0_f64;
        let mut peak = 0.0_f64;
        let mut max_dd = 0.0_f64;
        
        for &ret in returns {
            cumulative += ret;
            peak = peak.max(cumulative);
            let drawdown = (peak - cumulative) / peak.max(1e-8);
            max_dd = max_dd.max(drawdown);
        }
        
        Ok(max_dd)
    }

    fn calculate_volatility(&self, returns: &[f64]) -> Result<f64> {
        self.calculate_std(returns).map(|std| std * (252.0_f64).sqrt())
    }

    fn calculate_hit_rate(&self, predictions: &[f64], targets: &[f64]) -> Result<f64> {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return Ok(0.0);
        }
        
        let hits = predictions.iter().zip(targets.iter())
            .filter(|(pred, target)| (**pred > 0.0) == (**target > 0.0))
            .count();
        
        Ok(hits as f64 / predictions.len() as f64)
    }

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

    fn calculate_avg_win(&self, returns: &[f64]) -> Result<f64> {
        let wins: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).cloned().collect();
        if wins.is_empty() {
            Ok(0.0)
        } else {
            Ok(wins.iter().sum::<f64>() / wins.len() as f64)
        }
    }

    fn calculate_avg_loss(&self, returns: &[f64]) -> Result<f64> {
        let losses: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        if losses.is_empty() {
            Ok(0.0)
        } else {
            Ok(losses.iter().sum::<f64>() / losses.len() as f64)
        }
    }

    fn calculate_profit_factor(&self, returns: &[f64]) -> Result<f64> {
        let total_wins: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
        let total_losses: f64 = returns.iter().filter(|&&r| r < 0.0).sum::<f64>().abs();
        
        if total_losses > 0.0 {
            Ok(total_wins / total_losses)
        } else if total_wins > 0.0 {
            Ok(f64::INFINITY)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_ic(&self, predictions: &[f64], targets: &[f64]) -> Result<f64> {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return Ok(0.0);
        }

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

    fn calculate_turnover(&self, predictions: &[f64]) -> Result<f64> {
        if predictions.len() < 2 {
            return Ok(0.0);
        }
        
        let position_changes: f64 = predictions.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum();
        
        Ok(position_changes / predictions.len() as f64)
    }

    fn estimate_slippage_cost(&self, returns: &[f64]) -> Result<f64> {
        // 简化滑点成本估算
        Ok(returns.iter().map(|r| r.abs() * 0.001).sum::<f64>())
    }

    fn estimate_transaction_cost(&self, returns: &[f64]) -> Result<f64> {
        // 简化交易成本估算
        Ok(returns.len() as f64 * 0.0005)
    }

    fn calculate_parameter_stability_score(&self, periods: &[WalkForwardPeriod]) -> Result<f64> {
        // 简化参数稳定性评分
        let total_changes: f64 = periods.iter()
            .map(|p| p.parameter_changes.len() as f64)
            .sum();
        
        let avg_changes = total_changes / periods.len() as f64;
        Ok((1.0 - avg_changes / 10.0).max(0.0)) // 归一化到 [0, 1]
    }

    fn calculate_robustness_score(&self, periods: &[WalkForwardPeriod]) -> Result<f64> {
        // 综合鲁棒性评分
        let efficiency_consistency = 1.0 - self.calculate_coefficient_of_variation(
            &periods.iter().map(|p| p.efficiency_ratio).collect::<Vec<_>>()
        )?;
        
        let performance_consistency = 1.0 - self.calculate_coefficient_of_variation(
            &periods.iter().map(|p| p.out_of_sample_metrics.return_pct).collect::<Vec<_>>()
        )?;
        
        Ok((efficiency_consistency + performance_consistency) / 2.0)
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

    fn generate_learning_curve(&self, periods: &[WalkForwardPeriod]) -> Result<Vec<LearningPoint>> {
        let mut learning_curve = Vec::new();
        let mut cumulative_learning = 0.0;
        
        for (i, period) in periods.iter().enumerate() {
            cumulative_learning += period.adaptations_applied.len() as f64 * 0.1;
            
            let adaptation_effectiveness = period.adaptations_applied.iter()
                .filter_map(|a| a.result.as_ref())
                .map(|r| if r.success { 1.0 } else { 0.0 })
                .sum::<f64>() / period.adaptations_applied.len().max(1) as f64;
            
            learning_curve.push(LearningPoint {
                period: i as u32,
                cumulative_learning,
                adaptation_effectiveness,
                knowledge_gained: format!("Period {} insights", i),
            });
        }
        
        Ok(learning_curve)
    }

    fn extract_best_practices(&self, adaptations: &[AdaptationAction]) -> Result<Vec<String>> {
        let mut practices = Vec::new();
        
        // 分析最成功的适应类型
        let mut success_by_type = HashMap::new();
        for adaptation in adaptations {
            if let Some(result) = &adaptation.result {
                let entry = success_by_type.entry(adaptation.action_type.clone()).or_insert((0, 0));
                entry.1 += 1;
                if result.success {
                    entry.0 += 1;
                }
            }
        }
        
        for (adaptation_type, (success, total)) in success_by_type {
            if total > 0 && (success as f64 / total as f64) > 0.7 {
                practices.push(format!("{:?} adaptations show high success rate", adaptation_type));
            }
        }
        
        if practices.is_empty() {
            practices.push("Continue monitoring and adapting based on performance".to_string());
        }
        
        Ok(practices)
    }
}

// 支持结构和接口定义
#[derive(Debug, Clone)]
pub struct TimeSplit {
    pub period_id: u32,
    pub train_start: DateTime<Utc>,
    pub train_end: DateTime<Utc>,
    pub test_start: DateTime<Utc>,
    pub test_end: DateTime<Utc>,
    pub train_samples: usize,
    pub test_samples: usize,
}

pub trait WalkForwardDataset: Send + Sync {
    fn start_time(&self) -> DateTime<Utc>;
    fn end_time(&self) -> DateTime<Utc>;
    fn count_samples_in_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<usize>;
    fn get_data_in_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<Box<dyn WalkForwardData>>;
}

pub trait WalkForwardData: Send + Sync {
    fn get_features(&self) -> Result<Vec<Vec<f64>>>;
    fn get_targets(&self) -> Result<Vec<f64>>;
    fn len(&self) -> usize;
}

pub trait WalkForwardModelFactory: Send + Sync {
    fn create_model(&self) -> Result<Box<dyn WalkForwardModel>>;
}

#[async_trait]
pub trait WalkForwardModel: Send + Sync {
    async fn optimize_parameters(&self, data: &dyn WalkForwardData) -> Result<HashMap<String, f64>>;
    fn set_parameters(&self, params: &HashMap<String, f64>) -> Result<()>;
    async fn fit(&self, data: &dyn WalkForwardData) -> Result<TrainingResult>;
    async fn predict(&self, data: &dyn WalkForwardData) -> Result<Vec<f64>>;
}

#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub training_score: f64,
    pub iterations: usize,
    pub convergence_achieved: bool,
}

// 组件实现
impl ParameterAnalyzer {
    pub fn new() -> Result<Self> {
        Ok(Self {
            stability_calculator: StabilityCalculator::new(10, 0.1),
            trend_analyzer: TrendAnalyzer::new(5, 0.05),
            correlation_analyzer: CorrelationAnalyzer::new(0.3),
        })
    }

    pub async fn analyze_stability(&self, periods: &[WalkForwardPeriod]) -> Result<ParameterStabilityAnalysis> {
        // 简化参数稳定性分析
        let mut parameter_statistics = HashMap::new();
        let stability_trends = HashMap::new();
        let correlation_matrix = HashMap::new();
        let regime_sensitivity = HashMap::new();
        
        // 计算整体稳定性评分
        let overall_stability_score = periods.iter()
            .map(|p| 1.0 - p.parameter_changes.len() as f64 / 10.0)
            .sum::<f64>() / periods.len() as f64;

        Ok(ParameterStabilityAnalysis {
            parameter_statistics,
            stability_trends,
            correlation_matrix,
            regime_sensitivity,
            overall_stability_score,
        })
    }
}

impl StabilityCalculator {
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self { window_size, stability_threshold: threshold }
    }
}

impl TrendAnalyzer {
    pub fn new(min_periods: usize, significance_threshold: f64) -> Self {
        Self { min_periods, significance_threshold }
    }
}

impl CorrelationAnalyzer {
    pub fn new(min_correlation: f64) -> Self {
        Self { min_correlation }
    }
}

impl EfficiencyCalculator {
    pub fn new() -> Result<Self> {
        let mut cost_models: HashMap<String, Box<dyn CostModel>> = HashMap::new();
        cost_models.insert("optimization".to_string(), Box::new(OptimizationCostModel));
        
        Ok(Self {
            efficiency_metrics: vec!["efficiency_ratio".to_string(), "net_efficiency".to_string()],
            cost_models,
        })
    }

    pub async fn analyze_efficiency(&self, periods: &[WalkForwardPeriod]) -> Result<EfficiencyAnalysis> {
        // 简化效率分析
        let period_efficiencies: Vec<PeriodEfficiency> = periods.iter().map(|p| {
            PeriodEfficiency {
                period_id: p.period_id,
                efficiency_ratio: p.efficiency_ratio,
                is_performance: p.in_sample_metrics.return_pct,
                oos_performance: p.out_of_sample_metrics.return_pct,
                optimization_cost: p.optimization_time_ms as f64 / 1000.0,
                net_efficiency: p.efficiency_ratio - 0.01, // 简化扣除成本
            }
        }).collect();

        Ok(EfficiencyAnalysis {
            period_efficiencies,
            efficiency_trends: EfficiencyTrends::default(),
            efficiency_factors: HashMap::new(),
            optimization_effectiveness: OptimizationEffectiveness::default(),
            cost_benefit_analysis: CostBenefitAnalysis::default(),
        })
    }
}

#[derive(Debug)]
pub struct OptimizationCostModel;

impl CostModel for OptimizationCostModel {
    fn calculate_cost(&self, period: &WalkForwardPeriod) -> Result<f64> {
        Ok(period.optimization_time_ms as f64 / 1000.0 * 0.01) // 简化成本计算
    }
}

impl DegradationDetector {
    pub fn new() -> Result<Self> {
        let mut detectors: HashMap<DegradationType, Box<dyn DegradationMetric>> = HashMap::new();
        detectors.insert(DegradationType::PerformanceDecline, Box::new(PerformanceDeclineDetector));
        
        let mut alert_thresholds = HashMap::new();
        alert_thresholds.insert(DegradationType::PerformanceDecline, 0.5);
        
        Ok(Self { detectors, alert_thresholds })
    }

    pub fn detect_degradation(
        &self, 
        historical_periods: &[WalkForwardPeriod],
        current_period: &WalkForwardPeriod,
    ) -> Result<Vec<DegradationSignal>> {
        let mut signals = Vec::new();
        
        // 检测性能下降
        if current_period.efficiency_ratio < 0.5 {
            signals.push(DegradationSignal {
                signal_type: DegradationType::PerformanceDecline,
                severity: DegradationSeverity::Medium,
                description: "Efficiency ratio below threshold".to_string(),
                metric_value: current_period.efficiency_ratio,
                threshold: 0.5,
                detection_timestamp: Utc::now(),
                recommended_actions: vec!["Consider parameter reoptimization".to_string()],
            });
        }
        
        Ok(signals)
    }

    pub fn analyze_degradation_patterns(&self, periods: &[WalkForwardPeriod]) -> Result<DegradationAnalysis> {
        // 简化退化模式分析
        let degradation_frequency = periods.iter()
            .map(|p| p.degradation_signals.len() as f64)
            .sum::<f64>() / periods.len() as f64;
        
        Ok(DegradationAnalysis {
            degradation_frequency,
            degradation_patterns: vec![],
            early_warning_indicators: vec![],
            recovery_strategies: HashMap::new(),
            prevention_recommendations: vec![
                "Regular parameter reoptimization".to_string(),
                "Market regime monitoring".to_string(),
            ],
        })
    }
}

#[derive(Debug)]
pub struct PerformanceDeclineDetector;

impl DegradationMetric for PerformanceDeclineDetector {
    fn name(&self) -> &str { "performance_decline" }
    
    fn calculate(&self, periods: &[WalkForwardPeriod]) -> Result<f64> {
        if periods.len() < 2 {
            return Ok(0.0);
        }
        
        let recent_performance = periods.last().unwrap().efficiency_ratio;
        let avg_performance = periods.iter()
            .map(|p| p.efficiency_ratio)
            .sum::<f64>() / periods.len() as f64;
        
        Ok((avg_performance - recent_performance) / avg_performance.max(1e-8))
    }
    
    fn get_threshold(&self) -> f64 { 0.2 }
}

impl AdaptationManager {
    pub fn new() -> Result<Self> {
        let mut adaptation_strategies: HashMap<AdaptationType, Box<dyn AdaptationStrategy>> = HashMap::new();
        adaptation_strategies.insert(
            AdaptationType::ParameterReoptimization, 
            Box::new(ParameterReoptimizationStrategy)
        );
        
        Ok(Self {
            adaptation_strategies,
            decision_engine: AdaptationDecisionEngine::new(),
        })
    }

    pub fn create_adaptations(&self, signals: &[DegradationSignal]) -> Result<Vec<AdaptationAction>> {
        let mut actions = Vec::new();
        
        for signal in signals {
            for strategy in self.adaptation_strategies.values() {
                if strategy.can_handle(signal) {
                    let action = strategy.create_action(signal)?;
                    actions.push(action);
                }
            }
        }
        
        Ok(actions)
    }
}

impl AdaptationDecisionEngine {
    pub fn new() -> Self {
        Self {
            decision_rules: vec![],
            risk_tolerance: 0.1,
        }
    }
}

#[derive(Debug)]
pub struct ParameterReoptimizationStrategy;

impl AdaptationStrategy for ParameterReoptimizationStrategy {
    fn name(&self) -> &str { "parameter_reoptimization" }
    
    fn can_handle(&self, degradation: &DegradationSignal) -> bool {
        matches!(degradation.signal_type, DegradationType::PerformanceDecline)
    }
    
    fn create_action(&self, degradation: &DegradationSignal) -> Result<AdaptationAction> {
        Ok(AdaptationAction {
            action_type: AdaptationType::ParameterReoptimization,
            description: "Reoptimize model parameters due to performance decline".to_string(),
            parameters_affected: vec!["all".to_string()],
            expected_impact: 0.1,
            success_probability: 0.7,
            implementation_timestamp: Utc::now(),
            result: None,
        })
    }
    
    fn estimate_success_probability(&self, _action: &AdaptationAction) -> f64 { 0.7 }
}

// 默认实现
impl Default for WalkForwardConfig {
    fn default() -> Self {
        Self {
            initial_train_window: Duration::days(365),
            test_window: Duration::days(30),
            step_size: Duration::days(30),
            min_train_samples: 1000,
            min_test_samples: 100,
            max_periods: Some(24), // 2年月度验证
            enable_parameter_optimization: true,
            enable_efficiency_analysis: true,
            enable_degradation_detection: true,
            reoptimization_frequency: 3,
            stability_threshold: 0.1,
            efficiency_threshold: 0.5,
        }
    }
}

impl Default for EfficiencyAnalysis {
    fn default() -> Self {
        Self {
            period_efficiencies: vec![],
            efficiency_trends: EfficiencyTrends::default(),
            efficiency_factors: HashMap::new(),
            optimization_effectiveness: OptimizationEffectiveness::default(),
            cost_benefit_analysis: CostBenefitAnalysis::default(),
        }
    }
}

impl Default for EfficiencyTrends {
    fn default() -> Self {
        Self {
            overall_trend: TrendDirection::Stable,
            trend_strength: 0.0,
            recent_efficiency: 0.0,
            efficiency_volatility: 0.0,
            declining_periods: vec![],
        }
    }
}

impl Default for OptimizationEffectiveness {
    fn default() -> Self {
        Self {
            avg_improvement: 0.0,
            success_rate: 0.0,
            diminishing_returns_detected: false,
            optimal_reopt_frequency: 3,
        }
    }
}

impl Default for CostBenefitAnalysis {
    fn default() -> Self {
        Self {
            total_optimization_cost: 0.0,
            total_performance_gain: 0.0,
            net_benefit: 0.0,
            roi: 0.0,
            break_even_periods: 0,
        }
    }
}

impl Default for DegradationAnalysis {
    fn default() -> Self {
        Self {
            degradation_frequency: 0.0,
            degradation_patterns: vec![],
            early_warning_indicators: vec![],
            recovery_strategies: HashMap::new(),
            prevention_recommendations: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_walk_forward_validator_creation() {
        let config = WalkForwardConfig::default();
        let validator = WalkForwardValidator::new(config);
        assert!(validator.is_ok());
    }

    #[test]
    fn test_degradation_severity_ordering() {
        assert!((DegradationSeverity::Critical as u8) > (DegradationSeverity::High as u8));
        assert!((DegradationSeverity::High as u8) > (DegradationSeverity::Medium as u8));
        assert!((DegradationSeverity::Medium as u8) > (DegradationSeverity::Low as u8));
    }

    #[test]
    fn test_efficiency_calculation() {
        let periods = vec![
            create_test_period(0, 0.8, 0.6),
            create_test_period(1, 0.9, 0.7),
            create_test_period(2, 0.7, 0.5),
        ];
        
        let config = WalkForwardConfig::default();
        let validator = WalkForwardValidator::new(config).unwrap();
        let metrics = validator.calculate_aggregate_metrics(&periods).unwrap();
        
        assert!(metrics.avg_efficiency_ratio > 0.0);
    }

    fn create_test_period(id: u32, is_return: f64, oos_return: f64) -> WalkForwardPeriod {
        WalkForwardPeriod {
            period_id: id,
            train_start: Utc::now(),
            train_end: Utc::now(),
            test_start: Utc::now(),
            test_end: Utc::now(),
            train_samples: 1000,
            test_samples: 100,
            optimization_time_ms: 1000,
            prediction_time_ms: 100,
            in_sample_metrics: InSampleMetrics {
                return_pct: is_return,
                sharpe_ratio: 1.5,
                max_drawdown: 0.05,
                volatility: 0.15,
                hit_rate: 0.6,
                avg_win: 0.02,
                avg_loss: -0.015,
                profit_factor: 1.3,
                information_coefficient: 0.05,
                turnover: 2.0,
            },
            out_of_sample_metrics: OutOfSampleMetrics {
                return_pct: oos_return,
                sharpe_ratio: 1.2,
                max_drawdown: 0.08,
                volatility: 0.18,
                hit_rate: 0.55,
                avg_win: 0.018,
                avg_loss: -0.016,
                profit_factor: 1.1,
                information_coefficient: 0.03,
                turnover: 2.2,
                slippage_cost: 0.001,
                transaction_cost: 0.0005,
            },
            optimized_parameters: HashMap::new(),
            parameter_changes: HashMap::new(),
            efficiency_ratio: oos_return / is_return,
            degradation_signals: vec![],
            adaptations_applied: vec![],
        }
    }
}