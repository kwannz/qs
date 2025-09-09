use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{Duration, Instant};
use tracing::{debug, info};
use uuid::Uuid;

/// A/B测试实验配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub id: String,
    pub name: String,
    pub description: String,
    pub hypothesis: String,
    pub owner: String,
    pub status: ExperimentStatus,
    
    // 实验设计
    pub traffic_allocation: TrafficAllocation,
    pub variants: Vec<ExperimentVariant>,
    pub control_group: String,
    pub target_metrics: Vec<TargetMetric>,
    
    // 时间配置
    pub start_time: i64,
    pub end_time: i64,
    pub duration_days: u32,
    pub ramp_up_duration: Duration,
    pub ramp_down_duration: Duration,
    
    // 统计配置
    pub statistical_config: StatisticalConfig,
    pub significance_level: f64,
    pub minimum_detectable_effect: f64,
    pub power: f64,
    
    // 安全配置
    pub safety_rules: Vec<SafetyRule>,
    pub circuit_breaker: CircuitBreakerConfig,
    pub rollback_conditions: Vec<RollbackCondition>,
    
    // 监控配置
    pub monitoring_config: MonitoringConfig,
    pub alert_conditions: Vec<AlertCondition>,
}

/// 实验状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExperimentStatus {
    Draft,          // 草稿
    Scheduled,      // 已调度
    Running,        // 运行中
    Paused,         // 暂停
    Completed,      // 完成
    Terminated,     // 终止
    RollingBack,    // 回滚中
    RolledBack,     // 已回滚
}

/// 流量分配
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficAllocation {
    pub allocation_strategy: AllocationStrategy,
    pub variant_weights: HashMap<String, f64>,
    pub inclusion_criteria: Vec<InclusionCriterion>,
    pub exclusion_criteria: Vec<ExclusionCriterion>,
    pub sticky_assignment: bool, // 用户粘性分配
}

/// 分配策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    Random,                    // 随机分配
    DeterministicHashing,      // 确定性哈希
    GeographicSplit,           // 地理分割
    TimeBased,                 // 基于时间
    UserAttribute,             // 基于用户属性
    Stratified,                // 分层抽样
}

/// 实验变体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentVariant {
    pub id: String,
    pub name: String,
    pub description: String,
    pub is_control: bool,
    pub traffic_percentage: f64,
    pub config_overrides: HashMap<String, serde_json::Value>,
    pub feature_flags: HashMap<String, bool>,
}

/// 目标指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetMetric {
    pub name: String,
    pub metric_type: MetricType,
    pub aggregation: MetricAggregation,
    pub success_criteria: SuccessCriteria,
    pub is_primary: bool,
    pub weight: f64,
    pub guardrail_bounds: Option<GuardrailBounds>,
}

/// 指标类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    Conversion,        // 转化率
    Revenue,          // 收入
    Engagement,       // 参与度
    Performance,      // 性能指标
    Quality,          // 质量指标
    Custom(String),   // 自定义指标
}

/// 指标聚合方式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricAggregation {
    Sum,
    Average,
    Median,
    P95,
    P99,
    Count,
    UniqueCount,
    Rate,
}

/// 成功标准
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriteria {
    pub improvement_type: ImprovementType,
    pub target_improvement: f64,
    pub minimum_improvement: f64,
    pub statistical_significance_required: bool,
}

/// 改进类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImprovementType {
    Increase,      // 增加
    Decrease,      // 减少
    NoChange,      // 无变化（非劣性测试）
}

/// 护栏边界
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailBounds {
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
    pub alert_threshold: f64,
    pub emergency_threshold: f64,
}

/// 包含标准
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InclusionCriterion {
    pub field: String,
    pub operator: ComparisonOperator,
    pub value: serde_json::Value,
    pub weight: f64,
}

/// 排除标准
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExclusionCriterion {
    pub field: String,
    pub operator: ComparisonOperator,
    pub value: serde_json::Value,
    pub reason: String,
}

/// 比较操作符
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    In,
    NotIn,
    Contains,
    StartsWith,
    EndsWith,
}

/// 统计配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalConfig {
    pub test_type: StatisticalTest,
    pub multiple_testing_correction: MultipleTesting,
    pub sequential_testing_enabled: bool,
    pub bayesian_analysis_enabled: bool,
    pub confidence_interval_type: ConfidenceIntervalType,
    pub bootstrap_samples: usize,
}

/// 统计测试类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticalTest {
    TTest,             // t检验
    ChiSquareTest,     // 卡方检验
    FisherExactTest,   // Fisher精确检验
    MannWhitneyU,      // Mann-Whitney U检验
    WelchTTest,        // Welch t检验
    BayesianTest,      // 贝叶斯检验
}

/// 多重检验校正
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultipleTesting {
    None,
    Bonferroni,
    BenjaminiHochberg,
    Holm,
    Sidak,
}

/// 置信区间类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfidenceIntervalType {
    Normal,
    Bootstrap,
    Bayesian,
}

/// 安全规则
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyRule {
    pub name: String,
    pub condition: SafetyCondition,
    pub action: SafetyAction,
    pub severity: AlertSeverity,
    pub cooldown_period: Duration,
}

/// 安全条件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyCondition {
    pub metric: String,
    pub operator: ComparisonOperator,
    pub threshold: f64,
    pub window_size: Duration,
    pub minimum_samples: usize,
}

/// 安全操作
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyAction {
    Alert,             // 仅告警
    PauseExperiment,   // 暂停实验
    RollbackTraffic,   // 回滚流量
    TerminateExperiment, // 终止实验
}

/// 告警严重程度
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// 熔断器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub enabled: bool,
    pub failure_threshold: f64,
    pub recovery_threshold: f64,
    pub check_interval: Duration,
    pub half_open_max_requests: usize,
}

/// 回滚条件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackCondition {
    pub name: String,
    pub condition: SafetyCondition,
    pub automatic: bool,
    pub priority: u32,
}

/// 监控配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub metrics_collection_interval: Duration,
    pub real_time_monitoring: bool,
    pub dashboard_refresh_interval: Duration,
    pub data_quality_checks: Vec<DataQualityCheck>,
}

/// 数据质量检查
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityCheck {
    pub name: String,
    pub check_type: QualityCheckType,
    pub threshold: f64,
    pub action: DataQualityAction,
}

/// 质量检查类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityCheckType {
    DataCompleteness,   // 数据完整性
    DataAccuracy,       // 数据准确性
    DataConsistency,    // 数据一致性
    DataFreshness,      // 数据时效性
    OutlierDetection,   // 异常值检测
}

/// 数据质量操作
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataQualityAction {
    Log,
    Alert,
    ExcludeData,
    PauseExperiment,
}

/// 告警条件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertCondition {
    pub name: String,
    pub condition: SafetyCondition,
    pub severity: AlertSeverity,
    pub notification_channels: Vec<String>,
}

/// 实验结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    pub experiment_id: String,
    pub status: ExperimentStatus,
    pub start_time: i64,
    pub end_time: Option<i64>,
    pub total_participants: u64,
    pub variant_results: HashMap<String, VariantResult>,
    pub statistical_analysis: StatisticalAnalysis,
    pub recommendations: Vec<Recommendation>,
    pub confidence_level: f64,
}

/// 变体结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantResult {
    pub variant_id: String,
    pub participants: u64,
    pub metric_values: HashMap<String, MetricResult>,
    pub conversion_rate: f64,
    pub statistical_significance: bool,
    pub confidence_interval: ConfidenceInterval,
}

/// 指标结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricResult {
    pub value: f64,
    pub sample_size: u64,
    pub variance: f64,
    pub standard_error: f64,
    pub improvement_over_control: f64,
    pub p_value: f64,
}

/// 置信区间
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub confidence_level: f64,
}

/// 统计分析
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysis {
    pub overall_significance: bool,
    pub winning_variant: Option<String>,
    pub effect_size: f64,
    pub power_achieved: f64,
    pub sample_size_adequacy: bool,
    pub multiple_testing_correction_applied: bool,
    pub bayesian_analysis: Option<BayesianAnalysis>,
}

/// 贝叶斯分析
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianAnalysis {
    pub probability_of_superiority: HashMap<String, f64>,
    pub expected_loss: HashMap<String, f64>,
    pub credible_intervals: HashMap<String, ConfidenceInterval>,
    pub posterior_distributions: HashMap<String, Vec<f64>>,
}

/// 实验建议
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub recommendation_type: RecommendationType,
    pub confidence: f64,
    pub reason: String,
    pub suggested_action: String,
    pub risk_assessment: RiskAssessment,
}

/// 建议类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    LaunchVariant,      // 启动变体
    ContinueTesting,    // 继续测试
    StopTesting,        // 停止测试
    ExtendTesting,      // 延长测试
    ModifyExperiment,   // 修改实验
}

/// 风险评估
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub risk_level: RiskLevel,
    pub potential_downside: f64,
    pub potential_upside: f64,
    pub probability_of_success: f64,
    pub risk_factors: Vec<String>,
}

/// 风险等级
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// 实验管理器
pub struct ExperimentManager {
    // 实验存储
    active_experiments: Arc<RwLock<HashMap<String, ExperimentConfig>>>,
    experiment_results: Arc<RwLock<HashMap<String, ExperimentResult>>>,
    
    // 分配器和统计引擎
    traffic_splitter: Arc<dyn TrafficSplitter + Send + Sync>,
    statistical_engine: Arc<dyn StatisticalEngine + Send + Sync>,
    
    // 监控和告警
    safety_monitor: Arc<SafetyMonitor>,
    alert_manager: Arc<AlertManager>,
    
    // 状态管理
    experiment_states: Arc<RwLock<HashMap<String, ExperimentState>>>,
}

/// 实验状态
#[derive(Debug, Clone)]
struct ExperimentState {
    pub current_status: ExperimentStatus,
    pub last_status_change: Instant,
    pub participant_count: u64,
    pub variant_assignments: HashMap<String, u64>,
    pub safety_violations: Vec<SafetyViolation>,
    pub last_analysis: Option<Instant>,
}

/// 安全违规
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyViolation {
    pub rule_name: String,
    pub violation_time: i64,
    pub metric_value: f64,
    pub threshold: f64,
    pub severity: AlertSeverity,
    pub action_taken: SafetyAction,
}

/// 流量分配器特征
pub trait TrafficSplitter {
    fn assign_variant(&self, user_id: &str, experiment: &ExperimentConfig) -> Result<String>;
    fn get_assignment(&self, user_id: &str, experiment_id: &str) -> Result<Option<String>>;
}

/// 统计引擎特征
pub trait StatisticalEngine {
    fn analyze_experiment(&self, experiment: &ExperimentConfig, data: &ExperimentData) -> Result<StatisticalAnalysis>;
    fn calculate_sample_size(&self, config: &StatisticalConfig, effect_size: f64, power: f64, alpha: f64) -> Result<usize>;
    fn perform_significance_test(&self, control_data: &[f64], treatment_data: &[f64]) -> Result<SignificanceTestResult>;
}

/// 实验数据
#[derive(Debug, Clone)]
pub struct ExperimentData {
    pub participants: HashMap<String, Vec<ParticipantData>>,
    pub metrics: HashMap<String, Vec<MetricObservation>>,
    pub time_series: HashMap<String, Vec<TimeSeriesPoint>>,
}

/// 参与者数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantData {
    pub user_id: String,
    pub variant: String,
    pub assignment_time: i64,
    pub attributes: HashMap<String, serde_json::Value>,
}

/// 指标观察
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricObservation {
    pub user_id: String,
    pub variant: String,
    pub metric_name: String,
    pub value: f64,
    pub timestamp: i64,
}

/// 时间序列点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    pub timestamp: i64,
    pub variant: String,
    pub metric_name: String,
    pub value: f64,
}

/// 显著性测试结果
#[derive(Debug, Clone)]
pub struct SignificanceTestResult {
    pub p_value: f64,
    pub test_statistic: f64,
    pub effect_size: f64,
    pub confidence_interval: ConfidenceInterval,
    pub is_significant: bool,
}

impl ExperimentManager {
    pub fn new(
        traffic_splitter: Arc<dyn TrafficSplitter + Send + Sync>,
        statistical_engine: Arc<dyn StatisticalEngine + Send + Sync>,
    ) -> Self {
        Self {
            active_experiments: Arc::new(RwLock::new(HashMap::new())),
            experiment_results: Arc::new(RwLock::new(HashMap::new())),
            traffic_splitter,
            statistical_engine,
            safety_monitor: Arc::new(SafetyMonitor::new()),
            alert_manager: Arc::new(AlertManager::new()),
            experiment_states: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 创建新实验
    pub async fn create_experiment(&self, mut config: ExperimentConfig) -> Result<String> {
        // 验证实验配置
        self.validate_experiment_config(&config).await?;
        
        // 生成实验ID（如果未提供）
        if config.id.is_empty() {
            config.id = Uuid::new_v4().to_string();
        }
        
        // 初始化实验状态
        let initial_state = ExperimentState {
            current_status: ExperimentStatus::Draft,
            last_status_change: Instant::now(),
            participant_count: 0,
            variant_assignments: HashMap::new(),
            safety_violations: Vec::new(),
            last_analysis: None,
        };
        
        // 存储实验
        {
            let mut experiments = self.active_experiments.write().await;
            experiments.insert(config.id.clone(), config.clone());
        }
        
        {
            let mut states = self.experiment_states.write().await;
            states.insert(config.id.clone(), initial_state);
        }
        
        info!("Created experiment: {} ({})", config.name, config.id);
        Ok(config.id)
    }

    /// 启动实验
    pub async fn start_experiment(&self, experiment_id: &str) -> Result<()> {
        let experiment = {
            let experiments = self.active_experiments.read().await;
            experiments.get(experiment_id)
                .context("Experiment not found")?
                .clone()
        };
        
        // 验证可以启动
        self.validate_experiment_start(&experiment).await?;
        
        // 更新状态
        self.update_experiment_status(experiment_id, ExperimentStatus::Running).await?;
        
        // 启动监控
        self.start_experiment_monitoring(experiment_id).await?;
        
        info!("Started experiment: {}", experiment_id);
        Ok(())
    }

    /// 停止实验
    pub async fn stop_experiment(&self, experiment_id: &str, reason: Option<String>) -> Result<()> {
        // 更新状态
        self.update_experiment_status(experiment_id, ExperimentStatus::Completed).await?;
        
        // 执行最终分析
        self.perform_final_analysis(experiment_id).await?;
        
        // 停止监控
        self.stop_experiment_monitoring(experiment_id).await?;
        
        info!("Stopped experiment: {} (reason: {:?})", experiment_id, reason);
        Ok(())
    }

    /// 分配用户到实验变体
    pub async fn assign_user(&self, user_id: &str, experiment_id: &str) -> Result<String> {
        let experiment = {
            let experiments = self.active_experiments.read().await;
            experiments.get(experiment_id)
                .context("Experiment not found")?
                .clone()
        };
        
        // 检查实验是否运行中
        if experiment.status != ExperimentStatus::Running {
            return Err(anyhow::anyhow!("Experiment is not running"));
        }
        
        // 检查用户是否符合条件
        if !self.check_user_eligibility(user_id, &experiment).await? {
            return Err(anyhow::anyhow!("User not eligible for experiment"));
        }
        
        // 分配变体
        let variant = self.traffic_splitter.assign_variant(user_id, &experiment)?;
        
        // 更新统计信息
        self.update_assignment_stats(experiment_id, &variant).await?;
        
        debug!("Assigned user {} to variant {} in experiment {}", user_id, variant, experiment_id);
        Ok(variant)
    }

    /// 记录指标观察
    pub async fn record_metric(&self, observation: MetricObservation) -> Result<()> {
        // 验证指标观察
        self.validate_metric_observation(&observation).await?;
        
        // 存储指标数据
        // 实际实现中会存储到数据库或时间序列数据库
        
        // 触发实时分析（如果启用）
        self.trigger_real_time_analysis(&observation).await?;
        
        debug!("Recorded metric observation: {} = {} for user {} in variant {}", 
               observation.metric_name, observation.value, observation.user_id, observation.variant);
        
        Ok(())
    }

    /// 获取实验结果
    pub async fn get_experiment_results(&self, experiment_id: &str) -> Result<ExperimentResult> {
        let results = self.experiment_results.read().await;
        Ok(results.get(experiment_id)
            .context("Experiment results not found")?
            .clone())
    }

    /// 列出活跃实验
    pub async fn list_active_experiments(&self) -> Result<Vec<ExperimentConfig>> {
        let experiments = self.active_experiments.read().await;
        Ok(experiments.values().cloned().collect())
    }

    // 私有辅助方法
    async fn validate_experiment_config(&self, config: &ExperimentConfig) -> Result<()> {
        // 验证变体配置
        if config.variants.is_empty() {
            return Err(anyhow::anyhow!("At least one variant is required"));
        }
        
        // 验证流量分配总和为100%
        let total_traffic: f64 = config.traffic_allocation.variant_weights.values().sum();
        if (total_traffic - 1.0).abs() > 0.01 {
            return Err(anyhow::anyhow!("Traffic allocation must sum to 100%"));
        }
        
        // 验证目标指标
        if config.target_metrics.is_empty() {
            return Err(anyhow::anyhow!("At least one target metric is required"));
        }
        
        Ok(())
    }

    async fn validate_experiment_start(&self, _experiment: &ExperimentConfig) -> Result<()> {
        // 检查依赖项、资源等
        Ok(())
    }

    async fn update_experiment_status(&self, experiment_id: &str, status: ExperimentStatus) -> Result<()> {
        {
            let mut experiments = self.active_experiments.write().await;
            if let Some(experiment) = experiments.get_mut(experiment_id) {
                experiment.status = status.clone();
            }
        }
        
        {
            let mut states = self.experiment_states.write().await;
            if let Some(state) = states.get_mut(experiment_id) {
                state.current_status = status;
                state.last_status_change = Instant::now();
            }
        }
        
        Ok(())
    }

    async fn start_experiment_monitoring(&self, _experiment_id: &str) -> Result<()> {
        // 启动安全监控、性能监控等
        Ok(())
    }

    async fn stop_experiment_monitoring(&self, _experiment_id: &str) -> Result<()> {
        // 停止监控
        Ok(())
    }

    async fn perform_final_analysis(&self, _experiment_id: &str) -> Result<()> {
        // 执行最终统计分析
        Ok(())
    }

    async fn check_user_eligibility(&self, _user_id: &str, _experiment: &ExperimentConfig) -> Result<bool> {
        // 检查包含/排除条件
        Ok(true)
    }

    async fn update_assignment_stats(&self, experiment_id: &str, variant: &str) -> Result<()> {
        let mut states = self.experiment_states.write().await;
        if let Some(state) = states.get_mut(experiment_id) {
            state.participant_count += 1;
            *state.variant_assignments.entry(variant.to_string()).or_insert(0) += 1;
        }
        Ok(())
    }

    async fn validate_metric_observation(&self, _observation: &MetricObservation) -> Result<()> {
        // 验证指标观察的有效性
        Ok(())
    }

    async fn trigger_real_time_analysis(&self, _observation: &MetricObservation) -> Result<()> {
        // 触发实时分析
        Ok(())
    }
}

/// 安全监控器
#[derive(Debug)]
pub struct SafetyMonitor {
    // 安全规则和监控逻辑
}

impl SafetyMonitor {
    pub fn new() -> Self {
        Self {}
    }
}

/// 告警管理器
#[derive(Debug)]
pub struct AlertManager {
    // 告警管理逻辑
}

impl AlertManager {
    pub fn new() -> Self {
        Self {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_experiment_creation() {
        let traffic_splitter = Arc::new(MockTrafficSplitter);
        let statistical_engine = Arc::new(MockStatisticalEngine);
        let manager = ExperimentManager::new(traffic_splitter, statistical_engine);
        
        let config = ExperimentConfig {
            id: "test_experiment".to_string(),
            name: "Test Experiment".to_string(),
            description: "Test experiment description".to_string(),
            hypothesis: "Test hypothesis".to_string(),
            owner: "test_owner".to_string(),
            status: ExperimentStatus::Draft,
            traffic_allocation: TrafficAllocation {
                allocation_strategy: AllocationStrategy::Random,
                variant_weights: [("control".to_string(), 0.5), ("treatment".to_string(), 0.5)]
                    .into_iter().collect(),
                inclusion_criteria: Vec::new(),
                exclusion_criteria: Vec::new(),
                sticky_assignment: true,
            },
            variants: vec![
                ExperimentVariant {
                    id: "control".to_string(),
                    name: "Control".to_string(),
                    description: "Control variant".to_string(),
                    is_control: true,
                    traffic_percentage: 50.0,
                    config_overrides: HashMap::new(),
                    feature_flags: HashMap::new(),
                },
            ],
            control_group: "control".to_string(),
            target_metrics: vec![
                TargetMetric {
                    name: "conversion_rate".to_string(),
                    metric_type: MetricType::Conversion,
                    aggregation: MetricAggregation::Rate,
                    success_criteria: SuccessCriteria {
                        improvement_type: ImprovementType::Increase,
                        target_improvement: 0.1,
                        minimum_improvement: 0.05,
                        statistical_significance_required: true,
                    },
                    is_primary: true,
                    weight: 1.0,
                    guardrail_bounds: None,
                },
            ],
            start_time: chrono::Utc::now().timestamp_millis(),
            end_time: chrono::Utc::now().timestamp_millis() + 7 * 24 * 3600 * 1000,
            duration_days: 7,
            ramp_up_duration: Duration::from_secs(3600),
            ramp_down_duration: Duration::from_secs(3600),
            statistical_config: StatisticalConfig {
                test_type: StatisticalTest::TTest,
                multiple_testing_correction: MultipleTesting::BenjaminiHochberg,
                sequential_testing_enabled: false,
                bayesian_analysis_enabled: false,
                confidence_interval_type: ConfidenceIntervalType::Normal,
                bootstrap_samples: 1000,
            },
            significance_level: 0.05,
            minimum_detectable_effect: 0.05,
            power: 0.8,
            safety_rules: Vec::new(),
            circuit_breaker: CircuitBreakerConfig {
                enabled: false,
                failure_threshold: 0.1,
                recovery_threshold: 0.05,
                check_interval: Duration::from_secs(60),
                half_open_max_requests: 100,
            },
            rollback_conditions: Vec::new(),
            monitoring_config: MonitoringConfig {
                metrics_collection_interval: Duration::from_secs(60),
                real_time_monitoring: true,
                dashboard_refresh_interval: Duration::from_secs(30),
                data_quality_checks: Vec::new(),
            },
            alert_conditions: Vec::new(),
        };
        
        let result = manager.create_experiment(config).await;
        assert!(result.is_ok());
    }

    // Mock implementations for testing
    struct MockTrafficSplitter;
    
    impl TrafficSplitter for MockTrafficSplitter {
        fn assign_variant(&self, _user_id: &str, experiment: &ExperimentConfig) -> Result<String> {
            Ok(experiment.control_group.clone())
        }
        
        fn get_assignment(&self, _user_id: &str, _experiment_id: &str) -> Result<Option<String>> {
            Ok(Some("control".to_string()))
        }
    }
    
    struct MockStatisticalEngine;
    
    impl StatisticalEngine for MockStatisticalEngine {
        fn analyze_experiment(&self, _experiment: &ExperimentConfig, _data: &ExperimentData) -> Result<StatisticalAnalysis> {
            Ok(StatisticalAnalysis {
                overall_significance: false,
                winning_variant: None,
                effect_size: 0.0,
                power_achieved: 0.0,
                sample_size_adequacy: false,
                multiple_testing_correction_applied: false,
                bayesian_analysis: None,
            })
        }
        
        fn calculate_sample_size(&self, _config: &StatisticalConfig, _effect_size: f64, _power: f64, _alpha: f64) -> Result<usize> {
            Ok(1000)
        }
        
        fn perform_significance_test(&self, _control_data: &[f64], _treatment_data: &[f64]) -> Result<SignificanceTestResult> {
            Ok(SignificanceTestResult {
                p_value: 0.5,
                test_statistic: 0.0,
                effect_size: 0.0,
                confidence_interval: ConfidenceInterval {
                    lower_bound: -0.1,
                    upper_bound: 0.1,
                    confidence_level: 0.95,
                },
                is_significant: false,
            })
        }
    }
}