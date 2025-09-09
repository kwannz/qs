use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::{Id, Symbol, PositionSide};

/// 策略定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Strategy {
    pub id: Id,
    pub name: String,
    pub description: Option<String>,
    pub category: StrategyCategory,
    pub version: String,
    pub author: String,
    pub language: StrategyLanguage,
    pub source_code: Option<String>,
    pub binary_path: Option<String>,
    pub parameters: Vec<StrategyParameter>,
    pub default_parameters: HashMap<String, serde_json::Value>,
    pub supported_symbols: Option<Vec<Symbol>>,
    pub risk_profile: RiskProfile,
    pub performance_benchmark: Option<PerformanceBenchmark>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub is_active: bool,
    pub tags: Vec<String>,
}

/// 策略分类
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StrategyCategory {
    Trend,
    MeanReversion,
    Momentum,
    Arbitrage,
    MarketMaking,
    Grid,
    Scalping,
    Swing,
    LongTerm,
    MultiAsset,
    Custom,
}

/// 策略语言
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StrategyLanguage {
    Python,
    Rust,
    JavaScript,
    WASM,
    DSL, // Domain Specific Language
}

/// 策略参数定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyParameter {
    pub name: String,
    pub param_type: ParameterType,
    pub description: Option<String>,
    pub default_value: serde_json::Value,
    pub min_value: Option<serde_json::Value>,
    pub max_value: Option<serde_json::Value>,
    pub allowed_values: Option<Vec<serde_json::Value>>,
    pub is_required: bool,
    pub is_optimizable: bool,
}

/// 参数类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterType {
    Integer,
    Float,
    Boolean,
    String,
    Symbol,
    Duration,
    Percentage,
    Array,
    Object,
}

/// 风险档案
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskProfile {
    pub risk_level: RiskLevel,
    pub max_drawdown: Decimal,
    pub var_95: Option<Decimal>,
    pub volatility: Option<Decimal>,
    pub leverage_limit: Decimal,
    pub position_size_limit: Decimal,
    pub daily_loss_limit: Decimal,
    pub correlation_limit: Option<Decimal>,
}

/// 风险级别
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RiskLevel {
    Conservative,
    Moderate,
    Aggressive,
    HighRisk,
}

/// 性能基准
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBenchmark {
    pub benchmark_symbol: Symbol,
    pub expected_return: Decimal,
    pub expected_volatility: Decimal,
    pub expected_sharpe: Decimal,
    pub max_correlation: Option<Decimal>,
}

/// 策略配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    pub id: Id,
    pub strategy_id: Id,
    pub name: String,
    pub description: Option<String>,
    pub parameters: HashMap<String, serde_json::Value>,
    pub symbols: Vec<Symbol>,
    pub enabled: bool,
    pub auto_start: bool,
    pub max_position_size: Decimal,
    pub risk_multiplier: Decimal,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// 策略信号（AG3 增强版 - 包含元数据）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySignal {
    pub id: Id,
    pub strategy_id: Id,
    pub symbol: Symbol,
    pub signal_type: StrategySignalType,
    pub strength: Decimal, // -1.0 to 1.0, where negative is sell, positive is buy
    pub confidence: Decimal, // 0.0 to 1.0
    pub entry_price: Option<Decimal>,
    pub stop_loss: Option<Decimal>,
    pub take_profit: Option<Decimal>,
    pub position_size: Option<Decimal>,
    pub holding_period: Option<u64>, // seconds
    
    // AG3 新增元数据字段
    pub half_life: Option<u64>, // 信号半衰期（秒）
    pub suggested_holding: Option<u64>, // 建议持有期（秒）
    pub risk_tags: Vec<String>, // 风险标签
    pub quality_score: Option<Decimal>, // 信号质量评分 (0.0 to 1.0)
    pub decay_factor: Option<Decimal>, // 衰减因子
    pub regime: Option<String>, // 体制标签
    pub ic_score: Option<Decimal>, // 信息系数
    pub turnover_cost: Option<Decimal>, // 预估换手成本
    
    pub metadata: HashMap<String, serde_json::Value>,
    pub timestamp: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

/// 策略信号类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StrategySignalType {
    Entry,
    Exit,
    StopLoss,
    TakeProfit,
    PositionAdjust,
    RiskReduce,
}

/// 策略状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyState {
    pub strategy_id: Id,
    pub status: StrategyStatus,
    pub current_positions: HashMap<Symbol, StrategyPosition>,
    pub pending_orders: Vec<Id>,
    pub total_pnl: Decimal,
    pub daily_pnl: Decimal,
    pub unrealized_pnl: Decimal,
    pub realized_pnl: Decimal,
    pub max_drawdown: Decimal,
    pub win_rate: Decimal,
    pub trades_count: u64,
    pub last_signal_time: Option<DateTime<Utc>>,
    pub last_trade_time: Option<DateTime<Utc>>,
    pub error_count: u32,
    pub last_error: Option<String>,
    pub updated_at: DateTime<Utc>,
}

/// 策略运行状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StrategyStatus {
    Stopped,
    Starting,
    Running,
    Pausing,
    Paused,
    Stopping,
    Error,
    Disabled,
}

/// 策略持仓
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyPosition {
    pub symbol: Symbol,
    pub side: PositionSide,
    pub size: Decimal,
    pub average_price: Decimal,
    pub current_price: Decimal,
    pub unrealized_pnl: Decimal,
    pub realized_pnl: Decimal,
    pub entry_time: DateTime<Utc>,
    pub last_update: DateTime<Utc>,
}

// PositionSide moved to common.rs to avoid duplication

/// 策略回测结果（AG3 增强版 - 支持新验证方法）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyBacktest {
    pub id: Id,
    pub strategy_id: Id,
    pub config_id: Option<Id>,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub initial_capital: Decimal,
    pub final_capital: Decimal,
    pub total_return: Decimal,
    pub annualized_return: Decimal,
    pub volatility: Decimal,
    pub sharpe_ratio: Decimal,
    pub sortino_ratio: Decimal,
    pub max_drawdown: Decimal,
    pub max_drawdown_duration: u64, // days
    pub win_rate: Decimal,
    pub profit_factor: Decimal,
    pub total_trades: u64,
    pub avg_trade_return: Decimal,
    pub trades: Vec<BacktestTrade>,
    pub equity_curve: Vec<EquityPoint>,
    
    // AG3 新增验证字段
    pub validation_method: Option<ValidationMethod>,
    pub purged_kfold_results: Option<PurgedKFoldResults>,
    pub triple_barrier_labels: Option<Vec<TripleBarrierLabel>>,
    pub ic_metrics: Option<ICMetrics>,
    pub regime_analysis: Option<RegimeAnalysis>,
    pub walk_forward_results: Option<Vec<WalkForwardResult>>,
    
    pub created_at: DateTime<Utc>,
}

/// 回测交易
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestTrade {
    pub symbol: Symbol,
    pub side: PositionSide,
    pub entry_price: Decimal,
    pub exit_price: Decimal,
    pub quantity: Decimal,
    pub entry_time: DateTime<Utc>,
    pub exit_time: DateTime<Utc>,
    pub pnl: Decimal,
    pub return_pct: Decimal,
    pub duration_hours: f64,
    pub commission: Decimal,
}

/// 净值点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityPoint {
    pub timestamp: DateTime<Utc>,
    pub equity: Decimal,
    pub benchmark: Option<Decimal>,
    pub drawdown: Decimal,
    pub returns: Decimal,
}

/// 策略优化任务
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyOptimization {
    pub id: Id,
    pub strategy_id: Id,
    pub optimization_type: OptimizationType,
    pub parameters: Vec<OptimizationParameter>,
    pub objective: OptimizationObjective,
    pub constraints: Vec<OptimizationConstraint>,
    pub status: OptimizationStatus,
    pub progress: u8, // 0-100
    pub best_result: Option<OptimizationResult>,
    pub all_results: Vec<OptimizationResult>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// 优化类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationType {
    GridSearch,
    RandomSearch,
    BayesianOptimization,
    GeneticAlgorithm,
    ParticleSwarm,
    SimulatedAnnealing,
}

/// 优化参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationParameter {
    pub name: String,
    pub param_type: ParameterType,
    pub min_value: serde_json::Value,
    pub max_value: serde_json::Value,
    pub step_size: Option<serde_json::Value>,
    pub discrete_values: Option<Vec<serde_json::Value>>,
}

/// 优化目标
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationObjective {
    MaximizeReturn,
    MaximizeSharpe,
    MaximizeSortino,
    MinimizeDrawdown,
    MinimizeVolatility,
    MaximizeProfitFactor,
    MaximizeWinRate,
    Custom(String),
}

/// 优化约束
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConstraint {
    pub constraint_type: ConstraintType,
    pub parameter: String,
    pub value: serde_json::Value,
    pub operator: ComparisonOperator,
}

/// 约束类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstraintType {
    MaxDrawdown,
    MinSharpe,
    MinWinRate,
    MaxVolatility,
    MinTrades,
    Custom(String),
}

/// 比较运算符
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    GreaterEqual,
    LessEqual,
    Equal,
    NotEqual,
}

/// 优化状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// 优化结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub parameters: HashMap<String, serde_json::Value>,
    pub objective_value: Decimal,
    pub backtest_result: Option<StrategyBacktest>,
    pub constraint_violations: Vec<String>,
    pub is_valid: bool,
}

/// 策略组合
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyPortfolio {
    pub id: Id,
    pub name: String,
    pub description: Option<String>,
    pub strategies: Vec<PortfolioStrategy>,
    pub rebalancing_frequency: RebalancingFrequency,
    pub risk_budget: Decimal,
    pub max_correlation: Decimal,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// 组合策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioStrategy {
    pub strategy_id: Id,
    pub weight: Decimal,
    pub max_allocation: Decimal,
    pub min_allocation: Decimal,
    pub risk_contribution: Option<Decimal>,
}

/// 再平衡频率
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RebalancingFrequency {
    Never,
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Annually,
    OnThreshold(Decimal), // rebalance when allocation differs by more than X%
}

// =================== AG3 新增结构定义 ===================

/// 验证方法
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationMethod {
    TraditionalSplit,
    PurgedKFold,
    WalkForward,
    CombinatorialPurging,
    EmbargoedPurging,
}

/// Purged K-Fold 验证结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurgedKFoldResults {
    pub n_folds: u32,
    pub purge_period: u64, // 清洗期（秒）
    pub embargo_period: u64, // 禁售期（秒）
    pub fold_results: Vec<FoldResult>,
    pub avg_score: Decimal,
    pub std_score: Decimal,
    pub information_leakage_score: Decimal, // 0.0-1.0，越低越好
}

/// 单折验证结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldResult {
    pub fold_id: u32,
    pub train_start: DateTime<Utc>,
    pub train_end: DateTime<Utc>,
    pub test_start: DateTime<Utc>,
    pub test_end: DateTime<Utc>,
    pub score: Decimal,
    pub sharpe_ratio: Decimal,
    pub max_drawdown: Decimal,
}

/// 三重障碍标签
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleBarrierLabel {
    pub timestamp: DateTime<Utc>,
    pub symbol: Symbol,
    pub entry_price: Decimal,
    pub upper_barrier: Decimal, // 止盈位
    pub lower_barrier: Decimal, // 止损位
    pub time_barrier: DateTime<Utc>, // 时间障碍
    pub exit_type: BarrierExitType,
    pub exit_price: Option<Decimal>,
    pub exit_time: Option<DateTime<Utc>>,
    pub return_pct: Decimal,
}

/// 障碍退出类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BarrierExitType {
    TakeProfit,  // 触及上障碍
    StopLoss,    // 触及下障碍
    TimeExpiry,  // 时间到期
    NoExit,      // 未退出
}

/// IC指标集
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ICMetrics {
    pub ic: Decimal, // 信息系数
    pub ic_ir: Decimal, // 信息比率
    pub rolling_ic: Vec<RollingICPoint>,
    pub rank_ic: Decimal, // 秩信息系数
    pub turnover: Decimal, // 换手率
    pub net_alpha: Decimal, // 净Alpha
    pub hit_rate: Decimal, // 命中率
    pub decay_analysis: DecayAnalysis,
}

/// 滚动IC点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingICPoint {
    pub timestamp: DateTime<Utc>,
    pub ic: Decimal,
    pub sample_size: u32,
}

/// 衰减分析
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayAnalysis {
    pub half_life_days: f64,
    pub decay_rate: Decimal,
    pub decay_curve: Vec<DecayPoint>,
}

/// 衰减点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayPoint {
    pub days_ahead: u32,
    pub ic: Decimal,
}

/// 体制分析
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeAnalysis {
    pub detected_regimes: Vec<RegimePeriod>,
    pub regime_performance: HashMap<String, RegimePerformance>,
    pub transition_matrix: Vec<Vec<Decimal>>, // 转换概率矩阵
    pub current_regime: Option<String>,
    pub regime_stability: Decimal, // 0.0-1.0
}

/// 体制期间
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimePeriod {
    pub regime_id: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub confidence: Decimal,
    pub characteristics: HashMap<String, Decimal>,
}

/// 体制表现
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimePerformance {
    pub total_return: Decimal,
    pub sharpe_ratio: Decimal,
    pub max_drawdown: Decimal,
    pub win_rate: Decimal,
    pub avg_trade_duration: f64, // 小时
}

/// Walk Forward结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardResult {
    pub period_id: u32,
    pub train_start: DateTime<Utc>,
    pub train_end: DateTime<Utc>,
    pub test_start: DateTime<Utc>,
    pub test_end: DateTime<Utc>,
    pub in_sample_return: Decimal,
    pub out_of_sample_return: Decimal,
    pub efficiency_ratio: Decimal, // OOS/IS比率
    pub parameter_stability: Decimal, // 参数稳定性
}