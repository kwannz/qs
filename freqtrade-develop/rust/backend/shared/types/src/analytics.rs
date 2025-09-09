use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::{Id, Symbol, Price, Quantity};

/// 因子值类型
pub type FactorValue = Decimal;

/// 因子数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Factor {
    pub id: Id,
    pub name: String,
    pub category: FactorCategory,
    pub description: Option<String>,
    pub formula: Option<String>,
    pub parameters: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// 因子分类
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FactorCategory {
    Technical,
    Fundamental,
    Sentiment,
    Market,
    Volume,
    Volatility,
    Momentum,
    MeanReversion,
    Custom,
}

/// 因子值数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorData {
    pub factor_id: Id,
    pub symbol: Symbol,
    pub timestamp: DateTime<Utc>,
    pub value: FactorValue,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// 回测配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub id: Id,
    pub name: String,
    pub description: Option<String>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub initial_capital: Decimal,
    pub symbols: Vec<Symbol>,
    pub benchmark: Option<Symbol>,
    pub commission: Decimal,
    pub slippage: Decimal,
    pub parameters: HashMap<String, serde_json::Value>,
}

/// 回测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub id: Id,
    pub config_id: Id,
    pub status: BacktestStatus,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub duration_seconds: Option<u64>,
    pub performance: Option<PerformanceMetrics>,
    pub trades: Vec<AnalyticsBacktestTrade>,
    pub equity_curve: Vec<AnalyticsEquityPoint>,
    pub error_message: Option<String>,
}

/// 回测状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BacktestStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// 性能指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
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
    pub winning_trades: u64,
    pub losing_trades: u64,
    pub average_trade: Decimal,
    pub largest_win: Decimal,
    pub largest_loss: Decimal,
    pub consecutive_wins: u64,
    pub consecutive_losses: u64,
    pub beta: Option<Decimal>,
    pub alpha: Option<Decimal>,
    pub information_ratio: Option<Decimal>,
}

/// 分析回测交易记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsBacktestTrade {
    pub id: Id,
    pub symbol: Symbol,
    pub side: AnalyticsTradeSide,
    pub quantity: Quantity,
    pub entry_price: Price,
    pub entry_time: DateTime<Utc>,
    pub exit_price: Option<Price>,
    pub exit_time: Option<DateTime<Utc>>,
    pub pnl: Option<Decimal>,
    pub commission: Decimal,
    pub slippage: Decimal,
    pub duration: Option<u64>, // seconds
    pub status: TradeStatus,
}

/// 分析交易方向
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AnalyticsTradeSide {
    Long,
    Short,
}

/// 交易状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TradeStatus {
    Open,
    Closed,
    Cancelled,
}

/// 分析净值曲线点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsEquityPoint {
    pub timestamp: DateTime<Utc>,
    pub equity: Decimal,
    pub benchmark: Option<Decimal>,
    pub drawdown: Decimal,
}

/// 分析任务
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisTask {
    pub id: Id,
    pub name: String,
    pub task_type: AnalysisTaskType,
    pub status: TaskStatus,
    pub parameters: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub progress: u8, // 0-100
    pub result: Option<serde_json::Value>,
    pub error_message: Option<String>,
}

/// 分析任务类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnalysisTaskType {
    FactorCalculation,
    BacktestExecution,
    OptimizationRun,
    RiskAnalysis,
    CorrelationAnalysis,
    PerformanceAttribution,
}

/// 任务状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TaskStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// 优化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub id: Id,
    pub name: String,
    pub strategy_id: Id,
    pub parameters: Vec<AnalyticsOptimizationParameter>,
    pub objective: AnalyticsOptimizationObjective,
    pub method: OptimizationMethod,
    pub max_iterations: Option<u32>,
    pub tolerance: Option<Decimal>,
}

/// 分析优化参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsOptimizationParameter {
    pub name: String,
    pub min_value: Decimal,
    pub max_value: Decimal,
    pub step: Option<Decimal>,
    pub values: Option<Vec<Decimal>>,
}

/// 分析优化目标
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnalyticsOptimizationObjective {
    MaximizeSharpeRatio,
    MaximizeReturn,
    MinimizeVolatility,
    MinimizeDrawdown,
    MaximizeProfitFactor,
    Custom(String),
}

/// 优化方法
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationMethod {
    GridSearch,
    GeneticAlgorithm,
    BayesianOptimization,
    ParticleSwarmOptimization,
}

/// 分析优化结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsOptimizationResult {
    pub id: Id,
    pub config_id: Id,
    pub best_parameters: HashMap<String, Decimal>,
    pub best_score: Decimal,
    pub iterations: u32,
    pub results: Vec<OptimizationIteration>,
}

/// 优化迭代结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationIteration {
    pub iteration: u32,
    pub parameters: HashMap<String, Decimal>,
    pub score: Decimal,
    pub metrics: Option<PerformanceMetrics>,
}

/// 相关性分析结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAnalysis {
    pub id: Id,
    pub symbols: Vec<Symbol>,
    pub factors: Vec<Id>,
    pub correlation_matrix: Vec<Vec<Decimal>>,
    pub timestamp: DateTime<Utc>,
    pub period_days: u32,
}

/// 风险分析报告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAnalysis {
    pub id: Id,
    pub portfolio_id: Option<Id>,
    pub symbols: Vec<Symbol>,
    pub var_95: Decimal,    // 95% VaR
    pub var_99: Decimal,    // 99% VaR
    pub cvar_95: Decimal,   // 95% CVaR
    pub maximum_drawdown: Decimal,
    pub volatility: Decimal,
    pub beta: Option<Decimal>,
    pub timestamp: DateTime<Utc>,
    pub analysis_period: String,
}