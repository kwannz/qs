use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Strategy {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub strategy_type: StrategyType,
    pub status: StrategyStatus,
    pub parameters: StrategyParameters,
    pub symbols: Vec<String>,
    pub exchanges: Vec<String>,
    pub created_by: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_execution: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrategyType {
    Momentum,
    MeanReversion,
    Arbitrage,
    GridTrading,
    Dca, // Dollar Cost Averaging
    PairsTrading,
    MLPrediction,
    FactorBased,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StrategyStatus {
    Draft,
    Active,
    Paused,
    Stopped,
    Error,
    Backtesting,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyParameters {
    // Common parameters
    pub risk_tolerance: f64,
    pub max_position_size: Decimal,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    pub evaluation_interval: u64, // seconds
    
    // Strategy-specific parameters
    pub momentum_lookback: Option<u32>,
    pub ma_short_period: Option<u32>,
    pub ma_long_period: Option<u32>,
    pub rsi_period: Option<u32>,
    pub rsi_oversold: Option<f64>,
    pub rsi_overbought: Option<f64>,
    pub bollinger_period: Option<u32>,
    pub bollinger_std: Option<f64>,
    pub grid_size: Option<u32>,
    pub grid_spacing_pct: Option<f64>,
    pub dca_interval: Option<u64>,
    pub dca_amount: Option<Decimal>,
    pub pairs_correlation_threshold: Option<f64>,
    pub ml_model_confidence_threshold: Option<f64>,
    pub factor_weights: Option<HashMap<String, f64>>,
    
    // Custom parameters
    pub custom_params: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub id: Uuid,
    pub strategy_id: Uuid,
    pub symbol: String,
    pub exchange: String,
    pub signal_type: SignalType,
    pub action: SignalAction,
    pub strength: f64, // 0.0 to 1.0
    pub confidence: f64, // 0.0 to 1.0
    pub price: Decimal,
    pub quantity: Option<Decimal>,
    pub reason: String,
    pub factors: HashMap<String, f64>,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub executed: bool,
    pub executed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalType {
    Entry,
    Exit,
    StopLoss,
    TakeProfit,
    Rebalance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalAction {
    Buy,
    Sell,
    Hold,
    Close,
    ReducePosition,
    IncreasePosition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyExecution {
    pub id: Uuid,
    pub strategy_id: Uuid,
    pub execution_time: DateTime<Utc>,
    pub status: ExecutionStatus,
    pub signals_generated: u32,
    pub signals_executed: u32,
    pub pnl: Decimal,
    pub execution_duration_ms: u64,
    pub error_message: Option<String>,
    pub metrics: ExecutionMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExecutionStatus {
    Running,
    Completed,
    Failed,
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    pub data_points_processed: u32,
    pub indicators_calculated: u32,
    pub factors_evaluated: u32,
    pub risk_checks_passed: u32,
    pub risk_checks_failed: u32,
    pub average_signal_strength: f64,
    pub average_signal_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyPerformance {
    pub strategy_id: Uuid,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub total_return: Decimal,
    pub total_return_pct: f64,
    pub annualized_return_pct: f64,
    pub volatility_pct: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown_pct: f64,
    pub win_rate_pct: f64,
    pub total_trades: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
    pub average_trade_return: Decimal,
    pub largest_win: Decimal,
    pub largest_loss: Decimal,
    pub calculated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketFactor {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: FactorCategory,
    pub value: f64,
    pub normalized_value: f64, // Z-score normalized
    pub percentile: f64, // Historical percentile (0-100)
    pub confidence: f64,
    pub last_updated: DateTime<Utc>,
    pub data_source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactorCategory {
    Technical,
    Fundamental,
    Sentiment,
    Macro,
    OnChain,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Indicator {
    pub name: String,
    pub symbol: String,
    pub timeframe: String,
    pub value: f64,
    pub previous_value: f64,
    pub change: f64,
    pub change_pct: f64,
    pub signal: IndicatorSignal,
    pub confidence: f64,
    pub calculated_at: DateTime<Utc>,
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndicatorSignal {
    StrongBuy,
    Buy,
    Neutral,
    Sell,
    StrongSell,
}

// Request/Response models for API

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateStrategyRequest {
    pub name: String,
    pub description: String,
    pub strategy_type: StrategyType,
    pub parameters: StrategyParameters,
    pub symbols: Vec<String>,
    pub exchanges: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStrategyRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub parameters: Option<StrategyParameters>,
    pub symbols: Option<Vec<String>>,
    pub exchanges: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateStrategyInstanceRequest {
    pub strategy_id: Uuid,
    pub name: String,
    pub parameters: StrategyParameters,
    pub symbols: Vec<String>,
    pub exchanges: Vec<String>,
    pub initial_capital: Decimal,
    pub max_loss_per_trade: Decimal,
    pub max_daily_loss: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyInstance {
    pub id: Uuid,
    pub strategy_id: Uuid,
    pub name: String,
    pub parameters: StrategyParameters,
    pub symbols: Vec<String>,
    pub exchanges: Vec<String>,
    pub initial_capital: Decimal,
    pub current_capital: Decimal,
    pub max_loss_per_trade: Decimal,
    pub max_daily_loss: Decimal,
    pub status: String,
    pub health_status: Option<String>,
    pub last_error: Option<String>,
    pub execution_count: u32,
    pub last_execution: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyStatusResponse {
    pub strategy_id: Uuid,
    pub status: StrategyStatus,
    pub last_execution: Option<DateTime<Utc>>,
    pub next_execution: Option<DateTime<Utc>>,
    pub active_signals: u32,
    pub open_positions: u32,
    pub current_pnl: Decimal,
    pub daily_pnl: Decimal,
    pub execution_metrics: Option<ExecutionMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalListResponse {
    pub signals: Vec<Signal>,
    pub total_count: u32,
    pub page: u32,
    pub per_page: u32,
    pub filters_applied: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorListResponse {
    pub indicators: Vec<Indicator>,
    pub symbol: String,
    pub timeframe: String,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorAnalysisResponse {
    pub factors: Vec<MarketFactor>,
    pub market_regime: MarketRegime,
    pub risk_assessment: RiskAssessment,
    pub recommendations: Vec<String>,
    pub calculated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketRegime {
    pub regime: String, // "Bull", "Bear", "Sideways", "Volatile"
    pub confidence: f64,
    pub duration_days: u32,
    pub characteristics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub overall_risk: f64, // 0-100 scale
    pub market_risk: f64,
    pub liquidity_risk: f64,
    pub volatility_risk: f64,
    pub correlation_risk: f64,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestRequest {
    pub strategy_id: Uuid,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub initial_capital: Decimal,
    pub benchmark: Option<String>,
    pub slippage_pct: Option<f64>,
    pub commission_pct: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResponse {
    pub backtest_id: Uuid,
    pub status: String,
    pub progress: f64, // 0-100
    pub estimated_completion: Option<DateTime<Utc>>,
    pub preliminary_results: Option<StrategyPerformance>,
}

impl Default for StrategyParameters {
    fn default() -> Self {
        Self {
            risk_tolerance: 0.5,
            max_position_size: Decimal::from(10000),
            stop_loss_pct: 5.0,
            take_profit_pct: 10.0,
            evaluation_interval: 300, // 5 minutes
            momentum_lookback: None,
            ma_short_period: None,
            ma_long_period: None,
            rsi_period: None,
            rsi_oversold: None,
            rsi_overbought: None,
            bollinger_period: None,
            bollinger_std: None,
            grid_size: None,
            grid_spacing_pct: None,
            dca_interval: None,
            dca_amount: None,
            pairs_correlation_threshold: None,
            ml_model_confidence_threshold: None,
            factor_weights: None,
            custom_params: HashMap::new(),
        }
    }
}

// Display implementations for filtering
impl fmt::Display for StrategyStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StrategyStatus::Draft => write!(f, "draft"),
            StrategyStatus::Active => write!(f, "active"),
            StrategyStatus::Paused => write!(f, "paused"),
            StrategyStatus::Stopped => write!(f, "stopped"),
            StrategyStatus::Error => write!(f, "error"),
            StrategyStatus::Backtesting => write!(f, "backtesting"),
        }
    }
}

impl fmt::Display for StrategyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StrategyType::Momentum => write!(f, "momentum"),
            StrategyType::MeanReversion => write!(f, "mean_reversion"),
            StrategyType::Arbitrage => write!(f, "arbitrage"),
            StrategyType::GridTrading => write!(f, "grid_trading"),
            StrategyType::Dca => write!(f, "dca"),
            StrategyType::PairsTrading => write!(f, "pairs_trading"),
            StrategyType::MLPrediction => write!(f, "ml_prediction"),
            StrategyType::FactorBased => write!(f, "factor_based"),
            StrategyType::Custom => write!(f, "custom"),
        }
    }
}