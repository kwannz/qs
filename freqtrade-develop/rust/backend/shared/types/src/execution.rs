use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::{Id, Symbol, Price, Quantity, PositionSide};

/// 订单类型
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    StopMarket,
    StopLimit,
    TakeProfit,
    TakeProfitLimit,
    TrailingStop,
    Iceberg,
    OCO, // One-Cancels-Other
}

/// 订单方向
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// 订单状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
    PendingCancel,
    PendingNew,
}

/// 订单时效类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TimeInForce {
    GTC, // Good Till Cancel
    IOC, // Immediate or Cancel
    FOK, // Fill or Kill
    GTD, // Good Till Date
}

/// 订单结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: Id,
    pub client_order_id: String,
    pub exchange_order_id: Option<String>,
    pub symbol: Symbol,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: Quantity,
    pub price: Option<Price>,
    pub stop_price: Option<Price>,
    pub time_in_force: TimeInForce,
    pub status: OrderStatus,
    pub filled_quantity: Quantity,
    pub remaining_quantity: Quantity,
    pub average_price: Option<Price>,
    pub commission: Decimal,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub executed_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
    pub strategy_id: Option<Id>,
    pub parent_order_id: Option<Id>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// 订单执行报告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionReport {
    pub id: Id,
    pub order_id: Id,
    pub trade_id: Option<String>,
    pub symbol: Symbol,
    pub side: OrderSide,
    pub quantity: Quantity,
    pub price: Price,
    pub commission: Decimal,
    pub commission_asset: String,
    pub timestamp: DateTime<Utc>,
    pub execution_type: ExecutionType,
    pub order_status: OrderStatus,
    pub reject_reason: Option<String>,
}

/// 执行类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExecutionType {
    New,
    Cancelled,
    Replaced,
    Rejected,
    Trade,
    Expired,
}

/// 持仓信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub id: Id,
    pub symbol: Symbol,
    pub side: PositionSide,
    pub size: Quantity,
    pub entry_price: Price,
    pub mark_price: Price,
    pub unrealized_pnl: Decimal,
    pub realized_pnl: Decimal,
    pub margin: Decimal,
    pub margin_ratio: Decimal,
    pub leverage: Decimal,
    pub liquidation_price: Option<Price>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub strategy_id: Option<Id>,
}

// PositionSide moved to common.rs to avoid duplication

/// 账户信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub id: Id,
    pub exchange: String,
    pub account_type: AccountType,
    pub balances: Vec<Balance>,
    pub total_wallet_balance: Decimal,
    pub total_unrealized_pnl: Decimal,
    pub total_margin_balance: Decimal,
    pub available_balance: Decimal,
    pub max_withdraw_amount: Decimal,
    pub updated_at: DateTime<Utc>,
}

/// 账户类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AccountType {
    Spot,
    Margin,
    Futures,
    Options,
}

/// 余额信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Balance {
    pub asset: String,
    pub free: Decimal,
    pub locked: Decimal,
    pub total: Decimal,
    pub wallet_balance: Option<Decimal>,
    pub unrealized_pnl: Option<Decimal>,
    pub margin_balance: Option<Decimal>,
}

/// 交易记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: Id,
    pub exchange_trade_id: String,
    pub order_id: Id,
    pub symbol: Symbol,
    pub side: OrderSide,
    pub quantity: Quantity,
    pub price: Price,
    pub commission: Decimal,
    pub commission_asset: String,
    pub realized_pnl: Option<Decimal>,
    pub timestamp: DateTime<Utc>,
    pub is_maker: bool,
    pub strategy_id: Option<Id>,
}

/// 风险限制
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    pub max_position_size: Decimal,
    pub max_order_size: Decimal,
    pub max_daily_loss: Decimal,
    pub max_drawdown: Decimal,
    pub max_leverage: Decimal,
    pub allowed_symbols: Option<Vec<Symbol>>,
    pub blocked_symbols: Option<Vec<Symbol>>,
    pub max_orders_per_second: Option<u32>,
    pub max_orders_per_day: Option<u32>,
}

/// 执行策略状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExecutionStrategyStatus {
    Stopped,
    Running,
    Paused,
    Error,
    Stopping,
    Starting,
}

/// 策略实例
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyInstance {
    pub id: Id,
    pub strategy_id: Id,
    pub name: String,
    pub status: ExecutionStrategyStatus,
    pub symbols: Vec<Symbol>,
    pub parameters: HashMap<String, serde_json::Value>,
    pub risk_limits: RiskLimits,
    pub positions: Vec<Position>,
    pub orders: Vec<Order>,
    pub pnl: Decimal,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub stopped_at: Option<DateTime<Utc>>,
    pub last_signal_at: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
}

/// 交易信号
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub id: Id,
    pub strategy_id: Id,
    pub symbol: Symbol,
    pub signal_type: TradingSignalType,
    pub strength: Decimal, // 0.0 to 1.0
    pub target_quantity: Option<Quantity>,
    pub target_price: Option<Price>,
    pub stop_loss: Option<Price>,
    pub take_profit: Option<Price>,
    pub expires_at: Option<DateTime<Utc>>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub processed: bool,
    pub processed_at: Option<DateTime<Utc>>,
}

/// 交易信号类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TradingSignalType {
    Buy,
    Sell,
    Hold,
    Close,
    IncreasePosition,
    DecreasePosition,
}

/// 订单请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderRequest {
    pub client_order_id: Option<String>,
    pub symbol: Symbol,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: Quantity,
    pub price: Option<Price>,
    pub stop_price: Option<Price>,
    pub time_in_force: Option<TimeInForce>,
    pub strategy_id: Option<Id>,
    pub reduce_only: Option<bool>,
    pub close_position: Option<bool>,
}

/// 取消订单请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelOrderRequest {
    pub order_id: Option<Id>,
    pub client_order_id: Option<String>,
    pub symbol: Option<Symbol>,
}

/// 修改订单请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModifyOrderRequest {
    pub order_id: Id,
    pub quantity: Option<Quantity>,
    pub price: Option<Price>,
    pub stop_price: Option<Price>,
}

/// 交易统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingStats {
    pub strategy_id: Option<Id>,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub total_trades: u64,
    pub winning_trades: u64,
    pub losing_trades: u64,
    pub total_volume: Decimal,
    pub total_commission: Decimal,
    pub gross_profit: Decimal,
    pub gross_loss: Decimal,
    pub net_profit: Decimal,
    pub profit_factor: Decimal,
    pub win_rate: Decimal,
    pub average_win: Decimal,
    pub average_loss: Decimal,
    pub largest_win: Decimal,
    pub largest_loss: Decimal,
    pub max_consecutive_wins: u32,
    pub max_consecutive_losses: u32,
    pub sharpe_ratio: Option<Decimal>,
    pub sortino_ratio: Option<Decimal>,
    pub max_drawdown: Decimal,
}

/// 交易所连接状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConnectionStatus {
    Connected,
    Disconnected,
    Connecting,
    Reconnecting,
    Error,
}

/// 交易所状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeStatus {
    pub exchange: String,
    pub connection_status: ConnectionStatus,
    pub last_heartbeat: DateTime<Utc>,
    pub latency_ms: Option<u64>,
    pub orders_pending: u32,
    pub trades_today: u32,
    pub error_count: u32,
    pub last_error: Option<String>,
    pub rate_limit_remaining: Option<u32>,
    pub rate_limit_reset: Option<DateTime<Utc>>,
}

// =================== AG3 自适应执行算法 ===================

/// 执行算法类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExecutionAlgorithm {
    TWAP,    // 时间加权平均价格
    VWAP,    // 成交量加权平均价格
    PoV,     // 参与量百分比
    IS,      // 实施缺口
    Adaptive, // 自适应算法
    Custom(String),
}

/// 自适应执行配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveExecutionConfig {
    pub algorithm: ExecutionAlgorithm,
    pub target_quantity: Quantity,
    pub time_horizon_seconds: u64,
    pub participation_rate: Decimal, // 0.0-1.0
    pub max_participation_rate: Decimal,
    pub min_participation_rate: Decimal,
    pub urgency_factor: Decimal, // 0.0-1.0，紧迫性
    pub risk_aversion: Decimal,  // 0.0-1.0，风险厌恶
    pub impact_tolerance: Decimal, // 可接受的市场冲击
    pub adaptive_params: AdaptiveParams,
}

/// 自适应参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveParams {
    pub liquidity_sensitivity: Decimal,
    pub volatility_adjustment: bool,
    pub momentum_detection: bool,
    pub adverse_selection_protection: bool,
    pub queue_position_awareness: bool,
    pub cross_venue_optimization: bool,
}

/// 执行策略状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionState {
    pub request_id: Id,
    pub algorithm: ExecutionAlgorithm,
    pub config: AdaptiveExecutionConfig,
    pub progress: ExecutionProgress,
    pub performance_metrics: ExecutionMetrics,
    pub current_orders: Vec<Id>,
    pub completed_fills: Vec<ExecutionFill>,
    pub status: ExecutionStatus,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub last_update: DateTime<Utc>,
}

/// 执行进度
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionProgress {
    pub target_quantity: Quantity,
    pub executed_quantity: Quantity,
    pub remaining_quantity: Quantity,
    pub completion_rate: Decimal, // 0.0-1.0
    pub elapsed_time_seconds: u64,
    pub estimated_completion_seconds: Option<u64>,
}

/// 执行状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Pending,
    Running,
    Paused,
    Completed,
    Cancelled,
    Failed,
}

/// 执行成交记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionFill {
    pub fill_id: Id,
    pub order_id: Id,
    pub timestamp: DateTime<Utc>,
    pub quantity: Quantity,
    pub price: Price,
    pub venue: String,
    pub commission: Decimal,
    pub is_maker: bool,
    pub market_impact: Option<Decimal>,
    pub queue_position: Option<u32>,
    pub liquidity_score: Option<Decimal>,
}

/// 执行指标（AG3关键指标埋点）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    pub arrival_price: Price,
    pub volume_weighted_price: Price,
    pub implementation_shortfall: Decimal,
    pub market_impact: Decimal,
    pub timing_cost: Decimal,
    pub opportunity_cost: Decimal,
    
    // AG3 新增指标
    pub effective_spread: Decimal,
    pub effective_slippage: Decimal,
    pub fill_rate: Decimal, // 成交率
    pub cancel_rate: Decimal, // 撤单率
    pub participation_deviation: Decimal, // 参与率偏差
    pub queue_jump_count: u32, // 跳队次数
    pub adverse_selection_cost: Decimal,
    pub venue_hit_rate: HashMap<String, Decimal>, // 各venue命中率
    
    pub total_commission: Decimal,
    pub total_cost: Decimal,
    pub cost_per_share: Decimal,
    pub execution_time_seconds: u64,
    pub average_order_size: Quantity,
    pub order_count: u32,
}

/// 流动性函数参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityFunction {
    pub linear_impact: Decimal,      // 线性冲击系数
    pub permanent_impact: Decimal,   // 永久冲击系数
    pub temporary_impact: Decimal,   // 临时冲击系数
    pub volatility_factor: Decimal,  // 波动率因子
    pub volume_curve_params: VolumeCurveParams,
}

/// 成交量曲线参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeCurveParams {
    pub intraday_pattern: Vec<Decimal>, // 日内模式
    pub decay_factor: Decimal,
    pub seasonality_adjustment: bool,
    pub weekend_adjustment: Decimal,
}

/// 智能路由配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartRoutingConfig {
    pub enabled_venues: Vec<VenueConfig>,
    pub routing_algorithm: RoutingAlgorithm,
    pub latency_weights: HashMap<String, Decimal>,
    pub cost_weights: HashMap<String, Decimal>,
    pub fill_probability_weights: HashMap<String, Decimal>,
    pub min_order_size: HashMap<String, Quantity>,
    pub max_order_size: HashMap<String, Quantity>,
}

/// 交易场所配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueConfig {
    pub venue_name: String,
    pub priority: u32,
    pub max_allocation_pct: Decimal,
    pub min_allocation_pct: Decimal,
    pub latency_ms: u64,
    pub maker_fee: Decimal,
    pub taker_fee: Decimal,
    pub market_data_quality: Decimal, // 0.0-1.0
    pub order_rejection_rate: Decimal,
    pub is_active: bool,
}

/// 路由算法
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RoutingAlgorithm {
    BestPrice,
    SmartOrder,
    ProRata,
    FIFO,
    MLOptimized,
    MultiObjective,
}

/// 微结构Alpha信号
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrostructureSignal {
    pub signal_id: Id,
    pub symbol: Symbol,
    pub timestamp: DateTime<Utc>,
    pub signal_type: MicrostructureSignalType,
    pub strength: Decimal, // -1.0 to 1.0
    pub confidence: Decimal, // 0.0 to 1.0
    pub horizon_seconds: u64,
    pub suggested_action: MicroAction,
    pub market_regime: Option<String>,
}

/// 微结构信号类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MicrostructureSignalType {
    OrderBookImbalance,
    FlowToxicity,
    LiquidityDrying,
    MomentumBurst,
    MeanReversion,
    InformedTrading,
}

/// 微结构建议动作
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MicroAction {
    Accelerate,   // 加速执行
    Decelerate,   // 减缓执行
    SwitchVenue,  // 切换venue
    WaitForFill,  // 等待成交
    Cancel,       // 撤销订单
    HideLiquidity, // 隐藏流动性
}

impl Default for AdaptiveParams {
    fn default() -> Self {
        Self {
            liquidity_sensitivity: Decimal::from_parts(7, 0, 0, false, 1), // 0.7
            volatility_adjustment: true,
            momentum_detection: true,
            adverse_selection_protection: true,
            queue_position_awareness: true,
            cross_venue_optimization: true,
        }
    }
}

impl Default for AdaptiveExecutionConfig {
    fn default() -> Self {
        Self {
            algorithm: ExecutionAlgorithm::Adaptive,
            target_quantity: Quantity::from(0),
            time_horizon_seconds: 3600, // 1小时
            participation_rate: Decimal::from_parts(1, 0, 0, false, 1), // 0.1 (10%)
            max_participation_rate: Decimal::from_parts(3, 0, 0, false, 1), // 0.3 (30%)
            min_participation_rate: Decimal::from_parts(5, 0, 0, false, 2), // 0.05 (5%)
            urgency_factor: Decimal::from_parts(5, 0, 0, false, 1), // 0.5
            risk_aversion: Decimal::from_parts(5, 0, 0, false, 1), // 0.5
            impact_tolerance: Decimal::from_parts(1, 0, 0, false, 2), // 0.01 (1%)
            adaptive_params: AdaptiveParams::default(),
        }
    }
}