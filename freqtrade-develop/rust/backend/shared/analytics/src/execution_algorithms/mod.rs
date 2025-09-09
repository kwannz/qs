use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod market_impact;
pub mod vwap;
pub mod pov;
pub mod twap;
pub mod is; // Implementation Shortfall
pub mod iceberg;

/// 执行算法特征
pub trait ExecutionAlgorithm: Send + Sync {
    /// 算法名称
    fn name(&self) -> &str;
    
    /// 计算子单分割方案
    fn calculate_child_orders(
        &self,
        parent_order: &ParentOrder,
        market_conditions: &MarketConditions,
        execution_params: &ExecutionParams,
    ) -> Result<Vec<ChildOrder>>;
    
    /// 实时参数调整
    fn adapt_parameters(
        &mut self,
        execution_state: &ExecutionState,
        market_update: &MarketUpdate,
    ) -> Result<()>;
    
    /// 获取算法统计信息
    fn get_statistics(&self) -> AlgorithmStatistics;
    
    /// 验证算法参数
    fn validate_parameters(&self, params: &HashMap<String, f64>) -> Result<()>;
}

/// 父订单
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParentOrder {
    pub id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub total_quantity: f64,
    pub order_type: OrderType,
    pub time_horizon: i64, // 执行时间窗口(秒)
    pub urgency: f64, // 0.0-1.0
    pub limit_price: Option<f64>,
    pub arrival_price: f64,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// 子订单
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChildOrder {
    pub id: String,
    pub parent_id: String,
    pub sequence_number: u32,
    pub quantity: f64,
    pub price: Option<f64>,
    pub venue: String,
    pub order_type: OrderType,
    pub time_in_force: TimeInForce,
    pub scheduled_time: DateTime<Utc>,
    pub execution_window: i64, // 执行窗口(秒)
    pub is_hidden: bool,
    pub display_quantity: Option<f64>,
    pub post_only: bool,
    pub reduce_only: bool,
}

/// 订单方向
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// 订单类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    StopMarket,
    StopLimit,
    PostOnly,
    IcebergLimit,
    HiddenLimit,
}

/// 时间有效性
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TimeInForce {
    GoodTillCancel,
    ImmediateOrCancel,
    FillOrKill,
    GoodTillTime(DateTime<Utc>),
    DayOrder,
}

/// 市场条件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    
    // 价格数据
    pub mid_price: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub spread_bps: f64,
    pub tick_size: f64,
    
    // 流动性数据
    pub bid_size: f64,
    pub ask_size: f64,
    pub market_depth: MarketDepth,
    pub average_daily_volume: f64,
    pub current_volume: f64,
    pub volume_profile: Vec<VolumeProfileBucket>,
    
    // 波动性数据
    pub realized_volatility: f64,
    pub implied_volatility: f64,
    pub price_momentum: f64,
    pub short_term_trend: f64,
    
    // 微观结构数据
    pub order_book_imbalance: f64,
    pub queue_position_estimate: f64,
    pub toxic_flow_indicator: f64,
    pub informed_trading_probability: f64,
    
    // 时间因素
    pub time_to_close: i64, // 距收盘秒数
    pub intraday_period: IntradayPeriod,
    pub is_auction_period: bool,
    pub trading_session: TradingSession,
}

/// 市场深度
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDepth {
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
    pub total_bid_volume: f64,
    pub total_ask_volume: f64,
}

/// 价格档位
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: f64,
    pub size: f64,
    pub order_count: u32,
}

/// 成交量分布
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeProfileBucket {
    pub time_bucket: u32, // 分钟为单位
    pub volume: f64,
    pub vwap: f64,
    pub participation_rate: f64,
}

/// 日内时段
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IntradayPeriod {
    PreMarket,
    OpeningAuction,
    MorningSession,
    MiddayLull,
    AfternoonSession,
    ClosingAuction,
    AfterHours,
}

/// 交易时段
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TradingSession {
    London,
    NewYork,
    Tokyo,
    Sydney,
    Overlap(Vec<String>),
}

/// 执行参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionParams {
    pub algorithm: String,
    pub max_participation_rate: f64, // 0.0-1.0
    pub price_improvement_target_bps: f64,
    pub max_market_impact_bps: f64,
    pub time_risk_tolerance: f64, // 0.0-1.0
    pub venue_preferences: HashMap<String, f64>,
    pub hidden_order_ratio: f64, // 0.0-1.0
    pub iceberg_size_ratio: f64, // 0.0-1.0
    pub price_limit_offset_bps: f64,
    pub enable_dark_pools: bool,
    pub enable_cross_trading: bool,
    pub parameters: HashMap<String, f64>,
}

/// 执行状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionState {
    pub parent_order_id: String,
    pub total_quantity: f64,
    pub filled_quantity: f64,
    pub remaining_quantity: f64,
    pub average_fill_price: f64,
    pub arrival_price: f64,
    pub current_market_price: f64,
    
    // 子订单状态
    pub active_child_orders: Vec<ChildOrderStatus>,
    pub completed_child_orders: Vec<ChildOrderStatus>,
    
    // 执行指标
    pub elapsed_time: i64, // 秒
    pub remaining_time: i64, // 秒
    pub participation_rate: f64,
    pub slippage_bps: f64,
    pub implementation_shortfall_bps: f64,
    pub market_impact_bps: f64,
    pub timing_cost_bps: f64,
    
    // 风险指标
    pub current_risk_score: f64,
    pub max_risk_score: f64,
    pub venue_concentration: HashMap<String, f64>,
    pub liquidity_consumption: f64,
    
    pub last_updated: DateTime<Utc>,
}

/// 子订单状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChildOrderStatus {
    pub child_order: ChildOrder,
    pub status: ChildOrderState,
    pub filled_quantity: f64,
    pub average_fill_price: f64,
    pub fills: Vec<Fill>,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

/// 子订单状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChildOrderState {
    Pending,
    Submitted,
    Working,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

/// 成交记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub fill_id: String,
    pub child_order_id: String,
    pub quantity: f64,
    pub price: f64,
    pub timestamp: DateTime<Utc>,
    pub venue: String,
    pub counterparty: Option<String>,
    pub liquidity_flag: LiquidityFlag,
    pub commission: f64,
}

/// 流动性标识
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LiquidityFlag {
    Maker,
    Taker,
    Unknown,
}

/// 市场更新
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketUpdate {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub update_type: MarketUpdateType,
    pub price_change_bps: f64,
    pub volume_change_ratio: f64,
    pub volatility_change_ratio: f64,
    pub spread_change_bps: f64,
    pub liquidity_change_ratio: f64,
    pub market_conditions: MarketConditions,
}

/// 市场更新类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MarketUpdateType {
    PriceMove,
    VolumeSpike,
    VolatilityChange,
    SpreadChange,
    LiquidityShock,
    NewsEvent,
    TechnicalLevel,
}

/// 算法统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmStatistics {
    pub algorithm_name: String,
    pub total_executions: u64,
    pub success_rate: f64,
    pub average_slippage_bps: f64,
    pub slippage_std_dev_bps: f64,
    pub average_market_impact_bps: f64,
    pub average_timing_cost_bps: f64,
    pub average_participation_rate: f64,
    pub venue_distribution: HashMap<String, f64>,
    pub time_distribution: HashMap<String, f64>,
    pub performance_by_size: HashMap<String, f64>,
    pub performance_by_urgency: HashMap<String, f64>,
    pub last_updated: DateTime<Utc>,
}

/// 执行算法工厂
pub struct AlgorithmFactory;

impl AlgorithmFactory {
    pub fn create_algorithm(algorithm_type: &str) -> Result<Box<dyn ExecutionAlgorithm>> {
        match algorithm_type.to_uppercase().as_str() {
            "VWAP" => Ok(Box::new(vwap::VwapAlgorithm::new()?)),
            "POV" => Ok(Box::new(pov::PovAlgorithm::new()?)),
            "TWAP" => Ok(Box::new(twap::TwapAlgorithm::new()?)),
            "IS" => Ok(Box::new(is::ImplementationShortfallAlgorithm::new()?)),
            "ICEBERG" => Ok(Box::new(iceberg::IcebergAlgorithm::new()?)),
            _ => Err(anyhow::anyhow!("Unknown algorithm type: {}", algorithm_type)),
        }
    }
    
    pub fn get_available_algorithms() -> Vec<String> {
        vec![
            "VWAP".to_string(),
            "POV".to_string(), 
            "TWAP".to_string(),
            "IS".to_string(),
            "ICEBERG".to_string(),
        ]
    }
}

impl Default for ExecutionParams {
    fn default() -> Self {
        Self {
            algorithm: "VWAP".to_string(),
            max_participation_rate: 0.2,
            price_improvement_target_bps: 0.5,
            max_market_impact_bps: 5.0,
            time_risk_tolerance: 0.5,
            venue_preferences: HashMap::new(),
            hidden_order_ratio: 0.0,
            iceberg_size_ratio: 0.1,
            price_limit_offset_bps: 1.0,
            enable_dark_pools: true,
            enable_cross_trading: true,
            parameters: HashMap::new(),
        }
    }
}

impl Default for AlgorithmStatistics {
    fn default() -> Self {
        Self {
            algorithm_name: String::new(),
            total_executions: 0,
            success_rate: 0.0,
            average_slippage_bps: 0.0,
            slippage_std_dev_bps: 0.0,
            average_market_impact_bps: 0.0,
            average_timing_cost_bps: 0.0,
            average_participation_rate: 0.0,
            venue_distribution: HashMap::new(),
            time_distribution: HashMap::new(),
            performance_by_size: HashMap::new(),
            performance_by_urgency: HashMap::new(),
            last_updated: Utc::now(),
        }
    }
}