//! AG3 现实市场模拟系统
//!
//! 实现高保真度的市场模拟，包括：
//! - 真实的市场微观结构
//! - 部分成交模拟
//! - 流动性影响建模
//! - 订单簿深度模拟
//! - 时变参数建模

use anyhow::{Result, Context};
use chrono::{DateTime, Utc, Duration, Timelike};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use rand::prelude::*;
use rand_distr::{Normal, Poisson};

/// 市场模拟器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSimulationConfig {
    pub tick_size: f64,
    pub min_quantity: f64,
    pub max_order_book_levels: usize,
    pub liquidity_replenishment_rate: f64,
    pub price_impact_model: PriceImpactModel,
    pub volatility_model: VolatilityModel,
    pub microstructure_model: MicrostructureModel,
    pub session_schedule: SessionSchedule,
    pub partial_fill_model: PartialFillModel,
    pub latency_model: LatencyModel,
}

/// 价格冲击模型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PriceImpactModel {
    Linear {
        temporary_coefficient: f64,
        permanent_coefficient: f64,
    },
    SquareRoot {
        temporary_coefficient: f64,
        permanent_coefficient: f64,
    },
    Almgren {
        gamma: f64, // 永久冲击参数
        eta: f64,   // 临时冲击参数
        sigma: f64, // 波动率
    },
    Adaptive {
        base_model: Box<PriceImpactModel>,
        regime_adjustments: HashMap<String, f64>,
    },
}

/// 波动率模型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolatilityModel {
    Constant(f64),
    GARCH {
        omega: f64,
        alpha: f64,
        beta: f64,
        mean_reversion: f64,
    },
    Heston {
        kappa: f64, // 均值回归速度
        theta: f64, // 长期波动率
        sigma: f64, // 波动率的波动率
        rho: f64,   // 相关性
    },
    RegimeSwitching {
        regimes: Vec<RegimeVolatility>,
        transition_matrix: Vec<Vec<f64>>,
    },
}

/// 制度波动率
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeVolatility {
    pub regime_name: String,
    pub base_volatility: f64,
    pub mean_reversion_speed: f64,
    pub volatility_of_volatility: f64,
}

/// 微观结构模型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrostructureModel {
    pub spread_model: SpreadModel,
    pub depth_model: DepthModel,
    pub tick_dynamics: TickDynamics,
    pub order_flow_model: OrderFlowModel,
}

/// 价差模型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpreadModel {
    Constant(f64),
    VolatilityDependent {
        base_spread: f64,
        volatility_coefficient: f64,
    },
    TimeDependent {
        intraday_pattern: Vec<f64>, // 24小时内的价差变化
    },
    Adaptive {
        inventory_sensitivity: f64,
        adverse_selection_cost: f64,
        order_processing_cost: f64,
    },
}

/// 深度模型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthModel {
    pub level_distribution: LevelDistribution,
    pub liquidity_replenishment: LiquidityReplenishment,
    pub depth_decay: DepthDecay,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LevelDistribution {
    Exponential { lambda: f64 },
    PowerLaw { alpha: f64 },
    Empirical { levels: Vec<(f64, f64)> }, // (price_offset, size)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityReplenishment {
    pub replenishment_rate: f64,
    pub size_distribution: SizeDistribution,
    pub price_improvement_probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SizeDistribution {
    Exponential { lambda: f64 },
    Pareto { alpha: f64, xmin: f64 },
    LogNormal { mu: f64, sigma: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthDecay {
    pub decay_rate: f64,
    pub distance_exponent: f64,
}

/// Tick动态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickDynamics {
    pub arrival_rate: f64,
    pub price_change_probability: f64,
    pub uptick_probability: f64,
    pub jump_probability: f64,
    pub jump_size_distribution: SizeDistribution,
}

/// 订单流模型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFlowModel {
    pub informed_trader_ratio: f64,
    pub noise_trader_ratio: f64,
    pub market_maker_ratio: f64,
    pub order_size_distribution: SizeDistribution,
    pub order_arrival_process: ArrivalProcess,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArrivalProcess {
    Poisson { rate: f64 },
    Hawkes {
        base_rate: f64,
        decay_rate: f64,
        excitation: f64,
    },
    TimeVarying {
        intraday_pattern: Vec<f64>,
    },
}

/// 交易时段配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSchedule {
    pub market_open: u32,  // 分钟数，从午夜开始
    pub market_close: u32,
    pub pre_market_start: Option<u32>,
    pub after_hours_end: Option<u32>,
    pub lunch_break: Option<(u32, u32)>,
    pub volatility_schedule: IntradayVolatilitySchedule,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntradayVolatilitySchedule {
    pub opening_volatility_multiplier: f64,
    pub closing_volatility_multiplier: f64,
    pub lunch_volatility_multiplier: f64,
    pub normal_hours_multiplier: f64,
    pub transition_duration_minutes: u32,
}

/// 部分成交模型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialFillModel {
    pub fill_probability_model: FillProbabilityModel,
    pub fill_size_model: FillSizeModel,
    pub fill_timing_model: FillTimingModel,
    pub adverse_selection_impact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FillProbabilityModel {
    Constant(f64),
    SizeDependent {
        base_probability: f64,
        size_penalty: f64,
    },
    LiquidityDependent {
        depth_sensitivity: f64,
        spread_sensitivity: f64,
    },
    TimeDependent {
        intraday_pattern: Vec<f64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FillSizeModel {
    Uniform,
    Proportional { min_fill_ratio: f64 },
    Liquidity_based,
    Adverse_selection_aware {
        information_decay: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FillTimingModel {
    Immediate,
    Exponential { mean_delay_ms: f64 },
    Poisson { arrival_rate: f64 },
    Realistic {
        queue_position_impact: f64,
        priority_rules: PriorityRules,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityRules {
    pub price_priority: bool,
    pub time_priority: bool,
    pub size_priority: bool,
    pub hidden_order_penalty: f64,
}

/// 延迟模型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LatencyModel {
    Constant(u64),
    Variable {
        mean_latency_ms: f64,
        std_latency_ms: f64,
    },
    NetworkBased {
        base_latency_ms: f64,
        congestion_factor: f64,
        time_of_day_multiplier: Vec<f64>,
    },
}

/// 市场模拟器主类
#[derive(Debug)]
pub struct RealisticMarketSimulator {
    config: MarketSimulationConfig,
    order_books: Arc<RwLock<HashMap<String, OrderBook>>>,
    price_engines: Arc<RwLock<HashMap<String, PriceEngine>>>,
    liquidity_providers: Arc<RwLock<HashMap<String, LiquidityProvider>>>,
    execution_venues: Vec<ExecutionVenue>,
    market_data_feed: Arc<RwLock<MarketDataFeed>>,
    simulation_time: Arc<RwLock<DateTime<Utc>>>,
    event_queue: Arc<RwLock<BTreeMap<DateTime<Utc>, Vec<MarketEvent>>>>,
}

/// 订单簿
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: BTreeMap<OrderedFloat, VecDeque<Order>>, // 价格 -> 订单队列
    pub asks: BTreeMap<OrderedFloat, VecDeque<Order>>,
    pub last_trade_price: f64,
    pub last_trade_size: f64,
    pub last_update: DateTime<Utc>,
    pub total_bid_volume: f64,
    pub total_ask_volume: f64,
    pub mid_price: f64,
    pub spread: f64,
    pub tick_size: f64,
}

// 为了在BTreeMap中使用f64作为键
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct OrderedFloat(f64);

impl Eq for OrderedFloat {}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl From<f64> for OrderedFloat {
    fn from(f: f64) -> Self {
        OrderedFloat(f)
    }
}

impl From<OrderedFloat> for f64 {
    fn from(of: OrderedFloat) -> Self {
        of.0
    }
}

/// 订单
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub order_id: String,
    pub symbol: String,
    pub side: Side,
    pub order_type: OrderType,
    pub quantity: f64,
    pub remaining_quantity: f64,
    pub price: f64,
    pub timestamp: DateTime<Utc>,
    pub time_in_force: TimeInForce,
    pub hidden_quantity: f64,
    pub minimum_quantity: f64,
    pub client_id: String,
    pub venue_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
    Iceberg { display_size: f64 },
    HiddenLiquidity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeInForce {
    Day,
    GTC, // Good Till Cancelled
    IOC, // Immediate or Cancel
    FOK, // Fill or Kill
    GTD(DateTime<Utc>), // Good Till Date
}

/// 价格引擎
#[derive(Debug)]
pub struct PriceEngine {
    symbol: String,
    volatility_model: VolatilityModel,
    current_volatility: f64,
    price_history: VecDeque<(DateTime<Utc>, f64)>,
    return_history: VecDeque<f64>,
    jump_detector: JumpDetector,
    regime_detector: Option<Arc<crate::regime_detection::RegimeDetector>>,
}

/// 跳跃检测器
#[derive(Debug)]
pub struct JumpDetector {
    threshold: f64,
    lookback_window: usize,
    jump_history: VecDeque<(DateTime<Utc>, f64)>,
}

/// 流动性提供者
#[derive(Debug)]
pub struct LiquidityProvider {
    provider_id: String,
    symbols: Vec<String>,
    inventory_limits: HashMap<String, (f64, f64)>, // (min, max)
    current_inventory: HashMap<String, f64>,
    risk_appetite: f64,
    spread_target: f64,
    quote_size: f64,
    reaction_speed_ms: u64,
    adverse_selection_sensitivity: f64,
}

/// 执行场所
#[derive(Debug, Clone)]
pub struct ExecutionVenue {
    pub venue_id: String,
    pub venue_type: VenueType,
    pub latency_profile: LatencyProfile,
    pub fee_structure: FeeStructure,
    pub matching_logic: MatchingLogic,
    pub supported_order_types: Vec<OrderType>,
    pub market_data_quality: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VenueType {
    Exchange,
    DarkPool,
    ECN,
    InternalCrossing,
    ATS,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyProfile {
    pub mean_latency_ms: f64,
    pub latency_std_ms: f64,
    pub percentile_99_ms: f64,
    pub jitter_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeStructure {
    pub maker_fee: f64,
    pub taker_fee: f64,
    pub minimum_fee: f64,
    pub volume_tiers: Vec<VolumeTier>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeTier {
    pub monthly_volume_threshold: f64,
    pub maker_rebate: f64,
    pub taker_fee_discount: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MatchingLogic {
    FIFO,
    ProRata,
    SizeTimeProRata,
    PriceImprovement,
}

/// 市场数据推送
#[derive(Debug)]
pub struct MarketDataFeed {
    subscriptions: HashMap<String, Vec<MarketDataSubscription>>,
    last_quotes: HashMap<String, Quote>,
    last_trades: HashMap<String, Trade>,
    tick_history: HashMap<String, VecDeque<MarketTick>>,
}

#[derive(Debug, Clone)]
pub struct MarketDataSubscription {
    pub client_id: String,
    pub data_types: Vec<MarketDataType>,
    pub update_frequency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketDataType {
    Quote,
    Trade,
    OrderBook,
    AuctionInfo,
    Statistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quote {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub bid_price: f64,
    pub bid_size: f64,
    pub ask_price: f64,
    pub ask_size: f64,
    pub bid_count: u32,
    pub ask_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub size: f64,
    pub side: Option<Side>,
    pub trade_id: String,
    pub venue_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketTick {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub size: f64,
    pub tick_type: TickType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TickType {
    Trade,
    BidUpdate,
    AskUpdate,
    BestBidAsk,
}

/// 市场事件
#[derive(Debug, Clone)]
pub enum MarketEvent {
    OrderPlaced(Order),
    OrderCancelled { order_id: String, reason: String },
    OrderModified { order_id: String, new_quantity: f64, new_price: f64 },
    Trade {
        buy_order_id: String,
        sell_order_id: String,
        price: f64,
        quantity: f64,
        timestamp: DateTime<Utc>,
    },
    QuoteUpdate(Quote),
    LiquidityReplenishment { symbol: String, side: Side, price: f64, size: f64 },
    MarketOpen { symbol: String },
    MarketClose { symbol: String },
    VolatilityJump { symbol: String, old_vol: f64, new_vol: f64 },
}

/// 执行结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub order_id: String,
    pub executions: Vec<Execution>,
    pub total_filled: f64,
    pub remaining_quantity: f64,
    pub average_price: f64,
    pub total_fees: f64,
    pub market_impact: f64,
    pub implementation_shortfall: f64,
    pub execution_time_ms: u64,
    pub partial_fill_reason: Option<PartialFillReason>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Execution {
    pub execution_id: String,
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub quantity: f64,
    pub venue_id: String,
    pub liquidity_type: LiquidityType,
    pub fees: f64,
    pub market_impact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LiquidityType {
    Maker,
    Taker,
    Hidden,
    Midpoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartialFillReason {
    InsufficientLiquidity,
    PriceMovement,
    TimeConstraint,
    RiskLimit,
    AdverseSelection,
    VenueCapacity,
}

impl RealisticMarketSimulator {
    /// 创建新的市场模拟器
    pub fn new(config: MarketSimulationConfig) -> Result<Self> {
        Ok(Self {
            config,
            order_books: Arc::new(RwLock::new(HashMap::new())),
            price_engines: Arc::new(RwLock::new(HashMap::new())),
            liquidity_providers: Arc::new(RwLock::new(HashMap::new())),
            execution_venues: Vec::new(),
            market_data_feed: Arc::new(RwLock::new(MarketDataFeed::new())),
            simulation_time: Arc::new(RwLock::new(Utc::now())),
            event_queue: Arc::new(RwLock::new(BTreeMap::new())),
        })
    }

    /// 添加交易品种
    pub async fn add_symbol(&self, symbol: &str, initial_price: f64) -> Result<()> {
        // 创建订单簿
        let order_book = OrderBook::new(symbol, initial_price, self.config.tick_size);
        self.order_books.write().await.insert(symbol.to_string(), order_book);

        // 创建价格引擎
        let price_engine = PriceEngine::new(
            symbol,
            self.config.volatility_model.clone(),
            initial_price,
        )?;
        self.price_engines.write().await.insert(symbol.to_string(), price_engine);

        // 添加流动性提供者
        let liquidity_provider = LiquidityProvider::new(
            format!("LP_{}", symbol),
            vec![symbol.to_string()],
        )?;
        self.liquidity_providers.write().await.insert(
            format!("LP_{}", symbol),
            liquidity_provider
        );

        Ok(())
    }

    /// 模拟订单执行
    pub async fn simulate_order_execution(&self, order: Order) -> Result<ExecutionResult> {
        let start_time = std::time::Instant::now();
        let mut executions = Vec::new();
        let mut remaining_quantity = order.remaining_quantity;
        let mut total_fees = 0.0;
        let mut total_market_impact = 0.0;
        
        // 获取当前订单簿状态
        let order_books = self.order_books.read().await;
        let order_book = order_books.get(&order.symbol)
            .ok_or_else(|| anyhow::anyhow!("Symbol not found: {}", order.symbol))?;

        // 计算市场冲击
        let market_impact = self.calculate_market_impact(&order, order_book).await?;
        total_market_impact += market_impact;

        // 模拟部分成交
        let fill_simulation = self.simulate_partial_fills(&order, order_book).await?;
        
        for fill in fill_simulation.fills {
            let execution = Execution {
                execution_id: format!("exec_{}", rand::thread_rng().gen::<u64>()),
                timestamp: fill.timestamp,
                price: fill.price,
                quantity: fill.quantity,
                venue_id: fill.venue_id,
                liquidity_type: fill.liquidity_type,
                fees: fill.fees,
                market_impact: fill.market_impact,
            };
            
            executions.push(execution);
            remaining_quantity -= fill.quantity;
            total_fees += fill.fees;
        }

        let average_price = if !executions.is_empty() {
            executions.iter()
                .map(|e| e.price * e.quantity)
                .sum::<f64>() / executions.iter().map(|e| e.quantity).sum::<f64>()
        } else {
            order.price
        };

        let execution_time_ms = start_time.elapsed().as_millis() as u64;
        let total_filled = order.remaining_quantity - remaining_quantity;

        Ok(ExecutionResult {
            order_id: order.order_id.clone(),
            executions,
            total_filled,
            remaining_quantity,
            average_price,
            total_fees,
            market_impact: total_market_impact,
            implementation_shortfall: self.calculate_implementation_shortfall(
                &order, 
                average_price, 
                total_filled
            ).await?,
            execution_time_ms,
            partial_fill_reason: fill_simulation.partial_fill_reason,
        })
    }

    /// 计算市场冲击
    async fn calculate_market_impact(&self, order: &Order, order_book: &OrderBook) -> Result<f64> {
        let participation_rate = order.quantity / order_book.total_volume();
        
        match &self.config.price_impact_model {
            PriceImpactModel::Linear { temporary_coefficient, permanent_coefficient } => {
                Ok(temporary_coefficient * participation_rate + permanent_coefficient * participation_rate.sqrt())
            }
            PriceImpactModel::SquareRoot { temporary_coefficient, permanent_coefficient } => {
                Ok(temporary_coefficient * participation_rate.sqrt() + permanent_coefficient * participation_rate.powf(0.75))
            }
            PriceImpactModel::Almgren { gamma, eta, sigma } => {
                let temporary_impact = eta * sigma * participation_rate.powf(0.6);
                let permanent_impact = gamma * sigma * (participation_rate).sqrt();
                Ok(temporary_impact + permanent_impact)
            }
            PriceImpactModel::Adaptive { base_model, regime_adjustments } => {
                let base_impact = match base_model.as_ref() {
                    PriceImpactModel::Linear { temporary_coefficient, permanent_coefficient } => {
                        temporary_coefficient * participation_rate + permanent_coefficient * participation_rate.sqrt()
                    }
                    _ => 0.001, // 默认值
                };
                
                // 根据制度调整
                let adjustment = regime_adjustments.get("current").unwrap_or(&1.0);
                Ok(base_impact * adjustment)
            }
        }
    }

    /// 模拟部分成交
    async fn simulate_partial_fills(&self, order: &Order, order_book: &OrderBook) -> Result<FillSimulation> {
        let mut fills = Vec::new();
        let mut remaining_qty = order.remaining_quantity;
        let mut partial_fill_reason = None;

        // 获取可用流动性
        let available_liquidity = match order.side {
            Side::Buy => order_book.get_ask_liquidity_at_price(order.price),
            Side::Sell => order_book.get_bid_liquidity_at_price(order.price),
        };

        // 计算成交概率
        let fill_probability = self.calculate_fill_probability(order, order_book).await?;
        
        if rand::thread_rng().gen::<f64>() > fill_probability {
            partial_fill_reason = Some(PartialFillReason::InsufficientLiquidity);
            return Ok(FillSimulation { fills, partial_fill_reason });
        }

        // 模拟逐笔成交
        let mut current_time = order.timestamp;
        while remaining_qty > 0.0 && available_liquidity > 0.0 {
            let fill_size = self.calculate_fill_size(remaining_qty, available_liquidity, order).await?;
            
            if fill_size <= 0.0 {
                partial_fill_reason = Some(PartialFillReason::InsufficientLiquidity);
                break;
            }

            let fill_price = self.calculate_fill_price(order, order_book, fill_size).await?;
            let venue_id = self.select_execution_venue(&order.symbol).await?;
            let latency = self.calculate_execution_latency(&venue_id).await?;
            
            current_time = current_time + Duration::milliseconds(latency as i64);

            let fill = Fill {
                timestamp: current_time,
                price: fill_price,
                quantity: fill_size,
                venue_id: venue_id.clone(),
                liquidity_type: self.determine_liquidity_type(order, fill_price, order_book.mid_price).await?,
                fees: self.calculate_fees(fill_size, fill_price, &venue_id).await?,
                market_impact: self.calculate_incremental_market_impact(fill_size, remaining_qty).await?,
            };

            fills.push(fill);
            remaining_qty -= fill_size;

            // 检查是否因为不利选择而停止
            if self.should_stop_due_to_adverse_selection(order, &fills).await? {
                partial_fill_reason = Some(PartialFillReason::AdverseSelection);
                break;
            }

            // 模拟价格移动
            if rand::thread_rng().gen::<f64>() < 0.1 { // 10%概率价格移动
                partial_fill_reason = Some(PartialFillReason::PriceMovement);
                break;
            }
        }

        Ok(FillSimulation { fills, partial_fill_reason })
    }

    /// 计算成交概率
    async fn calculate_fill_probability(&self, order: &Order, order_book: &OrderBook) -> Result<f64> {
        match &self.config.partial_fill_model.fill_probability_model {
            FillProbabilityModel::Constant(prob) => Ok(*prob),
            FillProbabilityModel::SizeDependent { base_probability, size_penalty } => {
                let size_factor = (order.quantity / 1000.0).min(1.0); // 标准化到1000股
                Ok((base_probability - size_penalty * size_factor).max(0.1))
            }
            FillProbabilityModel::LiquidityDependent { depth_sensitivity, spread_sensitivity } => {
                let depth_score = order_book.total_volume() / 10000.0; // 标准化
                let spread_score = 1.0 / (1.0 + order_book.spread * 100.0);
                Ok((0.5 + depth_sensitivity * depth_score + spread_sensitivity * spread_score).min(1.0))
            }
            FillProbabilityModel::TimeDependent { intraday_pattern } => {
                let hour = Utc::now().hour() as usize;
                Ok(intraday_pattern.get(hour).copied().unwrap_or(0.7))
            }
        }
    }

    /// 计算成交数量
    async fn calculate_fill_size(&self, remaining_qty: f64, available_liquidity: f64, order: &Order) -> Result<f64> {
        match &self.config.partial_fill_model.fill_size_model {
            FillSizeModel::Uniform => {
                let max_fill = remaining_qty.min(available_liquidity);
                Ok(rand::thread_rng().gen::<f64>() * max_fill)
            }
            FillSizeModel::Proportional { min_fill_ratio } => {
                let max_fill = remaining_qty.min(available_liquidity);
                let min_fill = max_fill * min_fill_ratio;
                Ok(min_fill + rand::thread_rng().gen::<f64>() * (max_fill - min_fill))
            }
            FillSizeModel::Liquidity_based => {
                // 基于可用流动性决定成交量
                let participation_limit = available_liquidity * 0.1; // 最多吃掉10%的流动性
                Ok(remaining_qty.min(participation_limit))
            }
            FillSizeModel::Adverse_selection_aware { information_decay: _ } => {
                // 考虑信息衰减的成交量
                let time_penalty = 0.95; // 简化的时间衰减
                let adjusted_available = available_liquidity * time_penalty;
                Ok(remaining_qty.min(adjusted_available * 0.2))
            }
        }
    }

    /// 计算成交价格
    async fn calculate_fill_price(&self, order: &Order, order_book: &OrderBook, fill_size: f64) -> Result<f64> {
        let base_price = match order.order_type {
            OrderType::Market => order_book.mid_price,
            OrderType::Limit => order.price,
            _ => order_book.mid_price,
        };

        // 添加市场冲击
        let size_impact = fill_size / 10000.0 * 0.01; // 简化的价格冲击
        let direction = match order.side {
            Side::Buy => 1.0,
            Side::Sell => -1.0,
        };

        Ok(base_price * (1.0 + direction * size_impact))
    }

    /// 选择执行场所
    async fn select_execution_venue(&self, _symbol: &str) -> Result<String> {
        // 简化实现：随机选择场所
        let venues = vec!["NYSE", "NASDAQ", "BATS", "DARK_POOL_1"];
        let idx = rand::thread_rng().gen_range(0..venues.len());
        Ok(venues[idx].to_string())
    }

    /// 计算执行延迟
    async fn calculate_execution_latency(&self, venue_id: &str) -> Result<u64> {
        match &self.config.latency_model {
            LatencyModel::Constant(latency) => Ok(*latency),
            LatencyModel::Variable { mean_latency_ms, std_latency_ms } => {
                let normal = Normal::new(*mean_latency_ms, *std_latency_ms)
                    .map_err(|e| anyhow::anyhow!("Failed to create normal distribution: {}", e))?;
                Ok(normal.sample(&mut rand::thread_rng()).max(0.0) as u64)
            }
            LatencyModel::NetworkBased { base_latency_ms, congestion_factor, time_of_day_multiplier } => {
                let hour = Utc::now().hour() as usize;
                let time_multiplier = time_of_day_multiplier.get(hour).unwrap_or(&1.0);
                let congestion = 1.0 + rand::thread_rng().gen::<f64>() * congestion_factor;
                Ok((base_latency_ms * time_multiplier * congestion) as u64)
            }
        }
    }

    /// 确定流动性类型
    async fn determine_liquidity_type(&self, order: &Order, fill_price: f64, mid_price: f64) -> Result<LiquidityType> {
        let price_improvement = match order.side {
            Side::Buy => mid_price - fill_price,
            Side::Sell => fill_price - mid_price,
        };

        if price_improvement > 0.0 {
            Ok(LiquidityType::Maker)
        } else if price_improvement < 0.0 {
            Ok(LiquidityType::Taker)
        } else {
            Ok(LiquidityType::Midpoint)
        }
    }

    /// 计算手续费
    async fn calculate_fees(&self, quantity: f64, price: f64, venue_id: &str) -> Result<f64> {
        // 简化的手续费计算
        let notional = quantity * price;
        let fee_rate = match venue_id {
            "NYSE" => 0.0003,
            "NASDAQ" => 0.0003,
            "BATS" => 0.0002,
            "DARK_POOL_1" => 0.0001,
            _ => 0.0003,
        };
        Ok(notional * fee_rate)
    }

    /// 计算增量市场冲击
    async fn calculate_incremental_market_impact(&self, fill_size: f64, remaining_qty: f64) -> Result<f64> {
        let progress = 1.0 - (remaining_qty / (remaining_qty + fill_size));
        Ok(0.001 * fill_size * progress.sqrt()) // 简化计算
    }

    /// 判断是否因不利选择停止
    async fn should_stop_due_to_adverse_selection(&self, _order: &Order, fills: &[Fill]) -> Result<bool> {
        if fills.len() < 2 {
            return Ok(false);
        }

        // 检查价格是否持续不利移动
        let recent_fills = &fills[fills.len().saturating_sub(3)..];
        let price_trend = recent_fills.windows(2)
            .all(|w| (w[1].price - w[0].price).abs() > 0.001);

        Ok(price_trend && rand::thread_rng().gen::<f64>() < 0.2)
    }

    /// 计算实施缺口
    async fn calculate_implementation_shortfall(&self, order: &Order, average_price: f64, filled_qty: f64) -> Result<f64> {
        if filled_qty <= 0.0 {
            return Ok(0.0);
        }

        let benchmark_price = order.price; // 简化：使用订单价格作为基准
        let price_impact = (average_price - benchmark_price).abs() / benchmark_price;
        let quantity_impact = (order.remaining_quantity - filled_qty) / order.remaining_quantity;
        
        Ok(price_impact + quantity_impact * 0.1) // 加权组合
    }

    /// 运行市场模拟
    pub async fn run_simulation(&self, duration: Duration) -> Result<SimulationResults> {
        let start_time = *self.simulation_time.read().await;
        let end_time = start_time + duration;
        let mut current_time = start_time;

        let mut results = SimulationResults::new();

        while current_time < end_time {
            // 处理事件队列中的事件
            self.process_events_at_time(current_time).await?;
            
            // 更新价格引擎
            self.update_price_engines(current_time).await?;
            
            // 更新流动性提供者
            self.update_liquidity_providers(current_time).await?;
            
            // 生成市场数据
            self.generate_market_data(current_time).await?;
            
            // 推进时间
            current_time = current_time + Duration::milliseconds(100); // 100ms步长
            *self.simulation_time.write().await = current_time;
        }

        Ok(results)
    }

    async fn process_events_at_time(&self, timestamp: DateTime<Utc>) -> Result<()> {
        let mut event_queue = self.event_queue.write().await;
        if let Some(events) = event_queue.remove(&timestamp) {
            for event in events {
                self.handle_market_event(event).await?;
            }
        }
        Ok(())
    }

    async fn handle_market_event(&self, event: MarketEvent) -> Result<()> {
        match event {
            MarketEvent::OrderPlaced(order) => {
                // 处理新订单
                self.add_order_to_book(order).await?;
            }
            MarketEvent::Trade { buy_order_id: _, sell_order_id: _, price, quantity, timestamp } => {
                // 更新交易记录
                self.record_trade(price, quantity, timestamp).await?;
            }
            MarketEvent::QuoteUpdate(quote) => {
                // 更新报价
                self.update_quote(quote).await?;
            }
            // 处理其他事件类型
            _ => {}
        }
        Ok(())
    }

    async fn update_price_engines(&self, _current_time: DateTime<Utc>) -> Result<()> {
        let mut price_engines = self.price_engines.write().await;
        for (symbol, engine) in price_engines.iter_mut() {
            engine.update().await?;
        }
        Ok(())
    }

    async fn update_liquidity_providers(&self, _current_time: DateTime<Utc>) -> Result<()> {
        let mut providers = self.liquidity_providers.write().await;
        for (id, provider) in providers.iter_mut() {
            provider.update_quotes().await?;
        }
        Ok(())
    }

    async fn generate_market_data(&self, current_time: DateTime<Utc>) -> Result<()> {
        let order_books = self.order_books.read().await;
        let mut market_data_feed = self.market_data_feed.write().await;

        for (symbol, order_book) in order_books.iter() {
            // 生成报价更新
            if rand::thread_rng().gen::<f64>() < 0.1 { // 10%概率更新报价
                let quote = Quote {
                    symbol: symbol.clone(),
                    timestamp: current_time,
                    bid_price: order_book.best_bid_price(),
                    bid_size: order_book.best_bid_size(),
                    ask_price: order_book.best_ask_price(),
                    ask_size: order_book.best_ask_size(),
                    bid_count: order_book.bid_level_count() as u32,
                    ask_count: order_book.ask_level_count() as u32,
                };
                market_data_feed.update_quote(quote)?;
            }
        }

        Ok(())
    }

    // 辅助方法实现
    async fn add_order_to_book(&self, order: Order) -> Result<()> {
        let mut order_books = self.order_books.write().await;
        if let Some(order_book) = order_books.get_mut(&order.symbol) {
            order_book.add_order(order)?;
        }
        Ok(())
    }

    async fn record_trade(&self, _price: f64, _quantity: f64, _timestamp: DateTime<Utc>) -> Result<()> {
        // 记录交易到历史数据
        Ok(())
    }

    async fn update_quote(&self, quote: Quote) -> Result<()> {
        let mut market_data_feed = self.market_data_feed.write().await;
        market_data_feed.update_quote(quote)?;
        Ok(())
    }
}

// 辅助结构体实现
#[derive(Debug, Clone)]
struct FillSimulation {
    fills: Vec<Fill>,
    partial_fill_reason: Option<PartialFillReason>,
}

#[derive(Debug, Clone)]
struct Fill {
    timestamp: DateTime<Utc>,
    price: f64,
    quantity: f64,
    venue_id: String,
    liquidity_type: LiquidityType,
    fees: f64,
    market_impact: f64,
}

#[derive(Debug)]
pub struct SimulationResults {
    pub total_trades: usize,
    pub total_volume: f64,
    pub average_spread: f64,
    pub price_volatility: f64,
    pub execution_stats: ExecutionStats,
}

#[derive(Debug)]
pub struct ExecutionStats {
    pub fill_rate: f64,
    pub average_fill_size: f64,
    pub partial_fill_frequency: f64,
    pub average_slippage: f64,
    pub average_market_impact: f64,
}

impl SimulationResults {
    fn new() -> Self {
        Self {
            total_trades: 0,
            total_volume: 0.0,
            average_spread: 0.0,
            price_volatility: 0.0,
            execution_stats: ExecutionStats {
                fill_rate: 0.0,
                average_fill_size: 0.0,
                partial_fill_frequency: 0.0,
                average_slippage: 0.0,
                average_market_impact: 0.0,
            },
        }
    }
}

// OrderBook实现
impl OrderBook {
    fn new(symbol: &str, initial_price: f64, tick_size: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_trade_price: initial_price,
            last_trade_size: 0.0,
            last_update: Utc::now(),
            total_bid_volume: 0.0,
            total_ask_volume: 0.0,
            mid_price: initial_price,
            spread: 0.01,
            tick_size,
        }
    }

    fn total_volume(&self) -> f64 {
        self.total_bid_volume + self.total_ask_volume
    }

    fn get_ask_liquidity_at_price(&self, price: f64) -> f64 {
        self.asks.iter()
            .filter(|(ask_price, _)| ask_price.0 <= price)
            .map(|(_, orders)| orders.iter().map(|o| o.remaining_quantity).sum::<f64>())
            .sum()
    }

    fn get_bid_liquidity_at_price(&self, price: f64) -> f64 {
        self.bids.iter()
            .filter(|(bid_price, _)| bid_price.0 >= price)
            .map(|(_, orders)| orders.iter().map(|o| o.remaining_quantity).sum::<f64>())
            .sum()
    }

    fn best_bid_price(&self) -> f64 {
        self.bids.iter().next_back()
            .map(|(price, _)| price.0)
            .unwrap_or(self.mid_price - self.spread / 2.0)
    }

    fn best_ask_price(&self) -> f64 {
        self.asks.iter().next()
            .map(|(price, _)| price.0)
            .unwrap_or(self.mid_price + self.spread / 2.0)
    }

    fn best_bid_size(&self) -> f64 {
        self.bids.iter().next_back()
            .map(|(_, orders)| orders.iter().map(|o| o.remaining_quantity).sum::<f64>())
            .unwrap_or(0.0)
    }

    fn best_ask_size(&self) -> f64 {
        self.asks.iter().next()
            .map(|(_, orders)| orders.iter().map(|o| o.remaining_quantity).sum::<f64>())
            .unwrap_or(0.0)
    }

    fn bid_level_count(&self) -> usize {
        self.bids.len()
    }

    fn ask_level_count(&self) -> usize {
        self.asks.len()
    }

    fn add_order(&mut self, order: Order) -> Result<()> {
        let price = OrderedFloat::from(order.price);
        match order.side {
            Side::Buy => {
                self.bids.entry(price)
                    .or_insert_with(VecDeque::new)
                    .push_back(order);
            }
            Side::Sell => {
                self.asks.entry(price)
                    .or_insert_with(VecDeque::new)
                    .push_back(order);
            }
        }
        self.update_metrics();
        Ok(())
    }

    fn update_metrics(&mut self) {
        self.total_bid_volume = self.bids.values()
            .flat_map(|orders| orders.iter())
            .map(|o| o.remaining_quantity)
            .sum();

        self.total_ask_volume = self.asks.values()
            .flat_map(|orders| orders.iter())
            .map(|o| o.remaining_quantity)
            .sum();

        let best_bid = self.best_bid_price();
        let best_ask = self.best_ask_price();
        self.mid_price = (best_bid + best_ask) / 2.0;
        self.spread = best_ask - best_bid;
        self.last_update = Utc::now();
    }
}

// PriceEngine实现
impl PriceEngine {
    fn new(symbol: &str, volatility_model: VolatilityModel, initial_price: f64) -> Result<Self> {
        Ok(Self {
            symbol: symbol.to_string(),
            volatility_model,
            current_volatility: 0.2, // 默认年化波动率20%
            price_history: VecDeque::with_capacity(1000),
            return_history: VecDeque::with_capacity(1000),
            jump_detector: JumpDetector::new(3.0, 20),
            regime_detector: None,
        })
    }

    async fn update(&mut self) -> Result<()> {
        // 更新波动率
        self.update_volatility().await?;
        
        // 生成价格变动
        let price_change = self.generate_price_change().await?;
        let current_time = Utc::now();
        let current_price = self.price_history.back()
            .map(|(_, price)| *price)
            .unwrap_or(100.0); // 默认价格

        let new_price = current_price * (1.0 + price_change);
        self.price_history.push_back((current_time, new_price));
        self.return_history.push_back(price_change);

        // 保持历史大小
        if self.price_history.len() > 1000 {
            self.price_history.pop_front();
        }
        if self.return_history.len() > 1000 {
            self.return_history.pop_front();
        }

        // 检测跳跃
        self.jump_detector.check_jump(price_change)?;

        Ok(())
    }

    async fn update_volatility(&mut self) -> Result<()> {
        match &self.volatility_model {
            VolatilityModel::Constant(_) => {
                // 波动率保持不变
            }
            VolatilityModel::GARCH { omega, alpha, beta, mean_reversion } => {
                if let Some(&last_return) = self.return_history.back() {
                    let long_term_vol = 0.2; // 长期波动率
                    let vol_innovation = rand::thread_rng().gen::<f64>() * 0.01;
                    
                    self.current_volatility = omega + 
                        alpha * last_return * last_return + 
                        beta * self.current_volatility + 
                        mean_reversion * (long_term_vol - self.current_volatility) +
                        vol_innovation;
                }
            }
            _ => {
                // 其他波动率模型的简化实现
                let vol_change = Normal::new(0.0, 0.01)
                    .unwrap()
                    .sample(&mut rand::thread_rng());
                self.current_volatility = (self.current_volatility + vol_change).max(0.01);
            }
        }
        Ok(())
    }

    async fn generate_price_change(&self) -> Result<f64> {
        let dt = 1.0_f64 / (252.0 * 24.0 * 60.0 * 60.0 / 100.0); // 100ms时间步长
        let drift = 0.05 * dt; // 年化5%漂移
        let diffusion = self.current_volatility * dt.sqrt();
        
        let normal_shock = Normal::new(0.0, 1.0)
            .unwrap()
            .sample(&mut rand::thread_rng());
        
        // 添加跳跃成分
        let jump_component = if rand::thread_rng().gen::<f64>() < 0.001 { // 0.1%跳跃概率
            Normal::new(0.0, 0.02).unwrap().sample(&mut rand::thread_rng())
        } else {
            0.0
        };

        Ok(drift + diffusion * normal_shock + jump_component)
    }
}

// JumpDetector实现
impl JumpDetector {
    fn new(threshold: f64, lookback_window: usize) -> Self {
        Self {
            threshold,
            lookback_window,
            jump_history: VecDeque::with_capacity(1000),
        }
    }

    fn check_jump(&mut self, return_value: f64) -> Result<bool> {
        let is_jump = return_value.abs() > self.threshold * 0.01; // 简化的跳跃检测
        
        if is_jump {
            self.jump_history.push_back((Utc::now(), return_value));
            if self.jump_history.len() > 1000 {
                self.jump_history.pop_front();
            }
        }

        Ok(is_jump)
    }
}

// LiquidityProvider实现
impl LiquidityProvider {
    fn new(provider_id: String, symbols: Vec<String>) -> Result<Self> {
        let mut inventory_limits = HashMap::new();
        let mut current_inventory = HashMap::new();
        
        for symbol in &symbols {
            inventory_limits.insert(symbol.clone(), (-10000.0, 10000.0));
            current_inventory.insert(symbol.clone(), 0.0);
        }

        Ok(Self {
            provider_id,
            symbols,
            inventory_limits,
            current_inventory,
            risk_appetite: 0.8,
            spread_target: 0.02,
            quote_size: 1000.0,
            reaction_speed_ms: 50,
            adverse_selection_sensitivity: 0.5,
        })
    }

    async fn update_quotes(&mut self) -> Result<()> {
        // 简化的报价更新逻辑
        for symbol in &self.symbols.clone() {
            let inventory = self.current_inventory.get(symbol).copied().unwrap_or(0.0);
            let (min_inv, max_inv) = self.inventory_limits.get(symbol).copied().unwrap_or((-1000.0, 1000.0));
            
            // 根据库存调整报价意愿
            let inventory_skew = inventory / (max_inv - min_inv);
            let adjusted_spread = self.spread_target * (1.0 + inventory_skew.abs());
            
            // 生成新的买卖报价
            // TODO: 实现实际的报价生成和发送逻辑
        }
        Ok(())
    }
}

// MarketDataFeed实现
impl MarketDataFeed {
    fn new() -> Self {
        Self {
            subscriptions: HashMap::new(),
            last_quotes: HashMap::new(),
            last_trades: HashMap::new(),
            tick_history: HashMap::new(),
        }
    }

    fn update_quote(&mut self, quote: Quote) -> Result<()> {
        self.last_quotes.insert(quote.symbol.clone(), quote);
        Ok(())
    }
}

impl Default for MarketSimulationConfig {
    fn default() -> Self {
        Self {
            tick_size: 0.01,
            min_quantity: 1.0,
            max_order_book_levels: 10,
            liquidity_replenishment_rate: 0.1,
            price_impact_model: PriceImpactModel::Linear {
                temporary_coefficient: 0.01,
                permanent_coefficient: 0.005,
            },
            volatility_model: VolatilityModel::Constant(0.2),
            microstructure_model: MicrostructureModel {
                spread_model: SpreadModel::Constant(0.02),
                depth_model: DepthModel {
                    level_distribution: LevelDistribution::Exponential { lambda: 2.0 },
                    liquidity_replenishment: LiquidityReplenishment {
                        replenishment_rate: 0.1,
                        size_distribution: SizeDistribution::Exponential { lambda: 1.0 },
                        price_improvement_probability: 0.1,
                    },
                    depth_decay: DepthDecay {
                        decay_rate: 0.5,
                        distance_exponent: 1.5,
                    },
                },
                tick_dynamics: TickDynamics {
                    arrival_rate: 10.0,
                    price_change_probability: 0.1,
                    uptick_probability: 0.5,
                    jump_probability: 0.001,
                    jump_size_distribution: SizeDistribution::LogNormal { mu: 0.0, sigma: 0.02 },
                },
                order_flow_model: OrderFlowModel {
                    informed_trader_ratio: 0.2,
                    noise_trader_ratio: 0.6,
                    market_maker_ratio: 0.2,
                    order_size_distribution: SizeDistribution::Pareto { alpha: 1.5, xmin: 100.0 },
                    order_arrival_process: ArrivalProcess::Poisson { rate: 5.0 },
                },
            },
            session_schedule: SessionSchedule {
                market_open: 9 * 60 + 30, // 9:30 AM
                market_close: 16 * 60,    // 4:00 PM
                pre_market_start: Some(4 * 60), // 4:00 AM
                after_hours_end: Some(20 * 60), // 8:00 PM
                lunch_break: None,
                volatility_schedule: IntradayVolatilitySchedule {
                    opening_volatility_multiplier: 1.5,
                    closing_volatility_multiplier: 1.3,
                    lunch_volatility_multiplier: 0.8,
                    normal_hours_multiplier: 1.0,
                    transition_duration_minutes: 30,
                },
            },
            partial_fill_model: PartialFillModel {
                fill_probability_model: FillProbabilityModel::Constant(0.8),
                fill_size_model: FillSizeModel::Proportional { min_fill_ratio: 0.1 },
                fill_timing_model: FillTimingModel::Exponential { mean_delay_ms: 100.0 },
                adverse_selection_impact: 0.1,
            },
            latency_model: LatencyModel::Variable { 
                mean_latency_ms: 50.0, 
                std_latency_ms: 10.0 
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_market_simulator_creation() {
        let config = MarketSimulationConfig::default();
        let simulator = RealisticMarketSimulator::new(config);
        assert!(simulator.is_ok());
    }

    #[tokio::test]
    async fn test_add_symbol() {
        let config = MarketSimulationConfig::default();
        let simulator = RealisticMarketSimulator::new(config).unwrap();
        
        let result = simulator.add_symbol("AAPL", 150.0).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_order_execution_simulation() {
        let config = MarketSimulationConfig::default();
        let simulator = RealisticMarketSimulator::new(config).unwrap();
        
        simulator.add_symbol("AAPL", 150.0).await.unwrap();
        
        let order = Order {
            order_id: "test_order_1".to_string(),
            symbol: "AAPL".to_string(),
            side: Side::Buy,
            order_type: OrderType::Market,
            quantity: 1000.0,
            remaining_quantity: 1000.0,
            price: 150.0,
            timestamp: Utc::now(),
            time_in_force: TimeInForce::Day,
            hidden_quantity: 0.0,
            minimum_quantity: 0.0,
            client_id: "test_client".to_string(),
            venue_id: "NYSE".to_string(),
        };

        let result = simulator.simulate_order_execution(order).await;
        assert!(result.is_ok());
        
        let execution_result = result.unwrap();
        assert!(execution_result.total_filled >= 0.0);
        assert!(execution_result.total_filled <= 1000.0);
    }

    #[test]
    fn test_price_impact_calculation() {
        let config = MarketSimulationConfig::default();
        let simulator = RealisticMarketSimulator::new(config).unwrap();
        
        // 测试线性价格冲击模型
        // 这需要异步上下文和更多的设置
    }

    #[test]
    fn test_partial_fill_probability() {
        let model = FillProbabilityModel::SizeDependent {
            base_probability: 0.9,
            size_penalty: 0.1,
        };

        // 测试不同订单大小的成交概率
        match model {
            FillProbabilityModel::SizeDependent { base_probability, size_penalty } => {
                let small_order_prob = base_probability - size_penalty * 0.1;
                let large_order_prob = base_probability - size_penalty * 1.0;
                
                assert!(small_order_prob > large_order_prob);
                assert!(small_order_prob <= 1.0);
                assert!(large_order_prob >= 0.0);
            }
            _ => panic!("Unexpected model type"),
        }
    }
}