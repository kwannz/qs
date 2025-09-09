//! AG3智能路由计分系统
//! 
//! 实现了多维度、自适应的交易路由评分和优化系统，包括：
//! - 实时流动性分析
//! - 执行成本预估
//! - 市场冲击建模
//! - 延迟优化
//! - 动态路由权重调整

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::convert::TryInto;
use std::sync::{Arc, RwLock};
use tokio::sync::RwLock as TokioRwLock;

/// 智能路由评分引擎
#[derive(Clone)]
pub struct SmartRoutingScorer {
    config: RoutingConfig,
    venue_monitors: Arc<TokioRwLock<HashMap<String, VenueMonitor>>>,
    score_calculator: Arc<ScoreCalculator>,
    cost_estimator: Arc<CostEstimator>,
    liquidity_analyzer: Arc<LiquidityAnalyzer>,
    latency_monitor: Arc<LatencyMonitor>,
    routing_optimizer: Arc<RoutingOptimizer>,
    performance_tracker: Arc<RwLock<PerformanceTracker>>,
    adaptive_weights: Arc<RwLock<AdaptiveWeights>>,
}

/// 路由配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// 支持的交易所
    pub supported_venues: Vec<VenueConfig>,
    /// 评分权重
    pub score_weights: ScoreWeights,
    /// 更新频率（毫秒）
    pub update_frequency_ms: u64,
    /// 历史数据窗口
    pub lookback_window_minutes: u32,
    /// 最小流动性阈值
    pub min_liquidity_threshold: f64,
    /// 最大延迟容忍度（毫秒）
    pub max_latency_tolerance_ms: u32,
    /// 启用自适应权重
    pub enable_adaptive_weights: bool,
    /// 启用成本预估
    pub enable_cost_prediction: bool,
    /// 启用市场冲击建模
    pub enable_market_impact_modeling: bool,
}

/// 交易所配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueConfig {
    pub venue_id: String,
    pub venue_name: String,
    pub venue_type: VenueType,
    pub supported_symbols: Vec<String>,
    pub api_endpoints: ApiEndpoints,
    pub fee_structure: FeeStructure,
    pub trading_hours: TradingHours,
    pub connectivity: ConnectivityConfig,
    pub constraints: VenueConstraints,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VenueType {
    Exchange,         // 传统交易所
    ECN,             // 电子通信网络
    DarkPool,        // 暗池
    CrossingNetwork, // 撮合网络
    Market,          // 做市商
    ATS,             // 另类交易系统
    Dex,             // 去中心化交易所
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiEndpoints {
    pub market_data: String,
    pub order_entry: String,
    pub order_status: String,
    pub fills: String,
    pub websocket: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeStructure {
    pub maker_fee: Decimal,
    pub taker_fee: Decimal,
    pub tiered_fees: Option<Vec<FeeTier>>,
    pub volume_discounts: Option<Vec<VolumeDiscount>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeTier {
    pub volume_threshold: Decimal,
    pub maker_fee: Decimal,
    pub taker_fee: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeDiscount {
    pub monthly_volume: Decimal,
    pub discount_rate: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingHours {
    pub timezone: String,
    pub regular_hours: Vec<TradingSession>,
    pub extended_hours: Option<Vec<TradingSession>>,
    pub holidays: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSession {
    pub start_time: String, // "09:30"
    pub end_time: String,   // "16:00"
    pub days_of_week: Vec<u8>, // 1-7, Monday-Sunday
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityConfig {
    pub connection_type: ConnectionType,
    pub max_connections: u32,
    pub heartbeat_interval_ms: u32,
    pub reconnect_strategy: ReconnectStrategy,
    pub rate_limits: RateLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionType {
    REST,
    WebSocket,
    FIX,
    Binary,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconnectStrategy {
    pub max_retries: u32,
    pub initial_delay_ms: u32,
    pub backoff_multiplier: f64,
    pub max_delay_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimits {
    pub orders_per_second: u32,
    pub requests_per_minute: u32,
    pub weight_per_request: Option<u32>,
    pub burst_limit: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueConstraints {
    pub min_order_size: Decimal,
    pub max_order_size: Option<Decimal>,
    pub tick_size: Decimal,
    pub lot_size: Decimal,
    pub max_position_size: Option<Decimal>,
    pub allowed_order_types: Vec<OrderType>,
    pub supported_time_in_force: Vec<TimeInForce>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    StopMarket,
    StopLimit,
    TrailingStop,
    Iceberg,
    TWAP,
    VWAP,
    PostOnly,
    FillOrKill,
    ImmediateOrCancel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeInForce {
    GTC, // Good Till Cancelled
    IOC, // Immediate Or Cancel
    FOK, // Fill Or Kill
    GTD, // Good Till Date
    DAY, // Day Order
    Session,
}

/// 评分权重
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreWeights {
    pub liquidity_weight: f64,
    pub cost_weight: f64,
    pub latency_weight: f64,
    pub fill_rate_weight: f64,
    pub market_impact_weight: f64,
    pub reliability_weight: f64,
    pub volume_weight: f64,
    pub spread_weight: f64,
    pub depth_weight: f64,
    pub momentum_weight: f64,
}

/// 交易所监控器
#[derive(Debug, Clone)]
pub struct VenueMonitor {
    pub venue_id: String,
    pub config: VenueConfig,
    pub connection_status: ConnectionStatus,
    pub market_data: Arc<RwLock<MarketDataSnapshot>>,
    pub order_book: Arc<RwLock<OrderBook>>,
    pub recent_trades: Arc<RwLock<VecDeque<TradeData>>>,
    pub performance_metrics: Arc<RwLock<VenuePerformanceMetrics>>,
    pub health_status: Arc<RwLock<VenueHealthStatus>>,
    pub last_update: Arc<RwLock<DateTime<Utc>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionStatus {
    Connected,
    Connecting,
    Disconnected,
    Reconnecting,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataSnapshot {
    pub symbol: String,
    pub bid: Decimal,
    pub ask: Decimal,
    pub last_price: Decimal,
    pub bid_size: Decimal,
    pub ask_size: Decimal,
    pub volume_24h: Decimal,
    pub high_24h: Decimal,
    pub low_24h: Decimal,
    pub price_change_24h: Decimal,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub timestamp: DateTime<Utc>,
    pub sequence: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: Decimal,
    pub size: Decimal,
    pub order_count: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeData {
    pub symbol: String,
    pub price: Decimal,
    pub size: Decimal,
    pub side: TradeSide,
    pub timestamp: DateTime<Utc>,
    pub trade_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenuePerformanceMetrics {
    pub fill_rate: f64,
    pub average_fill_time_ms: f64,
    pub slippage_bps: f64,
    pub market_impact_bps: f64,
    pub rejection_rate: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub uptime_percentage: f64,
    pub total_volume_24h: Decimal,
    pub order_count_24h: u64,
    pub error_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueHealthStatus {
    pub overall_score: f64, // 0.0-1.0
    pub liquidity_score: f64,
    pub latency_score: f64,
    pub reliability_score: f64,
    pub cost_score: f64,
    pub last_check: DateTime<Utc>,
    pub issues: Vec<HealthIssue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIssue {
    pub issue_type: HealthIssueType,
    pub severity: IssueSeverity,
    pub message: String,
    pub detected_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthIssueType {
    HighLatency,
    LowLiquidity,
    ConnectionIssue,
    PricingAnomalies,
    OrderRejections,
    DataStale,
    ApiErrors,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// 评分计算器
#[derive(Debug)]
pub struct ScoreCalculator {
    config: RoutingConfig,
    scoring_models: HashMap<String, Box<dyn ScoringModel>>,
    ensemble_weights: HashMap<String, f64>,
}

/// 评分模型接口
#[async_trait]
pub trait ScoringModel: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;
    async fn calculate_score(&self, venue: &VenueMonitor, request: &RoutingRequest) -> Result<VenueScore>;
    fn feature_importance(&self) -> HashMap<String, f64>;
}

/// 路由请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRequest {
    pub symbol: String,
    pub side: TradeSide,
    pub quantity: Decimal,
    pub order_type: OrderType,
    pub time_in_force: TimeInForce,
    pub price_limit: Option<Decimal>,
    pub urgency: RoutingUrgency,
    pub strategy_id: String,
    pub client_id: String,
    pub risk_profile: RiskProfile,
    pub execution_preferences: ExecutionPreferences,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingUrgency {
    Low,      // 可以等待更好的价格
    Normal,   // 平衡价格和执行速度
    High,     // 优先执行速度
    Critical, // 立即执行
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskProfile {
    pub max_slippage_bps: f64,
    pub max_market_impact_bps: f64,
    pub max_venue_concentration: f64,
    pub preferred_venues: Vec<String>,
    pub restricted_venues: Vec<String>,
    pub max_execution_time_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPreferences {
    pub prefer_dark_pools: bool,
    pub minimize_market_impact: bool,
    pub maximize_fill_rate: bool,
    pub cost_vs_speed_preference: f64, // 0.0 = cost优先, 1.0 = speed优先
    pub venue_diversification: bool,
    pub iceberg_size: Option<Decimal>,
}

/// 交易所评分
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueScore {
    pub venue_id: String,
    pub overall_score: f64, // 0.0-1.0
    pub component_scores: ComponentScores,
    pub expected_cost_bps: f64,
    pub expected_fill_time_ms: f64,
    pub expected_fill_rate: f64,
    pub risk_score: f64,
    pub confidence: f64,
    pub reasoning: Vec<ScoreReason>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentScores {
    pub liquidity_score: f64,
    pub cost_score: f64,
    pub latency_score: f64,
    pub fill_rate_score: f64,
    pub market_impact_score: f64,
    pub reliability_score: f64,
    pub spread_score: f64,
    pub depth_score: f64,
    pub momentum_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreReason {
    pub component: String,
    pub score: f64,
    pub weight: f64,
    pub explanation: String,
    pub impact: ScoreImpact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScoreImpact {
    Positive,
    Negative,
    Neutral,
}

/// 成本估计器
#[derive(Debug)]
pub struct CostEstimator {
    models: HashMap<String, Box<dyn CostModel>>,
    market_impact_model: Box<dyn MarketImpactModel>,
    spread_model: Box<dyn SpreadModel>,
    fee_calculator: Box<dyn FeeCalculator>,
}

/// 成本模型接口
#[async_trait]
pub trait CostModel: Send + Sync + std::fmt::Debug {
    async fn estimate_cost(&self, venue: &VenueMonitor, request: &RoutingRequest) -> Result<CostEstimate>;
}

/// 市场冲击模型接口
#[async_trait]
pub trait MarketImpactModel: Send + Sync + std::fmt::Debug {
    async fn estimate_impact(&self, venue: &VenueMonitor, request: &RoutingRequest) -> Result<MarketImpactEstimate>;
}

/// 价差模型接口
#[async_trait]
pub trait SpreadModel: Send + Sync + std::fmt::Debug {
    async fn predict_spread(&self, venue: &VenueMonitor, symbol: &str) -> Result<SpreadPrediction>;
}

/// 费用计算器接口
#[async_trait]
pub trait FeeCalculator: Send + Sync + std::fmt::Debug {
    async fn calculate_fees(&self, venue: &VenueMonitor, request: &RoutingRequest) -> Result<FeeBreakdown>;
}

/// 成本估算结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    pub total_cost_bps: f64,
    pub spread_cost_bps: f64,
    pub market_impact_bps: f64,
    pub fee_cost_bps: f64,
    pub timing_cost_bps: f64,
    pub opportunity_cost_bps: f64,
    pub confidence_interval: (f64, f64), // 95% confidence interval
    pub cost_breakdown: CostBreakdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub fixed_costs: Decimal,
    pub variable_costs: Decimal,
    pub fees: FeeBreakdown,
    pub slippage: f64,
    pub market_impact: MarketImpactBreakdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeBreakdown {
    pub commission: Decimal,
    pub exchange_fees: Decimal,
    pub clearing_fees: Decimal,
    pub regulatory_fees: Decimal,
    pub other_fees: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketImpactEstimate {
    pub temporary_impact_bps: f64,
    pub permanent_impact_bps: f64,
    pub total_impact_bps: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketImpactBreakdown {
    pub linear_impact: f64,
    pub square_root_impact: f64,
    pub log_impact: f64,
    pub participation_rate_impact: f64,
    pub volatility_impact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadPrediction {
    pub predicted_spread_bps: f64,
    pub current_spread_bps: f64,
    pub spread_volatility: f64,
    pub prediction_horizon_seconds: u32,
    pub confidence: f64,
}

/// 流动性分析器
#[derive(Debug)]
pub struct LiquidityAnalyzer {
    historical_data: Arc<RwLock<HashMap<String, VecDeque<LiquiditySnapshot>>>>,
    liquidity_models: HashMap<String, Box<dyn LiquidityModel>>,
}

/// 流动性模型接口
#[async_trait]
pub trait LiquidityModel: Send + Sync + std::fmt::Debug {
    async fn analyze_liquidity(&self, venue: &VenueMonitor, symbol: &str) -> Result<LiquidityAnalysis>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquiditySnapshot {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub bid_liquidity: f64,
    pub ask_liquidity: f64,
    pub total_liquidity: f64,
    pub effective_spread: f64,
    pub price_impact_curve: Vec<PriceImpactPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceImpactPoint {
    pub quantity: Decimal,
    pub price_impact_bps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityAnalysis {
    pub current_liquidity: f64,
    pub liquidity_trend: LiquidityTrend,
    pub depth_analysis: DepthAnalysis,
    pub participation_limit: f64, // 建议的最大参与率
    pub optimal_chunk_size: Decimal,
    pub liquidity_forecast: Vec<LiquidityForecast>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LiquidityTrend {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthAnalysis {
    pub top_of_book_liquidity: f64,
    pub total_depth_10_levels: f64,
    pub weighted_average_spread: f64,
    pub liquidity_imbalance: f64, // bid vs ask imbalance
    pub depth_quality_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityForecast {
    pub timestamp: DateTime<Utc>,
    pub predicted_liquidity: f64,
    pub confidence: f64,
}

/// 延迟监控器
#[derive(Debug)]
pub struct LatencyMonitor {
    latency_measurements: Arc<RwLock<HashMap<String, VecDeque<LatencyMeasurement>>>>,
    network_topology: NetworkTopology,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMeasurement {
    pub venue_id: String,
    pub measurement_type: LatencyType,
    pub latency_ms: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LatencyType {
    OrderEntry,
    MarketData,
    OrderStatus,
    Execution,
    Cancellation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkTopology {
    pub data_centers: Vec<DataCenter>,
    pub connections: Vec<NetworkConnection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCenter {
    pub id: String,
    pub location: String,
    pub latency_to_venues: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConnection {
    pub from: String,
    pub to: String,
    pub latency_ms: f64,
    pub bandwidth_mbps: f64,
    pub reliability: f64,
}

/// 路由优化器
#[derive(Debug)]
pub struct RoutingOptimizer {
    optimization_algorithms: HashMap<String, Box<dyn OptimizationAlgorithm>>,
    constraint_manager: ConstraintManager,
    execution_simulator: ExecutionSimulator,
}

/// 优化算法接口
#[async_trait]
pub trait OptimizationAlgorithm: Send + Sync + std::fmt::Debug {
    async fn optimize(&self, request: &RoutingRequest, venue_scores: &[VenueScore]) -> Result<RoutingPlan>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingPlan {
    pub plan_id: String,
    pub total_quantity: Decimal,
    pub venue_allocations: Vec<VenueAllocation>,
    pub execution_sequence: Vec<ExecutionStep>,
    pub expected_total_cost_bps: f64,
    pub expected_fill_rate: f64,
    pub expected_completion_time_ms: f64,
    pub risk_metrics: RoutingRiskMetrics,
    pub confidence_score: f64,
    pub alternative_plans: Vec<AlternativePlan>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueAllocation {
    pub venue_id: String,
    pub allocation_percentage: f64,
    pub quantity: Decimal,
    pub priority: u32,
    pub timing_constraints: Option<TimingConstraints>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStep {
    pub step_id: u32,
    pub venue_id: String,
    pub quantity: Decimal,
    pub order_type: OrderType,
    pub price_limit: Option<Decimal>,
    pub timing: ExecutionTiming,
    pub conditions: Vec<ExecutionCondition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTiming {
    pub start_after_ms: u32,
    pub max_duration_ms: u32,
    pub dependencies: Vec<u32>, // 依赖的步骤ID
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionCondition {
    pub condition_type: ConditionType,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionType {
    PriceLimit,
    VolumeLimit,
    TimeLimit,
    MarketCondition,
    LiquidityThreshold,
    LatencyThreshold,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingConstraints {
    pub earliest_start: Option<DateTime<Utc>>,
    pub latest_start: Option<DateTime<Utc>>,
    pub max_duration: Option<Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRiskMetrics {
    pub concentration_risk: f64,
    pub liquidity_risk: f64,
    pub timing_risk: f64,
    pub execution_risk: f64,
    pub venue_risk: HashMap<String, f64>,
    pub overall_risk_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativePlan {
    pub plan_name: String,
    pub description: String,
    pub venue_allocations: Vec<VenueAllocation>,
    pub expected_cost_bps: f64,
    pub trade_offs: Vec<TradeOff>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeOff {
    pub aspect: String,
    pub current_plan_value: f64,
    pub alternative_value: f64,
    pub impact_description: String,
}

/// 约束管理器
#[derive(Debug)]
pub struct ConstraintManager {
    constraints: Vec<Box<dyn RoutingConstraint>>,
}

/// 路由约束接口
pub trait RoutingConstraint: Send + Sync + std::fmt::Debug {
    fn validate(&self, plan: &RoutingPlan, request: &RoutingRequest) -> Result<ConstraintValidation>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintValidation {
    pub is_valid: bool,
    pub violations: Vec<ConstraintViolation>,
    pub warnings: Vec<ConstraintWarning>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    pub constraint_name: String,
    pub severity: ViolationSeverity,
    pub description: String,
    pub suggested_fix: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintWarning {
    pub constraint_name: String,
    pub message: String,
    pub impact: ImpactLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactLevel {
    Low,
    Medium,
    High,
}

/// 执行模拟器
#[derive(Debug)]
pub struct ExecutionSimulator {
    simulation_models: HashMap<String, Box<dyn SimulationModel>>,
    market_scenario_generator: MarketScenarioGenerator,
}

/// 模拟模型接口
#[async_trait]
pub trait SimulationModel: Send + Sync + std::fmt::Debug {
    async fn simulate(&self, plan: &RoutingPlan, scenarios: &[MarketScenario]) -> Result<SimulationResult>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketScenario {
    pub scenario_id: String,
    pub scenario_name: String,
    pub probability: f64,
    pub market_conditions: MarketConditions,
    pub venue_states: HashMap<String, VenueState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub volatility_regime: VolatilityRegime,
    pub liquidity_regime: LiquidityRegime,
    pub trend_direction: TrendDirection,
    pub market_stress_level: f64, // 0.0-1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolatilityRegime {
    Low,
    Normal,
    High,
    Extreme,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LiquidityRegime {
    Abundant,
    Normal,
    Scarce,
    Fragmented,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Bullish,
    Bearish,
    Sideways,
    Volatile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueState {
    pub venue_id: String,
    pub availability: f64, // 0.0-1.0
    pub latency_multiplier: f64,
    pub liquidity_multiplier: f64,
    pub error_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub scenario_results: Vec<ScenarioResult>,
    pub aggregated_metrics: AggregatedMetrics,
    pub risk_analysis: SimulationRiskAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResult {
    pub scenario_id: String,
    pub total_cost_bps: f64,
    pub fill_rate: f64,
    pub completion_time_ms: f64,
    pub slippage_bps: f64,
    pub venue_performance: HashMap<String, VenueSimulationResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueSimulationResult {
    pub venue_id: String,
    pub filled_quantity: Decimal,
    pub average_price: Decimal,
    pub execution_time_ms: f64,
    pub fees_paid: Decimal,
    pub rejections: u32,
    pub partial_fills: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    pub expected_cost_bps: f64,
    pub cost_volatility_bps: f64,
    pub expected_fill_rate: f64,
    pub fill_rate_volatility: f64,
    pub expected_completion_time_ms: f64,
    pub completion_time_volatility_ms: f64,
    pub worst_case_cost_bps: f64,  // 95th percentile
    pub best_case_cost_bps: f64,   // 5th percentile
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationRiskAnalysis {
    pub value_at_risk_bps: f64,      // VaR at 95% confidence
    pub expected_shortfall_bps: f64,  // CVaR
    pub tail_risk_scenarios: Vec<String>,
    pub sensitivity_analysis: HashMap<String, f64>,
}

/// 市场情景生成器
#[derive(Debug)]
pub struct MarketScenarioGenerator {
    scenario_templates: Vec<ScenarioTemplate>,
    historical_patterns: HashMap<String, Vec<HistoricalPattern>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioTemplate {
    pub template_id: String,
    pub name: String,
    pub description: String,
    pub base_probability: f64,
    pub parameter_ranges: HashMap<String, ParameterRange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterRange {
    pub min_value: f64,
    pub max_value: f64,
    pub distribution_type: DistributionType,
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionType {
    Uniform,
    Normal { mean: f64, std_dev: f64 },
    LogNormal { mu: f64, sigma: f64 },
    Beta { alpha: f64, beta: f64 },
    Exponential { lambda: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalPattern {
    pub pattern_id: String,
    pub conditions: HashMap<String, f64>,
    pub outcomes: HashMap<String, f64>,
    pub frequency: f64,
    pub last_observed: DateTime<Utc>,
}

/// 性能跟踪器
#[derive(Debug, Clone)]
pub struct PerformanceTracker {
    routing_history: VecDeque<RoutingExecution>,
    venue_performance: HashMap<String, VenuePerformanceHistory>,
    model_performance: HashMap<String, ModelPerformanceMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingExecution {
    pub execution_id: String,
    pub request: RoutingRequest,
    pub plan: RoutingPlan,
    pub actual_results: ExecutionResults,
    pub performance_metrics: ExecutionPerformanceMetrics,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResults {
    pub total_filled: Decimal,
    pub average_fill_price: Decimal,
    pub total_cost_bps: f64,
    pub execution_time_ms: f64,
    pub venue_breakdown: HashMap<String, VenueExecutionResult>,
    pub unexpected_outcomes: Vec<UnexpectedOutcome>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueExecutionResult {
    pub venue_id: String,
    pub filled_quantity: Decimal,
    pub average_price: Decimal,
    pub fees_paid: Decimal,
    pub execution_time_ms: f64,
    pub number_of_orders: u32,
    pub fill_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnexpectedOutcome {
    pub outcome_type: OutcomeType,
    pub description: String,
    pub impact: f64,
    pub venue_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutcomeType {
    HigherThanExpectedCost,
    LowerThanExpectedFillRate,
    UnexpectedLatency,
    VenueUnavailability,
    PriceMovement,
    LiquidityDryUp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPerformanceMetrics {
    pub prediction_accuracy: f64, // 预测准确度
    pub cost_prediction_error_bps: f64,
    pub fill_rate_prediction_error: f64,
    pub timing_prediction_error_ms: f64,
    pub overall_satisfaction_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenuePerformanceHistory {
    pub venue_id: String,
    pub execution_count: u64,
    pub average_fill_rate: f64,
    pub average_cost_bps: f64,
    pub average_latency_ms: f64,
    pub reliability_score: f64,
    pub recent_performance_trend: PerformanceTrend,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Degrading,
    Volatile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceMetrics {
    pub model_name: String,
    pub prediction_accuracy: f64,
    pub mean_absolute_error: f64,
    pub root_mean_square_error: f64,
    pub directional_accuracy: f64, // 预测方向正确性
    pub calibration_score: f64,    // 概率校准度
    pub last_evaluation: DateTime<Utc>,
}

/// 自适应权重管理器
#[derive(Debug, Clone)]
pub struct AdaptiveWeights {
    current_weights: ScoreWeights,
    weight_history: VecDeque<WeightSnapshot>,
    adaptation_algorithm: WeightAdaptationAlgorithm,
    performance_feedback: VecDeque<PerformanceFeedback>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightSnapshot {
    pub weights: ScoreWeights,
    pub timestamp: DateTime<Utc>,
    pub performance_score: f64,
    pub market_regime: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightAdaptationAlgorithm {
    GradientDescent,
    BayesianOptimization,
    ReinforcementLearning,
    EnsembleVoting,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceFeedback {
    pub execution_id: String,
    pub predicted_score: f64,
    pub actual_performance: f64,
    pub error: f64,
    pub market_conditions: String,
    pub timestamp: DateTime<Utc>,
}

// 实现主要的 SmartRoutingScorer 结构
impl SmartRoutingScorer {
    /// 创建新的智能路由评分引擎
    pub fn new(config: RoutingConfig) -> Result<Self> {
        let venue_monitors = Arc::new(TokioRwLock::new(HashMap::new()));
        let score_calculator = Arc::new(ScoreCalculator::new(config.clone())?);
        let cost_estimator = Arc::new(CostEstimator::new()?);
        let liquidity_analyzer = Arc::new(LiquidityAnalyzer::new()?);
        let latency_monitor = Arc::new(LatencyMonitor::new()?);
        let routing_optimizer = Arc::new(RoutingOptimizer::new()?);
        let performance_tracker = Arc::new(RwLock::new(PerformanceTracker::new()));
        let adaptive_weights = Arc::new(RwLock::new(AdaptiveWeights::new(config.score_weights.clone())));

        Ok(Self {
            config,
            venue_monitors,
            score_calculator,
            cost_estimator,
            liquidity_analyzer,
            latency_monitor,
            routing_optimizer,
            performance_tracker,
            adaptive_weights,
        })
    }

    /// 初始化交易所监控器
    pub async fn initialize_venue_monitors(&self) -> Result<()> {
        let mut monitors = self.venue_monitors.write().await;
        
        for venue_config in &self.config.supported_venues {
            let monitor = VenueMonitor::new(venue_config.clone()).await?;
            monitors.insert(venue_config.venue_id.clone(), monitor);
        }

        log::info!("Initialized {} venue monitors", monitors.len());
        Ok(())
    }

    /// 生成路由评分和计划
    pub async fn generate_routing_plan(&self, request: &RoutingRequest) -> Result<RoutingPlan> {
        // 1. 获取所有交易所评分
        let venue_scores = self.score_all_venues(request).await?;
        
        // 2. 过滤不符合要求的交易所
        let filtered_scores = self.filter_venues(&venue_scores, request)?;
        
        // 3. 生成路由计划
        let routing_plan = self.routing_optimizer.optimize(request, &filtered_scores).await?;
        
        // 4. 验证约束
        let validation = self.routing_optimizer.validate_plan(&routing_plan, request)?;
        if !validation.is_valid {
            return Err(anyhow::anyhow!("Routing plan validation failed: {:?}", validation.violations));
        }
        
        // 5. 执行模拟
        let simulation_result = self.routing_optimizer.simulate_execution(&routing_plan).await?;
        
        log::info!("Generated routing plan with {} venue allocations, expected cost: {:.2} bps", 
            routing_plan.venue_allocations.len(), routing_plan.expected_total_cost_bps);
            
        Ok(routing_plan)
    }

    /// 为所有交易所计算评分
    async fn score_all_venues(&self, request: &RoutingRequest) -> Result<Vec<VenueScore>> {
        let monitors = self.venue_monitors.read().await;
        let mut venue_scores = Vec::new();
        
        for monitor in monitors.values() {
            // 检查交易所是否支持该交易对
            if !monitor.config.supported_symbols.contains(&request.symbol) {
                continue;
            }
            
            // 检查连接状态
            if !matches!(monitor.connection_status, ConnectionStatus::Connected) {
                continue;
            }
            
            // 计算评分
            match self.score_calculator.calculate_score(monitor, request).await {
                Ok(score) => venue_scores.push(score),
                Err(e) => {
                    log::warn!("Failed to calculate score for venue {}: {}", monitor.venue_id, e);
                    continue;
                }
            }
        }
        
        // 按评分排序
        venue_scores.sort_by(|a, b| b.overall_score.partial_cmp(&a.overall_score).unwrap());
        
        Ok(venue_scores)
    }

    /// 过滤交易所
    fn filter_venues(&self, venue_scores: &[VenueScore], request: &RoutingRequest) -> Result<Vec<VenueScore>> {
        let mut filtered = Vec::new();
        
        for score in venue_scores {
            // 检查风险限制
            if score.expected_cost_bps > request.risk_profile.max_slippage_bps {
                continue;
            }
            
            // 检查流动性阈值
            if score.component_scores.liquidity_score < self.config.min_liquidity_threshold {
                continue;
            }
            
            // 检查延迟容忍度
            if score.expected_fill_time_ms > request.risk_profile.max_execution_time_ms as f64 {
                continue;
            }
            
            // 检查限制交易所列表
            if request.risk_profile.restricted_venues.contains(&score.venue_id) {
                continue;
            }
            
            filtered.push(score.clone());
        }
        
        Ok(filtered)
    }

    /// 更新交易所监控数据
    pub async fn update_venue_data(&self, venue_id: &str, market_data: MarketDataSnapshot) -> Result<()> {
        let monitors = self.venue_monitors.read().await;
        
        if let Some(monitor) = monitors.get(venue_id) {
            {
                let mut data = monitor.market_data.write().unwrap();
                *data = market_data;
            }
            
            {
                let mut last_update = monitor.last_update.write().unwrap();
                *last_update = Utc::now();
            }
            
            // 更新健康状态
            self.update_venue_health(venue_id).await?;
        }
        
        Ok(())
    }

    /// 更新交易所健康状态
    async fn update_venue_health(&self, venue_id: &str) -> Result<()> {
        let monitors = self.venue_monitors.read().await;
        
        if let Some(monitor) = monitors.get(venue_id) {
            let health_status = self.calculate_venue_health(monitor).await?;
            
            let mut current_health = monitor.health_status.write().unwrap();
            *current_health = health_status;
        }
        
        Ok(())
    }

    /// 计算交易所健康评分
    async fn calculate_venue_health(&self, monitor: &VenueMonitor) -> Result<VenueHealthStatus> {
        let mut overall_score = 1.0;
        let mut issues = Vec::new();
        
        // 检查连接状态
        if !matches!(monitor.connection_status, ConnectionStatus::Connected) {
            overall_score *= 0.5;
            issues.push(HealthIssue {
                issue_type: HealthIssueType::ConnectionIssue,
                severity: IssueSeverity::High,
                message: format!("Venue {} is not connected", monitor.venue_id),
                detected_at: Utc::now(),
                resolved_at: None,
            });
        }
        
        // 检查数据新鲜度
        let last_update = *monitor.last_update.read().unwrap();
        let age = Utc::now().signed_duration_since(last_update);
        if age.num_seconds() > 30 {
            overall_score *= 0.8;
            issues.push(HealthIssue {
                issue_type: HealthIssueType::DataStale,
                severity: IssueSeverity::Medium,
                message: format!("Data is {} seconds old", age.num_seconds()),
                detected_at: Utc::now(),
                resolved_at: None,
            });
        }
        
        // 检查性能指标
        let performance = monitor.performance_metrics.read().unwrap();
        
        let liquidity_score = if performance.total_volume_24h > Decimal::ZERO { 1.0 } else { 0.3 };
        let latency_score = if performance.latency_p95_ms < 100.0 { 1.0 } else { 0.7 };
        let reliability_score = performance.uptime_percentage;
        let cost_score = if performance.slippage_bps < 10.0 { 1.0 } else { 0.6 };
        
        overall_score *= (liquidity_score + latency_score + reliability_score + cost_score) / 4.0;
        
        Ok(VenueHealthStatus {
            overall_score,
            liquidity_score,
            latency_score,
            reliability_score,
            cost_score,
            last_check: Utc::now(),
            issues,
        })
    }

    /// 记录执行结果并更新模型
    pub async fn record_execution_result(&self, execution: RoutingExecution) -> Result<()> {
        // 更新性能跟踪器
        {
            let mut tracker = self.performance_tracker.write().unwrap();
            tracker.add_execution(execution.clone());
        }
        
        // 更新自适应权重
        if self.config.enable_adaptive_weights {
            self.update_adaptive_weights(&execution).await?;
        }
        
        // 更新交易所性能历史
        for (venue_id, venue_result) in &execution.actual_results.venue_breakdown {
            self.update_venue_performance_history(venue_id, venue_result).await?;
        }
        
        log::info!("Recorded execution result for order {}", execution.execution_id);
        Ok(())
    }

    /// 更新自适应权重
    async fn update_adaptive_weights(&self, execution: &RoutingExecution) -> Result<()> {
        let mut adaptive_weights = self.adaptive_weights.write().unwrap();
        
        // 计算性能反馈
        let predicted_cost = execution.plan.expected_total_cost_bps;
        let actual_cost = execution.actual_results.total_cost_bps;
        let performance_error = (actual_cost - predicted_cost).abs();
        
        let feedback = PerformanceFeedback {
            execution_id: execution.execution_id.clone(),
            predicted_score: predicted_cost,
            actual_performance: actual_cost,
            error: performance_error,
            market_conditions: "normal".to_string(), // 简化
            timestamp: execution.timestamp,
        };
        
        adaptive_weights.add_feedback(feedback);
        
        // 调整权重
        if adaptive_weights.should_adapt() {
            adaptive_weights.adapt_weights()?;
            log::info!("Updated adaptive weights based on recent performance feedback");
        }
        
        Ok(())
    }

    /// 更新交易所性能历史
    async fn update_venue_performance_history(
        &self, 
        venue_id: &str, 
        venue_result: &VenueExecutionResult
    ) -> Result<()> {
        let mut tracker = self.performance_tracker.write().unwrap();
        tracker.update_venue_performance(venue_id, venue_result);
        Ok(())
    }

    /// 获取实时评分
    pub async fn get_real_time_scores(&self, symbol: &str) -> Result<HashMap<String, VenueScore>> {
        let monitors = self.venue_monitors.read().await;
        let mut scores = HashMap::new();
        
        // 创建简单的请求用于评分
        let sample_request = RoutingRequest {
            symbol: symbol.to_string(),
            side: TradeSide::Buy,
            quantity: Decimal::from(100),
            order_type: OrderType::Market,
            time_in_force: TimeInForce::IOC,
            price_limit: None,
            urgency: RoutingUrgency::Normal,
            strategy_id: "real_time_scoring".to_string(),
            client_id: "system".to_string(),
            risk_profile: RiskProfile::default(),
            execution_preferences: ExecutionPreferences::default(),
            metadata: HashMap::new(),
        };
        
        for monitor in monitors.values() {
            if monitor.config.supported_symbols.contains(&symbol.to_string()) {
                if let Ok(score) = self.score_calculator.calculate_score(monitor, &sample_request).await {
                    scores.insert(monitor.venue_id.clone(), score);
                }
            }
        }
        
        Ok(scores)
    }

    /// 启动实时监控
    pub async fn start_monitoring(&self) -> Result<()> {
        let update_interval = Duration::milliseconds(self.config.update_frequency_ms as i64);
        
        // 启动数据更新任务
        let venue_monitors = Arc::clone(&self.venue_monitors);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                tokio::time::Duration::from_millis(update_interval.num_milliseconds() as u64)
            );
            
            loop {
                interval.tick().await;
                
                let monitors = venue_monitors.read().await;
                for monitor in monitors.values() {
                    // 这里会实际更新市场数据
                    // 简化实现，实际中会从API获取数据
                }
            }
        });
        
        log::info!("Started real-time monitoring with {}ms update frequency", 
            self.config.update_frequency_ms);
            
        Ok(())
    }

    /// 获取系统统计信息
    pub async fn get_statistics(&self) -> Result<RoutingStatistics> {
        let tracker = self.performance_tracker.read().unwrap();
        let monitors = self.venue_monitors.read().await;
        
        let total_executions = tracker.routing_history.len();
        let average_cost_bps = if total_executions > 0 {
            tracker.routing_history.iter()
                .map(|e| e.actual_results.total_cost_bps)
                .sum::<f64>() / total_executions as f64
        } else {
            0.0
        };
        
        let venue_count = monitors.len();
        let connected_venues = monitors.values()
            .filter(|m| matches!(m.connection_status, ConnectionStatus::Connected))
            .count();
        
        Ok(RoutingStatistics {
            total_executions,
            average_cost_bps,
            venue_count,
            connected_venues,
            last_updated: Utc::now(),
        })
    }
}

/// 路由统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingStatistics {
    pub total_executions: usize,
    pub average_cost_bps: f64,
    pub venue_count: usize,
    pub connected_venues: usize,
    pub last_updated: DateTime<Utc>,
}

// 实现各个组件
impl VenueMonitor {
    pub async fn new(config: VenueConfig) -> Result<Self> {
        Ok(Self {
            venue_id: config.venue_id.clone(),
            config,
            connection_status: ConnectionStatus::Disconnected,
            market_data: Arc::new(RwLock::new(MarketDataSnapshot::default())),
            order_book: Arc::new(RwLock::new(OrderBook::default())),
            recent_trades: Arc::new(RwLock::new(VecDeque::new())),
            performance_metrics: Arc::new(RwLock::new(VenuePerformanceMetrics::default())),
            health_status: Arc::new(RwLock::new(VenueHealthStatus::default())),
            last_update: Arc::new(RwLock::new(Utc::now())),
        })
    }
}

impl ScoreCalculator {
    pub fn new(config: RoutingConfig) -> Result<Self> {
        let mut scoring_models: HashMap<String, Box<dyn ScoringModel>> = HashMap::new();
        
        // 添加基础评分模型
        scoring_models.insert("composite".to_string(), Box::new(CompositeScoreModel::new(config.score_weights.clone())));
        
        let ensemble_weights = {
            let mut weights = HashMap::new();
            weights.insert("composite".to_string(), 1.0);
            weights
        };
        
        Ok(Self {
            config,
            scoring_models,
            ensemble_weights,
        })
    }

    pub async fn calculate_score(&self, venue: &VenueMonitor, request: &RoutingRequest) -> Result<VenueScore> {
        // 使用组合模型计算评分
        if let Some(model) = self.scoring_models.get("composite") {
            model.calculate_score(venue, request).await
        } else {
            Err(anyhow::anyhow!("No scoring model available"))
        }
    }
}

impl CostEstimator {
    pub fn new() -> Result<Self> {
        let mut models: HashMap<String, Box<dyn CostModel>> = HashMap::new();
        models.insert("linear".to_string(), Box::new(LinearCostModel::new()));
        
        Ok(Self {
            models,
            market_impact_model: Box::new(SquareRootLawModel::new()),
            spread_model: Box::new(SimpleSpreadModel::new()),
            fee_calculator: Box::new(TieredFeeCalculator::new()),
        })
    }
}

impl LiquidityAnalyzer {
    pub fn new() -> Result<Self> {
        let mut liquidity_models: HashMap<String, Box<dyn LiquidityModel>> = HashMap::new();
        liquidity_models.insert("order_book".to_string(), Box::new(OrderBookLiquidityModel::new()));
        
        Ok(Self {
            historical_data: Arc::new(RwLock::new(HashMap::new())),
            liquidity_models,
        })
    }
}

impl LatencyMonitor {
    pub fn new() -> Result<Self> {
        Ok(Self {
            latency_measurements: Arc::new(RwLock::new(HashMap::new())),
            network_topology: NetworkTopology {
                data_centers: vec![],
                connections: vec![],
            },
        })
    }
}

impl RoutingOptimizer {
    pub fn new() -> Result<Self> {
        let mut optimization_algorithms: HashMap<String, Box<dyn OptimizationAlgorithm>> = HashMap::new();
        optimization_algorithms.insert("greedy".to_string(), Box::new(GreedyOptimizer::new()));
        
        Ok(Self {
            optimization_algorithms,
            constraint_manager: ConstraintManager::new(),
            execution_simulator: ExecutionSimulator::new(),
        })
    }

    pub async fn optimize(&self, request: &RoutingRequest, venue_scores: &[VenueScore]) -> Result<RoutingPlan> {
        if let Some(optimizer) = self.optimization_algorithms.get("greedy") {
            optimizer.optimize(request, venue_scores).await
        } else {
            Err(anyhow::anyhow!("No optimization algorithm available"))
        }
    }

    pub fn validate_plan(&self, plan: &RoutingPlan, request: &RoutingRequest) -> Result<ConstraintValidation> {
        self.constraint_manager.validate(plan, request)
    }

    pub async fn simulate_execution(&self, _plan: &RoutingPlan) -> Result<SimulationResult> {
        // 简化实现
        Ok(SimulationResult {
            scenario_results: vec![],
            aggregated_metrics: AggregatedMetrics::default(),
            risk_analysis: SimulationRiskAnalysis::default(),
        })
    }
}

impl ConstraintManager {
    pub fn new() -> Self {
        Self {
            constraints: vec![],
        }
    }

    pub fn validate(&self, _plan: &RoutingPlan, _request: &RoutingRequest) -> Result<ConstraintValidation> {
        // 简化验证
        Ok(ConstraintValidation {
            is_valid: true,
            violations: vec![],
            warnings: vec![],
        })
    }
}

impl ExecutionSimulator {
    pub fn new() -> Self {
        Self {
            simulation_models: HashMap::new(),
            market_scenario_generator: MarketScenarioGenerator::new(),
        }
    }
}

impl MarketScenarioGenerator {
    pub fn new() -> Self {
        Self {
            scenario_templates: vec![],
            historical_patterns: HashMap::new(),
        }
    }
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            routing_history: VecDeque::new(),
            venue_performance: HashMap::new(),
            model_performance: HashMap::new(),
        }
    }

    pub fn add_execution(&mut self, execution: RoutingExecution) {
        self.routing_history.push_back(execution);
        
        // 保持历史记录大小
        while self.routing_history.len() > 1000 {
            self.routing_history.pop_front();
        }
    }

    pub fn update_venue_performance(&mut self, venue_id: &str, result: &VenueExecutionResult) {
        let history = self.venue_performance.entry(venue_id.to_string())
            .or_insert_with(|| VenuePerformanceHistory {
                venue_id: venue_id.to_string(),
                execution_count: 0,
                average_fill_rate: 0.0,
                average_cost_bps: 0.0,
                average_latency_ms: 0.0,
                reliability_score: 1.0,
                recent_performance_trend: PerformanceTrend::Stable,
                last_updated: Utc::now(),
            });
        
        history.execution_count += 1;
        history.average_fill_rate = (history.average_fill_rate * (history.execution_count - 1) as f64 + result.fill_rate) / history.execution_count as f64;
        history.average_latency_ms = (history.average_latency_ms * (history.execution_count - 1) as f64 + result.execution_time_ms) / history.execution_count as f64;
        history.last_updated = Utc::now();
    }
}

impl AdaptiveWeights {
    pub fn new(initial_weights: ScoreWeights) -> Self {
        Self {
            current_weights: initial_weights,
            weight_history: VecDeque::new(),
            adaptation_algorithm: WeightAdaptationAlgorithm::GradientDescent,
            performance_feedback: VecDeque::new(),
        }
    }

    pub fn add_feedback(&mut self, feedback: PerformanceFeedback) {
        self.performance_feedback.push_back(feedback);
        
        // 保持反馈历史大小
        while self.performance_feedback.len() > 100 {
            self.performance_feedback.pop_front();
        }
    }

    pub fn should_adapt(&self) -> bool {
        self.performance_feedback.len() >= 10
    }

    pub fn adapt_weights(&mut self) -> Result<()> {
        // 简化的权重调整逻辑
        // 实际实现会使用更复杂的优化算法
        
        let recent_errors: Vec<f64> = self.performance_feedback.iter()
            .rev()
            .take(10)
            .map(|f| f.error)
            .collect();
        
        let avg_error = recent_errors.iter().sum::<f64>() / recent_errors.len() as f64;
        
        // 如果错误太大，调整权重
        if avg_error > 5.0 { // 5 bps threshold
            // 简单调整：增加成本权重
            self.current_weights.cost_weight = (self.current_weights.cost_weight * 1.1).min(1.0);
            self.current_weights.liquidity_weight = (self.current_weights.liquidity_weight * 0.9).max(0.1);
        }
        
        Ok(())
    }
}

// 具体模型实现
#[derive(Debug)]
pub struct CompositeScoreModel {
    weights: ScoreWeights,
}

impl CompositeScoreModel {
    pub fn new(weights: ScoreWeights) -> Self {
        Self { weights }
    }
}

#[async_trait]
impl ScoringModel for CompositeScoreModel {
    fn name(&self) -> &str {
        "composite"
    }

    async fn calculate_score(&self, venue: &VenueMonitor, request: &RoutingRequest) -> Result<VenueScore> {
        let performance = venue.performance_metrics.read().unwrap();
        let health = venue.health_status.read().unwrap();
        let market_data = venue.market_data.read().unwrap();
        
        // 计算各个组件评分
        let liquidity_score = health.liquidity_score;
        let cost_score = if performance.slippage_bps > 0.0 {
            (20.0 - performance.slippage_bps).max(0.0) / 20.0
        } else { 1.0 };
        let latency_score = if performance.latency_p95_ms > 0.0 {
            (200.0 - performance.latency_p95_ms).max(0.0) / 200.0
        } else { 1.0 };
        let fill_rate_score = performance.fill_rate;
        let market_impact_score = if performance.market_impact_bps > 0.0 {
            (10.0 - performance.market_impact_bps).max(0.0) / 10.0
        } else { 1.0 };
        let reliability_score = health.reliability_score;
        
        // 简化的价差和深度评分
        let spread_score = if market_data.ask > market_data.bid {
            let spread_decimal = (market_data.ask - market_data.bid) / market_data.bid * Decimal::from(10000);
            let spread_bps: f64 = spread_decimal.try_into().unwrap_or(0.0);
            (20.0 - spread_bps).max(0.0) / 20.0
        } else { 0.5 };
        
        let volume_score = if market_data.volume_24h > Decimal::ZERO {
            let volume_f64: f64 = market_data.volume_24h.try_into().unwrap_or(1.0);
            (volume_f64.ln() / 20.0).min(1.0)
        } else { 0.1 };
        
        let depth_size: f64 = (market_data.bid_size + market_data.ask_size).try_into().unwrap_or(0.0);
        let depth_score = depth_size / 10000.0;
        let momentum_score = 0.5; // 简化
        
        let component_scores = ComponentScores {
            liquidity_score,
            cost_score,
            latency_score,
            fill_rate_score,
            market_impact_score,
            reliability_score,
            spread_score,
            depth_score,
            momentum_score,
        };
        
        // 计算加权总分
        let overall_score = (
            liquidity_score * self.weights.liquidity_weight +
            cost_score * self.weights.cost_weight +
            latency_score * self.weights.latency_weight +
            fill_rate_score * self.weights.fill_rate_weight +
            market_impact_score * self.weights.market_impact_weight +
            reliability_score * self.weights.reliability_weight +
            volume_score * self.weights.volume_weight +
            spread_score * self.weights.spread_weight +
            depth_score * self.weights.depth_weight +
            momentum_score * self.weights.momentum_weight
        ) / (
            self.weights.liquidity_weight +
            self.weights.cost_weight +
            self.weights.latency_weight +
            self.weights.fill_rate_weight +
            self.weights.market_impact_weight +
            self.weights.reliability_weight +
            self.weights.volume_weight +
            self.weights.spread_weight +
            self.weights.depth_weight +
            self.weights.momentum_weight
        );
        
        // 估算期望成本和执行时间
        let expected_cost_bps = performance.slippage_bps + performance.market_impact_bps;
        let expected_fill_time_ms = performance.average_fill_time_ms;
        let expected_fill_rate = performance.fill_rate;
        
        let risk_score = 1.0 - health.overall_score;
        let confidence = if performance.order_count_24h > 10 { 0.9 } else { 0.5 };
        
        let reasoning = vec![
            ScoreReason {
                component: "liquidity".to_string(),
                score: liquidity_score,
                weight: self.weights.liquidity_weight,
                explanation: "Based on order book depth and recent volume".to_string(),
                impact: if liquidity_score > 0.7 { ScoreImpact::Positive } else { ScoreImpact::Negative },
            },
            ScoreReason {
                component: "cost".to_string(),
                score: cost_score,
                weight: self.weights.cost_weight,
                explanation: format!("Slippage: {:.2} bps", performance.slippage_bps),
                impact: if cost_score > 0.8 { ScoreImpact::Positive } else { ScoreImpact::Negative },
            },
        ];
        
        Ok(VenueScore {
            venue_id: venue.venue_id.clone(),
            overall_score,
            component_scores,
            expected_cost_bps,
            expected_fill_time_ms,
            expected_fill_rate,
            risk_score,
            confidence,
            reasoning,
        })
    }

    fn feature_importance(&self) -> HashMap<String, f64> {
        let mut importance = HashMap::new();
        importance.insert("liquidity".to_string(), self.weights.liquidity_weight);
        importance.insert("cost".to_string(), self.weights.cost_weight);
        importance.insert("latency".to_string(), self.weights.latency_weight);
        importance
    }
}

// 其他模型的简化实现
#[derive(Debug)]
pub struct LinearCostModel;

impl LinearCostModel {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl CostModel for LinearCostModel {
    async fn estimate_cost(&self, venue: &VenueMonitor, _request: &RoutingRequest) -> Result<CostEstimate> {
        let performance = venue.performance_metrics.read().unwrap();
        
        Ok(CostEstimate {
            total_cost_bps: performance.slippage_bps + 2.0, // 加上费用估算
            spread_cost_bps: 1.0,
            market_impact_bps: performance.market_impact_bps,
            fee_cost_bps: 1.0,
            timing_cost_bps: 0.5,
            opportunity_cost_bps: 0.0,
            confidence_interval: (performance.slippage_bps - 1.0, performance.slippage_bps + 1.0),
            cost_breakdown: CostBreakdown::default(),
        })
    }
}

#[derive(Debug)]
pub struct SquareRootLawModel;

impl SquareRootLawModel {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl MarketImpactModel for SquareRootLawModel {
    async fn estimate_impact(&self, venue: &VenueMonitor, request: &RoutingRequest) -> Result<MarketImpactEstimate> {
        let market_data = venue.market_data.read().unwrap();
        
        // 简化的平方根定律实现
        let quantity_f64: f64 = request.quantity.try_into().unwrap_or(0.0);
        let volume_f64: f64 = market_data.volume_24h.try_into().unwrap_or(1.0);
        let participation_rate = quantity_f64 / volume_f64;
        let base_impact = participation_rate.sqrt() * 10.0; // 简化系数
        
        Ok(MarketImpactEstimate {
            temporary_impact_bps: base_impact * 0.6,
            permanent_impact_bps: base_impact * 0.4,
            total_impact_bps: base_impact,
            confidence: 0.7,
        })
    }
}

#[derive(Debug)]
pub struct SimpleSpreadModel;

impl SimpleSpreadModel {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl SpreadModel for SimpleSpreadModel {
    async fn predict_spread(&self, venue: &VenueMonitor, _symbol: &str) -> Result<SpreadPrediction> {
        let market_data = venue.market_data.read().unwrap();
        let current_spread_bps = if market_data.bid > Decimal::ZERO {
            let spread_decimal = (market_data.ask - market_data.bid) / market_data.bid * Decimal::from(10000);
            spread_decimal.try_into().unwrap_or(10.0)
        } else { 10.0 };
        
        Ok(SpreadPrediction {
            predicted_spread_bps: current_spread_bps,
            current_spread_bps,
            spread_volatility: current_spread_bps * 0.1,
            prediction_horizon_seconds: 60,
            confidence: 0.8,
        })
    }
}

#[derive(Debug)]
pub struct TieredFeeCalculator;

impl TieredFeeCalculator {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl FeeCalculator for TieredFeeCalculator {
    async fn calculate_fees(&self, venue: &VenueMonitor, request: &RoutingRequest) -> Result<FeeBreakdown> {
        let notional = request.quantity * request.price_limit.unwrap_or(Decimal::from(100));
        let commission = notional * venue.config.fee_structure.taker_fee;
        
        Ok(FeeBreakdown {
            commission,
            exchange_fees: Decimal::ZERO,
            clearing_fees: Decimal::ZERO,
            regulatory_fees: Decimal::ZERO,
            other_fees: Decimal::ZERO,
        })
    }
}

#[derive(Debug)]
pub struct OrderBookLiquidityModel;

impl OrderBookLiquidityModel {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl LiquidityModel for OrderBookLiquidityModel {
    async fn analyze_liquidity(&self, venue: &VenueMonitor, _symbol: &str) -> Result<LiquidityAnalysis> {
        let order_book = venue.order_book.read().unwrap();
        
        let bid_liquidity: f64 = order_book.bids.iter().map(|l| l.size.try_into().unwrap_or(0.0)).sum();
        let ask_liquidity: f64 = order_book.asks.iter().map(|l| l.size.try_into().unwrap_or(0.0)).sum();
        let total_liquidity = bid_liquidity + ask_liquidity;
        
        Ok(LiquidityAnalysis {
            current_liquidity: total_liquidity,
            liquidity_trend: LiquidityTrend::Stable,
            depth_analysis: DepthAnalysis::default(),
            participation_limit: 0.1,
            optimal_chunk_size: Decimal::from(100),
            liquidity_forecast: vec![],
        })
    }
}

#[derive(Debug)]
pub struct GreedyOptimizer;

impl GreedyOptimizer {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl OptimizationAlgorithm for GreedyOptimizer {
    async fn optimize(&self, request: &RoutingRequest, venue_scores: &[VenueScore]) -> Result<RoutingPlan> {
        let plan_id = format!("plan_{}", Utc::now().timestamp_millis());
        let mut venue_allocations = Vec::new();
        let mut remaining_quantity = request.quantity;
        
        // 贪心分配：按评分从高到低分配
        for (i, score) in venue_scores.iter().take(3).enumerate() { // 最多使用3个交易所
            if remaining_quantity <= Decimal::ZERO {
                break;
            }
            
            let allocation_pct = match i {
                0 => 0.6, // 最高评分交易所60%
                1 => 0.3, // 第二高30%
                _ => 0.1, // 其他10%
            };
            
            let quantity = (request.quantity * Decimal::try_from(allocation_pct).unwrap_or(Decimal::ZERO)).min(remaining_quantity);
            
            venue_allocations.push(VenueAllocation {
                venue_id: score.venue_id.clone(),
                allocation_percentage: allocation_pct,
                quantity,
                priority: (i + 1) as u32,
                timing_constraints: None,
            });
            
            remaining_quantity -= quantity;
        }
        
        // 生成执行步骤
        let execution_sequence: Vec<ExecutionStep> = venue_allocations.iter().enumerate().map(|(i, alloc)| {
            ExecutionStep {
                step_id: i as u32,
                venue_id: alloc.venue_id.clone(),
                quantity: alloc.quantity,
                order_type: request.order_type.clone(),
                price_limit: request.price_limit,
                timing: ExecutionTiming {
                    start_after_ms: (i as u32) * 100, // 100ms间隔
                    max_duration_ms: 30000, // 30秒最大执行时间
                    dependencies: vec![],
                },
                conditions: vec![],
            }
        }).collect();
        
        // 计算期望指标
        let expected_total_cost_bps = venue_scores.iter()
            .take(venue_allocations.len())
            .map(|s| s.expected_cost_bps)
            .fold(0.0, |acc, cost| acc + cost) / venue_allocations.len() as f64;
        
        let expected_fill_rate = venue_scores.iter()
            .take(venue_allocations.len())
            .map(|s| s.expected_fill_rate)
            .fold(0.0, |acc, rate| acc + rate) / venue_allocations.len() as f64;
        
        let expected_completion_time_ms = venue_scores.iter()
            .take(venue_allocations.len())
            .map(|s| s.expected_fill_time_ms)
            .fold(0.0_f64, |acc, time| acc.max(time));
        
        Ok(RoutingPlan {
            plan_id,
            total_quantity: request.quantity,
            venue_allocations,
            execution_sequence,
            expected_total_cost_bps,
            expected_fill_rate,
            expected_completion_time_ms,
            risk_metrics: RoutingRiskMetrics::default(),
            confidence_score: 0.8,
            alternative_plans: vec![],
        })
    }
}

// 默认实现
impl Default for MarketDataSnapshot {
    fn default() -> Self {
        Self {
            symbol: "".to_string(),
            bid: Decimal::ZERO,
            ask: Decimal::ZERO,
            last_price: Decimal::ZERO,
            bid_size: Decimal::ZERO,
            ask_size: Decimal::ZERO,
            volume_24h: Decimal::ZERO,
            high_24h: Decimal::ZERO,
            low_24h: Decimal::ZERO,
            price_change_24h: Decimal::ZERO,
            timestamp: Utc::now(),
        }
    }
}

impl Default for OrderBook {
    fn default() -> Self {
        Self {
            symbol: "".to_string(),
            bids: vec![],
            asks: vec![],
            timestamp: Utc::now(),
            sequence: 0,
        }
    }
}

impl Default for VenuePerformanceMetrics {
    fn default() -> Self {
        Self {
            fill_rate: 0.95,
            average_fill_time_ms: 100.0,
            slippage_bps: 2.0,
            market_impact_bps: 1.0,
            rejection_rate: 0.01,
            latency_p50_ms: 50.0,
            latency_p95_ms: 150.0,
            latency_p99_ms: 300.0,
            uptime_percentage: 0.999,
            total_volume_24h: Decimal::from(1000000),
            order_count_24h: 1000,
            error_rate: 0.001,
        }
    }
}

impl Default for VenueHealthStatus {
    fn default() -> Self {
        Self {
            overall_score: 1.0,
            liquidity_score: 1.0,
            latency_score: 1.0,
            reliability_score: 1.0,
            cost_score: 1.0,
            last_check: Utc::now(),
            issues: vec![],
        }
    }
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            liquidity_weight: 0.25,
            cost_weight: 0.20,
            latency_weight: 0.15,
            fill_rate_weight: 0.15,
            market_impact_weight: 0.10,
            reliability_weight: 0.05,
            volume_weight: 0.03,
            spread_weight: 0.03,
            depth_weight: 0.02,
            momentum_weight: 0.02,
        }
    }
}

impl Default for RiskProfile {
    fn default() -> Self {
        Self {
            max_slippage_bps: 20.0,
            max_market_impact_bps: 10.0,
            max_venue_concentration: 0.8,
            preferred_venues: vec![],
            restricted_venues: vec![],
            max_execution_time_ms: 60000,
        }
    }
}

impl Default for ExecutionPreferences {
    fn default() -> Self {
        Self {
            prefer_dark_pools: false,
            minimize_market_impact: true,
            maximize_fill_rate: true,
            cost_vs_speed_preference: 0.5,
            venue_diversification: true,
            iceberg_size: None,
        }
    }
}

impl Default for CostBreakdown {
    fn default() -> Self {
        Self {
            fixed_costs: Decimal::ZERO,
            variable_costs: Decimal::ZERO,
            fees: FeeBreakdown::default(),
            slippage: 0.0,
            market_impact: MarketImpactBreakdown::default(),
        }
    }
}

impl Default for FeeBreakdown {
    fn default() -> Self {
        Self {
            commission: Decimal::ZERO,
            exchange_fees: Decimal::ZERO,
            clearing_fees: Decimal::ZERO,
            regulatory_fees: Decimal::ZERO,
            other_fees: Decimal::ZERO,
        }
    }
}

impl Default for MarketImpactBreakdown {
    fn default() -> Self {
        Self {
            linear_impact: 0.0,
            square_root_impact: 0.0,
            log_impact: 0.0,
            participation_rate_impact: 0.0,
            volatility_impact: 0.0,
        }
    }
}

impl Default for DepthAnalysis {
    fn default() -> Self {
        Self {
            top_of_book_liquidity: 0.0,
            total_depth_10_levels: 0.0,
            weighted_average_spread: 0.0,
            liquidity_imbalance: 0.0,
            depth_quality_score: 0.0,
        }
    }
}

impl Default for RoutingRiskMetrics {
    fn default() -> Self {
        Self {
            concentration_risk: 0.0,
            liquidity_risk: 0.0,
            timing_risk: 0.0,
            execution_risk: 0.0,
            venue_risk: HashMap::new(),
            overall_risk_score: 0.0,
        }
    }
}

impl Default for AggregatedMetrics {
    fn default() -> Self {
        Self {
            expected_cost_bps: 0.0,
            cost_volatility_bps: 0.0,
            expected_fill_rate: 0.0,
            fill_rate_volatility: 0.0,
            expected_completion_time_ms: 0.0,
            completion_time_volatility_ms: 0.0,
            worst_case_cost_bps: 0.0,
            best_case_cost_bps: 0.0,
        }
    }
}

impl Default for SimulationRiskAnalysis {
    fn default() -> Self {
        Self {
            value_at_risk_bps: 0.0,
            expected_shortfall_bps: 0.0,
            tail_risk_scenarios: vec![],
            sensitivity_analysis: HashMap::new(),
        }
    }
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            supported_venues: vec![],
            score_weights: ScoreWeights::default(),
            update_frequency_ms: 1000,
            lookback_window_minutes: 60,
            min_liquidity_threshold: 0.3,
            max_latency_tolerance_ms: 1000,
            enable_adaptive_weights: true,
            enable_cost_prediction: true,
            enable_market_impact_modeling: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_smart_routing_scorer_creation() {
        let config = RoutingConfig::default();
        let scorer = SmartRoutingScorer::new(config).unwrap();
        
        assert!(true); // 基本创建测试
    }
    
    #[tokio::test]
    async fn test_venue_monitor_creation() {
        let venue_config = VenueConfig {
            venue_id: "test_venue".to_string(),
            venue_name: "Test Venue".to_string(),
            venue_type: VenueType::Exchange,
            supported_symbols: vec!["BTC-USD".to_string()],
            api_endpoints: ApiEndpoints {
                market_data: "https://api.test.com/market".to_string(),
                order_entry: "https://api.test.com/orders".to_string(),
                order_status: "https://api.test.com/orders/status".to_string(),
                fills: "https://api.test.com/fills".to_string(),
                websocket: None,
            },
            fee_structure: FeeStructure {
                maker_fee: Decimal::from_str("0.001").unwrap(),
                taker_fee: Decimal::from_str("0.002").unwrap(),
                tiered_fees: None,
                volume_discounts: None,
            },
            trading_hours: TradingHours {
                timezone: "UTC".to_string(),
                regular_hours: vec![],
                extended_hours: None,
                holidays: vec![],
            },
            connectivity: ConnectivityConfig {
                connection_type: ConnectionType::REST,
                max_connections: 10,
                heartbeat_interval_ms: 30000,
                reconnect_strategy: ReconnectStrategy {
                    max_retries: 3,
                    initial_delay_ms: 1000,
                    backoff_multiplier: 2.0,
                    max_delay_ms: 10000,
                },
                rate_limits: RateLimits {
                    orders_per_second: 10,
                    requests_per_minute: 600,
                    weight_per_request: None,
                    burst_limit: None,
                },
            },
            constraints: VenueConstraints {
                min_order_size: Decimal::from_str("0.001").unwrap(),
                max_order_size: None,
                tick_size: Decimal::from_str("0.01").unwrap(),
                lot_size: Decimal::from_str("0.001").unwrap(),
                max_position_size: None,
                allowed_order_types: vec![OrderType::Market, OrderType::Limit],
                supported_time_in_force: vec![TimeInForce::GTC, TimeInForce::IOC],
            },
            metadata: HashMap::new(),
        };
        
        let monitor = VenueMonitor::new(venue_config).await.unwrap();
        assert_eq!(monitor.venue_id, "test_venue");
    }
    
    #[tokio::test]
    async fn test_composite_score_model() {
        let weights = ScoreWeights::default();
        let model = CompositeScoreModel::new(weights);
        
        // 创建测试交易所监控器
        let venue_config = VenueConfig {
            venue_id: "test".to_string(),
            venue_name: "Test".to_string(),
            venue_type: VenueType::Exchange,
            supported_symbols: vec!["BTC-USD".to_string()],
            api_endpoints: ApiEndpoints {
                market_data: "test".to_string(),
                order_entry: "test".to_string(),
                order_status: "test".to_string(),
                fills: "test".to_string(),
                websocket: None,
            },
            fee_structure: FeeStructure {
                maker_fee: Decimal::from_str("0.001").unwrap(),
                taker_fee: Decimal::from_str("0.002").unwrap(),
                tiered_fees: None,
                volume_discounts: None,
            },
            trading_hours: TradingHours {
                timezone: "UTC".to_string(),
                regular_hours: vec![],
                extended_hours: None,
                holidays: vec![],
            },
            connectivity: ConnectivityConfig {
                connection_type: ConnectionType::REST,
                max_connections: 1,
                heartbeat_interval_ms: 1000,
                reconnect_strategy: ReconnectStrategy {
                    max_retries: 1,
                    initial_delay_ms: 1000,
                    backoff_multiplier: 1.0,
                    max_delay_ms: 1000,
                },
                rate_limits: RateLimits {
                    orders_per_second: 1,
                    requests_per_minute: 60,
                    weight_per_request: None,
                    burst_limit: None,
                },
            },
            constraints: VenueConstraints {
                min_order_size: Decimal::ONE,
                max_order_size: None,
                tick_size: Decimal::from_str("0.01").unwrap(),
                lot_size: Decimal::ONE,
                max_position_size: None,
                allowed_order_types: vec![OrderType::Market],
                supported_time_in_force: vec![TimeInForce::IOC],
            },
            metadata: HashMap::new(),
        };
        
        let monitor = VenueMonitor::new(venue_config).await.unwrap();
        
        let request = RoutingRequest {
            symbol: "BTC-USD".to_string(),
            side: TradeSide::Buy,
            quantity: Decimal::from(100),
            order_type: OrderType::Market,
            time_in_force: TimeInForce::IOC,
            price_limit: None,
            urgency: RoutingUrgency::Normal,
            strategy_id: "test".to_string(),
            client_id: "test".to_string(),
            risk_profile: RiskProfile::default(),
            execution_preferences: ExecutionPreferences::default(),
            metadata: HashMap::new(),
        };
        
        let score = model.calculate_score(&monitor, &request).await.unwrap();
        assert!(score.overall_score >= 0.0 && score.overall_score <= 1.0);
    }
}

use std::str::FromStr;