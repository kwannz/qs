use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use std::collections::HashMap;

/// Execution protocol for order management, position tracking, and risk control
/// Defines how trading orders are transmitted, executed, and monitored across the platform

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRequest {
    pub request_id: String,
    pub strategy_id: String,
    pub request_type: ExecutionRequestType,
    pub priority: ExecutionPriority,
    pub timestamp: DateTime<Utc>,
    pub metadata: ExecutionMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionRequestType {
    PlaceOrder(OrderRequest),
    CancelOrder(CancelRequest),
    ModifyOrder(ModifyRequest),
    ClosePosition(ClosePositionRequest),
    BatchOperation(BatchRequest),
    RiskCheck(RiskCheckRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionPriority {
    Low,
    Normal,
    High,
    Urgent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetadata {
    pub source: String,
    pub correlation_id: Option<String>,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub tags: HashMap<String, String>,
}

// Order management protocols

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderRequest {
    pub symbol: String,
    pub order_type: OrderType,
    pub side: OrderSide,
    pub quantity: f64,
    pub price: Option<f64>,
    pub time_in_force: TimeInForce,
    pub execution_instructions: ExecutionInstructions,
    pub risk_limits: OrderRiskLimits,
    pub routing: OrderRouting,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
    TrailingStop { trail_amount: f64 },
    Iceberg { display_quantity: f64 },
    TWAP { duration_minutes: u32 },
    VWAP { participation_rate: f64 },
    Custom { algorithm: String, parameters: HashMap<String, serde_json::Value> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeInForce {
    GTC, // Good Till Cancelled
    IOC, // Immediate Or Cancel
    FOK, // Fill Or Kill
    GTD { expire_time: DateTime<Utc> }, // Good Till Date
    Session, // Good for session
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionInstructions {
    pub allow_partial_fills: bool,
    pub min_fill_size: Option<f64>,
    pub max_fill_size: Option<f64>,
    pub execution_style: ExecutionStyle,
    pub latency_target: LatencyTarget,
    pub anti_gaming: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStyle {
    Aggressive, // Market impact acceptable for speed
    Passive,    // Minimize market impact
    Balanced,   // Balance speed and market impact
    Stealth,    // Minimize detection
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LatencyTarget {
    Ultra,   // < 1ms
    High,    // < 10ms
    Medium,  // < 100ms
    Low,     // < 1s
    Batch,   // No specific latency requirement
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderRiskLimits {
    pub max_order_value: Option<f64>,
    pub max_position_size: Option<f64>,
    pub price_deviation_tolerance: Option<f64>,
    pub reject_if_crossed: bool,
    pub pre_trade_risk_check: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderRouting {
    pub preferred_venues: Vec<String>,
    pub venue_selection_strategy: VenueSelectionStrategy,
    pub smart_order_routing: bool,
    pub dark_pool_preference: DarkPoolPreference,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VenueSelectionStrategy {
    BestPrice,
    LowestLatency,
    HighestLiquidity,
    ProportionalAllocation,
    Custom { strategy: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DarkPoolPreference {
    Avoid,
    Prefer,
    Only,
    Conditional { conditions: Vec<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelRequest {
    pub order_id: String,
    pub symbol: String,
    pub reason: CancelReason,
    pub force_cancel: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CancelReason {
    UserRequested,
    RiskLimit,
    MarketClosed,
    StrategySignal,
    SystemShutdown,
    Error,
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModifyRequest {
    pub order_id: String,
    pub symbol: String,
    pub modifications: OrderModifications,
    pub preserve_priority: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderModifications {
    pub new_quantity: Option<f64>,
    pub new_price: Option<f64>,
    pub new_time_in_force: Option<TimeInForce>,
    pub new_execution_instructions: Option<ExecutionInstructions>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosePositionRequest {
    pub symbol: String,
    pub close_method: CloseMethod,
    pub urgency: CloseUrgency,
    pub price_limit: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloseMethod {
    Market,
    Limit { price: f64 },
    StopLoss { trigger_price: f64 },
    TakeProfit { target_price: f64 },
    TWAP { duration_minutes: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloseUrgency {
    Normal,
    High,
    Emergency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequest {
    pub operations: Vec<BatchOperation>,
    pub execution_mode: BatchExecutionMode,
    pub rollback_on_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchOperation {
    PlaceOrder(OrderRequest),
    CancelOrder(CancelRequest),
    ModifyOrder(ModifyRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchExecutionMode {
    Sequential,
    Parallel,
    OptimalOrder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskCheckRequest {
    pub check_type: RiskCheckType,
    pub scope: RiskCheckScope,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskCheckType {
    PreTrade,
    PostTrade,
    Portfolio,
    Credit,
    Concentration,
    Compliance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskCheckScope {
    Order { order_id: String },
    Position { symbol: String },
    Portfolio,
    Account,
}

// Execution responses

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResponse {
    pub request_id: String,
    pub response_type: ExecutionResponseType,
    pub status: ExecutionStatus,
    pub timestamp: DateTime<Utc>,
    pub latency_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionResponseType {
    OrderAcknowledged(OrderAck),
    OrderRejected(OrderReject),
    OrderFilled(Fill),
    OrderCancelled(CancelAck),
    OrderModified(ModifyAck),
    PositionUpdate(PositionUpdate),
    RiskCheckResult(RiskCheckResult),
    BatchResult(BatchResult),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Success,
    Pending,
    Failed { error_code: String, error_message: String },
    PartialSuccess { details: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderAck {
    pub order_id: String,
    pub client_order_id: Option<String>,
    pub symbol: String,
    pub status: OrderStatus,
    pub exchange_order_id: Option<String>,
    pub venue: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
    Pending,
    Working,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderReject {
    pub client_order_id: Option<String>,
    pub symbol: String,
    pub reject_reason: ExecutionRejectReason,
    pub reject_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionRejectReason {
    InvalidSymbol,
    InvalidQuantity,
    InvalidPrice,
    InsufficientFunds,
    RiskLimit,
    MarketClosed,
    InvalidOrderType,
    SystemError,
    ComplianceViolation,
    DuplicateOrder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub order_id: String,
    pub execution_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub price: f64,
    pub venue: String,
    pub commission: f64,
    pub trade_time: DateTime<Utc>,
    pub is_final: bool,
    pub execution_report: ExecutionReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionReport {
    pub total_quantity: f64,
    pub filled_quantity: f64,
    pub remaining_quantity: f64,
    pub average_price: f64,
    pub last_price: f64,
    pub last_quantity: f64,
    pub cumulative_commission: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelAck {
    pub order_id: String,
    pub symbol: String,
    pub cancel_status: CancelStatus,
    pub remaining_quantity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CancelStatus {
    Cancelled,
    Rejected,
    TooLateToCancel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModifyAck {
    pub order_id: String,
    pub symbol: String,
    pub modify_status: ModifyStatus,
    pub new_order_details: Option<OrderDetails>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModifyStatus {
    Modified,
    Rejected,
    ReplacedWithNewOrder { new_order_id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderDetails {
    pub quantity: f64,
    pub price: Option<f64>,
    pub time_in_force: TimeInForce,
}

// Position management

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionUpdate {
    pub symbol: String,
    pub position_id: String,
    pub side: PositionSide,
    pub size: f64,
    pub average_price: f64,
    pub market_value: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub margin_used: f64,
    pub margin_available: f64,
    pub last_updated: DateTime<Utc>,
    pub risk_metrics: PositionRiskMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionSide {
    Long,
    Short,
    Flat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionRiskMetrics {
    pub var: f64,
    pub expected_shortfall: f64,
    pub beta: Option<f64>,
    pub delta: Option<f64>,
    pub gamma: Option<f64>,
    pub vega: Option<f64>,
    pub theta: Option<f64>,
}

// Risk management

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskCheckResult {
    pub check_type: RiskCheckType,
    pub result: RiskResult,
    pub risk_score: f64,
    pub violations: Vec<RiskViolation>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskResult {
    Approved,
    Rejected,
    Warning,
    RequiresApproval,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskViolation {
    pub violation_type: RiskViolationType,
    pub severity: RiskSeverity,
    pub description: String,
    pub threshold: f64,
    pub current_value: f64,
    pub suggested_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskViolationType {
    PositionLimit,
    ConcentrationLimit,
    VaRLimit,
    DrawdownLimit,
    LeverageLimit,
    CorrelationLimit,
    LiquidityRisk,
    OperationalRisk,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskSeverity {
    Low,
    Medium,
    High,
    Critical,
}

// Batch operations

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    pub total_operations: u32,
    pub successful_operations: u32,
    pub failed_operations: u32,
    pub operation_results: Vec<BatchOperationResult>,
    pub overall_status: BatchStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchStatus {
    AllSuccessful,
    PartialSuccess,
    AllFailed,
    RolledBack,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchOperationResult {
    pub operation_index: u32,
    pub operation_type: String,
    pub status: ExecutionStatus,
    pub result: Option<serde_json::Value>,
}

// Market data integration

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMarketData {
    pub symbol: String,
    pub bid: f64,
    pub ask: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub last_price: f64,
    pub volume: f64,
    pub timestamp: DateTime<Utc>,
    pub venue_data: HashMap<String, VenueMarketData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueMarketData {
    pub venue: String,
    pub bid: f64,
    pub ask: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub latency_ms: f32,
    pub liquidity_score: f64,
}

// Execution analytics

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionAnalytics {
    pub order_id: String,
    pub symbol: String,
    pub execution_quality: ExecutionQuality,
    pub cost_analysis: CostAnalysis,
    pub timing_analysis: TimingAnalysis,
    pub venue_analysis: VenueAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionQuality {
    pub implementation_shortfall: f64,
    pub price_improvement: f64,
    pub effective_spread: f64,
    pub market_impact: f64,
    pub timing_risk: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAnalysis {
    pub explicit_costs: f64,
    pub implicit_costs: f64,
    pub opportunity_cost: f64,
    pub total_cost: f64,
    pub cost_per_share: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingAnalysis {
    pub decision_time: DateTime<Utc>,
    pub order_sent_time: DateTime<Utc>,
    pub order_ack_time: DateTime<Utc>,
    pub first_fill_time: Option<DateTime<Utc>>,
    pub completion_time: Option<DateTime<Utc>>,
    pub total_latency_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueAnalysis {
    pub venues_used: Vec<String>,
    pub fill_rates: HashMap<String, f64>,
    pub average_latencies: HashMap<String, u32>,
    pub cost_by_venue: HashMap<String, f64>,
    pub optimal_venue: Option<String>,
}

// Utility functions

impl ExecutionRequest {
    pub fn new(
        strategy_id: String,
        request_type: ExecutionRequestType,
        priority: ExecutionPriority,
    ) -> Self {
        Self {
            request_id: Uuid::new_v4().to_string(),
            strategy_id,
            request_type,
            priority,
            timestamp: Utc::now(),
            metadata: ExecutionMetadata {
                source: "platform".to_string(),
                correlation_id: None,
                user_id: None,
                session_id: None,
                tags: HashMap::new(),
            },
        }
    }

    #[must_use]
    pub fn with_metadata(mut self, metadata: ExecutionMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

impl OrderRequest {
    pub fn market_order(symbol: String, side: OrderSide, quantity: f64) -> Self {
        Self {
            symbol,
            order_type: OrderType::Market,
            side,
            quantity,
            price: None,
            time_in_force: TimeInForce::IOC,
            execution_instructions: ExecutionInstructions {
                allow_partial_fills: true,
                min_fill_size: None,
                max_fill_size: None,
                execution_style: ExecutionStyle::Aggressive,
                latency_target: LatencyTarget::High,
                anti_gaming: false,
            },
            risk_limits: OrderRiskLimits {
                max_order_value: None,
                max_position_size: None,
                price_deviation_tolerance: Some(0.05),
                reject_if_crossed: false,
                pre_trade_risk_check: true,
            },
            routing: OrderRouting {
                preferred_venues: Vec::new(),
                venue_selection_strategy: VenueSelectionStrategy::BestPrice,
                smart_order_routing: true,
                dark_pool_preference: DarkPoolPreference::Conditional {
                    conditions: vec!["size > 1000".to_string()],
                },
            },
        }
    }

    pub fn limit_order(symbol: String, side: OrderSide, quantity: f64, price: f64) -> Self {
        Self {
            symbol,
            order_type: OrderType::Limit,
            side,
            quantity,
            price: Some(price),
            time_in_force: TimeInForce::GTC,
            execution_instructions: ExecutionInstructions {
                allow_partial_fills: true,
                min_fill_size: None,
                max_fill_size: None,
                execution_style: ExecutionStyle::Passive,
                latency_target: LatencyTarget::Medium,
                anti_gaming: true,
            },
            risk_limits: OrderRiskLimits {
                max_order_value: None,
                max_position_size: None,
                price_deviation_tolerance: Some(0.02),
                reject_if_crossed: true,
                pre_trade_risk_check: true,
            },
            routing: OrderRouting {
                preferred_venues: Vec::new(),
                venue_selection_strategy: VenueSelectionStrategy::LowestLatency,
                smart_order_routing: true,
                dark_pool_preference: DarkPoolPreference::Prefer,
            },
        }
    }
}

pub fn create_execution_metadata(
    source: &str,
    user_id: Option<String>,
    session_id: Option<String>,
) -> ExecutionMetadata {
    ExecutionMetadata {
        source: source.to_string(),
        correlation_id: Some(Uuid::new_v4().to_string()),
        user_id,
        session_id,
        tags: HashMap::new(),
    }
}