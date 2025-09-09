use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use std::collections::HashMap;

/// Options for configuring quality features
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct QualityOptions {
    pub require_sequence: bool,
    pub detect_gaps: bool,
    pub validate_timestamps: bool,
    pub check_duplicates: bool,
}

impl Default for QualityOptions {
    fn default() -> Self {
        Self {
            require_sequence: true,
            detect_gaps: true,
            validate_timestamps: true,
            check_duplicates: true,
        }
    }
}

/// Market data protocol for real-time and historical data streaming
/// Defines how market data flows through the platform ecosystem

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataRequest {
    pub request_id: String,
    pub request_type: DataRequestType,
    pub subscription_config: SubscriptionConfig,
    pub delivery_config: DeliveryConfig,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataRequestType {
    Subscribe(SubscriptionRequest),
    Unsubscribe(UnsubscriptionRequest),
    Snapshot(SnapshotRequest),
    Historical(HistoricalRequest),
    Bulk(BulkRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionRequest {
    pub symbols: Vec<String>,
    pub data_types: Vec<MarketDataType>,
    pub venues: Vec<String>,
    pub filters: Vec<DataFilter>,
    pub aggregation: Option<AggregationConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketDataType {
    /// Level 1 market data (best bid/offer)
    Level1,
    /// Level 2 market data (order book)
    Level2 { depth: u32 },
    /// Full order book
    FullOrderBook,
    /// Trade data
    Trades,
    /// OHLC bars
    Bars { interval: TimeInterval },
    /// Volume data
    Volume,
    /// Market statistics
    Statistics,
    /// Index data
    Index,
    /// Options data
    Options { underlying: String },
    /// Futures data
    Futures { contract_month: String },
    /// Custom data type
    Custom { data_type: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeInterval {
    Milliseconds(u64),
    Seconds(u32),
    Minutes(u32),
    Hours(u32),
    Days(u32),
    Weeks(u32),
    Months(u32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFilter {
    pub filter_type: FilterType,
    pub condition: FilterCondition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterType {
    Price,
    Volume,
    Time,
    Venue,
    Symbol,
    TradeSize,
    Spread,
    Custom { field: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterCondition {
    GreaterThan(f64),
    LessThan(f64),
    Between { min: f64, max: f64 },
    Equals(f64),
    Contains(String),
    Matches(String), // Regex pattern
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationConfig {
    pub method: AggregationMethod,
    pub window: TimeWindow,
    pub emit_policy: EmitPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationMethod {
    OHLC,
    VWAP,
    TWAP,
    Volume,
    Count,
    Custom { algorithm: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    pub duration: TimeInterval,
    pub alignment: WindowAlignment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowAlignment {
    Calendar, // Align to calendar boundaries
    Session,  // Align to trading sessions
    Custom { offset: i64 }, // Custom offset in milliseconds
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmitPolicy {
    OnClose,     // Emit when window closes
    Continuous,  // Emit on every update
    OnInterval { interval: TimeInterval }, // Emit at regular intervals
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsubscriptionRequest {
    pub subscription_id: String,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotRequest {
    pub symbols: Vec<String>,
    pub data_types: Vec<MarketDataType>,
    pub venues: Vec<String>,
    pub include_stale: bool,
    pub max_age_seconds: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalRequest {
    pub symbols: Vec<String>,
    pub data_types: Vec<MarketDataType>,
    pub time_range: TimeRange,
    pub venues: Vec<String>,
    pub format: DataFormat,
    pub compression: Option<CompressionType>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub timezone: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataFormat {
    Json,
    Csv,
    Parquet,
    Avro,
    MessagePack,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionType {
    Gzip,
    Lz4,
    Zstd,
    Snappy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkRequest {
    pub requests: Vec<DataRequestType>,
    pub execution_mode: BulkExecutionMode,
    pub callback_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BulkExecutionMode {
    Sequential,
    Parallel { max_concurrency: u32 },
    Batch { batch_size: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionConfig {
    pub priority: DataPriority,
    pub conflation: ConflationPolicy,
    pub recovery: RecoveryConfig,
    pub quality: QualityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataPriority {
    Low,
    Normal,
    High,
    RealTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflationPolicy {
    None,         // No conflation, send all updates
    Latest,       // Only send latest update if queue backs up
    Time { interval: TimeInterval }, // Conflate based on time
    Count { max_updates: u32 },      // Conflate based on update count
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryConfig {
    pub enabled: bool,
    pub max_recovery_time: TimeInterval,
    pub recovery_method: RecoveryMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryMethod {
    Snapshot,     // Request snapshot on reconnect
    Replay,       // Replay missed messages
    BestEffort,   // Use whatever is available
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityConfig {
    pub quality_features: QualityFeatures,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityFeatures {
    pub sequence_validation: SequenceValidation,
    pub gap_detection: GapDetection,
    pub timestamp_validation: TimestampValidation,
    pub duplicate_checking: DuplicateChecking,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SequenceValidation {
    Required,
    Optional,
    Disabled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GapDetection {
    Enabled,
    Disabled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimestampValidation {
    Strict,
    Loose,
    Disabled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DuplicateChecking {
    Enabled,
    Disabled,
}

impl QualityConfig {
    /// Backward compatibility method for require_sequence boolean
    pub fn require_sequence(&self) -> bool {
        matches!(self.quality_features.sequence_validation, SequenceValidation::Required)
    }

    /// Backward compatibility method for detect_gaps boolean
    pub fn detect_gaps(&self) -> bool {
        matches!(self.quality_features.gap_detection, GapDetection::Enabled)
    }

    /// Backward compatibility method for validate_timestamps boolean
    pub fn validate_timestamps(&self) -> bool {
        !matches!(self.quality_features.timestamp_validation, TimestampValidation::Disabled)
    }

    /// Backward compatibility method for check_duplicates boolean
    pub fn check_duplicates(&self) -> bool {
        matches!(self.quality_features.duplicate_checking, DuplicateChecking::Enabled)
    }

    /// Create a QualityConfig from quality options
    pub fn from_quality_options(options: QualityOptions) -> Self {
        Self {
            quality_features: QualityFeatures {
                sequence_validation: if options.require_sequence {
                    SequenceValidation::Required
                } else {
                    SequenceValidation::Disabled
                },
                gap_detection: if options.detect_gaps {
                    GapDetection::Enabled
                } else {
                    GapDetection::Disabled
                },
                timestamp_validation: if options.validate_timestamps {
                    TimestampValidation::Strict
                } else {
                    TimestampValidation::Disabled
                },
                duplicate_checking: if options.check_duplicates {
                    DuplicateChecking::Enabled
                } else {
                    DuplicateChecking::Disabled
                },
            },
        }
    }

    /// Create a QualityConfig from boolean values for backward compatibility
    #[allow(clippy::fn_params_excessive_bools)]
    pub fn from_bools(
        require_sequence: bool,
        detect_gaps: bool,
        validate_timestamps: bool,
        check_duplicates: bool,
    ) -> Self {
        Self::from_quality_options(QualityOptions {
            require_sequence,
            detect_gaps,
            validate_timestamps,
            check_duplicates,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryConfig {
    pub transport: TransportType,
    pub encoding: EncodingType,
    pub batching: BatchingConfig,
    pub reliability: ReliabilityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransportType {
    WebSocket { compression: bool },
    Http { polling_interval: TimeInterval },
    Udp { multicast: bool },
    Tcp,
    MessageQueue { queue_name: String },
    Custom { protocol: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncodingType {
    Json,
    MessagePack,
    Protobuf,
    Avro,
    Binary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchingConfig {
    pub enabled: bool,
    pub max_batch_size: u32,
    pub max_wait_ms: u32,
    pub batch_by_symbol: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityConfig {
    pub delivery_guarantee: DeliveryGuarantee,
    pub acknowledgements: bool,
    pub retry_config: RetryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeliveryGuarantee {
    AtMostOnce,
    AtLeastOnce,
    ExactlyOnce,
    BestEffort,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub backoff_strategy: MarketDataBackoffStrategy,
    pub jitter: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketDataBackoffStrategy {
    Fixed { delay_ms: u32 },
    Linear { increment_ms: u32 },
    Exponential { base_ms: u32, multiplier: f64 },
}

// Market data messages

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataMessage {
    pub message_id: String,
    pub subscription_id: Option<String>,
    pub symbol: String,
    pub venue: String,
    pub message_type: MarketDataMessageType,
    pub timestamp: DateTime<Utc>,
    pub sequence_number: Option<u64>,
    pub metadata: MessageMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketDataMessageType {
    Quote(Quote),
    Trade(Trade),
    OrderBook(OrderBook),
    Bar(Bar),
    Statistics(MarketStatistics),
    Status(MarketStatus),
    News(NewsItem),
    Corporate(CorporateAction),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMetadata {
    pub source: String,
    pub latency_ms: Option<u32>,
    pub quality_flags: Vec<QualityFlag>,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityFlag {
    Delayed,
    Interpolated,
    Stale,
    OutOfSequence,
    Duplicate,
    Corrected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quote {
    pub bid_price: f64,
    pub bid_size: f64,
    pub ask_price: f64,
    pub ask_size: f64,
    pub bid_count: Option<u32>,
    pub ask_count: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub price: f64,
    pub size: f64,
    pub side: TradeSide,
    pub trade_id: String,
    pub conditions: Vec<TradeCondition>,
    pub buyer_maker: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeCondition {
    RegularTrade,
    Cash,
    NextDay,
    Seller,
    SoldLast,
    OutOfSequence,
    AveragePrice,
    AutoExecution,
    CrossTrade,
    Intermarket,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
    pub checksum: Option<String>,
    pub update_type: UpdateType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: f64,
    pub size: f64,
    pub order_count: Option<u32>,
    pub update_action: UpdateAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateAction {
    Insert,
    Update,
    Delete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateType {
    Snapshot,
    Incremental,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub vwap: Option<f64>,
    pub trade_count: Option<u32>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub interval: TimeInterval,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketStatistics {
    pub open_price: Option<f64>,
    pub high_price: Option<f64>,
    pub low_price: Option<f64>,
    pub close_price: Option<f64>,
    pub volume: f64,
    pub vwap: Option<f64>,
    pub change: Option<f64>,
    pub change_percent: Option<f64>,
    pub total_trades: Option<u64>,
    pub market_cap: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketStatus {
    pub status: TradingStatus,
    pub reason: Option<String>,
    pub expected_resume: Option<DateTime<Utc>>,
    pub session_info: Option<SessionInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradingStatus {
    PreOpen,
    Open,
    Closed,
    Halted,
    Suspended,
    PostClose,
    Maintenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub session_start: DateTime<Utc>,
    pub session_end: DateTime<Utc>,
    pub timezone: String,
    pub session_type: SessionType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionType {
    Regular,
    Extended,
    PreMarket,
    PostMarket,
    Auction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsItem {
    pub headline: String,
    pub body: Option<String>,
    pub source: String,
    pub urgency: NewsUrgency,
    pub categories: Vec<String>,
    pub related_symbols: Vec<String>,
    pub url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NewsUrgency {
    Low,
    Normal,
    High,
    Alert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorporateAction {
    pub action_type: CorporateActionType,
    pub effective_date: DateTime<Utc>,
    pub ex_date: Option<DateTime<Utc>>,
    pub record_date: Option<DateTime<Utc>>,
    pub details: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorporateActionType {
    Dividend,
    Split,
    Spinoff,
    Merger,
    Rights,
    SpecialDividend,
    StockDividend,
    Delisting,
}

// Response messages

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataResponse {
    pub request_id: String,
    pub response_type: ResponseType,
    pub status: ResponseStatus,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseType {
    SubscriptionAck(SubscriptionAck),
    SubscriptionReject(SubscriptionReject),
    Snapshot(SnapshotResponse),
    Historical(HistoricalResponse),
    Status(StatusUpdate),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseStatus {
    Success,
    Pending,
    Failed { error_code: String, error_message: String },
    PartialSuccess { details: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionAck {
    pub subscription_id: String,
    pub symbols: Vec<String>,
    pub data_types: Vec<MarketDataType>,
    pub venues: Vec<String>,
    pub estimated_rate: f64, // Messages per second
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionReject {
    pub reject_reason: MarketDataRejectReason,
    pub rejected_symbols: Vec<String>,
    pub rejected_data_types: Vec<MarketDataType>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketDataRejectReason {
    InvalidSymbol,
    InvalidDataType,
    InvalidVenue,
    PermissionDenied,
    QuotaExceeded,
    RateLimitExceeded,
    ServiceUnavailable,
    InternalError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotResponse {
    pub data: Vec<MarketDataMessage>,
    pub total_symbols: u32,
    pub snapshot_time: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalResponse {
    pub data_location: DataLocation,
    pub record_count: u64,
    pub file_size_bytes: u64,
    pub compression_ratio: Option<f64>,
    pub download_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataLocation {
    Inline { data: Vec<MarketDataMessage> },
    File { path: String },
    Url { url: String, expires_at: DateTime<Utc> },
    Stream { stream_id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusUpdate {
    pub subscription_id: String,
    pub status: SubscriptionStatus,
    pub message: Option<String>,
    pub affected_symbols: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubscriptionStatus {
    Active,
    Paused,
    Recovering,
    Error,
    Terminated,
}

// Utility functions and implementations

impl MarketDataRequest {
    pub fn new(request_type: DataRequestType) -> Self {
        Self {
            request_id: Uuid::new_v4().to_string(),
            request_type,
            subscription_config: SubscriptionConfig::default(),
            delivery_config: DeliveryConfig::default(),
            timestamp: Utc::now(),
        }
    }
}

impl Default for SubscriptionConfig {
    fn default() -> Self {
        Self {
            priority: DataPriority::Normal,
            conflation: ConflationPolicy::None,
            recovery: RecoveryConfig {
                enabled: true,
                max_recovery_time: TimeInterval::Minutes(5),
                recovery_method: RecoveryMethod::Snapshot,
            },
            quality: QualityConfig::from_bools(false, true, true, true),
        }
    }
}

impl Default for DeliveryConfig {
    fn default() -> Self {
        Self {
            transport: TransportType::WebSocket { compression: true },
            encoding: EncodingType::Json,
            batching: BatchingConfig {
                enabled: false,
                max_batch_size: 100,
                max_wait_ms: 10,
                batch_by_symbol: false,
            },
            reliability: ReliabilityConfig {
                delivery_guarantee: DeliveryGuarantee::AtLeastOnce,
                acknowledgements: false,
                retry_config: RetryConfig {
                    max_retries: 3,
                    backoff_strategy: MarketDataBackoffStrategy::Exponential {
                        base_ms: 100,
                        multiplier: 2.0,
                    },
                    jitter: true,
                },
            },
        }
    }
}

pub fn create_level1_subscription(symbols: Vec<String>, venues: Vec<String>) -> SubscriptionRequest {
    SubscriptionRequest {
        symbols,
        data_types: vec![MarketDataType::Level1, MarketDataType::Trades],
        venues,
        filters: Vec::new(),
        aggregation: None,
    }
}

pub fn create_bars_subscription(
    symbols: Vec<String>,
    interval: TimeInterval,
    venues: Vec<String>,
) -> SubscriptionRequest {
    SubscriptionRequest {
        symbols,
        data_types: vec![MarketDataType::Bars { interval }],
        venues,
        filters: Vec::new(),
        aggregation: None,
    }
}

pub fn create_historical_request(
    symbols: Vec<String>,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    data_types: Vec<MarketDataType>,
) -> HistoricalRequest {
    HistoricalRequest {
        symbols,
        data_types,
        time_range: TimeRange {
            start,
            end,
            timezone: Some("UTC".to_string()),
        },
        venues: Vec::new(),
        format: DataFormat::Json,
        compression: Some(CompressionType::Gzip),
    }
}