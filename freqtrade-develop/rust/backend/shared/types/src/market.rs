use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 市场数据时间戳
pub type MarketTimestamp = DateTime<Utc>;

/// 价格类型
pub type Price = Decimal;

/// 数量类型
pub type Quantity = Decimal;

/// 交易对
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Symbol {
    pub base: String,
    pub quote: String,
}

impl Symbol {
    pub fn new(base: &str, quote: &str) -> Self {
        Self {
            base: base.to_uppercase(),
            quote: quote.to_uppercase(),
        }
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.base, self.quote)
    }
}

impl std::str::FromStr for Symbol {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // 简单解析，实际应该更复杂
        if s.len() >= 6 {
            let base = &s[..3];
            let quote = &s[3..];
            Ok(Symbol::new(base, quote))
        } else {
            Err(format!("Invalid symbol format: {s}"))
        }
    }
}

/// K线数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub symbol: Symbol,
    pub timestamp: MarketTimestamp,
    pub open: Price,
    pub high: Price,
    pub low: Price,
    pub close: Price,
    pub volume: Quantity,
    pub quote_volume: Option<Quantity>,
    pub trades_count: Option<u64>,
    pub interval: KlineInterval,
}

/// K线周期
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KlineInterval {
    #[serde(rename = "1m")]
    OneMinute,
    #[serde(rename = "3m")]
    ThreeMinutes,
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "15m")]
    FifteenMinutes,
    #[serde(rename = "30m")]
    ThirtyMinutes,
    #[serde(rename = "1h")]
    OneHour,
    #[serde(rename = "2h")]
    TwoHours,
    #[serde(rename = "4h")]
    FourHours,
    #[serde(rename = "6h")]
    SixHours,
    #[serde(rename = "8h")]
    EightHours,
    #[serde(rename = "12h")]
    TwelveHours,
    #[serde(rename = "1d")]
    OneDay,
    #[serde(rename = "3d")]
    ThreeDays,
    #[serde(rename = "1w")]
    OneWeek,
    #[serde(rename = "1M")]
    OneMonth,
}

impl KlineInterval {
    pub fn to_seconds(&self) -> u64 {
        match self {
            KlineInterval::OneMinute => 60,
            KlineInterval::ThreeMinutes => 180,
            KlineInterval::FiveMinutes => 300,
            KlineInterval::FifteenMinutes => 900,
            KlineInterval::ThirtyMinutes => 1800,
            KlineInterval::OneHour => 3600,
            KlineInterval::TwoHours => 7200,
            KlineInterval::FourHours => 14400,
            KlineInterval::SixHours => 21600,
            KlineInterval::EightHours => 28800,
            KlineInterval::TwelveHours => 43200,
            KlineInterval::OneDay => 86400,
            KlineInterval::ThreeDays => 259200,
            KlineInterval::OneWeek => 604800,
            KlineInterval::OneMonth => 2629746, // 30.44 days average
        }
    }
}

/// Tick数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tick {
    pub symbol: Symbol,
    pub timestamp: MarketTimestamp,
    pub price: Price,
    pub quantity: Quantity,
    pub side: TradeSide,
    pub trade_id: Option<String>,
}

/// 交易方向
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// 订单簿深度
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: Symbol,
    pub timestamp: MarketTimestamp,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub last_update_id: Option<u64>,
}

/// 订单簿价格档位
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: Price,
    pub quantity: Quantity,
}

/// 24小时统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker24h {
    pub symbol: Symbol,
    pub timestamp: MarketTimestamp,
    pub price_change: Price,
    pub price_change_percent: Decimal,
    pub weighted_avg_price: Price,
    pub prev_close_price: Price,
    pub last_price: Price,
    pub last_quantity: Quantity,
    pub bid_price: Price,
    pub bid_quantity: Quantity,
    pub ask_price: Price,
    pub ask_quantity: Quantity,
    pub open_price: Price,
    pub high_price: Price,
    pub low_price: Price,
    pub volume: Quantity,
    pub quote_volume: Quantity,
    pub open_time: MarketTimestamp,
    pub close_time: MarketTimestamp,
    pub first_id: u64,
    pub last_id: u64,
    pub count: u64,
}

/// 交易所信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Exchange {
    pub id: String,
    pub name: String,
    pub enabled: bool,
    pub sandbox: bool,
    pub rate_limits: HashMap<String, RateLimit>,
    pub supported_intervals: Vec<KlineInterval>,
    pub min_notional: Option<Decimal>,
    pub max_notional: Option<Decimal>,
    pub fee_rates: FeeRates,
}

/// 费率信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeRates {
    pub maker: Decimal,
    pub taker: Decimal,
}

/// 速率限制
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    pub requests: u32,
    pub interval: u32, // seconds
    pub weight: Option<u32>,
}

/// 市场状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MarketStatus {
    Open,
    Closed,
    PreOpen,
    PreClose,
    Maintenance,
}

/// 交易对信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolInfo {
    pub symbol: Symbol,
    pub status: MarketStatus,
    pub base_asset: String,
    pub quote_asset: String,
    pub base_precision: u8,
    pub quote_precision: u8,
    pub min_quantity: Quantity,
    pub max_quantity: Quantity,
    pub min_notional: Quantity,
    pub max_notional: Option<Quantity>,
    pub tick_size: Price,
    pub lot_size: Quantity,
    pub is_spot_trading: bool,
    pub is_margin_trading: bool,
}

/// 市场数据请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataRequest {
    pub symbol: Symbol,
    pub interval: Option<KlineInterval>,
    pub start_time: Option<MarketTimestamp>,
    pub end_time: Option<MarketTimestamp>,
    pub limit: Option<u32>,
}

/// 市场数据响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataResponse<T> {
    pub symbol: Symbol,
    pub data: Vec<T>,
    pub timestamp: MarketTimestamp,
    pub count: usize,
}

/// 聚合交易数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggTrade {
    pub id: u64,
    pub symbol: Symbol,
    pub price: Price,
    pub quantity: Quantity,
    pub first_trade_id: u64,
    pub last_trade_id: u64,
    pub timestamp: MarketTimestamp,
    pub is_buyer_maker: bool,
}

/// 资金费率
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRate {
    pub symbol: Symbol,
    pub timestamp: MarketTimestamp,
    pub funding_rate: Decimal,
    pub mark_price: Price,
}

/// 持仓量
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenInterest {
    pub symbol: Symbol,
    pub timestamp: MarketTimestamp,
    pub open_interest: Quantity,
    pub mark_price: Price,
}

/// 多空比率
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongShortRatio {
    pub symbol: Symbol,
    pub timestamp: MarketTimestamp,
    pub long_ratio: Decimal,
    pub short_ratio: Decimal,
    pub long_account_ratio: Decimal,
    pub short_account_ratio: Decimal,
}