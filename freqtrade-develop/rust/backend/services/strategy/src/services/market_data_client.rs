use anyhow::Result;
use chrono::{DateTime, Utc};
use reqwest::Client;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{sleep, Duration};

use crate::config::Config;

// Type aliases to simplify complex types
type MarketDataCache = Arc<RwLock<HashMap<String, (MarketData, DateTime<Utc>)>>>;
type HistoricalCache = Arc<RwLock<HashMap<String, Vec<Ohlcv>>>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub symbol: String,
    pub exchange: String,
    pub price: Decimal,
    pub volume: Decimal,
    pub timestamp: DateTime<Utc>,
    pub bid: Option<Decimal>,
    pub ask: Option<Decimal>,
    pub high_24h: Option<Decimal>,
    pub low_24h: Option<Decimal>,
    pub change_24h: Option<Decimal>,
    pub change_24h_pct: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ohlcv {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
}

#[allow(dead_code)]
pub struct MarketDataClient {
    #[allow(dead_code)]
    config: Arc<Config>,
    #[allow(dead_code)]
    http_client: Client,
    #[allow(dead_code)]
    cache: MarketDataCache,
    #[allow(dead_code)]
    historical_cache: HistoricalCache,
}

#[allow(dead_code)]
impl MarketDataClient {
    pub async fn new(config: &Config) -> Result<Self> {
        let http_client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()?;

        Ok(Self {
            config: Arc::new(config.clone()),
            http_client,
            cache: Arc::new(RwLock::new(HashMap::new())),
            historical_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    // Real-time Market Data
    pub async fn get_latest_data(&self, symbols: &[String]) -> Result<HashMap<String, MarketData>> {
        let mut result = HashMap::new();
        
        // Check cache first
        let cache = self.cache.read().await;
        let cache_ttl = Duration::from_secs(self.config.strategy.market_data_cache_ttl);
        let now = Utc::now();
        
        let mut symbols_to_fetch = Vec::new();
        
        for symbol in symbols {
            if let Some((data, cached_at)) = cache.get(symbol) {
                if now.signed_duration_since(*cached_at).to_std().unwrap_or(Duration::MAX) < cache_ttl {
                    result.insert(symbol.clone(), data.clone());
                } else {
                    symbols_to_fetch.push(symbol.clone());
                }
            } else {
                symbols_to_fetch.push(symbol.clone());
            }
        }
        
        drop(cache);
        
        // Fetch missing symbols
        if !symbols_to_fetch.is_empty() {
            let fetched_data = self.fetch_market_data(&symbols_to_fetch).await?;
            
            // Update cache
            let mut cache = self.cache.write().await;
            for (symbol, data) in &fetched_data {
                cache.insert(symbol.clone(), (data.clone(), now));
                result.insert(symbol.clone(), data.clone());
            }
        }
        
        Ok(result)
    }

    pub async fn get_single_price(&self, symbol: &str) -> Result<Decimal> {
        let data = self.get_latest_data(&[symbol.to_string()]).await?;
        
        data.get(symbol)
            .map(|d| d.price)
            .ok_or_else(|| anyhow::anyhow!("Price not available for symbol: {}", symbol))
    }

    // Historical Data
    pub async fn get_historical_data(
        &self,
        symbol: &str,
        timeframe: &str,
        limit: Option<u32>,
    ) -> Result<Vec<Ohlcv>> {
        let cache_key = format!("{symbol}:{timeframe}");
        
        // Check cache
        let historical_cache = self.historical_cache.read().await;
        if let Some(cached_data) = historical_cache.get(&cache_key) {
            if !cached_data.is_empty() {
                let Some(last_data) = cached_data.last() else {
                    drop(historical_cache);
                    return self.fetch_historical_data(symbol, timeframe, limit).await;
                };
                let last_update = last_data.timestamp;
                let cache_age = Utc::now().signed_duration_since(last_update);
                
                // Use cache if data is less than 5 minutes old
                if cache_age.num_minutes() < 5 {
                    let limit = limit.unwrap_or(cached_data.len() as u32) as usize;
                    return Ok(cached_data.iter().rev().take(limit).rev().cloned().collect());
                }
            }
        }
        
        drop(historical_cache);
        
        // Fetch from API
        let data = self.fetch_historical_data(symbol, timeframe, limit).await?;
        
        // Update cache
        let mut historical_cache = self.historical_cache.write().await;
        historical_cache.insert(cache_key, data.clone());
        
        Ok(data)
    }

    // Utility Methods
    pub async fn get_orderbook(&self, symbol: &str, depth: Option<u32>) -> Result<OrderBook> {
        // Simulate orderbook data - in production, this would fetch from exchange API
        let latest_data = self.get_latest_data(&[symbol.to_string()]).await?;
        
        if let Some(data) = latest_data.get(symbol) {
            let spread = data.ask.unwrap_or(data.price) - data.bid.unwrap_or(data.price);
            let step = spread / Decimal::from(10); // 10 levels
            
            let mut bids = Vec::new();
            let mut asks = Vec::new();
            
            let depth = depth.unwrap_or(10).min(20) as usize;
            
            for i in 0..depth {
                let price_offset = Decimal::from(i) * step;
                
                bids.push(OrderBookLevel {
                    price: data.bid.unwrap_or(data.price) - price_offset,
                    quantity: Decimal::from(100 + i * 10), // Simulate quantity
                });
                
                asks.push(OrderBookLevel {
                    price: data.ask.unwrap_or(data.price) + price_offset,
                    quantity: Decimal::from(100 + i * 10),
                });
            }
            
            Ok(OrderBook {
                symbol: symbol.to_string(),
                bids,
                asks,
                timestamp: Utc::now(),
            })
        } else {
            Err(anyhow::anyhow!("Market data not available for symbol: {}", symbol))
        }
    }

    pub async fn get_24h_stats(&self, symbols: &[String]) -> Result<HashMap<String, DailyStats>> {
        let market_data = self.get_latest_data(symbols).await?;
        
        let mut stats = HashMap::new();
        for (symbol, data) in market_data {
            stats.insert(symbol.clone(), DailyStats {
                symbol: symbol.clone(),
                price: data.price,
                volume_24h: data.volume,
                high_24h: data.high_24h.unwrap_or(data.price),
                low_24h: data.low_24h.unwrap_or(data.price),
                change_24h: data.change_24h.unwrap_or(Decimal::ZERO),
                change_24h_pct: data.change_24h_pct.unwrap_or(0.0),
                timestamp: data.timestamp,
            });
        }
        
        Ok(stats)
    }

    // Private Methods
    async fn fetch_market_data(&self, symbols: &[String]) -> Result<HashMap<String, MarketData>> {
        // Simulate market data fetching - in production, integrate with real exchanges
        let mut result = HashMap::new();
        
        for symbol in symbols {
            // Simulate API call delay
            sleep(Duration::from_millis(10)).await;
            
            // Generate simulated market data
            let base_price = match symbol.as_str() {
                "BTC/USDT" => Decimal::from(45000),
                "ETH/USDT" => Decimal::from(3000),
                "BNB/USDT" => Decimal::from(400),
                "ADA/USDT" => Decimal::from_f64_retain(1.2).unwrap_or_default(),
                "SOL/USDT" => Decimal::from(100),
                _ => Decimal::from(100), // Default price
            };
            
            // Add some random variation (±2%)
            let variation = (fastrand::f64() - 0.5) * 0.04; // -2% to +2%
            let current_price = base_price * (Decimal::from_f64_retain(1.0 + variation).unwrap_or(Decimal::ONE));
            
            let market_data = MarketData {
                symbol: symbol.clone(),
                exchange: "binance".to_string(),
                price: current_price,
                volume: base_price * Decimal::from(1000 + fastrand::u32(0..5000)),
                timestamp: Utc::now(),
                bid: Some(current_price * Decimal::from_f64_retain(0.9995).unwrap_or(Decimal::ONE)),
                ask: Some(current_price * Decimal::from_f64_retain(1.0005).unwrap_or(Decimal::ONE)),
                high_24h: Some(current_price * Decimal::from_f64_retain(1.05).unwrap_or(Decimal::ONE)),
                low_24h: Some(current_price * Decimal::from_f64_retain(0.95).unwrap_or(Decimal::ONE)),
                change_24h: Some(current_price * Decimal::from_f64_retain(variation).unwrap_or_default()),
                change_24h_pct: Some(variation * 100.0),
            };
            
            result.insert(symbol.clone(), market_data);
        }
        
        Ok(result)
    }

    async fn fetch_historical_data(&self, symbol: &str, timeframe: &str, limit: Option<u32>) -> Result<Vec<Ohlcv>> {
        let limit = limit.unwrap_or(100).min(1000) as usize;
        let mut result = Vec::new();
        
        // Get current price as base
        let current_data = self.fetch_market_data(&[symbol.to_string()]).await?;
        let base_price = current_data.get(symbol)
            .map(|d| d.price)
            .unwrap_or(Decimal::from(100));
        
        // Generate historical data (simplified simulation)
        let interval_minutes = match timeframe {
            "1m" => 1,
            "5m" => 5,
            "15m" => 15,
            "1h" => 60,
            "4h" => 240,
            "1d" => 1440,
            _ => 60, // default to 1h
        };
        
        let mut current_time = Utc::now() - chrono::Duration::minutes((limit * interval_minutes) as i64);
        let mut last_close = base_price * Decimal::from_f64_retain(0.95).unwrap_or(Decimal::ONE); // Start 5% lower
        
        for _ in 0..limit {
            // Generate Ohlcv data with some realistic movement
            let open = last_close;
            let change = (fastrand::f64() - 0.5) * 0.02; // ±1% change
            let close = open * Decimal::from_f64_retain(1.0 + change).unwrap_or(Decimal::ONE);
            
            let high_change = fastrand::f64() * 0.01; // Up to 0.5% higher than open/close
            let low_change = fastrand::f64() * 0.01; // Up to 0.5% lower than open/close
            
            let high = open.max(close) * Decimal::from_f64_retain(1.0 + high_change).unwrap_or(Decimal::ONE);
            let low = open.min(close) * Decimal::from_f64_retain(1.0 - low_change).unwrap_or(Decimal::ONE);
            
            let volume = base_price * Decimal::from(500 + fastrand::u32(0..1000));
            
            result.push(Ohlcv {
                symbol: symbol.to_string(),
                timestamp: current_time,
                open,
                high,
                low,
                close,
                volume,
            });
            
            current_time += chrono::Duration::minutes(interval_minutes as i64);
            last_close = close;
        }
        
        Ok(result)
    }
}

// Supporting Data Structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: Decimal,
    pub quantity: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyStats {
    pub symbol: String,
    pub price: Decimal,
    pub volume_24h: Decimal,
    pub high_24h: Decimal,
    pub low_24h: Decimal,
    pub change_24h: Decimal,
    pub change_24h_pct: f64,
    pub timestamp: DateTime<Utc>,
}