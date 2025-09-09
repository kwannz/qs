// Re-export common types for convenience
pub use crate::generated::*;

// Additional type definitions and extensions
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Extended signal metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalMetadata {
    pub confidence: f64,
    pub risk_score: f64,
    pub tags: Vec<String>,
    pub parameters: HashMap<String, String>,
}

/// Enhanced position data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedPosition {
    pub symbol: String,
    pub size: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub metadata: SignalMetadata,
}

/// Market data snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSnapshot {
    pub symbol: String,
    pub timestamp: i64,
    pub price: f64,
    pub volume: f64,
    pub bid: f64,
    pub ask: f64,
    pub spread: f64,
}