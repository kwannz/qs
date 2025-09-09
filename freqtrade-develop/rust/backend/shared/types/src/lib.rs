pub mod common;
pub mod analytics;
pub mod execution;
pub mod strategy;
pub mod market;
pub mod error;
pub mod factor_cache;

pub use common::*;
pub use analytics::*;
pub use execution::*;
pub use strategy::*;
pub use market::*;
pub use error::*;
pub use factor_cache::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_creation() {
        let symbol = Symbol::new("BTC", "USDT");
        assert_eq!(symbol.base, "BTC");
        assert_eq!(symbol.quote, "USDT");
        assert_eq!(symbol.to_string(), "BTCUSDT");
    }

    #[test]
    fn test_symbol_parsing() {
        let symbol = "ETHUSDT".parse::<Symbol>().unwrap();
        assert_eq!(symbol.base, "ETH");
        assert_eq!(symbol.quote, "USDT");
    }

    #[test]
    fn test_kline_interval_seconds() {
        assert_eq!(KlineInterval::OneMinute.to_seconds(), 60);
        assert_eq!(KlineInterval::OneHour.to_seconds(), 3600);
        assert_eq!(KlineInterval::OneDay.to_seconds(), 86400);
    }
}