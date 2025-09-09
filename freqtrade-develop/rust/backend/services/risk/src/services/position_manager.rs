use anyhow::Result;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use chrono::Utc;

use crate::config::Config;
use crate::models::{Position, PositionSide};

pub struct PositionManager {
    #[allow(dead_code)]
    config: Arc<Config>,
    positions: Arc<RwLock<HashMap<String, Position>>>, // symbol -> position
}

#[allow(dead_code)]
impl PositionManager {
    pub async fn new(config: &Config) -> Result<Self> {
        Ok(Self {
            config: Arc::new(config.clone()),
            positions: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn get_all_positions(&self) -> Result<Vec<Position>> {
        let positions = self.positions.read().await;
        Ok(positions.values().cloned().collect())
    }

    pub async fn get_position_by_symbol(&self, symbol: &str) -> Result<Option<Position>> {
        let positions = self.positions.read().await;
        Ok(positions.get(symbol).cloned())
    }

    pub async fn update_position(&self, symbol: &str, size: Decimal, price: Decimal, side: PositionSide) -> Result<()> {
        self.update_position_with_exchange("default", symbol, size, price, side).await
    }

    pub async fn update_position_with_exchange(&self, exchange: &str, symbol: &str, size: Decimal, price: Decimal, side: PositionSide) -> Result<()> {
        let mut positions = self.positions.write().await;
        
        match positions.get_mut(symbol) {
            Some(position) => {
                // Update existing position
                position.size = size;
                position.current_price = price;
                position.unrealized_pnl = position.calculate_unrealized_pnl();
                position.updated_at = Utc::now();
            },
            None => {
                // Create new position
                let position = Position {
                    id: Uuid::new_v4(),
                    symbol: symbol.to_string(),
                    exchange: exchange.to_string(),
                    side,
                    size,
                    entry_price: price,
                    current_price: price,
                    unrealized_pnl: Decimal::ZERO,
                    realized_pnl: Decimal::ZERO,
                    margin_used: self.calculate_required_margin(size, price)?,
                    leverage: Decimal::ONE,
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                };
                positions.insert(symbol.to_string(), position);
            }
        }

        Ok(())
    }

    pub async fn set_leverage(&self, symbol: &str, leverage: Decimal) -> Result<Option<Decimal>> {
        let mut positions = self.positions.write().await;
        if let Some(position) = positions.get_mut(symbol) {
            let old = position.leverage;
            // Update leverage and recompute margin_used with a simple model:
            // margin = notional * margin_ratio / max(leverage, 1)
            position.leverage = leverage.max(Decimal::ONE);
            let notional = position.size * position.current_price;
            let mut margin = notional * self.config.risk.margin_ratio;
            if !position.leverage.is_zero() {
                margin = margin / position.leverage;
            }
            position.margin_used = margin;
            position.updated_at = Utc::now();
            Ok(Some(old))
        } else {
            Ok(None)
        }
    }

    pub async fn close_position(&self, symbol: &str, closing_price: Decimal) -> Result<Option<Decimal>> {
        let mut positions = self.positions.write().await;
        
        if let Some(position) = positions.remove(symbol) {
            let realized_pnl = match position.side {
                PositionSide::Long => (closing_price - position.entry_price) * position.size,
                PositionSide::Short => (position.entry_price - closing_price) * position.size,
            };
            Ok(Some(realized_pnl))
        } else {
            Ok(None)
        }
    }

    pub async fn update_market_prices(&self, price_updates: HashMap<String, Decimal>) -> Result<()> {
        let mut positions = self.positions.write().await;
        
        for (symbol, price) in price_updates {
            if let Some(position) = positions.get_mut(&symbol) {
                position.current_price = price;
                position.unrealized_pnl = position.calculate_unrealized_pnl();
                position.updated_at = Utc::now();
            }
        }

        Ok(())
    }

    pub async fn get_total_exposure(&self) -> Result<Decimal> {
        let positions = self.positions.read().await;
        let total = positions.values()
            .map(|p| p.size * p.current_price)
            .sum();
        Ok(total)
    }

    pub async fn get_total_unrealized_pnl(&self) -> Result<Decimal> {
        let positions = self.positions.read().await;
        let total = positions.values()
            .map(|p| p.unrealized_pnl)
            .sum();
        Ok(total)
    }

    pub async fn get_positions_by_exchange(&self, exchange: &str) -> Result<Vec<Position>> {
        let positions = self.positions.read().await;
        let filtered = positions.values()
            .filter(|p| p.exchange == exchange)
            .cloned()
            .collect();
        Ok(filtered)
    }

    pub async fn get_long_positions(&self) -> Result<Vec<Position>> {
        let positions = self.positions.read().await;
        let longs = positions.values()
            .filter(|p| matches!(p.side, PositionSide::Long))
            .cloned()
            .collect();
        Ok(longs)
    }

    pub async fn get_short_positions(&self) -> Result<Vec<Position>> {
        let positions = self.positions.read().await;
        let shorts = positions.values()
            .filter(|p| matches!(p.side, PositionSide::Short))
            .cloned()
            .collect();
        Ok(shorts)
    }

    pub async fn get_positions_at_risk(&self, risk_threshold: Decimal) -> Result<Vec<Position>> {
        let positions = self.positions.read().await;
        let at_risk = positions.values()
            .filter(|p| {
                let loss_percentage = if p.entry_price.is_zero() {
                    Decimal::ZERO
                } else {
                    (p.entry_price - p.current_price).abs() / p.entry_price
                };
                loss_percentage > risk_threshold && p.unrealized_pnl < Decimal::ZERO
            })
            .cloned()
            .collect();
        Ok(at_risk)
    }

    pub async fn check_stop_loss_triggers(&self) -> Result<Vec<Position>> {
        let positions = self.positions.read().await;
        let stop_loss_threshold = self.config.risk.stop_loss_threshold;
        
        let triggered = positions.values()
            .filter(|p| {
                let loss_percentage = if p.entry_price.is_zero() {
                    Decimal::ZERO
                } else {
                    match p.side {
                        PositionSide::Long => (p.entry_price - p.current_price) / p.entry_price,
                        PositionSide::Short => (p.current_price - p.entry_price) / p.entry_price,
                    }
                };
                loss_percentage > stop_loss_threshold
            })
            .cloned()
            .collect();
        Ok(triggered)
    }

    pub async fn get_margin_utilization(&self) -> Result<Decimal> {
        let positions = self.positions.read().await;
        let total_margin_used: Decimal = positions.values()
            .map(|p| p.margin_used)
            .sum();
        // 账户余额来源优先级：环境变量 RISK_ACCOUNT_BALANCE -> 配置 min_account_balance
        let account_balance = std::env::var("RISK_ACCOUNT_BALANCE")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .and_then(|v| Decimal::from_f64_retain(v))
            .unwrap_or(self.config.risk.min_account_balance)
            .max(Decimal::ONE);
        
        if account_balance.is_zero() {
            Ok(Decimal::ZERO)
        } else {
            Ok(total_margin_used / account_balance)
        }
    }

    // Private helper methods
    fn calculate_required_margin(&self, size: Decimal, price: Decimal) -> Result<Decimal> {
        let position_value = size * price;
        let margin = position_value * self.config.risk.margin_ratio;
        Ok(margin)
    }

    pub async fn simulate_position_close(&self, symbol: &str, closing_price: Decimal) -> Result<Option<Decimal>> {
        let positions = self.positions.read().await;
        
        if let Some(position) = positions.get(symbol) {
            let realized_pnl = match position.side {
                PositionSide::Long => (closing_price - position.entry_price) * position.size,
                PositionSide::Short => (position.entry_price - closing_price) * position.size,
            };
            Ok(Some(realized_pnl))
        } else {
            Ok(None)
        }
    }

    pub async fn get_position_summary(&self) -> Result<PositionSummary> {
        let positions = self.positions.read().await;
        
        let total_positions = positions.len();
        let long_positions = positions.values().filter(|p| matches!(p.side, PositionSide::Long)).count();
        let short_positions = positions.values().filter(|p| matches!(p.side, PositionSide::Short)).count();
        
        let total_exposure: Decimal = positions.values()
            .map(|p| p.size * p.current_price)
            .sum();
        
        let total_unrealized_pnl: Decimal = positions.values()
            .map(|p| p.unrealized_pnl)
            .sum();
        
        let total_margin_used: Decimal = positions.values()
            .map(|p| p.margin_used)
            .sum();

        Ok(PositionSummary {
            total_positions,
            long_positions,
            short_positions,
            total_exposure,
            total_unrealized_pnl,
            total_margin_used,
        })
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PositionSummary {
    pub total_positions: usize,
    pub long_positions: usize,
    pub short_positions: usize,
    pub total_exposure: Decimal,
    pub total_unrealized_pnl: Decimal,
    pub total_margin_used: Decimal,
}
