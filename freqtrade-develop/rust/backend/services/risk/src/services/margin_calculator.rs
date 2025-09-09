use anyhow::Result;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use crate::config::Config;
use crate::models::*;

pub struct MarginCalculator {
    #[allow(dead_code)]
    config: Config,
}

#[allow(dead_code)]
impl MarginCalculator {
    pub fn new(config: &Config) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub async fn calculate_order_margin(&self, request: &OrderValidationRequest) -> Result<Decimal> {
        let position_value = request.quantity * request.price.unwrap_or(Decimal::ONE);
        let leverage = request.leverage.unwrap_or(Decimal::ONE);
        
        // Base margin calculation
        let base_margin = position_value / leverage;
        
        // Add margin multipliers based on order type and market conditions
        let margin_multiplier = self.get_margin_multiplier(&request.order_type, &request.symbol);
        let required_margin = base_margin * margin_multiplier;
        
        Ok(required_margin)
    }

    pub async fn calculate_portfolio_margin_utilization(&self, positions: &[Position]) -> Result<Decimal> {
        let total_margin_used = self.calculate_total_used_margin(positions).await?;
        
        // Mock account balance - in practice would get from account service
        let account_balance = dec!(100000);
        
        if account_balance.is_zero() {
            Ok(Decimal::ZERO)
        } else {
            Ok(total_margin_used / account_balance)
        }
    }

    pub async fn calculate_total_used_margin(&self, positions: &[Position]) -> Result<Decimal> {
        let total = positions.iter()
            .map(|p| p.margin_used)
            .sum();
        Ok(total)
    }

    pub async fn calculate_margin_info(&self, positions: &[Position]) -> Result<MarginInfo> {
        let used_margin = self.calculate_total_used_margin(positions).await?;
        
        // Mock account balance
        let total_margin = dec!(100000);
        let available_margin = total_margin - used_margin;
        let margin_ratio = if total_margin.is_zero() { 
            Decimal::ZERO 
        } else { 
            used_margin / total_margin 
        };

        // Calculate maintenance margin (typically lower than initial margin)
        let maintenance_margin = used_margin * dec!(0.75); // 75% of used margin

        // Calculate margin call and liquidation levels
        let margin_call_level = total_margin * dec!(0.8); // Margin call at 80% utilization
        let liquidation_level = total_margin * dec!(0.95); // Liquidation at 95% utilization

        Ok(MarginInfo {
            total_margin,
            used_margin,
            available_margin,
            margin_ratio,
            maintenance_margin,
            margin_call_level,
            liquidation_level,
        })
    }

    pub async fn calculate_leverage_change_impact(
        &self,
        _symbol: &str,
        old_leverage: Decimal,
        new_leverage: Decimal,
    ) -> Result<Decimal> {
        // Mock position size and price for calculation
        let position_size = dec!(1.0);
        let price = dec!(50000); // Mock BTC price
        let position_value = position_size * price;

        let old_margin = position_value / old_leverage;
        let new_margin = position_value / new_leverage;
        
        // Return the difference in margin requirement
        Ok(new_margin - old_margin)
    }

    pub async fn calculate_liquidation_price(&self, position: &Position) -> Result<Decimal> {
        // Calculate liquidation price based on position and margin
        let margin_ratio = if position.margin_used.is_zero() {
            self.config.risk.margin_ratio
        } else {
            position.margin_used / (position.size * position.entry_price)
        };

        let liquidation_price = match position.side {
            PositionSide::Long => {
                position.entry_price * (Decimal::ONE - margin_ratio)
            },
            PositionSide::Short => {
                position.entry_price * (Decimal::ONE + margin_ratio)
            },
        };

        Ok(liquidation_price)
    }

    pub async fn calculate_margin_call_distance(&self, position: &Position) -> Result<Decimal> {
        let liquidation_price = self.calculate_liquidation_price(position).await?;
        let current_price = position.current_price;

        let distance = match position.side {
            PositionSide::Long => {
                if current_price > liquidation_price {
                    (current_price - liquidation_price) / current_price
                } else {
                    Decimal::ZERO
                }
            },
            PositionSide::Short => {
                if liquidation_price > current_price {
                    (liquidation_price - current_price) / current_price
                } else {
                    Decimal::ZERO
                }
            },
        };

        Ok(distance)
    }

    pub async fn calculate_cross_margin_impact(&self, positions: &[Position]) -> Result<CrossMarginInfo> {
        let total_margin_used = self.calculate_total_used_margin(positions).await?;
        let total_unrealized_pnl: Decimal = positions.iter()
            .map(|p| p.unrealized_pnl)
            .sum();

        // Mock account balance
        let account_balance = dec!(100000);
        let net_account_value = account_balance + total_unrealized_pnl;
        let margin_ratio = if net_account_value.is_zero() {
            Decimal::ZERO
        } else {
            total_margin_used / net_account_value
        };

        // Calculate positions at risk of liquidation
        let mut positions_at_risk = Vec::new();
        for position in positions {
            let distance = self.calculate_margin_call_distance(position).await?;
            if distance < dec!(0.1) { // Within 10% of liquidation
                positions_at_risk.push(position.clone());
            }
        }

        Ok(CrossMarginInfo {
            total_margin_used,
            net_account_value,
            margin_ratio,
            positions_at_risk,
            liquidation_risk: margin_ratio > dec!(0.8), // High risk if >80% margin used
        })
    }

    pub fn validate_leverage_change(&self, symbol: &str, new_leverage: Decimal) -> Result<LeverageValidationResult> {
        let mut valid = true;
        let mut warnings = Vec::new();
        let mut errors = Vec::new();

        // Check maximum leverage
        if new_leverage > self.config.risk.max_leverage {
            valid = false;
            errors.push(format!(
                "Leverage {} exceeds maximum allowed: {}", 
                new_leverage, 
                self.config.risk.max_leverage
            ));
        }

        // Check minimum leverage
        if new_leverage < Decimal::ONE {
            valid = false;
            errors.push("Leverage cannot be less than 1".to_string());
        }

        // Add warnings for high leverage
        if new_leverage > dec!(5) {
            warnings.push("High leverage increases liquidation risk".to_string());
        }

        // Symbol-specific leverage limits (simplified)
        let max_leverage_for_symbol = match symbol {
            s if s.contains("BTC") => dec!(20),
            s if s.contains("ETH") => dec!(15),
            _ => dec!(10),
        };

        if new_leverage > max_leverage_for_symbol {
            valid = false;
            errors.push(format!(
                "Leverage {new_leverage} exceeds maximum for {symbol}: {max_leverage_for_symbol}"
            ));
        }

        Ok(LeverageValidationResult {
            valid,
            warnings,
            errors,
            max_allowed_leverage: max_leverage_for_symbol,
        })
    }

    // Private helper methods
    fn get_margin_multiplier(&self, order_type: &OrderType, symbol: &str) -> Decimal {
        let mut multiplier = Decimal::ONE;

        // Order type multiplier
        match order_type {
            OrderType::Market => multiplier *= dec!(1.1), // 10% extra margin for market orders
            OrderType::StopMarket => multiplier *= dec!(1.15), // 15% extra for stop market
            OrderType::Limit => multiplier *= dec!(1.0), // No extra margin
            OrderType::StopLimit => multiplier *= dec!(1.05), // 5% extra for stop limit
        }

        // Symbol volatility multiplier (simplified)
        if symbol.contains("BTC") || symbol.contains("ETH") {
            multiplier *= dec!(1.0); // Major coins, no extra margin
        } else {
            multiplier *= dec!(1.2); // Altcoins, 20% extra margin
        }

        multiplier
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CrossMarginInfo {
    pub total_margin_used: Decimal,
    pub net_account_value: Decimal,
    pub margin_ratio: Decimal,
    pub positions_at_risk: Vec<Position>,
    pub liquidation_risk: bool,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LeverageValidationResult {
    pub valid: bool,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub max_allowed_leverage: Decimal,
}