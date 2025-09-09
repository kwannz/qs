use anyhow::Result;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use chrono::Utc;

use crate::config::Config;
use crate::models::*;

pub struct RiskCalculator {
    config: Config,
}

impl RiskCalculator {
    pub fn new(config: &Config) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub async fn calculate_order_risk(
        &self,
        request: &OrderValidationRequest,
        warnings: &[RiskWarning],
        errors: &[RiskError],
    ) -> Result<Decimal> {
        let mut risk_score = Decimal::ZERO;

        // Base risk from order size relative to max position size
        let order_value = request.quantity * request.price.unwrap_or(Decimal::ONE);
        let size_risk = (order_value / self.config.risk.max_position_size) * dec!(30);
        risk_score += size_risk;

        // Leverage risk
        if let Some(leverage) = request.leverage {
            let leverage_risk = (leverage / self.config.risk.max_leverage) * dec!(25);
            risk_score += leverage_risk;
        }

        // Market order risk (higher than limit orders)
        match request.order_type {
            OrderType::Market => risk_score += dec!(15),
            OrderType::StopMarket => risk_score += dec!(20),
            _ => risk_score += dec!(5),
        }

        // Account balance risk
        let required_margin = order_value / request.leverage.unwrap_or(Decimal::ONE);
        let balance_utilization = required_margin / request.account_balance;
        let balance_risk = balance_utilization * dec!(20);
        risk_score += balance_risk;

        // Penalty for warnings and errors
        risk_score += Decimal::from(warnings.len()) * dec!(5);
        risk_score += Decimal::from(errors.len()) * dec!(15);

        // Ensure risk score is within 0-100 range
        Ok(risk_score.min(dec!(100)).max(Decimal::ZERO))
    }

    pub async fn calculate_position_risk(
        &self,
        request: &PositionValidationRequest,
        total_exposure: Decimal,
        correlation_risk: Decimal,
        margin_utilization: Decimal,
    ) -> Result<Decimal> {
        let mut risk_score = Decimal::ZERO;

        // Exposure risk
        let max_exposure = self.config.risk.max_position_size * dec!(10); // Assume max 10 positions
        let exposure_risk = (total_exposure / max_exposure) * dec!(30);
        risk_score += exposure_risk;

        // Correlation risk
        let correlation_risk_score = (correlation_risk / self.config.risk.max_correlation_exposure) * dec!(25);
        risk_score += correlation_risk_score;

        // Margin utilization risk
        let margin_risk = margin_utilization * dec!(25);
        risk_score += margin_risk;

        // Position concentration risk
        let position_count = request.current_positions.len();
        if position_count < 3 {
            risk_score += dec!(10); // Penalty for lack of diversification
        }

        // Volatility risk (simplified - would need market data in practice)
        risk_score += dec!(10);

        Ok(risk_score.min(dec!(100)).max(Decimal::ZERO))
    }

    pub async fn calculate_correlation_risk(
        &self,
        positions: &[Position],
        new_symbol: &str,
    ) -> Result<Decimal> {
        if positions.is_empty() {
            return Ok(Decimal::ZERO);
        }

        // Simplified correlation calculation
        // In practice, this would use historical price correlations
        let mut total_correlation = Decimal::ZERO;
        let mut correlation_count = 0;

        for position in positions {
            let correlation = self.get_symbol_correlation(&position.symbol, new_symbol).await?;
            total_correlation += correlation.abs();
            correlation_count += 1;
        }

        if correlation_count == 0 {
            Ok(Decimal::ZERO)
        } else {
            Ok(total_correlation / Decimal::from(correlation_count))
        }
    }

    pub async fn calculate_portfolio_metrics(&self, positions: &[Position]) -> Result<RiskMetrics> {
        let total_exposure = self.calculate_total_exposure(positions);
        let unrealized_pnl = self.calculate_total_unrealized_pnl(positions);
        let realized_pnl_today = Decimal::ZERO; // Would need to track daily P&L
        let margin_utilization = self.calculate_margin_utilization(positions);
        let portfolio_var = self.calculate_value_at_risk(positions).await?;
        let sharpe_ratio = self.calculate_sharpe_ratio(positions).await?;
        let max_drawdown = self.calculate_max_drawdown(positions).await?;
        let correlation_risk = self.calculate_portfolio_correlation_risk(positions).await?;
        let leverage_ratio = self.calculate_average_leverage(positions);

        Ok(RiskMetrics {
            total_exposure,
            unrealized_pnl,
            realized_pnl_today,
            margin_utilization,
            portfolio_var,
            sharpe_ratio,
            max_drawdown,
            correlation_risk,
            leverage_ratio,
            calculated_at: Utc::now(),
        })
    }

    pub async fn calculate_exposure_info(&self, positions: &[Position]) -> Result<ExposureInfo> {
        let mut total_long_exposure = Decimal::ZERO;
        let mut total_short_exposure = Decimal::ZERO;
        let mut currency_exposures = HashMap::new();
        let mut sector_exposures = HashMap::new();

        for position in positions {
            let position_value = position.size * position.current_price;
            
            match position.side {
                PositionSide::Long => total_long_exposure += position_value,
                PositionSide::Short => total_short_exposure += position_value,
            }

            // Extract currency from symbol (e.g., BTCUSDT -> BTC, USDT)
            let currency = self.extract_base_currency(&position.symbol);
            *currency_exposures.entry(currency).or_insert(Decimal::ZERO) += position_value;

            // Simplified sector classification
            let sector = self.classify_asset_sector(&position.symbol);
            *sector_exposures.entry(sector).or_insert(Decimal::ZERO) += position_value;
        }

        let net_exposure = total_long_exposure - total_short_exposure;
        let gross_exposure = total_long_exposure + total_short_exposure;

        Ok(ExposureInfo {
            total_long_exposure,
            total_short_exposure,
            net_exposure,
            gross_exposure,
            currency_exposures,
            sector_exposures,
        })
    }

    // Private helper methods
    fn calculate_total_exposure(&self, positions: &[Position]) -> Decimal {
        positions.iter()
            .map(|p| p.size * p.current_price)
            .sum()
    }

    fn calculate_total_unrealized_pnl(&self, positions: &[Position]) -> Decimal {
        positions.iter()
            .map(|p| p.unrealized_pnl)
            .sum()
    }

    fn calculate_margin_utilization(&self, positions: &[Position]) -> Decimal {
        let total_margin_used: Decimal = positions.iter()
            .map(|p| p.margin_used)
            .sum();

        // Mock account balance for calculation
        let account_balance = dec!(100000);
        
        if account_balance.is_zero() {
            Decimal::ZERO
        } else {
            total_margin_used / account_balance
        }
    }

    async fn calculate_value_at_risk(&self, positions: &[Position]) -> Result<Decimal> {
        // Simplified VaR calculation (1-day, 95% confidence)
        // In practice, this would use historical volatility data
        let total_exposure = self.calculate_total_exposure(positions);
        let assumed_volatility = dec!(0.02); // 2% daily volatility
        let confidence_multiplier = dec!(1.65); // 95% confidence
        
        Ok(total_exposure * assumed_volatility * confidence_multiplier)
    }

    async fn calculate_sharpe_ratio(&self, positions: &[Position]) -> Result<Option<Decimal>> {
        // Simplified Sharpe ratio calculation
        // Would need historical returns data in practice
        if positions.is_empty() {
            return Ok(None);
        }

        let total_return = self.calculate_total_unrealized_pnl(positions);
        let total_exposure = self.calculate_total_exposure(positions);
        
        if total_exposure.is_zero() {
            return Ok(None);
        }

        let return_rate = total_return / total_exposure;
        let excess_return = return_rate - self.config.risk.risk_free_rate;
        let assumed_volatility = dec!(0.15); // 15% annualized volatility
        
        if assumed_volatility.is_zero() {
            Ok(None)
        } else {
            Ok(Some(excess_return / assumed_volatility))
        }
    }

    async fn calculate_max_drawdown(&self, positions: &[Position]) -> Result<Decimal> {
        // Simplified max drawdown calculation
        // Would need historical P&L data in practice
        let unrealized_losses: Decimal = positions.iter()
            .filter(|p| p.unrealized_pnl < Decimal::ZERO)
            .map(|p| p.unrealized_pnl.abs())
            .sum();
        
        let total_exposure = self.calculate_total_exposure(positions);
        
        if total_exposure.is_zero() {
            Ok(Decimal::ZERO)
        } else {
            Ok(unrealized_losses / total_exposure)
        }
    }

    async fn calculate_portfolio_correlation_risk(&self, positions: &[Position]) -> Result<Decimal> {
        if positions.len() < 2 {
            return Ok(Decimal::ZERO);
        }

        let mut total_correlation = Decimal::ZERO;
        let mut pair_count = 0;

        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let correlation = self.get_symbol_correlation(&positions[i].symbol, &positions[j].symbol).await?;
                total_correlation += correlation.abs();
                pair_count += 1;
            }
        }

        if pair_count == 0 {
            Ok(Decimal::ZERO)
        } else {
            Ok(total_correlation / Decimal::from(pair_count))
        }
    }

    fn calculate_average_leverage(&self, positions: &[Position]) -> Decimal {
        if positions.is_empty() {
            return Decimal::ZERO;
        }

        let total_leverage: Decimal = positions.iter()
            .map(|p| p.leverage)
            .sum();

        total_leverage / Decimal::from(positions.len())
    }

    async fn get_symbol_correlation(&self, symbol1: &str, symbol2: &str) -> Result<Decimal> {
        // Mock correlation data - in practice would query historical correlations
        match (symbol1, symbol2) {
            _ if symbol1 == symbol2 => Ok(Decimal::ONE),
            ("BTCUSDT", "ETHUSDT") | ("ETHUSDT", "BTCUSDT") => Ok(dec!(0.8)),
            ("BTCUSDT", "ADAUSDT") | ("ADAUSDT", "BTCUSDT") => Ok(dec!(0.6)),
            ("ETHUSDT", "ADAUSDT") | ("ADAUSDT", "ETHUSDT") => Ok(dec!(0.7)),
            _ => Ok(dec!(0.3)), // Default moderate correlation
        }
    }

    fn extract_base_currency(&self, symbol: &str) -> String {
        // Extract base currency from trading pair
        if symbol.contains("USDT") {
            symbol.replace("USDT", "")
        } else if symbol.contains("BTC") && symbol != "BTC" {
            symbol.replace("BTC", "")
        } else if symbol.contains("ETH") && symbol != "ETH" {
            symbol.replace("ETH", "")
        } else {
            symbol.to_string()
        }
    }

    fn classify_asset_sector(&self, symbol: &str) -> String {
        // Simplified asset sector classification
        match symbol {
            s if s.starts_with("BTC") => "Cryptocurrency".to_string(),
            s if s.starts_with("ETH") => "Smart Contract Platform".to_string(),
            s if s.starts_with("ADA") || s.starts_with("DOT") => "Proof of Stake".to_string(),
            s if s.starts_with("UNI") || s.starts_with("SUSHI") => "DeFi".to_string(),
            _ => "Other".to_string(),
        }
    }
}