use anyhow::Result;
use rust_decimal::Decimal;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::config::Config;
use crate::models::*;
use super::position_manager::PositionManager;
use super::risk_calculator::RiskCalculator;
use super::margin_calculator::MarginCalculator;

pub struct RiskService {
    config: Arc<Config>,
    position_manager: Arc<PositionManager>,
    risk_calculator: Arc<RiskCalculator>,
    margin_calculator: Arc<MarginCalculator>,
    risk_events: Arc<RwLock<Vec<RiskEvent>>>,
}

impl RiskService {
    pub async fn new(config: &Config) -> Result<Self> {
        let position_manager = Arc::new(PositionManager::new(config).await?);
        let risk_calculator = Arc::new(RiskCalculator::new(config));
        let margin_calculator = Arc::new(MarginCalculator::new(config));
        
        Ok(Self {
            config: Arc::new(config.clone()),
            position_manager,
            risk_calculator,
            margin_calculator,
            risk_events: Arc::new(RwLock::new(Vec::new())),
        })
    }

    pub async fn validate_order(&self, request: OrderValidationRequest) -> Result<OrderValidationResponse> {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        let mut valid = true;

        // Calculate required margin
        let required_margin = self.margin_calculator
            .calculate_order_margin(&request)
            .await?;

        // Check if account has sufficient balance
        if request.account_balance < required_margin {
            errors.push(RiskError {
                code: "INSUFFICIENT_BALANCE".to_string(),
                message: format!(
                    "Insufficient balance. Required: {}, Available: {}",
                    required_margin, request.account_balance
                ),
                details: None,
            });
            valid = false;
        }

        // Check position size limits
        if request.quantity * request.price.unwrap_or(Decimal::ZERO) > self.config.risk.max_position_size {
            errors.push(RiskError {
                code: "POSITION_SIZE_EXCEEDED".to_string(),
                message: format!(
                    "Position size exceeds maximum allowed: {}",
                    self.config.risk.max_position_size
                ),
                details: None,
            });
            valid = false;
        }

        // Check leverage limits
        if let Some(leverage) = request.leverage {
            if leverage > self.config.risk.max_leverage {
                errors.push(RiskError {
                    code: "LEVERAGE_EXCEEDED".to_string(),
                    message: format!(
                        "Leverage {} exceeds maximum allowed: {}",
                        leverage, self.config.risk.max_leverage
                    ),
                    details: None,
                });
                valid = false;
            }
        }

        // Calculate risk score
        let risk_score = self.risk_calculator
            .calculate_order_risk(&request, &warnings, &errors)
            .await?;

        // Add warnings for high risk
        if risk_score > Decimal::from(70) {
            warnings.push(RiskWarning {
                code: "HIGH_RISK_ORDER".to_string(),
                message: format!("Order has high risk score: {risk_score}"),
                severity: WarningSeverity::High,
                recommendation: Some("Consider reducing position size or using stop loss".to_string()),
            });
        }

        // Calculate max allowed quantity
        let max_allowed_quantity = self.calculate_max_allowed_quantity(&request).await?;

        // Suggest stop loss
        let suggested_stop_loss = self.calculate_suggested_stop_loss(&request).await?;

        // Log risk event
        self.log_risk_event(RiskEvent {
            id: Uuid::new_v4(),
            event_type: RiskEventType::OrderValidation,
            symbol: request.symbol.clone(),
            exchange: request.exchange.clone(),
            risk_score,
            details: serde_json::to_value(&request)?,
            created_at: chrono::Utc::now(),
        }).await;

        Ok(OrderValidationResponse {
            valid,
            risk_score,
            warnings,
            errors,
            required_margin,
            max_allowed_quantity,
            suggested_stop_loss,
        })
    }

    pub async fn validate_position(&self, request: PositionValidationRequest) -> Result<PositionValidationResponse> {
        let mut warnings = Vec::new();
        let errors = Vec::new();
        let valid = true;

        // Calculate total exposure
        let total_exposure = self.calculate_total_exposure(&request.current_positions, &request).await?;

        // Check correlation risk
        let correlation_risk = self.risk_calculator
            .calculate_correlation_risk(&request.current_positions, &request.symbol)
            .await?;

        if correlation_risk > self.config.risk.max_correlation_exposure {
            warnings.push(RiskWarning {
                code: "HIGH_CORRELATION_RISK".to_string(),
                message: format!(
                    "High correlation risk: {}. Maximum allowed: {}",
                    correlation_risk, self.config.risk.max_correlation_exposure
                ),
                severity: WarningSeverity::Medium,
                recommendation: Some("Consider diversifying across uncorrelated assets".to_string()),
            });
        }

        // Calculate margin utilization
        let margin_utilization = self.margin_calculator
            .calculate_portfolio_margin_utilization(&request.current_positions)
            .await?;

        if margin_utilization > Decimal::from_f64_retain(0.8).unwrap() {
            warnings.push(RiskWarning {
                code: "HIGH_MARGIN_UTILIZATION".to_string(),
                message: format!("High margin utilization: {}%", margin_utilization * Decimal::from(100)),
                severity: WarningSeverity::High,
                recommendation: Some("Consider closing some positions to reduce margin usage".to_string()),
            });
        }

        // Calculate risk score
        let risk_score = self.risk_calculator
            .calculate_position_risk(&request, total_exposure, correlation_risk, margin_utilization)
            .await?;

        Ok(PositionValidationResponse {
            valid,
            risk_score,
            warnings,
            errors,
            total_exposure,
            correlation_risk,
            margin_utilization,
        })
    }

    pub async fn check_limits(&self, request: RiskLimitsCheckRequest) -> Result<RiskLimitsCheckResponse> {
        let mut within_limits = true;
        let mut limit_violations = Vec::new();
        let warnings = Vec::new();

        // Check daily loss limit
        if request.daily_pnl < -self.config.risk.max_daily_loss {
            within_limits = false;
            limit_violations.push(LimitViolation {
                limit_type: LimitType::MaxDailyLoss,
                current_value: request.daily_pnl.abs(),
                limit_value: self.config.risk.max_daily_loss,
                violation_percentage: (request.daily_pnl.abs() / self.config.risk.max_daily_loss - Decimal::ONE) * Decimal::from(100),
            });
        }

        // Check minimum account balance
        if request.account_balance < self.config.risk.min_account_balance {
            within_limits = false;
            limit_violations.push(LimitViolation {
                limit_type: LimitType::MinAccountBalance,
                current_value: request.account_balance,
                limit_value: self.config.risk.min_account_balance,
                violation_percentage: (Decimal::ONE - request.account_balance / self.config.risk.min_account_balance) * Decimal::from(100),
            });
        }

        // Calculate margin utilization
        let margin_utilization = self.margin_calculator
            .calculate_portfolio_margin_utilization(&request.open_positions)
            .await?;

        // Check margin utilization limit
        let margin_limit = Decimal::from_f64_retain(0.9).unwrap(); // 90% margin limit
        if margin_utilization > margin_limit {
            within_limits = false;
            limit_violations.push(LimitViolation {
                limit_type: LimitType::MarginRequirement,
                current_value: margin_utilization,
                limit_value: margin_limit,
                violation_percentage: (margin_utilization / margin_limit - Decimal::ONE) * Decimal::from(100),
            });
        }

        // Calculate max additional exposure
        let used_margin = self.margin_calculator
            .calculate_total_used_margin(&request.open_positions)
            .await?;
        
        let available_margin = request.account_balance - used_margin;
        let max_additional_exposure = available_margin * self.config.risk.max_leverage;

        Ok(RiskLimitsCheckResponse {
            within_limits,
            limit_violations,
            warnings,
            margin_utilization,
            max_additional_exposure,
        })
    }

    pub async fn get_positions(&self) -> Result<Vec<Position>> {
        self.position_manager.get_all_positions().await
    }

    pub async fn get_position_by_symbol(&self, symbol: &str) -> Result<Option<Position>> {
        self.position_manager.get_position_by_symbol(symbol).await
    }

    pub async fn get_risk_metrics(&self) -> Result<RiskMetrics> {
        let positions = self.get_positions().await?;
        self.risk_calculator.calculate_portfolio_metrics(&positions).await
    }

    pub async fn get_exposure(&self) -> Result<ExposureInfo> {
        let positions = self.get_positions().await?;
        self.risk_calculator.calculate_exposure_info(&positions).await
    }

    pub async fn get_margin_info(&self) -> Result<MarginInfo> {
        let positions = self.get_positions().await?;
        self.margin_calculator.calculate_margin_info(&positions).await
    }

    pub async fn update_position(&self, symbol: &str, exchange: Option<&str>, size: Decimal, price: Decimal, side: PositionSide) -> Result<()> {
        let ex = exchange.unwrap_or("default");
        self.position_manager.update_position_with_exchange(ex, symbol, size, price, side).await
    }

    pub async fn set_leverage(&self, symbol: &str, leverage: Decimal) -> Result<LeverageResponse> {
        // Validate leverage
        if leverage > self.config.risk.max_leverage {
            return Ok(LeverageResponse {
                success: false,
                symbol: symbol.to_string(),
                old_leverage: Decimal::ZERO,
                new_leverage: leverage,
                margin_impact: Decimal::ZERO,
                warnings: vec![RiskWarning {
                    code: "LEVERAGE_EXCEEDED".to_string(),
                    message: format!("Leverage {} exceeds maximum allowed: {}", leverage, self.config.risk.max_leverage),
                    severity: WarningSeverity::Critical,
                    recommendation: Some(format!("Use maximum leverage of {}", self.config.risk.max_leverage)),
                }],
            });
        }

        // Get current position leverage (if any)
        let old_leverage = self.get_position_by_symbol(symbol).await?.map(|p| p.leverage).unwrap_or(Decimal::ONE);

        // Calculate margin impact
        let margin_impact = self.margin_calculator
            .calculate_leverage_change_impact(symbol, old_leverage, leverage)
            .await?;

        // Update leverage in position manager if position exists
        let _ = self.position_manager.set_leverage(symbol, leverage).await?;
        
        Ok(LeverageResponse {
            success: true,
            symbol: symbol.to_string(),
            old_leverage,
            new_leverage: leverage,
            margin_impact,
            warnings: vec![],
        })
    }

    // Private helper methods
    async fn calculate_max_allowed_quantity(&self, request: &OrderValidationRequest) -> Result<Decimal> {
        let available_balance = request.account_balance;
        let price = request.price.unwrap_or(Decimal::ONE);
        let leverage = request.leverage.unwrap_or(Decimal::ONE);
        
        let max_by_balance = (available_balance * leverage) / price;
        let max_by_position_limit = self.config.risk.max_position_size / price;
        
        Ok(max_by_balance.min(max_by_position_limit))
    }

    async fn calculate_suggested_stop_loss(&self, request: &OrderValidationRequest) -> Result<Option<Decimal>> {
        if let Some(price) = request.price {
            let stop_loss_distance = price * self.config.risk.stop_loss_threshold;
            let suggested_stop = match request.side {
                OrderSide::Buy => price - stop_loss_distance,
                OrderSide::Sell => price + stop_loss_distance,
            };
            Ok(Some(suggested_stop))
        } else {
            Ok(None)
        }
    }

    async fn calculate_total_exposure(&self, positions: &[Position], request: &PositionValidationRequest) -> Result<Decimal> {
        let existing_exposure: Decimal = positions.iter()
            .map(|p| p.size * p.current_price)
            .sum();
        
        let new_exposure = request.proposed_size * positions.iter()
            .find(|p| p.symbol == request.symbol)
            .map(|p| p.current_price)
            .unwrap_or(Decimal::ONE);

        Ok(existing_exposure + new_exposure)
    }

    async fn log_risk_event(&self, event: RiskEvent) {
        let mut events = self.risk_events.write().await;
        events.push(event);
        
        // Keep only last 1000 events in memory
        if events.len() > 1000 {
            let excess = events.len() - 1000;
            events.drain(0..excess);
        }
    }
}
