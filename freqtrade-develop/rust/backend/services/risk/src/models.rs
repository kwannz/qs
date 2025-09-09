use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub id: Uuid,
    pub symbol: String,
    pub exchange: String,
    pub side: PositionSide,
    pub size: Decimal,
    pub entry_price: Decimal,
    pub current_price: Decimal,
    pub unrealized_pnl: Decimal,
    pub realized_pnl: Decimal,
    pub margin_used: Decimal,
    pub leverage: Decimal,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionSide {
    Long,
    Short,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderValidationRequest {
    pub symbol: String,
    pub exchange: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: Decimal,
    pub price: Option<Decimal>,
    pub leverage: Option<Decimal>,
    pub account_balance: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    StopMarket,
    StopLimit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderValidationResponse {
    pub valid: bool,
    pub risk_score: Decimal,
    pub warnings: Vec<RiskWarning>,
    pub errors: Vec<RiskError>,
    pub required_margin: Decimal,
    pub max_allowed_quantity: Decimal,
    pub suggested_stop_loss: Option<Decimal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionValidationRequest {
    pub symbol: String,
    pub exchange: String,
    pub current_positions: Vec<Position>,
    pub proposed_size: Decimal,
    pub proposed_side: PositionSide,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionValidationResponse {
    pub valid: bool,
    pub risk_score: Decimal,
    pub warnings: Vec<RiskWarning>,
    pub errors: Vec<RiskError>,
    pub total_exposure: Decimal,
    pub correlation_risk: Decimal,
    pub margin_utilization: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimitsCheckRequest {
    pub account_balance: Decimal,
    pub open_positions: Vec<Position>,
    pub daily_pnl: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimitsCheckResponse {
    pub within_limits: bool,
    pub limit_violations: Vec<LimitViolation>,
    pub warnings: Vec<RiskWarning>,
    pub margin_utilization: Decimal,
    pub max_additional_exposure: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskWarning {
    pub code: String,
    pub message: String,
    pub severity: WarningSeverity,
    pub recommendation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarningSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskError {
    pub code: String,
    pub message: String,
    pub details: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitViolation {
    pub limit_type: LimitType,
    pub current_value: Decimal,
    pub limit_value: Decimal,
    pub violation_percentage: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LimitType {
    MaxPositionSize,
    MaxLeverage,
    MaxDailyLoss,
    MinAccountBalance,
    MaxCorrelationExposure,
    MarginRequirement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub total_exposure: Decimal,
    pub unrealized_pnl: Decimal,
    pub realized_pnl_today: Decimal,
    pub margin_utilization: Decimal,
    pub portfolio_var: Decimal, // Value at Risk
    pub sharpe_ratio: Option<Decimal>,
    pub max_drawdown: Decimal,
    pub correlation_risk: Decimal,
    pub leverage_ratio: Decimal,
    pub calculated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExposureInfo {
    pub total_long_exposure: Decimal,
    pub total_short_exposure: Decimal,
    pub net_exposure: Decimal,
    pub gross_exposure: Decimal,
    pub currency_exposures: std::collections::HashMap<String, Decimal>,
    pub sector_exposures: std::collections::HashMap<String, Decimal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginInfo {
    pub total_margin: Decimal,
    pub used_margin: Decimal,
    pub available_margin: Decimal,
    pub margin_ratio: Decimal,
    pub maintenance_margin: Decimal,
    pub margin_call_level: Decimal,
    pub liquidation_level: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeverageRequest {
    pub symbol: String,
    pub leverage: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeverageResponse {
    pub success: bool,
    pub symbol: String,
    pub old_leverage: Decimal,
    pub new_leverage: Decimal,
    pub margin_impact: Decimal,
    pub warnings: Vec<RiskWarning>,
}

// Database models for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskEvent {
    pub id: Uuid,
    pub event_type: RiskEventType,
    pub symbol: String,
    pub exchange: String,
    pub risk_score: Decimal,
    pub details: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskEventType {
    OrderValidation,
    PositionValidation,
    LimitViolation,
    MarginCall,
    StopLoss,
    LeverageChange,
}

impl Position {
    pub fn calculate_unrealized_pnl(&self) -> Decimal {
        match self.side {
            PositionSide::Long => (self.current_price - self.entry_price) * self.size,
            PositionSide::Short => (self.entry_price - self.current_price) * self.size,
        }
    }

    pub fn calculate_margin_ratio(&self) -> Decimal {
        if self.margin_used.is_zero() {
            Decimal::ZERO
        } else {
            (self.size * self.current_price) / self.margin_used
        }
    }

    pub fn is_profitable(&self) -> bool {
        self.unrealized_pnl > Decimal::ZERO
    }

    pub fn get_liquidation_price(&self, margin_ratio: Decimal) -> Decimal {
        match self.side {
            PositionSide::Long => {
                self.entry_price * (Decimal::ONE - margin_ratio)
            },
            PositionSide::Short => {
                self.entry_price * (Decimal::ONE + margin_ratio)
            },
        }
    }
}