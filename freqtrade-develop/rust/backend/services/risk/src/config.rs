use anyhow::Result;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub risk: RiskConfig,
    pub messaging: MessagingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub port: u16,
    pub host: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub min_connections: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Maximum position size per symbol (in base currency)
    pub max_position_size: Decimal,
    
    /// Maximum leverage allowed
    pub max_leverage: Decimal,
    
    /// Stop loss threshold (percentage)
    pub stop_loss_threshold: Decimal,
    
    /// Maximum daily loss (in base currency)
    pub max_daily_loss: Decimal,
    
    /// Minimum account balance required (in base currency)
    pub min_account_balance: Decimal,
    
    /// Maximum correlation exposure (percentage)
    pub max_correlation_exposure: Decimal,
    
    /// Margin ratio for position calculation
    pub margin_ratio: Decimal,
    
    /// Risk-free rate for calculations (annual percentage)
    pub risk_free_rate: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagingConfig {
    pub nats_url: String,
    pub subject_prefix: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                port: 8083,
                host: "0.0.0.0".to_string(),
            },
            database: DatabaseConfig {
                url: "postgresql://trading_user:dev_password_123@localhost:5432/trading_db".to_string(),
                max_connections: 10,
                min_connections: 2,
            },
            risk: RiskConfig {
                max_position_size: Decimal::from(10000), // $10,000 max position
                max_leverage: Decimal::from(10),         // 10x max leverage
                stop_loss_threshold: Decimal::from_f64_retain(0.05).unwrap(), // 5% stop loss
                max_daily_loss: Decimal::from(1000),     // $1,000 max daily loss
                min_account_balance: Decimal::from(1000), // $1,000 min balance
                max_correlation_exposure: Decimal::from_f64_retain(0.6).unwrap(), // 60% max correlation
                margin_ratio: Decimal::from_f64_retain(0.1).unwrap(), // 10% margin ratio
                risk_free_rate: Decimal::from_f64_retain(0.05).unwrap(), // 5% annual risk-free rate
            },
            messaging: MessagingConfig {
                nats_url: "nats://localhost:4222".to_string(),
                subject_prefix: "risk".to_string(),
            },
        }
    }
}

impl Config {
    pub fn new() -> Result<Self> {
        // Start with default config
        let mut config = Config::default();

        // Override with environment variables if available
        if let Ok(port) = std::env::var("RISK_PORT") {
            config.server.port = port.parse().unwrap_or(8083);
        }

        if let Ok(db_url) = std::env::var("DATABASE_URL") {
            config.database.url = db_url;
        }

        if let Ok(nats_url) = std::env::var("NATS_URL") {
            config.messaging.nats_url = nats_url;
        }

        // Risk parameter overrides
        if let Ok(max_pos) = std::env::var("RISK_MAX_POSITION_SIZE") {
            if let Ok(value) = max_pos.parse::<f64>() {
                config.risk.max_position_size = Decimal::from_f64_retain(value).unwrap_or(config.risk.max_position_size);
            }
        }

        if let Ok(max_lev) = std::env::var("RISK_MAX_LEVERAGE") {
            if let Ok(value) = max_lev.parse::<f64>() {
                config.risk.max_leverage = Decimal::from_f64_retain(value).unwrap_or(config.risk.max_leverage);
            }
        }

        if let Ok(stop_loss) = std::env::var("RISK_STOP_LOSS_THRESHOLD") {
            if let Ok(value) = stop_loss.parse::<f64>() {
                config.risk.stop_loss_threshold = Decimal::from_f64_retain(value).unwrap_or(config.risk.stop_loss_threshold);
            }
        }

        Ok(config)
    }
}