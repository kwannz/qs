use anyhow::Result;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub strategy: StrategyConfig,
    pub messaging: MessagingConfig,
    pub market_data: MarketDataConfig,
    pub ai: AiConfig,
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
pub struct StrategyConfig {
    /// Maximum number of concurrent running strategies
    pub max_concurrent_strategies: usize,
    
    /// Strategy evaluation interval in seconds
    pub evaluation_interval: u64,
    
    /// Maximum position size per strategy (in base currency)
    pub max_position_size: Decimal,
    
    /// Strategy performance tracking window (days)
    pub performance_window_days: u32,
    
    /// Factor calculation lookback period (hours)
    pub factor_lookback_hours: u32,
    
    /// Minimum data points required for factor calculation
    pub min_data_points: usize,
    
    /// Signal confidence threshold (0.0 - 1.0)
    pub signal_confidence_threshold: f64,
    
    /// Enable machine learning features
    pub enable_ml_features: bool,
    
    /// Risk score threshold for strategy execution
    pub max_risk_score: f64,
    
    /// Market data cache TTL in seconds
    pub market_data_cache_ttl: u64,
    
    /// Maximum execution history records to keep per strategy
    pub max_execution_history: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagingConfig {
    pub nats_url: String,
    pub subject_prefix: String,
    pub market_data_subject: String,
    pub signals_subject: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataConfig {
    pub service_url: String,
    pub websocket_url: String,
    pub supported_exchanges: Vec<String>,
    pub update_frequency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiConfig {
    pub deepseek_api_url: String,
    pub deepseek_api_key: String,
    pub gemini_api_url: String,
    pub gemini_api_key: String,
    pub enable_deepseek: bool,
    pub enable_gemini: bool,
    pub model_cache_ttl: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                port: 8084,
                host: "0.0.0.0".to_string(),
            },
            database: DatabaseConfig {
                url: "postgresql://trading_user:dev_password_123@localhost:5432/trading_db".to_string(),
                max_connections: 10,
                min_connections: 2,
            },
            strategy: StrategyConfig {
                max_concurrent_strategies: 10,
                evaluation_interval: 60, // 1 minute
                max_position_size: Decimal::from(50000), // $50,000 max per strategy
                performance_window_days: 30,
                factor_lookback_hours: 168, // 1 week
                min_data_points: 100,
                signal_confidence_threshold: 0.7,
                enable_ml_features: true,
                max_risk_score: 75.0,
                market_data_cache_ttl: 300, // 5 minutes
                max_execution_history: 1000, // Keep 1000 execution records per strategy
            },
            messaging: MessagingConfig {
                nats_url: "nats://localhost:4222".to_string(),
                subject_prefix: "strategy".to_string(),
                market_data_subject: "market.data.stream".to_string(),
                signals_subject: "strategy.signals".to_string(),
            },
            market_data: MarketDataConfig {
                service_url: "http://localhost:8081".to_string(),
                websocket_url: "ws://localhost:8081/ws".to_string(),
                supported_exchanges: vec![
                    "binance".to_string(),
                    "okx".to_string(),
                ],
                update_frequency_ms: 1000,
            },
            ai: AiConfig {
                deepseek_api_url: "https://api.deepseek.com/v1".to_string(),
                deepseek_api_key: "".to_string(),
                gemini_api_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
                gemini_api_key: "".to_string(),
                enable_deepseek: false,
                enable_gemini: false,
                model_cache_ttl: 3600, // 1 hour
            },
        }
    }
}

impl Config {
    pub fn new() -> Result<Self> {
        // Start with default config
        let mut config = Config::default();

        // Override with environment variables if available
        if let Ok(port) = std::env::var("STRATEGY_PORT") {
            config.server.port = port.parse().unwrap_or(8084);
        }

        if let Ok(db_url) = std::env::var("DATABASE_URL") {
            config.database.url = db_url;
        }

        if let Ok(nats_url) = std::env::var("NATS_URL") {
            config.messaging.nats_url = nats_url;
        }

        if let Ok(market_data_url) = std::env::var("MARKET_DATA_SERVICE_URL") {
            config.market_data.service_url = market_data_url;
        }

        // AI API configurations
        if let Ok(deepseek_key) = std::env::var("DEEPSEEK_API_KEY") {
            config.ai.deepseek_api_key = deepseek_key;
            config.ai.enable_deepseek = true;
        }

        if let Ok(gemini_key) = std::env::var("GEMINI_API_KEY") {
            config.ai.gemini_api_key = gemini_key;
            config.ai.enable_gemini = true;
        }

        // Strategy parameter overrides
        if let Ok(max_strategies) = std::env::var("STRATEGY_MAX_CONCURRENT") {
            if let Ok(value) = max_strategies.parse::<usize>() {
                config.strategy.max_concurrent_strategies = value;
            }
        }

        if let Ok(eval_interval) = std::env::var("STRATEGY_EVAL_INTERVAL") {
            if let Ok(value) = eval_interval.parse::<u64>() {
                config.strategy.evaluation_interval = value;
            }
        }

        if let Ok(max_pos) = std::env::var("STRATEGY_MAX_POSITION_SIZE") {
            if let Ok(value) = max_pos.parse::<f64>() {
                config.strategy.max_position_size = Decimal::from_f64_retain(value).unwrap_or(config.strategy.max_position_size);
            }
        }

        Ok(config)
    }
}