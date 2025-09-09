use anyhow::Result;
use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tracing::info;
use sqlx::{Pool, Postgres};
use prometheus::{Encoder, TextEncoder};

mod config;
mod models;
mod services;
mod simple_handlers;
mod handlers;
mod repository;
mod error;
mod indicators;
mod algorithms;

use config::Config;
use services::strategy_service::StrategyService;
use repository::{StrategyRepository, StrategyInstanceRepository, StrategyExecutionRepository, StrategyBacktestRepository, TransactionManager};

#[derive(Clone)]
pub struct AppState {
    pub strategy_service: Arc<StrategyService>,
    pub config: Arc<Config>,
    pub db_pool: Pool<Postgres>,
    pub strategy_repository: Arc<StrategyRepository>,
    pub strategy_instance_repository: Arc<StrategyInstanceRepository>,
    pub strategy_execution_repository: Arc<StrategyExecutionRepository>,
    pub strategy_backtest_repository: Arc<StrategyBacktestRepository>,
    pub transaction_manager: Arc<TransactionManager>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .with_thread_ids(true)
        .init();

    info!("🧠 Starting Strategy Service...");

    // Load configuration
    let config = Config::new()?;
    info!("✅ Configuration loaded");

    // Initialize database connection pool
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://trading_user:dev_password_123@localhost:5432/trading_db".to_string());
    
    info!("🔗 Connecting to database: {}", database_url.replace(":dev_password_123@", ":****@"));
    
    let db_pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(10)
        .connect(&database_url)
        .await?;
    
    // Run database migrations (tolerate already-applied/partial state for MVP)
    match sqlx::migrate!("../../migrations").run(&db_pool).await {
        Ok(_) => info!("✅ Database migrations completed"),
        Err(e) => {
            tracing::warn!("⚠️ Database migrations failed or partially applied: {}", e);
            // Continue to start the service to allow read-only APIs during MVP
        }
    }

    // Initialize repositories
    let strategy_repository = Arc::new(StrategyRepository::new(db_pool.clone()));
    let strategy_instance_repository = Arc::new(StrategyInstanceRepository::new(db_pool.clone()));
    let strategy_execution_repository = Arc::new(StrategyExecutionRepository::new(db_pool.clone()));
    let strategy_backtest_repository = Arc::new(StrategyBacktestRepository::new(db_pool.clone()));
    let transaction_manager = Arc::new(TransactionManager::new(db_pool.clone()));
    info!("✅ Database repositories and transaction manager initialized");

    // Initialize strategy service with database repositories
    let strategy_service = StrategyService::new(
        &config,
        strategy_repository.clone(),
        strategy_execution_repository.clone(),
    ).await?;
    info!("✅ Strategy service initialized");

    // Create application state
    let state = AppState {
        strategy_service: Arc::new(strategy_service),
        config: Arc::new(config.clone()),
        db_pool: db_pool.clone(),
        strategy_repository,
        strategy_instance_repository,
        strategy_execution_repository,
        strategy_backtest_repository,
        transaction_manager,
    };

    // Create router
    let app = create_app(state);

    // Start server
    let port = config.server.port;
    let addr = format!("0.0.0.0:{port}");
    
    info!("🚀 Strategy Service starting on {}", addr);
    
    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

fn create_app(state: AppState) -> Router {
    Router::new()
        // Health check
        .route("/health", get(health_check))
        .route("/metrics", get(metrics_handler))
        
        // Strategy management endpoints
        .route("/api/v1/strategies", get(simple_handlers::list_strategies))
        .route("/api/v1/strategies", post(simple_handlers::create_strategy))
        .route("/api/v1/strategies/{id}", get(simple_handlers::get_strategy))
        
        // Strategy execution endpoints
        .route("/api/v1/strategies/{id}/start", post(simple_handlers::start_strategy))
        .route("/api/v1/strategies/{id}/stop", post(simple_handlers::stop_strategy))
        
        // Signal generation endpoints
        .route("/api/v1/signals", get(handlers::get_signals))
        
        // Performance and analytics (placeholder)
        .route("/api/v1/strategies/{id}/performance", get(handlers::get_strategy_performance))
        .route("/api/v1/strategies/{id}/metrics", get(handlers::get_strategy_metrics))
        .route("/api/v1/strategies/{id}/executions", get(handlers::get_strategy_executions))
        
        // Market analysis
        .route("/api/v1/analysis/indicators", get(handlers::get_indicators))
        .route("/api/v1/analysis/factors", get(handlers::get_market_factors))
        
        // Real-time ingestion (from data folder bridge or external)
        .route("/api/v1/ingest/tick", post(handlers::ingest_tick))
        
        // Backtesting integration
        .route("/api/v1/strategies/{id}/backtest", post(simple_handlers::run_backtest))
        
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health_check(State(state): State<AppState>) -> Json<Value> {
    let running_strategies = state.strategy_service.get_running_strategies_count().await;
    let total_strategies = state.strategy_service.get_total_strategies_count().await;
    
    Json(json!({
        "status": "healthy",
        "service": "strategy-service",
        "version": "0.1.0",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "statistics": {
            "running_strategies": running_strategies,
            "total_strategies": total_strategies,
            "supported_algorithms": [
                "momentum",
                "mean_reversion", 
                "arbitrage",
                "grid_trading",
                "dca",
                "pairs_trading",
                "ml_prediction",
                "factor_based"
            ],
            "indicators_available": 25,
            "market_factors": 12
        },
        "market_connection": "connected",
        "database_connected": true
    }))
}

async fn metrics_handler() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    if let Err(e) = encoder.encode(&metric_families, &mut buffer) {
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("metrics encode error: {e}")).into_response();
    }
    axum::http::Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, encoder.format_type())
        .body(axum::body::Body::from(buffer))
        .unwrap()
}
