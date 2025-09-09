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
use prometheus::{Encoder, TextEncoder};
use chrono::{DateTime, Utc};

mod config;
mod models;
mod services;
mod handlers;
mod error;

use config::Config;
use services::risk_service::RiskService;

#[derive(Clone)]
pub struct AppState {
    pub risk_service: Arc<RiskService>,
    pub config: Arc<Config>,
    pub started_at: DateTime<Utc>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .with_thread_ids(true)
        .init();

    info!("🛡️ Starting Risk Service...");

    // Load configuration
    let config = Config::new()?;
    info!("✅ Configuration loaded");

    // Initialize risk service
    let risk_service = RiskService::new(&config).await?;
    info!("✅ Risk service initialized");

    // Create application state
    let state = AppState {
        risk_service: Arc::new(risk_service),
        config: Arc::new(config.clone()),
        started_at: Utc::now(),
    };

    // Create router
    let app = create_app(state);

    // Start server
    let port = config.server.port;
    let addr = format!("0.0.0.0:{port}");
    
    info!("🚀 Risk Service starting on {}", addr);
    
    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

fn create_app(state: AppState) -> Router {
    Router::new()
        // Health check
        .route("/health", get(health_check))
        .route("/metrics", get(metrics_handler))
        
        // Risk validation endpoints
        .route("/api/v1/risk/validate-order", post(handlers::validate_order))
        .route("/api/v1/risk/validate-position", post(handlers::validate_position))
        .route("/api/v1/risk/check-limits", post(handlers::check_limits))
        
        // Position management
        .route("/api/v1/risk/positions", get(handlers::get_positions))
        .route("/api/v1/risk/positions/{symbol}", get(handlers::get_position_by_symbol))
        .route("/api/v1/risk/positions/{symbol}", post(handlers::update_position))
        
        // Risk metrics
        .route("/api/v1/risk/metrics", get(handlers::get_risk_metrics))
        .route("/api/v1/risk/exposure", get(handlers::get_exposure))
        
        // Margin and leverage
        .route("/api/v1/risk/margin", get(handlers::get_margin_info))
        .route("/api/v1/risk/leverage/{symbol}", post(handlers::set_leverage))
        
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health_check(State(state): State<AppState>) -> Json<Value> {
    let uptime = (Utc::now() - state.started_at).num_seconds();
    Json(json!({
        "status": "healthy",
        "service": "risk-service",
        "version": "0.1.0",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "uptime_seconds": uptime,
        "database_connected": false,
        "config": {
            "max_position_size": state.config.risk.max_position_size,
            "max_leverage": state.config.risk.max_leverage,
            "stop_loss_threshold": state.config.risk.stop_loss_threshold
        }
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
