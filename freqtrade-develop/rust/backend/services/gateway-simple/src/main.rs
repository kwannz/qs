//! Simplified Gateway Service - 微服务API网关
//! 
//! 专注功能:
//! - API路由到各个微服务
//! - 负载均衡
//! - 请求/响应日志
//! - 健康检查聚合
//! - 无认证(个人使用)

use std::net::SocketAddr;
use std::sync::Arc;
// Remove unused import

use axum::{
    Router,
    routing::{get, post, any},
    extract::{Request, State},
    response::{Response, IntoResponse},
    http::StatusCode,
    body::Body,
};
use tower::ServiceBuilder;
use tower_http::{trace::TraceLayer, cors::CorsLayer, timeout::TimeoutLayer};
use axum::extract::DefaultBodyLimit;
use std::time::Duration;
use tracing::{info, error, warn};
use tokio::signal;

mod config;
mod proxy;
mod health;

use crate::config::Config;
use crate::proxy::ProxyClient;
use crate::health::HealthChecker;
use prometheus::{Encoder, TextEncoder};

/// Gateway应用状态
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub proxy_client: Arc<ProxyClient>,
    pub health_checker: Arc<HealthChecker>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 初始化日志
    tracing_subscriber::fmt()
        .with_env_filter("gateway_simple=debug,tower_http=debug")
        .init();

    info!("🚀 启动简化Gateway服务");

    // 加载配置
    let config = Arc::new(Config::from_env()?);
    let bind_addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;

    // 初始化组件
    let proxy_client = Arc::new(ProxyClient::new((*config).clone())?);
    let health_checker = Arc::new(HealthChecker::new((*config).clone()));
    
    let app_state = AppState {
        config: config.clone(),
        proxy_client,
        health_checker: health_checker.clone(),
    };

    // 构建路由
    let app = create_app(app_state);

    // 启动健康检查任务
    let health_checker_task = health_checker.clone();
    tokio::spawn(async move {
        health_checker_task.start_monitoring().await;
    });

    info!("🌐 Gateway服务监听 {}", bind_addr);

    // 启动服务器
    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("✅ Gateway服务优雅关闭");
    Ok(())
}

fn create_app(state: AppState) -> Router {
    Router::new()
        // 健康检查 - Gateway本身
        .route("/health", get(gateway_health_check))
        .route("/api/health", get(gateway_health_check))  // Frontend expects /api/health
        .route("/metrics", get(metrics_handler))
        .route("/ready", get(gateway_ready_check))
        .route("/health/services", get(services_health_check))
        
        // API路由代理
        // Market Data Service - 8081
        .route("/api/v1/market/{*path}", any(proxy_to_market))
        .route("/ws/market/{*path}", any(proxy_to_market))
        
        // Trading Service - 8082  
        .route("/api/v1/orders", post(proxy_to_trading))
        .route("/api/v1/orders/{*path}", any(proxy_to_trading))
        .route("/api/v1/positions/{*path}", any(proxy_to_trading))
        .route("/api/v1/algorithms/{*path}", any(proxy_to_trading))
        
        // Risk Service - 8083 (待实现)
        .route("/api/v1/risk/{*path}", any(proxy_to_risk))
        
        // Strategy Service - 8084 (待实现)
        .route("/api/v1/strategy/{*path}", any(proxy_to_strategy))
        .route("/api/v1/strategies/{*path}", any(proxy_to_strategy))
        
        // Backtest Service - 8085 (待实现)
        .route("/api/v1/backtest/{*path}", any(proxy_to_backtest))
        
        // 默认路由 - API版本信息
        .route("/", get(api_info))
        .route("/api/v1", get(api_v1_info))
        
        .with_state(state)
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive())
                .layer(DefaultBodyLimit::max(1 * 1024 * 1024)) // 1 MiB body limit
                .layer(TimeoutLayer::new(std::time::Duration::from_secs(30)))
        )
}

// =================
// 健康检查处理器
// =================

async fn gateway_health_check() -> impl IntoResponse {
    axum::response::Json(serde_json::json!({
        "status": "healthy",
        "service": "gateway-simple",
        "version": env!("CARGO_PKG_VERSION"),
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

async fn gateway_ready_check(State(state): State<AppState>) -> impl IntoResponse {
    // 检查是否能连接到至少一个后端服务
    let services_status = state.health_checker.check_all_services().await;
    let healthy_services = services_status.iter()
        .filter(|(_, status)| **status)
        .count();

    if healthy_services > 0 {
        (StatusCode::OK, axum::response::Json(serde_json::json!({
            "status": "ready",
            "healthy_services": healthy_services,
            "total_services": services_status.len()
        })))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, axum::response::Json(serde_json::json!({
            "status": "not_ready",
            "healthy_services": 0,
            "total_services": services_status.len()
        })))
    }
}

async fn services_health_check(State(state): State<AppState>) -> impl IntoResponse {
    let services_status = state.health_checker.check_all_services().await;
    axum::response::Json(serde_json::json!({
        "gateway_status": "healthy",
        "services": services_status,
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

// =================
// API信息处理器
// =================

async fn api_info() -> impl IntoResponse {
    axum::response::Json(serde_json::json!({
        "service": "Cryptocurrency Trading Platform API Gateway",
        "version": "v1.0.0",
        "description": "个人加密货币量化交易平台 - 微服务网关",
        "endpoints": {
            "market_data": "/api/v1/market/*",
            "trading": "/api/v1/orders, /api/v1/positions, /api/v1/algorithms",
            "risk": "/api/v1/risk/*",
            "strategy": "/api/v1/strategies/*",
            "backtest": "/api/v1/backtest/*"
        },
        "websocket": {
            "market_data": "/ws/market/*"
        },
        "health": {
            "gateway": "/health",
            "services": "/health/services"
        },
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

async fn api_v1_info() -> impl IntoResponse {
    axum::response::Json(serde_json::json!({
        "api_version": "v1",
        "status": "active",
        "features": [
            "market_data_streaming",
            "order_management",
            "algorithm_trading",
            "risk_management",
            "strategy_execution",
            "backtesting"
        ],
        "no_authentication": true,
        "description": "个人使用，无需认证",
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

// =================
// 代理处理器
// =================

async fn proxy_to_market(
    State(state): State<AppState>,
    req: Request<Body>
) -> Result<Response, StatusCode> {
    state.proxy_client.proxy_request("market-data", req).await
        .map_err(|e| {
            error!("Market Data Service代理错误: {}", e);
            StatusCode::BAD_GATEWAY
        })
}

async fn proxy_to_trading(
    State(state): State<AppState>,
    req: Request<Body>
) -> Result<Response, StatusCode> {
    state.proxy_client.proxy_request("trading", req).await
        .map_err(|e| {
            error!("Trading Service代理错误: {}", e);
            StatusCode::BAD_GATEWAY
        })
}

async fn proxy_to_risk(
    State(state): State<AppState>,
    req: Request<Body>
) -> Result<Response, StatusCode> {
    state.proxy_client.proxy_request("risk", req).await
        .map_err(|e| {
            error!("Risk Service代理错误: {}", e);
            StatusCode::BAD_GATEWAY
        })
}

async fn proxy_to_strategy(
    State(_state): State<AppState>,
    _req: Request<Body>
) -> Result<Response, StatusCode> {
    // Strategy Service尚未实现，返回503  
    warn!("Strategy Service尚未实现");
    Err(StatusCode::SERVICE_UNAVAILABLE)
}

async fn proxy_to_backtest(
    State(_state): State<AppState>,
    _req: Request<Body>
) -> Result<Response, StatusCode> {
    // Backtest Service尚未实现，返回503
    warn!("Backtest Service尚未实现");
    Err(StatusCode::SERVICE_UNAVAILABLE)
}

// =================
// 优雅关闭
// =================

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("🛑 接收到 Ctrl+C，开始优雅关闭...");
        }
        _ = terminate => {
            info!("🛑 接收到 SIGTERM，开始优雅关闭...");
        }
    }
}

// =================
// 指标导出
// =================

async fn metrics_handler() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    if let Err(e) = encoder.encode(&metric_families, &mut buffer) {
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("metrics encode error: {e}")).into_response();
    }
    let mut resp = Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, encoder.format_type())
        .body(Body::from(buffer))
        .unwrap();
    resp
}
