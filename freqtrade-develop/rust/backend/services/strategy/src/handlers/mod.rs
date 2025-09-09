use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
// use std::collections::HashMap;
use uuid::Uuid;

use crate::models::*;
use std::time::Instant;
use once_cell::sync::Lazy;
use prometheus::{register_histogram_vec, register_int_counter_vec, HistogramVec, IntCounterVec};
use crate::repository::strategy_execution_repository::StrategyExecution as ExecRow;
use crate::services::strategy_service::StrategyService;
use crate::simple_handlers::SuccessResponse;
// use crate::services::strategy_service::StrategyService;

// Import AppState from main.rs
use crate::AppState;

// Strategy Management Handlers
// These are development placeholders, not yet used by router

#[allow(dead_code)]
pub async fn create_strategy(
    State(state): State<AppState>,
    Json(request): Json<CreateStrategyRequest>,
) -> Result<Json<Strategy>, (StatusCode, Json<ErrorResponse>)> {
    match state.strategy_service.create_strategy(request).await {
        Ok(strategy) => Ok(Json(strategy)),
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Failed to create strategy".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

#[allow(dead_code)]
pub async fn get_strategy(
    State(state): State<AppState>,
    Path(strategy_id): Path<Uuid>,
) -> Result<Json<Strategy>, (StatusCode, Json<ErrorResponse>)> {
    match state.strategy_service.get_strategy(strategy_id).await {
        Ok(Some(strategy)) => Ok(Json(strategy)),
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Strategy not found".to_string(),
                message: format!("No strategy found with ID: {strategy_id}"),
            }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to get strategy".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

#[allow(dead_code)]
pub async fn update_strategy(
    State(state): State<AppState>,
    Path(strategy_id): Path<Uuid>,
    Json(request): Json<UpdateStrategyRequest>,
) -> Result<Json<Strategy>, (StatusCode, Json<ErrorResponse>)> {
    match state.strategy_service.update_strategy(strategy_id, request).await {
        Ok(Some(strategy)) => Ok(Json(strategy)),
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Strategy not found".to_string(),
                message: format!("No strategy found with ID: {strategy_id}"),
            }),
        )),
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Failed to update strategy".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

#[allow(dead_code)]
pub async fn delete_strategy(
    State(state): State<AppState>,
    Path(strategy_id): Path<Uuid>,
) -> Result<Json<DeleteResponse>, (StatusCode, Json<ErrorResponse>)> {
    match state.strategy_service.delete_strategy(strategy_id).await {
        Ok(true) => Ok(Json(DeleteResponse {
            success: true,
            message: "Strategy deleted successfully".to_string(),
        })),
        Ok(false) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Strategy not found".to_string(),
                message: format!("No strategy found with ID: {strategy_id}"),
            }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to delete strategy".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

#[allow(dead_code)]
pub async fn list_strategies(
    State(state): State<AppState>,
    Query(params): Query<ListStrategiesQuery>,
) -> Result<Json<StrategyListResponse>, (StatusCode, Json<ErrorResponse>)> {
    match state.strategy_service.list_strategies().await {
        Ok(mut strategies) => {
            // Apply filters
            if let Some(status) = &params.status {
                strategies.retain(|s| s.status.to_string().to_lowercase() == status.to_lowercase());
            }
            
            if let Some(strategy_type) = &params.strategy_type {
                strategies.retain(|s| s.strategy_type.to_string().to_lowercase() == strategy_type.to_lowercase());
            }

            // Apply pagination
            let total = strategies.len();
            let page = params.page.unwrap_or(1).max(1);
            let per_page = params.per_page.unwrap_or(10).min(100);
            let start = ((page - 1) * per_page) as usize;
            let end = (start + per_page as usize).min(total);

            let paginated_strategies = if start < total {
                strategies[start..end].to_vec()
            } else {
                Vec::new()
            };

            Ok(Json(StrategyListResponse {
                strategies: paginated_strategies,
                total: total as u32,
                page,
                per_page,
            }))
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to list strategies".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

// Strategy Execution Handlers
// Development placeholders

#[allow(dead_code)]
pub async fn start_strategy(
    State(state): State<AppState>,
    Path(strategy_id): Path<Uuid>,
) -> Result<Json<StrategyActionResponse>, (StatusCode, Json<ErrorResponse>)> {
    match state.strategy_service.start_strategy(strategy_id).await {
        Ok(true) => Ok(Json(StrategyActionResponse {
            success: true,
            message: "Strategy started successfully".to_string(),
            strategy_id,
        })),
        Ok(false) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Failed to start strategy".to_string(),
                message: "Strategy not found or already running".to_string(),
            }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to start strategy".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

#[allow(dead_code)]
pub async fn stop_strategy(
    State(state): State<AppState>,
    Path(strategy_id): Path<Uuid>,
) -> Result<Json<StrategyActionResponse>, (StatusCode, Json<ErrorResponse>)> {
    match state.strategy_service.stop_strategy(strategy_id).await {
        Ok(true) => Ok(Json(StrategyActionResponse {
            success: true,
            message: "Strategy stopped successfully".to_string(),
            strategy_id,
        })),
        Ok(false) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Strategy not found".to_string(),
                message: format!("No strategy found with ID: {strategy_id}"),
            }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to stop strategy".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

#[allow(dead_code)]
pub async fn get_strategy_status(
    State(state): State<AppState>,
    Path(strategy_id): Path<Uuid>,
) -> Result<Json<StrategyStatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    match state.strategy_service.get_strategy_status(strategy_id).await {
        Ok(Some(status)) => Ok(Json(status)),
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Strategy not found".to_string(),
                message: format!("No strategy found with ID: {strategy_id}"),
            }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to get strategy status".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

// Signal Handlers
// Development placeholders

#[allow(dead_code)]
pub async fn get_signals(
    State(state): State<AppState>,
    Query(params): Query<SignalQuery>,
) -> Result<Json<SignalListResponse>, (StatusCode, Json<ErrorResponse>)> {
    let ep = "get_signals";
    REQ_TOTAL.with_label_values(&[ep]).inc();
    let start = Instant::now();
    match state.strategy_service
        .get_signals(params.limit, params.offset, params.symbol.clone(), params.action.clone(), params.strategy_id)
        .await
    {
        Ok(response) => {
            REQ_DURATION.with_label_values(&[ep]).observe(start.elapsed().as_secs_f64());
            Ok(Json(response))
        },
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to get signals".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

#[allow(dead_code)]
pub async fn get_signals_by_strategy(
    State(state): State<AppState>,
    Path(strategy_id): Path<Uuid>,
) -> Result<Json<Vec<Signal>>, (StatusCode, Json<ErrorResponse>)> {
    match state.strategy_service.get_signals_by_strategy(strategy_id).await {
        Ok(signals) => Ok(Json(signals)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to get strategy signals".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

// Performance and Analytics Handlers

pub async fn get_strategy_performance(
    State(state): State<AppState>,
    Path(strategy_id): Path<Uuid>,
    Query(params): Query<PerformanceQuery>,
) -> Result<Json<StrategyPerformance>, (StatusCode, Json<ErrorResponse>)> {
    let ep = "get_performance";
    REQ_TOTAL.with_label_values(&[ep]).inc();
    let start = Instant::now();
    match state.strategy_service.get_performance(strategy_id, params.days).await {
        Ok(Some(performance)) => {
            REQ_DURATION.with_label_values(&[ep]).observe(start.elapsed().as_secs_f64());
            Ok(Json(performance))
        },
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Performance data not found".to_string(),
                message: format!("No performance data found for strategy: {strategy_id}"),
            }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to get performance data".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

pub async fn get_strategy_metrics(
    State(state): State<AppState>,
    Path(strategy_id): Path<Uuid>,
) -> Result<Json<ExecutionMetrics>, (StatusCode, Json<ErrorResponse>)> {
    let ep = "get_metrics";
    REQ_TOTAL.with_label_values(&[ep]).inc();
    let start = Instant::now();
    match state.strategy_service.get_metrics(strategy_id).await {
        Ok(Some(metrics)) => {
            REQ_DURATION.with_label_values(&[ep]).observe(start.elapsed().as_secs_f64());
            Ok(Json(metrics))
        },
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Metrics not found".to_string(),
                message: format!("No metrics found for strategy: {strategy_id}"),
            }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to get metrics".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

// Executions history
#[derive(Debug, Deserialize)]
pub struct ExecQuery { pub limit: Option<u32>, pub offset: Option<u32>, pub since: Option<String> }

#[derive(Debug, Serialize)]
pub struct ExecListResponse { pub executions: Vec<ExecRow>, pub count: u32, pub limit: u32, pub offset: u32 }

pub async fn get_strategy_executions(
    State(state): State<AppState>,
    Path(strategy_id): Path<Uuid>,
    Query(params): Query<ExecQuery>,
) -> Result<Json<ExecListResponse>, (StatusCode, Json<ErrorResponse>)> {
    let ep = "get_executions";
    REQ_TOTAL.with_label_values(&[ep]).inc();
    let start = Instant::now();
    let limit = params.limit.unwrap_or(20).min(100);
    let offset = params.offset.unwrap_or(0);
    // If since provided, fetch a larger window and filter client-side (MVP)
    let base_rows = if params.since.is_some() {
        state.strategy_execution_repository.find_by_instance_id(strategy_id, Some(1000), Some(0)).await
    } else {
        state.strategy_execution_repository.find_by_instance_id(strategy_id, Some(limit), Some(offset)).await
    };
    match base_rows {
        Ok(mut rows) => {
            if let Some(since_str) = params.since.as_ref() {
                if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(since_str) {
                    let ts_utc = ts.with_timezone(&chrono::Utc);
                    rows.retain(|r| r.created_at >= ts_utc);
                }
            }
            // apply pagination after filtering
            let total = rows.len() as u32;
            let start_idx = (offset as usize).min(rows.len());
            let end_idx = (start_idx + limit as usize).min(rows.len());
            let page_rows = rows[start_idx..end_idx].to_vec();
            REQ_DURATION.with_label_values(&[ep]).observe(start.elapsed().as_secs_f64());
            Ok(Json(ExecListResponse { executions: page_rows, count: total, limit, offset }))
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: "Failed to get executions".to_string(), message: e.to_string() })
        )),
    }
}

// Ingestion endpoint for real-time ticks (symbol, price)
#[derive(Debug, Deserialize)]
pub struct IngestTickRequest { pub symbol: String, pub price: f64 }

pub async fn ingest_tick(
    State(state): State<AppState>,
    Json(req): Json<IngestTickRequest>,
) -> Result<Json<SuccessResponse>, (StatusCode, Json<ErrorResponse>)> {
    let ep = "ingest_tick";
    REQ_TOTAL.with_label_values(&[ep]).inc();
    let start = Instant::now();
    state.strategy_service.ingest_tick(&req.symbol, req.price).await;
    REQ_DURATION.with_label_values(&[ep]).observe(start.elapsed().as_secs_f64());
    Ok(Json(SuccessResponse { success: true, message: "ingested".to_string() }))
}

// Market Analysis Handlers

pub async fn get_indicators(
    State(state): State<AppState>,
    Query(params): Query<IndicatorQuery>,
) -> Result<Json<IndicatorListResponse>, (StatusCode, Json<ErrorResponse>)> {
    let symbol = params.symbol.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Missing parameter".to_string(),
                message: "Symbol parameter is required".to_string(),
            }),
        )
    })?;

    let timeframe = params.timeframe.unwrap_or_else(|| "1h".to_string());

    match state.strategy_service.get_indicators(&symbol, &timeframe).await {
        Ok(response) => Ok(Json(response)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to get indicators".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

pub async fn get_market_factors(
    State(state): State<AppState>,
) -> Result<Json<FactorAnalysisResponse>, (StatusCode, Json<ErrorResponse>)> {
    match state.strategy_service.get_factors().await {
        Ok(response) => Ok(Json(response)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to get market factors".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

// Remove unused aliases - functions are called directly by router

// Backtesting Handlers
// Development placeholder

#[allow(dead_code)]
pub async fn run_backtest(
    State(state): State<AppState>,
    Json(request): Json<BacktestRequest>,
) -> Result<Json<BacktestResponse>, (StatusCode, Json<ErrorResponse>)> {
    match state.strategy_service.run_backtest(request).await {
        Ok(response) => Ok(Json(response)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to start backtest".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

// System Status Handlers
// Development placeholder

#[allow(dead_code)]
pub async fn get_system_status(
    State(state): State<AppState>,
) -> Result<Json<SystemStatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    let running_strategies = state.strategy_service.get_running_strategies_count().await;
    let total_strategies = state.strategy_service.get_total_strategies_count().await;

    Ok(Json(SystemStatusResponse {
        status: "healthy".to_string(),
        running_strategies: running_strategies as u32,
        total_strategies: total_strategies as u32,
        uptime_seconds: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    }))
}

// Query Parameter Structs

#[derive(Debug, Deserialize)]
pub struct ListStrategiesQuery {
    pub status: Option<String>,
    pub strategy_type: Option<String>,
    pub page: Option<u32>,
    pub per_page: Option<u32>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct SignalQuery {
    #[allow(dead_code)]
    pub limit: Option<u32>,
    #[allow(dead_code)]
    pub offset: Option<u32>,
    pub symbol: Option<String>,
    pub action: Option<String>,
    pub strategy_id: Option<Uuid>,
}

#[derive(Debug, Deserialize)]
pub struct PerformanceQuery {
    pub days: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct IndicatorQuery {
    pub symbol: Option<String>,
    pub timeframe: Option<String>,
}

// Response Structs

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
pub struct DeleteResponse {
    pub success: bool,
    pub message: String,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
pub struct StrategyActionResponse {
    pub success: bool,
    pub message: String,
    pub strategy_id: Uuid,
}
static REQ_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    register_int_counter_vec!("strategy_requests_total", "Strategy API requests", &["endpoint"]).unwrap()
});
static ERR_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    register_int_counter_vec!("strategy_errors_total", "Strategy API errors", &["endpoint"]).unwrap()
});
static REQ_DURATION: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!("strategy_request_duration_seconds", "Strategy API duration", &["endpoint"]).unwrap()
});

#[allow(dead_code)]
#[derive(Debug, Serialize)]
pub struct StrategyListResponse {
    pub strategies: Vec<Strategy>,
    pub total: u32,
    pub page: u32,
    pub per_page: u32,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
pub struct SystemStatusResponse {
    pub status: String,
    pub running_strategies: u32,
    pub total_strategies: u32,
    pub uptime_seconds: u64,
}
