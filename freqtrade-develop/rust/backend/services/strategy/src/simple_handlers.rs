use anyhow::Result;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
};
use serde::Serialize;
use serde_json::json;
use uuid::Uuid;

use crate::models::*;
use crate::AppState;

// Response Structs
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct SuccessResponse {
    pub success: bool,
    pub message: String,
}

// Simple Strategy Management Handlers

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

pub async fn list_strategies(
    State(state): State<AppState>,
) -> Result<Json<Vec<Strategy>>, (StatusCode, Json<ErrorResponse>)> {
    match state.strategy_service.list_strategies().await {
        Ok(strategies) => Ok(Json(strategies)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Failed to list strategies".to_string(),
                message: e.to_string(),
            }),
        )),
    }
}

pub async fn start_strategy(
    State(state): State<AppState>,
    Path(strategy_id): Path<Uuid>,
) -> Result<Json<SuccessResponse>, (StatusCode, Json<ErrorResponse>)> {
    match state.strategy_service.start_strategy(strategy_id).await {
        Ok(true) => Ok(Json(SuccessResponse {
            success: true,
            message: "Strategy started successfully".to_string(),
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

pub async fn stop_strategy(
    State(state): State<AppState>,
    Path(strategy_id): Path<Uuid>,
) -> Result<Json<SuccessResponse>, (StatusCode, Json<ErrorResponse>)> {
    match state.strategy_service.stop_strategy(strategy_id).await {
        Ok(true) => Ok(Json(SuccessResponse {
            success: true,
            message: "Strategy stopped successfully".to_string(),
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

// Placeholder endpoints

pub async fn get_signals(
    State(state): State<AppState>,
) -> Result<Json<SignalListResponse>, (StatusCode, Json<ErrorResponse>)> {
    match state
        .strategy_service
        .get_signals(Some(10), Some(0), None, None, None)
        .await
    {
        Ok(resp) => Ok(Json(resp)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: "Failed to get signals".to_string(), message: e.to_string() }),
        )),
    }
}

#[allow(dead_code)]
pub async fn get_performance(
    State(_state): State<AppState>,
    Path(_strategy_id): Path<Uuid>,
) -> Json<serde_json::Value> {
    Json(json!({
        "performance": null,
        "message": "Performance tracking not yet implemented"
    }))
}

#[allow(dead_code)]
pub async fn get_indicators(
    State(_state): State<AppState>,
) -> Json<serde_json::Value> {
    Json(json!({
        "indicators": [],
        "message": "Technical indicators not yet implemented"
    }))
}

pub async fn run_backtest(
    State(state): State<AppState>,
    Path(_strategy_id): Path<Uuid>,
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
