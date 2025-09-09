use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{Json, IntoResponse},
};
use serde_json::json;
use rust_decimal::Decimal;

use crate::{models::*, AppState};
use std::time::Instant;
use once_cell::sync::Lazy;
use prometheus::{register_histogram_vec, register_int_counter_vec, HistogramVec, IntCounterVec};

static REQ_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    register_int_counter_vec!("risk_requests_total", "Risk API requests", &["endpoint"]).unwrap()
});
static ERR_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    register_int_counter_vec!("risk_errors_total", "Risk API errors", &["endpoint"]).unwrap()
});
static REQ_DURATION: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!("risk_request_duration_seconds", "Risk API duration", &["endpoint"]).unwrap()
});

pub async fn validate_order(
    State(state): State<AppState>,
    Json(request): Json<OrderValidationRequest>,
) -> impl IntoResponse {
    let ep = "validate_order";
    REQ_TOTAL.with_label_values(&[ep]).inc();
    let start = Instant::now();
    match state.risk_service.validate_order(request).await {
        Ok(response) => {
            REQ_DURATION.with_label_values(&[ep]).observe(start.elapsed().as_secs_f64());
            Json(response).into_response()
        },
        Err(err) => {
            tracing::error!("Order validation error: {}", err);
            ERR_TOTAL.with_label_values(&[ep]).inc();
            REQ_DURATION.with_label_values(&[ep]).observe(start.elapsed().as_secs_f64());
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Internal server error"})),
            ).into_response()
        }
    }
}

pub async fn validate_position(
    State(state): State<AppState>,
    Json(request): Json<PositionValidationRequest>,
) -> impl IntoResponse {
    match state.risk_service.validate_position(request).await {
        Ok(response) => Json(response).into_response(),
        Err(err) => {
            tracing::error!("Position validation error: {}", err);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Internal server error"})),
            ).into_response()
        }
    }
}

pub async fn check_limits(
    State(state): State<AppState>,
    Json(request): Json<RiskLimitsCheckRequest>,
) -> impl IntoResponse {
    let ep = "check_limits";
    REQ_TOTAL.with_label_values(&[ep]).inc();
    let start = Instant::now();
    match state.risk_service.check_limits(request).await {
        Ok(response) => {
            REQ_DURATION.with_label_values(&[ep]).observe(start.elapsed().as_secs_f64());
            Json(response).into_response()
        },
        Err(err) => {
            tracing::error!("Risk limits check error: {}", err);
            ERR_TOTAL.with_label_values(&[ep]).inc();
            REQ_DURATION.with_label_values(&[ep]).observe(start.elapsed().as_secs_f64());
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Internal server error"})),
            ).into_response()
        }
    }
}

pub async fn get_positions(State(state): State<AppState>) -> impl IntoResponse {
    match state.risk_service.get_positions().await {
        Ok(positions) => Json(json!({
            "positions": positions,
            "count": positions.len()
        })).into_response(),
        Err(err) => {
            tracing::error!("Get positions error: {}", err);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Internal server error"})),
            ).into_response()
        }
    }
}

pub async fn get_position_by_symbol(
    State(state): State<AppState>,
    Path(symbol): Path<String>,
) -> impl IntoResponse {
    match state.risk_service.get_position_by_symbol(&symbol).await {
        Ok(Some(position)) => Json(position).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!("Position not found for symbol: {}", symbol)})),
        ).into_response(),
        Err(err) => {
            tracing::error!("Get position by symbol error: {}", err);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Internal server error"})),
            ).into_response()
        }
    }
}

pub async fn get_risk_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let ep = "get_risk_metrics";
    REQ_TOTAL.with_label_values(&[ep]).inc();
    let start = Instant::now();
    match state.risk_service.get_risk_metrics().await {
        Ok(metrics) => {
            REQ_DURATION.with_label_values(&[ep]).observe(start.elapsed().as_secs_f64());
            Json(metrics).into_response()
        },
        Err(err) => {
            tracing::error!("Get risk metrics error: {}", err);
            ERR_TOTAL.with_label_values(&[ep]).inc();
            REQ_DURATION.with_label_values(&[ep]).observe(start.elapsed().as_secs_f64());
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Internal server error"})),
            ).into_response()
        }
    }
}

pub async fn get_exposure(State(state): State<AppState>) -> impl IntoResponse {
    match state.risk_service.get_exposure().await {
        Ok(exposure) => Json(exposure).into_response(),
        Err(err) => {
            tracing::error!("Get exposure error: {}", err);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Internal server error"})),
            ).into_response()
        }
    }
}

pub async fn get_margin_info(State(state): State<AppState>) -> impl IntoResponse {
    match state.risk_service.get_margin_info().await {
        Ok(margin_info) => Json(margin_info).into_response(),
        Err(err) => {
            tracing::error!("Get margin info error: {}", err);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Internal server error"})),
            ).into_response()
        }
    }
}

pub async fn set_leverage(
    State(state): State<AppState>,
    Path(symbol): Path<String>,
    Json(request): Json<LeverageRequest>,
) -> impl IntoResponse {
    let ep = "set_leverage";
    REQ_TOTAL.with_label_values(&[ep]).inc();
    let start = Instant::now();
    match state.risk_service.set_leverage(&symbol, request.leverage).await {
        Ok(response) => {
            REQ_DURATION.with_label_values(&[ep]).observe(start.elapsed().as_secs_f64());
            Json(response).into_response()
        },
        Err(err) => {
            tracing::error!("Set leverage error: {}", err);
            ERR_TOTAL.with_label_values(&[ep]).inc();
            REQ_DURATION.with_label_values(&[ep]).observe(start.elapsed().as_secs_f64());
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Internal server error"})),
            ).into_response()
        }
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct UpdatePositionRequest {
    pub exchange: Option<String>,
    pub size: Decimal,
    pub price: Decimal,
    pub side: PositionSide,
}

pub async fn update_position(
    State(state): State<AppState>,
    Path(symbol): Path<String>,
    Json(req): Json<UpdatePositionRequest>,
) -> impl IntoResponse {
    match state.risk_service.update_position(&symbol, req.exchange.as_deref(), req.size, req.price, req.side).await {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({
                "status": "updated",
                "symbol": symbol,
            })),
        ).into_response(),
        Err(err) => {
            tracing::error!("Update position error: {}", err);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Internal server error"})),
            ).into_response()
        }
    }
}
