# Sprint 1 Implementation Guide - MVP Features

## Overview

This guide provides step-by-step implementation guidance for Sprint 1 MVP features, following Rust-first principles with unified gateway architecture.

## Implementation Principles

### MVP-First Development
- **Build only what's needed**: No over-engineering or speculative features
- **Sandbox-first**: Use in-memory implementations before external integrations
- **Gradual enhancement**: Start with placeholders, evolve to real implementations
- **Error gracefully**: Always provide fallback responses

### Code Quality Standards
- **No compilation warnings**: Zero tolerance for compiler warnings
- **Complete implementations**: No TODO comments or unimplemented functions
- **Consistent patterns**: Follow established project conventions
- **Observable code**: Built-in logging and tracing

## Current System State

### ✅ Completed Components
- **Gateway Core**: Axum web server with middleware stack
- **Execution MVP**: Sandbox trading engine with full CRUD operations
- **Markets MVP**: Placeholder market data with realistic responses
- **WebSocket**: Real-time communication infrastructure
- **Health System**: Comprehensive health checks and system info
- **Cache System**: Memory/Redis caching with batch operations
- **Middleware**: Tracing, idempotency, rate limiting, CORS

### 🔄 Partial Implementations
- **Analytics MVP**: Basic structure, needs completion
- **Risk MVP**: Placeholder endpoints, needs business logic
- **Audit MVP**: Basic logging, needs export functionality
- **Reports MVP**: Skeleton structure, needs implementation

### ❌ Not Implemented
- Real external service integration
- Persistent database storage
- Authentication system (intentionally disabled for MVP)

## Implementation Tasks by Priority

## Phase 1: Core Infrastructure Completion

### Task 1.1: Complete Analytics MVP Handlers
**File**: `services/gateway/src/handlers/analytics_mvp.rs`

**Current State**: Partial implementation with placeholder responses

**Implementation Steps**:

1. **Complete Trading Stats Endpoint**:
```rust
pub async fn get_trading_stats(
    State(app_state): State<AppState>,
    Query(params): Query<TradingStatsQuery>,
) -> Result<Json<ApiResponse<TradingStats>>, ApiError> {
    // Calculate stats from sandbox trading data
    let sandbox = &app_state.sandbox_engine;
    let account_id = params.account_id.unwrap_or_else(|| "demo".to_string());
    
    let stats = sandbox.get_account_stats(&account_id).await
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    
    Ok(Json(ApiResponse::success(stats)))
}

#[derive(Debug, Deserialize)]
pub struct TradingStatsQuery {
    pub account_id: Option<String>,
    pub start_date: Option<String>,
    pub end_date: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TradingStats {
    pub account_id: String,
    pub total_orders: u32,
    pub filled_orders: u32,
    pub total_volume: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub win_rate: f64,
    pub period_start: String,
    pub period_end: String,
}
```

2. **Implement Portfolio Summary**:
```rust
pub async fn get_portfolio_summary(
    State(app_state): State<AppState>,
    Query(params): Query<PortfolioQuery>,
) -> Result<Json<ApiResponse<PortfolioSummary>>, ApiError> {
    let account_id = params.account_id.unwrap_or_else(|| "demo".to_string());
    
    // Get current positions and balances
    let positions = app_state.sandbox_engine.get_positions(&account_id).await
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    
    let balances = app_state.sandbox_engine.get_balances(&account_id).await
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    
    // Calculate portfolio metrics
    let total_value = calculate_portfolio_value(&positions, &balances).await;
    let allocation = calculate_asset_allocation(&positions, &balances).await;
    
    let summary = PortfolioSummary {
        account_id,
        total_value,
        available_balance: balances.iter().find(|b| b.asset == "USDT")
            .map(|b| b.free).unwrap_or(0.0),
        positions_count: positions.len() as u32,
        asset_allocation: allocation,
        updated_at: chrono::Utc::now().to_rfc3339(),
    };
    
    Ok(Json(ApiResponse::success(summary)))
}
```

3. **Add Technical Indicators Calculation**:
```rust
pub async fn calculate_indicators(
    State(_app_state): State<AppState>,
    Query(params): Query<IndicatorQuery>,
) -> Result<Json<ApiResponse<IndicatorResult>>, ApiError> {
    // For MVP, return placeholder technical indicators
    let indicators = IndicatorResult {
        symbol: params.symbol.clone(),
        indicators: vec![
            Indicator {
                name: "RSI".to_string(),
                value: 65.5,
                signal: "NEUTRAL".to_string(),
            },
            Indicator {
                name: "MACD".to_string(), 
                value: 0.12,
                signal: "BUY".to_string(),
            },
            Indicator {
                name: "SMA_20".to_string(),
                value: 45000.0,
                signal: "NEUTRAL".to_string(),
            },
        ],
        timestamp: chrono::Utc::now().to_rfc3339(),
    };
    
    Ok(Json(ApiResponse::success(indicators)))
}
```

### Task 1.2: Complete Risk MVP Handlers
**File**: `services/gateway/src/handlers/risk_mvp.rs`

**Implementation Steps**:

1. **Risk Assessment Implementation**:
```rust
pub async fn get_risk_assessment(
    State(app_state): State<AppState>,
    Query(params): Query<RiskQuery>,
) -> Result<Json<ApiResponse<RiskAssessment>>, ApiError> {
    let account_id = params.account_id.unwrap_or_else(|| "demo".to_string());
    
    // Get current positions and calculate risk metrics
    let positions = app_state.sandbox_engine.get_positions(&account_id).await
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    
    let balances = app_state.sandbox_engine.get_balances(&account_id).await
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    
    let risk_score = calculate_risk_score(&positions, &balances).await;
    let portfolio_var = calculate_portfolio_var(&positions).await;
    
    let assessment = RiskAssessment {
        account_id,
        risk_score,
        risk_level: get_risk_level(risk_score),
        portfolio_var,
        concentration_risk: calculate_concentration_risk(&positions),
        leverage_ratio: calculate_leverage_ratio(&positions, &balances),
        warnings: generate_risk_warnings(&positions, &balances),
        updated_at: chrono::Utc::now().to_rfc3339(),
    };
    
    Ok(Json(ApiResponse::success(assessment)))
}

async fn calculate_risk_score(positions: &[Position], balances: &[Balance]) -> f64 {
    // Simple risk scoring algorithm
    let position_count = positions.len() as f64;
    let total_exposure = positions.iter().map(|p| p.quantity * p.average_price).sum::<f64>();
    let total_balance = balances.iter().map(|b| b.total).sum::<f64>();
    
    let leverage = if total_balance > 0.0 { total_exposure / total_balance } else { 0.0 };
    let diversification_score = (position_count / 10.0).min(1.0);
    
    // Risk score from 1-10 (lower is better)
    (leverage * 2.0 + (1.0 - diversification_score) * 3.0).min(10.0).max(1.0)
}
```

2. **VaR Calculation**:
```rust
pub async fn calculate_var(
    State(app_state): State<AppState>,
    Json(request): Json<VarRequest>,
) -> Result<Json<ApiResponse<VarResult>>, ApiError> {
    // Simple historical VaR calculation for MVP
    let var_95 = calculate_historical_var(&request.account_id, 0.95).await;
    let var_99 = calculate_historical_var(&request.account_id, 0.99).await;
    
    let result = VarResult {
        account_id: request.account_id,
        confidence_level_95: var_95,
        confidence_level_99: var_99,
        time_horizon_days: request.time_horizon_days.unwrap_or(1),
        method: "historical_simulation".to_string(),
        calculated_at: chrono::Utc::now().to_rfc3339(),
    };
    
    Ok(Json(ApiResponse::success(result)))
}

async fn calculate_historical_var(account_id: &str, confidence: f64) -> f64 {
    // Placeholder VaR calculation - in real implementation, 
    // this would use historical price data and portfolio positions
    match confidence {
        0.95 => 1250.0,  // 95% VaR
        0.99 => 2100.0,  // 99% VaR
        _ => 1000.0,
    }
}
```

### Task 1.3: Complete Audit MVP Export Functionality
**File**: `services/gateway/src/handlers/audit_mvp.rs`

**Implementation Steps**:

1. **Export Audit Events**:
```rust
pub async fn export_audit_events(
    State(app_state): State<AppState>,
    Query(params): Query<ExportQuery>,
) -> Result<Json<ApiResponse<ExportResult>>, ApiError> {
    let format = params.format.unwrap_or_else(|| "csv".to_string());
    let start_time = params.start_time;
    let end_time = params.end_time;
    
    // Generate export task
    let export_id = format!("export_{}", uuid::Uuid::new_v4());
    let filename = format!("audit_events_{}_{}.{}", 
        format_timestamp(start_time), 
        format_timestamp(end_time), 
        format
    );
    
    // In MVP, create the export immediately (small dataset)
    let events = get_audit_events_for_export(start_time, end_time).await
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    
    let export_path = generate_export_file(&filename, &format, &events).await
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    
    let result = ExportResult {
        export_id,
        format,
        download_url: format!("/api/v1/audit/download/{}", filename),
        expires_at: (chrono::Utc::now() + chrono::Duration::hours(1)).to_rfc3339(),
        record_count: events.len() as u32,
    };
    
    Ok(Json(ApiResponse::success(result)))
}

async fn generate_export_file(
    filename: &str, 
    format: &str, 
    events: &[AuditEvent]
) -> Result<String, Box<dyn std::error::Error>> {
    let export_dir = std::path::Path::new("tmp/exports");
    std::fs::create_dir_all(export_dir)?;
    
    let file_path = export_dir.join(filename);
    
    match format {
        "csv" => {
            let mut writer = csv::Writer::from_path(&file_path)?;
            for event in events {
                writer.serialize(event)?;
            }
            writer.flush()?;
        }
        "json" => {
            let json_data = serde_json::to_string_pretty(events)?;
            std::fs::write(&file_path, json_data)?;
        }
        _ => return Err("Unsupported format".into()),
    }
    
    Ok(file_path.to_string_lossy().to_string())
}
```

2. **Download Handler**:
```rust
pub async fn download_audit_export(
    Path(filename): Path<String>,
) -> Result<Response, ApiError> {
    let export_path = std::path::Path::new("tmp/exports").join(&filename);
    
    if !export_path.exists() {
        return Err(ApiError::NotFound("Export file not found".to_string()));
    }
    
    let file_contents = std::fs::read(&export_path)
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    
    let content_type = if filename.ends_with(".csv") {
        "text/csv"
    } else if filename.ends_with(".json") {
        "application/json"
    } else {
        "application/octet-stream"
    };
    
    Ok(Response::builder()
        .status(200)
        .header("Content-Type", content_type)
        .header("Content-Disposition", format!("attachment; filename=\"{}\"", filename))
        .body(file_contents.into())
        .unwrap())
}
```

## Phase 2: Sandbox System Enhancement

### Task 2.1: Enhanced Sandbox Trading Engine
**File**: `services/gateway/src/sandbox.rs`

**Enhancement Areas**:

1. **Order Matching Engine**:
```rust
impl SandboxTradingEngine {
    pub async fn process_market_order(&self, order: &mut Order) -> Result<(), SandboxError> {
        // Get current market price (from market data or use last price)
        let market_price = self.get_market_price(&order.symbol).await?;
        
        // Execute market order immediately at market price
        order.status = OrderStatus::Filled;
        order.filled_quantity = order.quantity;
        order.average_fill_price = Some(market_price);
        order.updated_at = chrono::Utc::now();
        
        // Update positions and balances
        self.update_position_from_fill(order).await?;
        self.update_balance_from_fill(order).await?;
        
        Ok(())
    }
    
    pub async fn process_limit_order(&self, order: &mut Order) -> Result<(), SandboxError> {
        let market_price = self.get_market_price(&order.symbol).await?;
        let order_price = order.price.unwrap_or(market_price);
        
        // Check if limit order can be filled
        let can_fill = match order.side {
            Side::Buy => market_price <= order_price,
            Side::Sell => market_price >= order_price,
        };
        
        if can_fill {
            // Fill the order
            order.status = OrderStatus::Filled;
            order.filled_quantity = order.quantity;
            order.average_fill_price = Some(order_price);
            order.updated_at = chrono::Utc::now();
            
            self.update_position_from_fill(order).await?;
            self.update_balance_from_fill(order).await?;
        } else {
            // Keep order open
            order.status = OrderStatus::New;
        }
        
        Ok(())
    }
}
```

2. **Portfolio Calculations**:
```rust
impl SandboxTradingEngine {
    pub async fn get_account_stats(&self, account_id: &str) -> Result<TradingStats, SandboxError> {
        let positions = self.get_positions(account_id).await?;
        let orders = self.get_orders_history(account_id).await?;
        
        let total_orders = orders.len() as u32;
        let filled_orders = orders.iter()
            .filter(|o| o.status == OrderStatus::Filled)
            .count() as u32;
        
        let total_volume = orders.iter()
            .filter(|o| o.status == OrderStatus::Filled)
            .map(|o| o.filled_quantity * o.average_fill_price.unwrap_or(0.0))
            .sum::<f64>();
        
        let realized_pnl = self.calculate_realized_pnl(account_id, &orders).await;
        let unrealized_pnl = self.calculate_unrealized_pnl(account_id, &positions).await;
        
        let winning_trades = orders.iter()
            .filter(|o| o.status == OrderStatus::Filled)
            .filter(|o| self.is_winning_trade(o))
            .count() as f64;
        
        let win_rate = if filled_orders > 0 {
            winning_trades / filled_orders as f64 * 100.0
        } else {
            0.0
        };
        
        Ok(TradingStats {
            account_id: account_id.to_string(),
            total_orders,
            filled_orders,
            total_volume,
            realized_pnl,
            unrealized_pnl,
            win_rate,
            period_start: chrono::Utc::now().date_naive().to_string(),
            period_end: chrono::Utc::now().date_naive().to_string(),
        })
    }
}
```

### Task 2.2: Enhanced Market Data Simulation
**File**: `services/gateway/src/handlers/markets_mvp.rs`

**Enhancement Steps**:

1. **Realistic Price Generation**:
```rust
use rand::Rng;

pub struct MarketDataSimulator {
    base_prices: HashMap<String, f64>,
    volatility: HashMap<String, f64>,
}

impl MarketDataSimulator {
    pub fn new() -> Self {
        let mut base_prices = HashMap::new();
        base_prices.insert("BTCUSDT".to_string(), 45000.0);
        base_prices.insert("ETHUSDT".to_string(), 2800.0);
        base_prices.insert("BNBUSDT".to_string(), 320.0);
        
        let mut volatility = HashMap::new();
        volatility.insert("BTCUSDT".to_string(), 0.02);
        volatility.insert("ETHUSDT".to_string(), 0.025);
        volatility.insert("BNBUSDT".to_string(), 0.03);
        
        Self { base_prices, volatility }
    }
    
    pub fn generate_realistic_candles(&self, symbol: &str, count: usize) -> Vec<Candle> {
        let base_price = self.base_prices.get(symbol).copied().unwrap_or(1000.0);
        let vol = self.volatility.get(symbol).copied().unwrap_or(0.02);
        
        let mut candles = Vec::new();
        let mut rng = rand::thread_rng();
        let mut current_price = base_price;
        
        for i in 0..count {
            let change_percent = rng.gen_range(-vol..vol);
            let new_price = current_price * (1.0 + change_percent);
            
            let high = current_price.max(new_price) * (1.0 + rng.gen_range(0.0..0.01));
            let low = current_price.min(new_price) * (1.0 - rng.gen_range(0.0..0.01));
            let volume = rng.gen_range(1000.0..5000.0);
            
            candles.push(Candle {
                open_time: chrono::Utc::now().timestamp() - (count - i) as i64 * 3600,
                close_time: chrono::Utc::now().timestamp() - (count - i - 1) as i64 * 3600,
                open: current_price,
                high,
                low,
                close: new_price,
                volume,
                trades: rng.gen_range(100..500),
            });
            
            current_price = new_price;
        }
        
        candles
    }
}
```

## Phase 3: Frontend Integration Support

### Task 3.1: WebSocket Message Broadcasting
**File**: `services/gateway/src/handlers/websocket.rs`

**Enhancement Steps**:

1. **Order Update Broadcasting**:
```rust
impl WebSocketManager {
    pub async fn broadcast_order_update(&self, account_id: &str, order: &Order) {
        let message = WebSocketMessage {
            message_type: "data".to_string(),
            channel: format!("orders:{}", account_id),
            data: serde_json::to_value(order).unwrap_or_default(),
            timestamp: chrono::Utc::now().timestamp(),
        };
        
        if let Err(e) = self.broadcast_to_channel(&message.channel, &message).await {
            tracing::error!("Failed to broadcast order update: {}", e);
        }
    }
    
    pub async fn broadcast_market_data(&self, symbol: &str, trade: &Trade) {
        let message = WebSocketMessage {
            message_type: "data".to_string(), 
            channel: format!("trades:{}", symbol),
            data: serde_json::to_value(trade).unwrap_or_default(),
            timestamp: chrono::Utc::now().timestamp(),
        };
        
        if let Err(e) = self.broadcast_to_channel(&message.channel, &message).await {
            tracing::error!("Failed to broadcast trade: {}", e);
        }
    }
}
```

2. **Integration with Trading Engine**:
```rust
// In sandbox.rs
impl SandboxTradingEngine {
    pub async fn create_order_with_broadcast(
        &self,
        request: CreateOrderRequest,
        ws_manager: Arc<WebSocketManager>,
    ) -> Result<Order, SandboxError> {
        let order = self.create_order(request).await?;
        
        // Broadcast order update
        ws_manager.broadcast_order_update(&order.account_id, &order).await;
        
        Ok(order)
    }
}
```

### Task 3.2: Error Handling Enhancement
**File**: `services/gateway/src/types.rs`

**Enhancement Steps**:

1. **Enhanced Error Context**:
```rust
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("Validation failed: {details}")]
    Validation { 
        details: String,
        errors: Vec<ValidationError>,
    },
    
    #[error("Resource not found: {resource}")]
    NotFound { resource: String },
    
    #[error("Business rule violation: {rule}")]
    BusinessRule { rule: String, context: String },
    
    #[error("External service error: {service}")]
    ExternalService { 
        service: String,
        error: String,
        fallback_used: bool,
    },
    
    // ... existing variants
}

#[derive(Debug, Serialize)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
    pub code: String,
}

impl ApiError {
    pub fn validation(field: &str, message: &str, code: &str) -> Self {
        Self::Validation {
            details: format!("Field '{}': {}", field, message),
            errors: vec![ValidationError {
                field: field.to_string(),
                message: message.to_string(),
                code: code.to_string(),
            }],
        }
    }
}
```

2. **Enhanced Error Responses**:
```rust
impl axum::response::IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        use axum::Json;
        
        let (status, error_data) = match &self {
            ApiError::Validation { errors, .. } => {
                (StatusCode::UNPROCESSABLE_ENTITY, Some(json!({
                    "errors": errors
                })))
            },
            ApiError::NotFound { resource } => {
                (StatusCode::NOT_FOUND, Some(json!({
                    "resource": resource
                })))
            },
            ApiError::ExternalService { service, fallback_used, .. } => {
                (StatusCode::SERVICE_UNAVAILABLE, Some(json!({
                    "service": service,
                    "fallback": fallback_used
                })))
            },
            _ => (self.status_code(), None),
        };
        
        let mut error_response = ApiResponse::<()>::error(self.to_string());
        if let Some(data) = error_data {
            error_response.data = Some(data);
        }
        
        (status, Json(error_response)).into_response()
    }
}
```

## Phase 4: Production Readiness

### Task 4.1: Configuration Management
**File**: `services/gateway/src/config.rs`

**Implementation**:

1. **Centralized Configuration**:
```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    pub server: ServerConfig,
    pub database: Option<DatabaseConfig>,
    pub cache: CacheConfig,
    pub external_services: ExternalServicesConfig,
    pub features: FeatureFlags,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: Option<usize>,
    pub keep_alive: u64,
    pub request_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlags {
    pub authentication: bool,
    pub sandbox_trading: bool,
    pub rate_limiting: bool,
    pub metrics_export: bool,
}

impl GatewayConfig {
    pub fn load() -> Result<Self, config::ConfigError> {
        let mut settings = config::Config::default();
        
        // Load from environment variables
        settings.merge(config::Environment::with_prefix("GATEWAY"))?;
        
        // Load from config file if exists
        if let Ok(config_path) = std::env::var("GATEWAY_CONFIG_PATH") {
            settings.merge(config::File::with_name(&config_path).required(false))?;
        }
        
        settings.try_into()
    }
}
```

### Task 4.2: Metrics and Observability
**File**: `services/gateway/src/metrics.rs`

**Implementation**:

1. **Business Metrics**:
```rust
use std::collections::HashMap;
use tokio::sync::RwLock;

pub struct BusinessMetrics {
    order_metrics: Arc<RwLock<OrderMetrics>>,
    market_metrics: Arc<RwLock<MarketMetrics>>,
}

#[derive(Default)]
pub struct OrderMetrics {
    pub orders_created: u64,
    pub orders_filled: u64,
    pub orders_cancelled: u64,
    pub total_volume: f64,
    pub orders_by_symbol: HashMap<String, u64>,
}

impl BusinessMetrics {
    pub async fn record_order_created(&self, symbol: &str, quantity: f64) {
        let mut metrics = self.order_metrics.write().await;
        metrics.orders_created += 1;
        metrics.total_volume += quantity;
        *metrics.orders_by_symbol.entry(symbol.to_string()).or_insert(0) += 1;
    }
    
    pub async fn get_metrics_snapshot(&self) -> MetricsSnapshot {
        let order_metrics = self.order_metrics.read().await;
        let market_metrics = self.market_metrics.read().await;
        
        MetricsSnapshot {
            timestamp: chrono::Utc::now().to_rfc3339(),
            orders: OrderMetricsSnapshot {
                created: order_metrics.orders_created,
                filled: order_metrics.orders_filled,
                cancelled: order_metrics.orders_cancelled,
                total_volume: order_metrics.total_volume,
            },
            // ... other metrics
        }
    }
}
```

### Task 4.3: Health Check Enhancement
**File**: `services/gateway/src/handlers/health.rs`

**Implementation**:

1. **Comprehensive Health Checks**:
```rust
pub async fn health_check_detailed(
    State(app_state): State<AppState>,
) -> Result<Json<ApiResponse<DetailedHealth>>, ApiError> {
    let start_time = std::time::Instant::now();
    
    // Check all subsystems
    let cache_health = check_cache_health(&app_state.cache).await;
    let sandbox_health = check_sandbox_health(&app_state.sandbox_engine).await;
    let websocket_health = check_websocket_health(&app_state.websocket_manager).await;
    
    let overall_status = if cache_health.healthy && sandbox_health.healthy && websocket_health.healthy {
        "healthy"
    } else {
        "degraded"
    };
    
    let health = DetailedHealth {
        service: "gateway".to_string(),
        status: overall_status.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: app_state.get_uptime_seconds(),
        subsystems: HashMap::from([
            ("cache".to_string(), cache_health),
            ("sandbox".to_string(), sandbox_health),
            ("websocket".to_string(), websocket_health),
        ]),
        check_duration_ms: start_time.elapsed().as_millis() as u64,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };
    
    Ok(Json(ApiResponse::success(health)))
}

async fn check_cache_health(cache: &CacheBackend) -> SubsystemHealth {
    match cache.ping().await {
        Ok(_) => SubsystemHealth {
            healthy: true,
            message: "Cache responsive".to_string(),
            last_check: chrono::Utc::now().to_rfc3339(),
            metrics: Some(json!({
                "backend": cache.backend_type(),
                "hit_rate": cache.hit_rate().await.unwrap_or(0.0)
            })),
        },
        Err(e) => SubsystemHealth {
            healthy: false,
            message: format!("Cache error: {}", e),
            last_check: chrono::Utc::now().to_rfc3339(),
            metrics: None,
        },
    }
}
```

## Testing Strategy

### Unit Testing
**File**: `services/gateway/tests/unit/`

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use axum_test::TestServer;
    
    #[tokio::test]
    async fn test_create_order_success() {
        let app = create_test_app().await;
        let server = TestServer::new(app).unwrap();
        
        let response = server
            .post("/api/v1/orders")
            .json(&json!({
                "symbol": "BTCUSDT",
                "side": "BUY",
                "order_type": "LIMIT",
                "quantity": 0.01,
                "price": 35000.0,
                "account_id": "test"
            }))
            .await;
        
        response.assert_status_ok();
        
        let body: ApiResponse<Order> = response.json();
        assert_eq!(body.status, "success");
        assert_eq!(body.data.symbol, "BTCUSDT");
    }
    
    #[tokio::test]
    async fn test_order_validation_error() {
        let app = create_test_app().await;
        let server = TestServer::new(app).unwrap();
        
        let response = server
            .post("/api/v1/orders")
            .json(&json!({
                "symbol": "",  // Invalid empty symbol
                "side": "BUY",
                "order_type": "LIMIT",
                "quantity": -0.01,  // Invalid negative quantity
                "price": 0.0,       // Invalid zero price
                "account_id": "test"
            }))
            .await;
        
        response.assert_status(422);
        
        let body: ApiResponse<()> = response.json();
        assert_eq!(body.status, "error");
        assert!(body.message.contains("Validation failed"));
    }
    
    async fn create_test_app() -> axum::Router {
        let config = GatewayConfig::default();
        let app_state = create_test_app_state(&config).await;
        crate::app::create_app(app_state, /* other deps */)
    }
}
```

### Integration Testing
**File**: `services/gateway/tests/integration/`

```rust
#[tokio::test]
async fn test_trading_workflow_end_to_end() {
    let server = start_test_server().await;
    
    // 1. Create an order
    let order_response = server
        .post("/api/v1/orders")
        .json(&create_order_request())
        .await;
    
    let order: Order = order_response.json::<ApiResponse<Order>>().data;
    
    // 2. Verify order exists
    let get_response = server
        .get(&format!("/api/v1/orders/{}", order.id))
        .await;
    
    get_response.assert_status_ok();
    
    // 3. Check positions updated
    let positions_response = server
        .get("/api/v1/positions?account_id=test")
        .await;
    
    let positions: PositionList = positions_response.json::<ApiResponse<PositionList>>().data;
    assert!(!positions.positions.is_empty());
    
    // 4. Cancel order
    let cancel_response = server
        .delete(&format!("/api/v1/orders/{}", order.id))
        .await;
    
    cancel_response.assert_status_ok();
}
```

## Deployment Instructions

### Local Development
```bash
# 1. Set environment variables
export DISABLE_AUTH=1
export SANDBOX_TRADING=1
export RATE_LIMIT_RPS=20
export RUST_LOG=debug

# 2. Run migrations (if database is configured)
cargo run --bin migrate

# 3. Start the gateway
cargo run -p gateway-service

# 4. Start frontend (separate terminal)
cd frontend && npm run dev
```

### Docker Deployment
**File**: `Dockerfile`

```dockerfile
FROM rust:1.82-slim as builder

WORKDIR /app
COPY . .
RUN cargo build --release -p gateway-service

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/gateway-service /usr/local/bin/

EXPOSE 8080

CMD ["gateway-service"]
```

### Production Checklist

#### Pre-deployment
- [ ] All tests passing
- [ ] No compilation warnings
- [ ] Configuration validated
- [ ] Health checks responding
- [ ] Metrics collection working
- [ ] Log aggregation configured

#### Post-deployment
- [ ] Health endpoints accessible
- [ ] API endpoints responding correctly
- [ ] WebSocket connections working
- [ ] Error rates within acceptable limits
- [ ] Response times meeting SLA
- [ ] Logs flowing to aggregation system

## Monitoring and Alerting

### Key Metrics to Monitor
1. **HTTP Metrics**:
   - Request rate (requests/second)
   - Response time (95th percentile)
   - Error rate (4xx/5xx responses)

2. **Business Metrics**:
   - Orders created/filled/cancelled per minute
   - Trading volume
   - Active WebSocket connections

3. **System Metrics**:
   - CPU and memory usage
   - Cache hit rate
   - External service response times

### Alert Thresholds
```yaml
alerts:
  - name: "High Error Rate"
    condition: "error_rate > 5%"
    duration: "5m"
    
  - name: "High Response Time"  
    condition: "response_time_p95 > 1000ms"
    duration: "2m"
    
  - name: "Low Cache Hit Rate"
    condition: "cache_hit_rate < 80%"
    duration: "10m"
```

## Troubleshooting Guide

### Common Issues

#### 1. High Memory Usage
**Symptoms**: Memory usage growing over time
**Diagnosis**: 
```bash
# Check for memory leaks in cache
curl http://localhost:8080/api/v1/system/info | jq '.data.memory'
```
**Solution**: Implement cache TTL and size limits

#### 2. WebSocket Connection Issues  
**Symptoms**: Clients unable to connect to WebSocket
**Diagnosis**:
```bash
# Test WebSocket connection
wscat -c ws://localhost:8080/api/v1/ws
```
**Solution**: Check CORS configuration and proxy settings

#### 3. Rate Limiting False Positives
**Symptoms**: Legitimate requests getting 429 responses
**Diagnosis**: Check rate limiting logs
**Solution**: Adjust `RATE_LIMIT_RPS` or implement user-specific limits

## Conclusion

This implementation guide provides a comprehensive roadmap for completing the Sprint 1 MVP. The key focus areas are:

1. **Complete missing MVP handlers** (Analytics, Risk, Audit, Reports)
2. **Enhance sandbox trading engine** with realistic behavior
3. **Improve error handling and observability**
4. **Prepare for production deployment**

Following this guide ensures:
- ✅ Zero compilation warnings
- ✅ Complete MVP functionality
- ✅ Production-ready observability
- ✅ Comprehensive testing coverage
- ✅ Clear deployment procedures

The implementation maintains the Rust-first, unified gateway architecture while providing a solid foundation for future enhancements.