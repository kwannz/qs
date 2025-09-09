# Sprint 1 API Specification - Unified Interface Pattern

## Overview

This document defines the complete API specification for the Sprint 1 MVP, following the unified interface pattern with consistent response structures across all domains.

## Global Conventions

### Base URL
- **Development**: `http://localhost:8080`
- **Environment Variable**: `VITE_GATEWAY_URL`

### Request Headers
```http
Content-Type: application/json
idempotency-key: <uuid>  # Required for write operations (POST/PUT/DELETE)
x-trace-id: <uuid>       # Optional, auto-generated if not provided
```

### Response Format
All APIs return the unified response structure:

```typescript
interface ApiResponse<T> {
  status: "success" | "error";
  data: T | null;
  message?: string;
  timestamp: string;        // ISO-8601 format
  trace_id?: string;        // Request correlation ID
  execution_time_ms: number;
}
```

### HTTP Status Codes
- `200 OK`: Successful operation
- `201 Created`: Resource created successfully
- `202 Accepted`: Async operation accepted
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource conflict (duplicate, constraint violation)
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: External service unavailable (placeholder mode)

## Health & System APIs

### Health Check
```http
GET /health
GET /healthz
GET /api/health  # Legacy compatibility
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "service": "gateway",
    "status": "healthy",
    "uptime_seconds": 3600,
    "timestamp": "2024-01-01T00:00:00Z"
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 1
}
```

### Readiness & Liveness
```http
GET /readyz    # Readiness probe
GET /livez     # Liveness probe
```

**Response:** Same format as health check with appropriate status.

### System Information
```http
GET /api/v1/system/info
```

**Response:**
```json
{
  "status": "success", 
  "data": {
    "name": "Trading Gateway",
    "version": "1.0.0",
    "uptime_seconds": 3600,
    "environment": "development",
    "features": {
      "sandbox_trading": true,
      "authentication": false,
      "rate_limiting": true
    },
    "services": {
      "execution": "sandbox",
      "markets": "placeholder",
      "analytics": "placeholder"
    }
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 2
}
```

## Execution APIs (Trading)

### Create Order
```http
POST /api/v1/orders
```

**Request:**
```json
{
  "symbol": "BTCUSDT",
  "side": "BUY" | "SELL",
  "order_type": "MARKET" | "LIMIT" | "STOP_LOSS" | "TAKE_PROFIT",
  "quantity": 0.01,
  "price": 35000.0,        // Required for LIMIT orders
  "stop_price": 34000.0,   // Required for STOP orders
  "account_id": "demo",
  "time_in_force": "GTC" | "IOC" | "FOK"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "ord_123456789",
    "symbol": "BTCUSDT", 
    "side": "BUY",
    "order_type": "LIMIT",
    "quantity": 0.01,
    "price": 35000.0,
    "status": "NEW",
    "account_id": "demo",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z"
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 10
}
```

### Get Order
```http
GET /api/v1/orders/{id}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "ord_123456789",
    "symbol": "BTCUSDT",
    "side": "BUY", 
    "order_type": "LIMIT",
    "quantity": 0.01,
    "price": 35000.0,
    "filled_quantity": 0.005,
    "remaining_quantity": 0.005,
    "status": "PARTIALLY_FILLED",
    "account_id": "demo",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:01Z",
    "fills": [
      {
        "price": 35000.0,
        "quantity": 0.005,
        "timestamp": "2024-01-01T00:00:01Z"
      }
    ]
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 5
}
```

### Cancel Order
```http
DELETE /api/v1/orders/{id}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "ord_123456789",
    "status": "CANCELLED",
    "cancelled_at": "2024-01-01T00:00:02Z"
  },
  "timestamp": "2024-01-01T00:00:02Z",
  "execution_time_ms": 8
}
```

### List Orders
```http
GET /api/v1/orders?symbol=BTCUSDT&status=FILLED&limit=50&offset=0
```

**Query Parameters:**
- `symbol`: Filter by trading symbol
- `status`: Filter by order status
- `side`: Filter by BUY/SELL
- `limit`: Number of results (default: 50, max: 100)
- `offset`: Pagination offset

**Response:**
```json
{
  "status": "success",
  "data": {
    "orders": [...],  // Array of order objects
    "total": 150,
    "limit": 50,
    "offset": 0,
    "has_more": true
  },
  "timestamp": "2024-01-01T00:00:00Z", 
  "execution_time_ms": 12
}
```

### List Positions
```http
GET /api/v1/positions?account_id=demo&limit=50&offset=0
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "positions": [
      {
        "symbol": "BTCUSDT",
        "side": "LONG",
        "quantity": 0.1,
        "average_price": 35000.0,
        "unrealized_pnl": 500.0,
        "account_id": "demo",
        "updated_at": "2024-01-01T00:00:00Z"
      }
    ],
    "total": 5,
    "limit": 50,
    "offset": 0
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 8
}
```

### List Balances
```http
GET /api/v1/balances?account_id=demo
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "balances": [
      {
        "asset": "BTC",
        "free": 1.5,
        "locked": 0.1,
        "total": 1.6,
        "account_id": "demo"
      },
      {
        "asset": "USDT", 
        "free": 50000.0,
        "locked": 5000.0,
        "total": 55000.0,
        "account_id": "demo"
      }
    ],
    "account_id": "demo",
    "updated_at": "2024-01-01T00:00:00Z"
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 6
}
```

## Markets APIs (Market Data)

### List Exchanges
```http
GET /api/v1/exchanges
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "exchanges": [
      {
        "id": "binance",
        "name": "Binance",
        "status": "active",
        "features": ["spot", "futures", "options"]
      },
      {
        "id": "coinbase", 
        "name": "Coinbase Pro",
        "status": "active",
        "features": ["spot"]
      }
    ]
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 5
}
```

### List Symbols
```http
GET /api/v1/symbols?exchange=binance&active_only=true&limit=100&offset=0
```

**Query Parameters:**
- `exchange`: Filter by exchange
- `base_asset`: Filter by base asset (e.g., BTC)
- `quote_asset`: Filter by quote asset (e.g., USDT)
- `active_only`: Only active trading pairs
- `limit`: Results per page
- `offset`: Pagination offset

**Response:**
```json
{
  "status": "success",
  "data": {
    "symbols": [
      {
        "symbol": "BTCUSDT",
        "base_asset": "BTC",
        "quote_asset": "USDT",
        "exchange": "binance",
        "status": "TRADING",
        "price_precision": 2,
        "quantity_precision": 6,
        "min_quantity": 0.000001,
        "max_quantity": 100000.0
      }
    ],
    "total": 2500,
    "limit": 100,
    "offset": 0,
    "has_more": true
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 15
}
```

### Search Symbols
```http
GET /api/v1/symbols/search?q=BTC&exchange=binance&active_only=true&limit=20
```

**Query Parameters:**
- `q`: Search query (symbol, base asset, or quote asset)
- `exchange`: Filter by exchange
- `active_only`: Only active pairs
- `limit`: Maximum results

**Response:** Same format as List Symbols

### Get Market Candles
```http
GET /api/v1/markets/{symbol}/candles?interval=1h&limit=100&start_time=1640995200&end_time=1641000000
```

**Path Parameters:**
- `symbol`: Trading symbol (e.g., BTCUSDT)

**Query Parameters:**
- `interval`: Candle interval (1m, 5m, 15m, 1h, 4h, 1d, 1w, 1M)
- `limit`: Number of candles (default: 100, max: 1000)
- `start_time`: Unix timestamp (seconds)
- `end_time`: Unix timestamp (seconds)

**Response:**
```json
{
  "status": "success",
  "data": {
    "symbol": "BTCUSDT",
    "interval": "1h",
    "candles": [
      {
        "open_time": 1640995200,
        "close_time": 1640998799,
        "open": 47000.0,
        "high": 47500.0,
        "low": 46800.0,
        "close": 47200.0,
        "volume": 1250.5,
        "trades": 8500
      }
    ]
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 25
}
```

### Get Order Book
```http
GET /api/v1/markets/{symbol}/orderbook?depth=20
```

**Query Parameters:**
- `depth`: Number of price levels (5, 10, 20, 50, 100)

**Response:**
```json
{
  "status": "success",
  "data": {
    "symbol": "BTCUSDT",
    "timestamp": 1640995200,
    "bids": [
      [47000.0, 1.5],    // [price, quantity]
      [46999.0, 2.1]
    ],
    "asks": [
      [47001.0, 0.8],
      [47002.0, 1.2]
    ]
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 10
}
```

### Get Recent Trades
```http
GET /api/v1/markets/{symbol}/trades?limit=100
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "symbol": "BTCUSDT",
    "trades": [
      {
        "id": "12345",
        "price": 47000.0,
        "quantity": 0.1,
        "side": "BUY",
        "timestamp": 1640995200
      }
    ]
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 12
}
```

### Get Market Stats
```http
GET /api/v1/markets/stats?symbols=BTCUSDT,ETHUSDT
```

**Response:**
```json
{
  "status": "success", 
  "data": {
    "stats": [
      {
        "symbol": "BTCUSDT",
        "price_change": 1200.0,
        "price_change_percent": 2.61,
        "last_price": 47000.0,
        "high_price": 47500.0,
        "low_price": 45800.0,
        "volume": 125000.5,
        "quote_volume": 5875000000.0,
        "open_time": 1640908800,
        "close_time": 1640995200
      }
    ]
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 18
}
```

## Strategy APIs

### List Strategies
```http
GET /api/v1/strategies?status=active&limit=20&offset=0
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "strategies": [
      {
        "id": "strat_123",
        "name": "MACD Cross Strategy",
        "type": "technical",
        "status": "active",
        "description": "Moving average convergence divergence strategy",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z"
      }
    ],
    "total": 10,
    "limit": 20,
    "offset": 0
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 8
}
```

### Create Strategy
```http
POST /api/v1/strategies
```

**Request:**
```json
{
  "name": "RSI Strategy",
  "type": "technical",
  "description": "RSI-based trading strategy",
  "parameters": {
    "rsi_period": 14,
    "overbought": 70,
    "oversold": 30
  },
  "symbols": ["BTCUSDT", "ETHUSDT"],
  "risk_limits": {
    "max_position_size": 1000.0,
    "stop_loss_percent": 2.0
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "strat_124",
    "name": "RSI Strategy", 
    "type": "technical",
    "status": "inactive",
    "parameters": {...},
    "created_at": "2024-01-01T00:00:00Z"
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 15
}
```

### Start/Stop Strategy
```http
POST /api/v1/strategies/{id}/start
POST /api/v1/strategies/{id}/stop
POST /api/v1/strategies/{id}/pause
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "strat_123",
    "status": "active",
    "updated_at": "2024-01-01T00:00:00Z"
  },
  "timestamp": "2024-01-01T00:00:00Z", 
  "execution_time_ms": 5
}
```

## Config & Alerts APIs

### Get User Config
```http
GET /api/v1/config/user
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "user_id": "demo",
    "preferences": {
      "theme": "dark",
      "language": "en",
      "timezone": "UTC",
      "notifications": {
        "email": true,
        "push": false,
        "sms": false
      }
    },
    "trading": {
      "default_account": "demo",
      "risk_tolerance": "medium",
      "auto_confirm_orders": false
    },
    "updated_at": "2024-01-01T00:00:00Z"
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 3
}
```

### Update User Config
```http
PUT /api/v1/config/user
```

**Request:**
```json
{
  "preferences": {
    "theme": "light",
    "notifications": {
      "email": false,
      "push": true
    }
  }
}
```

### Get System Config
```http
GET /api/v1/config/system
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "features": {
      "authentication": false,
      "sandbox_mode": true,
      "rate_limiting": true
    },
    "limits": {
      "max_orders_per_user": 1000,
      "max_positions_per_user": 100,
      "rate_limit_rps": 20
    },
    "supported_exchanges": ["binance", "coinbase"],
    "supported_intervals": ["1m", "5m", "15m", "1h", "4h", "1d"]
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 2
}
```

## Audit APIs

### Create Audit Event
```http
POST /api/v1/audit
```

**Request:**
```json
{
  "actor": "user:demo",
  "action": "order.create",
  "target": "order:ord_123456789",
  "success": true,
  "details": {
    "symbol": "BTCUSDT",
    "quantity": 0.01,
    "price": 35000.0
  },
  "ip_address": "192.168.1.100",
  "user_agent": "Mozilla/5.0..."
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "audit_789",
    "timestamp": "2024-01-01T00:00:00Z",
    "stored": true
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 8
}
```

### List Audit Events
```http
GET /api/v1/audit?actor=user:demo&action=order.create&limit=50&offset=0&start_time=1640995200&end_time=1641000000
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "events": [
      {
        "id": "audit_789",
        "actor": "user:demo",
        "action": "order.create",
        "target": "order:ord_123456789",
        "success": true,
        "timestamp": "2024-01-01T00:00:00Z",
        "ip_address": "192.168.1.100"
      }
    ],
    "total": 500,
    "limit": 50,
    "offset": 0,
    "has_more": true
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 12
}
```

### Export Audit Events
```http
GET /api/v1/audit/export?format=csv&start_time=1640995200&end_time=1641000000
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "format": "csv",
    "download_url": "/api/v1/audit/download/audit_20240101_20240102.csv",
    "expires_at": "2024-01-01T01:00:00Z",
    "record_count": 1500
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 25
}
```

## WebSocket API

### Connection
```http
GET /api/v1/ws
Upgrade: websocket
Connection: Upgrade
```

### Message Format
```json
{
  "type": "subscribe" | "unsubscribe" | "data" | "error" | "heartbeat",
  "channel": "orders:BTCUSDT" | "trades:BTCUSDT" | "orderbook:BTCUSDT",
  "data": {...},
  "timestamp": 1640995200
}
```

### Subscribe to Channel
```json
{
  "type": "subscribe",
  "channel": "trades:BTCUSDT"
}
```

### Market Data Updates
```json
{
  "type": "data",
  "channel": "trades:BTCUSDT",
  "data": {
    "symbol": "BTCUSDT",
    "price": 47000.0,
    "quantity": 0.1,
    "side": "BUY",
    "timestamp": 1640995200
  },
  "timestamp": 1640995200
}
```

### Order Updates
```json
{
  "type": "data",
  "channel": "orders:user:demo",
  "data": {
    "id": "ord_123456789",
    "status": "FILLED",
    "filled_quantity": 0.01,
    "updated_at": "2024-01-01T00:00:00Z"
  },
  "timestamp": 1640995200
}
```

## Analytics APIs (Placeholder)

### Create Analysis Task
```http
POST /api/v1/analytics/tasks
```

**Request:**
```json
{
  "type": "portfolio_analysis",
  "parameters": {
    "account_id": "demo",
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "metrics": ["sharpe_ratio", "max_drawdown", "total_return"]
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "task_id": "task_456",
    "type": "portfolio_analysis", 
    "status": "accepted",
    "estimated_completion": "2024-01-01T00:05:00Z",
    "created_at": "2024-01-01T00:00:00Z"
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 10
}
```

### Get Analysis Task
```http
GET /api/v1/analytics/tasks/{id}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "task_id": "task_456",
    "type": "portfolio_analysis",
    "status": "completed",
    "progress": 100,
    "result": {
      "sharpe_ratio": 1.25,
      "max_drawdown": -5.2,
      "total_return": 12.5
    },
    "created_at": "2024-01-01T00:00:00Z",
    "completed_at": "2024-01-01T00:04:30Z"
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 5
}
```

## Risk APIs (Placeholder)

### Risk Assessment
```http
GET /api/v1/risk/assessment?account_id=demo
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "account_id": "demo",
    "risk_score": 6.5,
    "risk_level": "medium",
    "metrics": {
      "portfolio_var": 1250.0,
      "concentration_risk": 0.3,
      "leverage_ratio": 1.8
    },
    "warnings": [
      "High concentration in BTC positions"
    ],
    "updated_at": "2024-01-01T00:00:00Z"
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "execution_time_ms": 15
}
```

## Error Responses

### Validation Error (422)
```json
{
  "status": "error",
  "data": null,
  "message": "Validation failed",
  "timestamp": "2024-01-01T00:00:00Z",
  "trace_id": "trace_123",
  "execution_time_ms": 2,
  "errors": [
    {
      "field": "quantity",
      "message": "Must be greater than minimum quantity 0.000001",
      "code": "MIN_VALUE_ERROR"
    }
  ]
}
```

### Rate Limit Error (429)
```json
{
  "status": "error",
  "data": null,
  "message": "Rate limit exceeded. Try again in 60 seconds.",
  "timestamp": "2024-01-01T00:00:00Z",
  "trace_id": "trace_124",
  "execution_time_ms": 1,
  "retry_after": 60
}
```

### Service Unavailable (503)
```json
{
  "status": "error",
  "data": null,
  "message": "External service temporarily unavailable. Using placeholder data.",
  "timestamp": "2024-01-01T00:00:00Z",
  "trace_id": "trace_125", 
  "execution_time_ms": 100,
  "service": "markets",
  "fallback": true
}
```

## Frontend Integration Examples

### TypeScript HTTP Client
```typescript
interface ApiResponse<T> {
  status: 'success' | 'error';
  data: T | null;
  message?: string;
  timestamp: string;
  trace_id?: string;
  execution_time_ms: number;
}

class GatewayClient {
  private baseUrl: string;

  constructor() {
    this.baseUrl = import.meta.env.VITE_GATEWAY_URL || 'http://localhost:8080';
  }

  async request<T>(
    path: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const headers = new Headers(options.headers);
    headers.set('Content-Type', 'application/json');

    // Auto-inject idempotency key for write operations
    if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(options.method?.toUpperCase() || '')) {
      headers.set('idempotency-key', crypto.randomUUID());
    }

    const response = await fetch(`${this.baseUrl}${path}`, {
      ...options,
      headers,
    });

    const data: ApiResponse<T> = await response.json();

    if (!response.ok) {
      throw new Error(data.message || `HTTP ${response.status}`);
    }

    return data;
  }

  // Trading APIs
  async createOrder(order: CreateOrderRequest) {
    return this.request<Order>('/api/v1/orders', {
      method: 'POST',
      body: JSON.stringify(order),
    });
  }

  async getOrder(orderId: string) {
    return this.request<Order>(`/api/v1/orders/${orderId}`);
  }

  async listPositions(accountId: string) {
    return this.request<PositionList>(`/api/v1/positions?account_id=${accountId}`);
  }

  // Market Data APIs
  async getCandles(symbol: string, interval: string, limit: number = 100) {
    return this.request<CandleData>(
      `/api/v1/markets/${symbol}/candles?interval=${interval}&limit=${limit}`
    );
  }

  async getOrderBook(symbol: string, depth: number = 20) {
    return this.request<OrderBook>(`/api/v1/markets/${symbol}/orderbook?depth=${depth}`);
  }
}

// Usage example
const client = new GatewayClient();

try {
  const response = await client.createOrder({
    symbol: 'BTCUSDT',
    side: 'BUY',
    order_type: 'LIMIT',
    quantity: 0.01,
    price: 35000,
    account_id: 'demo',
  });
  
  console.log('Order created:', response.data);
} catch (error) {
  console.error('Order failed:', error.message);
}
```

### React Hook Integration
```typescript
import { useState, useEffect } from 'react';

function useOrderBook(symbol: string) {
  const [orderBook, setOrderBook] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchOrderBook = async () => {
      try {
        setLoading(true);
        const response = await client.getOrderBook(symbol);
        setOrderBook(response.data);
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchOrderBook();
    const interval = setInterval(fetchOrderBook, 1000);
    
    return () => clearInterval(interval);
  }, [symbol]);

  return { orderBook, loading, error };
}
```

## Summary

This API specification provides a comprehensive interface for the Sprint 1 MVP, following the unified response pattern and Rust-first architecture principles. The APIs are designed to:

1. **Consistency**: All APIs follow the same response structure and error handling
2. **Observability**: Built-in tracing, timing, and error tracking
3. **Reliability**: Graceful degradation when external services are unavailable
4. **Performance**: Efficient pagination and caching strategies
5. **Developer Experience**: Clear documentation and TypeScript integration examples

The specification supports both MVP requirements (sandbox trading, placeholder data) and future evolution (real external service integration, advanced features).