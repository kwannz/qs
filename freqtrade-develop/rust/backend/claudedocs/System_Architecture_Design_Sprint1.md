# Sprint 1 System Architecture Design - Rust-First Gateway

## Executive Summary

**Architecture Philosophy**: Unified Rust Gateway as Single Point of Entry
- **Rust-first**: Backend centered on Rust gateway, no separate Admin/Monitoring/Reporting services
- **MVP**: No authentication, no JWT, no API Key - frontend calls directly  
- **Unified Interface**: Consistent API paths and response structure across all domains

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend (React/Vite)                       │
│                  http://localhost:5173                         │
│                                                                 │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────┐  │
│  │  Dashboard  │  Services   │   Trading   │      Data       │  │
│  │     Page    │    Page     │    Page     │      Page       │  │
│  └─────────────┴─────────────┴─────────────┴─────────────────┘  │
│  │                  System Info Page                         │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ /api/* proxy
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Rust Gateway (Axum)                            │
│               http://localhost:8080                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 Middleware Stack                            ││
│  │  • CORS • Rate Limiting • Tracing • Timing • Idempotency   ││
│  │  • Compression • Timeout • Error Handling                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   Core Handlers                             ││
│  │                                                             ││
│  │  ┌─────────────┬─────────────┬─────────────┬─────────────┐  ││
│  │  │ execution_  │ markets_    │ analytics_  │ audit_mvp   │  ││
│  │  │ mvp         │ mvp         │ mvp         │             │  ││
│  │  │ (Trading)   │ (Data)      │ (Analysis)  │ (Audit)     │  ││
│  │  └─────────────┴─────────────┴─────────────┴─────────────┘  ││
│  │  ┌─────────────┬─────────────┬─────────────┬─────────────┐  ││
│  │  │ risk_mvp    │ reports_mvp │ config      │ websocket   │  ││
│  │  │ (Risk)      │ (Reports)   │ (Config)    │ (WS)        │  ││
│  │  └─────────────┴─────────────┴─────────────┴─────────────┘  ││
│  │  ┌─────────────┬─────────────┬─────────────┬─────────────┐  ││
│  │  │ strategy    │ health      │ sandbox     │ backtest    │  ││
│  │  │ (Strategy)  │ (Health)    │ (Trading)   │ (Backtest)  │  ││
│  │  └─────────────┴─────────────┴─────────────┴─────────────┘  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  Internal Systems                           ││
│  │                                                             ││
│  │  ┌────────────────┬────────────────┬────────────────────────┐││
│  │  │ SandboxTrading │ CacheBackend   │ WebSocketManager       │││
│  │  │ Engine         │ (Memory/Redis) │                        │││
│  │  └────────────────┴────────────────┴────────────────────────┘││
│  │  ┌────────────────┬────────────────┬────────────────────────┐││
│  │  │ MetricsCollector│IdempotencyMgr │ MarketDataClient       │││
│  │  │ (Observability)│                │                        │││
│  │  └────────────────┴────────────────┴────────────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ gRPC (Future)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              External Services (Future/Optional)               │
│                                                                 │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────┐  │
│  │ Execution   │ Markets     │ Analytics   │ Other Services  │  │
│  │ Service     │ Service     │ Service     │                 │  │
│  └─────────────┴─────────────┴─────────────┴─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Frontend Architecture
- **Technology**: React 18 + TypeScript + Vite
- **Pages**: Dashboard, Services, Trading, Data, System Info (5 core pages)
- **Communication**: Direct HTTP calls to gateway, no authentication layer
- **Proxy**: `/api/*` routes proxied to gateway at `http://localhost:8080`

### 2. Rust Gateway (Core Backend)
- **Framework**: Axum web framework with tokio async runtime
- **Architecture**: Single service handling all business domains
- **Port**: `8080` (configurable via `GATEWAY_PORT`)
- **Responsibilities**: 
  - API routing and request handling
  - Business logic execution
  - External service coordination
  - Observability and monitoring

### 3. Internal Systems

#### SandboxTradingEngine
- **Purpose**: MVP trading execution without external dependencies
- **Location**: `rust-services/services/gateway/src/sandbox.rs`
- **Features**: In-memory order management, position tracking, balance management

#### CacheBackend
- **Purpose**: High-performance caching with Redis fallback to memory
- **Location**: `rust-services/services/gateway/src/cache.rs`
- **Features**: Batch operations, TTL support, pipeline operations

#### WebSocketManager
- **Purpose**: Real-time communication for market data and order updates
- **Location**: `rust-services/services/gateway/src/handlers/websocket.rs`
- **Protocol**: WebSocket with JSON message format

## API Architecture

### Unified Response Format
All APIs follow consistent response structure:

```json
{
  "status": "success" | "error",
  "data": "any_type",
  "message": "optional_description",
  "timestamp": "2024-01-01T00:00:00Z",
  "trace_id": "uuid",
  "execution_time_ms": 12
}
```

### Domain Organization
```
/api/v1/
├── orders/                 # Trading execution
├── positions/              # Position management  
├── balances/               # Balance queries
├── exchanges/              # Market data
├── symbols/                # Symbol management
├── markets/{symbol}/       # Market-specific data
├── strategies/             # Strategy management
├── config/                 # Configuration
├── alerts/                 # Alert management
├── risk/                   # Risk management
├── analytics/              # Analysis tasks
├── audit/                  # Audit logging
├── reports/                # Report generation
├── backtests/              # Backtesting
├── factors/                # Factor calculation
└── ws                      # WebSocket endpoint
```

### Cross-cutting Endpoints
```
/health                     # Health check
/healthz                    # Kubernetes health
/readyz                     # Readiness probe
/livez                      # Liveness probe
/api/v1/system/info         # System information
```

## Data Flow Architecture

### Request Flow
```
Frontend Request → Gateway Router → Middleware Stack → Handler → Internal System → Response
```

### Middleware Stack (Order of Execution)
1. **CORS**: Cross-origin resource sharing
2. **Rate Limiting**: Token bucket per IP
3. **Tracing**: Request ID generation and tracking
4. **Timing**: Execution time measurement
5. **Idempotency**: Write operation deduplication
6. **Compression**: Response compression
7. **Timeout**: Request timeout handling
8. **Error Handling**: Structured error responses

### Internal Data Flow
```
Handler → Validation → Business Logic → Cache Check → External Service (if needed) → Response
```

## Security Architecture (MVP - Minimal)

### Authentication: DISABLED
- No JWT tokens required
- No API keys needed
- Direct frontend access to all endpoints
- Environment variable: `DISABLE_AUTH=1`

### Rate Limiting
- **Algorithm**: Token bucket per client IP
- **Default**: 20 requests per second per IP
- **Configuration**: `RATE_LIMIT_RPS` environment variable
- **Response**: 429 Too Many Requests when exceeded

### Request Security
- **CORS**: Configured for development (permissive)
- **Headers**: Standard security headers applied
- **Validation**: Input validation on all endpoints
- **Sanitization**: SQL injection and XSS prevention

## Observability Architecture

### Logging
- **Framework**: `tracing` with structured JSON output
- **Levels**: ERROR, WARN, INFO, DEBUG, TRACE
- **Context**: Request ID propagation through all operations
- **Location**: `logs/gateway.log.{date}`

### Metrics
- **System**: HTTP request metrics (count, duration, status)
- **Domain**: Business metrics per domain (execution, markets, etc.)
- **Custom**: Application-specific metrics via MetricsCollector
- **Export**: Prometheus format (future endpoint: `/metrics`)

### Tracing
- **Implementation**: Distributed tracing with request IDs
- **Headers**: `x-trace-id` injection and response echo
- **Spans**: Request lifecycle tracking
- **Integration**: Ready for OpenTelemetry integration

## Storage Architecture

### Cache Layer
- **Primary**: Redis (when available)
- **Fallback**: In-memory HashMap
- **Features**: TTL support, batch operations, pipelining
- **Use Cases**: Idempotency keys, market data caching, session storage

### Sandbox Storage
- **Type**: In-memory data structures
- **Scope**: Trading orders, positions, balances
- **Persistence**: None (resets on restart)
- **Purpose**: MVP demonstration and development

### Future Database
- **Planned**: PostgreSQL for persistent data
- **Schema**: Trading data, audit logs, user preferences
- **Migration**: SQLx migration system ready

## Network Architecture

### Port Allocation
- **Frontend**: 5173 (Vite dev server)
- **Gateway**: 8080 (Rust Axum server)
- **Future Services**: 9100-9199 range reserved

### Protocol Support
- **HTTP/1.1**: Primary protocol for REST APIs
- **WebSocket**: Real-time communication support
- **gRPC**: Ready for external service integration (future)

### Load Balancing (Future)
- **Strategy**: Round-robin for horizontal scaling
- **Health Checks**: Integration with `/health` endpoints
- **Session Affinity**: WebSocket connection stickiness

## Deployment Architecture

### Development Environment
```bash
# Start all services
./start_local.sh

# Or individually:
cd rust-services && cargo run -p gateway-service
cd frontend && npm run dev
```

### Environment Variables
```bash
# Core Settings
DISABLE_AUTH=1                    # MVP: No authentication
SANDBOX_TRADING=1                 # Enable sandbox mode
PERSONAL_API_KEY_ENABLED=0        # Disable API keys

# Network
GATEWAY_PORT=8080                 # Gateway port
VITE_GATEWAY_URL=http://localhost:8080

# Performance
RATE_LIMIT_RPS=20                # Rate limiting
HTTP_REQUEST_TIMEOUT_SECS=30     # HTTP timeout
HTTP_POOL_MAX_IDLE_PER_HOST=10   # Connection pooling

# Observability  
RUST_LOG=debug                   # Logging level
```

### Production Considerations
- **Reverse Proxy**: Nginx for static file serving and SSL termination
- **Database**: PostgreSQL for persistent storage
- **Cache**: Redis cluster for distributed caching
- **Monitoring**: Prometheus + Grafana for metrics
- **Logging**: Centralized log aggregation (ELK stack)

## Integration Patterns

### External Service Integration
```rust
// Pattern: Graceful degradation
match external_service_call().await {
    Ok(response) => Ok(response),
    Err(_) => Ok(placeholder_response()), // Return placeholder on failure
}
```

### Frontend Integration
```typescript
// Unified HTTP client with automatic headers
const http = async (path: string, options: RequestInit = {}) => {
  const headers = new Headers(options.headers || {});
  headers.set('Content-Type', 'application/json');
  
  // Auto-inject idempotency key for write operations
  if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(options.method?.toUpperCase())) {
    headers.set('idempotency-key', crypto.randomUUID());
  }
  
  const response = await fetch(`${GATEWAY_URL}${path}`, {
    ...options,
    headers,
  });
  
  return response.json();
};
```

## Performance Architecture

### Concurrency Model
- **Runtime**: Tokio async runtime with work-stealing scheduler
- **Connection Pool**: HTTP client with connection pooling
- **Database**: SQLx with async connection pooling (future)
- **Cache**: Async Redis client with connection multiplexing

### Optimization Strategies
- **Response Caching**: Aggressive caching of market data and static content
- **Request Batching**: Batch operations for multiple items
- **Connection Reuse**: HTTP/1.1 keep-alive and HTTP/2 multiplexing
- **Lazy Loading**: On-demand initialization of expensive resources

### Scalability Considerations
- **Horizontal**: Multiple gateway instances behind load balancer
- **Vertical**: Efficient async I/O with minimal thread overhead
- **Caching**: Distributed caching with Redis cluster
- **Database**: Read replicas and connection pooling

## Error Handling Architecture

### Error Classification
1. **Client Errors (4xx)**: Invalid requests, validation failures
2. **Server Errors (5xx)**: Internal failures, external service timeouts
3. **Business Errors**: Domain-specific error conditions
4. **Infrastructure Errors**: Database, cache, network failures

### Error Response Format
```json
{
  "status": "error",
  "data": null,
  "message": "Human-readable error description",
  "timestamp": "2024-01-01T00:00:00Z",
  "trace_id": "uuid",
  "execution_time_ms": 5
}
```

### Error Recovery Strategies
- **Circuit Breaker**: Fail fast on repeated external service failures
- **Retry Logic**: Exponential backoff for transient failures
- **Fallback**: Graceful degradation to cached or placeholder data
- **Error Propagation**: Structured error context through call stack

## Testing Architecture

### Unit Testing
- **Framework**: Rust built-in test framework
- **Coverage**: Handler functions, business logic, utilities
- **Mocking**: Mock external services for isolated testing

### Integration Testing
- **Scope**: End-to-end API testing
- **Database**: Test database for integration scenarios
- **External Services**: Mock services for external dependencies

### Performance Testing
- **Load Testing**: Concurrent request handling
- **Stress Testing**: Resource exhaustion scenarios
- **Endurance Testing**: Long-running stability validation

## Documentation Architecture

### API Documentation
- **Format**: OpenAPI 3.0 specification
- **Generation**: Automated from code annotations
- **Interactive**: Swagger UI for API exploration

### Code Documentation
- **Rust**: Standard rustdoc documentation
- **TypeScript**: TSDoc for frontend components
- **Architecture**: Living documentation in markdown

## Future Evolution Path

### Phase 1: MVP Completion (Current)
- ✅ Core gateway functionality
- ✅ Sandbox trading system
- ✅ Basic frontend integration
- ✅ Health and monitoring

### Phase 2: External Service Integration
- gRPC client integration
- Database persistence
- Advanced caching strategies
- Enhanced error handling

### Phase 3: Production Readiness
- Authentication and authorization
- Advanced security features
- Performance optimization
- Comprehensive monitoring

### Phase 4: Advanced Features
- Real-time analytics
- Machine learning integration
- Advanced trading algorithms
- Multi-tenant support

## Implementation Guidelines

### Code Organization
```
rust-services/
├── services/gateway/src/
│   ├── handlers/           # API request handlers
│   ├── middleware/         # Cross-cutting concerns  
│   ├── models/             # Data structures
│   ├── services/           # Business logic
│   └── utils/              # Utility functions
├── shared/                 # Shared libraries
│   ├── logging/            # Structured logging
│   ├── cache/              # Cache abstraction
│   └── contracts/          # API contracts
└── tests/                  # Integration tests
```

### Development Workflow
1. **Feature Development**: Create feature branch
2. **Implementation**: Follow MVP principles, avoid over-engineering
3. **Testing**: Unit tests + integration tests
4. **Documentation**: Update API specs and architecture docs
5. **Review**: Code review focusing on MVP compliance
6. **Deployment**: Local testing, then production deployment

### Quality Gates
- **Compilation**: Zero compilation errors or warnings
- **Testing**: All tests passing with adequate coverage
- **Documentation**: API changes documented
- **Performance**: Response time within acceptable limits
- **Security**: Security scan passing (when enabled)

## Conclusion

This architecture design provides a solid foundation for the Sprint 1 MVP while maintaining flexibility for future evolution. The Rust-first approach ensures performance and reliability, while the unified gateway pattern simplifies frontend integration and reduces operational complexity.

The design prioritizes:
- **Simplicity**: Single service architecture reduces deployment complexity
- **Performance**: Async Rust with efficient resource utilization  
- **Observability**: Built-in logging, tracing, and metrics
- **Flexibility**: Extensible design for future service integration
- **Developer Experience**: Clear APIs and comprehensive documentation

This architecture successfully addresses the Sprint 1 requirements while establishing a foundation for the platform's future growth and evolution.