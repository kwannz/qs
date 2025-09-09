# Multi-stage build for Strategy Service
FROM rust:1.88-bookworm as builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    protobuf-compiler \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY .cargo/ .cargo/

# Copy all shared dependencies first for better caching
COPY shared/ shared/
COPY pkg/ pkg/

# Copy workspace members needed by Cargo workspace
COPY services/strategy/ services/strategy/
COPY services/gateway-simple/ services/gateway-simple/
COPY services/risk/ services/risk/
COPY pkg/ pkg/
COPY migrations/ migrations/

# Set SQLX to offline mode to avoid requiring database during build
ENV SQLX_OFFLINE=true

# Build the strategy service with optimizations
RUN --mount=type=cache,target=/app/target \
    --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo build --release --package strategy-service && \
    cp target/release/strategy-service /usr/local/bin/strategy-service

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies (include curl for HEALTHCHECK)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# Create non-root user for security
RUN useradd -r -s /bin/false -m -d /app strategy

# Copy the compiled binary
COPY --from=builder /usr/local/bin/strategy-service /usr/local/bin/strategy-service

# Set ownership and permissions
RUN chown strategy:strategy /usr/local/bin/strategy-service && \
    chmod +x /usr/local/bin/strategy-service

# Switch to non-root user
USER strategy
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9200/health || exit 1

# Default port for Strategy Service
EXPOSE 9200

# Run the strategy service
CMD ["strategy-service"]

# Build metadata
LABEL maintainer="Trading Platform Team"
LABEL service="strategy"
LABEL version="1.0.0"
LABEL description="Strategy Execution Microservice with AI Integration"
