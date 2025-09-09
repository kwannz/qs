# Risk Service Dockerfile
FROM rust:1.88-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy workspace files and members needed by Cargo workspace
COPY Cargo.toml Cargo.lock ./
COPY .cargo/ .cargo/
COPY shared/ ./shared/
COPY pkg/ ./pkg/
COPY services/risk/ ./services/risk/
COPY services/gateway-simple/ ./services/gateway-simple/
COPY services/strategy/ ./services/strategy/

# Build application
WORKDIR /app/services/risk
RUN cargo build --release

# Runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -r -u 1000 risk

# Copy binary
COPY --from=builder /app/target/release/risk-service /usr/local/bin/risk-service

# Set permissions
RUN chown risk:risk /usr/local/bin/risk-service
USER risk

# Expose port
EXPOSE 8083

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8083/health || exit 1

# Start application
CMD ["risk-service"]
