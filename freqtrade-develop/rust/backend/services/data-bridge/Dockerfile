# Data Bridge Dockerfile
FROM rust:1.88-slim as builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY .cargo/ .cargo/
COPY shared/ ./shared/
COPY services/data-bridge/ ./services/data-bridge/

WORKDIR /app/services/data-bridge
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/services/data-bridge/target/release/data-bridge /usr/local/bin/data-bridge
ENV DATA_DIR=/data
ENV NATS_URL=nats://nats:4222
VOLUME ["/data"]
CMD ["data-bridge"]

