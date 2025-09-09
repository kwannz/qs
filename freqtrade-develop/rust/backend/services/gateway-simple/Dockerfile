# Gateway Service Dockerfile
FROM rust:1.88-slim as builder

# 安装编译依赖
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 复制工作空间文件与成员（满足 Cargo workspace）
COPY Cargo.toml Cargo.lock ./
COPY .cargo/ .cargo/
COPY shared/ ./shared/
COPY pkg/ ./pkg/
COPY services/gateway-simple/ ./services/gateway-simple/
COPY services/risk/ ./services/risk/
COPY services/strategy/ ./services/strategy/

# 构建应用
WORKDIR /app/services/gateway-simple
RUN cargo build --release

# 运行时镜像
FROM debian:bookworm-slim

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 创建应用用户
RUN useradd -r -u 1000 gateway

# 复制二进制文件
COPY --from=builder /app/target/release/gateway-simple /usr/local/bin/gateway-simple

# 设置权限
RUN chown gateway:gateway /usr/local/bin/gateway-simple
USER gateway

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 启动应用
CMD ["gateway-simple"]
