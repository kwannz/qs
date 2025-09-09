-- 统一PostgreSQL Schema初始化
-- 替代分散的migration文件，创建按功能域隔离的schema结构

-- 创建基础schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS market;
CREATE SCHEMA IF NOT EXISTS analytics; 
CREATE SCHEMA IF NOT EXISTS admin;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS shared;

-- 创建用户和权限 (PostgreSQL compatible)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'trading_user') THEN
        CREATE USER trading_user WITH PASSWORD 'secure_password';
    END IF;
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'trading_reader') THEN
        CREATE USER trading_reader WITH PASSWORD 'reader_password';
    END IF;
END
$$;

-- 授权
GRANT USAGE ON SCHEMA trading, market, analytics, admin, monitoring, shared TO trading_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading, market, analytics, admin, monitoring, shared TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading, market, analytics, admin, monitoring, shared TO trading_user;

-- 只读用户权限
GRANT USAGE ON SCHEMA trading, market, analytics, admin, monitoring, shared TO trading_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA trading, market, analytics, admin, monitoring, shared TO trading_reader;

-- 设置默认权限
ALTER DEFAULT PRIVILEGES IN SCHEMA trading, market, analytics, admin, monitoring, shared 
GRANT ALL ON TABLES TO trading_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA trading, market, analytics, admin, monitoring, shared 
GRANT SELECT ON TABLES TO trading_reader;

-- 共享基础表
CREATE TABLE IF NOT EXISTS shared.system_config (
    id SERIAL PRIMARY KEY,
    key VARCHAR(255) NOT NULL UNIQUE,
    value TEXT,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS shared.audit_log (
    id BIGSERIAL PRIMARY KEY,
    table_schema VARCHAR(63) NOT NULL,
    table_name VARCHAR(63) NOT NULL,
    operation VARCHAR(10) NOT NULL,
    user_name VARCHAR(63) DEFAULT CURRENT_USER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    old_values JSONB,
    new_values JSONB
);

-- Trading schema 核心表
CREATE TABLE IF NOT EXISTS trading.orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    type VARCHAR(20) NOT NULL CHECK (type IN ('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT')),
    quantity DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8),
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trading.executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID NOT NULL REFERENCES trading.orders(id),
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    commission DECIMAL(20,8) DEFAULT 0,
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Market schema 表
CREATE TABLE IF NOT EXISTS market.klines (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    interval VARCHAR(10) NOT NULL,
    open_price DECIMAL(20,8) NOT NULL,
    high_price DECIMAL(20,8) NOT NULL,
    low_price DECIMAL(20,8) NOT NULL,
    close_price DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    UNIQUE(symbol, interval, timestamp)
);

CREATE TABLE IF NOT EXISTS market.tickers (
    symbol VARCHAR(50) PRIMARY KEY,
    price DECIMAL(20,8) NOT NULL,
    change_24h DECIMAL(10,4),
    volume_24h DECIMAL(20,8),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Analytics schema 表
CREATE TABLE IF NOT EXISTS analytics.backtests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_balance DECIMAL(20,8) NOT NULL,
    final_balance DECIMAL(20,8),
    total_return DECIMAL(10,4),
    max_drawdown DECIMAL(10,4),
    sharpe_ratio DECIMAL(10,4),
    total_trades INTEGER,
    win_rate DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analytics.factors (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    factor_name VARCHAR(100) NOT NULL,
    value DECIMAL(20,8) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    UNIQUE(symbol, factor_name, timestamp)
);

-- Admin schema 表
CREATE TABLE IF NOT EXISTS admin.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS admin.sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES admin.users(id),
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Monitoring schema 表
CREATE TABLE IF NOT EXISTS monitoring.health_checks (
    id BIGSERIAL PRIMARY KEY,
    service_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL,
    response_time_ms INTEGER,
    error_message TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS monitoring.alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_name VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'ACTIVE',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_orders_symbol_status ON trading.orders(symbol, status);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON trading.orders(created_at);
CREATE INDEX IF NOT EXISTS idx_executions_order_id ON trading.executions(order_id);
CREATE INDEX IF NOT EXISTS idx_klines_symbol_interval_timestamp ON market.klines(symbol, interval, timestamp);
CREATE INDEX IF NOT EXISTS idx_factors_symbol_timestamp ON analytics.factors(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_health_checks_service_timestamp ON monitoring.health_checks(service_name, timestamp);

-- 创建更新timestamp的触发器函数
CREATE OR REPLACE FUNCTION shared.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 应用触发器到需要的表
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger t
        JOIN pg_class c ON c.oid = t.tgrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE t.tgname = 'update_orders_updated_at'
          AND c.relname = 'orders'
          AND n.nspname = 'trading'
    ) THEN
        CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON trading.orders
            FOR EACH ROW EXECUTE FUNCTION shared.update_updated_at_column();
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger t
        JOIN pg_class c ON c.oid = t.tgrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE t.tgname = 'update_users_updated_at'
          AND c.relname = 'users'
          AND n.nspname = 'admin'
    ) THEN
        CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON admin.users
            FOR EACH ROW EXECUTE FUNCTION shared.update_updated_at_column();
    END IF;
END
$$;

-- 插入基础配置数据
INSERT INTO shared.system_config (key, value, description) VALUES
    ('system_name', 'Trading Platform', '系统名称'),
    ('version', '1.0.0', '系统版本'),
    ('timezone', 'UTC', '系统时区'),
    ('max_orders_per_second', '100', '每秒最大订单数'),
    ('maintenance_mode', 'false', '维护模式')
ON CONFLICT (key) DO NOTHING;
