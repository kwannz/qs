-- Migration: 001_create_trading_tables.sql
-- Sprint 5 Phase 1: Core Trading Tables
-- Created: 2025-01-20

-- ========================================
-- Trading Core Tables
-- ========================================

-- Orders table: 订单表
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_order_id VARCHAR(64) UNIQUE NOT NULL,
    exchange VARCHAR(32) NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type VARCHAR(16) NOT NULL CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit')),
    algorithm_type VARCHAR(16) CHECK (algorithm_type IN ('twap', 'vwap', 'pov', 'iceberg')),
    
    -- Quantities and Prices
    quantity DECIMAL(20,8) NOT NULL CHECK (quantity > 0),
    price DECIMAL(20,8),
    stop_price DECIMAL(20,8),
    filled_quantity DECIMAL(20,8) NOT NULL DEFAULT 0,
    remaining_quantity DECIMAL(20,8) NOT NULL DEFAULT 0,
    
    -- Algorithm Parameters (JSON for flexibility)
    algorithm_params JSONB,
    
    -- Status and Timing
    status VARCHAR(16) NOT NULL DEFAULT 'pending' CHECK (
        status IN ('pending', 'new', 'partially_filled', 'filled', 'canceled', 'rejected', 'expired')
    ),
    time_in_force VARCHAR(8) DEFAULT 'GTC' CHECK (time_in_force IN ('GTC', 'IOC', 'FOK', 'DAY')),
    
    -- Exchange specific
    exchange_order_id VARCHAR(64),
    exchange_status VARCHAR(32),
    
    -- Financial calculations
    average_fill_price DECIMAL(20,8),
    commission DECIMAL(20,8) DEFAULT 0,
    commission_asset VARCHAR(16),
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    
    -- Indexing
    CONSTRAINT orders_symbol_time_idx UNIQUE (symbol, created_at, id)
);

-- Positions table: 持仓表
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exchange VARCHAR(32) NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    position_side VARCHAR(8) NOT NULL DEFAULT 'both' CHECK (position_side IN ('long', 'short', 'both')),
    
    -- Position metrics
    quantity DECIMAL(20,8) NOT NULL DEFAULT 0,
    entry_price DECIMAL(20,8),
    mark_price DECIMAL(20,8),
    unrealized_pnl DECIMAL(20,8) DEFAULT 0,
    realized_pnl DECIMAL(20,8) DEFAULT 0,
    
    -- Margin info (for futures)
    margin_type VARCHAR(8) CHECK (margin_type IN ('cross', 'isolated')),
    initial_margin DECIMAL(20,8) DEFAULT 0,
    maintenance_margin DECIMAL(20,8) DEFAULT 0,
    
    -- Risk metrics
    leverage DECIMAL(8,2) DEFAULT 1.0,
    notional_value DECIMAL(20,8) DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Unique constraint per exchange-symbol-side
    UNIQUE(exchange, symbol, position_side)
);

-- Balances table: 余额表
CREATE TABLE IF NOT EXISTS balances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exchange VARCHAR(32) NOT NULL,
    asset VARCHAR(16) NOT NULL,
    account_type VARCHAR(16) NOT NULL DEFAULT 'spot' CHECK (account_type IN ('spot', 'futures', 'margin')),
    
    -- Balance breakdown
    free DECIMAL(20,8) NOT NULL DEFAULT 0,
    locked DECIMAL(20,8) NOT NULL DEFAULT 0,
    total DECIMAL(20,8) NOT NULL DEFAULT 0,
    
    -- For futures accounts
    wallet_balance DECIMAL(20,8),
    unrealized_pnl DECIMAL(20,8) DEFAULT 0,
    margin_balance DECIMAL(20,8),
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Unique constraint per exchange-asset-account
    UNIQUE(exchange, asset, account_type)
);

-- Trades table: 成交记录表
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID NOT NULL REFERENCES trading.orders(id) ON DELETE CASCADE,
    exchange_trade_id VARCHAR(64) NOT NULL,
    exchange VARCHAR(32) NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    
    -- Trade details
    side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity DECIMAL(20,8) NOT NULL CHECK (quantity > 0),
    price DECIMAL(20,8) NOT NULL CHECK (price > 0),
    notional DECIMAL(20,8) NOT NULL DEFAULT 0,
    
    -- Costs
    commission DECIMAL(20,8) NOT NULL DEFAULT 0,
    commission_asset VARCHAR(16),
    
    -- Trade classification
    is_maker BOOLEAN DEFAULT FALSE,
    trade_type VARCHAR(16) DEFAULT 'regular' CHECK (trade_type IN ('regular', 'liquidation', 'adl')),
    
    -- Timestamps
    executed_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Indexing for performance
    UNIQUE(exchange, exchange_trade_id)
);

-- ========================================
-- Indexes for Performance
-- ========================================

-- Orders indexes
CREATE INDEX IF NOT EXISTS idx_orders_status_created ON orders(status, created_at);
CREATE INDEX IF NOT EXISTS idx_orders_exchange_symbol ON orders(exchange, symbol);
CREATE INDEX IF NOT EXISTS idx_orders_algorithm_type ON orders(algorithm_type) WHERE algorithm_type IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_orders_client_order_id ON orders(client_order_id);

-- Positions indexes
CREATE INDEX IF NOT EXISTS idx_positions_exchange ON positions(exchange);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_updated ON positions(updated_at);

-- Balances indexes
CREATE INDEX IF NOT EXISTS idx_balances_exchange_asset ON balances(exchange, asset);
CREATE INDEX IF NOT EXISTS idx_balances_account_type ON balances(account_type);

-- Trades indexes
CREATE INDEX IF NOT EXISTS idx_trades_order_id ON trades(order_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, executed_at);
CREATE INDEX IF NOT EXISTS idx_trades_exchange_symbol ON trades(exchange, symbol);

-- ========================================
-- Triggers for Updated At
-- ========================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers
CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_balances_updated_at BEFORE UPDATE ON balances 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ========================================
-- Views for Common Queries
-- ========================================

-- Active orders view
CREATE OR REPLACE VIEW active_orders AS
SELECT * FROM orders 
WHERE status IN ('pending', 'new', 'partially_filled')
ORDER BY created_at;

-- Open positions view
CREATE OR REPLACE VIEW open_positions AS
SELECT * FROM positions 
WHERE ABS(quantity) > 0
ORDER BY notional_value DESC;

-- Daily trades summary view
CREATE OR REPLACE VIEW daily_trades_summary AS
SELECT 
    DATE(executed_at) as trade_date,
    symbol,
    COUNT(*) as trade_count,
    SUM(quantity) as total_quantity,
    SUM(notional) as total_notional,
    SUM(commission) as total_commission,
    AVG(price) as avg_price,
    MIN(executed_at) as first_trade,
    MAX(executed_at) as last_trade
FROM trades
GROUP BY DATE(executed_at), symbol
ORDER BY trade_date DESC, total_notional DESC;

-- Portfolio overview view
CREATE OR REPLACE VIEW portfolio_overview AS
SELECT 
    p.exchange,
    p.symbol,
    p.quantity,
    p.entry_price,
    p.mark_price,
    p.unrealized_pnl,
    p.realized_pnl,
    p.notional_value,
    p.leverage,
    COUNT(o.id) as active_orders_count,
    p.updated_at
FROM positions p
LEFT JOIN orders o ON o.symbol = p.symbol AND o.exchange = p.exchange 
    AND o.status IN ('pending', 'new', 'partially_filled')
WHERE ABS(p.quantity) > 0
GROUP BY p.id, p.exchange, p.symbol, p.quantity, p.entry_price, p.mark_price, 
         p.unrealized_pnl, p.realized_pnl, p.notional_value, p.leverage, p.updated_at
ORDER BY p.notional_value DESC;
