-- Migration: 003_normalize_trading_schema.sql
-- 目标：将核心交易对象统一到 trading schema，提供从 public 的兼容迁移，并创建幂等触发器与视图

-- 兼容迁移：如 public 下存在旧表而 trading 下不存在，则迁移到 trading
DO $$
BEGIN
    IF to_regclass('public.orders') IS NOT NULL AND to_regclass('trading.orders') IS NULL THEN
        ALTER TABLE public.orders SET SCHEMA trading;
    END IF;
    IF to_regclass('public.positions') IS NOT NULL AND to_regclass('trading.positions') IS NULL THEN
        ALTER TABLE public.positions SET SCHEMA trading;
    END IF;
    IF to_regclass('public.balances') IS NOT NULL AND to_regclass('trading.balances') IS NULL THEN
        ALTER TABLE public.balances SET SCHEMA trading;
    END IF;
    IF to_regclass('public.trades') IS NOT NULL AND to_regclass('trading.trades') IS NULL THEN
        ALTER TABLE public.trades SET SCHEMA trading;
    END IF;
END
$$;

-- 统一索引（如已在 trading 上存在则忽略）
CREATE INDEX IF NOT EXISTS idx_orders_status_created ON trading.orders(status, created_at);
CREATE INDEX IF NOT EXISTS idx_orders_exchange_symbol ON trading.orders(exchange, symbol);
CREATE INDEX IF NOT EXISTS idx_orders_algorithm_type ON trading.orders(algorithm_type) WHERE algorithm_type IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_orders_client_order_id ON trading.orders(client_order_id);

CREATE INDEX IF NOT EXISTS idx_positions_exchange ON trading.positions(exchange);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON trading.positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_updated ON trading.positions(updated_at);

CREATE INDEX IF NOT EXISTS idx_balances_exchange_asset ON trading.balances(exchange, asset);
CREATE INDEX IF NOT EXISTS idx_balances_account_type ON trading.balances(account_type);

CREATE INDEX IF NOT EXISTS idx_trades_order_id ON trading.trades(order_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trading.trades(symbol, executed_at);
CREATE INDEX IF NOT EXISTS idx_trades_exchange_symbol ON trading.trades(exchange, symbol);

-- 删除 002 中可能创建的本地函数，避免与 shared 下的函数重复
DO $$ BEGIN
    IF to_regproc('update_updated_at_column()') IS NOT NULL THEN
        DROP FUNCTION update_updated_at_column();
    END IF;
END $$;

-- 在 trading.* 上创建幂等触发器，使用 shared.update_updated_at_column
DO $$
BEGIN
    IF to_regclass('trading.orders') IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM pg_trigger t
        JOIN pg_class c ON c.oid = t.tgrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE t.tgname = 'update_orders_updated_at'
          AND c.relname = 'orders' AND n.nspname = 'trading'
    ) THEN
        CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON trading.orders
            FOR EACH ROW EXECUTE FUNCTION shared.update_updated_at_column();
    END IF;

    IF to_regclass('trading.positions') IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM pg_trigger t
        JOIN pg_class c ON c.oid = t.tgrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE t.tgname = 'update_positions_updated_at'
          AND c.relname = 'positions' AND n.nspname = 'trading'
    ) THEN
        CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON trading.positions
            FOR EACH ROW EXECUTE FUNCTION shared.update_updated_at_column();
    END IF;

    IF to_regclass('trading.balances') IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM pg_trigger t
        JOIN pg_class c ON c.oid = t.tgrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE t.tgname = 'update_balances_updated_at'
          AND c.relname = 'balances' AND n.nspname = 'trading'
    ) THEN
        CREATE TRIGGER update_balances_updated_at BEFORE UPDATE ON trading.balances
            FOR EACH ROW EXECUTE FUNCTION shared.update_updated_at_column();
    END IF;
END
$$;

-- 规范化视图到 trading schema
CREATE OR REPLACE VIEW trading.active_orders AS
SELECT * FROM trading.orders 
WHERE status IN ('pending', 'new', 'partially_filled')
ORDER BY created_at;

CREATE OR REPLACE VIEW trading.open_positions AS
SELECT * FROM trading.positions 
WHERE ABS(quantity) > 0
ORDER BY notional_value DESC;

CREATE OR REPLACE VIEW trading.daily_trades_summary AS
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
FROM trading.trades
GROUP BY DATE(executed_at), symbol
ORDER BY trade_date DESC, total_notional DESC;

CREATE OR REPLACE VIEW trading.portfolio_overview AS
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
FROM trading.positions p
LEFT JOIN trading.orders o ON o.symbol = p.symbol AND o.exchange = p.exchange 
    AND o.status IN ('pending', 'new', 'partially_filled')
WHERE ABS(p.quantity) > 0
GROUP BY p.id, p.exchange, p.symbol, p.quantity, p.entry_price, p.mark_price, 
         p.unrealized_pnl, p.realized_pnl, p.notional_value, p.leverage, p.updated_at
ORDER BY p.notional_value DESC;

