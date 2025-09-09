-- Migration: 004_cleanup_public_trading_tables.sql
-- 目的：在已标准化到 trading schema 后，清理由于历史原因残留在 public schema 下的交易表。
-- 策略：
--   1) 如 trading.* 与 public.* 同时存在且 trading.* 为空而 public.* 有数据，则先搬迁数据到 trading.*。
--   2) 删除 public.* 上的更新时间触发器（若存在）。
--   3) 仅在 public.* 表为空时删除该表（保守策略，避免误删仍有数据的旧表）。
--   4) 删除 public.update_updated_at_column() 函数（若存在）。

DO $$
DECLARE
    pub_cnt bigint;
    trg_cnt bigint;
BEGIN
    -- orders
    IF to_regclass('public.orders') IS NOT NULL AND to_regclass('trading.orders') IS NOT NULL THEN
        SELECT COUNT(*) INTO pub_cnt FROM public.orders;
        SELECT COUNT(*) INTO trg_cnt FROM trading.orders;
        IF trg_cnt = 0 AND pub_cnt > 0 THEN
            INSERT INTO trading.orders (
                id, client_order_id, exchange, symbol, side, order_type, algorithm_type,
                quantity, price, stop_price, filled_quantity, remaining_quantity,
                algorithm_params, status, time_in_force, exchange_order_id, exchange_status,
                average_fill_price, commission, commission_asset,
                created_at, updated_at, expires_at
            )
            SELECT 
                id, client_order_id, exchange, symbol, side, order_type, algorithm_type,
                quantity, price, stop_price, filled_quantity, remaining_quantity,
                algorithm_params, status, time_in_force, exchange_order_id, exchange_status,
                average_fill_price, commission, commission_asset,
                created_at, updated_at, expires_at
            FROM public.orders;
        END IF;

        -- drop trigger if exists on public.orders
        IF EXISTS (
            SELECT 1 FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE t.tgname = 'update_orders_updated_at' AND c.relname = 'orders' AND n.nspname = 'public'
        ) THEN
            EXECUTE 'DROP TRIGGER IF EXISTS update_orders_updated_at ON public.orders';
        END IF;

        SELECT COUNT(*) INTO pub_cnt FROM public.orders;
        IF pub_cnt = 0 THEN
            DROP TABLE public.orders CASCADE;
        END IF;
    END IF;

    -- positions
    IF to_regclass('public.positions') IS NOT NULL AND to_regclass('trading.positions') IS NOT NULL THEN
        SELECT COUNT(*) INTO pub_cnt FROM public.positions;
        SELECT COUNT(*) INTO trg_cnt FROM trading.positions;
        IF trg_cnt = 0 AND pub_cnt > 0 THEN
            INSERT INTO trading.positions (
                id, exchange, symbol, position_side, quantity, entry_price, mark_price,
                unrealized_pnl, realized_pnl, margin_type, initial_margin, maintenance_margin,
                leverage, notional_value, created_at, updated_at
            )
            SELECT 
                id, exchange, symbol, position_side, quantity, entry_price, mark_price,
                unrealized_pnl, realized_pnl, margin_type, initial_margin, maintenance_margin,
                leverage, notional_value, created_at, updated_at
            FROM public.positions;
        END IF;

        IF EXISTS (
            SELECT 1 FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE t.tgname = 'update_positions_updated_at' AND c.relname = 'positions' AND n.nspname = 'public'
        ) THEN
            EXECUTE 'DROP TRIGGER IF EXISTS update_positions_updated_at ON public.positions';
        END IF;

        SELECT COUNT(*) INTO pub_cnt FROM public.positions;
        IF pub_cnt = 0 THEN
            DROP TABLE public.positions CASCADE;
        END IF;
    END IF;

    -- balances
    IF to_regclass('public.balances') IS NOT NULL AND to_regclass('trading.balances') IS NOT NULL THEN
        SELECT COUNT(*) INTO pub_cnt FROM public.balances;
        SELECT COUNT(*) INTO trg_cnt FROM trading.balances;
        IF trg_cnt = 0 AND pub_cnt > 0 THEN
            INSERT INTO trading.balances (
                id, exchange, asset, account_type, free, locked, total,
                wallet_balance, unrealized_pnl, margin_balance,
                created_at, updated_at
            )
            SELECT 
                id, exchange, asset, account_type, free, locked, total,
                wallet_balance, unrealized_pnl, margin_balance,
                created_at, updated_at
            FROM public.balances;
        END IF;

        IF EXISTS (
            SELECT 1 FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE t.tgname = 'update_balances_updated_at' AND c.relname = 'balances' AND n.nspname = 'public'
        ) THEN
            EXECUTE 'DROP TRIGGER IF EXISTS update_balances_updated_at ON public.balances';
        END IF;

        SELECT COUNT(*) INTO pub_cnt FROM public.balances;
        IF pub_cnt = 0 THEN
            DROP TABLE public.balances CASCADE;
        END IF;
    END IF;

    -- trades (注意外键依赖 orders)
    IF to_regclass('public.trades') IS NOT NULL AND to_regclass('trading.trades') IS NOT NULL THEN
        SELECT COUNT(*) INTO pub_cnt FROM public.trades;
        SELECT COUNT(*) INTO trg_cnt FROM trading.trades;
        IF trg_cnt = 0 AND pub_cnt > 0 THEN
            INSERT INTO trading.trades (
                id, order_id, exchange_trade_id, exchange, symbol, side, quantity, price,
                notional, commission, commission_asset, is_maker, trade_type, executed_at, created_at
            )
            SELECT 
                id, order_id, exchange_trade_id, exchange, symbol, side, quantity, price,
                notional, commission, commission_asset, is_maker, trade_type, executed_at, created_at
            FROM public.trades;
        END IF;

        SELECT COUNT(*) INTO pub_cnt FROM public.trades;
        IF pub_cnt = 0 THEN
            DROP TABLE public.trades CASCADE;
        END IF;
    END IF;

    -- 删除 public 下的本地更新时间函数（若存在）
    IF to_regproc('public.update_updated_at_column()') IS NOT NULL THEN
        DROP FUNCTION public.update_updated_at_column();
    END IF;
END $$;

