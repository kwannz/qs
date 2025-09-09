use sqlx::{Pool, Postgres, Row};
use uuid::Uuid;
use anyhow::Result;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

use super::Repository;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct StrategyExecution {
    pub id: Uuid,
    pub instance_id: Uuid,
    pub execution_id: String,
    pub exchange: String,
    pub symbol: String,
    pub execution_type: String,
    pub input_data: Option<serde_json::Value>,
    pub output_data: Option<serde_json::Value>,
    pub decisions: Option<serde_json::Value>,
    pub market_price: Option<Decimal>,
    pub volume_24h: Option<Decimal>,
    pub volatility: Option<Decimal>,
    pub status: String,
    pub orders_created: Option<i32>,
    pub orders_filled: Option<i32>,
    pub total_volume: Option<Decimal>,
    pub realized_pnl: Option<Decimal>,
    pub execution_time_ms: Option<i32>,
    pub latency_ms: Option<i32>,
    pub slippage_bps: Option<i32>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
    pub retry_count: Option<i32>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

pub struct StrategyExecutionRepository {
    pool: Pool<Postgres>,
}

impl StrategyExecutionRepository {
    pub fn new(pool: Pool<Postgres>) -> Self {
        Self { pool }
    }

    pub async fn find_by_instance_id(&self, instance_id: Uuid, limit: Option<u32>, offset: Option<u32>) -> Result<Vec<StrategyExecution>> {
        let limit = limit.unwrap_or(50).min(100) as i64;
        let offset = offset.unwrap_or(0) as i64;

        let rows = sqlx::query(
            r#"
            SELECT id, instance_id, execution_id, exchange, symbol, execution_type,
                   input_data, output_data, decisions, market_price, volume_24h,
                   volatility, status, orders_created, orders_filled, total_volume,
                   realized_pnl, execution_time_ms, latency_ms, slippage_bps,
                   error_code, error_message, retry_count, created_at, started_at,
                   completed_at
            FROM strategy_executions 
            WHERE instance_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(instance_id)
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;

        let executions: Result<Vec<StrategyExecution>> = rows.into_iter().map(|row| {
            Ok(StrategyExecution {
                id: row.get("id"),
                instance_id: row.get("instance_id"),
                execution_id: row.get("execution_id"),
                exchange: row.get("exchange"),
                symbol: row.get("symbol"),
                execution_type: row.get("execution_type"),
                input_data: row.get("input_data"),
                output_data: row.get("output_data"),
                decisions: row.get("decisions"),
                market_price: row.get("market_price"),
                volume_24h: row.get("volume_24h"),
                volatility: row.get("volatility"),
                status: row.get("status"),
                orders_created: row.get("orders_created"),
                orders_filled: row.get("orders_filled"),
                total_volume: row.get("total_volume"),
                realized_pnl: row.get("realized_pnl"),
                execution_time_ms: row.get("execution_time_ms"),
                latency_ms: row.get("latency_ms"),
                slippage_bps: row.get("slippage_bps"),
                error_code: row.get("error_code"),
                error_message: row.get("error_message"),
                retry_count: row.get("retry_count"),
                created_at: row.get("created_at"),
                started_at: row.get("started_at"),
                completed_at: row.get("completed_at"),
            })
        }).collect();

        executions
    }

    pub async fn find_by_status(&self, status: &str, limit: Option<u32>, offset: Option<u32>) -> Result<Vec<StrategyExecution>> {
        let limit = limit.unwrap_or(50).min(100) as i64;
        let offset = offset.unwrap_or(0) as i64;

        let rows = sqlx::query(
            r#"
            SELECT id, instance_id, execution_id, exchange, symbol, execution_type,
                   input_data, output_data, decisions, market_price, volume_24h,
                   volatility, status, orders_created, orders_filled, total_volume,
                   realized_pnl, execution_time_ms, latency_ms, slippage_bps,
                   error_code, error_message, retry_count, created_at, started_at,
                   completed_at
            FROM strategy_executions 
            WHERE status = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(status)
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;

        let executions: Result<Vec<StrategyExecution>> = rows.into_iter().map(|row| {
            Ok(StrategyExecution {
                id: row.get("id"),
                instance_id: row.get("instance_id"),
                execution_id: row.get("execution_id"),
                exchange: row.get("exchange"),
                symbol: row.get("symbol"),
                execution_type: row.get("execution_type"),
                input_data: row.get("input_data"),
                output_data: row.get("output_data"),
                decisions: row.get("decisions"),
                market_price: row.get("market_price"),
                volume_24h: row.get("volume_24h"),
                volatility: row.get("volatility"),
                status: row.get("status"),
                orders_created: row.get("orders_created"),
                orders_filled: row.get("orders_filled"),
                total_volume: row.get("total_volume"),
                realized_pnl: row.get("realized_pnl"),
                execution_time_ms: row.get("execution_time_ms"),
                latency_ms: row.get("latency_ms"),
                slippage_bps: row.get("slippage_bps"),
                error_code: row.get("error_code"),
                error_message: row.get("error_message"),
                retry_count: row.get("retry_count"),
                created_at: row.get("created_at"),
                started_at: row.get("started_at"),
                completed_at: row.get("completed_at"),
            })
        }).collect();

        executions
    }

    pub async fn find_recent_by_symbol(&self, symbol: &str, hours: i32, limit: Option<u32>) -> Result<Vec<StrategyExecution>> {
        let limit = limit.unwrap_or(100).min(500) as i64;
        
        let hours_param = format!("{hours} hours");
        let rows = sqlx::query(
            r#"
            SELECT id, instance_id, execution_id, exchange, symbol, execution_type,
                   input_data, output_data, decisions, market_price, volume_24h,
                   volatility, status, orders_created, orders_filled, total_volume,
                   realized_pnl, execution_time_ms, latency_ms, slippage_bps,
                   error_code, error_message, retry_count, created_at, started_at,
                   completed_at
            FROM strategy_executions 
            WHERE symbol = $1 
              AND created_at >= NOW() - INTERVAL $2
            ORDER BY created_at DESC
            LIMIT $3
            "#
        )
        .bind(symbol)
        .bind(hours_param)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        let executions: Result<Vec<StrategyExecution>> = rows.into_iter().map(|row| {
            Ok(StrategyExecution {
                id: row.get("id"),
                instance_id: row.get("instance_id"),
                execution_id: row.get("execution_id"),
                exchange: row.get("exchange"),
                symbol: row.get("symbol"),
                execution_type: row.get("execution_type"),
                input_data: row.get("input_data"),
                output_data: row.get("output_data"),
                decisions: row.get("decisions"),
                market_price: row.get("market_price"),
                volume_24h: row.get("volume_24h"),
                volatility: row.get("volatility"),
                status: row.get("status"),
                orders_created: row.get("orders_created"),
                orders_filled: row.get("orders_filled"),
                total_volume: row.get("total_volume"),
                realized_pnl: row.get("realized_pnl"),
                execution_time_ms: row.get("execution_time_ms"),
                latency_ms: row.get("latency_ms"),
                slippage_bps: row.get("slippage_bps"),
                error_code: row.get("error_code"),
                error_message: row.get("error_message"),
                retry_count: row.get("retry_count"),
                created_at: row.get("created_at"),
                started_at: row.get("started_at"),
                completed_at: row.get("completed_at"),
            })
        }).collect();

        executions
    }

    pub async fn update_status(&self, id: Uuid, status: &str, completed_at: Option<DateTime<Utc>>) -> Result<()> {
        if let Some(completed) = completed_at {
            sqlx::query(
                r#"
                UPDATE strategy_executions 
                SET status = $2, completed_at = $3
                WHERE id = $1
                "#,
            )
            .bind(id)
            .bind(status)
            .bind(completed)
            .execute(&self.pool)
            .await?;
        } else {
            sqlx::query(
                r#"
                UPDATE strategy_executions 
                SET status = $2
                WHERE id = $1
                "#,
            )
            .bind(id)
            .bind(status)
            .execute(&self.pool)
            .await?;
        }

        Ok(())
    }

    pub async fn update_results(&self, id: Uuid, orders_created: i32, orders_filled: i32, total_volume: Decimal, realized_pnl: Decimal) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE strategy_executions 
            SET orders_created = $2, orders_filled = $3, total_volume = $4, realized_pnl = $5
            WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(orders_created)
        .bind(orders_filled)
        .bind(total_volume)
        .bind(realized_pnl)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn increment_retry(&self, id: Uuid, error_code: Option<&str>, error_message: Option<&str>) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE strategy_executions 
            SET retry_count = COALESCE(retry_count, 0) + 1,
                error_code = $2,
                error_message = $3
            WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(error_code)
        .bind(error_message)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_performance_metrics(&self, instance_id: Uuid, days: i32) -> Result<(i64, i64, Decimal, Decimal)> {
        let days_param = format!("{days} days");
        let row = sqlx::query(
            r#"
            SELECT 
                COUNT(*) as total_executions,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_executions,
                COALESCE(SUM(realized_pnl), 0) as total_pnl,
                COALESCE(AVG(execution_time_ms), 0) as avg_execution_time
            FROM strategy_executions 
            WHERE instance_id = $1 
              AND created_at >= NOW() - INTERVAL $2
            "#
        )
        .bind(instance_id)
        .bind(days_param)
        .fetch_one(&self.pool)
        .await?;

        Ok((
            row.get::<i64, _>("total_executions"),
            row.get::<i64, _>("successful_executions"),
            row.get::<Decimal, _>("total_pnl"),
            row.get::<Decimal, _>("avg_execution_time"),
        ))
    }
}

impl Repository<StrategyExecution> for StrategyExecutionRepository {
    async fn create(&self, execution: &StrategyExecution) -> Result<StrategyExecution> {
        sqlx::query(
            r#"
            INSERT INTO strategy_executions (
                id, instance_id, execution_id, exchange, symbol, execution_type,
                input_data, output_data, decisions, market_price, volume_24h,
                volatility, status, orders_created, orders_filled, total_volume,
                realized_pnl, execution_time_ms, latency_ms, slippage_bps,
                error_code, error_message, retry_count, created_at, started_at,
                completed_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26
            )
            "#,
        )
        .bind(execution.id)
        .bind(execution.instance_id)
        .bind(&execution.execution_id)
        .bind(&execution.exchange)
        .bind(&execution.symbol)
        .bind(&execution.execution_type)
        .bind(&execution.input_data)
        .bind(&execution.output_data)
        .bind(&execution.decisions)
        .bind(execution.market_price)
        .bind(execution.volume_24h)
        .bind(execution.volatility)
        .bind(&execution.status)
        .bind(execution.orders_created)
        .bind(execution.orders_filled)
        .bind(execution.total_volume)
        .bind(execution.realized_pnl)
        .bind(execution.execution_time_ms)
        .bind(execution.latency_ms)
        .bind(execution.slippage_bps)
        .bind(&execution.error_code)
        .bind(&execution.error_message)
        .bind(execution.retry_count)
        .bind(execution.created_at)
        .bind(execution.started_at)
        .bind(execution.completed_at)
        .execute(&self.pool)
        .await?;

        Ok(execution.clone())
    }

    async fn find_by_id(&self, id: Uuid) -> Result<Option<StrategyExecution>> {
        let row = sqlx::query(
            r#"
            SELECT id, instance_id, execution_id, exchange, symbol, execution_type,
                   input_data, output_data, decisions, market_price, volume_24h,
                   volatility, status, orders_created, orders_filled, total_volume,
                   realized_pnl, execution_time_ms, latency_ms, slippage_bps,
                   error_code, error_message, retry_count, created_at, started_at,
                   completed_at
            FROM strategy_executions 
            WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => Ok(Some(StrategyExecution {
                id: row.get("id"),
                instance_id: row.get("instance_id"),
                execution_id: row.get("execution_id"),
                exchange: row.get("exchange"),
                symbol: row.get("symbol"),
                execution_type: row.get("execution_type"),
                input_data: row.get("input_data"),
                output_data: row.get("output_data"),
                decisions: row.get("decisions"),
                market_price: row.get("market_price"),
                volume_24h: row.get("volume_24h"),
                volatility: row.get("volatility"),
                status: row.get("status"),
                orders_created: row.get("orders_created"),
                orders_filled: row.get("orders_filled"),
                total_volume: row.get("total_volume"),
                realized_pnl: row.get("realized_pnl"),
                execution_time_ms: row.get("execution_time_ms"),
                latency_ms: row.get("latency_ms"),
                slippage_bps: row.get("slippage_bps"),
                error_code: row.get("error_code"),
                error_message: row.get("error_message"),
                retry_count: row.get("retry_count"),
                created_at: row.get("created_at"),
                started_at: row.get("started_at"),
                completed_at: row.get("completed_at"),
            })),
            None => Ok(None),
        }
    }

    async fn update(&self, id: Uuid, execution: &StrategyExecution) -> Result<Option<StrategyExecution>> {
        let result = sqlx::query(
            r#"
            UPDATE strategy_executions SET 
                execution_id = $2, exchange = $3, symbol = $4, execution_type = $5,
                input_data = $6, output_data = $7, decisions = $8,
                market_price = $9, volume_24h = $10, volatility = $11,
                status = $12, orders_created = $13, orders_filled = $14,
                total_volume = $15, realized_pnl = $16, execution_time_ms = $17,
                latency_ms = $18, slippage_bps = $19, error_code = $20,
                error_message = $21, started_at = $22, completed_at = $23
            WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(&execution.execution_id)
        .bind(&execution.exchange)
        .bind(&execution.symbol)
        .bind(&execution.execution_type)
        .bind(&execution.input_data)
        .bind(&execution.output_data)
        .bind(&execution.decisions)
        .bind(execution.market_price)
        .bind(execution.volume_24h)
        .bind(execution.volatility)
        .bind(&execution.status)
        .bind(execution.orders_created)
        .bind(execution.orders_filled)
        .bind(execution.total_volume)
        .bind(execution.realized_pnl)
        .bind(execution.execution_time_ms)
        .bind(execution.latency_ms)
        .bind(execution.slippage_bps)
        .bind(&execution.error_code)
        .bind(&execution.error_message)
        .bind(execution.started_at)
        .bind(execution.completed_at)
        .execute(&self.pool)
        .await?;

        if result.rows_affected() > 0 {
            self.find_by_id(id).await
        } else {
            Ok(None)
        }
    }

    async fn delete(&self, id: Uuid) -> Result<bool> {
        let result = sqlx::query("DELETE FROM strategy_executions WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected() > 0)
    }

    async fn list(&self, limit: Option<u32>, offset: Option<u32>) -> Result<Vec<StrategyExecution>> {
        let limit = limit.unwrap_or(50).min(100) as i64;
        let offset = offset.unwrap_or(0) as i64;

        let rows = sqlx::query(
            r#"
            SELECT id, instance_id, execution_id, exchange, symbol, execution_type,
                   input_data, output_data, decisions, market_price, volume_24h,
                   volatility, status, orders_created, orders_filled, total_volume,
                   realized_pnl, execution_time_ms, latency_ms, slippage_bps,
                   error_code, error_message, retry_count, created_at, started_at,
                   completed_at
            FROM strategy_executions
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            "#,
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;

        let executions: Result<Vec<StrategyExecution>> = rows.into_iter().map(|row| {
            Ok(StrategyExecution {
                id: row.get("id"),
                instance_id: row.get("instance_id"),
                execution_id: row.get("execution_id"),
                exchange: row.get("exchange"),
                symbol: row.get("symbol"),
                execution_type: row.get("execution_type"),
                input_data: row.get("input_data"),
                output_data: row.get("output_data"),
                decisions: row.get("decisions"),
                market_price: row.get("market_price"),
                volume_24h: row.get("volume_24h"),
                volatility: row.get("volatility"),
                status: row.get("status"),
                orders_created: row.get("orders_created"),
                orders_filled: row.get("orders_filled"),
                total_volume: row.get("total_volume"),
                realized_pnl: row.get("realized_pnl"),
                execution_time_ms: row.get("execution_time_ms"),
                latency_ms: row.get("latency_ms"),
                slippage_bps: row.get("slippage_bps"),
                error_code: row.get("error_code"),
                error_message: row.get("error_message"),
                retry_count: row.get("retry_count"),
                created_at: row.get("created_at"),
                started_at: row.get("started_at"),
                completed_at: row.get("completed_at"),
            })
        }).collect();

        executions
    }
}
