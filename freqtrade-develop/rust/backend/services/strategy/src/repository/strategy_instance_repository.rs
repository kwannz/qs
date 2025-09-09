use sqlx::{Pool, Postgres, Row};
use uuid::Uuid;
use anyhow::Result;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

use super::Repository;

#[derive(Debug, Clone)]
pub struct StrategyInstance {
    pub id: Uuid,
    pub strategy_id: Uuid,
    pub name: String,
    pub params: serde_json::Value,
    pub exchanges: Vec<String>,
    pub symbols: Vec<String>,
    pub allocated_capital: Decimal,
    pub max_capital_utilization: Option<Decimal>,
    pub stop_loss_percent: Option<Decimal>,
    pub take_profit_percent: Option<Decimal>,
    pub max_daily_trades: Option<i32>,
    pub max_open_positions: Option<i32>,
    pub status: String,
    pub health_status: Option<String>,
    pub auto_start: Option<bool>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub trading_schedule: Option<serde_json::Value>,
    pub total_pnl: Option<Decimal>,
    pub daily_pnl: Option<Decimal>,
    pub win_rate: Option<Decimal>,
    pub total_trades: Option<i32>,
    pub winning_trades: Option<i32>,
    pub losing_trades: Option<i32>,
    pub last_execution: Option<DateTime<Utc>>,
    pub next_execution: Option<DateTime<Utc>>,
    pub execution_count: Option<i64>,
    pub error_count: Option<i32>,
    pub last_error: Option<String>,
    pub created_by: String,
    pub environment: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub stopped_at: Option<DateTime<Utc>>,
}

pub struct StrategyInstanceRepository {
    pool: Pool<Postgres>,
}

impl StrategyInstanceRepository {
    pub fn new(pool: Pool<Postgres>) -> Self {
        Self { pool }
    }

    pub async fn find_by_strategy_id(&self, strategy_id: Uuid) -> Result<Vec<StrategyInstance>> {
        let rows = sqlx::query(
            r#"
            SELECT id, strategy_id, name, params, exchanges, symbols, allocated_capital,
                   max_capital_utilization, stop_loss_percent, take_profit_percent,
                   max_daily_trades, max_open_positions, status, health_status,
                   auto_start, start_time, end_time, trading_schedule, total_pnl,
                   daily_pnl, win_rate, total_trades, winning_trades, losing_trades,
                   last_execution, next_execution, execution_count, error_count,
                   last_error, created_by, environment, created_at, updated_at,
                   started_at, stopped_at
            FROM strategy_instances 
            WHERE strategy_id = $1
            ORDER BY created_at DESC
            "#,
        )
        .bind(strategy_id)
        .fetch_all(&self.pool)
        .await?;

        let instances: Result<Vec<StrategyInstance>> = rows.into_iter().map(|row| {
            Ok(StrategyInstance {
                id: row.get("id"),
                strategy_id: row.get("strategy_id"),
                name: row.get("name"),
                params: row.get("params"),
                exchanges: row.get("exchanges"),
                symbols: row.get("symbols"),
                allocated_capital: row.get("allocated_capital"),
                max_capital_utilization: row.get("max_capital_utilization"),
                stop_loss_percent: row.get("stop_loss_percent"),
                take_profit_percent: row.get("take_profit_percent"),
                max_daily_trades: row.get("max_daily_trades"),
                max_open_positions: row.get("max_open_positions"),
                status: row.get("status"),
                health_status: row.get("health_status"),
                auto_start: row.get("auto_start"),
                start_time: row.get("start_time"),
                end_time: row.get("end_time"),
                trading_schedule: row.get("trading_schedule"),
                total_pnl: row.get("total_pnl"),
                daily_pnl: row.get("daily_pnl"),
                win_rate: row.get("win_rate"),
                total_trades: row.get("total_trades"),
                winning_trades: row.get("winning_trades"),
                losing_trades: row.get("losing_trades"),
                last_execution: row.get("last_execution"),
                next_execution: row.get("next_execution"),
                execution_count: row.get("execution_count"),
                error_count: row.get("error_count"),
                last_error: row.get("last_error"),
                created_by: row.get("created_by"),
                environment: row.get("environment"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
                started_at: row.get("started_at"),
                stopped_at: row.get("stopped_at"),
            })
        }).collect();

        instances
    }

    pub async fn find_by_status(&self, status: &str) -> Result<Vec<StrategyInstance>> {
        let rows = sqlx::query(
            r#"
            SELECT id, strategy_id, name, params, exchanges, symbols, allocated_capital,
                   max_capital_utilization, stop_loss_percent, take_profit_percent,
                   max_daily_trades, max_open_positions, status, health_status,
                   auto_start, start_time, end_time, trading_schedule, total_pnl,
                   daily_pnl, win_rate, total_trades, winning_trades, losing_trades,
                   last_execution, next_execution, execution_count, error_count,
                   last_error, created_by, environment, created_at, updated_at,
                   started_at, stopped_at
            FROM strategy_instances 
            WHERE status = $1
            ORDER BY created_at DESC
            "#,
        )
        .bind(status)
        .fetch_all(&self.pool)
        .await?;

        let instances: Result<Vec<StrategyInstance>> = rows.into_iter().map(|row| {
            Ok(StrategyInstance {
                id: row.get("id"),
                strategy_id: row.get("strategy_id"),
                name: row.get("name"),
                params: row.get("params"),
                exchanges: row.get("exchanges"),
                symbols: row.get("symbols"),
                allocated_capital: row.get("allocated_capital"),
                max_capital_utilization: row.get("max_capital_utilization"),
                stop_loss_percent: row.get("stop_loss_percent"),
                take_profit_percent: row.get("take_profit_percent"),
                max_daily_trades: row.get("max_daily_trades"),
                max_open_positions: row.get("max_open_positions"),
                status: row.get("status"),
                health_status: row.get("health_status"),
                auto_start: row.get("auto_start"),
                start_time: row.get("start_time"),
                end_time: row.get("end_time"),
                trading_schedule: row.get("trading_schedule"),
                total_pnl: row.get("total_pnl"),
                daily_pnl: row.get("daily_pnl"),
                win_rate: row.get("win_rate"),
                total_trades: row.get("total_trades"),
                winning_trades: row.get("winning_trades"),
                losing_trades: row.get("losing_trades"),
                last_execution: row.get("last_execution"),
                next_execution: row.get("next_execution"),
                execution_count: row.get("execution_count"),
                error_count: row.get("error_count"),
                last_error: row.get("last_error"),
                created_by: row.get("created_by"),
                environment: row.get("environment"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
                started_at: row.get("started_at"),
                stopped_at: row.get("stopped_at"),
            })
        }).collect();

        instances
    }

    pub async fn update_pnl(&self, id: Uuid, total_pnl: Decimal, daily_pnl: Decimal) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE strategy_instances 
            SET total_pnl = $2, daily_pnl = $3, updated_at = $4
            WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(total_pnl)
        .bind(daily_pnl)
        .bind(Utc::now())
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn update_status(&self, id: Uuid, status: &str, health_status: Option<&str>) -> Result<()> {
        let now = Utc::now();
        
        if let Some(health) = health_status {
            sqlx::query(
                r#"
                UPDATE strategy_instances 
                SET status = $2, health_status = $3, updated_at = $4 
                WHERE id = $1
                "#
            )
            .bind(id)
            .bind(status)
            .bind(health)
            .bind(now)
            .execute(&self.pool)
            .await?;
        } else {
            sqlx::query(
                r#"
                UPDATE strategy_instances 
                SET status = $2, updated_at = $3 
                WHERE id = $1
                "#
            )
            .bind(id)
            .bind(status)
            .bind(now)
            .execute(&self.pool)
            .await?;
        }
        Ok(())
    }

    pub async fn update_execution_count(&self, id: Uuid) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE strategy_instances 
            SET execution_count = COALESCE(execution_count, 0) + 1,
                last_execution = $2,
                updated_at = $2
            WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(Utc::now())
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

impl Repository<StrategyInstance> for StrategyInstanceRepository {
    async fn create(&self, instance: &StrategyInstance) -> Result<StrategyInstance> {
        sqlx::query(
            r#"
            INSERT INTO strategy_instances (
                id, strategy_id, name, params, exchanges, symbols, allocated_capital,
                max_capital_utilization, stop_loss_percent, take_profit_percent,
                max_daily_trades, max_open_positions, status, health_status,
                auto_start, start_time, end_time, trading_schedule, total_pnl,
                daily_pnl, win_rate, total_trades, winning_trades, losing_trades,
                last_execution, next_execution, execution_count, error_count,
                last_error, created_by, environment, created_at, updated_at,
                started_at, stopped_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26,
                $27, $28, $29, $30, $31, $32, $33, $34, $35
            )
            "#,
        )
        .bind(instance.id)
        .bind(instance.strategy_id)
        .bind(&instance.name)
        .bind(&instance.params)
        .bind(&instance.exchanges)
        .bind(&instance.symbols)
        .bind(instance.allocated_capital)
        .bind(instance.max_capital_utilization)
        .bind(instance.stop_loss_percent)
        .bind(instance.take_profit_percent)
        .bind(instance.max_daily_trades)
        .bind(instance.max_open_positions)
        .bind(&instance.status)
        .bind(&instance.health_status)
        .bind(instance.auto_start)
        .bind(instance.start_time)
        .bind(instance.end_time)
        .bind(&instance.trading_schedule)
        .bind(instance.total_pnl)
        .bind(instance.daily_pnl)
        .bind(instance.win_rate)
        .bind(instance.total_trades)
        .bind(instance.winning_trades)
        .bind(instance.losing_trades)
        .bind(instance.last_execution)
        .bind(instance.next_execution)
        .bind(instance.execution_count)
        .bind(instance.error_count)
        .bind(&instance.last_error)
        .bind(&instance.created_by)
        .bind(&instance.environment)
        .bind(instance.created_at)
        .bind(instance.updated_at)
        .bind(instance.started_at)
        .bind(instance.stopped_at)
        .execute(&self.pool)
        .await?;

        Ok(instance.clone())
    }

    async fn find_by_id(&self, id: Uuid) -> Result<Option<StrategyInstance>> {
        let row = sqlx::query(
            r#"
            SELECT id, strategy_id, name, params, exchanges, symbols, allocated_capital,
                   max_capital_utilization, stop_loss_percent, take_profit_percent,
                   max_daily_trades, max_open_positions, status, health_status,
                   auto_start, start_time, end_time, trading_schedule, total_pnl,
                   daily_pnl, win_rate, total_trades, winning_trades, losing_trades,
                   last_execution, next_execution, execution_count, error_count,
                   last_error, created_by, environment, created_at, updated_at,
                   started_at, stopped_at
            FROM strategy_instances 
            WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => Ok(Some(StrategyInstance {
                id: row.get("id"),
                strategy_id: row.get("strategy_id"),
                name: row.get("name"),
                params: row.get("params"),
                exchanges: row.get("exchanges"),
                symbols: row.get("symbols"),
                allocated_capital: row.get("allocated_capital"),
                max_capital_utilization: row.get("max_capital_utilization"),
                stop_loss_percent: row.get("stop_loss_percent"),
                take_profit_percent: row.get("take_profit_percent"),
                max_daily_trades: row.get("max_daily_trades"),
                max_open_positions: row.get("max_open_positions"),
                status: row.get("status"),
                health_status: row.get("health_status"),
                auto_start: row.get("auto_start"),
                start_time: row.get("start_time"),
                end_time: row.get("end_time"),
                trading_schedule: row.get("trading_schedule"),
                total_pnl: row.get("total_pnl"),
                daily_pnl: row.get("daily_pnl"),
                win_rate: row.get("win_rate"),
                total_trades: row.get("total_trades"),
                winning_trades: row.get("winning_trades"),
                losing_trades: row.get("losing_trades"),
                last_execution: row.get("last_execution"),
                next_execution: row.get("next_execution"),
                execution_count: row.get("execution_count"),
                error_count: row.get("error_count"),
                last_error: row.get("last_error"),
                created_by: row.get("created_by"),
                environment: row.get("environment"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
                started_at: row.get("started_at"),
                stopped_at: row.get("stopped_at"),
            })),
            None => Ok(None),
        }
    }

    async fn update(&self, id: Uuid, instance: &StrategyInstance) -> Result<Option<StrategyInstance>> {
        let result = sqlx::query(
            r#"
            UPDATE strategy_instances SET 
                name = $2, params = $3, exchanges = $4, symbols = $5,
                allocated_capital = $6, max_capital_utilization = $7,
                stop_loss_percent = $8, take_profit_percent = $9,
                max_daily_trades = $10, max_open_positions = $11,
                status = $12, health_status = $13, auto_start = $14,
                start_time = $15, end_time = $16, trading_schedule = $17,
                updated_at = $18
            WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(&instance.name)
        .bind(&instance.params)
        .bind(&instance.exchanges)
        .bind(&instance.symbols)
        .bind(instance.allocated_capital)
        .bind(instance.max_capital_utilization)
        .bind(instance.stop_loss_percent)
        .bind(instance.take_profit_percent)
        .bind(instance.max_daily_trades)
        .bind(instance.max_open_positions)
        .bind(&instance.status)
        .bind(&instance.health_status)
        .bind(instance.auto_start)
        .bind(instance.start_time)
        .bind(instance.end_time)
        .bind(&instance.trading_schedule)
        .bind(Utc::now())
        .execute(&self.pool)
        .await?;

        if result.rows_affected() > 0 {
            self.find_by_id(id).await
        } else {
            Ok(None)
        }
    }

    async fn delete(&self, id: Uuid) -> Result<bool> {
        let result = sqlx::query("DELETE FROM strategy_instances WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected() > 0)
    }

    async fn list(&self, limit: Option<u32>, offset: Option<u32>) -> Result<Vec<StrategyInstance>> {
        let limit = limit.unwrap_or(50).min(100) as i64;
        let offset = offset.unwrap_or(0) as i64;

        let rows = sqlx::query(
            r#"
            SELECT id, strategy_id, name, params, exchanges, symbols, allocated_capital,
                   max_capital_utilization, stop_loss_percent, take_profit_percent,
                   max_daily_trades, max_open_positions, status, health_status,
                   auto_start, start_time, end_time, trading_schedule, total_pnl,
                   daily_pnl, win_rate, total_trades, winning_trades, losing_trades,
                   last_execution, next_execution, execution_count, error_count,
                   last_error, created_by, environment, created_at, updated_at,
                   started_at, stopped_at
            FROM strategy_instances
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            "#,
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;

        let instances: Result<Vec<StrategyInstance>> = rows.into_iter().map(|row| {
            Ok(StrategyInstance {
                id: row.get("id"),
                strategy_id: row.get("strategy_id"),
                name: row.get("name"),
                params: row.get("params"),
                exchanges: row.get("exchanges"),
                symbols: row.get("symbols"),
                allocated_capital: row.get("allocated_capital"),
                max_capital_utilization: row.get("max_capital_utilization"),
                stop_loss_percent: row.get("stop_loss_percent"),
                take_profit_percent: row.get("take_profit_percent"),
                max_daily_trades: row.get("max_daily_trades"),
                max_open_positions: row.get("max_open_positions"),
                status: row.get("status"),
                health_status: row.get("health_status"),
                auto_start: row.get("auto_start"),
                start_time: row.get("start_time"),
                end_time: row.get("end_time"),
                trading_schedule: row.get("trading_schedule"),
                total_pnl: row.get("total_pnl"),
                daily_pnl: row.get("daily_pnl"),
                win_rate: row.get("win_rate"),
                total_trades: row.get("total_trades"),
                winning_trades: row.get("winning_trades"),
                losing_trades: row.get("losing_trades"),
                last_execution: row.get("last_execution"),
                next_execution: row.get("next_execution"),
                execution_count: row.get("execution_count"),
                error_count: row.get("error_count"),
                last_error: row.get("last_error"),
                created_by: row.get("created_by"),
                environment: row.get("environment"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
                started_at: row.get("started_at"),
                stopped_at: row.get("stopped_at"),
            })
        }).collect();

        instances
    }
}