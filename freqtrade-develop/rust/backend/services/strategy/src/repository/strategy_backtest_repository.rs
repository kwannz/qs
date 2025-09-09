use sqlx::{Pool, Postgres, Row};
use uuid::Uuid;
use anyhow::Result;
use chrono::{DateTime, Utc, NaiveDate};
use rust_decimal::Decimal;

use super::Repository;

#[derive(Debug)]
pub struct BacktestCompletion {
    pub id: Uuid,
    pub final_capital: Decimal,
    pub total_return: Decimal,
    pub annualized_return: Decimal,
    pub max_drawdown: Decimal,
    pub sharpe_ratio: Decimal,
    pub total_trades: i32,
    pub winning_trades: i32,
    pub execution_time_seconds: i32,
    pub detailed_results: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct StrategyBacktest {
    pub id: Uuid,
    pub strategy_id: Uuid,
    pub name: String,
    pub params: serde_json::Value,
    pub symbols: Vec<String>,
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub initial_capital: Decimal,
    pub final_capital: Option<Decimal>,
    pub total_return: Option<Decimal>,
    pub annualized_return: Option<Decimal>,
    pub max_drawdown: Option<Decimal>,
    pub sharpe_ratio: Option<Decimal>,
    pub sortino_ratio: Option<Decimal>,
    pub calmar_ratio: Option<Decimal>,
    pub total_trades: Option<i32>,
    pub winning_trades: Option<i32>,
    pub losing_trades: Option<i32>,
    pub win_rate: Option<Decimal>,
    pub avg_win: Option<Decimal>,
    pub avg_loss: Option<Decimal>,
    pub profit_factor: Option<Decimal>,
    pub value_at_risk: Option<Decimal>,
    pub expected_shortfall: Option<Decimal>,
    pub volatility: Option<Decimal>,
    pub beta: Option<Decimal>,
    pub daily_returns: Option<serde_json::Value>,
    pub equity_curve: Option<serde_json::Value>,
    pub drawdown_periods: Option<serde_json::Value>,
    pub monthly_returns: Option<serde_json::Value>,
    pub status: String,
    pub progress_percent: Option<i32>,
    pub execution_time_seconds: Option<i32>,
    pub error_message: Option<String>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

pub struct StrategyBacktestRepository {
    pool: Pool<Postgres>,
}

impl StrategyBacktestRepository {
    pub fn new(pool: Pool<Postgres>) -> Self {
        Self { pool }
    }

    pub async fn find_by_strategy_id(&self, strategy_id: Uuid, limit: Option<u32>, offset: Option<u32>) -> Result<Vec<StrategyBacktest>> {
        let limit = limit.unwrap_or(50).min(100) as i64;
        let offset = offset.unwrap_or(0) as i64;

        let rows = sqlx::query(
            r#"
            SELECT id, strategy_id, name, params, symbols, start_date, end_date,
                   initial_capital, final_capital, total_return, annualized_return,
                   max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio,
                   total_trades, winning_trades, losing_trades, win_rate,
                   avg_win, avg_loss, profit_factor, value_at_risk,
                   expected_shortfall, volatility, beta, daily_returns,
                   equity_curve, drawdown_periods, monthly_returns, status,
                   progress_percent, execution_time_seconds, error_message,
                   created_at, started_at, completed_at
            FROM strategy_backtests 
            WHERE strategy_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(strategy_id)
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;

        let backtests: Result<Vec<StrategyBacktest>> = rows.into_iter().map(|row| {
            Ok(StrategyBacktest {
                id: row.get("id"),
                strategy_id: row.get("strategy_id"),
                name: row.get("name"),
                params: row.get("params"),
                symbols: row.get("symbols"),
                start_date: row.get("start_date"),
                end_date: row.get("end_date"),
                initial_capital: row.get("initial_capital"),
                final_capital: row.get("final_capital"),
                total_return: row.get("total_return"),
                annualized_return: row.get("annualized_return"),
                max_drawdown: row.get("max_drawdown"),
                sharpe_ratio: row.get("sharpe_ratio"),
                sortino_ratio: row.get("sortino_ratio"),
                calmar_ratio: row.get("calmar_ratio"),
                total_trades: row.get("total_trades"),
                winning_trades: row.get("winning_trades"),
                losing_trades: row.get("losing_trades"),
                win_rate: row.get("win_rate"),
                avg_win: row.get("avg_win"),
                avg_loss: row.get("avg_loss"),
                profit_factor: row.get("profit_factor"),
                value_at_risk: row.get("value_at_risk"),
                expected_shortfall: row.get("expected_shortfall"),
                volatility: row.get("volatility"),
                beta: row.get("beta"),
                daily_returns: row.get("daily_returns"),
                equity_curve: row.get("equity_curve"),
                drawdown_periods: row.get("drawdown_periods"),
                monthly_returns: row.get("monthly_returns"),
                status: row.get("status"),
                progress_percent: row.get("progress_percent"),
                execution_time_seconds: row.get("execution_time_seconds"),
                error_message: row.get("error_message"),
                created_at: row.get("created_at"),
                started_at: row.get("started_at"),
                completed_at: row.get("completed_at"),
            })
        }).collect();

        backtests
    }

    pub async fn find_by_status(&self, status: &str, limit: Option<u32>, offset: Option<u32>) -> Result<Vec<StrategyBacktest>> {
        let limit = limit.unwrap_or(50).min(100) as i64;
        let offset = offset.unwrap_or(0) as i64;

        let rows = sqlx::query(
            r#"
            SELECT id, strategy_id, name, params, symbols, start_date, end_date,
                   initial_capital, final_capital, total_return, annualized_return,
                   max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio,
                   total_trades, winning_trades, losing_trades, win_rate,
                   avg_win, avg_loss, profit_factor, value_at_risk,
                   expected_shortfall, volatility, beta, daily_returns,
                   equity_curve, drawdown_periods, monthly_returns, status,
                   progress_percent, execution_time_seconds, error_message,
                   created_at, started_at, completed_at
            FROM strategy_backtests 
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

        let backtests: Result<Vec<StrategyBacktest>> = rows.into_iter().map(|row| {
            Ok(StrategyBacktest {
                id: row.get("id"),
                strategy_id: row.get("strategy_id"),
                name: row.get("name"),
                params: row.get("params"),
                symbols: row.get("symbols"),
                start_date: row.get("start_date"),
                end_date: row.get("end_date"),
                initial_capital: row.get("initial_capital"),
                final_capital: row.get("final_capital"),
                total_return: row.get("total_return"),
                annualized_return: row.get("annualized_return"),
                max_drawdown: row.get("max_drawdown"),
                sharpe_ratio: row.get("sharpe_ratio"),
                sortino_ratio: row.get("sortino_ratio"),
                calmar_ratio: row.get("calmar_ratio"),
                total_trades: row.get("total_trades"),
                winning_trades: row.get("winning_trades"),
                losing_trades: row.get("losing_trades"),
                win_rate: row.get("win_rate"),
                avg_win: row.get("avg_win"),
                avg_loss: row.get("avg_loss"),
                profit_factor: row.get("profit_factor"),
                value_at_risk: row.get("value_at_risk"),
                expected_shortfall: row.get("expected_shortfall"),
                volatility: row.get("volatility"),
                beta: row.get("beta"),
                daily_returns: row.get("daily_returns"),
                equity_curve: row.get("equity_curve"),
                drawdown_periods: row.get("drawdown_periods"),
                monthly_returns: row.get("monthly_returns"),
                status: row.get("status"),
                progress_percent: row.get("progress_percent"),
                execution_time_seconds: row.get("execution_time_seconds"),
                error_message: row.get("error_message"),
                created_at: row.get("created_at"),
                started_at: row.get("started_at"),
                completed_at: row.get("completed_at"),
            })
        }).collect();

        backtests
    }

    pub async fn find_completed_by_performance(&self, limit: Option<u32>, offset: Option<u32>) -> Result<Vec<StrategyBacktest>> {
        let limit = limit.unwrap_or(50).min(100) as i64;
        let offset = offset.unwrap_or(0) as i64;

        let rows = sqlx::query(
            r#"
            SELECT id, strategy_id, name, params, symbols, start_date, end_date,
                   initial_capital, final_capital, total_return, annualized_return,
                   max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio,
                   total_trades, winning_trades, losing_trades, win_rate,
                   avg_win, avg_loss, profit_factor, value_at_risk,
                   expected_shortfall, volatility, beta, daily_returns,
                   equity_curve, drawdown_periods, monthly_returns, status,
                   progress_percent, execution_time_seconds, error_message,
                   created_at, started_at, completed_at
            FROM strategy_backtests 
            WHERE status = 'completed'
            ORDER BY annualized_return DESC NULLS LAST, sharpe_ratio DESC NULLS LAST
            LIMIT $1 OFFSET $2
            "#,
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;

        let backtests: Result<Vec<StrategyBacktest>> = rows.into_iter().map(|row| {
            Ok(StrategyBacktest {
                id: row.get("id"),
                strategy_id: row.get("strategy_id"),
                name: row.get("name"),
                params: row.get("params"),
                symbols: row.get("symbols"),
                start_date: row.get("start_date"),
                end_date: row.get("end_date"),
                initial_capital: row.get("initial_capital"),
                final_capital: row.get("final_capital"),
                total_return: row.get("total_return"),
                annualized_return: row.get("annualized_return"),
                max_drawdown: row.get("max_drawdown"),
                sharpe_ratio: row.get("sharpe_ratio"),
                sortino_ratio: row.get("sortino_ratio"),
                calmar_ratio: row.get("calmar_ratio"),
                total_trades: row.get("total_trades"),
                winning_trades: row.get("winning_trades"),
                losing_trades: row.get("losing_trades"),
                win_rate: row.get("win_rate"),
                avg_win: row.get("avg_win"),
                avg_loss: row.get("avg_loss"),
                profit_factor: row.get("profit_factor"),
                value_at_risk: row.get("value_at_risk"),
                expected_shortfall: row.get("expected_shortfall"),
                volatility: row.get("volatility"),
                beta: row.get("beta"),
                daily_returns: row.get("daily_returns"),
                equity_curve: row.get("equity_curve"),
                drawdown_periods: row.get("drawdown_periods"),
                monthly_returns: row.get("monthly_returns"),
                status: row.get("status"),
                progress_percent: row.get("progress_percent"),
                execution_time_seconds: row.get("execution_time_seconds"),
                error_message: row.get("error_message"),
                created_at: row.get("created_at"),
                started_at: row.get("started_at"),
                completed_at: row.get("completed_at"),
            })
        }).collect();

        backtests
    }

    pub async fn update_progress(&self, id: Uuid, progress_percent: i32) -> Result<()> {
        sqlx::query(
            "UPDATE strategy_backtests SET progress_percent = $2 WHERE id = $1"
        )
        .bind(id)
        .bind(progress_percent)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn update_status(&self, id: Uuid, status: &str, error_message: Option<&str>) -> Result<()> {
        if let Some(error) = error_message {
            sqlx::query(
                "UPDATE strategy_backtests SET status = $2, error_message = $3 WHERE id = $1"
            )
            .bind(id)
            .bind(status)
            .bind(error)
            .execute(&self.pool)
            .await?;
        } else {
            sqlx::query(
                "UPDATE strategy_backtests SET status = $2 WHERE id = $1"
            )
            .bind(id)
            .bind(status)
            .execute(&self.pool)
            .await?;
        }

        Ok(())
    }

    pub async fn complete_backtest(&self, completion: BacktestCompletion) -> Result<()> {
        let now = Utc::now();
        
        sqlx::query(
            r#"
            UPDATE strategy_backtests SET 
                status = 'completed',
                progress_percent = 100,
                final_capital = $2,
                total_return = $3,
                annualized_return = $4,
                max_drawdown = $5,
                sharpe_ratio = $6,
                total_trades = $7,
                winning_trades = $8,
                losing_trades = $9,
                win_rate = $10,
                execution_time_seconds = $11,
                completed_at = $12,
                daily_returns = $13
            WHERE id = $1
            "#,
        )
        .bind(completion.id)
        .bind(completion.final_capital)
        .bind(completion.total_return)
        .bind(completion.annualized_return)
        .bind(completion.max_drawdown)
        .bind(completion.sharpe_ratio)
        .bind(completion.total_trades)
        .bind(completion.winning_trades)
        .bind(completion.total_trades - completion.winning_trades)
        .bind(if completion.total_trades > 0 { Some((completion.winning_trades as f64 / completion.total_trades as f64 * 100.0) as i32) } else { None })
        .bind(completion.execution_time_seconds)
        .bind(now)
        .bind(&completion.detailed_results)
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

impl Repository<StrategyBacktest> for StrategyBacktestRepository {
    async fn create(&self, backtest: &StrategyBacktest) -> Result<StrategyBacktest> {
        sqlx::query(
            r#"
            INSERT INTO strategy_backtests (
                id, strategy_id, name, params, symbols, start_date, end_date,
                initial_capital, final_capital, total_return, annualized_return,
                max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio,
                total_trades, winning_trades, losing_trades, win_rate,
                avg_win, avg_loss, profit_factor, value_at_risk,
                expected_shortfall, volatility, beta, daily_returns,
                equity_curve, drawdown_periods, monthly_returns, status,
                progress_percent, execution_time_seconds, error_message,
                created_at, started_at, completed_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26,
                $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37
            )
            "#,
        )
        .bind(backtest.id)
        .bind(backtest.strategy_id)
        .bind(&backtest.name)
        .bind(&backtest.params)
        .bind(&backtest.symbols)
        .bind(backtest.start_date)
        .bind(backtest.end_date)
        .bind(backtest.initial_capital)
        .bind(backtest.final_capital)
        .bind(backtest.total_return)
        .bind(backtest.annualized_return)
        .bind(backtest.max_drawdown)
        .bind(backtest.sharpe_ratio)
        .bind(backtest.sortino_ratio)
        .bind(backtest.calmar_ratio)
        .bind(backtest.total_trades)
        .bind(backtest.winning_trades)
        .bind(backtest.losing_trades)
        .bind(backtest.win_rate)
        .bind(backtest.avg_win)
        .bind(backtest.avg_loss)
        .bind(backtest.profit_factor)
        .bind(backtest.value_at_risk)
        .bind(backtest.expected_shortfall)
        .bind(backtest.volatility)
        .bind(backtest.beta)
        .bind(&backtest.daily_returns)
        .bind(&backtest.equity_curve)
        .bind(&backtest.drawdown_periods)
        .bind(&backtest.monthly_returns)
        .bind(&backtest.status)
        .bind(backtest.progress_percent)
        .bind(backtest.execution_time_seconds)
        .bind(&backtest.error_message)
        .bind(backtest.created_at)
        .bind(backtest.started_at)
        .bind(backtest.completed_at)
        .execute(&self.pool)
        .await?;

        Ok(backtest.clone())
    }

    async fn find_by_id(&self, id: Uuid) -> Result<Option<StrategyBacktest>> {
        let row = sqlx::query(
            r#"
            SELECT id, strategy_id, name, params, symbols, start_date, end_date,
                   initial_capital, final_capital, total_return, annualized_return,
                   max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio,
                   total_trades, winning_trades, losing_trades, win_rate,
                   avg_win, avg_loss, profit_factor, value_at_risk,
                   expected_shortfall, volatility, beta, daily_returns,
                   equity_curve, drawdown_periods, monthly_returns, status,
                   progress_percent, execution_time_seconds, error_message,
                   created_at, started_at, completed_at
            FROM strategy_backtests 
            WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => Ok(Some(StrategyBacktest {
                id: row.get("id"),
                strategy_id: row.get("strategy_id"),
                name: row.get("name"),
                params: row.get("params"),
                symbols: row.get("symbols"),
                start_date: row.get("start_date"),
                end_date: row.get("end_date"),
                initial_capital: row.get("initial_capital"),
                final_capital: row.get("final_capital"),
                total_return: row.get("total_return"),
                annualized_return: row.get("annualized_return"),
                max_drawdown: row.get("max_drawdown"),
                sharpe_ratio: row.get("sharpe_ratio"),
                sortino_ratio: row.get("sortino_ratio"),
                calmar_ratio: row.get("calmar_ratio"),
                total_trades: row.get("total_trades"),
                winning_trades: row.get("winning_trades"),
                losing_trades: row.get("losing_trades"),
                win_rate: row.get("win_rate"),
                avg_win: row.get("avg_win"),
                avg_loss: row.get("avg_loss"),
                profit_factor: row.get("profit_factor"),
                value_at_risk: row.get("value_at_risk"),
                expected_shortfall: row.get("expected_shortfall"),
                volatility: row.get("volatility"),
                beta: row.get("beta"),
                daily_returns: row.get("daily_returns"),
                equity_curve: row.get("equity_curve"),
                drawdown_periods: row.get("drawdown_periods"),
                monthly_returns: row.get("monthly_returns"),
                status: row.get("status"),
                progress_percent: row.get("progress_percent"),
                execution_time_seconds: row.get("execution_time_seconds"),
                error_message: row.get("error_message"),
                created_at: row.get("created_at"),
                started_at: row.get("started_at"),
                completed_at: row.get("completed_at"),
            })),
            None => Ok(None),
        }
    }

    async fn update(&self, id: Uuid, backtest: &StrategyBacktest) -> Result<Option<StrategyBacktest>> {
        let result = sqlx::query(
            r#"
            UPDATE strategy_backtests SET 
                name = $2, params = $3, symbols = $4, start_date = $5,
                end_date = $6, initial_capital = $7, final_capital = $8,
                total_return = $9, annualized_return = $10, max_drawdown = $11,
                sharpe_ratio = $12, sortino_ratio = $13, calmar_ratio = $14,
                total_trades = $15, winning_trades = $16, losing_trades = $17,
                win_rate = $18, avg_win = $19, avg_loss = $20, profit_factor = $21,
                value_at_risk = $22, expected_shortfall = $23, volatility = $24,
                beta = $25, daily_returns = $26, equity_curve = $27,
                drawdown_periods = $28, monthly_returns = $29, status = $30,
                progress_percent = $31, execution_time_seconds = $32,
                error_message = $33, started_at = $34, completed_at = $35
            WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(&backtest.name)
        .bind(&backtest.params)
        .bind(&backtest.symbols)
        .bind(backtest.start_date)
        .bind(backtest.end_date)
        .bind(backtest.initial_capital)
        .bind(backtest.final_capital)
        .bind(backtest.total_return)
        .bind(backtest.annualized_return)
        .bind(backtest.max_drawdown)
        .bind(backtest.sharpe_ratio)
        .bind(backtest.sortino_ratio)
        .bind(backtest.calmar_ratio)
        .bind(backtest.total_trades)
        .bind(backtest.winning_trades)
        .bind(backtest.losing_trades)
        .bind(backtest.win_rate)
        .bind(backtest.avg_win)
        .bind(backtest.avg_loss)
        .bind(backtest.profit_factor)
        .bind(backtest.value_at_risk)
        .bind(backtest.expected_shortfall)
        .bind(backtest.volatility)
        .bind(backtest.beta)
        .bind(&backtest.daily_returns)
        .bind(&backtest.equity_curve)
        .bind(&backtest.drawdown_periods)
        .bind(&backtest.monthly_returns)
        .bind(&backtest.status)
        .bind(backtest.progress_percent)
        .bind(backtest.execution_time_seconds)
        .bind(&backtest.error_message)
        .bind(backtest.started_at)
        .bind(backtest.completed_at)
        .execute(&self.pool)
        .await?;

        if result.rows_affected() > 0 {
            self.find_by_id(id).await
        } else {
            Ok(None)
        }
    }

    async fn delete(&self, id: Uuid) -> Result<bool> {
        let result = sqlx::query("DELETE FROM strategy_backtests WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected() > 0)
    }

    async fn list(&self, limit: Option<u32>, offset: Option<u32>) -> Result<Vec<StrategyBacktest>> {
        let limit = limit.unwrap_or(50).min(100) as i64;
        let offset = offset.unwrap_or(0) as i64;

        let rows = sqlx::query(
            r#"
            SELECT id, strategy_id, name, params, symbols, start_date, end_date,
                   initial_capital, final_capital, total_return, annualized_return,
                   max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio,
                   total_trades, winning_trades, losing_trades, win_rate,
                   avg_win, avg_loss, profit_factor, value_at_risk,
                   expected_shortfall, volatility, beta, daily_returns,
                   equity_curve, drawdown_periods, monthly_returns, status,
                   progress_percent, execution_time_seconds, error_message,
                   created_at, started_at, completed_at
            FROM strategy_backtests
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            "#,
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;

        let backtests: Result<Vec<StrategyBacktest>> = rows.into_iter().map(|row| {
            Ok(StrategyBacktest {
                id: row.get("id"),
                strategy_id: row.get("strategy_id"),
                name: row.get("name"),
                params: row.get("params"),
                symbols: row.get("symbols"),
                start_date: row.get("start_date"),
                end_date: row.get("end_date"),
                initial_capital: row.get("initial_capital"),
                final_capital: row.get("final_capital"),
                total_return: row.get("total_return"),
                annualized_return: row.get("annualized_return"),
                max_drawdown: row.get("max_drawdown"),
                sharpe_ratio: row.get("sharpe_ratio"),
                sortino_ratio: row.get("sortino_ratio"),
                calmar_ratio: row.get("calmar_ratio"),
                total_trades: row.get("total_trades"),
                winning_trades: row.get("winning_trades"),
                losing_trades: row.get("losing_trades"),
                win_rate: row.get("win_rate"),
                avg_win: row.get("avg_win"),
                avg_loss: row.get("avg_loss"),
                profit_factor: row.get("profit_factor"),
                value_at_risk: row.get("value_at_risk"),
                expected_shortfall: row.get("expected_shortfall"),
                volatility: row.get("volatility"),
                beta: row.get("beta"),
                daily_returns: row.get("daily_returns"),
                equity_curve: row.get("equity_curve"),
                drawdown_periods: row.get("drawdown_periods"),
                monthly_returns: row.get("monthly_returns"),
                status: row.get("status"),
                progress_percent: row.get("progress_percent"),
                execution_time_seconds: row.get("execution_time_seconds"),
                error_message: row.get("error_message"),
                created_at: row.get("created_at"),
                started_at: row.get("started_at"),
                completed_at: row.get("completed_at"),
            })
        }).collect();

        backtests
    }
}