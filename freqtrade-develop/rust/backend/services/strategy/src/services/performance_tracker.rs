use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::config::Config;
use crate::models::*;

#[allow(dead_code)]
pub struct PerformanceTracker {
    #[allow(dead_code)]
    config: Arc<Config>,
    #[allow(dead_code)]
    executions: Arc<RwLock<HashMap<Uuid, Vec<StrategyExecution>>>>,
    #[allow(dead_code)]
    performances: Arc<RwLock<HashMap<Uuid, StrategyPerformance>>>,
    #[allow(dead_code)]
    pnl_cache: Arc<RwLock<HashMap<Uuid, (Decimal, Decimal)>>>, // (current_pnl, daily_pnl)
}

#[allow(dead_code)]
impl PerformanceTracker {
    pub async fn new(config: &Config) -> Result<Self> {
        Ok(Self {
            config: Arc::new(config.clone()),
            executions: Arc::new(RwLock::new(HashMap::new())),
            performances: Arc::new(RwLock::new(HashMap::new())),
            pnl_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    // Execution Tracking
    pub async fn record_execution(
        &self,
        strategy_id: Uuid,
        signals: &[Signal],
        duration_ms: u64,
        metrics: ExecutionMetrics,
    ) -> Result<()> {
        let execution = StrategyExecution {
            id: Uuid::new_v4(),
            strategy_id,
            execution_time: Utc::now(),
            status: ExecutionStatus::Completed,
            signals_generated: signals.len() as u32,
            signals_executed: signals.iter().filter(|s| s.executed).count() as u32,
            pnl: self.calculate_execution_pnl(signals).await?,
            execution_duration_ms: duration_ms,
            error_message: None,
            metrics,
        };

        let mut executions = self.executions.write().await;
        executions.entry(strategy_id).or_insert_with(Vec::new).push(execution);

        // Keep only recent executions to prevent memory bloat
        let max_executions = self.config.strategy.max_execution_history;
        if let Some(strategy_executions) = executions.get_mut(&strategy_id) {
            if strategy_executions.len() > max_executions {
                strategy_executions.drain(0..strategy_executions.len() - max_executions);
            }
        }

        // Update performance metrics
        self.update_performance_metrics(strategy_id).await?;

        Ok(())
    }

    pub async fn record_execution_error(
        &self,
        strategy_id: Uuid,
        error: &str,
        duration_ms: u64,
    ) -> Result<()> {
        let execution = StrategyExecution {
            id: Uuid::new_v4(),
            strategy_id,
            execution_time: Utc::now(),
            status: ExecutionStatus::Failed,
            signals_generated: 0,
            signals_executed: 0,
            pnl: Decimal::ZERO,
            execution_duration_ms: duration_ms,
            error_message: Some(error.to_string()),
            metrics: ExecutionMetrics {
                data_points_processed: 0,
                indicators_calculated: 0,
                factors_evaluated: 0,
                risk_checks_passed: 0,
                risk_checks_failed: 0,
                average_signal_strength: 0.0,
                average_signal_confidence: 0.0,
            },
        };

        let mut executions = self.executions.write().await;
        executions.entry(strategy_id).or_insert_with(Vec::new).push(execution);

        Ok(())
    }

    // Performance Metrics
    pub async fn get_performance(&self, strategy_id: Uuid, days: Option<u32>) -> Result<Option<StrategyPerformance>> {
        let performances = self.performances.read().await;
        if let Some(performance) = performances.get(&strategy_id) {
            if let Some(days) = days {
                // Calculate performance for specific time period
                let cutoff_date = Utc::now() - Duration::days(days as i64);
                if performance.period_start >= cutoff_date {
                    return Ok(Some(performance.clone()));
                }
                
                // Recalculate for specific period
                drop(performances);
                self.calculate_performance_for_period(strategy_id, cutoff_date, Utc::now()).await
            } else {
                Ok(Some(performance.clone()))
            }
        } else {
            Ok(None)
        }
    }

    pub async fn get_latest_metrics(&self, strategy_id: Uuid) -> Result<Option<ExecutionMetrics>> {
        let executions = self.executions.read().await;
        if let Some(strategy_executions) = executions.get(&strategy_id) {
            Ok(strategy_executions.last().map(|e| e.metrics.clone()))
        } else {
            Ok(None)
        }
    }

    pub async fn get_current_pnl(&self, strategy_id: Uuid) -> Result<(Decimal, Decimal)> {
        let pnl_cache = self.pnl_cache.read().await;
        Ok(pnl_cache.get(&strategy_id).cloned().unwrap_or((Decimal::ZERO, Decimal::ZERO)))
    }

    // Internal Methods
    async fn calculate_execution_pnl(&self, signals: &[Signal]) -> Result<Decimal> {
        // Simple PnL calculation based on signals
        // In production, this would integrate with position manager and market data
        let mut pnl = Decimal::ZERO;
        
        for signal in signals.iter().filter(|s| s.executed) {
            if let Some(quantity) = signal.quantity {
                // Simplified PnL calculation (assumes price movement favorable to signal)
                let signal_pnl = match signal.action {
                    SignalAction::Buy => quantity * signal.price * Decimal::from_f64_retain(0.001).unwrap_or_default(), // 0.1% gain assumption
                    SignalAction::Sell => quantity * signal.price * Decimal::from_f64_retain(0.001).unwrap_or_default(),
                    _ => Decimal::ZERO,
                };
                pnl += signal_pnl;
            }
        }

        Ok(pnl)
    }

    async fn update_performance_metrics(&self, strategy_id: Uuid) -> Result<()> {
        let executions = self.executions.read().await;
        if let Some(strategy_executions) = executions.get(&strategy_id) {
            if strategy_executions.is_empty() {
                return Ok(());
            }

            // Calculate performance metrics
            let total_executions = strategy_executions.len() as u32;
            let _successful_executions = strategy_executions.iter()
                .filter(|e| e.status == ExecutionStatus::Completed)
                .count() as u32;

            let total_return: Decimal = strategy_executions.iter().map(|e| e.pnl).sum();
            let returns: Vec<f64> = strategy_executions.iter()
                .map(|e| e.pnl.to_f64().unwrap_or(0.0))
                .collect();

            let Some(first_execution) = strategy_executions.first() else {
                return Ok(());
            };
            let Some(last_execution) = strategy_executions.last() else {
                return Ok(());
            };
            let period_start = first_execution.execution_time;
            let period_end = last_execution.execution_time;
            let period_days = (period_end - period_start).num_days().max(1) as f64;

            // Basic statistical calculations
            let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>() / returns.len() as f64;
            let volatility = variance.sqrt();

            let annualized_return = (mean_return * 365.0) / period_days;
            let annualized_volatility = volatility * (365.0 / period_days).sqrt();
            
            let sharpe_ratio = if annualized_volatility > 0.0 {
                annualized_return / annualized_volatility
            } else {
                0.0
            };

            // Calculate drawdown
            let mut running_max: f64 = 0.0;
            let mut max_drawdown: f64 = 0.0;
            let mut running_pnl = 0.0;

            for execution in strategy_executions {
                running_pnl += execution.pnl.to_f64().unwrap_or(0.0);
                running_max = running_max.max(running_pnl);
                let drawdown = (running_max - running_pnl) / running_max.max(1.0);
                max_drawdown = max_drawdown.max(drawdown);
            }

            let winning_executions = strategy_executions.iter()
                .filter(|e| e.pnl > Decimal::ZERO)
                .count() as u32;
            let losing_executions = strategy_executions.iter()
                .filter(|e| e.pnl < Decimal::ZERO)
                .count() as u32;

            let win_rate = if total_executions > 0 {
                (winning_executions as f64 / total_executions as f64) * 100.0
            } else {
                0.0
            };

            let largest_win = strategy_executions.iter()
                .map(|e| e.pnl)
                .max()
                .unwrap_or(Decimal::ZERO);
            let largest_loss = strategy_executions.iter()
                .map(|e| e.pnl)
                .min()
                .unwrap_or(Decimal::ZERO);

            let performance = StrategyPerformance {
                strategy_id,
                period_start,
                period_end,
                total_return,
                total_return_pct: (total_return.to_f64().unwrap_or(0.0) / 10000.0) * 100.0, // Assuming $10k initial
                annualized_return_pct: annualized_return * 100.0,
                volatility_pct: annualized_volatility * 100.0,
                sharpe_ratio,
                max_drawdown_pct: max_drawdown * 100.0,
                win_rate_pct: win_rate,
                total_trades: total_executions,
                winning_trades: winning_executions,
                losing_trades: losing_executions,
                average_trade_return: if total_executions > 0 {
                    total_return / Decimal::from(total_executions)
                } else {
                    Decimal::ZERO
                },
                largest_win,
                largest_loss,
                calculated_at: Utc::now(),
            };

            // Update cache
            let mut performances = self.performances.write().await;
            performances.insert(strategy_id, performance);

            // Update PnL cache
            let daily_cutoff = Utc::now() - Duration::hours(24);
            let daily_pnl: Decimal = strategy_executions.iter()
                .filter(|e| e.execution_time >= daily_cutoff)
                .map(|e| e.pnl)
                .sum();

            let mut pnl_cache = self.pnl_cache.write().await;
            pnl_cache.insert(strategy_id, (total_return, daily_pnl));
        }

        Ok(())
    }

    async fn calculate_performance_for_period(
        &self,
        strategy_id: Uuid,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Option<StrategyPerformance>> {
        let executions = self.executions.read().await;
        if let Some(strategy_executions) = executions.get(&strategy_id) {
            let filtered_executions: Vec<_> = strategy_executions.iter()
                .filter(|e| e.execution_time >= start && e.execution_time <= end)
                .collect();

            if filtered_executions.is_empty() {
                return Ok(None);
            }

            // Similar calculation as update_performance_metrics but for filtered data
            let total_return: Decimal = filtered_executions.iter().map(|e| e.pnl).sum();
            let total_executions = filtered_executions.len() as u32;
            
            // Simplified metrics for period-specific calculation
            let winning_executions = filtered_executions.iter()
                .filter(|e| e.pnl > Decimal::ZERO)
                .count() as u32;
            let losing_executions = filtered_executions.iter()
                .filter(|e| e.pnl < Decimal::ZERO)
                .count() as u32;

            let win_rate = if total_executions > 0 {
                (winning_executions as f64 / total_executions as f64) * 100.0
            } else {
                0.0
            };

            Ok(Some(StrategyPerformance {
                strategy_id,
                period_start: start,
                period_end: end,
                total_return,
                total_return_pct: (total_return.to_f64().unwrap_or(0.0) / 10000.0) * 100.0,
                annualized_return_pct: 0.0, // Simplified
                volatility_pct: 0.0, // Simplified
                sharpe_ratio: 0.0, // Simplified
                max_drawdown_pct: 0.0, // Simplified
                win_rate_pct: win_rate,
                total_trades: total_executions,
                winning_trades: winning_executions,
                losing_trades: losing_executions,
                average_trade_return: if total_executions > 0 {
                    total_return / Decimal::from(total_executions)
                } else {
                    Decimal::ZERO
                },
                largest_win: filtered_executions.iter()
                    .map(|e| e.pnl)
                    .max()
                    .unwrap_or(Decimal::ZERO),
                largest_loss: filtered_executions.iter()
                    .map(|e| e.pnl)
                    .min()
                    .unwrap_or(Decimal::ZERO),
                calculated_at: Utc::now(),
            }))
        } else {
            Ok(None)
        }
    }
}