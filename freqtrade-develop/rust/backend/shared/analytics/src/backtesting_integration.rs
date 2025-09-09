use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// 回测集成模块
/// 提供与回测系统的集成接口
#[derive(Debug, Clone)]
pub struct BacktestingIntegration {
    config: BacktestConfig,
    results_cache: HashMap<String, BacktestResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub initial_capital: f64,
    pub commission_rate: f64,
    pub slippage_model: SlippageModel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlippageModel {
    Fixed(f64),
    Linear(f64),
    Square(f64),
    Market(MarketSlippageParams),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSlippageParams {
    pub base_slippage: f64,
    pub volume_impact: f64,
    pub volatility_impact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub strategy_id: String,
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub trades: Vec<Trade>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub side: TradeSide,
    pub quantity: f64,
    pub price: f64,
    pub commission: f64,
    pub slippage: f64,
    pub pnl: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

impl BacktestingIntegration {
    pub fn new(config: BacktestConfig) -> Self {
        Self {
            config,
            results_cache: HashMap::new(),
        }
    }

    pub fn run_backtest(&mut self, strategy_id: &str, signals: Vec<Signal>) -> Result<BacktestResult> {
        // 简化的回测逻辑
        let mut trades = Vec::new();
        let mut capital = self.config.initial_capital;
        let mut position = 0.0;
        let mut total_pnl = 0.0;
        let mut wins = 0;
        let mut losses = 0;

        for signal in signals {
            let trade = self.execute_signal(signal, &mut capital, &mut position)?;
            if trade.pnl > 0.0 {
                wins += 1;
            } else if trade.pnl < 0.0 {
                losses += 1;
            }
            total_pnl += trade.pnl;
            trades.push(trade);
        }

        let total_return = total_pnl / self.config.initial_capital;
        let win_rate = if wins + losses > 0 {
            wins as f64 / (wins + losses) as f64
        } else {
            0.0
        };

        let result = BacktestResult {
            strategy_id: strategy_id.to_string(),
            total_return,
            sharpe_ratio: self.calculate_sharpe_ratio(&trades),
            max_drawdown: self.calculate_max_drawdown(&trades),
            win_rate,
            profit_factor: self.calculate_profit_factor(&trades),
            trades,
        };

        self.results_cache.insert(strategy_id.to_string(), result.clone());
        Ok(result)
    }

    fn execute_signal(&self, signal: Signal, capital: &mut f64, position: &mut f64) -> Result<Trade> {
        let commission = signal.quantity * signal.price * self.config.commission_rate;
        let slippage = self.calculate_slippage(signal.quantity, signal.price);
        
        let effective_price = match signal.side {
            TradeSide::Buy => signal.price + slippage,
            TradeSide::Sell => signal.price - slippage,
        };

        let cost = signal.quantity * effective_price + commission;
        
        let pnl = match signal.side {
            TradeSide::Buy => {
                *capital -= cost;
                *position += signal.quantity;
                0.0 // PnL calculated on close
            }
            TradeSide::Sell => {
                let revenue = signal.quantity * effective_price - commission;
                *capital += revenue;
                let pnl = revenue - (*position * signal.price);
                *position -= signal.quantity;
                pnl
            }
        };

        Ok(Trade {
            timestamp: signal.timestamp,
            symbol: signal.symbol,
            side: signal.side,
            quantity: signal.quantity,
            price: effective_price,
            commission,
            slippage,
            pnl,
        })
    }

    fn calculate_slippage(&self, quantity: f64, price: f64) -> f64 {
        match &self.config.slippage_model {
            SlippageModel::Fixed(s) => *s,
            SlippageModel::Linear(factor) => quantity * factor,
            SlippageModel::Square(factor) => quantity * quantity * factor,
            SlippageModel::Market(params) => {
                params.base_slippage + 
                params.volume_impact * quantity.ln() +
                params.volatility_impact * price * 0.01
            }
        }
    }

    fn calculate_sharpe_ratio(&self, trades: &[Trade]) -> f64 {
        if trades.is_empty() {
            return 0.0;
        }

        let returns: Vec<f64> = trades.iter().map(|t| t.pnl).collect();
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        if variance > 0.0 {
            mean_return / variance.sqrt()
        } else {
            0.0
        }
    }

    fn calculate_max_drawdown(&self, trades: &[Trade]) -> f64 {
        let mut peak = 0.0;
        let mut max_dd = 0.0;
        let mut cumulative_pnl = 0.0;

        for trade in trades {
            cumulative_pnl += trade.pnl;
            if cumulative_pnl > peak {
                peak = cumulative_pnl;
            }
            let drawdown = (peak - cumulative_pnl) / peak.max(1.0);
            if drawdown > max_dd {
                max_dd = drawdown;
            }
        }

        max_dd
    }

    fn calculate_profit_factor(&self, trades: &[Trade]) -> f64 {
        let gross_profit: f64 = trades.iter()
            .filter(|t| t.pnl > 0.0)
            .map(|t| t.pnl)
            .sum();
        
        let gross_loss: f64 = trades.iter()
            .filter(|t| t.pnl < 0.0)
            .map(|t| t.pnl.abs())
            .sum();

        if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else {
            f64::INFINITY
        }
    }

    pub fn get_cached_result(&self, strategy_id: &str) -> Option<&BacktestResult> {
        self.results_cache.get(strategy_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub side: TradeSide,
    pub quantity: f64,
    pub price: f64,
}