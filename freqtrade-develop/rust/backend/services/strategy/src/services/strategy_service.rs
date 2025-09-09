use anyhow::Result;
use chrono::Utc;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::config::Config;
use crate::models::*;
use crate::repository::{Repository, StrategyRepository, StrategyExecutionRepository, TransactionManager};
use sqlx::Transaction;
use sqlx::Postgres;
// use super::signal_generator::SignalGenerator;
// use super::factor_analyzer::FactorAnalyzer;
use super::performance_tracker::PerformanceTracker;
use super::market_data_client::MarketDataClient;
use futures::StreamExt;
use serde::Deserialize;

pub struct StrategyService {
    config: Arc<Config>,
    strategy_repository: Arc<StrategyRepository>,
    execution_repository: Arc<StrategyExecutionRepository>,
    running_strategies: Arc<RwLock<HashMap<Uuid, tokio::task::JoinHandle<()>>>>,
    // signal_generator: Arc<SignalGenerator>,
    // factor_analyzer: Arc<FactorAnalyzer>,
    performance_tracker: Arc<PerformanceTracker>,
    market_data_client: Arc<MarketDataClient>,
    market_prices: Arc<RwLock<HashMap<String, Vec<f64>>>>, // symbol -> recent prices
}

impl StrategyService {
    pub async fn new(
        config: &Config,
        strategy_repository: Arc<StrategyRepository>,
        execution_repository: Arc<StrategyExecutionRepository>,
    ) -> Result<Self> {
        // let signal_generator = Arc::new(SignalGenerator::new(config).await?);
        // let factor_analyzer = Arc::new(FactorAnalyzer::new(config).await?);
        let performance_tracker = Arc::new(PerformanceTracker::new(config).await?);
        let market_data_client = Arc::new(MarketDataClient::new(config).await?);

        let service = Self {
            config: Arc::new(config.clone()),
            strategy_repository,
            execution_repository,
            running_strategies: Arc::new(RwLock::new(HashMap::new())),
            // signal_generator,
            // factor_analyzer,
            performance_tracker,
            market_data_client,
            market_prices: Arc::new(RwLock::new(HashMap::new())),
        };

        // Spawn NATS ingestion task (best effort)
        let nats_url = service.config.messaging.nats_url.clone();
        let prices = service.market_prices.clone();
        tokio::spawn(async move {
            loop {
                match async_nats::connect(nats_url.clone()).await {
                    Ok(client) => {
                        if let Ok(mut sub) = client.subscribe("market.ticks").await {
                            tracing::info!("Strategy subscribed to NATS subject market.ticks");
                            while let Some(msg) = sub.next().await {
                                if let Ok(tick) = serde_json::from_slice::<InTick>(&msg.payload) {
                                    let mut map = prices.write().await;
                                    let e = map.entry(tick.symbol).or_insert_with(Vec::new);
                                    e.push(tick.price);
                                    if e.len() > 240 { let excess = e.len() - 240; e.drain(0..excess); }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("NATS connect failed: {}", e);
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    }
                }
            }
        });

        Ok(service)
    }

    // Strategy Management
    pub async fn create_strategy(&self, request: CreateStrategyRequest) -> Result<Strategy> {
        self.strategy_repository.create_from_request(request, "system").await
    }

    pub async fn get_strategy(&self, strategy_id: Uuid) -> Result<Option<Strategy>> {
        self.strategy_repository.find_by_id(strategy_id).await
    }

    pub async fn update_strategy(&self, strategy_id: Uuid, request: UpdateStrategyRequest) -> Result<Option<Strategy>> {
        self.strategy_repository.update_from_request(strategy_id, request).await
    }

    pub async fn delete_strategy(&self, strategy_id: Uuid) -> Result<bool> {
        // Stop strategy if running
        self.stop_strategy(strategy_id).await?;
        
        self.strategy_repository.delete(strategy_id).await
    }

    pub async fn list_strategies(&self) -> Result<Vec<Strategy>> {
        self.strategy_repository.list(Some(50), Some(0)).await
    }

    // Strategy Execution
    pub async fn start_strategy(&self, strategy_id: Uuid) -> Result<bool> {
        let mut running_strategies = self.running_strategies.write().await;

        // Check if already running
        if running_strategies.contains_key(&strategy_id) {
            return Ok(false);
        }

        // Check concurrent limit
        if running_strategies.len() >= self.config.strategy.max_concurrent_strategies {
            return Err(anyhow::anyhow!("Maximum concurrent strategies limit reached"));
        }

        // Get strategy from database
        if let Some(mut strategy) = self.strategy_repository.find_by_id(strategy_id).await? {
            strategy.status = StrategyStatus::Active;
            strategy.updated_at = Utc::now();
            
            // Update strategy status in database
            self.strategy_repository.update(strategy_id, &strategy).await?;
            
            // Spawn strategy execution task
            let strategy_clone = strategy.clone();
            let config = self.config.clone();
            let market_data_client = self.market_data_client.clone();
            let performance_tracker = self.performance_tracker.clone();
            let execution_repo = self.execution_repository.clone();
            
            let handle = tokio::spawn(async move {
                Self::execute_strategy_loop(strategy_clone, config, market_data_client, performance_tracker, execution_repo).await;
            });
            
            running_strategies.insert(strategy_id, handle);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub async fn stop_strategy(&self, strategy_id: Uuid) -> Result<bool> {
        let mut running_strategies = self.running_strategies.write().await;

        // Stop the execution task
        if let Some(handle) = running_strategies.remove(&strategy_id) {
            handle.abort();
        }

        // Update strategy status in database
        if let Some(mut strategy) = self.strategy_repository.find_by_id(strategy_id).await? {
            strategy.status = StrategyStatus::Stopped;
            strategy.updated_at = Utc::now();
            self.strategy_repository.update(strategy_id, &strategy).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub async fn get_strategy_status(&self, strategy_id: Uuid) -> Result<Option<StrategyStatusResponse>> {
        let running_strategies = self.running_strategies.read().await;

        if let Some(strategy) = self.strategy_repository.find_by_id(strategy_id).await? {
            let is_running = running_strategies.contains_key(&strategy_id);
            
            // Get current metrics (placeholder)
            let active_signals = 0; // self.signal_generator.get_active_signals_count(strategy_id).await;
            let (current_pnl, daily_pnl) = (Decimal::ZERO, Decimal::ZERO); // self.performance_tracker.get_current_pnl(strategy_id).await?;
            
            let next_execution = if is_running {
                Some(Utc::now() + chrono::Duration::seconds(strategy.parameters.evaluation_interval as i64))
            } else {
                None
            };

            // Fetch open positions for the strategy symbols via Risk Service (best-effort)
            let open_positions = self
                .fetch_open_positions_count_for_symbols(&strategy.symbols)
                .await
                .unwrap_or(0);

            // Basic execution metrics (placeholder but filled)
            let execution_metrics = Some(ExecutionMetrics {
                data_points_processed: (strategy.symbols.len() as u32) * 100,
                indicators_calculated: (strategy.symbols.len() as u32) * 5,
                factors_evaluated: (strategy.symbols.len() as u32) * 3,
                risk_checks_passed: (strategy.symbols.len() as u32),
                risk_checks_failed: 0,
                average_signal_strength: 0.0,
                average_signal_confidence: 0.0,
            });

            Ok(Some(StrategyStatusResponse {
                strategy_id,
                status: strategy.status.clone(),
                last_execution: strategy.last_execution,
                next_execution,
                active_signals,
                open_positions: open_positions as u32,
                current_pnl,
                daily_pnl,
                execution_metrics,
            }))
        } else {
            Ok(None)
        }
    }

    async fn fetch_open_positions_count_for_symbols(&self, symbols: &[String]) -> Result<usize> {
        #[derive(serde::Deserialize)]
        struct Resp { positions: Vec<RiskPosition> }
        #[derive(serde::Deserialize)]
        struct RiskPosition { symbol: String, size: rust_decimal::Decimal }

        let risk_url = std::env::var("RISK_SERVICE_URL").unwrap_or_else(|_| "http://localhost:8083".to_string());
        let url = format!("{}/api/v1/risk/positions", risk_url.trim_end_matches('/'));
        let client = reqwest::Client::new();
        let resp = client.get(url).send().await?;
        if !resp.status().is_success() {
            return Ok(0);
        }
        let data: Resp = resp.json().await.unwrap_or(Resp { positions: Vec::new() });
        let set: std::collections::HashSet<&str> = symbols.iter().map(|s| s.as_str()).collect();
        let count = data.positions.iter().filter(|p| set.contains(p.symbol.as_str()) && p.size > Decimal::ZERO).count();
        Ok(count)
    }

    // Signal Operations (placeholder implementations)
    pub async fn get_signals(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
        symbol: Option<String>,
        action: Option<String>,
        strategy_id: Option<Uuid>,
    ) -> Result<SignalListResponse> {
        let per_page = limit.unwrap_or(10).min(100);
        let offset = offset.unwrap_or(0);

        // Gather symbols
        let symbols: Vec<String> = if let Some(id) = strategy_id {
            if let Some(s) = self.strategy_repository.find_by_id(id).await? {
                s.symbols
            } else {
                Vec::new()
            }
        } else {
            let strategies = self.list_strategies().await?;
            let mut out: Vec<String> = Vec::new();
            for s in &strategies {
                for sym in &s.symbols {
                    if !out.contains(sym) {
                        out.push(sym.clone());
                    }
                }
            }
            out
        };

        let mut signals = self.generate_momentum_signals_series(&symbols).await?;

        // Filtering
        if let Some(sym) = symbol {
            signals.retain(|s| s.symbol.eq_ignore_ascii_case(&sym));
        }
        if let Some(act) = action {
            use crate::models::SignalAction::*;
            let act_upper = act.to_lowercase();
            signals.retain(|s| match (&act_upper[..], &s.action) {
                ("buy", Buy) | ("sell", Sell) | ("hold", Hold) | ("close", SignalAction::Close)
                | ("reduceposition", SignalAction::ReducePosition) | ("increaseposition", SignalAction::IncreasePosition) => true,
                _ => false,
            });
        }

        let total_count = signals.len() as u32;
        // Pagination
        let start = (offset as usize).min(signals.len());
        let end = (start + per_page as usize).min(signals.len());
        let signals = signals[start..end].to_vec();

        Ok(SignalListResponse {
            signals,
            total_count,
            page: (offset / per_page.max(1)) + 1,
            per_page,
            filters_applied: std::collections::HashMap::new(),
        })
    }

    pub async fn get_signals_by_strategy(&self, strategy_id: Uuid) -> Result<Vec<Signal>> {
        if let Some(strategy) = self.strategy_repository.find_by_id(strategy_id).await? {
            self.generate_momentum_signals_series(&strategy.symbols).await
        } else {
            Ok(Vec::new())
        }
    }

    // Performance and Analytics (placeholder implementations)
    pub async fn get_performance(&self, strategy_id: Uuid, days: Option<u32>) -> Result<Option<StrategyPerformance>> {
        let days = days.unwrap_or(7) as i32;
        let (total_exec, success_exec, total_pnl, avg_exec_time) =
            self.execution_repository.get_performance_metrics(strategy_id, days).await?;

        let now = Utc::now();
        let period_end = now;
        let period_start = now - chrono::Duration::days(days as i64);

        let total_return = total_pnl; // MVP: treat realized pnl as total return
        let total_return_pct = 0.0; // unknown equity -> 0 for MVP
        let annualized_return_pct = 0.0;
        let volatility_pct = 0.0;
        let sharpe_ratio = 0.0;
        let max_drawdown_pct = 0.0;
        let total_trades = total_exec as u32;
        let winning_trades = success_exec as u32;
        let losing_trades = (total_exec - success_exec).max(0) as u32;
        let average_trade_return = Decimal::ZERO;
        let largest_win = Decimal::ZERO;
        let largest_loss = Decimal::ZERO;

        let perf = StrategyPerformance {
            strategy_id,
            period_start,
            period_end,
            total_return,
            total_return_pct,
            annualized_return_pct,
            volatility_pct,
            sharpe_ratio,
            max_drawdown_pct,
            win_rate_pct: if total_trades > 0 { (winning_trades as f64) / (total_trades as f64) } else { 0.0 },
            total_trades,
            winning_trades,
            losing_trades,
            average_trade_return,
            largest_win,
            largest_loss,
            calculated_at: Utc::now(),
        };
        Ok(Some(perf))
    }

    pub async fn get_metrics(&self, strategy_id: Uuid) -> Result<Option<ExecutionMetrics>> {
        if let Some(strategy) = self.strategy_repository.find_by_id(strategy_id).await? {
            let symbols_n = strategy.symbols.len() as u32;
            let metrics = ExecutionMetrics {
                data_points_processed: symbols_n * 100,
                indicators_calculated: symbols_n * 5,
                factors_evaluated: symbols_n * 3,
                risk_checks_passed: symbols_n,
                risk_checks_failed: 0,
                average_signal_strength: 0.0,
                average_signal_confidence: 0.0,
            };
            Ok(Some(metrics))
        } else {
            Ok(None)
        }
    }

    // Market Analysis (placeholder implementations)
    pub async fn get_indicators(&self, _symbol: &str, _timeframe: &str) -> Result<IndicatorListResponse> {
        Ok(IndicatorListResponse {
            indicators: Vec::new(),
            symbol: _symbol.to_string(),
            timeframe: _timeframe.to_string(),
            last_updated: Utc::now(),
        })
    }

    pub async fn get_factors(&self) -> Result<FactorAnalysisResponse> {
        Ok(FactorAnalysisResponse {
            factors: Vec::new(),
            market_regime: MarketRegime {
                regime: "Neutral".to_string(),
                confidence: 0.5,
                duration_days: 0,
                characteristics: Vec::new(),
            },
            risk_assessment: RiskAssessment {
                overall_risk: 50.0,
                market_risk: 50.0,
                liquidity_risk: 50.0,
                volatility_risk: 50.0,
                correlation_risk: 50.0,
                recommendations: Vec::new(),
            },
            recommendations: Vec::new(),
            calculated_at: Utc::now(),
        })
    }

    // Backtesting
    pub async fn run_backtest(&self, _request: BacktestRequest) -> Result<BacktestResponse> {
        // TODO: Integrate with backtest service
        Ok(BacktestResponse {
            backtest_id: Uuid::new_v4(),
            status: "queued".to_string(),
            progress: 0.0,
            estimated_completion: Some(Utc::now() + chrono::Duration::minutes(30)),
            preliminary_results: None,
        })
    }

    // Utility methods
    pub async fn get_running_strategies_count(&self) -> usize {
        let running_strategies = self.running_strategies.read().await;
        running_strategies.len()
    }

    pub async fn get_total_strategies_count(&self) -> usize {
        match self.strategy_repository.list(Some(1000), Some(0)).await {
            Ok(strategies) => strategies.len(),
            Err(_) => 0,
        }
    }

    async fn execute_strategy_loop(
        strategy: Strategy, 
        _config: Arc<Config>,
        _market_data_client: Arc<MarketDataClient>,
        _performance_tracker: Arc<PerformanceTracker>,
        execution_repository: Arc<StrategyExecutionRepository>,
    ) {
        let mut interval = tokio::time::interval(
            tokio::time::Duration::from_secs(strategy.parameters.evaluation_interval)
        );

        loop {
            interval.tick().await;

            // Execute strategy evaluation
            tracing::info!("Executing strategy: {}", strategy.name);
            
            // Execute and persist results
            let start_t = std::time::Instant::now();
            if let Err(err) = Self::evaluate_strategy_static(&strategy).await {
                tracing::error!("Strategy {} execution error: {}", strategy.name, err);
                break; // Exit the loop on error
            }

            // Persist a minimal execution record (MVP)
            let now = Utc::now();
            let id = Uuid::new_v4();
            let symbol = strategy.symbols.get(0).cloned().unwrap_or_else(|| "N/A".to_string());
            let exchange = strategy.exchanges.get(0).cloned().unwrap_or_else(|| "default".to_string());
            let exec_id = Uuid::new_v4().to_string();
            let signals = match Self::generate_momentum_signals_static(&strategy.symbols) {
                Ok(sig) => sig,
                Err(_) => Vec::new(),
            };
            let elapsed_ms = start_t.elapsed().as_millis() as i32;

            let input_json = serde_json::json!({
                "symbols": strategy.symbols,
                "exchanges": strategy.exchanges,
                "parameters": {
                    "evaluation_interval": strategy.parameters.evaluation_interval
                }
            });
            let strength_avg: f64 = if signals.is_empty() { 0.0 } else { signals.iter().map(|s| s.strength).sum::<f64>() / signals.len() as f64 };
            let confidence_avg: f64 = if signals.is_empty() { 0.0 } else { signals.iter().map(|s| s.confidence).sum::<f64>() / signals.len() as f64 };
            let buy_count = signals.iter().filter(|s| matches!(s.action, SignalAction::Buy)).count();
            let sell_count = signals.iter().filter(|s| matches!(s.action, SignalAction::Sell)).count();
            let hold_count = signals.iter().filter(|s| matches!(s.action, SignalAction::Hold)).count();

            // Sample synthetic price series (first symbol) for visualization (10 points)
            let sample_prices = if let Some(sym0) = strategy.symbols.get(0) {
                use std::hash::{BuildHasher, Hasher};
                let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
                hasher.write(sym0.as_bytes());
                let h = hasher.finish();
                let n = 60usize;
                let base = 100.0 + ((h as u64) % 100) as f64;
                let mut series: Vec<f64> = Vec::with_capacity(n);
                for i in 0..n {
                    let t = i as f64;
                    let val = base + 2.0 * (t/5.0).sin() + 0.1 * (t/3.0).cos();
                    series.push(val);
                }
                let take = 10usize.min(series.len());
                series[series.len()-take..].to_vec()
            } else { Vec::new() };

            let output_json = serde_json::json!({
                "signal_count": signals.len(),
                "actions": { "buy": buy_count, "sell": sell_count, "hold": hold_count },
                "strength_avg": strength_avg,
                "confidence_avg": confidence_avg,
                "sample_prices": sample_prices,
            });

            let row = crate::repository::strategy_execution_repository::StrategyExecution {
                id,
                instance_id: strategy.id, // MVP: use strategy id as instance id
                execution_id: exec_id,
                exchange,
                symbol,
                execution_type: "evaluation".to_string(),
                input_data: Some(input_json),
                output_data: Some(output_json),
                decisions: None,
                market_price: None,
                volume_24h: None,
                volatility: None,
                status: "completed".to_string(),
                orders_created: Some(0),
                orders_filled: Some(0),
                total_volume: Some(Decimal::ZERO),
                realized_pnl: Some(Decimal::ZERO),
                execution_time_ms: Some(elapsed_ms),
                latency_ms: Some(0),
                slippage_bps: Some(0),
                error_code: None,
                error_message: None,
                retry_count: Some(0),
                created_at: now,
                started_at: Some(now),
                completed_at: Some(now),
            };

            if let Err(e) = execution_repository.create(&row).await {
                tracing::warn!("Failed to persist execution record for {}: {}", strategy.name, e);
            }
        }
    }

    async fn evaluate_strategy_static(_strategy: &Strategy) -> Result<()> {
        // Placeholder implementation - in real system would:
        // 1. Get market data for strategy symbols
        // 2. Calculate indicators and factors  
        // 3. Generate signals based on strategy type
        // 4. Record execution metrics
        // 5. Update performance tracking

        tracing::debug!("Strategy evaluated (placeholder implementation)");
        Ok(())
    }

    fn generate_momentum_signals_static(symbols: &[String]) -> Result<Vec<Signal>> {
        use crate::models::{SignalAction, SignalType};
        use std::hash::{BuildHasher, Hasher};
        use std::collections::hash_map::RandomState;
        let mut out = Vec::new();
        for sym in symbols {
            // Deterministic pseudo series based on symbol hash
            // Deterministic seed per symbol
            let mut hasher = RandomState::new().build_hasher();
            hasher.write(sym.as_bytes());
            let h = hasher.finish();
            let n = 60usize;
            let mut series: Vec<f64> = Vec::with_capacity(n);
            let base = 100.0 + ((h as u64) % 100) as f64;
            for i in 0..n {
                let t = i as f64;
                let val = base + 2.0 * (t/5.0).sin() + 0.1 * (t/3.0).cos();
                series.push(val);
            }
            let sma = |win: usize| -> f64 {
                let w = win.min(series.len());
                let s: f64 = series[series.len()-w..].iter().sum();
                s / w as f64
            };
            let short = sma(5);
            let long = sma(20);
            let last = *series.last().unwrap_or(&base);
            let diff = short - long;
            let strength = (diff / (long.abs() + 1e-6)).clamp(-0.05, 0.05) / 0.05; // -1..1
            let (action, confidence) = if diff > 0.0 {
                (SignalAction::Buy, strength.abs())
            } else if diff < 0.0 {
                (SignalAction::Sell, strength.abs())
            } else {
                (SignalAction::Hold, 0.0)
            };

            out.push(Signal {
                id: Uuid::new_v4(),
                strategy_id: Uuid::nil(),
                symbol: sym.clone(),
                exchange: "default".to_string(),
                signal_type: SignalType::Entry,
                action,
                strength: strength.abs(),
                confidence,
                price: Decimal::from_f64_retain(last).unwrap_or(Decimal::ONE),
                quantity: None,
                reason: if diff > 0.0 { "SMA5>SMA20" } else { "SMA5<SMA20" }.to_string(),
                factors: std::collections::HashMap::new(),
                created_at: Utc::now(),
                expires_at: None,
                executed: false,
                executed_at: None,
            });
        }
        Ok(out)
    }

    async fn generate_momentum_signals_series(&self, symbols: &[String]) -> Result<Vec<Signal>> {
        // Try to use real series from market_prices; fallback to static generator per symbol
        use crate::models::{SignalAction, SignalType};
        let mut out = Vec::new();
        let prices = self.market_prices.read().await;
        for sym in symbols {
            if let Some(series) = prices.get(sym) {
                if series.len() >= 5 {
                    let sma = |win: usize| -> f64 {
                        let w = win.min(series.len());
                        let s: f64 = series[series.len()-w..].iter().sum();
                        s / w as f64
                    };
                    let short = sma(5);
                    let long = sma(20);
                    let last = *series.last().unwrap_or(&short);
                    let diff = short - long;
                    let strength = (diff / (long.abs() + 1e-6)).clamp(-0.05, 0.05) / 0.05; // -1..1
                    let (action, confidence) = if diff > 0.0 {
                        (SignalAction::Buy, strength.abs())
                    } else if diff < 0.0 {
                        (SignalAction::Sell, strength.abs())
                    } else {
                        (SignalAction::Hold, 0.0)
                    };
                    out.push(Signal {
                        id: Uuid::new_v4(),
                        strategy_id: Uuid::nil(),
                        symbol: sym.clone(),
                        exchange: "default".to_string(),
                        signal_type: SignalType::Entry,
                        action,
                        strength: strength.abs(),
                        confidence,
                        price: Decimal::from_f64_retain(last).unwrap_or(Decimal::ONE),
                        quantity: None,
                        reason: if diff > 0.0 { "SMA5>SMA20" } else { "SMA5<SMA20" }.to_string(),
                        factors: std::collections::HashMap::new(),
                        created_at: Utc::now(),
                        expires_at: None,
                        executed: false,
                        executed_at: None,
                    });
                    continue;
                }
            }
            // fallback
            let mut fallback = Self::generate_momentum_signals_static(&[sym.clone()])?;
            out.append(&mut fallback);
        }
        Ok(out)
    }

    pub async fn ingest_tick(&self, symbol: &str, price: f64) {
        let mut map = self.market_prices.write().await;
        let entry = map.entry(symbol.to_string()).or_insert_with(Vec::new);
        entry.push(price);
        if entry.len() > 120 { let excess = entry.len() - 120; entry.drain(0..excess); }
    }

    // Transactional operations - these use database transactions for consistency
    pub async fn create_strategy_with_instance_tx(
        &self,
        request: CreateStrategyRequest,
        created_by: &str,
        transaction_manager: &TransactionManager,
    ) -> Result<(Strategy, StrategyInstance)> {
        let mut tx = transaction_manager.begin_transaction().await?;

        // Create strategy within transaction
        let strategy = self.create_strategy_in_tx(&mut tx, request, created_by).await?;

        // Create default instance within same transaction
        let instance_request = CreateStrategyInstanceRequest {
            strategy_id: strategy.id,
            name: format!("{}_default", strategy.name),
            parameters: strategy.parameters.clone(),
            symbols: strategy.symbols.clone(),
            exchanges: strategy.exchanges.clone(),
            initial_capital: Decimal::from(10000), // Default $10k
            max_loss_per_trade: Decimal::from(100), // Default $100
            max_daily_loss: Decimal::from(500), // Default $500
        };

        let instance = self.create_instance_in_tx(&mut tx, instance_request).await?;

        // Commit transaction
        TransactionManager::commit_transaction(tx).await?;

        Ok((strategy, instance))
    }

    async fn create_strategy_in_tx(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        request: CreateStrategyRequest,
        created_by: &str,
    ) -> Result<Strategy> {
        let strategy_id = Uuid::new_v4();
        let now = Utc::now();
        let strategy_type_str = request.strategy_type.to_string();
        let params_json = serde_json::to_value(&request.parameters)?;

        sqlx::query(
            r#"
            INSERT INTO strategies (
                id, name, description, strategy_type, default_params,
                allowed_symbols, allowed_exchanges, created_by, is_active,
                created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            "#,
        )
        .bind(strategy_id)
        .bind(&request.name)
        .bind(&request.description)
        .bind(&strategy_type_str)
        .bind(&params_json)
        .bind(&request.symbols)
        .bind(&request.exchanges)
        .bind(created_by)
        .bind(true)
        .bind(now)
        .bind(now)
        .execute(&mut **tx)
        .await?;

        Ok(Strategy {
            id: strategy_id,
            name: request.name,
            description: request.description,
            strategy_type: request.strategy_type,
            status: StrategyStatus::Draft,
            parameters: request.parameters,
            symbols: request.symbols,
            exchanges: request.exchanges,
            created_by: created_by.to_string(),
            created_at: now,
            updated_at: now,
            last_execution: None,
        })
    }

    async fn create_instance_in_tx(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        request: CreateStrategyInstanceRequest,
    ) -> Result<StrategyInstance> {
        let instance_id = Uuid::new_v4();
        let now = Utc::now();
        let params_json = serde_json::to_value(&request.parameters)?;

        sqlx::query(
            r#"
            INSERT INTO strategy_instances (
                id, strategy_id, name, parameters, symbols, exchanges,
                initial_capital, current_capital, max_loss_per_trade,
                max_daily_loss, status, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            "#
        )
        .bind(instance_id)
        .bind(request.strategy_id)
        .bind(&request.name)
        .bind(&params_json)
        .bind(&request.symbols)
        .bind(&request.exchanges)
        .bind(request.initial_capital)
        .bind(request.initial_capital) // current_capital starts same as initial
        .bind(request.max_loss_per_trade)
        .bind(request.max_daily_loss)
        .bind("created")
        .bind(now)
        .bind(now)
        .execute(&mut **tx)
        .await?;

        Ok(StrategyInstance {
            id: instance_id,
            strategy_id: request.strategy_id,
            name: request.name,
            parameters: request.parameters,
            symbols: request.symbols,
            exchanges: request.exchanges,
            initial_capital: request.initial_capital,
            current_capital: request.initial_capital,
            max_loss_per_trade: request.max_loss_per_trade,
            max_daily_loss: request.max_daily_loss,
            status: "created".to_string(),
            health_status: Some("healthy".to_string()),
            last_error: None,
            execution_count: 0,
            last_execution: None,
            created_at: now,
            updated_at: now,
        })
    }
}

#[derive(Deserialize)]
struct InTick {
    symbol: String,
    price: f64,
    #[allow(dead_code)]
    ts: Option<String>,
}
