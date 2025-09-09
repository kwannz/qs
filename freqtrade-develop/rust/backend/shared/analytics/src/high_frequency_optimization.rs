use anyhow::{Result, Context};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc, Duration};
use ndarray::{Array1, Array2};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HFOptimizationConfig {
    pub optimization_window: Duration,
    pub signal_decay_halflife: Duration,
    pub latency_threshold_ms: u64,
    pub min_signal_strength: f64,
    pub max_position_size: f64,
    pub inventory_target: f64,
    pub risk_limit: f64,
    pub tick_size: f64,
    pub lot_size: f64,
    pub max_orders_per_second: u64,
    pub adverse_selection_penalty: f64,
    pub momentum_decay_factor: f64,
    pub mean_reversion_strength: f64,
}

impl Default for HFOptimizationConfig {
    fn default() -> Self {
        Self {
            optimization_window: Duration::minutes(5),
            signal_decay_halflife: Duration::seconds(30),
            latency_threshold_ms: 10,
            min_signal_strength: 0.01,
            max_position_size: 1000000.0,
            inventory_target: 0.0,
            risk_limit: 50000.0,
            tick_size: 0.01,
            lot_size: 100.0,
            max_orders_per_second: 100,
            adverse_selection_penalty: 0.5,
            momentum_decay_factor: 0.95,
            mean_reversion_strength: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMicrostructureSignal {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub signal_type: String,
    pub strength: f64,
    pub confidence: f64,
    pub expected_duration: Duration,
    pub price_impact_estimate: f64,
    pub urgency_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookState {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub mid_price: f64,
    pub spread: f64,
    pub imbalance: f64,
    pub depth_ratio: f64,
    pub last_trade_price: f64,
    pub last_trade_size: f64,
    pub book_pressure: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationTarget {
    pub symbol: String,
    pub target_position: f64,
    pub max_participation_rate: f64,
    pub time_horizon: Duration,
    pub urgency: f64,
    pub risk_tolerance: f64,
    pub cost_penalty: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSlice {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub side: String, // "buy" or "sell"
    pub quantity: f64,
    pub price: f64,
    pub urgency: f64,
    pub expected_fill_rate: f64,
    pub adverse_selection_risk: f64,
    pub market_impact_cost: f64,
}

pub struct SignalProcessor {
    config: HFOptimizationConfig,
    signal_history: Arc<RwLock<HashMap<String, VecDeque<MarketMicrostructureSignal>>>>,
    momentum_tracker: Arc<RwLock<HashMap<String, MomentumState>>>,
}

#[derive(Debug, Clone)]
struct MomentumState {
    price_momentum: f64,
    volume_momentum: f64,
    spread_momentum: f64,
    last_update: DateTime<Utc>,
}

impl SignalProcessor {
    pub fn new(config: HFOptimizationConfig) -> Self {
        Self {
            config,
            signal_history: Arc::new(RwLock::new(HashMap::new())),
            momentum_tracker: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn process_orderbook_update(
        &self,
        orderbook: &OrderBookState,
    ) -> Result<Vec<MarketMicrostructureSignal>> {
        let mut signals = Vec::new();

        // Update momentum state
        self.update_momentum_state(orderbook).await?;

        // Generate microstructure signals
        if let Some(imbalance_signal) = self.detect_order_flow_imbalance(orderbook).await? {
            signals.push(imbalance_signal);
        }

        if let Some(momentum_signal) = self.detect_momentum_signal(orderbook).await? {
            signals.push(momentum_signal);
        }

        if let Some(mean_reversion_signal) = self.detect_mean_reversion_signal(orderbook).await? {
            signals.push(mean_reversion_signal);
        }

        if let Some(liquidity_signal) = self.detect_liquidity_signal(orderbook).await? {
            signals.push(liquidity_signal);
        }

        // Store signals
        let mut history = self.signal_history.write().await;
        let symbol_history = history.entry(orderbook.symbol.clone())
            .or_insert_with(VecDeque::new);

        for signal in &signals {
            symbol_history.push_back(signal.clone());
        }

        // Clean old signals
        let cutoff_time = Utc::now() - self.config.optimization_window;
        symbol_history.retain(|s| s.timestamp > cutoff_time);

        Ok(signals)
    }

    async fn update_momentum_state(&self, orderbook: &OrderBookState) -> Result<()> {
        let mut tracker = self.momentum_tracker.write().await;
        let mut momentum_state = tracker.entry(orderbook.symbol.clone())
            .or_insert_with(|| MomentumState {
                price_momentum: 0.0,
                volume_momentum: 0.0,
                spread_momentum: 0.0,
                last_update: orderbook.timestamp,
            });

        let time_diff = (orderbook.timestamp - momentum_state.last_update)
            .num_milliseconds() as f64 / 1000.0;

        if time_diff > 0.0 {
            let decay = (-time_diff / 30.0).exp(); // 30-second decay
            
            // Update price momentum
            let price_change = orderbook.mid_price;
            momentum_state.price_momentum = momentum_state.price_momentum * decay + price_change * (1.0 - decay);

            // Update volume momentum
            let volume_signal = (orderbook.bid_size + orderbook.ask_size).ln();
            momentum_state.volume_momentum = momentum_state.volume_momentum * decay + volume_signal * (1.0 - decay);

            // Update spread momentum
            momentum_state.spread_momentum = momentum_state.spread_momentum * decay + orderbook.spread * (1.0 - decay);

            momentum_state.last_update = orderbook.timestamp;
        }

        Ok(())
    }

    async fn detect_order_flow_imbalance(&self, orderbook: &OrderBookState) -> Result<Option<MarketMicrostructureSignal>> {
        let imbalance = orderbook.imbalance;
        
        if imbalance.abs() > self.config.min_signal_strength {
            let signal_strength = imbalance.tanh(); // Bounded signal
            let confidence = imbalance.abs().min(1.0);
            
            let signal = MarketMicrostructureSignal {
                timestamp: orderbook.timestamp,
                symbol: orderbook.symbol.clone(),
                signal_type: "order_flow_imbalance".to_string(),
                strength: signal_strength,
                confidence,
                expected_duration: Duration::seconds(10),
                price_impact_estimate: signal_strength * orderbook.spread * 0.5,
                urgency_score: confidence,
            };
            
            return Ok(Some(signal));
        }

        Ok(None)
    }

    async fn detect_momentum_signal(&self, orderbook: &OrderBookState) -> Result<Option<MarketMicrostructureSignal>> {
        let tracker = self.momentum_tracker.read().await;
        
        if let Some(momentum_state) = tracker.get(&orderbook.symbol) {
            let momentum_strength = momentum_state.price_momentum * self.config.momentum_decay_factor;
            
            if momentum_strength.abs() > self.config.min_signal_strength {
                let signal_strength = momentum_strength.tanh();
                let confidence = momentum_strength.abs().min(1.0);
                
                let signal = MarketMicrostructureSignal {
                    timestamp: orderbook.timestamp,
                    symbol: orderbook.symbol.clone(),
                    signal_type: "momentum".to_string(),
                    strength: signal_strength,
                    confidence,
                    expected_duration: Duration::seconds(20),
                    price_impact_estimate: signal_strength * orderbook.spread * 0.3,
                    urgency_score: confidence * 0.8,
                };
                
                return Ok(Some(signal));
            }
        }

        Ok(None)
    }

    async fn detect_mean_reversion_signal(&self, orderbook: &OrderBookState) -> Result<Option<MarketMicrostructureSignal>> {
        // Simple mean reversion based on spread deviation
        let normal_spread = orderbook.spread / orderbook.mid_price;
        let spread_deviation = normal_spread - 0.001; // Assume 0.1% is normal spread
        
        if spread_deviation.abs() > self.config.min_signal_strength {
            let reversion_strength = -spread_deviation * self.config.mean_reversion_strength;
            let confidence = spread_deviation.abs().min(1.0);
            
            let signal = MarketMicrostructureSignal {
                timestamp: orderbook.timestamp,
                symbol: orderbook.symbol.clone(),
                signal_type: "mean_reversion".to_string(),
                strength: reversion_strength.tanh(),
                confidence,
                expected_duration: Duration::seconds(15),
                price_impact_estimate: reversion_strength.abs() * orderbook.spread * 0.2,
                urgency_score: confidence * 0.6,
            };
            
            return Ok(Some(signal));
        }

        Ok(None)
    }

    async fn detect_liquidity_signal(&self, orderbook: &OrderBookState) -> Result<Option<MarketMicrostructureSignal>> {
        let liquidity_score = (orderbook.bid_size * orderbook.ask_size).sqrt();
        let depth_signal = orderbook.depth_ratio - 1.0; // Deviation from balanced depth
        
        if depth_signal.abs() > self.config.min_signal_strength && liquidity_score > 1000.0 {
            let signal_strength = depth_signal * (liquidity_score / 10000.0).ln();
            let confidence = depth_signal.abs().min(1.0);
            
            let signal = MarketMicrostructureSignal {
                timestamp: orderbook.timestamp,
                symbol: orderbook.symbol.clone(),
                signal_type: "liquidity_provision".to_string(),
                strength: signal_strength.tanh(),
                confidence,
                expected_duration: Duration::seconds(25),
                price_impact_estimate: signal_strength.abs() * orderbook.spread * 0.1,
                urgency_score: confidence * 0.5,
            };
            
            return Ok(Some(signal));
        }

        Ok(None)
    }

    pub async fn get_combined_signal(&self, symbol: &str) -> Result<Option<f64>> {
        let history = self.signal_history.read().await;
        
        if let Some(signals) = history.get(symbol) {
            let mut combined_strength = 0.0;
            let mut total_weight = 0.0;
            let now = Utc::now();
            
            for signal in signals {
                let age = now - signal.timestamp;
                let age_seconds = age.num_milliseconds() as f64 / 1000.0;
                
                // Exponential decay based on signal age
                let decay_factor = (-age_seconds / self.config.signal_decay_halflife.num_seconds() as f64).exp();
                let weight = signal.confidence * decay_factor;
                
                combined_strength += signal.strength * weight;
                total_weight += weight;
            }
            
            if total_weight > 0.0 {
                return Ok(Some(combined_strength / total_weight));
            }
        }
        
        Ok(None)
    }
}

pub struct ExecutionOptimizer {
    config: HFOptimizationConfig,
    market_impact_model: Arc<dyn MarketImpactModel>,
    adverse_selection_model: Arc<dyn AdverseSelectionModel>,
    inventory_manager: Arc<dyn InventoryManager>,
}

impl ExecutionOptimizer {
    pub fn new(
        config: HFOptimizationConfig,
        market_impact_model: Arc<dyn MarketImpactModel>,
        adverse_selection_model: Arc<dyn AdverseSelectionModel>,
        inventory_manager: Arc<dyn InventoryManager>,
    ) -> Self {
        Self {
            config,
            market_impact_model,
            adverse_selection_model,
            inventory_manager,
        }
    }

    pub async fn optimize_execution(
        &self,
        target: &OptimizationTarget,
        current_orderbook: &OrderBookState,
        combined_signal: f64,
        current_position: f64,
    ) -> Result<Vec<ExecutionSlice>> {
        let mut slices = Vec::new();
        
        let remaining_quantity = target.target_position - current_position;
        if remaining_quantity.abs() < self.config.lot_size {
            return Ok(slices);
        }

        // Calculate optimal slice size and timing
        let optimal_slices = self.calculate_optimal_slicing(
            target,
            current_orderbook,
            combined_signal,
            remaining_quantity,
        ).await?;

        // Apply inventory risk adjustments
        let inventory_adjusted_slices = self.inventory_manager
            .adjust_for_inventory_risk(&optimal_slices, current_position).await?;

        // Apply adverse selection protection
        let protected_slices = self.adverse_selection_model
            .apply_protection(&inventory_adjusted_slices, current_orderbook).await?;

        Ok(protected_slices)
    }

    async fn calculate_optimal_slicing(
        &self,
        target: &OptimizationTarget,
        orderbook: &OrderBookState,
        signal: f64,
        remaining_quantity: f64,
    ) -> Result<Vec<ExecutionSlice>> {
        let mut slices = Vec::new();
        
        let side = if remaining_quantity > 0.0 { "buy" } else { "sell" };
        let abs_quantity = remaining_quantity.abs();
        
        // Calculate base slice size based on market conditions
        let market_depth = if side == "buy" { orderbook.ask_size } else { orderbook.bid_size };
        let max_slice_size = (market_depth * target.max_participation_rate)
            .min(self.config.max_position_size)
            .max(self.config.lot_size);

        // Adjust for signal strength and urgency
        let urgency_multiplier = 1.0 + target.urgency * signal.abs();
        let adjusted_slice_size = max_slice_size * urgency_multiplier;
        
        let num_slices = (abs_quantity / adjusted_slice_size).ceil() as usize;
        let slice_quantity = abs_quantity / num_slices as f64;

        // Generate execution slices
        for i in 0..num_slices {
            let slice_urgency = target.urgency + (i as f64 / num_slices as f64) * 0.1;
            let expected_fill_rate = self.estimate_fill_rate(orderbook, slice_quantity, side).await?;
            
            let adverse_selection_risk = self.adverse_selection_model
                .estimate_risk(orderbook, slice_quantity, side).await?;

            let market_impact_cost = self.market_impact_model
                .estimate_cost(orderbook, slice_quantity, side).await?;

            let execution_price = if side == "buy" {
                orderbook.ask_price + market_impact_cost
            } else {
                orderbook.bid_price - market_impact_cost
            };

            let slice = ExecutionSlice {
                timestamp: Utc::now() + Duration::seconds(i as i64),
                symbol: target.symbol.clone(),
                side: side.to_string(),
                quantity: if side == "buy" { slice_quantity } else { -slice_quantity },
                price: execution_price,
                urgency: slice_urgency,
                expected_fill_rate,
                adverse_selection_risk,
                market_impact_cost,
            };

            slices.push(slice);
        }

        Ok(slices)
    }

    async fn estimate_fill_rate(&self, orderbook: &OrderBookState, quantity: f64, side: &str) -> Result<f64> {
        let available_size = if side == "buy" { orderbook.ask_size } else { orderbook.bid_size };
        
        if quantity <= available_size {
            Ok(1.0) // Full fill expected
        } else {
            // Partial fill probability based on size ratio
            let fill_ratio = available_size / quantity;
            Ok(fill_ratio.min(1.0))
        }
    }
}

#[async_trait]
pub trait MarketImpactModel: Send + Sync {
    async fn estimate_cost(&self, orderbook: &OrderBookState, quantity: f64, side: &str) -> Result<f64>;
    async fn update_model(&self, execution_data: &[ExecutionSlice], actual_costs: &[f64]) -> Result<()>;
}

pub struct SquareRootImpactModel {
    base_cost: f64,
    liquidity_coefficient: f64,
    volatility_coefficient: f64,
}

impl SquareRootImpactModel {
    pub fn new(base_cost: f64, liquidity_coefficient: f64, volatility_coefficient: f64) -> Self {
        Self {
            base_cost,
            liquidity_coefficient,
            volatility_coefficient,
        }
    }
}

#[async_trait]
impl MarketImpactModel for SquareRootImpactModel {
    async fn estimate_cost(&self, orderbook: &OrderBookState, quantity: f64, _side: &str) -> Result<f64> {
        let liquidity = (orderbook.bid_size + orderbook.ask_size) / 2.0;
        let volatility_proxy = orderbook.spread / orderbook.mid_price;
        
        let participation_rate = quantity / liquidity;
        let impact = self.base_cost + 
                    self.liquidity_coefficient * participation_rate.sqrt() +
                    self.volatility_coefficient * volatility_proxy;
        
        Ok(impact * orderbook.mid_price)
    }

    async fn update_model(&self, _execution_data: &[ExecutionSlice], _actual_costs: &[f64]) -> Result<()> {
        // In a real implementation, this would use machine learning to update model parameters
        Ok(())
    }
}

#[async_trait]
pub trait AdverseSelectionModel: Send + Sync {
    async fn estimate_risk(&self, orderbook: &OrderBookState, quantity: f64, side: &str) -> Result<f64>;
    async fn apply_protection(&self, slices: &[ExecutionSlice], orderbook: &OrderBookState) -> Result<Vec<ExecutionSlice>>;
}

pub struct SimpleAdverseSelectionModel {
    risk_coefficient: f64,
    protection_threshold: f64,
}

impl SimpleAdverseSelectionModel {
    pub fn new(risk_coefficient: f64, protection_threshold: f64) -> Self {
        Self {
            risk_coefficient,
            protection_threshold,
        }
    }
}

#[async_trait]
impl AdverseSelectionModel for SimpleAdverseSelectionModel {
    async fn estimate_risk(&self, orderbook: &OrderBookState, quantity: f64, side: &str) -> Result<f64> {
        let imbalance = if side == "buy" { orderbook.imbalance } else { -orderbook.imbalance };
        let size_impact = quantity / ((orderbook.bid_size + orderbook.ask_size) / 2.0);
        
        let risk = self.risk_coefficient * imbalance.max(0.0) * size_impact;
        Ok(risk)
    }

    async fn apply_protection(&self, slices: &[ExecutionSlice], orderbook: &OrderBookState) -> Result<Vec<ExecutionSlice>> {
        let mut protected_slices = Vec::new();
        
        for slice in slices {
            let mut protected_slice = slice.clone();
            
            if slice.adverse_selection_risk > self.protection_threshold {
                // Reduce urgency and adjust price for protection
                protected_slice.urgency *= 0.8;
                
                let protection_adjustment = slice.adverse_selection_risk * orderbook.spread * 0.2;
                if slice.side == "buy" {
                    protected_slice.price += protection_adjustment;
                } else {
                    protected_slice.price -= protection_adjustment;
                }
            }
            
            protected_slices.push(protected_slice);
        }

        Ok(protected_slices)
    }
}

#[async_trait]
pub trait InventoryManager: Send + Sync {
    async fn adjust_for_inventory_risk(&self, slices: &[ExecutionSlice], current_position: f64) -> Result<Vec<ExecutionSlice>>;
    async fn get_optimal_inventory(&self, symbol: &str) -> Result<f64>;
}

pub struct SimpleInventoryManager {
    target_inventory: f64,
    risk_aversion: f64,
    max_inventory: f64,
}

impl SimpleInventoryManager {
    pub fn new(target_inventory: f64, risk_aversion: f64, max_inventory: f64) -> Self {
        Self {
            target_inventory,
            risk_aversion,
            max_inventory,
        }
    }
}

#[async_trait]
impl InventoryManager for SimpleInventoryManager {
    async fn adjust_for_inventory_risk(&self, slices: &[ExecutionSlice], current_position: f64) -> Result<Vec<ExecutionSlice>> {
        let mut adjusted_slices = Vec::new();
        let mut running_position = current_position;
        
        for slice in slices {
            let new_position = running_position + slice.quantity;
            
            // Check inventory limits
            if new_position.abs() > self.max_inventory {
                // Reduce slice size to stay within limits
                let max_allowed_quantity = if slice.quantity > 0.0 {
                    (self.max_inventory - running_position).max(0.0)
                } else {
                    (-self.max_inventory - running_position).min(0.0)
                };
                
                if max_allowed_quantity.abs() > 0.0 {
                    let mut adjusted_slice = slice.clone();
                    adjusted_slice.quantity = max_allowed_quantity;
                    adjusted_slices.push(adjusted_slice);
                    running_position += max_allowed_quantity;
                }
            } else {
                // Apply inventory risk adjustment
                let inventory_deviation = (new_position - self.target_inventory).abs();
                let risk_penalty = self.risk_aversion * inventory_deviation / self.max_inventory;
                
                let mut adjusted_slice = slice.clone();
                adjusted_slice.urgency *= (1.0 - risk_penalty).max(0.1);
                
                adjusted_slices.push(adjusted_slice);
                running_position = new_position;
            }
        }

        Ok(adjusted_slices)
    }

    async fn get_optimal_inventory(&self, _symbol: &str) -> Result<f64> {
        Ok(self.target_inventory)
    }
}

pub struct HighFrequencyOptimizer {
    config: HFOptimizationConfig,
    signal_processor: Arc<SignalProcessor>,
    execution_optimizer: Arc<ExecutionOptimizer>,
    performance_tracker: Arc<RwLock<PerformanceTracker>>,
}

#[derive(Debug, Clone)]
struct PerformanceTracker {
    total_pnl: f64,
    total_trades: u64,
    win_rate: f64,
    sharpe_ratio: f64,
    max_drawdown: f64,
    avg_fill_rate: f64,
    avg_latency_ms: f64,
}

impl HighFrequencyOptimizer {
    pub fn new(
        config: HFOptimizationConfig,
        market_impact_model: Arc<dyn MarketImpactModel>,
        adverse_selection_model: Arc<dyn AdverseSelectionModel>,
        inventory_manager: Arc<dyn InventoryManager>,
    ) -> Self {
        let signal_processor = Arc::new(SignalProcessor::new(config.clone()));
        let execution_optimizer = Arc::new(ExecutionOptimizer::new(
            config.clone(),
            market_impact_model,
            adverse_selection_model,
            inventory_manager,
        ));

        Self {
            config,
            signal_processor,
            execution_optimizer,
            performance_tracker: Arc::new(RwLock::new(PerformanceTracker {
                total_pnl: 0.0,
                total_trades: 0,
                win_rate: 0.0,
                sharpe_ratio: 0.0,
                max_drawdown: 0.0,
                avg_fill_rate: 0.0,
                avg_latency_ms: 0.0,
            })),
        }
    }

    pub async fn optimize_for_symbol(
        &self,
        symbol: &str,
        orderbook: &OrderBookState,
        target: &OptimizationTarget,
        current_position: f64,
    ) -> Result<Vec<ExecutionSlice>> {
        let start_time = std::time::Instant::now();

        // Process market microstructure signals
        let signals = self.signal_processor.process_orderbook_update(orderbook).await?;
        
        // Get combined signal strength
        let combined_signal = self.signal_processor.get_combined_signal(symbol).await?
            .unwrap_or(0.0);

        // Check if signal strength meets threshold
        if combined_signal.abs() < self.config.min_signal_strength {
            return Ok(Vec::new());
        }

        // Optimize execution
        let execution_slices = self.execution_optimizer.optimize_execution(
            target,
            orderbook,
            combined_signal,
            current_position,
        ).await?;

        // Check latency constraint
        let processing_time = start_time.elapsed().as_millis() as u64;
        if processing_time > self.config.latency_threshold_ms {
            // Log latency warning but continue
            eprintln!("Warning: Processing time {}ms exceeds threshold {}ms", 
                     processing_time, self.config.latency_threshold_ms);
        }

        Ok(execution_slices)
    }

    pub async fn update_performance(&self, execution_result: &ExecutionResult) -> Result<()> {
        let mut tracker = self.performance_tracker.write().await;
        
        tracker.total_pnl += execution_result.realized_pnl;
        tracker.total_trades += 1;
        tracker.avg_fill_rate = (tracker.avg_fill_rate * (tracker.total_trades - 1) as f64 + 
                                execution_result.fill_rate) / tracker.total_trades as f64;
        tracker.avg_latency_ms = (tracker.avg_latency_ms * (tracker.total_trades - 1) as f64 + 
                                 execution_result.latency_ms) / tracker.total_trades as f64;

        Ok(())
    }

    pub async fn get_performance_metrics(&self) -> Result<PerformanceTracker> {
        let tracker = self.performance_tracker.read().await;
        Ok(tracker.clone())
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub execution_id: String,
    pub symbol: String,
    pub realized_pnl: f64,
    pub fill_rate: f64,
    pub latency_ms: f64,
    pub market_impact_cost: f64,
    pub adverse_selection_cost: f64,
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_signal_processor() {
        let config = HFOptimizationConfig::default();
        let processor = SignalProcessor::new(config);

        let orderbook = OrderBookState {
            timestamp: Utc::now(),
            symbol: "AAPL".to_string(),
            bid_price: 100.0,
            ask_price: 100.1,
            bid_size: 1000.0,
            ask_size: 800.0,
            mid_price: 100.05,
            spread: 0.1,
            imbalance: 0.2,
            depth_ratio: 1.25,
            last_trade_price: 100.05,
            last_trade_size: 100.0,
            book_pressure: 0.1,
        };

        let signals = processor.process_orderbook_update(&orderbook).await.unwrap();
        assert!(!signals.is_empty());
    }

    #[tokio::test]
    async fn test_execution_optimizer() {
        let config = HFOptimizationConfig::default();
        let impact_model = Arc::new(SquareRootImpactModel::new(0.001, 0.5, 1.0));
        let adverse_model = Arc::new(SimpleAdverseSelectionModel::new(0.1, 0.05));
        let inventory_mgr = Arc::new(SimpleInventoryManager::new(0.0, 0.5, 10000.0));

        let optimizer = ExecutionOptimizer::new(
            config,
            impact_model,
            adverse_model,
            inventory_mgr,
        );

        let target = OptimizationTarget {
            symbol: "AAPL".to_string(),
            target_position: 1000.0,
            max_participation_rate: 0.2,
            time_horizon: Duration::minutes(5),
            urgency: 0.5,
            risk_tolerance: 0.1,
            cost_penalty: 0.01,
        };

        let orderbook = OrderBookState {
            timestamp: Utc::now(),
            symbol: "AAPL".to_string(),
            bid_price: 100.0,
            ask_price: 100.1,
            bid_size: 2000.0,
            ask_size: 2000.0,
            mid_price: 100.05,
            spread: 0.1,
            imbalance: 0.0,
            depth_ratio: 1.0,
            last_trade_price: 100.05,
            last_trade_size: 100.0,
            book_pressure: 0.0,
        };

        let slices = optimizer.optimize_execution(&target, &orderbook, 0.05, 0.0).await.unwrap();
        assert!(!slices.is_empty());
    }
}