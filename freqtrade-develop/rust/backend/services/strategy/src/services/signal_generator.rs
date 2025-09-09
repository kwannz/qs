use anyhow::Result;
use chrono::Utc;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::config::Config;
use crate::models::*;
use crate::indicators::*;

pub struct SignalGenerator {
    config: Arc<Config>,
    signals: Arc<RwLock<HashMap<Uuid, Signal>>>,
    active_signals: Arc<RwLock<HashMap<Uuid, Vec<Uuid>>>>, // strategy_id -> signal_ids
}

impl SignalGenerator {
    pub async fn new(config: &Config) -> Result<Self> {
        Ok(Self {
            config: Arc::new(config.clone()),
            signals: Arc::new(RwLock::new(HashMap::new())),
            active_signals: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn generate_signals(
        &self,
        strategy: &Strategy,
        market_data: &HashMap<String, MarketData>,
        indicators: &HashMap<String, HashMap<String, Indicator>>,
        factors: &[MarketFactor],
    ) -> Result<Vec<Signal>> {
        let mut signals = Vec::new();

        match strategy.strategy_type {
            StrategyType::Momentum => {
                signals.extend(self.generate_momentum_signals(strategy, market_data, indicators).await?);
            },
            StrategyType::MeanReversion => {
                signals.extend(self.generate_mean_reversion_signals(strategy, market_data, indicators).await?);
            },
            StrategyType::Arbitrage => {
                signals.extend(self.generate_arbitrage_signals(strategy, market_data).await?);
            },
            StrategyType::GridTrading => {
                signals.extend(self.generate_grid_signals(strategy, market_data).await?);
            },
            StrategyType::Dca => {
                signals.extend(self.generate_dca_signals(strategy, market_data).await?);
            },
            StrategyType::PairsTrading => {
                signals.extend(self.generate_pairs_signals(strategy, market_data, indicators).await?);
            },
            StrategyType::MLPrediction => {
                signals.extend(self.generate_ml_signals(strategy, market_data, indicators, factors).await?);
            },
            StrategyType::FactorBased => {
                signals.extend(self.generate_factor_signals(strategy, market_data, factors).await?);
            },
            StrategyType::Custom => {
                signals.extend(self.generate_custom_signals(strategy, market_data, indicators, factors).await?);
            },
        }

        // Store signals
        self.store_signals(&signals).await;

        Ok(signals)
    }

    async fn generate_momentum_signals(
        &self,
        strategy: &Strategy,
        market_data: &HashMap<String, MarketData>,
        indicators: &HashMap<String, HashMap<String, Indicator>>,
    ) -> Result<Vec<Signal>> {
        let mut signals = Vec::new();

        for symbol in &strategy.symbols {
            if let (Some(data), Some(symbol_indicators)) = (market_data.get(symbol), indicators.get(symbol)) {
                // RSI-based momentum signals
                if let Some(rsi) = symbol_indicators.get("rsi") {
                    let rsi_overbought = strategy.parameters.rsi_overbought.unwrap_or(70.0);
                    let rsi_oversold = strategy.parameters.rsi_oversold.unwrap_or(30.0);

                    if rsi.value < rsi_oversold && rsi.previous_value >= rsi_oversold {
                        // RSI oversold - potential buy signal
                        signals.push(Signal {
                            id: Uuid::new_v4(),
                            strategy_id: strategy.id,
                            symbol: symbol.clone(),
                            exchange: data.exchange.clone(),
                            signal_type: SignalType::Entry,
                            action: SignalAction::Buy,
                            strength: (rsi_oversold - rsi.value) / rsi_oversold,
                            confidence: rsi.confidence,
                            price: data.close_price,
                            quantity: Some(self.calculate_position_size(strategy, data.close_price)),
                            reason: format!("RSI oversold: {:.2}", rsi.value),
                            factors: HashMap::from([("rsi".to_string(), rsi.value)]),
                            created_at: Utc::now(),
                            expires_at: Some(Utc::now() + chrono::Duration::minutes(15)),
                            executed: false,
                            executed_at: None,
                        });
                    } else if rsi.value > rsi_overbought && rsi.previous_value <= rsi_overbought {
                        // RSI overbought - potential sell signal
                        signals.push(Signal {
                            id: Uuid::new_v4(),
                            strategy_id: strategy.id,
                            symbol: symbol.clone(),
                            exchange: data.exchange.clone(),
                            signal_type: SignalType::Entry,
                            action: SignalAction::Sell,
                            strength: (rsi.value - rsi_overbought) / (100.0 - rsi_overbought),
                            confidence: rsi.confidence,
                            price: data.close_price,
                            quantity: Some(self.calculate_position_size(strategy, data.close_price)),
                            reason: format!("RSI overbought: {:.2}", rsi.value),
                            factors: HashMap::from([("rsi".to_string(), rsi.value)]),
                            created_at: Utc::now(),
                            expires_at: Some(Utc::now() + chrono::Duration::minutes(15)),
                            executed: false,
                            executed_at: None,
                        });
                    }
                }

                // Moving Average crossover signals
                if let (Some(ma_short), Some(ma_long)) = (symbol_indicators.get("ma_short"), symbol_indicators.get("ma_long")) {
                    if ma_short.value > ma_long.value && ma_short.previous_value <= ma_long.previous_value {
                        // Golden cross - buy signal
                        let strength = (ma_short.value - ma_long.value) / ma_long.value;
                        signals.push(Signal {
                            id: Uuid::new_v4(),
                            strategy_id: strategy.id,
                            symbol: symbol.clone(),
                            exchange: data.exchange.clone(),
                            signal_type: SignalType::Entry,
                            action: SignalAction::Buy,
                            strength: strength.abs(),
                            confidence: (ma_short.confidence + ma_long.confidence) / 2.0,
                            price: data.close_price,
                            quantity: Some(self.calculate_position_size(strategy, data.close_price)),
                            reason: format!("Golden cross: MA{} > MA{}", ma_short.parameters.get("period").unwrap_or(&14.0), ma_long.parameters.get("period").unwrap_or(&50.0)),
                            factors: HashMap::from([
                                ("ma_short".to_string(), ma_short.value),
                                ("ma_long".to_string(), ma_long.value),
                            ]),
                            created_at: Utc::now(),
                            expires_at: Some(Utc::now() + chrono::Duration::hours(1)),
                            executed: false,
                            executed_at: None,
                        });
                    } else if ma_short.value < ma_long.value && ma_short.previous_value >= ma_long.previous_value {
                        // Death cross - sell signal
                        let strength = (ma_long.value - ma_short.value) / ma_long.value;
                        signals.push(Signal {
                            id: Uuid::new_v4(),
                            strategy_id: strategy.id,
                            symbol: symbol.clone(),
                            exchange: data.exchange.clone(),
                            signal_type: SignalType::Entry,
                            action: SignalAction::Sell,
                            strength: strength.abs(),
                            confidence: (ma_short.confidence + ma_long.confidence) / 2.0,
                            price: data.close_price,
                            quantity: Some(self.calculate_position_size(strategy, data.close_price)),
                            reason: format!("Death cross: MA{} < MA{}", ma_short.parameters.get("period").unwrap_or(&14.0), ma_long.parameters.get("period").unwrap_or(&50.0)),
                            factors: HashMap::from([
                                ("ma_short".to_string(), ma_short.value),
                                ("ma_long".to_string(), ma_long.value),
                            ]),
                            created_at: Utc::now(),
                            expires_at: Some(Utc::now() + chrono::Duration::hours(1)),
                            executed: false,
                            executed_at: None,
                        });
                    }
                }
            }
        }

        Ok(signals)
    }

    async fn generate_mean_reversion_signals(
        &self,
        strategy: &Strategy,
        market_data: &HashMap<String, MarketData>,
        indicators: &HashMap<String, HashMap<String, Indicator>>,
    ) -> Result<Vec<Signal>> {
        let mut signals = Vec::new();

        for symbol in &strategy.symbols {
            if let (Some(data), Some(symbol_indicators)) = (market_data.get(symbol), indicators.get(symbol)) {
                // Bollinger Bands mean reversion
                if let (Some(bb_upper), Some(bb_middle), Some(bb_lower)) = (
                    symbol_indicators.get("bb_upper"),
                    symbol_indicators.get("bb_middle"),
                    symbol_indicators.get("bb_lower")
                ) {
                    let current_price = data.close_price.to_f64().unwrap_or(0.0);
                    
                    if current_price <= bb_lower.value && data.volume > data.average_volume * 1.2 {
                        // Price below lower band with high volume - buy signal
                        let distance_from_middle = (bb_middle.value - current_price) / bb_middle.value;
                        signals.push(Signal {
                            id: Uuid::new_v4(),
                            strategy_id: strategy.id,
                            symbol: symbol.clone(),
                            exchange: data.exchange.clone(),
                            signal_type: SignalType::Entry,
                            action: SignalAction::Buy,
                            strength: distance_from_middle.abs(),
                            confidence: 0.8,
                            price: data.close_price,
                            quantity: Some(self.calculate_position_size(strategy, data.close_price)),
                            reason: "Price below Bollinger lower band".to_string(),
                            factors: HashMap::from([
                                ("bb_position".to_string(), (current_price - bb_lower.value) / (bb_upper.value - bb_lower.value)),
                                ("volume_ratio".to_string(), data.volume / data.average_volume),
                            ]),
                            created_at: Utc::now(),
                            expires_at: Some(Utc::now() + chrono::Duration::hours(2)),
                            executed: false,
                            executed_at: None,
                        });
                    } else if current_price >= bb_upper.value && data.volume > data.average_volume * 1.2 {
                        // Price above upper band with high volume - sell signal
                        let distance_from_middle = (current_price - bb_middle.value) / bb_middle.value;
                        signals.push(Signal {
                            id: Uuid::new_v4(),
                            strategy_id: strategy.id,
                            symbol: symbol.clone(),
                            exchange: data.exchange.clone(),
                            signal_type: SignalType::Entry,
                            action: SignalAction::Sell,
                            strength: distance_from_middle.abs(),
                            confidence: 0.8,
                            price: data.close_price,
                            quantity: Some(self.calculate_position_size(strategy, data.close_price)),
                            reason: "Price above Bollinger upper band".to_string(),
                            factors: HashMap::from([
                                ("bb_position".to_string(), (current_price - bb_lower.value) / (bb_upper.value - bb_lower.value)),
                                ("volume_ratio".to_string(), data.volume / data.average_volume),
                            ]),
                            created_at: Utc::now(),
                            expires_at: Some(Utc::now() + chrono::Duration::hours(2)),
                            executed: false,
                            executed_at: None,
                        });
                    }
                }
            }
        }

        Ok(signals)
    }

    async fn generate_arbitrage_signals(
        &self,
        strategy: &Strategy,
        market_data: &HashMap<String, MarketData>,
    ) -> Result<Vec<Signal>> {
        let mut signals = Vec::new();

        // Simple price arbitrage between exchanges
        for symbol in &strategy.symbols {
            let mut exchange_prices: Vec<(String, Decimal)> = Vec::new();
            
            // Collect prices from different exchanges
            for (key, data) in market_data {
                if key.starts_with(&format!("{}:", symbol)) {
                    exchange_prices.push((data.exchange.clone(), data.close_price));
                }
            }

            // Look for arbitrage opportunities
            if exchange_prices.len() >= 2 {
                exchange_prices.sort_by(|a, b| a.1.cmp(&b.1));
                let lowest = &exchange_prices[0];
                let highest = &exchange_prices[exchange_prices.len() - 1];
                
                let spread_pct = ((highest.1 - lowest.1) / lowest.1).to_f64().unwrap_or(0.0) * 100.0;
                
                if spread_pct > 0.5 { // Minimum 0.5% spread for arbitrage
                    // Buy on lowest price exchange
                    signals.push(Signal {
                        id: Uuid::new_v4(),
                        strategy_id: strategy.id,
                        symbol: symbol.clone(),
                        exchange: lowest.0.clone(),
                        signal_type: SignalType::Entry,
                        action: SignalAction::Buy,
                        strength: spread_pct / 5.0, // Normalize to 0-1 (assuming max 5% spread)
                        confidence: 0.9,
                        price: lowest.1,
                        quantity: Some(self.calculate_position_size(strategy, lowest.1)),
                        reason: format!("Arbitrage opportunity: {:.2}% spread", spread_pct),
                        factors: HashMap::from([
                            ("spread_pct".to_string(), spread_pct),
                            ("low_price".to_string(), lowest.1.to_f64().unwrap_or(0.0)),
                            ("high_price".to_string(), highest.1.to_f64().unwrap_or(0.0)),
                        ]),
                        created_at: Utc::now(),
                        expires_at: Some(Utc::now() + chrono::Duration::minutes(5)),
                        executed: false,
                        executed_at: None,
                    });

                    // Sell on highest price exchange
                    signals.push(Signal {
                        id: Uuid::new_v4(),
                        strategy_id: strategy.id,
                        symbol: symbol.clone(),
                        exchange: highest.0.clone(),
                        signal_type: SignalType::Entry,
                        action: SignalAction::Sell,
                        strength: spread_pct / 5.0,
                        confidence: 0.9,
                        price: highest.1,
                        quantity: Some(self.calculate_position_size(strategy, highest.1)),
                        reason: format!("Arbitrage opportunity: {:.2}% spread", spread_pct),
                        factors: HashMap::from([
                            ("spread_pct".to_string(), spread_pct),
                            ("low_price".to_string(), lowest.1.to_f64().unwrap_or(0.0)),
                            ("high_price".to_string(), highest.1.to_f64().unwrap_or(0.0)),
                        ]),
                        created_at: Utc::now(),
                        expires_at: Some(Utc::now() + chrono::Duration::minutes(5)),
                        executed: false,
                        executed_at: None,
                    });
                }
            }
        }

        Ok(signals)
    }

    async fn generate_grid_signals(
        &self,
        strategy: &Strategy,
        market_data: &HashMap<String, MarketData>,
    ) -> Result<Vec<Signal>> {
        let mut signals = Vec::new();

        let grid_size = strategy.parameters.grid_size.unwrap_or(10) as f64;
        let grid_spacing_pct = strategy.parameters.grid_spacing_pct.unwrap_or(2.0);

        for symbol in &strategy.symbols {
            if let Some(data) = market_data.get(symbol) {
                let current_price = data.close_price.to_f64().unwrap_or(0.0);
                let base_price = data.open_price.to_f64().unwrap_or(current_price);
                
                // Generate grid levels
                for i in 1..=(grid_size as i32) {
                    let buy_price = base_price * (1.0 - (i as f64 * grid_spacing_pct / 100.0));
                    let sell_price = base_price * (1.0 + (i as f64 * grid_spacing_pct / 100.0));
                    
                    // Generate buy signals below current price
                    if current_price <= buy_price * 1.01 { // Within 1% tolerance
                        signals.push(Signal {
                            id: Uuid::new_v4(),
                            strategy_id: strategy.id,
                            symbol: symbol.clone(),
                            exchange: data.exchange.clone(),
                            signal_type: SignalType::Entry,
                            action: SignalAction::Buy,
                            strength: (base_price - buy_price) / base_price,
                            confidence: 0.7,
                            price: Decimal::from_f64_retain(buy_price).unwrap_or(data.close_price),
                            quantity: Some(self.calculate_position_size(strategy, data.close_price) / Decimal::from(grid_size)),
                            reason: format!("Grid buy level {}: ${:.2}", i, buy_price),
                            factors: HashMap::from([
                                ("grid_level".to_string(), i as f64),
                                ("distance_from_base".to_string(), (base_price - buy_price) / base_price * 100.0),
                            ]),
                            created_at: Utc::now(),
                            expires_at: Some(Utc::now() + chrono::Duration::hours(24)),
                            executed: false,
                            executed_at: None,
                        });
                    }

                    // Generate sell signals above current price
                    if current_price >= sell_price * 0.99 { // Within 1% tolerance
                        signals.push(Signal {
                            id: Uuid::new_v4(),
                            strategy_id: strategy.id,
                            symbol: symbol.clone(),
                            exchange: data.exchange.clone(),
                            signal_type: SignalType::Entry,
                            action: SignalAction::Sell,
                            strength: (sell_price - base_price) / base_price,
                            confidence: 0.7,
                            price: Decimal::from_f64_retain(sell_price).unwrap_or(data.close_price),
                            quantity: Some(self.calculate_position_size(strategy, data.close_price) / Decimal::from(grid_size)),
                            reason: format!("Grid sell level {}: ${:.2}", i, sell_price),
                            factors: HashMap::from([
                                ("grid_level".to_string(), i as f64),
                                ("distance_from_base".to_string(), (sell_price - base_price) / base_price * 100.0),
                            ]),
                            created_at: Utc::now(),
                            expires_at: Some(Utc::now() + chrono::Duration::hours(24)),
                            executed: false,
                            executed_at: None,
                        });
                    }
                }
            }
        }

        Ok(signals)
    }

    async fn generate_dca_signals(
        &self,
        strategy: &Strategy,
        market_data: &HashMap<String, MarketData>,
    ) -> Result<Vec<Signal>> {
        let mut signals = Vec::new();

        let dca_amount = strategy.parameters.dca_amount.unwrap_or(Decimal::from(1000));
        
        for symbol in &strategy.symbols {
            if let Some(data) = market_data.get(symbol) {
                // Simple DCA: buy on every interval regardless of price
                signals.push(Signal {
                    id: Uuid::new_v4(),
                    strategy_id: strategy.id,
                    symbol: symbol.clone(),
                    exchange: data.exchange.clone(),
                    signal_type: SignalType::Entry,
                    action: SignalAction::Buy,
                    strength: 0.5, // Constant strength for DCA
                    confidence: 1.0, // High confidence for DCA
                    price: data.close_price,
                    quantity: Some(dca_amount / data.close_price),
                    reason: format!("DCA buy: ${}", dca_amount),
                    factors: HashMap::from([
                        ("dca_amount".to_string(), dca_amount.to_f64().unwrap_or(0.0)),
                        ("current_price".to_string(), data.close_price.to_f64().unwrap_or(0.0)),
                    ]),
                    created_at: Utc::now(),
                    expires_at: Some(Utc::now() + chrono::Duration::hours(1)),
                    executed: false,
                    executed_at: None,
                });
            }
        }

        Ok(signals)
    }

    async fn generate_pairs_signals(
        &self,
        strategy: &Strategy,
        market_data: &HashMap<String, MarketData>,
        indicators: &HashMap<String, HashMap<String, Indicator>>,
    ) -> Result<Vec<Signal>> {
        let mut signals = Vec::new();

        // Simplified pairs trading - requires at least 2 symbols
        if strategy.symbols.len() >= 2 {
            let correlation_threshold = strategy.parameters.pairs_correlation_threshold.unwrap_or(0.8);
            
            for i in 0..strategy.symbols.len() {
                for j in (i+1)..strategy.symbols.len() {
                    let symbol1 = &strategy.symbols[i];
                    let symbol2 = &strategy.symbols[j];
                    
                    if let (Some(data1), Some(data2)) = (market_data.get(symbol1), market_data.get(symbol2)) {
                        // Calculate price ratio
                        let ratio = data1.close_price / data2.close_price;
                        let ratio_f64 = ratio.to_f64().unwrap_or(1.0);
                        
                        // Simple mean reversion on ratio (assuming historical mean of 1.0)
                        let deviation = (ratio_f64 - 1.0).abs();
                        
                        if deviation > 0.1 { // 10% deviation threshold
                            if ratio_f64 > 1.1 {
                                // Symbol1 overvalued relative to Symbol2
                                signals.push(Signal {
                                    id: Uuid::new_v4(),
                                    strategy_id: strategy.id,
                                    symbol: symbol1.clone(),
                                    exchange: data1.exchange.clone(),
                                    signal_type: SignalType::Entry,
                                    action: SignalAction::Sell,
                                    strength: deviation,
                                    confidence: 0.75,
                                    price: data1.close_price,
                                    quantity: Some(self.calculate_position_size(strategy, data1.close_price) / Decimal::from(2)),
                                    reason: format!("Pairs trade: {} overvalued vs {}", symbol1, symbol2),
                                    factors: HashMap::from([("ratio".to_string(), ratio_f64)]),
                                    created_at: Utc::now(),
                                    expires_at: Some(Utc::now() + chrono::Duration::hours(4)),
                                    executed: false,
                                    executed_at: None,
                                });
                                
                                signals.push(Signal {
                                    id: Uuid::new_v4(),
                                    strategy_id: strategy.id,
                                    symbol: symbol2.clone(),
                                    exchange: data2.exchange.clone(),
                                    signal_type: SignalType::Entry,
                                    action: SignalAction::Buy,
                                    strength: deviation,
                                    confidence: 0.75,
                                    price: data2.close_price,
                                    quantity: Some(self.calculate_position_size(strategy, data2.close_price) / Decimal::from(2)),
                                    reason: format!("Pairs trade: {} undervalued vs {}", symbol2, symbol1),
                                    factors: HashMap::from([("ratio".to_string(), ratio_f64)]),
                                    created_at: Utc::now(),
                                    expires_at: Some(Utc::now() + chrono::Duration::hours(4)),
                                    executed: false,
                                    executed_at: None,
                                });
                            } else if ratio_f64 < 0.9 {
                                // Symbol2 overvalued relative to Symbol1
                                signals.push(Signal {
                                    id: Uuid::new_v4(),
                                    strategy_id: strategy.id,
                                    symbol: symbol1.clone(),
                                    exchange: data1.exchange.clone(),
                                    signal_type: SignalType::Entry,
                                    action: SignalAction::Buy,
                                    strength: deviation,
                                    confidence: 0.75,
                                    price: data1.close_price,
                                    quantity: Some(self.calculate_position_size(strategy, data1.close_price) / Decimal::from(2)),
                                    reason: format!("Pairs trade: {} undervalued vs {}", symbol1, symbol2),
                                    factors: HashMap::from([("ratio".to_string(), ratio_f64)]),
                                    created_at: Utc::now(),
                                    expires_at: Some(Utc::now() + chrono::Duration::hours(4)),
                                    executed: false,
                                    executed_at: None,
                                });
                                
                                signals.push(Signal {
                                    id: Uuid::new_v4(),
                                    strategy_id: strategy.id,
                                    symbol: symbol2.clone(),
                                    exchange: data2.exchange.clone(),
                                    signal_type: SignalType::Entry,
                                    action: SignalAction::Sell,
                                    strength: deviation,
                                    confidence: 0.75,
                                    price: data2.close_price,
                                    quantity: Some(self.calculate_position_size(strategy, data2.close_price) / Decimal::from(2)),
                                    reason: format!("Pairs trade: {} overvalued vs {}", symbol2, symbol1),
                                    factors: HashMap::from([("ratio".to_string(), ratio_f64)]),
                                    created_at: Utc::now(),
                                    expires_at: Some(Utc::now() + chrono::Duration::hours(4)),
                                    executed: false,
                                    executed_at: None,
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(signals)
    }

    async fn generate_ml_signals(
        &self,
        strategy: &Strategy,
        market_data: &HashMap<String, MarketData>,
        indicators: &HashMap<String, HashMap<String, Indicator>>,
        factors: &[MarketFactor],
    ) -> Result<Vec<Signal>> {
        let mut signals = Vec::new();

        // Simplified ML-based signal generation
        // In a real implementation, this would call an ML model
        
        let ml_threshold = strategy.parameters.ml_model_confidence_threshold.unwrap_or(0.8);

        for symbol in &strategy.symbols {
            if let Some(data) = market_data.get(symbol) {
                // Combine multiple indicators for ML-like scoring
                let mut feature_score = 0.0;
                let mut feature_count = 0;

                if let Some(symbol_indicators) = indicators.get(symbol) {
                    // RSI feature
                    if let Some(rsi) = symbol_indicators.get("rsi") {
                        if rsi.value < 30.0 {
                            feature_score += 1.0; // Bullish
                        } else if rsi.value > 70.0 {
                            feature_score -= 1.0; // Bearish
                        }
                        feature_count += 1;
                    }

                    // MACD feature
                    if let Some(macd) = symbol_indicators.get("macd") {
                        if macd.value > macd.previous_value {
                            feature_score += 0.5; // Momentum increasing
                        } else {
                            feature_score -= 0.5; // Momentum decreasing
                        }
                        feature_count += 1;
                    }

                    // Volume feature
                    if data.volume > data.average_volume * 1.5 {
                        feature_score += 0.3; // High volume is bullish
                        feature_count += 1;
                    }
                }

                // Market factors influence
                let market_sentiment = factors.iter()
                    .filter(|f| f.category == FactorCategory::Sentiment)
                    .map(|f| f.normalized_value)
                    .sum::<f64>() / factors.iter().filter(|f| f.category == FactorCategory::Sentiment).count().max(1) as f64;
                
                feature_score += market_sentiment * 0.2;
                feature_count += 1;

                if feature_count > 0 {
                    let final_score = feature_score / feature_count as f64;
                    let confidence = final_score.abs().min(1.0);

                    if confidence > ml_threshold {
                        let action = if final_score > 0.0 { SignalAction::Buy } else { SignalAction::Sell };
                        
                        signals.push(Signal {
                            id: Uuid::new_v4(),
                            strategy_id: strategy.id,
                            symbol: symbol.clone(),
                            exchange: data.exchange.clone(),
                            signal_type: SignalType::Entry,
                            action,
                            strength: confidence,
                            confidence,
                            price: data.close_price,
                            quantity: Some(self.calculate_position_size(strategy, data.close_price)),
                            reason: format!("ML prediction: score {:.2}", final_score),
                            factors: HashMap::from([
                                ("ml_score".to_string(), final_score),
                                ("feature_count".to_string(), feature_count as f64),
                                ("market_sentiment".to_string(), market_sentiment),
                            ]),
                            created_at: Utc::now(),
                            expires_at: Some(Utc::now() + chrono::Duration::hours(2)),
                            executed: false,
                            executed_at: None,
                        });
                    }
                }
            }
        }

        Ok(signals)
    }

    async fn generate_factor_signals(
        &self,
        strategy: &Strategy,
        market_data: &HashMap<String, MarketData>,
        factors: &[MarketFactor],
    ) -> Result<Vec<Signal>> {
        let mut signals = Vec::new();

        // Factor-based signal generation
        let factor_weights = strategy.parameters.factor_weights.as_ref()
            .unwrap_or(&HashMap::new());

        for symbol in &strategy.symbols {
            if let Some(data) = market_data.get(symbol) {
                let mut weighted_score = 0.0;
                let mut total_weight = 0.0;

                for factor in factors {
                    if let Some(&weight) = factor_weights.get(&factor.id) {
                        weighted_score += factor.normalized_value * weight;
                        total_weight += weight.abs();
                    }
                }

                if total_weight > 0.0 {
                    let final_score = weighted_score / total_weight;
                    let confidence = final_score.abs().min(1.0);

                    if confidence > 0.6 {
                        let action = if final_score > 0.0 { SignalAction::Buy } else { SignalAction::Sell };
                        
                        signals.push(Signal {
                            id: Uuid::new_v4(),
                            strategy_id: strategy.id,
                            symbol: symbol.clone(),
                            exchange: data.exchange.clone(),
                            signal_type: SignalType::Entry,
                            action,
                            strength: confidence,
                            confidence,
                            price: data.close_price,
                            quantity: Some(self.calculate_position_size(strategy, data.close_price)),
                            reason: format!("Factor-based signal: score {:.2}", final_score),
                            factors: HashMap::from([
                                ("factor_score".to_string(), final_score),
                                ("factors_used".to_string(), factor_weights.len() as f64),
                            ]),
                            created_at: Utc::now(),
                            expires_at: Some(Utc::now() + chrono::Duration::hours(3)),
                            executed: false,
                            executed_at: None,
                        });
                    }
                }
            }
        }

        Ok(signals)
    }

    async fn generate_custom_signals(
        &self,
        strategy: &Strategy,
        market_data: &HashMap<String, MarketData>,
        indicators: &HashMap<String, HashMap<String, Indicator>>,
        factors: &[MarketFactor],
    ) -> Result<Vec<Signal>> {
        let mut signals = Vec::new();

        // Custom signal generation based on strategy parameters
        // This would be implemented based on user-defined rules
        
        for symbol in &strategy.symbols {
            if let Some(data) = market_data.get(symbol) {
                // Example: Custom rule from parameters
                if let Some(custom_threshold) = strategy.parameters.custom_params.get("price_threshold") {
                    if let Some(threshold_value) = custom_threshold.as_f64() {
                        let current_price = data.close_price.to_f64().unwrap_or(0.0);
                        
                        if current_price < threshold_value {
                            signals.push(Signal {
                                id: Uuid::new_v4(),
                                strategy_id: strategy.id,
                                symbol: symbol.clone(),
                                exchange: data.exchange.clone(),
                                signal_type: SignalType::Entry,
                                action: SignalAction::Buy,
                                strength: (threshold_value - current_price) / threshold_value,
                                confidence: 0.7,
                                price: data.close_price,
                                quantity: Some(self.calculate_position_size(strategy, data.close_price)),
                                reason: format!("Custom rule: price below threshold ${:.2}", threshold_value),
                                factors: HashMap::from([
                                    ("threshold".to_string(), threshold_value),
                                    ("current_price".to_string(), current_price),
                                ]),
                                created_at: Utc::now(),
                                expires_at: Some(Utc::now() + chrono::Duration::hours(1)),
                                executed: false,
                                executed_at: None,
                            });
                        }
                    }
                }
            }
        }

        Ok(signals)
    }

    // Helper methods
    fn calculate_position_size(&self, strategy: &Strategy, price: Decimal) -> Decimal {
        // Simple position sizing: use percentage of max position size
        let risk_tolerance = strategy.parameters.risk_tolerance;
        let max_size = strategy.parameters.max_position_size;
        
        // Calculate position size based on risk tolerance
        let base_size = max_size * Decimal::from_f64_retain(risk_tolerance).unwrap_or(Decimal::from(1));
        
        // Convert to quantity
        base_size / price
    }

    async fn store_signals(&self, signals: &[Signal]) {
        let mut stored_signals = self.signals.write().await;
        let mut active_signals = self.active_signals.write().await;

        for signal in signals {
            stored_signals.insert(signal.id, signal.clone());
            
            active_signals
                .entry(signal.strategy_id)
                .or_insert_with(Vec::new)
                .push(signal.id);
        }
    }

    // Public methods for querying signals
    pub async fn get_signals(&self, limit: Option<u32>, offset: Option<u32>) -> Result<SignalListResponse> {
        let signals = self.signals.read().await;
        let all_signals: Vec<Signal> = signals.values().cloned().collect();
        
        let total_count = all_signals.len() as u32;
        let offset = offset.unwrap_or(0) as usize;
        let limit = limit.unwrap_or(50) as usize;
        
        let signals_page = all_signals
            .into_iter()
            .skip(offset)
            .take(limit)
            .collect();

        Ok(SignalListResponse {
            signals: signals_page,
            total_count,
            page: (offset / limit.max(1)) as u32 + 1,
            per_page: limit as u32,
            filters_applied: HashMap::new(),
        })
    }

    pub async fn get_signals_by_strategy(&self, strategy_id: Uuid) -> Result<Vec<Signal>> {
        let signals = self.signals.read().await;
        let active_signals = self.active_signals.read().await;
        
        let mut result = Vec::new();
        
        if let Some(signal_ids) = active_signals.get(&strategy_id) {
            for signal_id in signal_ids {
                if let Some(signal) = signals.get(signal_id) {
                    result.push(signal.clone());
                }
            }
        }
        
        Ok(result)
    }

    pub async fn get_active_signals_count(&self, strategy_id: Uuid) -> u32 {
        let active_signals = self.active_signals.read().await;
        active_signals.get(&strategy_id)
            .map(|signals| signals.len())
            .unwrap_or(0) as u32
    }
}

// Helper struct for market data
#[derive(Debug, Clone)]
pub struct MarketData {
    pub symbol: String,
    pub exchange: String,
    pub open_price: Decimal,
    pub high_price: Decimal,
    pub low_price: Decimal,
    pub close_price: Decimal,
    pub volume: f64,
    pub average_volume: f64,
    pub timestamp: chrono::DateTime<Utc>,
}