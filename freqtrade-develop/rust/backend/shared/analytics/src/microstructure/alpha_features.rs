use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::warn;

use crate::microstructure::{BookSnapshot, BookLevel, TradeData, TradeSide};

/// Alpha特征配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaConfig {
    pub lookback_periods: Vec<usize>,      // 回看周期：[5, 10, 20, 50]
    pub decay_factors: Vec<f64>,           // 衰减因子：[0.9, 0.95, 0.98]
    pub volatility_window: usize,          // 波动率计算窗口
    pub momentum_window: usize,            // 动量计算窗口
    pub mean_reversion_window: usize,      // 均值回归窗口
    pub regime_sensitivity: f64,           // 体制敏感度
    pub signal_threshold: f64,             // 信号阈值
}

impl Default for AlphaConfig {
    fn default() -> Self {
        Self {
            lookback_periods: vec![5, 10, 20, 50],
            decay_factors: vec![0.9, 0.95, 0.98],
            volatility_window: 20,
            momentum_window: 10,
            mean_reversion_window: 50,
            regime_sensitivity: 0.02,
            signal_threshold: 0.1,
        }
    }
}

/// 微结构Alpha特征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrostructureAlphaFeatures {
    pub timestamp: i64,
    pub symbol: String,
    
    // 基础价格特征
    pub price_features: PriceFeatures,
    
    // 订单簿特征  
    pub order_book_features: OrderBookFeatures,
    
    // 交易流特征
    pub trade_flow_features: TradeFlowFeatures,
    
    // 动量特征
    pub momentum_features: MomentumFeatures,
    
    // 均值回归特征
    pub mean_reversion_features: MeanReversionFeatures,
    
    // 体制感知特征
    pub regime_features: RegimeFeatures,
    
    // 综合Alpha信号
    pub composite_alpha: CompositeAlpha,
}

/// 价格特征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceFeatures {
    pub mid_price: f64,
    pub weighted_mid_price: f64,
    pub micro_price: f64,                  // 微观价格
    pub price_acceleration: f64,            // 价格加速度
    pub price_volatility: f64,             // 价格波动率
    pub relative_spread: f64,              // 相对价差
    pub effective_spread: f64,             // 有效价差
    pub realized_spread: f64,              // 实现价差
}

/// 订单簿特征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookFeatures {
    pub depth_imbalance: f64,              // 深度不平衡
    pub slope_imbalance: f64,              // 斜率不平衡
    pub resilience: f64,                   // 弹性恢复
    pub book_pressure: f64,                // 订单簿压力
    pub liquidity_density: f64,            // 流动性密度
    pub order_arrival_intensity: f64,      // 订单到达强度
    pub cancellation_rate: f64,            // 撤单率
}

/// 交易流特征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeFlowFeatures {
    pub order_flow_imbalance: f64,         // 订单流不平衡
    pub volume_weighted_price: f64,        // 成交量加权价格
    pub trade_intensity: f64,              // 交易强度
    pub size_weighted_flow: f64,           // 尺寸加权流
    pub tick_direction: f64,               // Tick方向
    pub trade_clustering: f64,             // 交易聚集度
    pub market_impact: f64,                // 市场冲击
}

/// 动量特征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MomentumFeatures {
    pub short_momentum: f64,               // 短期动量
    pub medium_momentum: f64,              // 中期动量
    pub long_momentum: f64,                // 长期动量
    pub momentum_acceleration: f64,        // 动量加速度
    pub momentum_persistence: f64,         // 动量持续性
    pub breakout_strength: f64,            // 突破强度
    pub trend_strength: f64,               // 趋势强度
}

/// 均值回归特征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanReversionFeatures {
    pub deviation_from_mean: f64,          // 与均值的偏离
    pub reversion_speed: f64,              // 回归速度
    pub reversion_probability: f64,        // 回归概率
    pub support_resistance: f64,           // 支撑阻力
    pub range_position: f64,               // 区间位置
    pub oversold_overbought: f64,          // 超买超卖
    pub cyclical_component: f64,           // 周期性成分
}

/// 体制感知特征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeFeatures {
    pub volatility_regime: f64,            // 波动率体制
    pub trend_regime: f64,                 // 趋势体制
    pub liquidity_regime: f64,             // 流动性体制
    pub correlation_regime: f64,           // 相关性体制
    pub regime_persistence: f64,           // 体制持续性
    pub regime_transition_prob: f64,       // 体制转换概率
    pub regime_stability: f64,             // 体制稳定性
}

/// 综合Alpha信号
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeAlpha {
    pub raw_alpha: f64,                    // 原始Alpha
    pub risk_adjusted_alpha: f64,          // 风险调整Alpha
    pub regime_adjusted_alpha: f64,        // 体制调整Alpha
    pub confidence_score: f64,             // 置信度
    pub signal_strength: f64,              // 信号强度
    pub decay_adjusted_alpha: f64,         // 衰减调整Alpha
    pub final_alpha: f64,                  // 最终Alpha信号
}

/// 微结构Alpha特征计算器
#[derive(Debug)]
pub struct MicrostructureAlphaCalculator {
    config: AlphaConfig,
    symbol: String,
    
    // 历史数据
    price_history: VecDeque<f64>,
    spread_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    imbalance_history: VecDeque<f64>,
    
    // 订单簿历史
    book_snapshots: VecDeque<BookSnapshot>,
    trade_data: VecDeque<TradeData>,
    
    // 缓存的特征
    cached_features: Option<MicrostructureAlphaFeatures>,
    last_update_time: i64,
}

impl MicrostructureAlphaCalculator {
    pub fn new(symbol: String, config: AlphaConfig) -> Self {
        let max_history = config.lookback_periods.iter().max().copied().unwrap_or(100) * 2;
        
        Self {
            config,
            symbol,
            price_history: VecDeque::with_capacity(max_history),
            spread_history: VecDeque::with_capacity(max_history),
            volume_history: VecDeque::with_capacity(max_history),
            imbalance_history: VecDeque::with_capacity(max_history),
            book_snapshots: VecDeque::with_capacity(max_history),
            trade_data: VecDeque::with_capacity(max_history),
            cached_features: None,
            last_update_time: 0,
        }
    }

    /// 更新市场数据
    pub fn update_market_data(&mut self, snapshot: BookSnapshot, trades: Vec<TradeData>) -> Result<()> {
        let max_history = self.config.lookback_periods.iter().max().copied().unwrap_or(100) * 2;
        
        // 计算基础指标
        let mid_price = self.calculate_mid_price(&snapshot)?;
        let spread = self.calculate_spread(&snapshot)?;
        let imbalance = self.calculate_basic_imbalance(&snapshot)?;
        let volume = trades.iter().map(|t| t.size).sum::<f64>();
        
        // 更新历史数据
        self.price_history.push_back(mid_price);
        self.spread_history.push_back(spread);
        self.volume_history.push_back(volume);
        self.imbalance_history.push_back(imbalance);
        
        // 限制历史长度
        while self.price_history.len() > max_history {
            self.price_history.pop_front();
        }
        while self.spread_history.len() > max_history {
            self.spread_history.pop_front();
        }
        while self.volume_history.len() > max_history {
            self.volume_history.pop_front();
        }
        while self.imbalance_history.len() > max_history {
            self.imbalance_history.pop_front();
        }
        
        // 更新订单簿和交易数据
        self.book_snapshots.push_back(snapshot);
        for trade in trades {
            self.trade_data.push_back(trade);
        }
        
        while self.book_snapshots.len() > max_history {
            self.book_snapshots.pop_front();
        }
        while self.trade_data.len() > max_history {
            self.trade_data.pop_front();
        }
        
        self.last_update_time = chrono::Utc::now().timestamp_millis();
        Ok(())
    }

    /// 计算完整的Alpha特征
    pub fn calculate_features(&mut self) -> Result<MicrostructureAlphaFeatures> {
        if self.price_history.len() < 10 {
            return Err(anyhow::anyhow!("Insufficient historical data"));
        }

        let timestamp = self.last_update_time;
        let current_snapshot = self.book_snapshots.back()
            .context("No order book snapshot available")?;

        // 计算各类特征
        let price_features = self.calculate_price_features(current_snapshot)?;
        let order_book_features = self.calculate_order_book_features(current_snapshot)?;
        let trade_flow_features = self.calculate_trade_flow_features()?;
        let momentum_features = self.calculate_momentum_features()?;
        let mean_reversion_features = self.calculate_mean_reversion_features()?;
        let regime_features = self.calculate_regime_features()?;
        let composite_alpha = self.calculate_composite_alpha(
            &price_features,
            &order_book_features,
            &trade_flow_features,
            &momentum_features,
            &mean_reversion_features,
            &regime_features,
        )?;

        let features = MicrostructureAlphaFeatures {
            timestamp,
            symbol: self.symbol.clone(),
            price_features,
            order_book_features,
            trade_flow_features,
            momentum_features,
            mean_reversion_features,
            regime_features,
            composite_alpha,
        };

        self.cached_features = Some(features.clone());
        Ok(features)
    }

    /// 计算价格特征
    fn calculate_price_features(&self, snapshot: &BookSnapshot) -> Result<PriceFeatures> {
        let mid_price = self.calculate_mid_price(snapshot)?;
        let weighted_mid_price = self.calculate_weighted_mid_price(snapshot)?;
        let micro_price = self.calculate_micro_price(snapshot)?;
        
        // 价格加速度（二阶导数）
        let price_acceleration = if self.price_history.len() >= 3 {
            let recent_prices: Vec<f64> = self.price_history.iter().rev().take(3).cloned().collect();
            recent_prices[0] - 2.0 * recent_prices[1] + recent_prices[2]
        } else {
            0.0
        };

        // 价格波动率
        let price_volatility = self.calculate_rolling_volatility(self.config.volatility_window)?;
        
        // 价差相关指标
        let relative_spread = self.calculate_relative_spread(snapshot)?;
        let effective_spread = self.calculate_effective_spread(snapshot)?;
        let realized_spread = self.calculate_realized_spread(snapshot)?;

        Ok(PriceFeatures {
            mid_price,
            weighted_mid_price,
            micro_price,
            price_acceleration,
            price_volatility,
            relative_spread,
            effective_spread,
            realized_spread,
        })
    }

    /// 计算订单簿特征
    fn calculate_order_book_features(&self, snapshot: &BookSnapshot) -> Result<OrderBookFeatures> {
        let depth_imbalance = self.calculate_depth_imbalance(snapshot)?;
        let slope_imbalance = self.calculate_slope_imbalance(snapshot)?;
        let resilience = self.calculate_resilience()?;
        let book_pressure = self.calculate_book_pressure(snapshot)?;
        let liquidity_density = self.calculate_liquidity_density(snapshot)?;
        let order_arrival_intensity = self.calculate_arrival_intensity()?;
        let cancellation_rate = self.calculate_cancellation_rate()?;

        Ok(OrderBookFeatures {
            depth_imbalance,
            slope_imbalance,
            resilience,
            book_pressure,
            liquidity_density,
            order_arrival_intensity,
            cancellation_rate,
        })
    }

    /// 计算交易流特征
    fn calculate_trade_flow_features(&self) -> Result<TradeFlowFeatures> {
        let order_flow_imbalance = self.calculate_order_flow_imbalance()?;
        let volume_weighted_price = self.calculate_vwap()?;
        let trade_intensity = self.calculate_trade_intensity()?;
        let size_weighted_flow = self.calculate_size_weighted_flow()?;
        let tick_direction = self.calculate_tick_direction()?;
        let trade_clustering = self.calculate_trade_clustering()?;
        let market_impact = self.calculate_market_impact()?;

        Ok(TradeFlowFeatures {
            order_flow_imbalance,
            volume_weighted_price,
            trade_intensity,
            size_weighted_flow,
            tick_direction,
            trade_clustering,
            market_impact,
        })
    }

    /// 计算动量特征
    fn calculate_momentum_features(&self) -> Result<MomentumFeatures> {
        let short_momentum = self.calculate_momentum(5)?;
        let medium_momentum = self.calculate_momentum(10)?;
        let long_momentum = self.calculate_momentum(20)?;
        
        let momentum_acceleration = if self.price_history.len() >= 10 {
            let recent_mom = self.calculate_momentum(3)?;
            let prev_mom = self.calculate_momentum_at_offset(3, 3)?;
            recent_mom - prev_mom
        } else {
            0.0
        };
        
        let momentum_persistence = self.calculate_momentum_persistence()?;
        let breakout_strength = self.calculate_breakout_strength()?;
        let trend_strength = self.calculate_trend_strength()?;

        Ok(MomentumFeatures {
            short_momentum,
            medium_momentum,
            long_momentum,
            momentum_acceleration,
            momentum_persistence,
            breakout_strength,
            trend_strength,
        })
    }

    /// 计算均值回归特征
    fn calculate_mean_reversion_features(&self) -> Result<MeanReversionFeatures> {
        let window = self.config.mean_reversion_window;
        let mean = self.calculate_rolling_mean(window)?;
        let current_price = self.price_history.back().copied().unwrap_or(0.0);
        
        let deviation_from_mean = (current_price - mean) / mean;
        let reversion_speed = self.calculate_reversion_speed()?;
        let reversion_probability = self.calculate_reversion_probability()?;
        let support_resistance = self.calculate_support_resistance()?;
        let range_position = self.calculate_range_position()?;
        let oversold_overbought = self.calculate_rsi_like_indicator()?;
        let cyclical_component = self.calculate_cyclical_component()?;

        Ok(MeanReversionFeatures {
            deviation_from_mean,
            reversion_speed,
            reversion_probability,
            support_resistance,
            range_position,
            oversold_overbought,
            cyclical_component,
        })
    }

    /// 计算体制感知特征
    fn calculate_regime_features(&self) -> Result<RegimeFeatures> {
        let volatility_regime = self.classify_volatility_regime()?;
        let trend_regime = self.classify_trend_regime()?;
        let liquidity_regime = self.classify_liquidity_regime()?;
        let correlation_regime = self.classify_correlation_regime()?;
        let regime_persistence = self.calculate_regime_persistence()?;
        let regime_transition_prob = self.calculate_transition_probability()?;
        let regime_stability = self.calculate_regime_stability()?;

        Ok(RegimeFeatures {
            volatility_regime,
            trend_regime,
            liquidity_regime,
            correlation_regime,
            regime_persistence,
            regime_transition_prob,
            regime_stability,
        })
    }

    /// 计算综合Alpha信号
    fn calculate_composite_alpha(
        &self,
        price: &PriceFeatures,
        book: &OrderBookFeatures,
        flow: &TradeFlowFeatures,
        momentum: &MomentumFeatures,
        mean_rev: &MeanReversionFeatures,
        regime: &RegimeFeatures,
    ) -> Result<CompositeAlpha> {
        // 基础Alpha组合
        let raw_alpha = 
            book.depth_imbalance * 0.2 +
            flow.order_flow_imbalance * 0.15 +
            momentum.short_momentum * 0.15 +
            mean_rev.deviation_from_mean * (-0.1) +
            flow.market_impact * (-0.1) +
            book.resilience * 0.1 +
            momentum.trend_strength * 0.1 +
            flow.trade_intensity * 0.05;

        // 风险调整
        let volatility_adj = (price.price_volatility + 1e-8).ln().tanh();
        let risk_adjusted_alpha = raw_alpha * (1.0 - volatility_adj * 0.3);

        // 体制调整
        let regime_factor = (regime.volatility_regime + regime.trend_regime + regime.liquidity_regime) / 3.0;
        let regime_adjusted_alpha = risk_adjusted_alpha * (0.7 + 0.3 * regime_factor);

        // 置信度计算
        let confidence_score = self.calculate_alpha_confidence(
            momentum.momentum_persistence,
            regime.regime_stability,
            book.liquidity_density,
        )?;

        // 信号强度
        let signal_strength = raw_alpha.abs().min(1.0);

        // 衰减调整
        let decay_adjusted_alpha = self.apply_decay_adjustment(regime_adjusted_alpha)?;

        // 最终Alpha（应用置信度权重）
        let final_alpha = decay_adjusted_alpha * confidence_score;

        Ok(CompositeAlpha {
            raw_alpha,
            risk_adjusted_alpha,
            regime_adjusted_alpha,
            confidence_score,
            signal_strength,
            decay_adjusted_alpha,
            final_alpha,
        })
    }

    // 辅助计算方法
    fn calculate_mid_price(&self, snapshot: &BookSnapshot) -> Result<f64> {
        if snapshot.bids.is_empty() || snapshot.asks.is_empty() {
            return Err(anyhow::anyhow!("Empty order book"));
        }
        Ok((snapshot.bids[0].price + snapshot.asks[0].price) / 2.0)
    }

    fn calculate_spread(&self, snapshot: &BookSnapshot) -> Result<f64> {
        if snapshot.bids.is_empty() || snapshot.asks.is_empty() {
            return Err(anyhow::anyhow!("Empty order book"));
        }
        Ok(snapshot.asks[0].price - snapshot.bids[0].price)
    }

    fn calculate_basic_imbalance(&self, snapshot: &BookSnapshot) -> Result<f64> {
        let bid_vol: f64 = snapshot.bids.iter().take(5).map(|l| l.size).sum();
        let ask_vol: f64 = snapshot.asks.iter().take(5).map(|l| l.size).sum();
        let total_vol = bid_vol + ask_vol;
        
        if total_vol == 0.0 {
            return Ok(0.0);
        }
        
        Ok((bid_vol - ask_vol) / total_vol)
    }

    fn calculate_weighted_mid_price(&self, snapshot: &BookSnapshot) -> Result<f64> {
        if snapshot.bids.is_empty() || snapshot.asks.is_empty() {
            return self.calculate_mid_price(snapshot);
        }
        
        let best_bid_size = snapshot.bids[0].size;
        let best_ask_size = snapshot.asks[0].size;
        let total_size = best_bid_size + best_ask_size;
        
        if total_size == 0.0 {
            return self.calculate_mid_price(snapshot);
        }
        
        Ok((snapshot.bids[0].price * best_ask_size + snapshot.asks[0].price * best_bid_size) / total_size)
    }

    fn calculate_micro_price(&self, snapshot: &BookSnapshot) -> Result<f64> {
        // 微观价格：考虑订单簿深度的价格
        let mut weighted_price = 0.0;
        let mut total_weight = 0.0;
        
        // 使用前3档价格，距离中间价越近权重越大
        let mid_price = self.calculate_mid_price(snapshot)?;
        
        for (i, level) in snapshot.bids.iter().take(3).enumerate() {
            let weight = level.size / (1.0 + i as f64);
            weighted_price += level.price * weight;
            total_weight += weight;
        }
        
        for (i, level) in snapshot.asks.iter().take(3).enumerate() {
            let weight = level.size / (1.0 + i as f64);
            weighted_price += level.price * weight;
            total_weight += weight;
        }
        
        if total_weight > 0.0 {
            Ok(weighted_price / total_weight)
        } else {
            Ok(mid_price)
        }
    }

    // 其他辅助方法的简化实现
    fn calculate_rolling_volatility(&self, window: usize) -> Result<f64> {
        if self.price_history.len() < window {
            return Ok(0.0);
        }
        
        let recent_prices: Vec<f64> = self.price_history.iter().rev().take(window).cloned().collect();
        let mean = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
        let variance = recent_prices.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / recent_prices.len() as f64;
            
        Ok(variance.sqrt())
    }

    fn calculate_rolling_mean(&self, window: usize) -> Result<f64> {
        if self.price_history.len() < window {
            return Ok(self.price_history.iter().sum::<f64>() / self.price_history.len() as f64);
        }
        
        let recent_prices: Vec<f64> = self.price_history.iter().rev().take(window).cloned().collect();
        Ok(recent_prices.iter().sum::<f64>() / recent_prices.len() as f64)
    }

    fn calculate_momentum(&self, period: usize) -> Result<f64> {
        if self.price_history.len() <= period {
            return Ok(0.0);
        }
        
        let current = *self.price_history.back().unwrap();
        let past = self.price_history[self.price_history.len() - 1 - period];
        
        Ok((current - past) / past)
    }

    fn calculate_momentum_at_offset(&self, period: usize, offset: usize) -> Result<f64> {
        if self.price_history.len() <= period + offset {
            return Ok(0.0);
        }
        
        let current_idx = self.price_history.len() - 1 - offset;
        let past_idx = current_idx - period;
        
        let current = self.price_history[current_idx];
        let past = self.price_history[past_idx];
        
        Ok((current - past) / past)
    }

    // 简化版本的其他计算方法
    fn calculate_relative_spread(&self, snapshot: &BookSnapshot) -> Result<f64> {
        let spread = self.calculate_spread(snapshot)?;
        let mid = self.calculate_mid_price(snapshot)?;
        Ok(spread / mid)
    }

    fn calculate_effective_spread(&self, _snapshot: &BookSnapshot) -> Result<f64> {
        // 简化：使用最近的交易数据估算
        if let Some(trade) = self.trade_data.back() {
            let mid = self.price_history.back().copied().unwrap_or(trade.price);
            Ok(2.0 * (trade.price - mid).abs() / mid)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_realized_spread(&self, _snapshot: &BookSnapshot) -> Result<f64> {
        // 简化实现
        Ok(self.spread_history.back().copied().unwrap_or(0.0))
    }

    // 简化的其他特征计算方法
    fn calculate_depth_imbalance(&self, snapshot: &BookSnapshot) -> Result<f64> {
        self.calculate_basic_imbalance(snapshot)
    }

    fn calculate_slope_imbalance(&self, _snapshot: &BookSnapshot) -> Result<f64> {
        Ok(0.0) // 简化实现
    }

    fn calculate_resilience(&self) -> Result<f64> {
        // 简化：基于历史价差恢复速度
        if self.spread_history.len() < 5 {
            return Ok(0.5);
        }
        
        let recent_spreads: Vec<f64> = self.spread_history.iter().rev().take(5).cloned().collect();
        let spread_change = recent_spreads.first().unwrap() - recent_spreads.last().unwrap();
        Ok((-spread_change).tanh())
    }

    // 其他简化实现
    fn calculate_book_pressure(&self, _snapshot: &BookSnapshot) -> Result<f64> { Ok(0.0) }
    fn calculate_liquidity_density(&self, _snapshot: &BookSnapshot) -> Result<f64> { Ok(0.5) }
    fn calculate_arrival_intensity(&self) -> Result<f64> { Ok(0.0) }
    fn calculate_cancellation_rate(&self) -> Result<f64> { Ok(0.0) }
    fn calculate_order_flow_imbalance(&self) -> Result<f64> { Ok(0.0) }
    fn calculate_vwap(&self) -> Result<f64> { 
        Ok(self.price_history.back().copied().unwrap_or(0.0)) 
    }
    fn calculate_trade_intensity(&self) -> Result<f64> { Ok(0.0) }
    fn calculate_size_weighted_flow(&self) -> Result<f64> { Ok(0.0) }
    fn calculate_tick_direction(&self) -> Result<f64> { Ok(0.0) }
    fn calculate_trade_clustering(&self) -> Result<f64> { Ok(0.0) }
    fn calculate_market_impact(&self) -> Result<f64> { Ok(0.0) }
    fn calculate_momentum_persistence(&self) -> Result<f64> { Ok(0.0) }
    fn calculate_breakout_strength(&self) -> Result<f64> { Ok(0.0) }
    fn calculate_trend_strength(&self) -> Result<f64> { Ok(0.0) }
    fn calculate_reversion_speed(&self) -> Result<f64> { Ok(0.0) }
    fn calculate_reversion_probability(&self) -> Result<f64> { Ok(0.0) }
    fn calculate_support_resistance(&self) -> Result<f64> { Ok(0.0) }
    fn calculate_range_position(&self) -> Result<f64> { Ok(0.0) }
    fn calculate_rsi_like_indicator(&self) -> Result<f64> { Ok(0.0) }
    fn calculate_cyclical_component(&self) -> Result<f64> { Ok(0.0) }
    fn classify_volatility_regime(&self) -> Result<f64> { Ok(0.5) }
    fn classify_trend_regime(&self) -> Result<f64> { Ok(0.5) }
    fn classify_liquidity_regime(&self) -> Result<f64> { Ok(0.5) }
    fn classify_correlation_regime(&self) -> Result<f64> { Ok(0.5) }
    fn calculate_regime_persistence(&self) -> Result<f64> { Ok(0.5) }
    fn calculate_transition_probability(&self) -> Result<f64> { Ok(0.5) }
    fn calculate_regime_stability(&self) -> Result<f64> { Ok(0.5) }
    
    fn calculate_alpha_confidence(&self, _momentum_persistence: f64, _regime_stability: f64, _liquidity_density: f64) -> Result<f64> {
        Ok(0.7) // 默认置信度
    }
    
    fn apply_decay_adjustment(&self, alpha: f64) -> Result<f64> {
        // 简化的衰减调整
        Ok(alpha * 0.95)
    }
}

/// Alpha特征服务
#[derive(Debug)]
pub struct AlphaFeatureService {
    calculators: Arc<RwLock<HashMap<String, MicrostructureAlphaCalculator>>>,
    default_config: AlphaConfig,
}

impl AlphaFeatureService {
    pub fn new(config: AlphaConfig) -> Self {
        Self {
            calculators: Arc::new(RwLock::new(HashMap::new())),
            default_config: config,
        }
    }

    /// 更新符号的市场数据并计算特征
    pub async fn update_and_calculate(&self, symbol: String, snapshot: BookSnapshot, trades: Vec<TradeData>) -> Result<MicrostructureAlphaFeatures> {
        let mut calculators = self.calculators.write().await;
        let calculator = calculators
            .entry(symbol.clone())
            .or_insert_with(|| MicrostructureAlphaCalculator::new(symbol.clone(), self.default_config.clone()));

        calculator.update_market_data(snapshot, trades)?;
        calculator.calculate_features()
    }

    /// 获取符号的最新Alpha特征
    pub async fn get_latest_features(&self, symbol: &str) -> Option<MicrostructureAlphaFeatures> {
        let calculators = self.calculators.read().await;
        calculators.get(symbol).and_then(|calc| calc.cached_features.clone())
    }

    /// 批量计算多个符号的特征
    pub async fn batch_calculate(&self, updates: Vec<(String, BookSnapshot, Vec<TradeData>)>) -> Result<Vec<(String, MicrostructureAlphaFeatures)>> {
        let mut results = Vec::new();
        
        for (symbol, snapshot, trades) in updates {
            match self.update_and_calculate(symbol.clone(), snapshot, trades).await {
                Ok(features) => results.push((symbol, features)),
                Err(e) => {
                    warn!("Failed to calculate features for {}: {}", symbol, e);
                    continue;
                }
            }
        }
        
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_snapshot(symbol: &str) -> BookSnapshot {
        BookSnapshot {
            symbol: symbol.to_string(),
            timestamp: Utc::now(),
            sequence: 1,
            bids: vec![
                BookLevel { price: 100.0, size: 10.0, order_count: 5 },
                BookLevel { price: 99.0, size: 20.0, order_count: 3 },
            ],
            asks: vec![
                BookLevel { price: 101.0, size: 15.0, order_count: 7 },
                BookLevel { price: 102.0, size: 25.0, order_count: 4 },
            ],
        }
    }

    fn create_test_trades(symbol: &str) -> Vec<TradeData> {
        vec![
            TradeData {
                symbol: symbol.to_string(),
                timestamp: Utc::now(),
                price: 100.5,
                size: 5.0,
                side: TradeSide::Buy,
                trade_id: "1".to_string(),
            },
        ]
    }

    #[test]
    fn test_alpha_calculator_creation() {
        let config = AlphaConfig::default();
        let calculator = MicrostructureAlphaCalculator::new("BTCUSD".to_string(), config);
        assert_eq!(calculator.symbol, "BTCUSD");
    }

    #[tokio::test]
    async fn test_alpha_service() {
        let config = AlphaConfig::default();
        let service = AlphaFeatureService::new(config);
        
        let snapshot = create_test_snapshot("BTCUSD");
        let trades = create_test_trades("BTCUSD");
        
        // 需要多次更新才能计算特征
        for _ in 0..15 {
            let _ = service.update_and_calculate("BTCUSD".to_string(), snapshot.clone(), trades.clone()).await;
        }
        
        let result = service.update_and_calculate("BTCUSD".to_string(), snapshot, trades).await;
        assert!(result.is_ok());
    }
}