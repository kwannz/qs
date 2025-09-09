use crate::microstructure::{BookSnapshot, TradeData, TradeSide};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// 流动性毒性检测器 - AG3 VPIN模型
pub struct ToxicityDetector {
    vpin_window: usize,
    volume_bucket_size: f64,
    alpha_smoothing: f64,
}

/// 毒性检测特征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToxicityFeatures {
    pub vpin: f64,                   // VPIN毒性指标
    pub price_impact: f64,           // 价格冲击
    pub volume_sync_prob: f64,       // 成交量同步概率
    pub tick_intensity: f64,         // Tick强度
    pub order_flow_toxicity: f64,    // 订单流毒性
    pub adverse_selection: f64,      // 逆向选择成本
    pub effective_spread_ratio: f64, // 有效价差比率
    pub voi_imbalance: f64,          // VOI（Volume-Order Imbalance）
}

impl ToxicityDetector {
    pub fn new() -> Self {
        Self {
            vpin_window: 50,
            volume_bucket_size: 1000.0,
            alpha_smoothing: 0.1,
        }
    }

    pub fn with_params(vpin_window: usize, volume_bucket_size: f64, alpha_smoothing: f64) -> Self {
        Self {
            vpin_window,
            volume_bucket_size,
            alpha_smoothing,
        }
    }

    /// 检测流动性毒性
    pub fn detect(
        &mut self,
        trade_history: &VecDeque<TradeData>,
        book_history: &VecDeque<BookSnapshot>,
    ) -> Result<ToxicityFeatures> {
        if trade_history.is_empty() || book_history.is_empty() {
            return Ok(ToxicityFeatures::default());
        }

        // 计算VPIN
        let vpin = self.calculate_vpin(trade_history)?;
        
        // 计算价格冲击
        let price_impact = self.calculate_price_impact(trade_history)?;
        
        // 计算成交量同步概率
        let volume_sync_prob = self.calculate_volume_sync_probability(trade_history)?;
        
        // 计算Tick强度
        let tick_intensity = self.calculate_tick_intensity(trade_history, book_history)?;
        
        // 计算订单流毒性
        let order_flow_toxicity = self.calculate_order_flow_toxicity(trade_history, book_history)?;
        
        // 计算逆向选择成本
        let adverse_selection = self.calculate_adverse_selection(trade_history, book_history)?;
        
        // 计算有效价差比率
        let effective_spread_ratio = self.calculate_effective_spread_ratio(trade_history, book_history)?;
        
        // 计算VOI不平衡
        let voi_imbalance = self.calculate_voi_imbalance(trade_history, book_history)?;

        Ok(ToxicityFeatures {
            vpin,
            price_impact,
            volume_sync_prob,
            tick_intensity,
            order_flow_toxicity,
            adverse_selection,
            effective_spread_ratio,
            voi_imbalance,
        })
    }

    /// 计算VPIN (Volume-Synchronized Probability of Informed Trading)
    fn calculate_vpin(&self, trade_history: &VecDeque<TradeData>) -> Result<f64> {
        if trade_history.len() < 10 {
            return Ok(0.0);
        }

        let trades: Vec<&TradeData> = trade_history.iter()
            .rev()
            .take(self.vpin_window)
            .collect();

        // 按成交量分桶
        let mut volume_buckets = Vec::new();
        let mut current_bucket_volume = 0.0;
        let mut current_bucket_buy_volume = 0.0;
        let mut current_bucket_sell_volume = 0.0;

        for trade in trades.iter().rev() {
            current_bucket_volume += trade.size;
            
            match trade.side {
                TradeSide::Buy => current_bucket_buy_volume += trade.size,
                TradeSide::Sell => current_bucket_sell_volume += trade.size,
                TradeSide::Unknown => {
                    // 简单分配策略
                    current_bucket_buy_volume += trade.size * 0.5;
                    current_bucket_sell_volume += trade.size * 0.5;
                }
            }

            if current_bucket_volume >= self.volume_bucket_size {
                let imbalance = (current_bucket_buy_volume - current_bucket_sell_volume).abs() / current_bucket_volume;
                volume_buckets.push(imbalance);
                
                // 重置桶
                current_bucket_volume = 0.0;
                current_bucket_buy_volume = 0.0;
                current_bucket_sell_volume = 0.0;
            }
        }

        if volume_buckets.is_empty() {
            return Ok(0.0);
        }

        // 计算VPIN（平均不平衡度）
        let vpin = volume_buckets.iter().sum::<f64>() / volume_buckets.len() as f64;
        Ok(vpin.min(1.0)) // 限制在[0,1]范围内
    }

    /// 计算价格冲击
    fn calculate_price_impact(&self, trade_history: &VecDeque<TradeData>) -> Result<f64> {
        if trade_history.len() < 5 {
            return Ok(0.0);
        }

        let mut impacts = Vec::new();
        let recent_trades: Vec<&TradeData> = trade_history.iter().rev().take(20).collect();

        for i in 1..recent_trades.len() {
            let current_trade = recent_trades[i];
            let prev_trade = recent_trades[i - 1];
            
            if prev_trade.price > 0.0 {
                let price_change = (current_trade.price - prev_trade.price) / prev_trade.price;
                let size_impact = current_trade.size.ln(); // 对数成交量
                
                // 价格冲击 = 价格变化 / 对数成交量
                if size_impact > 0.0 {
                    impacts.push(price_change.abs() / size_impact);
                }
            }
        }

        if impacts.is_empty() {
            return Ok(0.0);
        }

        Ok(impacts.iter().sum::<f64>() / impacts.len() as f64)
    }

    /// 计算成交量同步概率
    fn calculate_volume_sync_probability(&self, trade_history: &VecDeque<TradeData>) -> Result<f64> {
        if trade_history.len() < 10 {
            return Ok(0.0);
        }

        let recent_trades: Vec<&TradeData> = trade_history.iter().rev().take(50).collect();
        let mut sync_events = 0;
        let mut total_events = 0;

        // 检查相邻交易的成交量和方向相关性
        for i in 1..recent_trades.len() {
            let current_trade = recent_trades[i];
            let prev_trade = recent_trades[i - 1];
            
            total_events += 1;
            
            // 成交量差异
            let volume_ratio = (current_trade.size / prev_trade.size).min(prev_trade.size / current_trade.size);
            
            // 方向一致性
            let direction_sync = match (current_trade.side, prev_trade.side) {
                (TradeSide::Buy, TradeSide::Buy) | (TradeSide::Sell, TradeSide::Sell) => true,
                _ => false,
            };
            
            // 如果成交量相近且方向一致，认为是同步事件
            if volume_ratio > 0.7 && direction_sync {
                sync_events += 1;
            }
        }

        if total_events == 0 {
            return Ok(0.0);
        }

        Ok(sync_events as f64 / total_events as f64)
    }

    /// 计算Tick强度
    fn calculate_tick_intensity(
        &self,
        trade_history: &VecDeque<TradeData>,
        book_history: &VecDeque<BookSnapshot>,
    ) -> Result<f64> {
        if trade_history.len() < 5 || book_history.is_empty() {
            return Ok(0.0);
        }

        let current_book = book_history.back().unwrap();
        let spread = if !current_book.bids.is_empty() && !current_book.asks.is_empty() {
            current_book.asks[0].price - current_book.bids[0].price
        } else {
            return Ok(0.0);
        };

        if spread <= 0.0 {
            return Ok(0.0);
        }

        let recent_trades: Vec<&TradeData> = trade_history.iter().rev().take(100).collect();
        let mut tick_moves = 0;
        let mut total_trades = 0;

        for i in 1..recent_trades.len() {
            let current_trade = recent_trades[i];
            let prev_trade = recent_trades[i - 1];
            
            total_trades += 1;
            let price_diff = (current_trade.price - prev_trade.price).abs();
            
            // 如果价格变化接近一个tick（价差的一半），记录为tick移动
            if price_diff >= spread * 0.3 && price_diff <= spread * 2.0 {
                tick_moves += 1;
            }
        }

        if total_trades == 0 {
            return Ok(0.0);
        }

        Ok(tick_moves as f64 / total_trades as f64)
    }

    /// 计算订单流毒性
    fn calculate_order_flow_toxicity(
        &self,
        trade_history: &VecDeque<TradeData>,
        book_history: &VecDeque<BookSnapshot>,
    ) -> Result<f64> {
        if trade_history.len() < 5 || book_history.len() < 2 {
            return Ok(0.0);
        }

        let mut toxicity_scores = Vec::new();
        let window_size = std::cmp::min(20, trade_history.len());

        for i in 0..window_size {
            if let Some(trade) = trade_history.iter().rev().nth(i) {
                // 找到交易时间最接近的订单簿快照
                let book_before = self.find_closest_book(book_history, trade.timestamp, true)?;
                let book_after = self.find_closest_book(book_history, trade.timestamp, false)?;

                if let (Some(before), Some(after)) = (book_before, book_after) {
                    // 计算交易前后的订单簿变化
                    let depth_change = self.calculate_depth_impact(before, after, trade);
                    let spread_change = self.calculate_spread_impact(before, after);
                    
                    // 毒性评分：深度减少和价差增加都是毒性信号
                    let toxicity = depth_change + spread_change;
                    toxicity_scores.push(toxicity);
                }
            }
        }

        if toxicity_scores.is_empty() {
            return Ok(0.0);
        }

        Ok(toxicity_scores.iter().sum::<f64>() / toxicity_scores.len() as f64)
    }

    /// 计算逆向选择成本
    fn calculate_adverse_selection(
        &self,
        trade_history: &VecDeque<TradeData>,
        book_history: &VecDeque<BookSnapshot>,
    ) -> Result<f64> {
        if trade_history.len() < 10 || book_history.is_empty() {
            return Ok(0.0);
        }

        let current_book = book_history.back().unwrap();
        if current_book.bids.is_empty() || current_book.asks.is_empty() {
            return Ok(0.0);
        }

        let mid_price = (current_book.bids[0].price + current_book.asks[0].price) / 2.0;
        let spread = current_book.asks[0].price - current_book.bids[0].price;

        let recent_trades: Vec<&TradeData> = trade_history.iter().rev().take(50).collect();
        let mut adverse_costs = Vec::new();

        for trade in recent_trades {
            // 计算交易价格相对于中间价的偏离
            let price_deviation = match trade.side {
                TradeSide::Buy => trade.price - mid_price,
                TradeSide::Sell => mid_price - trade.price,
                TradeSide::Unknown => (trade.price - mid_price).abs(),
            };

            // 标准化为价差的比例
            if spread > 0.0 {
                let normalized_cost = price_deviation / spread;
                adverse_costs.push(normalized_cost.max(0.0)); // 只考虑正向成本
            }
        }

        if adverse_costs.is_empty() {
            return Ok(0.0);
        }

        Ok(adverse_costs.iter().sum::<f64>() / adverse_costs.len() as f64)
    }

    /// 计算有效价差比率
    fn calculate_effective_spread_ratio(
        &self,
        trade_history: &VecDeque<TradeData>,
        book_history: &VecDeque<BookSnapshot>,
    ) -> Result<f64> {
        if trade_history.len() < 5 || book_history.is_empty() {
            return Ok(0.0);
        }

        let current_book = book_history.back().unwrap();
        if current_book.bids.is_empty() || current_book.asks.is_empty() {
            return Ok(0.0);
        }

        let quoted_spread = current_book.asks[0].price - current_book.bids[0].price;
        if quoted_spread <= 0.0 {
            return Ok(1.0);
        }

        let mid_price = (current_book.bids[0].price + current_book.asks[0].price) / 2.0;
        let recent_trades: Vec<&TradeData> = trade_history.iter().rev().take(20).collect();

        let mut effective_spreads = Vec::new();
        for trade in recent_trades {
            let effective_spread = 2.0 * (trade.price - mid_price).abs();
            effective_spreads.push(effective_spread);
        }

        if effective_spreads.is_empty() {
            return Ok(1.0);
        }

        let avg_effective_spread = effective_spreads.iter().sum::<f64>() / effective_spreads.len() as f64;
        Ok((avg_effective_spread / quoted_spread).min(5.0)) // 限制最大比率
    }

    /// 计算VOI不平衡
    fn calculate_voi_imbalance(
        &self,
        trade_history: &VecDeque<TradeData>,
        book_history: &VecDeque<BookSnapshot>,
    ) -> Result<f64> {
        if trade_history.len() < 5 || book_history.len() < 2 {
            return Ok(0.0);
        }

        // VOI = Volume - Order Imbalance
        // 基于最近的订单簿变化和交易量计算
        
        let current_book = book_history.back().unwrap();
        let prev_book = &book_history[book_history.len() - 2];

        // 计算订单簿深度变化
        let current_bid_depth: f64 = current_book.bids.iter().take(5).map(|l| l.size).sum();
        let current_ask_depth: f64 = current_book.asks.iter().take(5).map(|l| l.size).sum();
        let prev_bid_depth: f64 = prev_book.bids.iter().take(5).map(|l| l.size).sum();
        let prev_ask_depth: f64 = prev_book.asks.iter().take(5).map(|l| l.size).sum();

        let depth_imbalance = ((current_bid_depth - prev_bid_depth) - (current_ask_depth - prev_ask_depth)) /
                             (current_bid_depth + current_ask_depth + prev_bid_depth + prev_ask_depth);

        // 结合最近的交易量不平衡
        let recent_trades: Vec<&TradeData> = trade_history.iter().rev().take(10).collect();
        let mut buy_volume = 0.0;
        let mut sell_volume = 0.0;

        for trade in recent_trades {
            match trade.side {
                TradeSide::Buy => buy_volume += trade.size,
                TradeSide::Sell => sell_volume += trade.size,
                TradeSide::Unknown => {
                    buy_volume += trade.size * 0.5;
                    sell_volume += trade.size * 0.5;
                }
            }
        }

        let volume_imbalance = if buy_volume + sell_volume > 0.0 {
            (buy_volume - sell_volume) / (buy_volume + sell_volume)
        } else {
            0.0
        };

        // 综合VOI不平衡
        Ok((depth_imbalance + volume_imbalance) / 2.0)
    }

    // 辅助方法
    fn find_closest_book<'a>(
        &self,
        book_history: &'a VecDeque<BookSnapshot>,
        target_time: chrono::DateTime<chrono::Utc>,
        before: bool,
    ) -> Result<Option<&'a BookSnapshot>> {
        for book in book_history.iter().rev() {
            if before && book.timestamp <= target_time {
                return Ok(Some(book));
            } else if !before && book.timestamp >= target_time {
                return Ok(Some(book));
            }
        }
        Ok(None)
    }

    fn calculate_depth_impact(&self, before: &BookSnapshot, after: &BookSnapshot, trade: &TradeData) -> f64 {
        let before_depth: f64 = before.bids.iter().take(5).map(|l| l.size).sum::<f64>() +
                               before.asks.iter().take(5).map(|l| l.size).sum::<f64>();
        let after_depth: f64 = after.bids.iter().take(5).map(|l| l.size).sum::<f64>() +
                              after.asks.iter().take(5).map(|l| l.size).sum::<f64>();
        
        if before_depth > 0.0 {
            ((before_depth - after_depth) / before_depth).max(0.0)
        } else {
            0.0
        }
    }

    fn calculate_spread_impact(&self, before: &BookSnapshot, after: &BookSnapshot) -> f64 {
        if before.bids.is_empty() || before.asks.is_empty() || after.bids.is_empty() || after.asks.is_empty() {
            return 0.0;
        }

        let before_spread = before.asks[0].price - before.bids[0].price;
        let after_spread = after.asks[0].price - after.bids[0].price;

        if before_spread > 0.0 {
            ((after_spread - before_spread) / before_spread).max(0.0)
        } else {
            0.0
        }
    }
}

impl Default for ToxicityDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ToxicityFeatures {
    fn default() -> Self {
        Self {
            vpin: 0.0,
            price_impact: 0.0,
            volume_sync_prob: 0.0,
            tick_intensity: 0.0,
            order_flow_toxicity: 0.0,
            adverse_selection: 0.0,
            effective_spread_ratio: 1.0,
            voi_imbalance: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_trade() -> TradeData {
        TradeData {
            symbol: "BTCUSDT".to_string(),
            timestamp: Utc::now(),
            price: 50000.0,
            size: 1.0,
            side: TradeSide::Buy,
            trade_id: "test_1".to_string(),
        }
    }

    #[test]
    fn test_toxicity_detection() {
        let mut detector = ToxicityDetector::new();
        let trade = create_test_trade();
        let mut trade_history = VecDeque::new();
        let book_history = VecDeque::new();
        
        trade_history.push_back(trade);
        
        let features = detector.detect(&trade_history, &book_history).unwrap();
        assert!(features.vpin >= 0.0 && features.vpin <= 1.0);
    }

    #[test]
    fn test_vpin_calculation() {
        let mut detector = ToxicityDetector::new();
        let mut trade_history = VecDeque::new();
        
        // 添加一些测试交易
        for i in 0..20 {
            let mut trade = create_test_trade();
            trade.size = 100.0;
            trade.side = if i % 2 == 0 { TradeSide::Buy } else { TradeSide::Sell };
            trade_history.push_back(trade);
        }
        
        let vpin = detector.calculate_vpin(&trade_history).unwrap();
        assert!(vpin >= 0.0 && vpin <= 1.0);
    }
}