use crate::microstructure::{BookSnapshot, BookLevel};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// 订单簿不平衡计算器 - AG3核心算法
pub struct OBICalculator {
    lookback_window: usize,
    alpha_ema: f64,  // EMA平滑参数
}

/// OBI特征输出
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OBIFeatures {
    pub imbalance: f64,              // 基础OBI
    pub weighted_imbalance: f64,     // 加权OBI
    pub spread: f64,                 // 买卖价差
    pub mid_price: f64,              // 中间价
    pub weighted_mid_price: f64,     // 成交量加权中间价
    pub momentum: f64,               // 价格动量
    pub reversion_signal: f64,       // 均值回归信号
    pub volatility: f64,             // 短期波动率
    pub flow_imbalance: f64,         // 订单流不平衡
    pub depth_ratio: f64,            // 深度比率
}

impl OBICalculator {
    pub fn new() -> Self {
        Self {
            lookback_window: 20,
            alpha_ema: 0.1,
        }
    }

    pub fn with_params(lookback_window: usize, alpha_ema: f64) -> Self {
        Self {
            lookback_window,
            alpha_ema,
        }
    }

    /// 计算订单簿不平衡特征
    pub fn calculate(&mut self, book_history: &VecDeque<BookSnapshot>) -> Result<OBIFeatures> {
        if book_history.is_empty() {
            return Err(anyhow::anyhow!("Empty book history"));
        }

        let current_book = book_history.back().unwrap();
        
        // 基础价格和深度计算
        let (mid_price, spread) = self.calculate_basic_prices(current_book)?;
        let (bid_depth, ask_depth, total_depth) = self.calculate_depths(current_book);
        
        // 计算基础OBI
        let basic_obi = self.calculate_basic_obi(bid_depth, ask_depth);
        
        // 计算加权OBI（考虑价格距离）
        let weighted_obi = self.calculate_weighted_obi(current_book)?;
        
        // 计算成交量加权中间价
        let weighted_mid_price = self.calculate_weighted_mid_price(current_book)?;
        
        // 计算动量和均值回归信号
        let (momentum, reversion_signal) = self.calculate_momentum_signals(book_history)?;
        
        // 计算短期波动率
        let volatility = self.calculate_short_term_volatility(book_history)?;
        
        // 计算订单流不平衡
        let flow_imbalance = self.calculate_flow_imbalance(book_history)?;
        
        // 计算深度比率
        let depth_ratio = if ask_depth > 0.0 { bid_depth / ask_depth } else { 0.0 };

        Ok(OBIFeatures {
            imbalance: basic_obi,
            weighted_imbalance: weighted_obi,
            spread,
            mid_price,
            weighted_mid_price,
            momentum,
            reversion_signal,
            volatility,
            flow_imbalance,
            depth_ratio,
        })
    }

    /// 计算基础价格信息
    fn calculate_basic_prices(&self, book: &BookSnapshot) -> Result<(f64, f64)> {
        let best_bid = book.bids.first()
            .ok_or_else(|| anyhow::anyhow!("No bids available"))?
            .price;
        let best_ask = book.asks.first()
            .ok_or_else(|| anyhow::anyhow!("No asks available"))?
            .price;
            
        let mid_price = (best_bid + best_ask) / 2.0;
        let spread = best_ask - best_bid;
        
        Ok((mid_price, spread))
    }

    /// 计算订单簿深度
    fn calculate_depths(&self, book: &BookSnapshot) -> (f64, f64, f64) {
        let bid_depth: f64 = book.bids.iter()
            .take(5)  // 取前5档
            .map(|level| level.size)
            .sum();
            
        let ask_depth: f64 = book.asks.iter()
            .take(5)  // 取前5档
            .map(|level| level.size)
            .sum();
            
        let total_depth = bid_depth + ask_depth;
        
        (bid_depth, ask_depth, total_depth)
    }

    /// 计算基础OBI（传统公式）
    fn calculate_basic_obi(&self, bid_depth: f64, ask_depth: f64) -> f64 {
        if bid_depth + ask_depth == 0.0 {
            return 0.0;
        }
        
        (bid_depth - ask_depth) / (bid_depth + ask_depth)
    }

    /// 计算加权OBI - 考虑价格距离的权重
    fn calculate_weighted_obi(&self, book: &BookSnapshot) -> Result<f64> {
        let best_bid = book.bids.first().unwrap().price;
        let best_ask = book.asks.first().unwrap().price;
        let mid_price = (best_bid + best_ask) / 2.0;

        // 计算加权买方深度
        let weighted_bid_depth: f64 = book.bids.iter()
            .take(10)
            .map(|level| {
                let weight = 1.0 / (1.0 + (mid_price - level.price) / mid_price);
                level.size * weight
            })
            .sum();

        // 计算加权卖方深度
        let weighted_ask_depth: f64 = book.asks.iter()
            .take(10)
            .map(|level| {
                let weight = 1.0 / (1.0 + (level.price - mid_price) / mid_price);
                level.size * weight
            })
            .sum();

        if weighted_bid_depth + weighted_ask_depth == 0.0 {
            return Ok(0.0);
        }

        Ok((weighted_bid_depth - weighted_ask_depth) / (weighted_bid_depth + weighted_ask_depth))
    }

    /// 计算成交量加权中间价
    fn calculate_weighted_mid_price(&self, book: &BookSnapshot) -> Result<f64> {
        let mut weighted_price = 0.0;
        let mut total_weight = 0.0;

        // 买方权重
        for level in book.bids.iter().take(5) {
            weighted_price += level.price * level.size;
            total_weight += level.size;
        }

        // 卖方权重
        for level in book.asks.iter().take(5) {
            weighted_price += level.price * level.size;
            total_weight += level.size;
        }

        if total_weight > 0.0 {
            Ok(weighted_price / total_weight)
        } else {
            // 退回到简单中间价
            let best_bid = book.bids.first().unwrap().price;
            let best_ask = book.asks.first().unwrap().price;
            Ok((best_bid + best_ask) / 2.0)
        }
    }

    /// 计算动量和均值回归信号
    fn calculate_momentum_signals(&self, book_history: &VecDeque<BookSnapshot>) -> Result<(f64, f64)> {
        if book_history.len() < 3 {
            return Ok((0.0, 0.0));
        }

        let mut mid_prices = Vec::new();
        for book in book_history.iter().rev().take(self.lookback_window) {
            let best_bid = book.bids.first().unwrap().price;
            let best_ask = book.asks.first().unwrap().price;
            mid_prices.push((best_bid + best_ask) / 2.0);
        }

        if mid_prices.len() < 3 {
            return Ok((0.0, 0.0));
        }

        // 动量计算（短期趋势）
        let current_price = mid_prices[0];
        let prev_price = mid_prices[mid_prices.len() / 2];
        let momentum = (current_price - prev_price) / prev_price;

        // 均值回归信号（偏离均值程度）
        let mean_price: f64 = mid_prices.iter().sum::<f64>() / mid_prices.len() as f64;
        let reversion_signal = (current_price - mean_price) / mean_price;

        Ok((momentum, reversion_signal))
    }

    /// 计算短期波动率
    fn calculate_short_term_volatility(&self, book_history: &VecDeque<BookSnapshot>) -> Result<f64> {
        if book_history.len() < 5 {
            return Ok(0.0);
        }

        let mut returns = Vec::new();
        let mut prev_mid = None;

        for book in book_history.iter().rev().take(20) {
            let best_bid = book.bids.first().unwrap().price;
            let best_ask = book.asks.first().unwrap().price;
            let mid_price = (best_bid + best_ask) / 2.0;

            if let Some(prev_price) = prev_mid {
                if prev_price > 0.0 {
                    let return_rate = (mid_price - prev_price) / prev_price;
                    returns.push(return_rate);
                }
            }
            prev_mid = Some(mid_price);
        }

        if returns.is_empty() {
            return Ok(0.0);
        }

        // 计算收益率标准差
        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;

        Ok(variance.sqrt())
    }

    /// 计算订单流不平衡（基于订单数量）
    fn calculate_flow_imbalance(&self, book_history: &VecDeque<BookSnapshot>) -> Result<f64> {
        if book_history.len() < 2 {
            return Ok(0.0);
        }

        let current_book = book_history.back().unwrap();
        let prev_book = &book_history[book_history.len() - 2];

        // 计算订单数量变化
        let current_bid_orders: u32 = current_book.bids.iter().take(5).map(|l| l.order_count).sum();
        let current_ask_orders: u32 = current_book.asks.iter().take(5).map(|l| l.order_count).sum();

        let prev_bid_orders: u32 = prev_book.bids.iter().take(5).map(|l| l.order_count).sum();
        let prev_ask_orders: u32 = prev_book.asks.iter().take(5).map(|l| l.order_count).sum();

        let bid_flow = current_bid_orders as f64 - prev_bid_orders as f64;
        let ask_flow = current_ask_orders as f64 - prev_ask_orders as f64;

        if bid_flow + ask_flow == 0.0 {
            return Ok(0.0);
        }

        Ok((bid_flow - ask_flow) / (bid_flow.abs() + ask_flow.abs()))
    }
}

impl Default for OBICalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_book() -> BookSnapshot {
        BookSnapshot {
            symbol: "BTCUSDT".to_string(),
            timestamp: Utc::now(),
            bids: vec![
                BookLevel { price: 50000.0, size: 1.0, order_count: 5 },
                BookLevel { price: 49999.0, size: 2.0, order_count: 3 },
                BookLevel { price: 49998.0, size: 1.5, order_count: 2 },
            ],
            asks: vec![
                BookLevel { price: 50001.0, size: 0.8, order_count: 4 },
                BookLevel { price: 50002.0, size: 1.2, order_count: 2 },
                BookLevel { price: 50003.0, size: 2.0, order_count: 6 },
            ],
            sequence: 1,
        }
    }

    #[test]
    fn test_obi_calculation() {
        let mut calculator = OBICalculator::new();
        let book = create_test_book();
        let mut history = VecDeque::new();
        history.push_back(book);

        let features = calculator.calculate(&history).unwrap();
        assert!(features.imbalance.abs() <= 1.0);  // OBI应该在[-1, 1]范围内
        assert!(features.spread > 0.0);  // 价差应该为正
    }

    #[test]
    fn test_basic_obi() {
        let calculator = OBICalculator::new();
        let obi = calculator.calculate_basic_obi(100.0, 50.0);
        assert!((obi - 1.0/3.0).abs() < 1e-6);
    }

    #[test]
    fn test_depths_calculation() {
        let calculator = OBICalculator::new();
        let book = create_test_book();
        let (bid_depth, ask_depth, total_depth) = calculator.calculate_depths(&book);
        
        assert!(bid_depth > 0.0);
        assert!(ask_depth > 0.0);
        assert_eq!(total_depth, bid_depth + ask_depth);
    }
}