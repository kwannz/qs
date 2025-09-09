use crate::microstructure::{BookSnapshot, BookLevel};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// 队列分析器 - AG3队列不平衡分析
pub struct QueueAnalyzer {
    depth_levels: usize,
    time_window: usize,
}

/// 队列分析特征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueFeatures {
    pub imbalance: f64,              // 队列不平衡
    pub bid_depth: f64,              // 买方深度
    pub ask_depth: f64,              // 卖方深度
    pub total_depth: f64,            // 总深度
    pub weighted_imbalance: f64,     // 加权队列不平衡
    pub queue_intensity: f64,        // 队列强度
    pub order_arrival_rate: f64,     // 订单到达率
    pub queue_position_advantage: f64, // 队列位置优势
    pub depth_stability: f64,        // 深度稳定性
}

impl QueueAnalyzer {
    pub fn new() -> Self {
        Self {
            depth_levels: 10,
            time_window: 50,
        }
    }

    pub fn with_params(depth_levels: usize, time_window: usize) -> Self {
        Self {
            depth_levels,
            time_window,
        }
    }

    /// 分析队列特征
    pub fn analyze(&mut self, book_history: &VecDeque<BookSnapshot>) -> Result<QueueFeatures> {
        if book_history.is_empty() {
            return Err(anyhow::anyhow!("Empty book history"));
        }

        let current_book = book_history.back().unwrap();
        
        // 计算基础深度
        let (bid_depth, ask_depth, total_depth) = self.calculate_depth(current_book);
        
        // 计算基础队列不平衡
        let basic_imbalance = self.calculate_basic_imbalance(bid_depth, ask_depth);
        
        // 计算加权队列不平衡
        let weighted_imbalance = self.calculate_weighted_imbalance(current_book)?;
        
        // 计算队列强度
        let queue_intensity = self.calculate_queue_intensity(current_book);
        
        // 计算订单到达率
        let order_arrival_rate = self.calculate_order_arrival_rate(book_history)?;
        
        // 计算队列位置优势
        let queue_position_advantage = self.calculate_position_advantage(current_book)?;
        
        // 计算深度稳定性
        let depth_stability = self.calculate_depth_stability(book_history)?;

        Ok(QueueFeatures {
            imbalance: basic_imbalance,
            bid_depth,
            ask_depth,
            total_depth,
            weighted_imbalance,
            queue_intensity,
            order_arrival_rate,
            queue_position_advantage,
            depth_stability,
        })
    }

    /// 计算订单簿深度
    fn calculate_depth(&self, book: &BookSnapshot) -> (f64, f64, f64) {
        let bid_depth: f64 = book.bids.iter()
            .take(self.depth_levels)
            .map(|level| level.size)
            .sum();
            
        let ask_depth: f64 = book.asks.iter()
            .take(self.depth_levels)
            .map(|level| level.size)
            .sum();
            
        let total_depth = bid_depth + ask_depth;
        
        (bid_depth, ask_depth, total_depth)
    }

    /// 计算基础队列不平衡
    fn calculate_basic_imbalance(&self, bid_depth: f64, ask_depth: f64) -> f64 {
        if bid_depth + ask_depth == 0.0 {
            return 0.0;
        }
        
        (bid_depth - ask_depth) / (bid_depth + ask_depth)
    }

    /// 计算加权队列不平衡 - 考虑订单数量权重
    fn calculate_weighted_imbalance(&self, book: &BookSnapshot) -> Result<f64> {
        if book.bids.is_empty() || book.asks.is_empty() {
            return Ok(0.0);
        }

        // 计算加权买方强度（深度 × 订单数量）
        let weighted_bid_strength: f64 = book.bids.iter()
            .take(self.depth_levels)
            .map(|level| level.size * (level.order_count as f64).sqrt()) // 订单数量开方权重
            .sum();

        // 计算加权卖方强度
        let weighted_ask_strength: f64 = book.asks.iter()
            .take(self.depth_levels)
            .map(|level| level.size * (level.order_count as f64).sqrt())
            .sum();

        if weighted_bid_strength + weighted_ask_strength == 0.0 {
            return Ok(0.0);
        }

        Ok((weighted_bid_strength - weighted_ask_strength) / 
           (weighted_bid_strength + weighted_ask_strength))
    }

    /// 计算队列强度 - 基于订单密度
    fn calculate_queue_intensity(&self, book: &BookSnapshot) -> f64 {
        if book.bids.is_empty() || book.asks.is_empty() {
            return 0.0;
        }

        let best_bid = book.bids[0].price;
        let best_ask = book.asks[0].price;
        let spread = best_ask - best_bid;

        if spread <= 0.0 {
            return 0.0;
        }

        // 计算价差内的订单密度
        let bid_intensity = book.bids.iter()
            .take(5)
            .map(|level| level.order_count as f64 / spread)
            .sum::<f64>();

        let ask_intensity = book.asks.iter()
            .take(5)
            .map(|level| level.order_count as f64 / spread)
            .sum::<f64>();

        (bid_intensity + ask_intensity) / 2.0
    }

    /// 计算订单到达率
    fn calculate_order_arrival_rate(&self, book_history: &VecDeque<BookSnapshot>) -> Result<f64> {
        if book_history.len() < 2 {
            return Ok(0.0);
        }

        let window_size = std::cmp::min(self.time_window, book_history.len() - 1);
        let mut total_order_changes = 0u32;

        for i in 1..=window_size {
            let current_idx = book_history.len() - 1;
            let prev_idx = book_history.len() - 1 - i;

            let current_book = &book_history[current_idx];
            let prev_book = &book_history[prev_idx];

            // 计算订单数量变化
            let current_orders = self.count_total_orders(current_book);
            let prev_orders = self.count_total_orders(prev_book);

            total_order_changes += (current_orders as i32 - prev_orders as i32).abs() as u32;
        }

        // 平均每个时间步的订单变化率
        Ok(total_order_changes as f64 / window_size as f64)
    }

    /// 统计总订单数
    fn count_total_orders(&self, book: &BookSnapshot) -> u32 {
        let bid_orders: u32 = book.bids.iter().take(self.depth_levels).map(|l| l.order_count).sum();
        let ask_orders: u32 = book.asks.iter().take(self.depth_levels).map(|l| l.order_count).sum();
        bid_orders + ask_orders
    }

    /// 计算队列位置优势
    fn calculate_position_advantage(&self, book: &BookSnapshot) -> Result<f64> {
        if book.bids.is_empty() || book.asks.is_empty() {
            return Ok(0.0);
        }

        // 计算最佳价位的相对优势
        let best_bid_size = book.bids[0].size;
        let best_ask_size = book.asks[0].size;
        
        // 计算第二档的大小
        let second_bid_size = book.bids.get(1).map_or(0.0, |l| l.size);
        let second_ask_size = book.asks.get(1).map_or(0.0, |l| l.size);

        // 最佳价位相对于次优价位的优势
        let bid_advantage = if second_bid_size > 0.0 { 
            best_bid_size / second_bid_size 
        } else { 
            1.0 
        };
        let ask_advantage = if second_ask_size > 0.0 { 
            best_ask_size / second_ask_size 
        } else { 
            1.0 
        };

        // 综合优势指标
        Ok((bid_advantage + ask_advantage) / 2.0)
    }

    /// 计算深度稳定性
    fn calculate_depth_stability(&self, book_history: &VecDeque<BookSnapshot>) -> Result<f64> {
        if book_history.len() < 5 {
            return Ok(0.0);
        }

        let window_size = std::cmp::min(20, book_history.len());
        let mut depth_changes = Vec::new();

        for i in 1..window_size {
            let current_idx = book_history.len() - 1 - i + 1;
            let prev_idx = book_history.len() - 1 - i;

            let (current_bid, current_ask, _) = self.calculate_depth(&book_history[current_idx]);
            let (prev_bid, prev_ask, _) = self.calculate_depth(&book_history[prev_idx]);

            // 计算深度相对变化
            let bid_change = if prev_bid > 0.0 { 
                ((current_bid - prev_bid) / prev_bid).abs() 
            } else { 
                0.0 
            };
            let ask_change = if prev_ask > 0.0 { 
                ((current_ask - prev_ask) / prev_ask).abs() 
            } else { 
                0.0 
            };

            depth_changes.push((bid_change + ask_change) / 2.0);
        }

        if depth_changes.is_empty() {
            return Ok(0.0);
        }

        // 计算深度变化的标准差（稳定性的逆指标）
        let mean_change: f64 = depth_changes.iter().sum::<f64>() / depth_changes.len() as f64;
        let variance: f64 = depth_changes.iter()
            .map(|&change| (change - mean_change).powi(2))
            .sum::<f64>() / depth_changes.len() as f64;

        let stability = 1.0 / (1.0 + variance.sqrt()); // 转换为稳定性指标
        Ok(stability)
    }
}

impl Default for QueueAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_book_with_orders() -> BookSnapshot {
        BookSnapshot {
            symbol: "BTCUSDT".to_string(),
            timestamp: Utc::now(),
            bids: vec![
                BookLevel { price: 50000.0, size: 2.0, order_count: 10 },
                BookLevel { price: 49999.0, size: 1.5, order_count: 8 },
                BookLevel { price: 49998.0, size: 1.0, order_count: 5 },
            ],
            asks: vec![
                BookLevel { price: 50001.0, size: 1.0, order_count: 6 },
                BookLevel { price: 50002.0, size: 2.5, order_count: 12 },
                BookLevel { price: 50003.0, size: 1.8, order_count: 9 },
            ],
            sequence: 1,
        }
    }

    #[test]
    fn test_queue_analysis() {
        let mut analyzer = QueueAnalyzer::new();
        let book = create_test_book_with_orders();
        let mut history = VecDeque::new();
        history.push_back(book);

        let features = analyzer.analyze(&history).unwrap();
        assert!(features.imbalance.abs() <= 1.0);
        assert!(features.bid_depth > 0.0);
        assert!(features.ask_depth > 0.0);
        assert!(features.queue_intensity >= 0.0);
    }

    #[test]
    fn test_weighted_imbalance() {
        let analyzer = QueueAnalyzer::new();
        let book = create_test_book_with_orders();
        let weighted_imbalance = analyzer.calculate_weighted_imbalance(&book).unwrap();
        assert!(weighted_imbalance.abs() <= 1.0);
    }

    #[test]
    fn test_position_advantage() {
        let analyzer = QueueAnalyzer::new();
        let book = create_test_book_with_orders();
        let advantage = analyzer.calculate_position_advantage(&book).unwrap();
        assert!(advantage >= 0.0);
    }
}