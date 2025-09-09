pub mod order_book_imbalance;
pub mod queue_analysis;
pub mod flow_toxicity;
// pub mod trade_clustering;
// pub mod impact_modeling;
pub mod alpha_features;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use chrono::{DateTime, Utc};

/// AG3微结构数据特征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrostructureFeatures {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    
    // 订单簿特征
    pub order_book_imbalance: f64,      // OBI
    pub bid_ask_spread: f64,            // 买卖价差
    pub mid_price: f64,                 // 中间价
    pub weighted_mid_price: f64,        // 加权中间价
    
    // 队列特征  
    pub queue_imbalance: f64,           // 队列不平衡
    pub bid_depth: f64,                 // 买方深度
    pub ask_depth: f64,                 // 卖方深度
    pub total_depth: f64,               // 总深度
    
    // 流毒性指标
    pub flow_toxicity: f64,             // VPIN流毒性
    pub price_impact: f64,              // 价格冲击
    pub volume_sync_probability: f64,   // 成交量同步概率
    
    // 交易簇特征
    pub trade_cluster_id: u64,          // 交易簇ID
    pub cluster_intensity: f64,         // 簇强度
    pub cluster_duration: u64,          // 簇持续时间（毫秒）
    
    // 高频指标
    pub micro_price_trend: f64,         // 微观价格趋势
    pub tick_intensity: f64,            // Tick强度
    pub order_flow_imbalance: f64,      // 订单流不平衡
    
    // 预测性指标
    pub short_term_alpha: f64,          // 短期Alpha信号
    pub regime_probability: f64,        // 体制概率
    pub predicted_impact: f64,          // 预期冲击
}

/// 订单簿快照
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookSnapshot {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub bids: Vec<BookLevel>,
    pub asks: Vec<BookLevel>,
    pub sequence: u64,
}

/// 订单簿层级
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookLevel {
    pub price: f64,
    pub size: f64,
    pub order_count: u32,
}

/// 交易数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeData {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub size: f64,
    pub side: TradeSide,
    pub trade_id: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
    Unknown,
}

// 临时结构体（替代缺失的模块）
#[derive(Debug, Clone)]
struct ClusterFeatures {
    pub cluster_id: u64,
    pub intensity: f64,
    pub duration_ms: u64,
}

#[derive(Debug, Clone)]
struct ImpactFeatures {
    pub trend: f64,
    pub predicted_impact: f64,
}

/// 微结构特征计算引擎
pub struct MicrostructureEngine {
    symbol: String,
    book_history: VecDeque<BookSnapshot>,
    trade_history: VecDeque<TradeData>,
    max_history: usize,
    
    // 计算模块
    obi_calculator: order_book_imbalance::OBICalculator,
    queue_analyzer: queue_analysis::QueueAnalyzer,
    toxicity_detector: flow_toxicity::ToxicityDetector,
    // trade_clusterer: trade_clustering::TradeClusterer,
    // impact_model: impact_modeling::ImpactModel,
}

impl MicrostructureEngine {
    pub fn new(symbol: String, max_history: usize) -> Self {
        Self {
            symbol: symbol.clone(),
            book_history: VecDeque::with_capacity(max_history),
            trade_history: VecDeque::with_capacity(max_history),
            max_history,
            obi_calculator: order_book_imbalance::OBICalculator::new(),
            queue_analyzer: queue_analysis::QueueAnalyzer::new(),
            toxicity_detector: flow_toxicity::ToxicityDetector::new(),
            // trade_clusterer: trade_clustering::TradeClusterer::new(),
            // impact_model: impact_modeling::ImpactModel::new(),
        }
    }

    /// 更新订单簿快照
    pub fn update_book(&mut self, snapshot: BookSnapshot) -> Result<()> {
        if self.book_history.len() >= self.max_history {
            self.book_history.pop_front();
        }
        self.book_history.push_back(snapshot);
        Ok(())
    }

    /// 更新交易数据
    pub fn update_trade(&mut self, trade: TradeData) -> Result<()> {
        if self.trade_history.len() >= self.max_history {
            self.trade_history.pop_front();
        }
        self.trade_history.push_back(trade);
        Ok(())
    }

    /// 计算所有微结构特征
    pub fn compute_features(&mut self) -> Result<Option<MicrostructureFeatures>> {
        if self.book_history.is_empty() {
            return Ok(None);
        }

        let current_book = self.book_history.back().unwrap();
        let timestamp = current_book.timestamp;

        // 计算各类特征
        let obi_features = self.obi_calculator.calculate(&self.book_history)?;
        let queue_features = self.queue_analyzer.analyze(&self.book_history)?;
        let toxicity_features = self.toxicity_detector.detect(&self.trade_history, &self.book_history)?;
        // let cluster_features = self.trade_clusterer.analyze(&self.trade_history)?;
        // let impact_features = self.impact_model.predict(&self.trade_history, &self.book_history)?;
        
        // 临时默认值
        let cluster_features = ClusterFeatures {
            cluster_id: 0,
            intensity: 0.0,
            duration_ms: 0,
        };
        let impact_features = ImpactFeatures {
            trend: 0.0,
            predicted_impact: 0.0,
        };

        // 计算短期Alpha信号
        let short_term_alpha = self.compute_short_term_alpha(&obi_features, &toxicity_features)?;

        let features = MicrostructureFeatures {
            symbol: self.symbol.clone(),
            timestamp,
            
            // 订单簿特征
            order_book_imbalance: obi_features.imbalance,
            bid_ask_spread: obi_features.spread,
            mid_price: obi_features.mid_price,
            weighted_mid_price: obi_features.weighted_mid_price,
            
            // 队列特征
            queue_imbalance: queue_features.imbalance,
            bid_depth: queue_features.bid_depth,
            ask_depth: queue_features.ask_depth,
            total_depth: queue_features.total_depth,
            
            // 流毒性指标
            flow_toxicity: toxicity_features.vpin,
            price_impact: toxicity_features.price_impact,
            volume_sync_probability: toxicity_features.volume_sync_prob,
            
            // 交易簇特征
            trade_cluster_id: cluster_features.cluster_id,
            cluster_intensity: cluster_features.intensity,
            cluster_duration: cluster_features.duration_ms,
            
            // 高频指标
            micro_price_trend: impact_features.trend,
            tick_intensity: toxicity_features.tick_intensity,
            order_flow_imbalance: obi_features.flow_imbalance,
            
            // 预测性指标
            short_term_alpha,
            regime_probability: self.estimate_regime_probability(&obi_features)?,
            predicted_impact: impact_features.predicted_impact,
        };

        Ok(Some(features))
    }

    /// 计算短期Alpha信号（AG3核心）
    fn compute_short_term_alpha(
        &self, 
        obi: &order_book_imbalance::OBIFeatures,
        toxicity: &flow_toxicity::ToxicityFeatures,
    ) -> Result<f64> {
        // 多因子Alpha模型
        let obi_alpha = obi.imbalance * 0.3;  // OBI权重30%
        let toxicity_alpha = -toxicity.vpin * 0.2;  // 毒性信号权重20%（反向）
        let momentum_alpha = obi.momentum * 0.25;   // 动量权重25%
        let mean_reversion_alpha = -obi.reversion_signal * 0.25; // 均值回归25%（反向）

        let raw_alpha = obi_alpha + toxicity_alpha + momentum_alpha + mean_reversion_alpha;
        
        // 标准化到[-1, 1]
        let normalized_alpha = (raw_alpha * 2.0).tanh();
        
        Ok(normalized_alpha)
    }

    /// 估计市场体制概率
    fn estimate_regime_probability(&self, obi: &order_book_imbalance::OBIFeatures) -> Result<f64> {
        // 简化的体制检测：基于波动率和不平衡度
        let volatility_regime = if obi.volatility > 0.02 { 1.0 } else { 0.0 };
        let imbalance_regime = if obi.imbalance.abs() > 0.3 { 1.0 } else { 0.0 };
        
        Ok((volatility_regime + imbalance_regime) / 2.0)
    }

    /// 获取特征维度信息
    pub fn get_feature_info() -> Vec<(&'static str, &'static str)> {
        vec![
            ("order_book_imbalance", "订单簿不平衡 (OBI)"),
            ("bid_ask_spread", "买卖价差"),
            ("queue_imbalance", "队列不平衡"),
            ("flow_toxicity", "流动性毒性 (VPIN)"),
            ("price_impact", "价格冲击"),
            ("trade_cluster_intensity", "交易簇强度"),
            ("short_term_alpha", "短期Alpha信号"),
            ("regime_probability", "体制概率"),
            ("predicted_impact", "预期市场冲击"),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_microstructure_engine_creation() {
        let engine = MicrostructureEngine::new("BTCUSDT".to_string(), 1000);
        assert_eq!(engine.symbol, "BTCUSDT");
        assert_eq!(engine.max_history, 1000);
    }

    #[test]
    fn test_feature_info() {
        let info = MicrostructureEngine::get_feature_info();
        assert!(!info.is_empty());
        assert_eq!(info[0].0, "order_book_imbalance");
    }
}