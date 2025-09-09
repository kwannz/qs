//! AG3 实施差额(Implementation Shortfall)算法
//!
//! 基于Perold (1988) 和 Almgren-Chriss (2000) 模型的实施差额计算
//! 将总交易成本分解为：
//! - 决策延迟成本 (Decision Delay)
//! - 市场冲击成本 (Market Impact)  
//! - 机会成本 (Opportunity Cost)
//! - 时机成本 (Timing Cost)

use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{ExecutionTransaction, MarketDataHistory, Fill, PricePoint};

/// 实施差额计算器
#[derive(Debug)]
pub struct ShortfallCalculator {
    decision_tracker: DecisionTracker,
    timing_analyzer: TimingAnalyzer,
    opportunity_analyzer: OpportunityAnalyzer,
    market_impact_estimator: MarketImpactEstimator,
    config: ShortfallConfig,
}

/// 实施差额配置
#[derive(Debug, Clone)]
pub struct ShortfallConfig {
    pub decision_delay_ms: i64,       // 决策延迟（毫秒）
    pub paper_portfolio_enabled: bool, // 是否启用纸面投资组合
    pub benchmark_method: ShortfallBenchmarkMethod,
    pub attribution_method: AttributionMethod,
    pub risk_free_rate: f64,          // 无风险利率
    pub confidence_intervals: bool,    // 是否计算置信区间
}

/// 基准方法
#[derive(Debug, Clone)]
pub enum ShortfallBenchmarkMethod {
    Arrival,                  // 到达价
    OpenPrice,               // 开盘价
    PreviousClose,           // 前收盘价
    CustomBenchmark(f64),    // 自定义基准价
}

/// 归因方法
#[derive(Debug, Clone)]
pub enum AttributionMethod {
    Perold,                  // Perold经典方法
    AlmgrenChriss,          // Almgren-Chriss扩展
    Hybrid,                  // 混合方法
}

/// 决策跟踪器
#[derive(Debug)]
pub struct DecisionTracker {
    decision_times: HashMap<String, DateTime<Utc>>,
    order_intents: HashMap<String, OrderIntent>,
}

/// 订单意图
#[derive(Debug, Clone)]
pub struct OrderIntent {
    pub intended_quantity: f64,
    pub decision_time: DateTime<Utc>,
    pub urgency_level: UrgencyLevel,
    pub strategy_context: StrategyContext,
}

#[derive(Debug, Clone)]
pub enum UrgencyLevel {
    Low,     // 低紧急度
    Medium,  // 中等紧急度  
    High,    // 高紧急度
    Critical, // 紧急
}

#[derive(Debug, Clone)]
pub struct StrategyContext {
    pub strategy_type: String,
    pub risk_tolerance: f64,
    pub alpha_decay_rate: f64,
}

/// 时机分析器
#[derive(Debug)]
pub struct TimingAnalyzer {
    timing_models: Vec<TimingModel>,
    performance_attribution: PerformanceAttribution,
}

#[derive(Debug)]
pub enum TimingModel {
    LinearDecay,         // 线性衰减模型
    ExponentialDecay,    // 指数衰减模型
    StepFunction,        // 阶跃函数
    MarketRegime,        // 市场制度感知
}

#[derive(Debug)]
pub struct PerformanceAttribution {
    alpha_capture: f64,
    timing_skill: f64,
    execution_efficiency: f64,
}

/// 机会分析器
#[derive(Debug)]
pub struct OpportunityAnalyzer {
    paper_portfolio: PaperPortfolio,
    counterfactual_analyzer: CounterfactualAnalyzer,
}

/// 纸面投资组合
#[derive(Debug)]
pub struct PaperPortfolio {
    virtual_positions: HashMap<String, VirtualPosition>,
    performance_tracking: PerformanceTracking,
}

#[derive(Debug)]
pub struct VirtualPosition {
    pub symbol: String,
    pub intended_quantity: f64,
    pub decision_price: f64,
    pub current_market_price: f64,
    pub unrealized_pnl: f64,
}

#[derive(Debug)]
pub struct PerformanceTracking {
    pub total_paper_value: f64,
    pub total_realized_value: f64,
    pub opportunity_cost: f64,
}

/// 反事实分析器
#[derive(Debug)]
pub struct CounterfactualAnalyzer {
    scenario_generator: ScenarioGenerator,
    what_if_calculator: WhatIfCalculator,
}

#[derive(Debug)]
pub struct ScenarioGenerator {
    pub scenario_count: usize,
    pub confidence_level: f64,
}

#[derive(Debug)]
pub struct WhatIfCalculator {
    pub alternative_strategies: Vec<String>,
}

/// 市场冲击估算器
#[derive(Debug)]
pub struct MarketImpactEstimator {
    impact_models: Vec<ImpactModel>,
    liquidity_analyzer: LiquidityAnalyzer,
}

#[derive(Debug)]
pub enum ImpactModel {
    AlmgrenChriss,       // Almgren-Chriss模型
    LinearImpact,        // 线性冲击模型
    SquareRootImpact,    // 平方根模型
    PowerLaw,           // 幂律模型
}

#[derive(Debug)]
pub struct LiquidityAnalyzer {
    pub depth_metrics: DepthMetrics,
    pub spread_metrics: SpreadMetrics,
}

#[derive(Debug)]
pub struct DepthMetrics {
    pub bid_depth: f64,
    pub ask_depth: f64,
    pub total_depth: f64,
    pub imbalance_ratio: f64,
}

#[derive(Debug)]
pub struct SpreadMetrics {
    pub bid_ask_spread: f64,
    pub effective_spread: f64,
    pub quoted_spread: f64,
}

/// 实施差额结果
#[derive(Debug, Serialize, Deserialize)]
pub struct ImplementationShortfallResult {
    pub total_shortfall_bps: f64,
    pub total_shortfall_currency: f64,
    
    // 成本分解
    pub delay_cost_bps: f64,          // 决策延迟成本
    pub market_impact_bps: f64,       // 市场冲击成本
    pub opportunity_cost_bps: f64,    // 机会成本
    pub timing_cost_bps: f64,         // 时机成本
    
    // 归因分析
    pub alpha_capture_rate: f64,      // Alpha捕获率
    pub execution_efficiency: f64,    // 执行效率
    pub timing_skill_score: f64,      // 时机技能得分
    
    // 基准和实际表现
    pub benchmark_return_bps: f64,    // 基准收益率
    pub paper_portfolio_return_bps: f64, // 纸面投资组合收益率
    pub actual_return_bps: f64,       // 实际收益率
    
    // 统计指标
    pub confidence_interval: Option<(f64, f64)>, // 置信区间
    pub statistical_significance: f64, // 统计显著性
    
    // 详细归因
    pub detailed_attribution: DetailedAttribution,
}

/// 详细归因
#[derive(Debug, Serialize, Deserialize)]
pub struct DetailedAttribution {
    pub decision_quality_score: f64,     // 决策质量得分
    pub execution_quality_score: f64,    // 执行质量得分
    pub market_timing_score: f64,        // 市场时机得分
    pub liquidity_timing_score: f64,     // 流动性时机得分
    pub cost_efficiency_score: f64,      // 成本效率得分
}

impl ShortfallCalculator {
    /// 创建新的实施差额计算器
    pub fn new() -> Self {
        let config = ShortfallConfig {
            decision_delay_ms: 100,
            paper_portfolio_enabled: true,
            benchmark_method: ShortfallBenchmarkMethod::Arrival,
            attribution_method: AttributionMethod::AlmgrenChriss,
            risk_free_rate: 0.02, // 2%
            confidence_intervals: true,
        };

        Self {
            decision_tracker: DecisionTracker::new(),
            timing_analyzer: TimingAnalyzer::new(),
            opportunity_analyzer: OpportunityAnalyzer::new(),
            market_impact_estimator: MarketImpactEstimator::new(),
            config,
        }
    }

    /// 计算实施差额
    pub fn calculate_shortfall(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
        order_intent: &OrderIntent,
    ) -> Result<ImplementationShortfallResult> {
        let calculation_start = std::time::Instant::now();
        
        // 1. 计算基准价格
        let benchmark_price = self.calculate_benchmark_price(
            transaction,
            market_data,
            order_intent,
        )?;
        
        // 2. 计算决策延迟成本
        let delay_cost_bps = self.calculate_delay_cost(
            transaction,
            market_data,
            order_intent,
            benchmark_price,
        )?;
        
        // 3. 计算市场冲击成本
        let market_impact_bps = self.market_impact_estimator.calculate_impact(
            transaction,
            market_data,
        )?;
        
        // 4. 计算机会成本（未执行部分）
        let opportunity_cost_bps = self.opportunity_analyzer.calculate_opportunity_cost(
            transaction,
            market_data,
            order_intent,
        )?;
        
        // 5. 计算时机成本
        let timing_cost_bps = self.timing_analyzer.calculate_timing_cost(
            transaction,
            market_data,
            order_intent,
        )?;
        
        // 6. 计算总实施差额
        let total_shortfall_bps = delay_cost_bps + market_impact_bps + 
                                 opportunity_cost_bps + timing_cost_bps;
        
        let total_notional = transaction.original_quantity * benchmark_price;
        let total_shortfall_currency = total_shortfall_bps * total_notional / 10000.0;
        
        // 7. 计算归因分析
        let attribution = self.calculate_detailed_attribution(
            transaction,
            market_data,
            order_intent,
            benchmark_price,
        )?;
        
        // 8. 计算收益率
        let benchmark_return_bps = 0.0; // 基准收益为0（相对自身）
        let paper_portfolio_return_bps = self.calculate_paper_portfolio_return(
            order_intent,
            market_data,
        )?;
        let actual_return_bps = self.calculate_actual_return(
            transaction,
            benchmark_price,
        )?;
        
        // 9. 计算统计显著性和置信区间
        let (confidence_interval, statistical_significance) = if self.config.confidence_intervals {
            let ci = self.calculate_confidence_interval(
                total_shortfall_bps,
                market_data,
            )?;
            let sig = self.calculate_statistical_significance(
                total_shortfall_bps,
                market_data,
            )?;
            (Some(ci), sig)
        } else {
            (None, 0.0)
        };
        
        let calculation_duration = calculation_start.elapsed();
        log::info!("Implementation Shortfall calculation completed in {:?}", calculation_duration);
        
        Ok(ImplementationShortfallResult {
            total_shortfall_bps,
            total_shortfall_currency,
            delay_cost_bps,
            market_impact_bps,
            opportunity_cost_bps,
            timing_cost_bps,
            alpha_capture_rate: attribution.alpha_capture,
            execution_efficiency: attribution.execution_efficiency,
            timing_skill_score: attribution.timing_skill,
            benchmark_return_bps,
            paper_portfolio_return_bps,
            actual_return_bps,
            confidence_interval,
            statistical_significance,
            detailed_attribution: attribution.into(),
        })
    }
    
    /// 计算基准价格
    fn calculate_benchmark_price(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
        order_intent: &OrderIntent,
    ) -> Result<f64> {
        match &self.config.benchmark_method {
            ShortfallBenchmarkMethod::Arrival => {
                self.calculate_arrival_price(transaction, market_data)
            },
            ShortfallBenchmarkMethod::OpenPrice => {
                self.calculate_open_price(market_data)
            },
            ShortfallBenchmarkMethod::PreviousClose => {
                self.calculate_previous_close(market_data)
            },
            ShortfallBenchmarkMethod::CustomBenchmark(price) => {
                Ok(*price)
            },
        }
    }
    
    /// 计算到达价
    fn calculate_arrival_price(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        if transaction.fills.is_empty() {
            return Err(anyhow::anyhow!("No fills in transaction"));
        }
        
        let first_fill_time = transaction.fills.iter()
            .map(|f| f.timestamp)
            .min().unwrap();
        
        let arrival_time = first_fill_time - Duration::milliseconds(self.config.decision_delay_ms);
        
        // 找到最接近到达时间的价格
        let arrival_price = market_data.price_data.iter()
            .min_by_key(|p| (p.timestamp - arrival_time).num_seconds().abs())
            .map(|p| p.price)
            .unwrap_or(transaction.fills[0].price);
            
        Ok(arrival_price)
    }
    
    /// 计算开盘价
    fn calculate_open_price(&self, market_data: &MarketDataHistory) -> Result<f64> {
        market_data.price_data.first()
            .map(|p| p.price)
            .ok_or_else(|| anyhow::anyhow!("No price data available"))
    }
    
    /// 计算前收盘价
    fn calculate_previous_close(&self, market_data: &MarketDataHistory) -> Result<f64> {
        // 简化实现：使用第一个价格作为前收盘价
        self.calculate_open_price(market_data)
    }
    
    /// 计算决策延迟成本
    fn calculate_delay_cost(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
        order_intent: &OrderIntent,
        benchmark_price: f64,
    ) -> Result<f64> {
        if transaction.fills.is_empty() {
            return Ok(0.0);
        }
        
        let decision_time = order_intent.decision_time;
        let first_fill_time = transaction.fills.iter()
            .map(|f| f.timestamp)
            .min().unwrap();
        
        // 计算决策期间的价格变动
        let decision_price = self.get_price_at_time(market_data, decision_time)?;
        let first_fill_price = transaction.fills[0].price;
        
        let price_change = if transaction.side == "BUY" {
            (first_fill_price - decision_price) / decision_price
        } else {
            (decision_price - first_fill_price) / decision_price
        };
        
        Ok(price_change * 10000.0) // 转换为基点
    }
    
    /// 计算实际收益
    fn calculate_actual_return(
        &self,
        transaction: &ExecutionTransaction,
        benchmark_price: f64,
    ) -> Result<f64> {
        if transaction.fills.is_empty() {
            return Ok(0.0);
        }
        
        let execution_vwap = transaction.fills.iter()
            .map(|f| f.quantity * f.price)
            .sum::<f64>() / transaction.fills.iter()
            .map(|f| f.quantity)
            .sum::<f64>();
        
        let return_rate = if transaction.side == "BUY" {
            (execution_vwap - benchmark_price) / benchmark_price
        } else {
            (benchmark_price - execution_vwap) / benchmark_price
        };
        
        Ok(-return_rate * 10000.0) // 负号因为这是成本
    }
    
    /// 计算纸面投资组合收益
    fn calculate_paper_portfolio_return(
        &self,
        order_intent: &OrderIntent,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        if market_data.price_data.len() < 2 {
            return Ok(0.0);
        }
        
        let start_price = market_data.price_data.first().unwrap().price;
        let end_price = market_data.price_data.last().unwrap().price;
        
        let return_rate = (end_price - start_price) / start_price;
        
        Ok(return_rate * 10000.0)
    }
    
    /// 获取特定时间的价格
    fn get_price_at_time(
        &self,
        market_data: &MarketDataHistory,
        target_time: DateTime<Utc>,
    ) -> Result<f64> {
        let price_point = market_data.price_data.iter()
            .min_by_key(|p| (p.timestamp - target_time).num_seconds().abs())
            .ok_or_else(|| anyhow::anyhow!("No price data available"))?;
            
        Ok(price_point.price)
    }
    
    /// 计算详细归因
    fn calculate_detailed_attribution(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
        order_intent: &OrderIntent,
        benchmark_price: f64,
    ) -> Result<PerformanceAttribution> {
        // 简化的归因计算
        let fill_rate = if order_intent.intended_quantity > 0.0 {
            transaction.fills.iter().map(|f| f.quantity).sum::<f64>() / order_intent.intended_quantity
        } else {
            0.0
        };
        
        let alpha_capture = fill_rate * 0.8; // 简化：填充率 * 效率因子
        let execution_efficiency = self.calculate_execution_efficiency(transaction, benchmark_price)?;
        let timing_skill = self.calculate_timing_skill(transaction, market_data, order_intent)?;
        
        Ok(PerformanceAttribution {
            alpha_capture,
            execution_efficiency,
            timing_skill,
        })
    }
    
    /// 计算执行效率
    fn calculate_execution_efficiency(
        &self,
        transaction: &ExecutionTransaction,
        benchmark_price: f64,
    ) -> Result<f64> {
        if transaction.fills.is_empty() {
            return Ok(0.0);
        }
        
        let execution_vwap = transaction.fills.iter()
            .map(|f| f.quantity * f.price)
            .sum::<f64>() / transaction.fills.iter()
            .map(|f| f.quantity)
            .sum::<f64>();
        
        let efficiency = 1.0 - ((execution_vwap - benchmark_price) / benchmark_price).abs();
        
        Ok(efficiency.max(0.0).min(1.0))
    }
    
    /// 计算时机技能
    fn calculate_timing_skill(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
        order_intent: &OrderIntent,
    ) -> Result<f64> {
        // 简化实现：基于执行期间的价格变动方向
        if transaction.fills.is_empty() || market_data.price_data.len() < 2 {
            return Ok(0.5); // 中性技能
        }
        
        let start_price = market_data.price_data.first().unwrap().price;
        let end_price = market_data.price_data.last().unwrap().price;
        let price_direction = if end_price > start_price { 1.0 } else { -1.0 };
        
        // 如果价格方向与交易方向一致，认为时机较好
        let trade_direction = if transaction.side == "BUY" { 1.0 } else { -1.0 };
        let timing_alignment = (price_direction * trade_direction + 1.0) / 2.0; // 归一化到[0,1]
        
        Ok(timing_alignment)
    }
    
    /// 计算置信区间
    fn calculate_confidence_interval(
        &self,
        shortfall_bps: f64,
        market_data: &MarketDataHistory,
    ) -> Result<(f64, f64)> {
        // 简化实现：基于历史波动率
        let volatility = self.estimate_volatility(market_data)?;
        let confidence_factor = 1.96; // 95%置信区间
        
        let lower_bound = shortfall_bps - confidence_factor * volatility;
        let upper_bound = shortfall_bps + confidence_factor * volatility;
        
        Ok((lower_bound, upper_bound))
    }
    
    /// 计算统计显著性
    fn calculate_statistical_significance(
        &self,
        shortfall_bps: f64,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        let volatility = self.estimate_volatility(market_data)?;
        
        if volatility <= 0.0 {
            return Ok(0.0);
        }
        
        let t_statistic = shortfall_bps / volatility;
        let significance = if t_statistic.abs() > 1.96 { 0.95 } else { 0.0 };
        
        Ok(significance)
    }
    
    /// 估算波动率
    fn estimate_volatility(&self, market_data: &MarketDataHistory) -> Result<f64> {
        if market_data.price_data.len() < 2 {
            return Ok(5.0); // 默认5bp
        }
        
        let returns: Vec<f64> = market_data.price_data
            .windows(2)
            .map(|w| (w[1].price / w[0].price - 1.0) * 10000.0)
            .collect();
        
        if returns.is_empty() {
            return Ok(5.0);
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        Ok(variance.sqrt())
    }
}

// 各个组件的实现

impl DecisionTracker {
    pub fn new() -> Self {
        Self {
            decision_times: HashMap::new(),
            order_intents: HashMap::new(),
        }
    }
}

impl TimingAnalyzer {
    pub fn new() -> Self {
        Self {
            timing_models: vec![
                TimingModel::LinearDecay,
                TimingModel::ExponentialDecay,
            ],
            performance_attribution: PerformanceAttribution {
                alpha_capture: 0.0,
                execution_efficiency: 0.0,
                timing_skill: 0.0,
            },
        }
    }
    
    pub fn calculate_timing_cost(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
        order_intent: &OrderIntent,
    ) -> Result<f64> {
        // 简化实现：基于执行期间的价格走势
        if transaction.fills.is_empty() || market_data.price_data.len() < 2 {
            return Ok(0.0);
        }
        
        let start_time = transaction.fills.iter().map(|f| f.timestamp).min().unwrap();
        let end_time = transaction.fills.iter().map(|f| f.timestamp).max().unwrap();
        
        let start_price = market_data.price_data.iter()
            .min_by_key(|p| (p.timestamp - start_time).num_seconds().abs())
            .map(|p| p.price)
            .unwrap_or(100.0);
            
        let end_price = market_data.price_data.iter()
            .min_by_key(|p| (p.timestamp - end_time).num_seconds().abs())
            .map(|p| p.price)
            .unwrap_or(100.0);
        
        let price_movement = (end_price - start_price) / start_price;
        
        // 时机成本：如果价格朝不利方向移动
        let timing_cost = if transaction.side == "BUY" {
            if price_movement > 0.0 { price_movement } else { 0.0 }
        } else {
            if price_movement < 0.0 { -price_movement } else { 0.0 }
        };
        
        Ok(timing_cost * 10000.0)
    }
}

impl OpportunityAnalyzer {
    pub fn new() -> Self {
        Self {
            paper_portfolio: PaperPortfolio {
                virtual_positions: HashMap::new(),
                performance_tracking: PerformanceTracking {
                    total_paper_value: 0.0,
                    total_realized_value: 0.0,
                    opportunity_cost: 0.0,
                },
            },
            counterfactual_analyzer: CounterfactualAnalyzer {
                scenario_generator: ScenarioGenerator {
                    scenario_count: 1000,
                    confidence_level: 0.95,
                },
                what_if_calculator: WhatIfCalculator {
                    alternative_strategies: vec!["TWAP".to_string(), "VWAP".to_string()],
                },
            },
        }
    }
    
    pub fn calculate_opportunity_cost(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
        order_intent: &OrderIntent,
    ) -> Result<f64> {
        // 计算未执行部分的机会成本
        let executed_quantity: f64 = transaction.fills.iter().map(|f| f.quantity).sum();
        let unexecuted_quantity = order_intent.intended_quantity - executed_quantity;
        
        if unexecuted_quantity <= 0.0 {
            return Ok(0.0);
        }
        
        // 计算期间价格变动
        if market_data.price_data.len() < 2 {
            return Ok(0.0);
        }
        
        let start_price = market_data.price_data.first().unwrap().price;
        let end_price = market_data.price_data.last().unwrap().price;
        let price_movement = (end_price - start_price) / start_price;
        
        // 机会成本：未执行数量的价格变动
        let opportunity_cost = if transaction.side == "BUY" {
            if price_movement > 0.0 { price_movement } else { 0.0 }
        } else {
            if price_movement < 0.0 { -price_movement } else { 0.0 }
        };
        
        let weight = unexecuted_quantity / order_intent.intended_quantity;
        
        Ok(opportunity_cost * weight * 10000.0)
    }
}

impl MarketImpactEstimator {
    pub fn new() -> Self {
        Self {
            impact_models: vec![ImpactModel::AlmgrenChriss, ImpactModel::SquareRootImpact],
            liquidity_analyzer: LiquidityAnalyzer {
                depth_metrics: DepthMetrics {
                    bid_depth: 0.0,
                    ask_depth: 0.0,
                    total_depth: 0.0,
                    imbalance_ratio: 0.0,
                },
                spread_metrics: SpreadMetrics {
                    bid_ask_spread: 0.0,
                    effective_spread: 0.0,
                    quoted_spread: 0.0,
                },
            },
        }
    }
    
    pub fn calculate_impact(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        // 简化的市场冲击计算
        let total_volume: f64 = transaction.fills.iter().map(|f| f.quantity).sum();
        let total_market_volume: f64 = market_data.volume_data.iter().map(|v| v.volume).sum();
        
        if total_market_volume <= 0.0 {
            return Ok(2.0); // 默认2bp冲击
        }
        
        let participation_rate = total_volume / total_market_volume;
        
        // 平方根法则：冲击 ∝ sqrt(参与率)
        let impact_bps = 10.0 * participation_rate.sqrt(); // 基础冲击系数为10bp
        
        Ok(impact_bps)
    }
}

impl From<PerformanceAttribution> for DetailedAttribution {
    fn from(attr: PerformanceAttribution) -> Self {
        Self {
            decision_quality_score: attr.alpha_capture,
            execution_quality_score: attr.execution_efficiency,
            market_timing_score: attr.timing_skill,
            liquidity_timing_score: (attr.timing_skill + attr.execution_efficiency) / 2.0,
            cost_efficiency_score: attr.execution_efficiency,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_shortfall_calculator_creation() {
        let calculator = ShortfallCalculator::new();
        // 基本创建测试通过
    }

    #[test]
    fn test_shortfall_calculation() {
        let calculator = ShortfallCalculator::new();
        
        let order_intent = OrderIntent {
            intended_quantity: 1000.0,
            decision_time: Utc::now() - Duration::minutes(5),
            urgency_level: UrgencyLevel::Medium,
            strategy_context: StrategyContext {
                strategy_type: "Momentum".to_string(),
                risk_tolerance: 0.5,
                alpha_decay_rate: 0.1,
            },
        };
        
        let transaction = ExecutionTransaction {
            transaction_id: "test_shortfall".to_string(),
            order_id: "order_1".to_string(),
            strategy_id: "strategy_1".to_string(),
            symbol: "AAPL".to_string(),
            side: "BUY".to_string(),
            original_quantity: 1000.0,
            fills: vec![
                Fill {
                    fill_id: "fill_1".to_string(),
                    quantity: 800.0,
                    price: 150.5,
                    timestamp: Utc::now(),
                    venue: "NYSE".to_string(),
                    commission: 2.0,
                    liquidity_flag: "TAKER".to_string(),
                }
            ],
            metadata: HashMap::new(),
        };
        
        let market_data = MarketDataHistory {
            symbol: "AAPL".to_string(),
            start_time: Utc::now() - Duration::hours(1),
            end_time: Utc::now(),
            price_data: vec![
                PricePoint { timestamp: Utc::now() - Duration::minutes(10), price: 150.0 },
                PricePoint { timestamp: Utc::now(), price: 150.8 },
            ],
            volume_data: vec![],
        };
        
        let result = calculator.calculate_shortfall(&transaction, &market_data, &order_intent);
        assert!(result.is_ok());
        
        let shortfall = result.unwrap();
        assert!(shortfall.total_shortfall_bps.is_finite());
        assert!(shortfall.execution_efficiency >= 0.0 && shortfall.execution_efficiency <= 1.0);
    }
}