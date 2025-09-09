//! AG3 TCA成本分解分析器
//!
//! 实现详细的交易成本分解功能：
//! - 市场冲击成本分析
//! - 时机成本计算
//! - 价差成本分解
//! - 滑点和机会成本分析

use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rust_decimal::Decimal;

use super::{TCAConfig, CostBreakdown, ExecutionTransaction, MarketDataHistory, Fill};

/// 成本分析器
#[derive(Debug)]
pub struct CostAnalyzer {
    impact_analyzer: MarketImpactAnalyzer,
    timing_analyzer: TimingCostAnalyzer,
    spread_analyzer: SpreadCostAnalyzer,
    slippage_analyzer: SlippageAnalyzer,
    commission_calculator: CommissionCalculator,
}

/// 市场冲击分析器
#[derive(Debug)]
pub struct MarketImpactAnalyzer {
    permanent_impact_model: PermanentImpactModel,
    temporary_impact_model: TemporaryImpactModel,
}

/// 时机成本分析器
#[derive(Debug)]
pub struct TimingCostAnalyzer {
    arrival_benchmark: ArrivalPriceBenchmark,
    trend_analyzer: TrendAnalyzer,
}

/// 价差成本分析器
#[derive(Debug)]
pub struct SpreadCostAnalyzer {
    bid_ask_analyzer: BidAskAnalyzer,
    effective_spread_calculator: EffectiveSpreadCalculator,
}

/// 滑点分析器
#[derive(Debug)]
pub struct SlippageAnalyzer {
    expected_slippage_model: ExpectedSlippageModel,
    realized_slippage_calculator: RealizedSlippageCalculator,
}

/// 佣金计算器
#[derive(Debug)]
pub struct CommissionCalculator {
    venue_fee_schedules: HashMap<String, FeeSchedule>,
}

#[derive(Debug, Clone)]
pub struct FeeSchedule {
    pub maker_fee_rate: f64,
    pub taker_fee_rate: f64,
    pub min_fee: f64,
    pub max_fee: f64,
}

/// 永久冲击模型 (Almgren-Chriss)
#[derive(Debug)]
pub struct PermanentImpactModel {
    gamma: f64,  // 永久冲击参数 γ
    volatility_scaling: bool,  // 是否使用波动率标准化
}

/// 临时冲击模型 (Almgren-Chriss)
#[derive(Debug)]
pub struct TemporaryImpactModel {
    eta: f64,    // 临时冲击参数 η
    decay_factor: f64,  // 衰减因子
}

/// 到达价基准
#[derive(Debug)]
pub struct ArrivalPriceBenchmark {
    decision_delay_ms: i64,
}

/// 趋势分析器
#[derive(Debug)]
pub struct TrendAnalyzer {
    lookback_windows: Vec<i64>, // 分析窗口（秒）
    momentum_threshold: f64,    // 动量阈值
}

/// 买卖价差分析器
#[derive(Debug)]
pub struct BidAskAnalyzer {
    spread_calculation_method: SpreadMethod,
}

#[derive(Debug)]
pub enum SpreadMethod {
    TimeWeighted,
    VolumeWeighted,
    Simple,
}

/// 有效价差计算器
#[derive(Debug)]
pub struct EffectiveSpreadCalculator {
    reference_price_method: ReferencePriceMethod,
}

#[derive(Debug)]
pub enum ReferencePriceMethod {
    Midpoint,
    NBBO,
    TWAP,
}

/// 预期滑点模型
#[derive(Debug)]
pub struct ExpectedSlippageModel {
    historical_data: HashMap<String, Vec<f64>>,
    confidence_level: f64,
}

/// 实际滑点计算器
#[derive(Debug)]
pub struct RealizedSlippageCalculator {
    benchmark_prices: HashMap<String, f64>,
}

impl CostAnalyzer {
    /// 创建新的成本分析器
    pub fn new(config: &TCAConfig) -> Self {
        Self {
            impact_analyzer: MarketImpactAnalyzer::new(),
            timing_analyzer: TimingCostAnalyzer::new(),
            spread_analyzer: SpreadCostAnalyzer::new(),
            slippage_analyzer: SlippageAnalyzer::new(),
            commission_calculator: CommissionCalculator::new(),
        }
    }

    /// 执行完整的成本分解分析
    pub fn analyze(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
        config: &TCAConfig,
    ) -> Result<CostBreakdown> {
        let analysis_start = std::time::Instant::now();
        
        // 1. 计算市场冲击成本
        let (permanent_impact, temporary_impact) = self.impact_analyzer
            .calculate_market_impact(transaction, market_data)?;
        let market_impact_bps = permanent_impact + temporary_impact;
        
        // 2. 计算时机成本
        let timing_cost_bps = self.timing_analyzer
            .calculate_timing_cost(transaction, market_data)?;
        
        // 3. 计算价差成本
        let (realized_spread, effective_spread) = self.spread_analyzer
            .calculate_spread_costs(transaction, market_data)?;
        let spread_cost_bps = realized_spread;
        
        // 4. 计算滑点
        let slippage_bps = self.slippage_analyzer
            .calculate_slippage(transaction, market_data)?;
        
        // 5. 计算佣金
        let commission_bps = self.commission_calculator
            .calculate_commission(transaction)?;
        
        // 6. 计算机会成本和延迟成本
        let opportunity_cost_bps = self.calculate_opportunity_cost(transaction, market_data)?;
        let delay_cost_bps = self.calculate_delay_cost(transaction, market_data)?;
        
        // 7. 汇总总成本
        let total_cost_bps = market_impact_bps + timing_cost_bps + spread_cost_bps + 
                            slippage_bps + commission_bps + opportunity_cost_bps + delay_cost_bps;
        
        let total_notional = transaction.fills.iter()
            .map(|f| f.quantity * f.price)
            .sum::<f64>();
        let total_cost_currency = total_cost_bps * total_notional / 10000.0;
        
        let analysis_duration = analysis_start.elapsed();
        log::info!("Cost breakdown analysis completed in {:?}", analysis_duration);
        
        Ok(CostBreakdown {
            total_cost_bps,
            total_cost_currency,
            market_impact_bps,
            timing_cost_bps,
            spread_cost_bps,
            commission_bps,
            slippage_bps,
            opportunity_cost_bps,
            delay_cost_bps,
            permanent_impact_bps: permanent_impact,
            temporary_impact_bps: temporary_impact,
            realized_spread_bps: realized_spread,
            effective_spread_bps: effective_spread,
            cost_confidence_interval: self.calculate_confidence_interval(
                total_cost_bps, 
                transaction, 
                market_data
            )?,
        })
    }
    
    /// 计算机会成本
    fn calculate_opportunity_cost(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        if transaction.fills.is_empty() {
            return Ok(0.0);
        }
        
        let first_fill_time = transaction.fills.iter()
            .map(|f| f.timestamp)
            .min().unwrap();
        let last_fill_time = transaction.fills.iter()
            .map(|f| f.timestamp)
            .max().unwrap();
        
        // 计算执行期间的价格变动
        let start_price = self.get_market_price_at_time(market_data, first_fill_time)?;
        let end_price = self.get_market_price_at_time(market_data, last_fill_time)?;
        
        let price_movement = (end_price - start_price) / start_price;
        
        // 根据买卖方向计算机会成本
        let opportunity_cost = if transaction.side == "BUY" {
            if price_movement > 0.0 {
                price_movement * 10000.0 // 转换为基点
            } else {
                0.0 // 价格下跌对买方有利，无机会成本
            }
        } else {
            if price_movement < 0.0 {
                -price_movement * 10000.0 // 价格上涨对卖方不利
            } else {
                0.0 // 价格上涨对卖方有利，无机会成本
            }
        };
        
        Ok(opportunity_cost)
    }
    
    /// 计算延迟成本
    fn calculate_delay_cost(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        if transaction.fills.is_empty() {
            return Ok(0.0);
        }
        
        // 假设决策时间到第一笔成交的延迟
        let decision_delay_ms = 1000; // 1秒延迟
        let first_fill_time = transaction.fills.iter()
            .map(|f| f.timestamp)
            .min().unwrap();
        
        let decision_time = first_fill_time - Duration::milliseconds(decision_delay_ms);
        
        let decision_price = self.get_market_price_at_time(market_data, decision_time)?;
        let first_fill_price = transaction.fills[0].price;
        
        let delay_cost = ((first_fill_price - decision_price) / decision_price).abs() * 10000.0;
        
        Ok(delay_cost)
    }
    
    /// 获取特定时间的市场价格
    fn get_market_price_at_time(
        &self,
        market_data: &MarketDataHistory,
        target_time: DateTime<Utc>,
    ) -> Result<f64> {
        // 找到最接近目标时间的价格点
        let price_point = market_data.price_data.iter()
            .min_by_key(|p| (p.timestamp - target_time).num_seconds().abs())
            .ok_or_else(|| anyhow::anyhow!("No price data available for target time"))?;
        
        Ok(price_point.price)
    }
    
    /// 计算成本置信区间
    fn calculate_confidence_interval(
        &self,
        total_cost_bps: f64,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<Option<(f64, f64)>> {
        // 简化实现：基于历史波动率估算置信区间
        let volatility = self.estimate_cost_volatility(market_data)?;
        let confidence_factor = 1.96; // 95%置信区间
        
        let lower_bound = total_cost_bps - confidence_factor * volatility;
        let upper_bound = total_cost_bps + confidence_factor * volatility;
        
        Ok(Some((lower_bound, upper_bound)))
    }
    
    /// 估算成本波动率
    fn estimate_cost_volatility(&self, market_data: &MarketDataHistory) -> Result<f64> {
        if market_data.price_data.len() < 2 {
            return Ok(1.0); // 默认1bp波动率
        }
        
        // 计算价格收益率的标准差
        let returns: Vec<f64> = market_data.price_data
            .windows(2)
            .map(|w| (w[1].price / w[0].price - 1.0) * 10000.0)
            .collect();
        
        if returns.is_empty() {
            return Ok(1.0);
        }
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        Ok(variance.sqrt())
    }
}

/// 市场冲击分析器实现
impl MarketImpactAnalyzer {
    pub fn new() -> Self {
        Self {
            permanent_impact_model: PermanentImpactModel {
                gamma: 0.314,  // Almgren-Chriss典型值
                volatility_scaling: true,
            },
            temporary_impact_model: TemporaryImpactModel {
                eta: 0.142,    // Almgren-Chriss典型值  
                decay_factor: 0.1,
            },
        }
    }
    
    /// 计算市场冲击成本
    pub fn calculate_market_impact(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<(f64, f64)> {
        let total_volume = transaction.fills.iter().map(|f| f.quantity).sum::<f64>();
        
        if total_volume <= 0.0 {
            return Ok((0.0, 0.0));
        }
        
        // 估算日均成交量 (Average Daily Volume)
        let avg_daily_volume = self.estimate_adv(market_data)?;
        let participation_rate = total_volume / avg_daily_volume;
        
        // 估算波动率
        let volatility = self.estimate_volatility(market_data)?;
        
        // 计算永久冲击: γ * σ * sqrt(V/VD)
        let permanent_impact = self.permanent_impact_model.gamma * 
            volatility * (participation_rate).sqrt() * 10000.0; // 转换为基点
        
        // 计算临时冲击: η * σ * (V/VD)^0.6
        let temporary_impact = self.temporary_impact_model.eta * 
            volatility * (participation_rate).powf(0.6) * 10000.0; // 转换为基点
        
        Ok((permanent_impact, temporary_impact))
    }
    
    fn estimate_adv(&self, market_data: &MarketDataHistory) -> Result<f64> {
        if market_data.volume_data.is_empty() {
            return Ok(100000.0); // 默认日均量
        }
        
        let avg_volume = market_data.volume_data.iter()
            .map(|v| v.volume)
            .sum::<f64>() / market_data.volume_data.len() as f64;
            
        Ok(avg_volume * 24.0) // 假设24小时交易
    }
    
    fn estimate_volatility(&self, market_data: &MarketDataHistory) -> Result<f64> {
        if market_data.price_data.len() < 2 {
            return Ok(0.02); // 2%默认波动率
        }
        
        let returns: Vec<f64> = market_data.price_data
            .windows(2)
            .map(|w| (w[1].price / w[0].price).ln())
            .collect();
        
        if returns.is_empty() {
            return Ok(0.02);
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        
        Ok(variance.sqrt() * (365.25_f64).sqrt()) // 年化波动率
    }
}

/// 时机成本分析器实现
impl TimingCostAnalyzer {
    pub fn new() -> Self {
        Self {
            arrival_benchmark: ArrivalPriceBenchmark {
                decision_delay_ms: 100, // 100ms决策延迟
            },
            trend_analyzer: TrendAnalyzer {
                lookback_windows: vec![30, 60, 300], // 30s, 1min, 5min
                momentum_threshold: 0.001, // 0.1%
            },
        }
    }
    
    /// 计算时机成本
    pub fn calculate_timing_cost(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        if transaction.fills.is_empty() {
            return Ok(0.0);
        }
        
        let arrival_price = self.calculate_arrival_price(transaction, market_data)?;
        let execution_vwap = self.calculate_execution_vwap(transaction)?;
        
        // 时机成本 = (执行价格 - 到达价) / 到达价
        let timing_cost = ((execution_vwap - arrival_price) / arrival_price).abs() * 10000.0;
        
        // 根据交易方向调整符号
        let signed_timing_cost = if transaction.side == "BUY" {
            if execution_vwap > arrival_price {
                timing_cost  // 买高了，正成本
            } else {
                -timing_cost // 买便宜了，负成本
            }
        } else {
            if execution_vwap < arrival_price {
                timing_cost  // 卖低了，正成本
            } else {
                -timing_cost // 卖高了，负成本
            }
        };
        
        Ok(signed_timing_cost)
    }
    
    fn calculate_arrival_price(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        let first_fill_time = transaction.fills.iter()
            .map(|f| f.timestamp)
            .min().unwrap();
        
        let arrival_time = first_fill_time - 
            Duration::milliseconds(self.arrival_benchmark.decision_delay_ms);
        
        // 找到最接近到达时间的价格
        let arrival_price = market_data.price_data.iter()
            .min_by_key(|p| (p.timestamp - arrival_time).num_seconds().abs())
            .map(|p| p.price)
            .unwrap_or(transaction.fills[0].price);
            
        Ok(arrival_price)
    }
    
    fn calculate_execution_vwap(&self, transaction: &ExecutionTransaction) -> Result<f64> {
        let total_notional: f64 = transaction.fills.iter()
            .map(|f| f.quantity * f.price)
            .sum();
        let total_quantity: f64 = transaction.fills.iter()
            .map(|f| f.quantity)
            .sum();
            
        if total_quantity <= 0.0 {
            return Err(anyhow::anyhow!("Zero total quantity"));
        }
        
        Ok(total_notional / total_quantity)
    }
}

/// 价差成本分析器实现
impl SpreadCostAnalyzer {
    pub fn new() -> Self {
        Self {
            bid_ask_analyzer: BidAskAnalyzer {
                spread_calculation_method: SpreadMethod::TimeWeighted,
            },
            effective_spread_calculator: EffectiveSpreadCalculator {
                reference_price_method: ReferencePriceMethod::Midpoint,
            },
        }
    }
    
    /// 计算价差成本
    pub fn calculate_spread_costs(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<(f64, f64)> {
        let mut realized_spread_total = 0.0;
        let mut effective_spread_total = 0.0;
        let mut total_notional = 0.0;
        
        for fill in &transaction.fills {
            let fill_notional = fill.quantity * fill.price;
            total_notional += fill_notional;
            
            // 计算实际价差
            let midpoint = self.get_midpoint_at_time(market_data, fill.timestamp)?;
            let realized_spread = if transaction.side == "BUY" {
                (fill.price - midpoint) / midpoint
            } else {
                (midpoint - fill.price) / midpoint
            };
            realized_spread_total += realized_spread * fill_notional;
            
            // 计算有效价差 (简化为实际价差的80%)
            effective_spread_total += realized_spread * 0.8 * fill_notional;
        }
        
        if total_notional <= 0.0 {
            return Ok((0.0, 0.0));
        }
        
        let avg_realized_spread = (realized_spread_total / total_notional) * 10000.0;
        let avg_effective_spread = (effective_spread_total / total_notional) * 10000.0;
        
        Ok((avg_realized_spread, avg_effective_spread))
    }
    
    fn get_midpoint_at_time(
        &self,
        market_data: &MarketDataHistory,
        target_time: DateTime<Utc>,
    ) -> Result<f64> {
        // 简化实现：使用价格作为中价
        let price_point = market_data.price_data.iter()
            .min_by_key(|p| (p.timestamp - target_time).num_seconds().abs())
            .ok_or_else(|| anyhow::anyhow!("No price data for target time"))?;
            
        Ok(price_point.price)
    }
}

/// 滑点分析器实现
impl SlippageAnalyzer {
    pub fn new() -> Self {
        Self {
            expected_slippage_model: ExpectedSlippageModel {
                historical_data: HashMap::new(),
                confidence_level: 0.95,
            },
            realized_slippage_calculator: RealizedSlippageCalculator {
                benchmark_prices: HashMap::new(),
            },
        }
    }
    
    /// 计算滑点
    pub fn calculate_slippage(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        if transaction.fills.is_empty() {
            return Ok(0.0);
        }
        
        // 计算预期执行价格（基于历史中价）
        let expected_price = self.calculate_expected_price(transaction, market_data)?;
        let actual_vwap = transaction.fills.iter()
            .map(|f| f.quantity * f.price)
            .sum::<f64>() / transaction.fills.iter().map(|f| f.quantity).sum::<f64>();
        
        // 滑点 = |实际价格 - 预期价格| / 预期价格
        let slippage = ((actual_vwap - expected_price) / expected_price).abs() * 10000.0;
        
        Ok(slippage)
    }
    
    fn calculate_expected_price(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        // 使用交易期间的TWAP作为预期价格
        let start_time = transaction.fills.iter().map(|f| f.timestamp).min().unwrap();
        let end_time = transaction.fills.iter().map(|f| f.timestamp).max().unwrap();
        
        let relevant_prices: Vec<f64> = market_data.price_data.iter()
            .filter(|p| p.timestamp >= start_time && p.timestamp <= end_time)
            .map(|p| p.price)
            .collect();
        
        if relevant_prices.is_empty() {
            return Ok(transaction.fills[0].price);
        }
        
        Ok(relevant_prices.iter().sum::<f64>() / relevant_prices.len() as f64)
    }
}

/// 佣金计算器实现
impl CommissionCalculator {
    pub fn new() -> Self {
        let mut fee_schedules = HashMap::new();
        
        // 默认费率表
        fee_schedules.insert("NYSE".to_string(), FeeSchedule {
            maker_fee_rate: -0.0002,  // -0.2bp (rebate)
            taker_fee_rate: 0.0030,   // 3bp
            min_fee: 0.01,
            max_fee: 1000.0,
        });
        
        fee_schedules.insert("NASDAQ".to_string(), FeeSchedule {
            maker_fee_rate: -0.0001,  // -0.1bp (rebate)
            taker_fee_rate: 0.0025,   // 2.5bp
            min_fee: 0.01,
            max_fee: 1000.0,
        });
        
        Self {
            venue_fee_schedules: fee_schedules,
        }
    }
    
    /// 计算佣金成本
    pub fn calculate_commission(&self, transaction: &ExecutionTransaction) -> Result<f64> {
        let mut total_commission = 0.0;
        let mut total_notional = 0.0;
        
        for fill in &transaction.fills {
            let notional = fill.quantity * fill.price;
            total_notional += notional;
            
            // 使用fill中的佣金，或根据场所费率计算
            let commission = if fill.commission > 0.0 {
                fill.commission
            } else {
                self.calculate_venue_commission(fill)?
            };
            
            total_commission += commission;
        }
        
        if total_notional <= 0.0 {
            return Ok(0.0);
        }
        
        // 转换为基点
        Ok((total_commission / total_notional) * 10000.0)
    }
    
    fn calculate_venue_commission(&self, fill: &Fill) -> Result<f64> {
        let fee_schedule = self.venue_fee_schedules.get(&fill.venue)
            .ok_or_else(|| anyhow::anyhow!("Unknown venue: {}", fill.venue))?;
        
        let notional = fill.quantity * fill.price;
        let fee_rate = if fill.liquidity_flag == "MAKER" {
            fee_schedule.maker_fee_rate
        } else {
            fee_schedule.taker_fee_rate
        };
        
        let commission = (notional * fee_rate).abs();
        Ok(commission.max(fee_schedule.min_fee).min(fee_schedule.max_fee))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_cost_analyzer_creation() {
        let config = TCAConfig::default();
        let analyzer = CostAnalyzer::new(&config);
        // 基本创建测试通过
    }

    #[test]
    fn test_market_impact_calculation() {
        let analyzer = MarketImpactAnalyzer::new();
        
        let transaction = ExecutionTransaction {
            transaction_id: "test_1".to_string(),
            order_id: "order_1".to_string(),
            strategy_id: "strategy_1".to_string(),
            symbol: "AAPL".to_string(),
            side: "BUY".to_string(),
            original_quantity: 1000.0,
            fills: vec![
                Fill {
                    fill_id: "fill_1".to_string(),
                    quantity: 1000.0,
                    price: 150.0,
                    timestamp: Utc::now(),
                    venue: "NYSE".to_string(),
                    commission: 1.0,
                    liquidity_flag: "TAKER".to_string(),
                }
            ],
            metadata: HashMap::new(),
        };
        
        // 创建模拟市场数据
        let market_data = MarketDataHistory {
            symbol: "AAPL".to_string(),
            start_time: Utc::now() - Duration::hours(1),
            end_time: Utc::now(),
            price_data: vec![],
            volume_data: vec![],
        };
        
        let result = analyzer.calculate_market_impact(&transaction, &market_data);
        assert!(result.is_ok());
        
        let (permanent, temporary) = result.unwrap();
        assert!(permanent >= 0.0);
        assert!(temporary >= 0.0);
    }
}