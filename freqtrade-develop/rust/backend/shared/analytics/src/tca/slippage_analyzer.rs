//! AG3 滑点分析器
//!
//! 实现全面的滑点分析功能：
//! - 实际vs预期滑点比较  
//! - 滑点归因分析
//! - 动态滑点预测

use anyhow::Result;
use chrono::{DateTime, Utc, Duration, Timelike};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{ExecutionTransaction, MarketDataHistory};

/// 滑点分析器
#[derive(Debug)]
pub struct SlippageAnalyzer {
    expected_model: ExpectedSlippageModel,
    realized_calculator: RealizedSlippageCalculator,
    attribution_engine: SlippageAttributionEngine,
    predictor: DynamicSlippagePredictor,
}

/// 预期滑点模型
#[derive(Debug)]
pub struct ExpectedSlippageModel {
    base_model: BaseSlippageModel,
    volatility_adjustor: VolatilityAdjustor,
    liquidity_adjustor: LiquidityAdjustor,
    timing_adjustor: TimingAdjustor,
}

/// 实际滑点计算器
#[derive(Debug)]
pub struct RealizedSlippageCalculator {
    benchmark_calculator: BenchmarkCalculator,
    execution_analyzer: ExecutionAnalyzer,
}

/// 滑点归因引擎
#[derive(Debug)]
pub struct SlippageAttributionEngine {
    factor_models: Vec<SlippageFactorModel>,
    interaction_analyzer: InteractionAnalyzer,
}

/// 动态滑点预测器
#[derive(Debug)]
pub struct DynamicSlippagePredictor {
    time_series_model: TimeSeriesSlippageModel,
    regime_model: RegimeBasedSlippageModel,
    ensemble_weights: Vec<f64>,
}

/// 滑点分析结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageAnalysisResult {
    pub expected_slippage_bps: f64,
    pub realized_slippage_bps: f64,
    pub excess_slippage_bps: f64,
    pub slippage_ratio: f64,
    
    // 分解分析
    pub component_breakdown: SlippageComponentBreakdown,
    pub attribution_analysis: SlippageAttribution,
    pub temporal_analysis: TemporalSlippageAnalysis,
    
    // 预测和诊断
    pub predicted_slippage_bps: f64,
    pub prediction_confidence: f64,
    pub model_diagnostics: SlippageDiagnostics,
    
    pub analysis_timestamp: DateTime<Utc>,
}

/// 滑点组件分解
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageComponentBreakdown {
    pub market_microstructure_bps: f64,
    pub adverse_selection_bps: f64,
    pub inventory_management_bps: f64,
    pub timing_mismatch_bps: f64,
    pub execution_delay_bps: f64,
    pub venue_routing_bps: f64,
    pub order_management_bps: f64,
}

/// 滑点归因
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageAttribution {
    pub size_effect_bps: f64,
    pub volatility_effect_bps: f64,
    pub liquidity_effect_bps: f64,
    pub timing_effect_bps: f64,
    pub venue_effect_bps: f64,
    pub algorithm_effect_bps: f64,
    pub market_regime_effect_bps: f64,
    pub idiosyncratic_effect_bps: f64,
}

/// 时间滑点分析
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalSlippageAnalysis {
    pub intraday_pattern: HashMap<u32, f64>, // 小时 -> 滑点
    pub execution_progression: Vec<ProgressionPoint>,
    pub decay_analysis: SlippageDecayAnalysis,
}

/// 进程点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressionPoint {
    pub timestamp: DateTime<Utc>,
    pub cumulative_quantity_ratio: f64,
    pub cumulative_slippage_bps: f64,
    pub incremental_slippage_bps: f64,
    pub execution_rate: f64,
}

/// 滑点衰减分析
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageDecayAnalysis {
    pub decay_halflife_minutes: f64,
    pub recovery_rate: f64,
    pub persistent_component_bps: f64,
    pub transient_component_bps: f64,
}

/// 滑点诊断
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageDiagnostics {
    pub model_accuracy: f64,
    pub prediction_bias: f64,
    pub volatility_clustering: f64,
    pub regime_stability: f64,
    pub data_quality_score: f64,
    pub outlier_detection: Vec<OutlierAlert>,
}

/// 异常警报
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierAlert {
    pub timestamp: DateTime<Utc>,
    pub severity: OutlierSeverity,
    pub description: String,
    pub suggested_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// 基础滑点模型
#[derive(Debug)]
pub struct BaseSlippageModel {
    alpha: f64,  // 基础滑点系数
    beta: f64,   // 规模弹性
    calibration_window: Duration,
}

/// 波动率调整器
#[derive(Debug)]
pub struct VolatilityAdjustor {
    sensitivity: f64,
    lookback_period: Duration,
    regime_thresholds: Vec<f64>,
}

/// 流动性调整器
#[derive(Debug)]
pub struct LiquidityAdjustor {
    depth_impact_factor: f64,
    spread_impact_factor: f64,
    venue_liquidity_scores: HashMap<String, f64>,
}

/// 时机调整器
#[derive(Debug)]
pub struct TimingAdjustor {
    intraday_patterns: HashMap<u32, f64>, // 小时模式
    market_open_effect: f64,
    market_close_effect: f64,
    lunch_time_effect: f64,
}

/// 基准计算器
#[derive(Debug)]
pub struct BenchmarkCalculator {
    arrival_price_calculator: ArrivalPriceCalculator,
    twap_calculator: TWAPCalculator,
    vwap_calculator: VWAPCalculator,
}

/// 执行分析器
#[derive(Debug)]
pub struct ExecutionAnalyzer {
    fill_progression_analyzer: FillProgressionAnalyzer,
    venue_performance_analyzer: VenuePerformanceAnalyzer,
}

/// 滑点因子模型
#[derive(Debug, Clone)]
pub struct SlippageFactorModel {
    pub factor_name: String,
    pub factor_loading: f64,
    pub r_squared: f64,
    pub factor_type: SlippageFactorType,
}

#[derive(Debug, Clone)]
pub enum SlippageFactorType {
    OrderSize,
    MarketVolatility,
    BookDepth,
    TimeOfDay,
    VenueChoice,
    ExecutionStyle,
    MarketRegime,
}

/// 相互作用分析器
#[derive(Debug)]
pub struct InteractionAnalyzer {
    correlation_matrix: Vec<Vec<f64>>,
    interaction_effects: HashMap<(String, String), f64>,
}

/// 时间序列滑点模型
#[derive(Debug)]
pub struct TimeSeriesSlippageModel {
    autoregressive_order: usize,
    moving_average_order: usize,
    seasonal_components: Vec<f64>,
}

/// 基于状态的滑点模型
#[derive(Debug)]
pub struct RegimeBasedSlippageModel {
    regime_probabilities: HashMap<String, f64>,
    regime_specific_models: HashMap<String, BaseSlippageModel>,
}

impl SlippageAnalyzer {
    pub fn new() -> Self {
        Self {
            expected_model: ExpectedSlippageModel::new(),
            realized_calculator: RealizedSlippageCalculator::new(),
            attribution_engine: SlippageAttributionEngine::new(),
            predictor: DynamicSlippagePredictor::new(),
        }
    }

    /// 执行完整滑点分析
    pub fn analyze_slippage(
        &mut self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<SlippageAnalysisResult> {
        // 1. 计算预期滑点
        let expected_slippage_bps = self.expected_model.calculate_expected_slippage(
            transaction, market_data
        )?;

        // 2. 计算实际滑点
        let realized_slippage_bps = self.realized_calculator.calculate_realized_slippage(
            transaction, market_data
        )?;

        // 3. 计算超额滑点
        let excess_slippage_bps = realized_slippage_bps - expected_slippage_bps;
        let slippage_ratio = if expected_slippage_bps != 0.0 {
            realized_slippage_bps / expected_slippage_bps
        } else {
            1.0
        };

        // 4. 组件分解
        let component_breakdown = self.analyze_slippage_components(
            transaction, market_data, realized_slippage_bps
        )?;

        // 5. 归因分析
        let attribution_analysis = self.attribution_engine.perform_attribution(
            transaction, market_data, &component_breakdown
        )?;

        // 6. 时间分析
        let temporal_analysis = self.analyze_temporal_patterns(
            transaction, market_data
        )?;

        // 7. 动态预测
        let predicted_slippage_bps = self.predictor.predict_future_slippage(
            transaction, market_data
        )?;

        let prediction_confidence = self.predictor.calculate_prediction_confidence(
            transaction, market_data
        )?;

        // 8. 模型诊断
        let model_diagnostics = self.run_diagnostics(
            transaction, market_data, expected_slippage_bps, realized_slippage_bps
        )?;

        Ok(SlippageAnalysisResult {
            expected_slippage_bps,
            realized_slippage_bps,
            excess_slippage_bps,
            slippage_ratio,
            component_breakdown,
            attribution_analysis,
            temporal_analysis,
            predicted_slippage_bps,
            prediction_confidence,
            model_diagnostics,
            analysis_timestamp: Utc::now(),
        })
    }

    fn analyze_slippage_components(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
        total_slippage: f64,
    ) -> Result<SlippageComponentBreakdown> {
        // 市场微观结构成分
        let market_microstructure_bps = self.calculate_microstructure_component(
            transaction, market_data
        )?;

        // 逆向选择成分
        let adverse_selection_bps = self.calculate_adverse_selection_component(
            transaction, total_slippage
        )?;

        // 库存管理成分
        let inventory_management_bps = self.calculate_inventory_component(
            transaction
        )?;

        // 时机错配成分
        let timing_mismatch_bps = self.calculate_timing_mismatch_component(
            transaction, market_data
        )?;

        // 执行延迟成分
        let execution_delay_bps = self.calculate_execution_delay_component(
            transaction
        )?;

        // 场所路由成分
        let venue_routing_bps = self.calculate_venue_routing_component(
            transaction
        )?;

        // 订单管理成分
        let order_management_bps = total_slippage - (
            market_microstructure_bps + adverse_selection_bps + inventory_management_bps
            + timing_mismatch_bps + execution_delay_bps + venue_routing_bps
        );

        Ok(SlippageComponentBreakdown {
            market_microstructure_bps,
            adverse_selection_bps,
            inventory_management_bps,
            timing_mismatch_bps,
            execution_delay_bps,
            venue_routing_bps,
            order_management_bps,
        })
    }

    fn calculate_microstructure_component(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        // 基于bid-ask spread和深度的微观结构成本
        let estimated_spread_bps = 5.0; // 简化：5bp平均价差
        let depth_penalty = self.calculate_depth_penalty(transaction)?;
        
        Ok(estimated_spread_bps * 0.5 + depth_penalty) // 价差的一半 + 深度惩罚
    }

    fn calculate_adverse_selection_component(
        &self,
        transaction: &ExecutionTransaction,
        total_slippage: f64,
    ) -> Result<f64> {
        // 逆向选择通常与订单规模相关
        let total_quantity = transaction.fills.iter().map(|f| f.quantity).sum::<f64>();
        let size_factor = (total_quantity / 10000.0).min(3.0); // 归一化规模因子
        
        Ok(total_slippage * 0.3 * size_factor / 3.0) // 约30%归因于逆向选择
    }

    fn calculate_inventory_component(&self, transaction: &ExecutionTransaction) -> Result<f64> {
        // 基于交易方向和市场状况的库存成本
        let side_multiplier = match transaction.side.as_str() {
            "BUY" => 1.0,
            "SELL" => 1.1, // 卖出通常有稍高库存成本
            _ => 1.0,
        };
        
        Ok(1.5 * side_multiplier) // 基础1.5bp库存成本
    }

    fn calculate_timing_mismatch_component(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        if transaction.fills.is_empty() {
            return Ok(0.0);
        }

        // 计算决策到执行的时间差异造成的成本
        let execution_span = if transaction.fills.len() > 1 {
            let first = transaction.fills.iter().map(|f| f.timestamp).min().unwrap();
            let last = transaction.fills.iter().map(|f| f.timestamp).max().unwrap();
            last.signed_duration_since(first).num_minutes() as f64
        } else {
            0.0
        };

        let timing_cost = (execution_span / 60.0) * 0.5; // 每小时0.5bp时机成本
        Ok(timing_cost.min(5.0)) // 最大5bp
    }

    fn calculate_execution_delay_component(&self, transaction: &ExecutionTransaction) -> Result<f64> {
        // 基于填单间隔的延迟成本
        if transaction.fills.len() < 2 {
            return Ok(0.0);
        }

        let mut intervals = Vec::new();
        let mut sorted_fills = transaction.fills.clone();
        sorted_fills.sort_by_key(|f| f.timestamp);

        for i in 1..sorted_fills.len() {
            let interval = sorted_fills[i].timestamp
                .signed_duration_since(sorted_fills[i-1].timestamp)
                .num_seconds() as f64;
            intervals.push(interval);
        }

        let avg_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let delay_cost = (avg_interval / 300.0) * 0.2; // 每5分钟间隔0.2bp延迟成本
        
        Ok(delay_cost.min(3.0)) // 最大3bp
    }

    fn calculate_venue_routing_component(&self, transaction: &ExecutionTransaction) -> Result<f64> {
        // 基于场所选择效率的路由成本
        let venues_used: std::collections::HashSet<&String> = 
            transaction.fills.iter().map(|f| &f.venue).collect();

        let routing_efficiency = match venues_used.len() {
            1 => 0.8,  // 单一场所可能错失机会
            2..=3 => 1.0,  // 适度多样化
            _ => 0.9,  // 过度分散可能增加成本
        };

        let base_routing_cost = 1.0; // 基础1bp路由成本
        Ok(base_routing_cost / routing_efficiency)
    }

    fn calculate_depth_penalty(&self, transaction: &ExecutionTransaction) -> Result<f64> {
        // 基于订单规模估算深度惩罚
        let total_quantity = transaction.fills.iter().map(|f| f.quantity).sum::<f64>();
        let depth_consumption = (total_quantity / 50000.0).min(2.0); // 5万股为基准深度
        
        Ok(depth_consumption * 1.5) // 每消耗基准深度1.5bp惩罚
    }

    fn analyze_temporal_patterns(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<TemporalSlippageAnalysis> {
        // 日内模式分析
        let intraday_pattern = self.calculate_intraday_pattern(transaction)?;
        
        // 执行进程分析
        let execution_progression = self.calculate_execution_progression(transaction)?;
        
        // 衰减分析
        let decay_analysis = self.calculate_decay_analysis(transaction, market_data)?;

        Ok(TemporalSlippageAnalysis {
            intraday_pattern,
            execution_progression,
            decay_analysis,
        })
    }

    fn calculate_intraday_pattern(&self, transaction: &ExecutionTransaction) -> Result<HashMap<u32, f64>> {
        let mut pattern = HashMap::new();

        for fill in &transaction.fills {
            let hour = fill.timestamp.hour();
            let hour_slippage = self.estimate_fill_slippage(fill)?;
            
            *pattern.entry(hour).or_insert(0.0) += hour_slippage;
        }

        Ok(pattern)
    }

    fn calculate_execution_progression(&self, transaction: &ExecutionTransaction) -> Result<Vec<ProgressionPoint>> {
        let mut progression: Vec<ProgressionPoint> = Vec::new();
        let mut cumulative_quantity = 0.0;
        let mut cumulative_slippage = 0.0;
        let total_quantity = transaction.fills.iter().map(|f| f.quantity).sum::<f64>();

        let mut sorted_fills = transaction.fills.clone();
        sorted_fills.sort_by_key(|f| f.timestamp);

        for fill in &sorted_fills {
            cumulative_quantity += fill.quantity;
            let fill_slippage = self.estimate_fill_slippage(fill)?;
            cumulative_slippage += fill_slippage * (fill.quantity / total_quantity);

            let execution_rate = if progression.is_empty() {
                fill.quantity
            } else {
                let prev_time = progression.last().unwrap().timestamp;
                let time_diff = fill.timestamp.signed_duration_since(prev_time).num_seconds() as f64;
                fill.quantity / time_diff.max(1.0)
            };

            progression.push(ProgressionPoint {
                timestamp: fill.timestamp,
                cumulative_quantity_ratio: cumulative_quantity / total_quantity,
                cumulative_slippage_bps: cumulative_slippage,
                incremental_slippage_bps: fill_slippage,
                execution_rate,
            });
        }

        Ok(progression)
    }

    fn calculate_decay_analysis(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<SlippageDecayAnalysis> {
        // 简化的衰减分析
        let decay_halflife_minutes = 15.0; // 15分钟半衰期
        let recovery_rate = 0.05; // 每分钟5%恢复
        
        let total_slippage = self.realized_calculator.calculate_realized_slippage(
            transaction, market_data
        )?;
        
        let persistent_component_bps = total_slippage * 0.3; // 30%永久成分
        let transient_component_bps = total_slippage * 0.7;  // 70%临时成分

        Ok(SlippageDecayAnalysis {
            decay_halflife_minutes,
            recovery_rate,
            persistent_component_bps,
            transient_component_bps,
        })
    }

    fn estimate_fill_slippage(&self, fill: &crate::tca::Fill) -> Result<f64> {
        // 简化的单笔成交滑点估算
        let base_slippage = 2.0; // 基础2bp滑点
        let size_adjustment = (fill.quantity / 1000.0).sqrt() * 0.5;
        Ok(base_slippage + size_adjustment)
    }

    fn run_diagnostics(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
        expected_slippage: f64,
        realized_slippage: f64,
    ) -> Result<SlippageDiagnostics> {
        let prediction_error = (realized_slippage - expected_slippage).abs();
        let model_accuracy = 1.0 - (prediction_error / realized_slippage.max(1.0)).min(1.0);
        
        let prediction_bias = (realized_slippage - expected_slippage) / expected_slippage.max(1.0);
        
        let volatility_clustering = 0.3; // 简化
        let regime_stability = 0.8; // 简化
        let data_quality_score = self.assess_data_quality(market_data)?;
        
        let outlier_detection = self.detect_outliers(transaction, realized_slippage)?;

        Ok(SlippageDiagnostics {
            model_accuracy,
            prediction_bias,
            volatility_clustering,
            regime_stability,
            data_quality_score,
            outlier_detection,
        })
    }

    fn assess_data_quality(&self, market_data: &MarketDataHistory) -> Result<f64> {
        let mut quality = 1.0;
        
        if market_data.price_data.is_empty() {
            quality *= 0.5;
        } else if market_data.price_data.len() < 50 {
            quality *= 0.8;
        }
        
        if market_data.volume_data.is_empty() {
            quality *= 0.9;
        }
        
        Ok(quality)
    }

    fn detect_outliers(
        &self,
        transaction: &ExecutionTransaction,
        realized_slippage: f64,
    ) -> Result<Vec<OutlierAlert>> {
        let mut alerts = Vec::new();
        
        // 检测异常高滑点
        if realized_slippage > 50.0 {
            alerts.push(OutlierAlert {
                timestamp: Utc::now(),
                severity: OutlierSeverity::High,
                description: format!("Exceptionally high slippage: {:.1}bp", realized_slippage),
                suggested_action: "Review execution strategy and market conditions".to_string(),
            });
        }
        
        // 检测异常执行模式
        if transaction.fills.len() > 100 {
            alerts.push(OutlierAlert {
                timestamp: Utc::now(),
                severity: OutlierSeverity::Medium,
                description: format!("High fragmentation: {} fills", transaction.fills.len()),
                suggested_action: "Consider larger child order sizes".to_string(),
            });
        }
        
        Ok(alerts)
    }
}

// 实现各个子组件
impl ExpectedSlippageModel {
    pub fn new() -> Self {
        Self {
            base_model: BaseSlippageModel::new(),
            volatility_adjustor: VolatilityAdjustor::new(),
            liquidity_adjustor: LiquidityAdjustor::new(),
            timing_adjustor: TimingAdjustor::new(),
        }
    }

    pub fn calculate_expected_slippage(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        // 基础滑点
        let base_slippage = self.base_model.calculate_base_slippage(transaction)?;
        
        // 各种调整
        let volatility_adjustment = self.volatility_adjustor.adjust(transaction, market_data)?;
        let liquidity_adjustment = self.liquidity_adjustor.adjust(transaction, market_data)?;
        let timing_adjustment = self.timing_adjustor.adjust(transaction)?;
        
        let expected_slippage = base_slippage + volatility_adjustment + liquidity_adjustment + timing_adjustment;
        Ok(expected_slippage.max(0.0))
    }
}

impl BaseSlippageModel {
    pub fn new() -> Self {
        Self {
            alpha: 2.0,  // 基础2bp滑点
            beta: 0.5,   // 平方根规模弹性
            calibration_window: Duration::days(30),
        }
    }

    pub fn calculate_base_slippage(&self, transaction: &ExecutionTransaction) -> Result<f64> {
        let total_quantity = transaction.fills.iter().map(|f| f.quantity).sum::<f64>();
        let size_factor = (total_quantity / 10000.0).powf(self.beta); // 10K股为基准
        
        Ok(self.alpha * size_factor)
    }
}

impl VolatilityAdjustor {
    pub fn new() -> Self {
        Self {
            sensitivity: 1.5,
            lookback_period: Duration::hours(1),
            regime_thresholds: vec![0.01, 0.02, 0.03],
        }
    }

    pub fn adjust(&self, transaction: &ExecutionTransaction, market_data: &MarketDataHistory) -> Result<f64> {
        let volatility = self.calculate_recent_volatility(market_data)?;
        let adjustment = volatility * self.sensitivity * 100.0; // 转换为bp
        Ok(adjustment.min(10.0)) // 最大10bp波动率调整
    }

    fn calculate_recent_volatility(&self, market_data: &MarketDataHistory) -> Result<f64> {
        if market_data.price_data.len() < 10 {
            return Ok(0.015); // 默认1.5%波动率
        }

        let recent_prices = &market_data.price_data[market_data.price_data.len()-10..];
        let returns: Vec<f64> = recent_prices.windows(2)
            .map(|w| (w[1].price / w[0].price - 1.0))
            .collect();

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        Ok(variance.sqrt())
    }
}

impl LiquidityAdjustor {
    pub fn new() -> Self {
        let mut venue_scores = HashMap::new();
        venue_scores.insert("NYSE".to_string(), 0.9);
        venue_scores.insert("NASDAQ".to_string(), 0.85);
        venue_scores.insert("BATS".to_string(), 0.8);

        Self {
            depth_impact_factor: 0.5,
            spread_impact_factor: 1.0,
            venue_liquidity_scores: venue_scores,
        }
    }

    pub fn adjust(&self, transaction: &ExecutionTransaction, market_data: &MarketDataHistory) -> Result<f64> {
        let venue_adjustment = self.calculate_venue_adjustment(transaction)?;
        let spread_adjustment = 2.0; // 简化：2bp价差调整
        let depth_adjustment = self.calculate_depth_adjustment(transaction)?;

        Ok(venue_adjustment + spread_adjustment + depth_adjustment)
    }

    fn calculate_venue_adjustment(&self, transaction: &ExecutionTransaction) -> Result<f64> {
        let mut weighted_score = 0.0;
        let mut total_notional = 0.0;

        for fill in &transaction.fills {
            let notional = fill.quantity * fill.price;
            total_notional += notional;
            
            let venue_score = self.venue_liquidity_scores
                .get(&fill.venue)
                .unwrap_or(&0.7);
            
            weighted_score += venue_score * notional;
        }

        if total_notional > 0.0 {
            let avg_score = weighted_score / total_notional;
            Ok((1.0 - avg_score) * 3.0) // 最大3bp场所调整
        } else {
            Ok(0.0)
        }
    }

    fn calculate_depth_adjustment(&self, transaction: &ExecutionTransaction) -> Result<f64> {
        let total_quantity = transaction.fills.iter().map(|f| f.quantity).sum::<f64>();
        let depth_consumption = (total_quantity / 100000.0).min(1.0); // 10万股为满深度
        Ok(depth_consumption * self.depth_impact_factor * 2.0) // 最大1bp深度调整
    }
}

impl TimingAdjustor {
    pub fn new() -> Self {
        let mut patterns = HashMap::new();
        patterns.insert(9, 1.3);   // 开盘时段
        patterns.insert(10, 1.1);  
        patterns.insert(11, 1.0);  // 正常时段
        patterns.insert(12, 0.9);  // 午间相对平静
        patterns.insert(13, 0.9);
        patterns.insert(14, 1.0);  
        patterns.insert(15, 1.2);  // 收盘前活跃
        patterns.insert(16, 1.4);  // 收盘时段

        Self {
            intraday_patterns: patterns,
            market_open_effect: 1.5,
            market_close_effect: 1.8,
            lunch_time_effect: 0.8,
        }
    }

    pub fn adjust(&self, transaction: &ExecutionTransaction) -> Result<f64> {
        if transaction.fills.is_empty() {
            return Ok(0.0);
        }

        let first_fill = &transaction.fills[0];
        let hour = first_fill.timestamp.hour();
        
        let pattern_multiplier = self.intraday_patterns.get(&hour).unwrap_or(&1.0);
        let base_timing_impact = 1.0; // 基础1bp时机影响
        
        Ok(base_timing_impact * (pattern_multiplier - 1.0))
    }
}

impl RealizedSlippageCalculator {
    pub fn new() -> Self {
        Self {
            benchmark_calculator: BenchmarkCalculator::new(),
            execution_analyzer: ExecutionAnalyzer::new(),
        }
    }

    pub fn calculate_realized_slippage(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        // 计算基准价格
        let benchmark_price = self.benchmark_calculator.calculate_arrival_price(
            transaction, market_data
        )?;

        // 计算平均执行价格
        let avg_execution_price = self.calculate_volume_weighted_price(transaction)?;

        // 计算滑点
        let slippage = match transaction.side.as_str() {
            "BUY" => (avg_execution_price - benchmark_price) / benchmark_price,
            "SELL" => (benchmark_price - avg_execution_price) / benchmark_price,
            _ => 0.0,
        };

        Ok(slippage * 10000.0) // 转换为bp
    }

    fn calculate_volume_weighted_price(&self, transaction: &ExecutionTransaction) -> Result<f64> {
        let total_notional: f64 = transaction.fills.iter()
            .map(|f| f.quantity * f.price)
            .sum();
        let total_quantity: f64 = transaction.fills.iter()
            .map(|f| f.quantity)
            .sum();

        if total_quantity > 0.0 {
            Ok(total_notional / total_quantity)
        } else {
            Ok(0.0)
        }
    }
}

// 其他组件的简化实现
impl BenchmarkCalculator {
    pub fn new() -> Self {
        Self {
            arrival_price_calculator: ArrivalPriceCalculator,
            twap_calculator: TWAPCalculator,
            vwap_calculator: VWAPCalculator,
        }
    }

    pub fn calculate_arrival_price(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        self.arrival_price_calculator.calculate(transaction, market_data)
    }
}

#[derive(Debug)]
pub struct ArrivalPriceCalculator;

impl ArrivalPriceCalculator {
    pub fn calculate(&self, transaction: &ExecutionTransaction, market_data: &MarketDataHistory) -> Result<f64> {
        if transaction.fills.is_empty() {
            return Err(anyhow::anyhow!("No fills in transaction"));
        }

        // 找到第一笔成交前的最近价格
        let first_fill_time = transaction.fills.iter().map(|f| f.timestamp).min().unwrap();
        
        for price_point in market_data.price_data.iter().rev() {
            if price_point.timestamp <= first_fill_time {
                return Ok(price_point.price);
            }
        }

        // 如果找不到，使用第一笔成交价格
        Ok(transaction.fills[0].price)
    }
}

#[derive(Debug)]
pub struct TWAPCalculator;

#[derive(Debug)]
pub struct VWAPCalculator;

impl ExecutionAnalyzer {
    pub fn new() -> Self {
        Self {
            fill_progression_analyzer: FillProgressionAnalyzer,
            venue_performance_analyzer: VenuePerformanceAnalyzer,
        }
    }
}

#[derive(Debug)]
pub struct FillProgressionAnalyzer;

#[derive(Debug)]
pub struct VenuePerformanceAnalyzer;

impl SlippageAttributionEngine {
    pub fn new() -> Self {
        let factor_models = vec![
            SlippageFactorModel {
                factor_name: "Order Size".to_string(),
                factor_loading: 0.6,
                r_squared: 0.4,
                factor_type: SlippageFactorType::OrderSize,
            },
            SlippageFactorModel {
                factor_name: "Market Volatility".to_string(),
                factor_loading: 0.8,
                r_squared: 0.3,
                factor_type: SlippageFactorType::MarketVolatility,
            },
        ];

        Self {
            factor_models,
            interaction_analyzer: InteractionAnalyzer::new(),
        }
    }

    pub fn perform_attribution(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
        component_breakdown: &SlippageComponentBreakdown,
    ) -> Result<SlippageAttribution> {
        // 简化的归因分析
        let total_component_slippage = component_breakdown.market_microstructure_bps
            + component_breakdown.adverse_selection_bps
            + component_breakdown.inventory_management_bps
            + component_breakdown.timing_mismatch_bps
            + component_breakdown.execution_delay_bps
            + component_breakdown.venue_routing_bps
            + component_breakdown.order_management_bps;

        Ok(SlippageAttribution {
            size_effect_bps: total_component_slippage * 0.35,
            volatility_effect_bps: total_component_slippage * 0.20,
            liquidity_effect_bps: total_component_slippage * 0.15,
            timing_effect_bps: total_component_slippage * 0.10,
            venue_effect_bps: total_component_slippage * 0.10,
            algorithm_effect_bps: total_component_slippage * 0.05,
            market_regime_effect_bps: total_component_slippage * 0.03,
            idiosyncratic_effect_bps: total_component_slippage * 0.02,
        })
    }
}

impl InteractionAnalyzer {
    pub fn new() -> Self {
        Self {
            correlation_matrix: vec![vec![0.0; 5]; 5],
            interaction_effects: HashMap::new(),
        }
    }
}

impl DynamicSlippagePredictor {
    pub fn new() -> Self {
        Self {
            time_series_model: TimeSeriesSlippageModel::new(),
            regime_model: RegimeBasedSlippageModel::new(),
            ensemble_weights: vec![0.6, 0.4], // 时间序列60%, 状态模型40%
        }
    }

    pub fn predict_future_slippage(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        let ts_prediction = self.time_series_model.predict(transaction, market_data)?;
        let regime_prediction = self.regime_model.predict(transaction, market_data)?;
        
        let ensemble_prediction = ts_prediction * self.ensemble_weights[0]
            + regime_prediction * self.ensemble_weights[1];
        
        Ok(ensemble_prediction)
    }

    pub fn calculate_prediction_confidence(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        // 基于模型一致性和数据质量计算置信度
        let ts_prediction = self.time_series_model.predict(transaction, market_data)?;
        let regime_prediction = self.regime_model.predict(transaction, market_data)?;
        
        let prediction_agreement = 1.0 - ((ts_prediction - regime_prediction).abs() / ts_prediction.max(regime_prediction).max(1.0));
        let data_quality = if market_data.price_data.len() > 100 { 0.9 } else { 0.6 };
        
        Ok((prediction_agreement + data_quality) / 2.0)
    }
}

impl TimeSeriesSlippageModel {
    pub fn new() -> Self {
        Self {
            autoregressive_order: 3,
            moving_average_order: 2,
            seasonal_components: vec![1.0; 24], // 24小时季节性
        }
    }

    pub fn predict(&self, transaction: &ExecutionTransaction, market_data: &MarketDataHistory) -> Result<f64> {
        // 简化的时间序列预测
        Ok(3.5) // 预测3.5bp滑点
    }
}

impl RegimeBasedSlippageModel {
    pub fn new() -> Self {
        let mut regime_models = HashMap::new();
        regime_models.insert("Low Vol".to_string(), BaseSlippageModel { alpha: 2.0, beta: 0.4, calibration_window: Duration::days(30) });
        regime_models.insert("High Vol".to_string(), BaseSlippageModel { alpha: 4.0, beta: 0.6, calibration_window: Duration::days(30) });

        Self {
            regime_probabilities: HashMap::new(),
            regime_specific_models: regime_models,
        }
    }

    pub fn predict(&self, transaction: &ExecutionTransaction, market_data: &MarketDataHistory) -> Result<f64> {
        // 简化的基于状态预测
        Ok(4.2) // 预测4.2bp滑点
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slippage_analyzer_creation() {
        let analyzer = SlippageAnalyzer::new();
        // 基础创建测试
    }

    #[test]
    fn test_base_slippage_calculation() {
        let model = BaseSlippageModel::new();
        
        let transaction = ExecutionTransaction {
            transaction_id: "test".to_string(),
            order_id: "order1".to_string(),
            strategy_id: "strategy1".to_string(),
            symbol: "AAPL".to_string(),
            side: "BUY".to_string(),
            original_quantity: 10000.0,
            fills: vec![crate::tca::Fill {
                fill_id: "fill1".to_string(),
                quantity: 10000.0,
                price: 150.0,
                timestamp: Utc::now(),
                venue: "NYSE".to_string(),
                commission: 10.0,
                liquidity_flag: "TAKER".to_string(),
            }],
            metadata: HashMap::new(),
        };
        
        let slippage = model.calculate_base_slippage(&transaction).unwrap();
        assert!(slippage > 0.0);
        assert!(slippage < 20.0); // 合理范围
    }
}