pub mod cost_breakdown;
pub mod implementation_shortfall;
pub mod market_impact;
pub mod slippage_analyzer;
pub mod benchmark_comparison;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration, Timelike};

/// AG3交易成本分析引擎
pub struct TCAEngine {
    cost_analyzer: cost_breakdown::CostAnalyzer,
    shortfall_calculator: implementation_shortfall::ShortfallCalculator,
    impact_model: market_impact::MarketImpactModel,
    slippage_analyzer: slippage_analyzer::SlippageAnalyzer,
    benchmark_engine: benchmark_comparison::BenchmarkEngine,
    config: TCAConfig,
}

/// TCA配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCAConfig {
    pub analysis_window_hours: i64,        // 分析时间窗口
    pub benchmark_methods: Vec<BenchmarkMethod>, // 基准方法
    pub cost_components: CostComponents,   // 成本组件
    pub attribution_levels: Vec<AttributionLevel>, // 归因层级
    pub reporting_currency: String,        // 报告货币
    pub include_opportunity_cost: bool,    // 包含机会成本
    pub real_time_analysis: bool,          // 实时分析
    pub confidence_intervals: bool,        // 置信区间
}

/// 基准方法
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BenchmarkMethod {
    Arrival,                    // 到达价
    TWAP(i64),                 // 时间加权平均价（分钟）
    VWAP(i64),                 // 成交量加权平均价（分钟）
    Implementation,             // 实施基准
    PreTrade,                   // 交易前基准
    PostTrade,                  // 交易后基准
    Close,                      // 收盘价
    Open,                       // 开盘价
}

/// 成本组件配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostComponents {
    pub market_impact: bool,        // 市场冲击
    pub timing_cost: bool,          // 时机成本
    pub spread_cost: bool,          // 价差成本
    pub commission: bool,           // 手续费
    pub slippage: bool,             // 滑点
    pub opportunity_cost: bool,     // 机会成本
    pub delay_cost: bool,           // 延迟成本
}

/// 归因层级
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributionLevel {
    Strategy,          // 策略层
    Asset,             // 资产层
    Venue,             // 场所层
    TimeOfDay,         // 时段层
    OrderSize,         // 订单规模层
    Liquidity,         // 流动性层
}

/// TCA分析结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCAResult {
    pub transaction_id: String,
    pub symbol: String,
    pub analysis_timestamp: DateTime<Utc>,
    pub trade_summary: TradeSummary,
    pub cost_breakdown: CostBreakdown,
    pub benchmark_performance: BenchmarkPerformance,
    pub attribution_analysis: AttributionAnalysis,
    pub quality_metrics: QualityMetrics,
    pub risk_metrics: RiskMetrics,
    pub recommendations: Vec<TradingRecommendation>,
}

/// 交易摘要
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeSummary {
    pub order_id: String,
    pub side: String,
    pub quantity: f64,
    pub notional_value: f64,
    pub average_price: f64,
    pub first_fill_time: DateTime<Utc>,
    pub last_fill_time: DateTime<Utc>,
    pub execution_duration_minutes: f64,
    pub fill_rate: f64,
    pub number_of_fills: u32,
    pub venues_used: Vec<String>,
}

/// 成本分解
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub total_cost_bps: f64,           // 总成本（基点）
    pub total_cost_currency: f64,      // 总成本（货币）
    
    // 主要成本组件
    pub market_impact_bps: f64,        // 市场冲击
    pub timing_cost_bps: f64,          // 时机成本
    pub spread_cost_bps: f64,          // 价差成本
    pub commission_bps: f64,           // 手续费
    pub slippage_bps: f64,             // 滑点
    pub opportunity_cost_bps: f64,     // 机会成本
    pub delay_cost_bps: f64,           // 延迟成本
    
    // 细分成本
    pub permanent_impact_bps: f64,     // 永久冲击
    pub temporary_impact_bps: f64,     // 临时冲击
    pub realized_spread_bps: f64,      // 已实现价差
    pub effective_spread_bps: f64,     // 有效价差
    
    // 成本置信区间
    pub cost_confidence_interval: Option<(f64, f64)>, // 95%置信区间
}

/// 基准表现
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkPerformance {
    pub benchmarks: HashMap<String, BenchmarkResult>,
    pub best_benchmark: String,
    pub worst_benchmark: String,
    pub benchmark_consistency: f64,    // 基准一致性得分
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub benchmark_name: String,
    pub benchmark_price: f64,
    pub performance_bps: f64,          // 相对基准表现
    pub outperformance_probability: f64, // 跑赢概率
    pub statistical_significance: f64,  // 统计显著性
}

/// 归因分析
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionAnalysis {
    pub strategy_attribution: HashMap<String, f64>,  // 策略归因
    pub venue_attribution: HashMap<String, f64>,     // 场所归因
    pub time_attribution: HashMap<String, f64>,      // 时间归因
    pub size_attribution: HashMap<String, f64>,      // 规模归因
    pub main_drivers: Vec<AttributionDriver>,         // 主要驱动因子
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionDriver {
    pub factor: String,
    pub contribution_bps: f64,
    pub significance: f64,
    pub explanation: String,
}

/// 质量指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub execution_score: f64,          // 执行评分 (0-100)
    pub efficiency_rank: String,       // 效率排名
    pub consistency_score: f64,        // 一致性评分
    pub speed_percentile: f64,         // 速度百分位
    pub cost_percentile: f64,          // 成本百分位
    pub liquidity_capture: f64,        // 流动性捕获率
    pub adverse_selection_ratio: f64,  // 逆向选择比率
    pub timing_alpha: f64,             // 时机Alpha
}

/// 风险指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub cost_volatility: f64,          // 成本波动率
    pub execution_risk: f64,           // 执行风险
    pub market_impact_risk: f64,       // 市场冲击风险
    pub liquidity_risk: f64,           // 流动性风险
    pub concentration_risk: f64,        // 集中度风险
    pub tail_cost_risk: f64,           // 尾部成本风险
    pub var_95_bps: f64,               // 95% VaR（基点）
    pub cvar_95_bps: f64,              // 95% CVaR（基点）
}

/// 交易建议
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingRecommendation {
    pub category: RecommendationCategory,
    pub priority: RecommendationPriority,
    pub title: String,
    pub description: String,
    pub expected_improvement_bps: Option<f64>,
    pub implementation_difficulty: f64, // 0-1, 1=最难
    pub confidence_level: f64,          // 0-1, 1=最高置信度
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Timing,           // 时机优化
    VenueSelection,   // 场所选择
    OrderSizing,      // 订单规模
    AlgorithmChoice,  // 算法选择
    RiskManagement,   // 风险管理
    CostReduction,    // 成本降低
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

impl TCAEngine {
    pub fn new(config: TCAConfig) -> Self {
        Self {
            cost_analyzer: cost_breakdown::CostAnalyzer::new(&config),
            shortfall_calculator: implementation_shortfall::ShortfallCalculator::new(),
            impact_model: market_impact::MarketImpactModel::new(),
            slippage_analyzer: slippage_analyzer::SlippageAnalyzer::new(),
            benchmark_engine: benchmark_comparison::BenchmarkEngine::new(),
            config,
        }
    }

    /// 执行完整TCA分析
    pub fn analyze_execution(
        &mut self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<TCAResult> {
        let analysis_start = std::time::Instant::now();
        
        // 1. 构建交易摘要
        let trade_summary = self.build_trade_summary(transaction)?;
        
        // 2. 成本分解分析
        let cost_breakdown = self.cost_analyzer.analyze(
            transaction, 
            market_data, 
            &self.config,
        )?;
        
        // 3. 基准表现分析
        // 计算平均执行价格
        let total_value: f64 = transaction.fills.iter()
            .map(|f| f.price * f.quantity)
            .sum();
        let total_quantity: f64 = transaction.fills.iter()
            .map(|f| f.quantity)
            .sum();
        let average_price = if total_quantity > 0.0 {
            total_value / total_quantity
        } else {
            0.0
        };
        
        // 将MarketDataHistory转换为HashMap<String, f64>
        let mut market_data_map = HashMap::new();
        
        // 获取最新价格
        let latest_price = market_data.price_data.last()
            .map(|p| p.price)
            .unwrap_or(0.0);
        
        // 计算总成交量
        let total_volume: f64 = market_data.volume_data.iter()
            .map(|v| v.volume)
            .sum();
        
        // 计算VWAP (Volume Weighted Average Price)
        let total_value: f64 = market_data.price_data.iter()
            .zip(market_data.volume_data.iter())
            .map(|(p, v)| p.price * v.volume)
            .sum();
        let vwap = if total_volume > 0.0 { total_value / total_volume } else { latest_price };
        
        market_data_map.insert("last_price".to_string(), latest_price);
        market_data_map.insert("volume".to_string(), total_volume);
        market_data_map.insert("vwap".to_string(), vwap);
        
        let benchmark_result = self.benchmark_engine.compare_execution(
            average_price,
            "vwap", // default benchmark
            &market_data_map,
        )?;
        
        // 创建BenchmarkPerformance结构 - 转换benchmark_comparison::BenchmarkResult到tca::BenchmarkResult
        let tca_benchmark_result = BenchmarkResult {
            benchmark_name: benchmark_result.benchmark_name.clone(),
            benchmark_price: benchmark_result.benchmark_price,
            performance_bps: benchmark_result.slippage_bps, // 使用slippage_bps作为performance_bps
            outperformance_probability: if benchmark_result.slippage_bps < 0.0 { 0.7 } else { 0.3 }, // 简单估计
            statistical_significance: benchmark_result.performance_ratio.abs().min(1.0), // 基于performance_ratio计算
        };
        
        let mut benchmarks = HashMap::new();
        benchmarks.insert("vwap".to_string(), tca_benchmark_result);
        let benchmark_performance = BenchmarkPerformance {
            benchmarks: benchmarks.clone(),
            best_benchmark: "vwap".to_string(),
            worst_benchmark: "vwap".to_string(),
            benchmark_consistency: 1.0, // 只有一个基准时一致性为1.0
        };
        
        // 4. 归因分析
        let attribution_analysis = self.perform_attribution_analysis(
            transaction,
            &cost_breakdown,
            market_data,
        )?;
        
        // 5. 计算质量指标
        let quality_metrics = self.calculate_quality_metrics(
            transaction,
            &cost_breakdown,
            &benchmark_performance,
        )?;
        
        // 6. 计算风险指标
        let risk_metrics = self.calculate_risk_metrics(
            transaction,
            &cost_breakdown,
            market_data,
        )?;
        
        // 7. 生成交易建议
        let recommendations = self.generate_recommendations(
            &cost_breakdown,
            &attribution_analysis,
            &quality_metrics,
        )?;
        
        let analysis_duration = analysis_start.elapsed();
        log::info!("TCA analysis completed in {:?}", analysis_duration);
        
        Ok(TCAResult {
            transaction_id: transaction.transaction_id.clone(),
            symbol: transaction.symbol.clone(),
            analysis_timestamp: Utc::now(),
            trade_summary,
            cost_breakdown,
            benchmark_performance,
            attribution_analysis,
            quality_metrics,
            risk_metrics,
            recommendations,
        })
    }

    /// 批量分析
    pub fn analyze_batch(
        &mut self,
        transactions: Vec<ExecutionTransaction>,
        market_data: &MarketDataHistory,
    ) -> Result<Vec<TCAResult>> {
        let mut results = Vec::new();
        
        for transaction in transactions {
            match self.analyze_execution(&transaction, market_data) {
                Ok(result) => results.push(result),
                Err(e) => tracing::error!("Failed to analyze transaction {}: {}", 
                    transaction.transaction_id, e),
            }
        }
        
        Ok(results)
    }

    /// 构建交易摘要
    fn build_trade_summary(&self, transaction: &ExecutionTransaction) -> Result<TradeSummary> {
        let fills = &transaction.fills;
        
        if fills.is_empty() {
            return Err(anyhow::anyhow!("No fills in transaction"));
        }
        
        // 基本统计
        let total_quantity: f64 = fills.iter().map(|f| f.quantity).sum();
        let total_notional: f64 = fills.iter().map(|f| f.quantity * f.price).sum();
        let average_price = if total_quantity > 0.0 { total_notional / total_quantity } else { 0.0 };
        
        // 时间统计
        let first_fill_time = fills.iter().map(|f| f.timestamp).min().unwrap();
        let last_fill_time = fills.iter().map(|f| f.timestamp).max().unwrap();
        let execution_duration = last_fill_time.signed_duration_since(first_fill_time);
        
        // 场所统计
        let venues_used: std::collections::HashSet<String> = fills.iter()
            .map(|f| f.venue.clone())
            .collect();
        
        Ok(TradeSummary {
            order_id: transaction.order_id.clone(),
            side: transaction.side.clone(),
            quantity: total_quantity,
            notional_value: total_notional,
            average_price,
            first_fill_time,
            last_fill_time,
            execution_duration_minutes: execution_duration.num_minutes() as f64,
            fill_rate: total_quantity / transaction.original_quantity,
            number_of_fills: fills.len() as u32,
            venues_used: venues_used.into_iter().collect(),
        })
    }

    /// 执行归因分析
    fn perform_attribution_analysis(
        &self,
        transaction: &ExecutionTransaction,
        cost_breakdown: &CostBreakdown,
        market_data: &MarketDataHistory,
    ) -> Result<AttributionAnalysis> {
        let mut strategy_attribution = HashMap::new();
        let mut venue_attribution = HashMap::new();
        let mut time_attribution = HashMap::new();
        let mut size_attribution = HashMap::new();
        let mut main_drivers = Vec::new();
        
        // 策略归因
        strategy_attribution.insert(
            transaction.strategy_id.clone(), 
            cost_breakdown.total_cost_bps,
        );
        
        // 场所归因
        for fill in &transaction.fills {
            let venue_cost = self.calculate_venue_cost(fill, cost_breakdown)?;
            venue_attribution.insert(fill.venue.clone(), venue_cost);
        }
        
        // 时间归因（按小时分组）
        for fill in &transaction.fills {
            let hour_key = format!("{}:00", fill.timestamp.hour());
            let hour_cost = self.calculate_time_cost(fill, cost_breakdown)?;
            *time_attribution.entry(hour_key).or_insert(0.0) += hour_cost;
        }
        
        // 规模归因
        let small_orders: f64 = transaction.fills.iter()
            .filter(|f| f.quantity < 1000.0)
            .map(|f| f.quantity * f.price)
            .sum();
        let large_orders: f64 = transaction.fills.iter()
            .filter(|f| f.quantity >= 1000.0)
            .map(|f| f.quantity * f.price)
            .sum();
            
        if small_orders > 0.0 {
            size_attribution.insert("Small (<1000)".to_string(), 
                cost_breakdown.total_cost_bps * 0.3);
        }
        if large_orders > 0.0 {
            size_attribution.insert("Large (>=1000)".to_string(), 
                cost_breakdown.total_cost_bps * 0.7);
        }
        
        // 识别主要驱动因子
        if cost_breakdown.market_impact_bps > cost_breakdown.total_cost_bps * 0.3 {
            main_drivers.push(AttributionDriver {
                factor: "Market Impact".to_string(),
                contribution_bps: cost_breakdown.market_impact_bps,
                significance: 0.8,
                explanation: "High market impact due to large order size or low liquidity".to_string(),
            });
        }
        
        if cost_breakdown.timing_cost_bps > cost_breakdown.total_cost_bps * 0.2 {
            main_drivers.push(AttributionDriver {
                factor: "Timing Cost".to_string(),
                contribution_bps: cost_breakdown.timing_cost_bps,
                significance: 0.6,
                explanation: "Unfavorable timing relative to market movement".to_string(),
            });
        }
        
        Ok(AttributionAnalysis {
            strategy_attribution,
            venue_attribution,
            time_attribution,
            size_attribution,
            main_drivers,
        })
    }

    /// 计算质量指标
    fn calculate_quality_metrics(
        &self,
        transaction: &ExecutionTransaction,
        cost_breakdown: &CostBreakdown,
        benchmark_performance: &BenchmarkPerformance,
    ) -> Result<QualityMetrics> {
        // 执行评分（基于总成本的倒数）
        let execution_score = (100.0 / (1.0 + cost_breakdown.total_cost_bps.abs())).min(100.0);
        
        // 效率排名（基于成本百分位）
        let cost_percentile = self.calculate_cost_percentile(cost_breakdown.total_cost_bps)?;
        let efficiency_rank = match cost_percentile {
            p if p <= 0.1 => "Excellent".to_string(),
            p if p <= 0.25 => "Good".to_string(),
            p if p <= 0.75 => "Average".to_string(),
            _ => "Poor".to_string(),
        };
        
        // 一致性评分（基于基准表现的标准差）
        let benchmark_values: Vec<f64> = benchmark_performance.benchmarks
            .values()
            .map(|b| b.performance_bps)
            .collect();
        
        let consistency_score = if benchmark_values.len() > 1 {
            let mean = benchmark_values.iter().sum::<f64>() / benchmark_values.len() as f64;
            let variance = benchmark_values.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / benchmark_values.len() as f64;
            let std_dev = variance.sqrt();
            (1.0 / (1.0 + std_dev)).max(0.0).min(1.0)
        } else {
            1.0
        };
        
        // 流动性捕获率
        let liquidity_capture = if cost_breakdown.spread_cost_bps > 0.0 {
            1.0 - (cost_breakdown.realized_spread_bps / cost_breakdown.effective_spread_bps).abs()
        } else {
            1.0
        };
        
        // 逆向选择比率
        let adverse_selection_ratio = if cost_breakdown.permanent_impact_bps > 0.0 {
            cost_breakdown.permanent_impact_bps / cost_breakdown.market_impact_bps
        } else {
            0.0
        };
        
        // 时机Alpha（基于timing cost的相反数）
        let timing_alpha = -cost_breakdown.timing_cost_bps / 10.0; // 归一化到合理范围
        
        Ok(QualityMetrics {
            execution_score,
            efficiency_rank,
            consistency_score,
            speed_percentile: self.calculate_speed_percentile(transaction)?,
            cost_percentile,
            liquidity_capture: liquidity_capture.max(0.0).min(1.0),
            adverse_selection_ratio: adverse_selection_ratio.max(0.0).min(1.0),
            timing_alpha,
        })
    }

    /// 计算风险指标
    fn calculate_risk_metrics(
        &self,
        transaction: &ExecutionTransaction,
        cost_breakdown: &CostBreakdown,
        market_data: &MarketDataHistory,
    ) -> Result<RiskMetrics> {
        // 成本波动率（基于历史类似交易）
        let cost_volatility = self.estimate_cost_volatility(transaction, market_data)?;
        
        // 执行风险（未完成执行的风险）
        let execution_risk = 1.0 - (transaction.fills.iter().map(|f| f.quantity).sum::<f64>() / transaction.original_quantity);
        
        // 市场冲击风险
        let market_impact_risk = cost_breakdown.market_impact_bps / 100.0; // 归一化
        
        // 流动性风险
        let liquidity_risk = self.assess_liquidity_risk(transaction, market_data)?;
        
        // 集中度风险（场所集中度）
        let venue_concentrations: Vec<f64> = transaction.fills.iter()
            .fold(HashMap::new(), |mut acc, fill| {
                *acc.entry(&fill.venue).or_insert(0.0) += fill.quantity * fill.price;
                acc
            })
            .values()
            .map(|&notional| notional / (transaction.fills.iter().map(|f| f.quantity * f.price).sum::<f64>()))
            .collect();
        
        let concentration_risk = venue_concentrations.iter().map(|&c| c.powi(2)).sum::<f64>();
        
        // 简化的VaR和CVaR计算
        let cost_distribution = self.simulate_cost_distribution(cost_breakdown)?;
        let var_95_bps = self.calculate_var(&cost_distribution, 0.95)?;
        let cvar_95_bps = self.calculate_cvar(&cost_distribution, 0.95)?;
        
        Ok(RiskMetrics {
            cost_volatility,
            execution_risk: execution_risk.max(0.0).min(1.0),
            market_impact_risk: market_impact_risk.max(0.0).min(1.0),
            liquidity_risk,
            concentration_risk,
            tail_cost_risk: (cvar_95_bps - var_95_bps).max(0.0),
            var_95_bps,
            cvar_95_bps,
        })
    }

    /// 生成交易建议
    fn generate_recommendations(
        &self,
        cost_breakdown: &CostBreakdown,
        attribution_analysis: &AttributionAnalysis,
        quality_metrics: &QualityMetrics,
    ) -> Result<Vec<TradingRecommendation>> {
        let mut recommendations = Vec::new();
        
        // 高市场冲击建议
        if cost_breakdown.market_impact_bps > 10.0 {
            recommendations.push(TradingRecommendation {
                category: RecommendationCategory::OrderSizing,
                priority: RecommendationPriority::High,
                title: "Reduce Market Impact".to_string(),
                description: "Consider breaking large orders into smaller pieces or using TWAP/VWAP algorithms".to_string(),
                expected_improvement_bps: Some(cost_breakdown.market_impact_bps * 0.3),
                implementation_difficulty: 0.3,
                confidence_level: 0.8,
            });
        }
        
        // 高时机成本建议
        if cost_breakdown.timing_cost_bps > 5.0 {
            recommendations.push(TradingRecommendation {
                category: RecommendationCategory::Timing,
                priority: RecommendationPriority::Medium,
                title: "Improve Execution Timing".to_string(),
                description: "Consider using predictive models to optimize order timing and reduce adverse price movements".to_string(),
                expected_improvement_bps: Some(cost_breakdown.timing_cost_bps * 0.4),
                implementation_difficulty: 0.7,
                confidence_level: 0.6,
            });
        }
        
        // 场所分散建议
        if attribution_analysis.venue_attribution.len() == 1 {
            recommendations.push(TradingRecommendation {
                category: RecommendationCategory::VenueSelection,
                priority: RecommendationPriority::Medium,
                title: "Diversify Venue Usage".to_string(),
                description: "Consider routing orders to multiple venues to improve liquidity access and reduce costs".to_string(),
                expected_improvement_bps: Some(2.0),
                implementation_difficulty: 0.4,
                confidence_level: 0.7,
            });
        }
        
        // 低效率算法建议
        if quality_metrics.execution_score < 60.0 {
            recommendations.push(TradingRecommendation {
                category: RecommendationCategory::AlgorithmChoice,
                priority: RecommendationPriority::High,
                title: "Optimize Algorithm Selection".to_string(),
                description: "Current execution algorithm may not be optimal for this order profile. Consider alternative strategies".to_string(),
                expected_improvement_bps: Some(cost_breakdown.total_cost_bps * 0.2),
                implementation_difficulty: 0.5,
                confidence_level: 0.75,
            });
        }
        
        recommendations.sort_by(|a, b| {
            // 按优先级和预期改善排序
            let priority_order = |p: &RecommendationPriority| match p {
                RecommendationPriority::Critical => 0,
                RecommendationPriority::High => 1,
                RecommendationPriority::Medium => 2,
                RecommendationPriority::Low => 3,
            };
            
            priority_order(&a.priority).cmp(&priority_order(&b.priority))
                .then_with(|| {
                    let a_improvement = a.expected_improvement_bps.unwrap_or(0.0);
                    let b_improvement = b.expected_improvement_bps.unwrap_or(0.0);
                    b_improvement.partial_cmp(&a_improvement).unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        
        Ok(recommendations)
    }

    // 辅助计算方法
    fn calculate_venue_cost(&self, fill: &Fill, cost_breakdown: &CostBreakdown) -> Result<f64> {
        // 简化：按成交量比例分配成本
        Ok(cost_breakdown.total_cost_bps * 0.1) // 占位实现
    }

    fn calculate_time_cost(&self, fill: &Fill, cost_breakdown: &CostBreakdown) -> Result<f64> {
        Ok(cost_breakdown.total_cost_bps * 0.1) // 占位实现
    }

    fn calculate_cost_percentile(&self, total_cost_bps: f64) -> Result<f64> {
        // 简化：基于经验分布
        let normalized_cost = (total_cost_bps / 50.0).min(2.0); // 50bp为中位数基准
        Ok(0.5 * (1.0 + normalized_cost.tanh())) // S曲线映射到[0,1]
    }

    fn calculate_speed_percentile(&self, transaction: &ExecutionTransaction) -> Result<f64> {
        // 基于执行时长计算速度百分位
        let execution_minutes = if !transaction.fills.is_empty() {
            let first = transaction.fills.iter().map(|f| f.timestamp).min().unwrap();
            let last = transaction.fills.iter().map(|f| f.timestamp).max().unwrap();
            last.signed_duration_since(first).num_minutes() as f64
        } else {
            0.0
        };
        
        // 简化映射：越快百分位越高
        let speed_score = 1.0 / (1.0 + execution_minutes / 10.0); // 10分钟为基准
        Ok(speed_score)
    }

    fn estimate_cost_volatility(&self, transaction: &ExecutionTransaction, market_data: &MarketDataHistory) -> Result<f64> {
        // 简化：基于市场波动率估算成本波动率
        Ok(0.05) // 5% 占位值
    }

    fn assess_liquidity_risk(&self, transaction: &ExecutionTransaction, market_data: &MarketDataHistory) -> Result<f64> {
        // 简化：基于平均价差估算流动性风险
        Ok(0.1) // 10% 占位值
    }

    fn simulate_cost_distribution(&self, cost_breakdown: &CostBreakdown) -> Result<Vec<f64>> {
        // 简化：正态分布模拟
        let mean = cost_breakdown.total_cost_bps;
        let std_dev = mean * 0.2; // 20%相对标准差
        let mut distribution = Vec::new();
        
        for _ in 0..10000 {
            let random_cost = mean + std_dev * rand::random::<f64>(); // 简化随机生成
            distribution.push(random_cost);
        }
        
        distribution.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(distribution)
    }

    fn calculate_var(&self, distribution: &[f64], confidence_level: f64) -> Result<f64> {
        let index = ((1.0 - confidence_level) * distribution.len() as f64) as usize;
        Ok(distribution.get(index).copied().unwrap_or(0.0))
    }

    fn calculate_cvar(&self, distribution: &[f64], confidence_level: f64) -> Result<f64> {
        let var_index = ((1.0 - confidence_level) * distribution.len() as f64) as usize;
        if var_index > 0 {
            let tail_values = &distribution[0..var_index];
            Ok(tail_values.iter().sum::<f64>() / tail_values.len() as f64)
        } else {
            Ok(0.0)
        }
    }

    /// 生成TCA报告
    pub fn generate_report(&self, result: &TCAResult) -> Result<String> {
        let mut report = String::new();
        
        report.push_str("=== 交易成本分析报告 (TCA) ===\n\n");
        
        // 交易摘要
        report.push_str("## 交易摘要\n");
        report.push_str(&format!("交易ID: {}\n", result.transaction_id));
        report.push_str(&format!("标的: {}\n", result.symbol));
        report.push_str(&format!("总成交量: {:.0}\n", result.trade_summary.quantity));
        report.push_str(&format!("平均价格: {:.4}\n", result.trade_summary.average_price));
        report.push_str(&format!("执行时长: {:.1} 分钟\n", result.trade_summary.execution_duration_minutes));
        report.push_str(&format!("成交率: {:.1}%\n", result.trade_summary.fill_rate * 100.0));
        
        // 成本分解
        report.push_str("\n## 成本分解\n");
        report.push_str(&format!("总成本: {:.2} bp\n", result.cost_breakdown.total_cost_bps));
        report.push_str(&format!("  市场冲击: {:.2} bp\n", result.cost_breakdown.market_impact_bps));
        report.push_str(&format!("  时机成本: {:.2} bp\n", result.cost_breakdown.timing_cost_bps));
        report.push_str(&format!("  价差成本: {:.2} bp\n", result.cost_breakdown.spread_cost_bps));
        report.push_str(&format!("  手续费: {:.2} bp\n", result.cost_breakdown.commission_bps));
        
        // 质量指标
        report.push_str("\n## 执行质量\n");
        report.push_str(&format!("执行评分: {:.1}/100\n", result.quality_metrics.execution_score));
        report.push_str(&format!("效率排名: {}\n", result.quality_metrics.efficiency_rank));
        report.push_str(&format!("一致性评分: {:.3}\n", result.quality_metrics.consistency_score));
        
        // 建议
        report.push_str("\n## 改进建议\n");
        for (i, rec) in result.recommendations.iter().enumerate() {
            report.push_str(&format!("{}. {} (优先级: {:?})\n", 
                i + 1, rec.title, rec.priority));
            report.push_str(&format!("   {}\n", rec.description));
            if let Some(improvement) = rec.expected_improvement_bps {
                report.push_str(&format!("   预期改善: {:.1} bp\n", improvement));
            }
        }
        
        Ok(report)
    }
}

// 数据结构定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTransaction {
    pub transaction_id: String,
    pub order_id: String,
    pub strategy_id: String,
    pub symbol: String,
    pub side: String,
    pub original_quantity: f64,
    pub fills: Vec<Fill>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub fill_id: String,
    pub quantity: f64,
    pub price: f64,
    pub timestamp: DateTime<Utc>,
    pub venue: String,
    pub commission: f64,
    pub liquidity_flag: String,
}

#[derive(Debug, Clone)]
pub struct MarketDataHistory {
    pub symbol: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub price_data: Vec<PricePoint>,
    pub volume_data: Vec<VolumePoint>,
}

#[derive(Debug, Clone)]
pub struct PricePoint {
    pub timestamp: DateTime<Utc>,
    pub price: f64,
}

#[derive(Debug, Clone)]
pub struct VolumePoint {
    pub timestamp: DateTime<Utc>,
    pub volume: f64,
}

impl Default for TCAConfig {
    fn default() -> Self {
        Self {
            analysis_window_hours: 24,
            benchmark_methods: vec![
                BenchmarkMethod::Arrival,
                BenchmarkMethod::TWAP(30),
                BenchmarkMethod::VWAP(30),
            ],
            cost_components: CostComponents {
                market_impact: true,
                timing_cost: true,
                spread_cost: true,
                commission: true,
                slippage: true,
                opportunity_cost: true,
                delay_cost: true,
            },
            attribution_levels: vec![
                AttributionLevel::Strategy,
                AttributionLevel::Venue,
                AttributionLevel::TimeOfDay,
            ],
            reporting_currency: "USD".to_string(),
            include_opportunity_cost: true,
            real_time_analysis: false,
            confidence_intervals: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tca_engine_creation() {
        let config = TCAConfig::default();
        let engine = TCAEngine::new(config);
        // 基本创建测试
    }

    #[test]
    fn test_trade_summary_calculation() {
        let config = TCAConfig::default();
        let engine = TCAEngine::new(config);
        
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
                    quantity: 500.0,
                    price: 150.0,
                    timestamp: Utc::now(),
                    venue: "NYSE".to_string(),
                    commission: 1.0,
                    liquidity_flag: "MAKER".to_string(),
                },
                Fill {
                    fill_id: "fill_2".to_string(),
                    quantity: 500.0,
                    price: 150.5,
                    timestamp: Utc::now(),
                    venue: "NASDAQ".to_string(),
                    commission: 1.5,
                    liquidity_flag: "TAKER".to_string(),
                },
            ],
            metadata: HashMap::new(),
        };
        
        let summary = engine.build_trade_summary(&transaction).unwrap();
        assert_eq!(summary.quantity, 1000.0);
        assert_eq!(summary.average_price, 150.25);
        assert_eq!(summary.fill_rate, 1.0);
    }
}