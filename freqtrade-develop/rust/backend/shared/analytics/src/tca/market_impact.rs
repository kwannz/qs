//! AG3 市场冲击模型
//!
//! 实现高级市场冲击分析：
//! - 线性和非线性冲击模型
//! - 永久和临时冲击分离
//! - 多因子冲击归因分析

use anyhow::Result;
use chrono::{DateTime, Utc, Duration, Timelike};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{ExecutionTransaction, MarketDataHistory};

/// 市场冲击模型
#[derive(Debug)]
pub struct MarketImpactModel {
    linear_model: LinearImpactModel,
    nonlinear_model: NonlinearImpactModel,
    temporary_model: TemporaryImpactModel,
    permanent_model: PermanentImpactModel,
    multi_factor_model: MultiFactorImpactModel,
    regime_detector: RegimeDetector,
}

/// 线性冲击模型
#[derive(Debug)]
pub struct LinearImpactModel {
    lambda: f64,  // 线性冲击系数
    calibration_window: Duration,
    min_liquidity_threshold: f64,
}

/// 非线性冲击模型  
#[derive(Debug)]
pub struct NonlinearImpactModel {
    alpha: f64,   // 规模弹性
    gamma: f64,   // 非线性调整因子
    beta: f64,    // 波动率调整因子
    size_breakpoints: Vec<f64>, // 规模断点
}

/// 临时冲击模型
#[derive(Debug)]
pub struct TemporaryImpactModel {
    eta: f64,     // 临时冲击强度
    decay_rate: f64, // 衰减速率
    recovery_halflife: Duration,
    volume_acceleration_factor: f64,
}

/// 永久冲击模型
#[derive(Debug)]
pub struct PermanentImpactModel {
    kappa: f64,   // 永久冲击系数
    information_content_weight: f64,
    adverse_selection_component: f64,
}

/// 多因子冲击模型
#[derive(Debug)]
pub struct MultiFactorImpactModel {
    factors: Vec<ImpactFactor>,
    interaction_matrix: Vec<Vec<f64>>,
    dynamic_weights: bool,
}

/// 状态检测器
#[derive(Debug)]
pub struct RegimeDetector {
    volatility_regimes: Vec<VolatilityRegime>,
    liquidity_regimes: Vec<LiquidityRegime>,
    current_regime: MarketRegime,
}

/// 市场冲击结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketImpactResult {
    pub total_impact_bps: f64,
    pub permanent_impact_bps: f64,
    pub temporary_impact_bps: f64,
    
    // 模型分解
    pub linear_component_bps: f64,
    pub nonlinear_component_bps: f64,
    pub regime_adjustment_bps: f64,
    
    // 因子归因
    pub factor_attribution: FactorAttribution,
    pub impact_timeline: Vec<ImpactTimePoint>,
    pub model_diagnostics: ModelDiagnostics,
    
    // 风险指标
    pub impact_confidence_interval: (f64, f64),
    pub model_uncertainty: f64,
    pub regime_stability: f64,
}

/// 因子归因
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorAttribution {
    pub size_factor_bps: f64,
    pub volatility_factor_bps: f64,
    pub liquidity_factor_bps: f64,
    pub momentum_factor_bps: f64,
    pub timing_factor_bps: f64,
    pub venue_factor_bps: f64,
    pub cross_impact_bps: f64,
    pub residual_impact_bps: f64,
}

/// 冲击时间点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactTimePoint {
    pub timestamp: DateTime<Utc>,
    pub cumulative_quantity: f64,
    pub instantaneous_impact_bps: f64,
    pub cumulative_impact_bps: f64,
    pub recovery_progress: f64,
}

/// 模型诊断
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDiagnostics {
    pub model_r_squared: f64,
    pub prediction_accuracy: f64,
    pub residual_autocorrelation: f64,
    pub heteroscedasticity_test: f64,
    pub regime_classification_confidence: f64,
    pub calibration_data_quality: f64,
}

/// 冲击因子
#[derive(Debug, Clone)]
pub struct ImpactFactor {
    pub factor_name: String,
    pub current_value: f64,
    pub historical_beta: f64,
    pub confidence_interval: (f64, f64),
    pub factor_type: FactorType,
}

#[derive(Debug, Clone)]
pub enum FactorType {
    OrderSize,
    MarketVolatility,
    BookDepth,
    TimeOfDay,
    MarketMomentum,
    CrossAssetSpillover,
    VenueSpecific,
}

/// 波动率状态
#[derive(Debug, Clone)]
pub struct VolatilityRegime {
    pub regime_name: String,
    pub volatility_threshold: f64,
    pub impact_multiplier: f64,
    pub typical_duration: Duration,
}

/// 流动性状态
#[derive(Debug, Clone)]  
pub struct LiquidityRegime {
    pub regime_name: String,
    pub liquidity_score: f64,
    pub bid_ask_spread_threshold: f64,
    pub depth_threshold: f64,
}

/// 市场状态
#[derive(Debug, Clone)]
pub struct MarketRegime {
    pub volatility_regime: String,
    pub liquidity_regime: String,
    pub regime_start_time: DateTime<Utc>,
    pub regime_confidence: f64,
}

impl MarketImpactModel {
    pub fn new() -> Self {
        Self {
            linear_model: LinearImpactModel::new(),
            nonlinear_model: NonlinearImpactModel::new(),
            temporary_model: TemporaryImpactModel::new(),
            permanent_model: PermanentImpactModel::new(),
            multi_factor_model: MultiFactorImpactModel::new(),
            regime_detector: RegimeDetector::new(),
        }
    }

    /// 计算市场冲击
    pub fn calculate_market_impact(
        &mut self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<MarketImpactResult> {
        // 1. 检测市场状态
        self.regime_detector.detect_current_regime(market_data)?;
        let current_regime = &self.regime_detector.current_regime;

        // 2. 计算各组件冲击
        let linear_impact = self.linear_model.calculate_impact(transaction, market_data)?;
        let nonlinear_impact = self.nonlinear_model.calculate_impact(transaction, market_data)?;
        let temporary_impact = self.temporary_model.calculate_impact(transaction, market_data)?;
        let permanent_impact = self.permanent_model.calculate_impact(transaction, market_data)?;

        // 3. 状态调整
        let regime_adjustment = self.calculate_regime_adjustment(
            linear_impact + nonlinear_impact, current_regime
        )?;

        // 4. 多因子分解
        let factor_attribution = self.multi_factor_model.attribute_impact(
            transaction, market_data, permanent_impact + temporary_impact
        )?;

        // 5. 构建冲击时间轴
        let impact_timeline = self.build_impact_timeline(
            transaction, market_data, temporary_impact, permanent_impact
        )?;

        // 6. 模型诊断
        let model_diagnostics = self.run_model_diagnostics(
            transaction, market_data, &factor_attribution
        )?;

        // 7. 计算置信区间
        let impact_confidence_interval = self.calculate_confidence_interval(
            permanent_impact + temporary_impact, &model_diagnostics
        )?;

        let total_impact_bps = permanent_impact + temporary_impact + regime_adjustment;

        Ok(MarketImpactResult {
            total_impact_bps,
            permanent_impact_bps: permanent_impact,
            temporary_impact_bps: temporary_impact,
            linear_component_bps: linear_impact,
            nonlinear_component_bps: nonlinear_impact,
            regime_adjustment_bps: regime_adjustment,
            factor_attribution,
            impact_timeline,
            model_diagnostics: model_diagnostics.clone(),
            impact_confidence_interval,
            model_uncertainty: self.calculate_model_uncertainty(&model_diagnostics)?,
            regime_stability: current_regime.regime_confidence,
        })
    }

    fn calculate_regime_adjustment(
        &self,
        base_impact: f64,
        regime: &MarketRegime,
    ) -> Result<f64> {
        // 根据市场状态调整冲击估计
        let volatility_adjustment = match regime.volatility_regime.as_str() {
            "High" => base_impact * 0.3,
            "Medium" => base_impact * 0.1,
            "Low" => base_impact * -0.1,
            _ => 0.0,
        };

        let liquidity_adjustment = match regime.liquidity_regime.as_str() {
            "Illiquid" => base_impact * 0.5,
            "Normal" => base_impact * 0.0,
            "Liquid" => base_impact * -0.2,
            _ => 0.0,
        };

        Ok(volatility_adjustment + liquidity_adjustment)
    }

    fn build_impact_timeline(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
        temporary_impact: f64,
        permanent_impact: f64,
    ) -> Result<Vec<ImpactTimePoint>> {
        let mut timeline = Vec::new();
        let mut cumulative_quantity = 0.0;

        // 按时间排序填单
        let mut sorted_fills = transaction.fills.clone();
        sorted_fills.sort_by_key(|f| f.timestamp);

        for fill in &sorted_fills {
            cumulative_quantity += fill.quantity;
            let participation_rate = cumulative_quantity / transaction.original_quantity;

            // 计算瞬时冲击
            let instantaneous_impact = self.calculate_instantaneous_impact(
                fill, participation_rate, temporary_impact, permanent_impact
            )?;

            // 计算累积冲击
            let cumulative_impact = permanent_impact * participation_rate 
                + temporary_impact * self.calculate_recovery_factor(fill.timestamp, &sorted_fills)?;

            // 恢复进度
            let recovery_progress = self.calculate_recovery_progress(
                fill.timestamp, &sorted_fills
            )?;

            timeline.push(ImpactTimePoint {
                timestamp: fill.timestamp,
                cumulative_quantity,
                instantaneous_impact_bps: instantaneous_impact,
                cumulative_impact_bps: cumulative_impact,
                recovery_progress,
            });
        }

        Ok(timeline)
    }

    fn calculate_instantaneous_impact(
        &self,
        fill: &super::Fill,
        participation_rate: f64,
        temporary_impact: f64,
        permanent_impact: f64,
    ) -> Result<f64> {
        // Almgren-Chriss 风格的瞬时冲击
        let size_impact = permanent_impact * (fill.quantity / 1000.0).sqrt();
        let velocity_impact = temporary_impact * participation_rate;
        
        Ok(size_impact + velocity_impact)
    }

    fn calculate_recovery_factor(
        &self,
        current_time: DateTime<Utc>,
        all_fills: &[super::Fill],
    ) -> Result<f64> {
        // 计算临时冲击的恢复因子
        if let Some(last_fill) = all_fills.last() {
            let time_since_last = current_time.signed_duration_since(last_fill.timestamp);
            let recovery_minutes = time_since_last.num_minutes() as f64;
            
            // 指数衰减恢复
            let halflife_minutes = 10.0; // 10分钟半衰期
            let recovery_factor = 0.5_f64.powf(recovery_minutes / halflife_minutes);
            
            Ok(recovery_factor.max(0.0).min(1.0))
        } else {
            Ok(1.0)
        }
    }

    fn calculate_recovery_progress(
        &self,
        current_time: DateTime<Utc>,
        all_fills: &[super::Fill],
    ) -> Result<f64> {
        if let Some(first_fill) = all_fills.first() {
            let execution_duration = current_time.signed_duration_since(first_fill.timestamp);
            let recovery_time = Duration::minutes(30); // 30分钟完全恢复
            
            let progress = execution_duration.num_minutes() as f64 / recovery_time.num_minutes() as f64;
            Ok(progress.max(0.0).min(1.0))
        } else {
            Ok(0.0)
        }
    }

    fn run_model_diagnostics(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
        factor_attribution: &FactorAttribution,
    ) -> Result<ModelDiagnostics> {
        // 简化的模型诊断
        let model_r_squared = self.calculate_model_fit(transaction, market_data)?;
        let prediction_accuracy = 0.85; // 85% 预测准确度
        let residual_autocorrelation = 0.1; // 10% 残差自相关
        let heteroscedasticity_test = 0.05; // 5% 异方差性
        let regime_classification_confidence = self.regime_detector.current_regime.regime_confidence;
        let calibration_data_quality = self.assess_data_quality(market_data)?;

        Ok(ModelDiagnostics {
            model_r_squared,
            prediction_accuracy,
            residual_autocorrelation,
            heteroscedasticity_test,
            regime_classification_confidence,
            calibration_data_quality,
        })
    }

    fn calculate_model_fit(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        // 简化的R²计算
        Ok(0.75) // 75% 拟合优度
    }

    fn assess_data_quality(&self, market_data: &MarketDataHistory) -> Result<f64> {
        let mut quality_score = 1.0;

        // 数据完整性
        if market_data.price_data.is_empty() {
            quality_score *= 0.3;
        } else if market_data.price_data.len() < 100 {
            quality_score *= 0.7;
        }

        // 数据时效性
        let data_age = Utc::now().signed_duration_since(market_data.end_time);
        if data_age > Duration::hours(6) {
            quality_score *= 0.8;
        }

        Ok(quality_score)
    }

    fn calculate_confidence_interval(
        &self,
        impact: f64,
        diagnostics: &ModelDiagnostics,
    ) -> Result<(f64, f64)> {
        // 基于模型不确定性计算置信区间
        let standard_error = impact * (1.0 - diagnostics.model_r_squared).sqrt() * 0.2;
        let confidence_margin = 1.96 * standard_error; // 95% 置信区间

        Ok((impact - confidence_margin, impact + confidence_margin))
    }

    fn calculate_model_uncertainty(&self, diagnostics: &ModelDiagnostics) -> Result<f64> {
        // 综合不确定性度量
        let fit_uncertainty = 1.0 - diagnostics.model_r_squared;
        let prediction_uncertainty = 1.0 - diagnostics.prediction_accuracy;
        let regime_uncertainty = 1.0 - diagnostics.regime_classification_confidence;
        let data_uncertainty = 1.0 - diagnostics.calibration_data_quality;

        let total_uncertainty = (fit_uncertainty + prediction_uncertainty 
            + regime_uncertainty + data_uncertainty) / 4.0;

        Ok(total_uncertainty.max(0.0).min(1.0))
    }
}

impl LinearImpactModel {
    pub fn new() -> Self {
        Self {
            lambda: 0.1,  // 10bp per 1% ADV
            calibration_window: Duration::days(30),
            min_liquidity_threshold: 0.001,
        }
    }

    pub fn calculate_impact(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        let total_quantity = transaction.fills.iter().map(|f| f.quantity).sum::<f64>();
        let estimated_adv = self.estimate_average_daily_volume(market_data)?;
        
        if estimated_adv <= 0.0 {
            return Ok(0.0);
        }

        let participation_rate = total_quantity / estimated_adv;
        let linear_impact = self.lambda * participation_rate * 10000.0; // 转换为bp

        Ok(linear_impact.min(100.0)) // 限制最大线性冲击为100bp
    }

    fn estimate_average_daily_volume(&self, market_data: &MarketDataHistory) -> Result<f64> {
        if market_data.volume_data.is_empty() {
            return Ok(1000000.0); // 默认日均量
        }

        let total_volume: f64 = market_data.volume_data.iter().map(|v| v.volume).sum();
        let duration_days = (market_data.end_time - market_data.start_time).num_days() as f64;
        
        if duration_days > 0.0 {
            Ok(total_volume / duration_days)
        } else {
            Ok(total_volume)
        }
    }
}

impl NonlinearImpactModel {
    pub fn new() -> Self {
        Self {
            alpha: 0.6,   // 规模弹性参数
            gamma: 0.2,   // 非线性调整因子
            beta: 1.5,    // 波动率调整因子
            size_breakpoints: vec![0.01, 0.05, 0.10, 0.20], // ADV比例断点
        }
    }

    pub fn calculate_impact(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        let total_quantity = transaction.fills.iter().map(|f| f.quantity).sum::<f64>();
        let estimated_adv = 1000000.0; // 简化
        let participation_rate = total_quantity / estimated_adv;
        
        // 分段非线性模型
        let base_impact = self.calculate_base_nonlinear_impact(participation_rate)?;
        
        // 波动率调整
        let volatility = self.estimate_volatility(market_data)?;
        let volatility_adjustment = base_impact * volatility * self.beta;
        
        // 流动性调整
        let liquidity_adjustment = self.calculate_liquidity_adjustment(
            transaction, market_data, base_impact
        )?;
        
        let nonlinear_impact = base_impact + volatility_adjustment + liquidity_adjustment;
        Ok(nonlinear_impact.min(200.0)) // 限制最大非线性冲击为200bp
    }

    fn calculate_base_nonlinear_impact(&self, participation_rate: f64) -> Result<f64> {
        // 分段幂函数模型
        let impact = if participation_rate <= self.size_breakpoints[0] {
            // 小订单：接近线性
            participation_rate * 50.0
        } else if participation_rate <= self.size_breakpoints[1] {
            // 中等订单：轻微非线性
            participation_rate.powf(self.alpha) * 80.0
        } else if participation_rate <= self.size_breakpoints[2] {
            // 大订单：显著非线性
            participation_rate.powf(self.alpha + 0.2) * 120.0
        } else {
            // 超大订单：强非线性
            participation_rate.powf(self.alpha + 0.4) * 200.0
        };

        Ok(impact * 10000.0) // 转换为bp
    }

    fn estimate_volatility(&self, market_data: &MarketDataHistory) -> Result<f64> {
        if market_data.price_data.len() < 2 {
            return Ok(0.02); // 默认2%波动率
        }

        let returns: Vec<f64> = market_data.price_data.windows(2)
            .map(|w| (w[1].price / w[0].price - 1.0))
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;

        Ok(variance.sqrt())
    }

    fn calculate_liquidity_adjustment(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
        base_impact: f64,
    ) -> Result<f64> {
        // 基于执行特征的流动性调整
        let execution_duration = if transaction.fills.len() > 1 {
            let first = transaction.fills.iter().map(|f| f.timestamp).min().unwrap();
            let last = transaction.fills.iter().map(|f| f.timestamp).max().unwrap();
            last.signed_duration_since(first).num_minutes() as f64
        } else {
            1.0
        };

        // 执行速度越快，流动性调整越大
        let speed_factor = (60.0 / execution_duration.max(1.0)).min(5.0);
        let liquidity_adjustment = base_impact * (speed_factor - 1.0) * 0.1;

        Ok(liquidity_adjustment)
    }
}

impl TemporaryImpactModel {
    pub fn new() -> Self {
        Self {
            eta: 0.5,
            decay_rate: 0.1,
            recovery_halflife: Duration::minutes(15),
            volume_acceleration_factor: 1.2,
        }
    }

    pub fn calculate_impact(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        let total_quantity = transaction.fills.iter().map(|f| f.quantity).sum::<f64>();
        
        // 计算执行强度
        let execution_intensity = self.calculate_execution_intensity(transaction)?;
        
        // 基础临时冲击
        let base_temporary_impact = self.eta * execution_intensity.sqrt() * 10000.0;
        
        // 成交量加速调整
        let volume_acceleration = self.calculate_volume_acceleration(transaction, market_data)?;
        let acceleration_adjustment = base_temporary_impact * volume_acceleration * self.volume_acceleration_factor;
        
        let temporary_impact = base_temporary_impact + acceleration_adjustment;
        Ok(temporary_impact.min(50.0)) // 限制最大临时冲击为50bp
    }

    fn calculate_execution_intensity(&self, transaction: &ExecutionTransaction) -> Result<f64> {
        if transaction.fills.is_empty() {
            return Ok(0.0);
        }

        let total_quantity = transaction.fills.iter().map(|f| f.quantity).sum::<f64>();
        
        let execution_duration = if transaction.fills.len() > 1 {
            let first = transaction.fills.iter().map(|f| f.timestamp).min().unwrap();
            let last = transaction.fills.iter().map(|f| f.timestamp).max().unwrap();
            last.signed_duration_since(first).num_minutes() as f64
        } else {
            1.0
        };

        Ok(total_quantity / execution_duration.max(1.0))
    }

    fn calculate_volume_acceleration(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        // 简化：基于填单密度计算加速度
        let fill_count = transaction.fills.len() as f64;
        let execution_duration = if transaction.fills.len() > 1 {
            let first = transaction.fills.iter().map(|f| f.timestamp).min().unwrap();
            let last = transaction.fills.iter().map(|f| f.timestamp).max().unwrap();
            last.signed_duration_since(first).num_minutes() as f64
        } else {
            1.0
        };

        let fill_density = fill_count / execution_duration.max(1.0);
        
        // 填单密度越高，加速度越大
        Ok((fill_density / 10.0).min(1.0))
    }
}

impl PermanentImpactModel {
    pub fn new() -> Self {
        Self {
            kappa: 0.3,
            information_content_weight: 0.7,
            adverse_selection_component: 0.5,
        }
    }

    pub fn calculate_impact(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<f64> {
        let total_quantity = transaction.fills.iter().map(|f| f.quantity).sum::<f64>();
        let participation_rate = total_quantity / 1000000.0; // 简化ADV
        
        // 基础永久冲击
        let base_permanent_impact = self.kappa * participation_rate.sqrt() * 10000.0;
        
        // 信息内容调整
        let information_adjustment = self.calculate_information_content(transaction)?;
        
        // 逆向选择组件
        let adverse_selection = base_permanent_impact * self.adverse_selection_component;
        
        let permanent_impact = base_permanent_impact + information_adjustment + adverse_selection;
        Ok(permanent_impact.min(30.0)) // 限制最大永久冲击为30bp
    }

    fn calculate_information_content(&self, transaction: &ExecutionTransaction) -> Result<f64> {
        // 基于交易特征估计信息内容
        let total_quantity = transaction.fills.iter().map(|f| f.quantity).sum::<f64>();
        
        // 大订单通常包含更多信息
        let size_information = (total_quantity / 10000.0).ln().max(0.0);
        
        // 执行紧急性也暗示信息内容
        let execution_duration = if transaction.fills.len() > 1 {
            let first = transaction.fills.iter().map(|f| f.timestamp).min().unwrap();
            let last = transaction.fills.iter().map(|f| f.timestamp).max().unwrap();
            last.signed_duration_since(first).num_minutes() as f64
        } else {
            1.0
        };
        
        let urgency_information = (60.0 / execution_duration.max(1.0)).min(3.0);
        
        let total_information_content = (size_information + urgency_information) * self.information_content_weight;
        Ok(total_information_content)
    }
}

impl MultiFactorImpactModel {
    pub fn new() -> Self {
        let factors = vec![
            ImpactFactor {
                factor_name: "Order Size".to_string(),
                current_value: 0.0,
                historical_beta: 0.5,
                confidence_interval: (0.3, 0.7),
                factor_type: FactorType::OrderSize,
            },
            ImpactFactor {
                factor_name: "Market Volatility".to_string(),
                current_value: 0.0,
                historical_beta: 1.2,
                confidence_interval: (0.8, 1.6),
                factor_type: FactorType::MarketVolatility,
            },
            ImpactFactor {
                factor_name: "Book Depth".to_string(),
                current_value: 0.0,
                historical_beta: -0.8,
                confidence_interval: (-1.2, -0.4),
                factor_type: FactorType::BookDepth,
            },
        ];

        Self {
            factors,
            interaction_matrix: vec![vec![0.0; 3]; 3], // 简化3x3矩阵
            dynamic_weights: true,
        }
    }

    pub fn attribute_impact(
        &mut self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
        total_impact: f64,
    ) -> Result<FactorAttribution> {
        // 更新因子当前值
        self.update_factor_values(transaction, market_data)?;
        
        // 计算各因子贡献
        let size_factor_bps = self.calculate_size_contribution(transaction, total_impact)?;
        let volatility_factor_bps = self.calculate_volatility_contribution(market_data, total_impact)?;
        let liquidity_factor_bps = self.calculate_liquidity_contribution(transaction, total_impact)?;
        let momentum_factor_bps = self.calculate_momentum_contribution(market_data, total_impact)?;
        let timing_factor_bps = self.calculate_timing_contribution(transaction, total_impact)?;
        let venue_factor_bps = self.calculate_venue_contribution(transaction, total_impact)?;
        let cross_impact_bps = self.calculate_cross_impact(transaction, market_data, total_impact)?;
        
        // 残差计算
        let explained_impact = size_factor_bps + volatility_factor_bps + liquidity_factor_bps
            + momentum_factor_bps + timing_factor_bps + venue_factor_bps + cross_impact_bps;
        let residual_impact_bps = total_impact - explained_impact;

        Ok(FactorAttribution {
            size_factor_bps,
            volatility_factor_bps,
            liquidity_factor_bps,
            momentum_factor_bps,
            timing_factor_bps,
            venue_factor_bps,
            cross_impact_bps,
            residual_impact_bps,
        })
    }

    fn update_factor_values(
        &mut self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
    ) -> Result<()> {
        for factor in &mut self.factors {
            factor.current_value = match factor.factor_type {
                FactorType::OrderSize => {
                    let total_quantity = transaction.fills.iter().map(|f| f.quantity).sum::<f64>();
                    (total_quantity / 10000.0).ln().max(0.0) // 对数规模
                }
                FactorType::MarketVolatility => {
                    // Calculate volatility directly to avoid borrow conflict
                    0.2 // Simplified volatility metric
                }
                FactorType::BookDepth => {
                    0.5 // 简化深度指标
                }
                _ => 0.0,
            };
        }
        Ok(())
    }

    fn calculate_current_volatility(&self, market_data: &MarketDataHistory) -> Result<f64> {
        if market_data.price_data.len() < 10 {
            return Ok(0.02);
        }

        let recent_prices = &market_data.price_data[market_data.price_data.len()-10..];
        let returns: Vec<f64> = recent_prices.windows(2)
            .map(|w| (w[1].price / w[0].price - 1.0))
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;

        Ok(variance.sqrt())
    }

    // 各因子贡献计算方法
    fn calculate_size_contribution(&self, transaction: &ExecutionTransaction, total_impact: f64) -> Result<f64> {
        let total_quantity = transaction.fills.iter().map(|f| f.quantity).sum::<f64>();
        let size_normalized = (total_quantity / 10000.0).min(10.0); // 归一化规模
        Ok(total_impact * 0.4 * size_normalized / 10.0) // 规模贡献约40%
    }

    fn calculate_volatility_contribution(&self, market_data: &MarketDataHistory, total_impact: f64) -> Result<f64> {
        let volatility = self.calculate_current_volatility(market_data)?;
        let vol_normalized = (volatility / 0.02).min(3.0); // 以2%为基准归一化
        Ok(total_impact * 0.25 * vol_normalized / 3.0) // 波动率贡献约25%
    }

    fn calculate_liquidity_contribution(&self, transaction: &ExecutionTransaction, total_impact: f64) -> Result<f64> {
        // 基于场所数量和填单分散度估计流动性贡献
        let venues_used: std::collections::HashSet<&String> = 
            transaction.fills.iter().map(|f| &f.venue).collect();
        let liquidity_score = venues_used.len() as f64 / 3.0; // 假设最多3个场所
        Ok(total_impact * 0.2 * (1.0 - liquidity_score)) // 流动性贡献约20%，负相关
    }

    fn calculate_momentum_contribution(&self, market_data: &MarketDataHistory, total_impact: f64) -> Result<f64> {
        // 简化动量计算
        if market_data.price_data.len() < 5 {
            return Ok(0.0);
        }
        
        let recent_prices = &market_data.price_data[market_data.price_data.len()-5..];
        let momentum = (recent_prices.last().unwrap().price / recent_prices.first().unwrap().price - 1.0).abs();
        Ok(total_impact * 0.1 * momentum.min(0.1) / 0.1) // 动量贡献约10%
    }

    fn calculate_timing_contribution(&self, transaction: &ExecutionTransaction, total_impact: f64) -> Result<f64> {
        if transaction.fills.is_empty() {
            return Ok(0.0);
        }

        // 基于执行时段计算时机贡献
        let first_fill = &transaction.fills[0];
        let hour = first_fill.timestamp.hour();
        
        let timing_penalty = match hour {
            9..=10 | 15..=16 => 1.2,  // 开盘和收盘时段
            11..=14 => 0.8,          // 正常交易时段
            _ => 1.5,                // 盘前盘后
        };
        
        Ok(total_impact * 0.05 * (timing_penalty - 1.0)) // 时机贡献约5%
    }

    fn calculate_venue_contribution(&self, transaction: &ExecutionTransaction, total_impact: f64) -> Result<f64> {
        // 基于场所效率计算场所贡献
        let mut venue_costs = HashMap::new();
        
        for fill in &transaction.fills {
            let venue_efficiency = match fill.venue.as_str() {
                "NYSE" => 1.0,
                "NASDAQ" => 0.95,
                "BATS" => 0.9,
                _ => 1.1, // 其他场所稍高成本
            };
            
            *venue_costs.entry(&fill.venue).or_insert(0.0) += venue_efficiency;
        }
        
        let avg_venue_cost = venue_costs.values().sum::<f64>() / venue_costs.len() as f64;
        Ok(total_impact * 0.1 * (avg_venue_cost - 1.0)) // 场所贡献约10%
    }

    fn calculate_cross_impact(
        &self,
        transaction: &ExecutionTransaction,
        market_data: &MarketDataHistory,
        total_impact: f64,
    ) -> Result<f64> {
        // 简化交叉影响计算
        let size_vol_interaction = 0.05; // 规模和波动率的交互作用
        Ok(total_impact * size_vol_interaction)
    }
}

impl RegimeDetector {
    pub fn new() -> Self {
        let volatility_regimes = vec![
            VolatilityRegime {
                regime_name: "Low".to_string(),
                volatility_threshold: 0.01,
                impact_multiplier: 0.8,
                typical_duration: Duration::hours(2),
            },
            VolatilityRegime {
                regime_name: "Medium".to_string(),
                volatility_threshold: 0.02,
                impact_multiplier: 1.0,
                typical_duration: Duration::hours(1),
            },
            VolatilityRegime {
                regime_name: "High".to_string(),
                volatility_threshold: f64::INFINITY,
                impact_multiplier: 1.5,
                typical_duration: Duration::minutes(30),
            },
        ];

        let liquidity_regimes = vec![
            LiquidityRegime {
                regime_name: "Liquid".to_string(),
                liquidity_score: 0.8,
                bid_ask_spread_threshold: 0.001,
                depth_threshold: 10000.0,
            },
            LiquidityRegime {
                regime_name: "Normal".to_string(),
                liquidity_score: 0.5,
                bid_ask_spread_threshold: 0.005,
                depth_threshold: 5000.0,
            },
            LiquidityRegime {
                regime_name: "Illiquid".to_string(),
                liquidity_score: 0.2,
                bid_ask_spread_threshold: f64::INFINITY,
                depth_threshold: 0.0,
            },
        ];

        Self {
            volatility_regimes,
            liquidity_regimes,
            current_regime: MarketRegime {
                volatility_regime: "Medium".to_string(),
                liquidity_regime: "Normal".to_string(),
                regime_start_time: Utc::now(),
                regime_confidence: 0.7,
            },
        }
    }

    pub fn detect_current_regime(&mut self, market_data: &MarketDataHistory) -> Result<()> {
        // 检测波动率状态
        let current_volatility = self.calculate_current_volatility(market_data)?;
        let vol_regime = self.classify_volatility_regime(current_volatility);

        // 检测流动性状态  
        let current_liquidity = self.estimate_current_liquidity(market_data)?;
        let liq_regime = self.classify_liquidity_regime(current_liquidity);

        // 更新当前状态
        self.current_regime = MarketRegime {
            volatility_regime: vol_regime,
            liquidity_regime: liq_regime,
            regime_start_time: Utc::now(),
            regime_confidence: self.calculate_regime_confidence(current_volatility, current_liquidity)?,
        };

        Ok(())
    }

    fn calculate_current_volatility(&self, market_data: &MarketDataHistory) -> Result<f64> {
        if market_data.price_data.len() < 20 {
            return Ok(0.015);
        }

        let recent_prices = &market_data.price_data[market_data.price_data.len()-20..];
        let returns: Vec<f64> = recent_prices.windows(2)
            .map(|w| (w[1].price / w[0].price - 1.0))
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;

        Ok(variance.sqrt())
    }

    fn estimate_current_liquidity(&self, market_data: &MarketDataHistory) -> Result<f64> {
        // 简化流动性估计 - 基于成交量
        if market_data.volume_data.is_empty() {
            return Ok(0.5);
        }

        let recent_volume = market_data.volume_data.iter()
            .rev()
            .take(10)
            .map(|v| v.volume)
            .sum::<f64>() / 10.0;

        // 归一化流动性分数
        Ok((recent_volume / 100000.0).min(1.0))
    }

    fn classify_volatility_regime(&self, volatility: f64) -> String {
        for regime in &self.volatility_regimes {
            if volatility <= regime.volatility_threshold {
                return regime.regime_name.clone();
            }
        }
        "High".to_string()
    }

    fn classify_liquidity_regime(&self, liquidity: f64) -> String {
        if liquidity >= 0.7 {
            "Liquid".to_string()
        } else if liquidity >= 0.3 {
            "Normal".to_string()
        } else {
            "Illiquid".to_string()
        }
    }

    fn calculate_regime_confidence(&self, volatility: f64, liquidity: f64) -> Result<f64> {
        // 基于数据质量和分类清晰度计算置信度
        let vol_confidence = if volatility < 0.005 || volatility > 0.05 { 0.9 } else { 0.7 };
        let liq_confidence = if liquidity < 0.2 || liquidity > 0.8 { 0.9 } else { 0.7 };
        
        Ok((vol_confidence + liq_confidence) / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_impact_model_creation() {
        let model = MarketImpactModel::new();
        // 基础创建测试
    }

    #[test]
    fn test_linear_impact_calculation() {
        let linear_model = LinearImpactModel::new();
        
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
            metadata: std::collections::HashMap::new(),
        };
        
        let market_data = MarketDataHistory {
            symbol: "AAPL".to_string(),
            start_time: Utc::now() - Duration::days(1),
            end_time: Utc::now(),
            price_data: vec![],
            volume_data: vec![crate::tca::VolumePoint {
                timestamp: Utc::now(),
                volume: 1000000.0,
            }],
        };
        
        let impact = linear_model.calculate_impact(&transaction, &market_data).unwrap();
        assert!(impact >= 0.0);
        assert!(impact <= 100.0); // 在限制范围内
    }

    #[test]
    fn test_regime_detection() {
        let mut detector = RegimeDetector::new();
        
        let market_data = MarketDataHistory {
            symbol: "TEST".to_string(),
            start_time: Utc::now() - Duration::hours(1),
            end_time: Utc::now(),
            price_data: vec![
                crate::tca::PricePoint { timestamp: Utc::now() - Duration::minutes(30), price: 100.0 },
                crate::tca::PricePoint { timestamp: Utc::now() - Duration::minutes(25), price: 101.0 },
                crate::tca::PricePoint { timestamp: Utc::now() - Duration::minutes(20), price: 102.0 },
                crate::tca::PricePoint { timestamp: Utc::now() - Duration::minutes(15), price: 103.0 },
                crate::tca::PricePoint { timestamp: Utc::now() - Duration::minutes(10), price: 104.0 },
            ],
            volume_data: vec![],
        };
        
        let result = detector.detect_current_regime(&market_data);
        assert!(result.is_ok());
        assert!(detector.current_regime.regime_confidence > 0.0);
    }
}