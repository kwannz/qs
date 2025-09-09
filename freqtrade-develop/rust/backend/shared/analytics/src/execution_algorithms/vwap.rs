use super::*;
use super::market_impact::{AlmgrenChrissModel, MarketImpactResult, ImpactExecutionStrategy};
use anyhow::{Result, Context};
use chrono::{Duration, Utc};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// VWAP (Volume Weighted Average Price) 执行算法
/// 
/// 该算法旨在匹配历史成交量分布模式来执行大额订单，
/// 以实现接近市场VWAP的平均成交价格
#[derive(Debug)]
pub struct VwapAlgorithm {
    config: VwapConfig,
    statistics: AlgorithmStatistics,
    volume_profile: Option<Vec<VolumeProfileBucket>>,
    adaptive_params: AdaptiveParameters,
    market_impact_model: AlmgrenChrissModel,
}

/// VWAP算法配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VwapConfig {
    pub lookback_days: u32,                    // 历史数据回望天数
    pub min_participation_rate: f64,           // 最小参与率
    pub max_participation_rate: f64,           // 最大参与率
    pub target_participation_rate: f64,        // 目标参与率
    pub participation_rate_tolerance: f64,     // 参与率容忍度
    pub price_improvement_threshold_bps: f64,  // 价格改善阈值
    pub market_impact_control: f64,            // 市场冲击控制系数
    pub volume_forecast_confidence: f64,       // 成交量预测置信度
    pub adaptive_adjustment_factor: f64,       // 自适应调整因子
    pub slice_duration_minutes: u32,          // 子单时间间隔(分钟)
    pub min_order_size: f64,                  // 最小订单大小
    pub max_order_size_adv_ratio: f64,        // 最大订单占ADV比例
    pub enable_volume_forecasting: bool,       // 启用成交量预测
    pub enable_intraday_adaptation: bool,      // 启用日内自适应
    pub enable_cross_venue_optimization: bool, // 启用跨场所优化
}

/// 自适应参数
#[derive(Debug, Clone)]
struct AdaptiveParameters {
    current_participation_rate: f64,
    volume_adjustment_factor: f64,
    urgency_multiplier: f64,
    liquidity_penalty: f64,
    momentum_adjustment: f64,
    last_adjustment_time: DateTime<Utc>,
}

/// 成交量预测器
#[derive(Debug)]
struct VolumeForecaster {
    historical_profiles: Vec<DailyVolumeProfile>,
    intraday_pattern: Vec<f64>,
    adjustment_factors: HashMap<String, f64>,
}

/// 日成交量分布
#[derive(Debug, Clone)]
struct DailyVolumeProfile {
    date: DateTime<Utc>,
    total_volume: f64,
    intraday_buckets: Vec<VolumeProfileBucket>,
    volatility_adjusted: bool,
    regime_classification: String,
}

/// VWAP执行计划
#[derive(Debug, Clone)]
struct VwapExecutionPlan {
    parent_order_id: String,
    total_slices: u32,
    slice_schedules: Vec<SliceSchedule>,
    expected_vwap: f64,
    risk_budget: f64,
    contingency_plan: Option<ContingencyPlan>,
}

/// 子单计划
#[derive(Debug, Clone)]
struct SliceSchedule {
    slice_id: u32,
    scheduled_time: DateTime<Utc>,
    target_quantity: f64,
    expected_volume: f64,
    participation_rate: f64,
    venue_allocation: HashMap<String, f64>,
    price_constraints: PriceConstraints,
}

/// 价格约束
#[derive(Debug, Clone)]
struct PriceConstraints {
    limit_price: Option<f64>,
    max_price_deviation_bps: f64,
    stop_loss_level: Option<f64>,
    improvement_target_bps: f64,
}

/// 应急计划
#[derive(Debug, Clone)]
enum ContingencyPlan {
    AcceleratedExecution { factor: f64 },
    MarketOrderFallback { threshold_bps: f64 },
    PostponeExecution { delay_minutes: u32 },
    CancelRemaining,
}

impl VwapAlgorithm {
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: VwapConfig::default(),
            statistics: AlgorithmStatistics {
                algorithm_name: "VWAP".to_string(),
                ..Default::default()
            },
            volume_profile: None,
            adaptive_params: AdaptiveParameters::default(),
            market_impact_model: AlmgrenChrissModel::new()?,
        })
    }
    
    pub fn with_config(config: VwapConfig) -> Result<Self> {
        let mut instance = Self::new()?;
        instance.config = config;
        Ok(instance)
    }
    
    pub fn with_market_impact_model(mut self, model: AlmgrenChrissModel) -> Self {
        self.market_impact_model = model;
        self
    }
    
    /// 创建VWAP执行计划
    fn create_execution_plan(
        &self,
        parent_order: &ParentOrder,
        market_conditions: &MarketConditions,
        execution_params: &ExecutionParams,
    ) -> Result<VwapExecutionPlan> {
        info!("Creating VWAP execution plan for order {}", parent_order.id);
        
        // 预测成交量分布
        let volume_forecast = self.forecast_volume_profile(
            &parent_order.symbol,
            parent_order.time_horizon,
            market_conditions,
        )?;
        
        // 计算子单分割
        let slice_schedules = self.calculate_slice_schedule(
            parent_order,
            &volume_forecast,
            market_conditions,
            execution_params,
        )?;
        
        // 估算期望VWAP
        let expected_vwap = self.estimate_expected_vwap(
            &volume_forecast,
            market_conditions,
            parent_order.side == OrderSide::Buy,
        )?;
        
        // 计算风险预算
        let risk_budget = self.calculate_risk_budget(
            parent_order,
            market_conditions,
            execution_params,
        )?;
        
        let plan = VwapExecutionPlan {
            parent_order_id: parent_order.id.clone(),
            total_slices: slice_schedules.len() as u32,
            slice_schedules,
            expected_vwap,
            risk_budget,
            contingency_plan: self.create_contingency_plan(parent_order, execution_params)?,
        };
        
        debug!("VWAP execution plan created with {} slices, expected VWAP: {:.4}", 
               plan.total_slices, plan.expected_vwap);
        
        Ok(plan)
    }
    
    /// 预测成交量分布
    fn forecast_volume_profile(
        &self,
        symbol: &str,
        time_horizon: i64,
        market_conditions: &MarketConditions,
    ) -> Result<Vec<VolumeProfileBucket>> {
        if !self.config.enable_volume_forecasting {
            return Ok(market_conditions.volume_profile.clone());
        }
        
        // 基于历史模式预测
        let base_profile = self.get_base_volume_pattern(symbol)?;
        
        // 应用实时调整
        let mut adjusted_profile = self.apply_realtime_adjustments(
            base_profile,
            market_conditions,
            time_horizon,
        )?;
        
        // 应用市场冲击模型优化
        self.apply_market_impact_optimization(
            &mut adjusted_profile,
            symbol,
            market_conditions,
        )?;
        
        debug!("Volume forecast completed for {} buckets with market impact optimization", adjusted_profile.len());
        Ok(adjusted_profile)
    }
    
    /// 获取基础成交量模式
    fn get_base_volume_pattern(&self, symbol: &str) -> Result<Vec<VolumeProfileBucket>> {
        // 简化实现 - 实际中会从数据库获取历史数据
        let typical_pattern = vec![
            VolumeProfileBucket { time_bucket: 0, volume: 1000.0, vwap: 100.0, participation_rate: 0.15 },   // 开盘
            VolumeProfileBucket { time_bucket: 30, volume: 800.0, vwap: 100.1, participation_rate: 0.12 },   // 开盘后30分钟
            VolumeProfileBucket { time_bucket: 60, volume: 600.0, vwap: 100.2, participation_rate: 0.10 },   // 1小时
            VolumeProfileBucket { time_bucket: 120, volume: 500.0, vwap: 100.3, participation_rate: 0.08 },  // 2小时
            VolumeProfileBucket { time_bucket: 180, volume: 400.0, vwap: 100.4, participation_rate: 0.06 },  // 3小时
            VolumeProfileBucket { time_bucket: 240, volume: 500.0, vwap: 100.5, participation_rate: 0.08 },  // 4小时
            VolumeProfileBucket { time_bucket: 300, volume: 600.0, vwap: 100.6, participation_rate: 0.10 },  // 5小时
            VolumeProfileBucket { time_bucket: 360, volume: 1200.0, vwap: 100.7, participation_rate: 0.20 }, // 收盘前
        ];
        
        Ok(typical_pattern)
    }
    
    /// 应用实时调整
    fn apply_realtime_adjustments(
        &self,
        mut base_profile: Vec<VolumeProfileBucket>,
        market_conditions: &MarketConditions,
        time_horizon: i64,
    ) -> Result<Vec<VolumeProfileBucket>> {
        let vol_adjustment = market_conditions.realized_volatility / 0.2; // 基准波动率
        let momentum_adjustment = 1.0 + market_conditions.price_momentum * 0.1;
        
        // 基于市场制度的调整
        let regime_adjustment = match market_conditions.intraday_period {
            IntradayPeriod::PreMarket => 0.6,
            IntradayPeriod::OpeningAuction => 1.5,
            IntradayPeriod::MorningSession => 1.2,
            IntradayPeriod::MiddayLull => 0.8,
            IntradayPeriod::AfternoonSession => 1.1,
            IntradayPeriod::ClosingAuction => 1.8,
            IntradayPeriod::AfterHours => 0.4,
        };
        
        for bucket in &mut base_profile {
            // 调整预期成交量
            bucket.volume *= vol_adjustment * momentum_adjustment * regime_adjustment;
            
            // 调整参与率
            bucket.participation_rate = (bucket.participation_rate * vol_adjustment)
                .min(self.config.max_participation_rate)
                .max(self.config.min_participation_rate);
                
            // 基于流动性冲击的VWAP调整
            if market_conditions.toxic_flow_indicator > 0.7 {
                bucket.participation_rate *= 0.7; // 降低参与率避免逆向选择
            }
        }
        
        Ok(base_profile)
    }
    
    /// 应用市场冲击模型优化
    fn apply_market_impact_optimization(
        &self,
        profile: &mut Vec<VolumeProfileBucket>,
        symbol: &str,
        market_conditions: &MarketConditions,
    ) -> Result<()> {
        // 创建模拟订单来计算市场冲击
        let test_order = ParentOrder {
            id: "test_vwap_impact".to_string(),
            symbol: symbol.to_string(),
            side: OrderSide::Buy, // 默认买单测试
            total_quantity: 10000.0, // 测试数量
            limit_price: Some(market_conditions.mid_price),
            time_horizon: 3600, // 1小时
            urgency: 0.5,
            ..Default::default()
        };
        
        // 使用VWAP策略计算市场冲击
        let impact_result = self.market_impact_model.calculate_impact(
            &test_order,
            market_conditions,
            &ImpactExecutionStrategy::Vwap,
        )?;
        
        debug!("Market impact analysis: permanent={:.4}bps, temporary={:.4}bps", 
               impact_result.permanent_impact_bps, impact_result.temporary_impact_bps);
        
        // 根据市场冲击调整成交量分布
        let total_impact_bps = impact_result.permanent_impact_bps + impact_result.temporary_impact_bps;
        let impact_penalty = if total_impact_bps > 10.0 { 0.8 } else { 1.0 - (total_impact_bps / 100.0) };
        
        for (i, bucket) in profile.iter_mut().enumerate() {
            // 基于时间线上的冲击分布调整参与率
            if let Some(timeline_impact) = impact_result.impact_timeline.get(i) {
                let time_specific_penalty = 1.0 - (timeline_impact / 50.0).min(0.3);
                bucket.participation_rate *= time_specific_penalty * impact_penalty;
            } else {
                bucket.participation_rate *= impact_penalty;
            }
            
            // 确保参与率在合理范围内
            bucket.participation_rate = bucket.participation_rate
                .min(self.config.max_participation_rate)
                .max(self.config.min_participation_rate);
        }
        
        Ok(())
    }
    
    /// 计算子单计划
    fn calculate_slice_schedule(
        &self,
        parent_order: &ParentOrder,
        volume_forecast: &[VolumeProfileBucket],
        market_conditions: &MarketConditions,
        execution_params: &ExecutionParams,
    ) -> Result<Vec<SliceSchedule>> {
        let mut schedules = Vec::new();
        let total_quantity = parent_order.total_quantity;
        let start_time = Utc::now();
        
        // 计算市场冲击优化的数量分配
        let impact_optimized_allocation = self.calculate_impact_optimized_allocation(
            parent_order,
            volume_forecast,
            market_conditions,
        )?;
        
        let mut allocated_quantity = 0.0;
        
        for (i, (bucket, optimal_quantity)) in volume_forecast.iter().zip(impact_optimized_allocation.iter()).enumerate() {
            let final_quantity = if i == volume_forecast.len() - 1 {
                total_quantity - allocated_quantity
            } else {
                *optimal_quantity
            };
            
            if final_quantity <= 0.0 {
                break;
            }
            
            let scheduled_time = start_time + Duration::minutes(bucket.time_bucket as i64);
            
            // 计算该slice的市场冲击调整参与率
            let impact_adjusted_participation = self.calculate_impact_adjusted_participation(
                parent_order,
                final_quantity,
                bucket,
                market_conditions,
                execution_params,
            )?;
            
            let schedule = SliceSchedule {
                slice_id: i as u32,
                scheduled_time,
                target_quantity: final_quantity,
                expected_volume: bucket.volume,
                participation_rate: impact_adjusted_participation,
                venue_allocation: self.calculate_venue_allocation(market_conditions, execution_params)?,
                price_constraints: self.calculate_impact_aware_price_constraints(
                    parent_order, 
                    final_quantity, 
                    market_conditions
                )?,
            };
            
            schedules.push(schedule);
            allocated_quantity += final_quantity;
        }
        
        info!("Created {} slice schedules with market impact optimization, total quantity: {}", 
              schedules.len(), total_quantity);
        Ok(schedules)
    }
    
    /// 计算市场冲击优化的数量分配
    fn calculate_impact_optimized_allocation(
        &self,
        parent_order: &ParentOrder,
        volume_forecast: &[VolumeProfileBucket],
        market_conditions: &MarketConditions,
    ) -> Result<Vec<f64>> {
        let total_quantity = parent_order.total_quantity;
        let total_forecast_volume: f64 = volume_forecast.iter().map(|b| b.volume).sum();
        
        let mut allocations = Vec::new();
        
        for bucket in volume_forecast {
            // 基础按成交量比例分配
            let volume_ratio = bucket.volume / total_forecast_volume;
            let base_quantity = total_quantity * volume_ratio;
            
            // 创建slice订单来计算市场冲击
            let slice_order = ParentOrder {
                id: format!("{}_slice_test", parent_order.id),
                symbol: parent_order.symbol.clone(),
                side: parent_order.side,
                total_quantity: base_quantity,
                limit_price: parent_order.limit_price,
                time_horizon: (self.config.slice_duration_minutes * 60) as i64,
                urgency: parent_order.urgency,
                ..Default::default()
            };
            
            // 计算该slice的市场冲击
            let impact = self.market_impact_model.calculate_impact(
                &slice_order,
                market_conditions,
                &ImpactExecutionStrategy::Vwap,
            )?;
            
            // 根据冲击大小调整数量
            let total_impact_bps = impact.permanent_impact_bps + impact.temporary_impact_bps;
            let impact_adjustment = if total_impact_bps > 15.0 {
                0.7 // 高冲击时减少数量
            } else if total_impact_bps > 5.0 {
                0.85 // 中等冲击时稍微减少
            } else {
                1.0 // 低冲击时保持原数量
            };
            
            let optimized_quantity = (base_quantity * impact_adjustment)
                .max(self.config.min_order_size);
            
            allocations.push(optimized_quantity);
        }
        
        // 规范化以确保总数量不变
        let total_allocated: f64 = allocations.iter().sum();
        let normalization_factor = total_quantity / total_allocated;
        
        for allocation in &mut allocations {
            *allocation *= normalization_factor;
        }
        
        Ok(allocations)
    }
    
    /// 计算市场冲击调整的参与率
    fn calculate_impact_adjusted_participation(
        &self,
        parent_order: &ParentOrder,
        slice_quantity: f64,
        bucket: &VolumeProfileBucket,
        market_conditions: &MarketConditions,
        execution_params: &ExecutionParams,
    ) -> Result<f64> {
        // 首先计算基础参与率
        let base_participation = self.calculate_adaptive_participation_rate(
            bucket.participation_rate,
            market_conditions,
            execution_params,
        )?;
        
        // 计算该slice的相对市场冲击
        let slice_order = ParentOrder {
            id: format!("{}_slice_impact", parent_order.id),
            symbol: parent_order.symbol.clone(),
            side: parent_order.side,
            total_quantity: slice_quantity,
            limit_price: parent_order.limit_price,
            time_horizon: (self.config.slice_duration_minutes * 60) as i64,
            urgency: parent_order.urgency,
            ..Default::default()
        };
        
        let impact = self.market_impact_model.calculate_impact(
            &slice_order,
            market_conditions,
            &ImpactExecutionStrategy::Vwap,
        )?;
        
        // 根据市场冲击调整参与率
        let total_impact_bps = impact.permanent_impact_bps + impact.temporary_impact_bps;
        let impact_adjustment = match total_impact_bps {
            x if x > 20.0 => 0.6, // 极高冲击，大幅降低参与率
            x if x > 10.0 => 0.75, // 高冲击，明显降低参与率
            x if x > 5.0 => 0.9,   // 中等冲击，轻微降低参与率
            _ => 1.0,              // 低冲击，保持原参与率
        };
        
        let adjusted_participation = (base_participation * impact_adjustment)
            .min(self.config.max_participation_rate)
            .max(self.config.min_participation_rate);
        
        debug!("Impact-adjusted participation: base={:.3}, impact={:.1}bps, adjusted={:.3}", 
               base_participation, total_impact_bps, adjusted_participation);
        
        Ok(adjusted_participation)
    }
    
    /// 计算市场冲击感知的价格约束
    fn calculate_impact_aware_price_constraints(
        &self,
        parent_order: &ParentOrder,
        slice_quantity: f64,
        market_conditions: &MarketConditions,
    ) -> Result<PriceConstraints> {
        // 计算该slice的预期市场冲击
        let slice_order = ParentOrder {
            id: format!("{}_price_impact", parent_order.id),
            symbol: parent_order.symbol.clone(),
            side: parent_order.side,
            total_quantity: slice_quantity,
            limit_price: parent_order.limit_price,
            time_horizon: (self.config.slice_duration_minutes * 60) as i64,
            urgency: parent_order.urgency,
            ..Default::default()
        };
        
        let impact = self.market_impact_model.calculate_impact(
            &slice_order,
            market_conditions,
            &ImpactExecutionStrategy::Vwap,
        )?;
        
        // 基于市场冲击调整价格约束
        let impact_buffer_bps = (impact.permanent_impact_bps + impact.temporary_impact_bps) * 1.2;
        let max_deviation_bps = self.config.price_improvement_threshold_bps * 2.0 + impact_buffer_bps;
        
        // 调整限价以考虑市场冲击
        let adjusted_limit_price = if let Some(limit_price) = parent_order.limit_price {
            let impact_adjustment = match parent_order.side {
                OrderSide::Buy => limit_price * (1.0 + impact_buffer_bps / 10000.0),
                OrderSide::Sell => limit_price * (1.0 - impact_buffer_bps / 10000.0),
            };
            Some(impact_adjustment)
        } else {
            parent_order.limit_price
        };
        
        Ok(PriceConstraints {
            limit_price: adjusted_limit_price,
            max_price_deviation_bps: max_deviation_bps,
            stop_loss_level: None, // VWAP通常不设置止损
            improvement_target_bps: self.config.price_improvement_threshold_bps,
        })
    }
    
    /// 计算目标参与率
    fn calculate_target_participation_rate(
        &self,
        base_rate: f64,
        market_conditions: &MarketConditions,
        execution_params: &ExecutionParams,
    ) -> Result<f64> {
        self.calculate_adaptive_participation_rate(base_rate, market_conditions, execution_params)
    }
    
    /// 计算自适应参与率
    fn calculate_adaptive_participation_rate(
        &self,
        base_rate: f64,
        market_conditions: &MarketConditions,
        execution_params: &ExecutionParams,
    ) -> Result<f64> {
        let mut adjusted_rate = base_rate;
        
        // 基于流动性调整
        let liquidity_factor = (market_conditions.current_volume / market_conditions.average_daily_volume)
            .min(2.0).max(0.5);
        adjusted_rate *= liquidity_factor;
        
        // 基于价差调整
        let spread_factor = if market_conditions.spread_bps > 10.0 {
            0.8 // 价差较大时降低参与率
        } else {
            1.0
        };
        adjusted_rate *= spread_factor;
        
        // 基于波动性调整  
        let volatility_factor = if market_conditions.realized_volatility > 0.3 {
            0.7 // 高波动时降低参与率
        } else {
            1.0
        };
        adjusted_rate *= volatility_factor;
        
        // 应用执行参数约束
        adjusted_rate = adjusted_rate
            .min(execution_params.max_participation_rate)
            .max(self.config.min_participation_rate);
        
        Ok(adjusted_rate)
    }
    
    /// 计算场所分配
    fn calculate_venue_allocation(
        &self,
        market_conditions: &MarketConditions,
        execution_params: &ExecutionParams,
    ) -> Result<HashMap<String, f64>> {
        let mut allocation = HashMap::new();
        
        if !execution_params.venue_preferences.is_empty() {
            // 使用配置的场所偏好
            let total_weight: f64 = execution_params.venue_preferences.values().sum();
            for (venue, weight) in &execution_params.venue_preferences {
                allocation.insert(venue.clone(), weight / total_weight);
            }
        } else {
            // 默认分配策略
            allocation.insert("PRIMARY".to_string(), 0.6);
            allocation.insert("DARK_POOL_1".to_string(), 0.25);
            allocation.insert("DARK_POOL_2".to_string(), 0.15);
        }
        
        Ok(allocation)
    }
    
    /// 计算价格约束
    fn calculate_price_constraints(
        &self,
        parent_order: &ParentOrder,
        market_conditions: &MarketConditions,
    ) -> Result<PriceConstraints> {
        let max_deviation_bps = self.config.price_improvement_threshold_bps * 2.0;
        
        Ok(PriceConstraints {
            limit_price: parent_order.limit_price,
            max_price_deviation_bps: max_deviation_bps,
            stop_loss_level: None, // VWAP通常不设置止损
            improvement_target_bps: self.config.price_improvement_threshold_bps,
        })
    }
    
    /// 估算期望VWAP（含市场冲击估算）
    fn estimate_expected_vwap(
        &self,
        volume_forecast: &[VolumeProfileBucket],
        market_conditions: &MarketConditions,
        is_buy: bool,
    ) -> Result<f64> {
        let total_volume: f64 = volume_forecast.iter().map(|b| b.volume).sum();
        let weighted_price_sum: f64 = volume_forecast.iter()
            .map(|b| b.vwap * b.volume)
            .sum();
        
        let base_vwap = if total_volume > 0.0 {
            weighted_price_sum / total_volume
        } else {
            market_conditions.mid_price
        };
        
        // 使用Almgren-Chriss模型估算市场冲击
        let test_order = ParentOrder {
            id: "vwap_impact_estimate".to_string(),
            symbol: market_conditions.symbol.clone(),
            side: if is_buy { OrderSide::Buy } else { OrderSide::Sell },
            total_quantity: total_volume * 0.1, // 使用10%的成交量作为估算基准
            limit_price: Some(base_vwap),
            time_horizon: 3600, // 1小时默认时间窗口
            urgency: 0.5,
            ..Default::default()
        };
        
        let impact_result = self.market_impact_model.calculate_impact(
            &test_order,
            market_conditions,
            &ImpactExecutionStrategy::Vwap,
        )?;
        
        // 计算总市场冲击调整
        let total_impact_bps = impact_result.permanent_impact_bps + 
                               impact_result.temporary_impact_bps * 0.5; // 临时冲击部分恢复
        
        let impact_adjustment = if is_buy {
            total_impact_bps / 10000.0 // 买单向上调整
        } else {
            -total_impact_bps / 10000.0 // 卖单向下调整
        };
        
        let expected_vwap = base_vwap * (1.0 + impact_adjustment);
        
        debug!("Expected VWAP: base={:.4}, impact={:.1}bps, final={:.4}", 
               base_vwap, total_impact_bps, expected_vwap);
        
        Ok(expected_vwap)
    }
    
    /// 计算风险预算
    fn calculate_risk_budget(
        &self,
        parent_order: &ParentOrder,
        market_conditions: &MarketConditions,
        execution_params: &ExecutionParams,
    ) -> Result<f64> {
        let base_risk = parent_order.total_quantity * market_conditions.mid_price * 0.001; // 0.1%基础风险
        let volatility_multiplier = market_conditions.realized_volatility / 0.2;
        let urgency_multiplier = 1.0 + parent_order.urgency * 0.5;
        
        Ok(base_risk * volatility_multiplier * urgency_multiplier)
    }
    
    /// 创建应急计划
    fn create_contingency_plan(
        &self,
        parent_order: &ParentOrder,
        execution_params: &ExecutionParams,
    ) -> Result<Option<ContingencyPlan>> {
        if parent_order.urgency > 0.8 {
            Ok(Some(ContingencyPlan::AcceleratedExecution { factor: 1.5 }))
        } else if execution_params.max_market_impact_bps < 5.0 {
            Ok(Some(ContingencyPlan::PostponeExecution { delay_minutes: 15 }))
        } else {
            Ok(None)
        }
    }
}

impl ExecutionAlgorithm for VwapAlgorithm {
    fn name(&self) -> &str {
        "VWAP"
    }
    
    fn calculate_child_orders(
        &self,
        parent_order: &ParentOrder,
        market_conditions: &MarketConditions,
        execution_params: &ExecutionParams,
    ) -> Result<Vec<ChildOrder>> {
        let execution_plan = self.create_execution_plan(parent_order, market_conditions, execution_params)?;
        let mut child_orders = Vec::new();
        
        for schedule in execution_plan.slice_schedules {
            for (venue, allocation) in schedule.venue_allocation {
                let quantity = schedule.target_quantity * allocation;
                
                if quantity >= self.config.min_order_size {
                    let child_order = ChildOrder {
                        id: format!("{}_{}_{}_{}", parent_order.id, schedule.slice_id, venue, child_orders.len()),
                        parent_id: parent_order.id.clone(),
                        sequence_number: child_orders.len() as u32,
                        quantity,
                        price: schedule.price_constraints.limit_price,
                        venue,
                        order_type: if schedule.price_constraints.limit_price.is_some() {
                            OrderType::Limit
                        } else {
                            OrderType::Market
                        },
                        time_in_force: TimeInForce::GoodTillTime(
                            schedule.scheduled_time + Duration::minutes(self.config.slice_duration_minutes as i64)
                        ),
                        scheduled_time: schedule.scheduled_time,
                        execution_window: self.config.slice_duration_minutes as i64 * 60,
                        is_hidden: execution_params.hidden_order_ratio > 0.5,
                        display_quantity: if execution_params.iceberg_size_ratio > 0.0 {
                            Some(quantity * execution_params.iceberg_size_ratio)
                        } else {
                            None
                        },
                        post_only: false, // VWAP通常需要立即执行
                        reduce_only: false,
                    };
                    
                    child_orders.push(child_order);
                }
            }
        }
        
        info!("VWAP algorithm generated {} child orders", child_orders.len());
        Ok(child_orders)
    }
    
    fn adapt_parameters(
        &mut self,
        execution_state: &ExecutionState,
        market_update: &MarketUpdate,
    ) -> Result<()> {
        if !self.config.enable_intraday_adaptation {
            return Ok(());
        }
        
        // 计算执行进度
        let completion_ratio = execution_state.filled_quantity / execution_state.total_quantity;
        let time_ratio = execution_state.elapsed_time as f64 / 
                        (execution_state.elapsed_time + execution_state.remaining_time) as f64;
        
        // 检查是否需要调整参与率
        if completion_ratio < time_ratio - 0.1 {
            // 执行落后，增加参与率
            self.adaptive_params.current_participation_rate *= 1.2;
            self.adaptive_params.urgency_multiplier *= 1.1;
            warn!("VWAP execution falling behind, increasing participation rate to {:.3}", 
                  self.adaptive_params.current_participation_rate);
        } else if completion_ratio > time_ratio + 0.1 {
            // 执行过快，降低参与率
            self.adaptive_params.current_participation_rate *= 0.9;
            self.adaptive_params.urgency_multiplier *= 0.95;
            debug!("VWAP execution ahead of schedule, reducing participation rate to {:.3}", 
                   self.adaptive_params.current_participation_rate);
        }
        
        // 根据市场更新调整
        match market_update.update_type {
            MarketUpdateType::VolumeSpike => {
                self.adaptive_params.volume_adjustment_factor *= 1.1;
            },
            MarketUpdateType::VolatilityChange => {
                if market_update.volatility_change_ratio > 1.2 {
                    self.adaptive_params.current_participation_rate *= 0.8;
                }
            },
            MarketUpdateType::LiquidityShock => {
                self.adaptive_params.liquidity_penalty += 0.1;
                self.adaptive_params.current_participation_rate *= 0.7;
            },
            _ => {}
        }
        
        // 应用边界约束
        self.adaptive_params.current_participation_rate = self.adaptive_params.current_participation_rate
            .min(self.config.max_participation_rate)
            .max(self.config.min_participation_rate);
        
        self.adaptive_params.last_adjustment_time = Utc::now();
        
        Ok(())
    }
    
    fn get_statistics(&self) -> AlgorithmStatistics {
        self.statistics.clone()
    }
    
    fn validate_parameters(&self, params: &HashMap<String, f64>) -> Result<()> {
        // 验证关键参数
        if let Some(&participation_rate) = params.get("participation_rate") {
            if participation_rate <= 0.0 || participation_rate > 1.0 {
                return Err(anyhow::anyhow!("Invalid participation_rate: must be between 0.0 and 1.0"));
            }
        }
        
        if let Some(&market_impact_limit) = params.get("market_impact_limit_bps") {
            if market_impact_limit < 0.0 || market_impact_limit > 100.0 {
                return Err(anyhow::anyhow!("Invalid market_impact_limit_bps: must be between 0.0 and 100.0"));
            }
        }
        
        Ok(())
    }
}

impl Default for VwapConfig {
    fn default() -> Self {
        Self {
            lookback_days: 20,
            min_participation_rate: 0.02,
            max_participation_rate: 0.30,
            target_participation_rate: 0.10,
            participation_rate_tolerance: 0.02,
            price_improvement_threshold_bps: 1.0,
            market_impact_control: 0.8,
            volume_forecast_confidence: 0.7,
            adaptive_adjustment_factor: 1.2,
            slice_duration_minutes: 15,
            min_order_size: 100.0,
            max_order_size_adv_ratio: 0.05,
            enable_volume_forecasting: true,
            enable_intraday_adaptation: true,
            enable_cross_venue_optimization: true,
        }
    }
}

impl Default for AdaptiveParameters {
    fn default() -> Self {
        Self {
            current_participation_rate: 0.10,
            volume_adjustment_factor: 1.0,
            urgency_multiplier: 1.0,
            liquidity_penalty: 0.0,
            momentum_adjustment: 1.0,
            last_adjustment_time: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vwap_algorithm_creation() {
        let algorithm = VwapAlgorithm::new().unwrap();
        assert_eq!(algorithm.name(), "VWAP");
    }
    
    #[test]
    fn test_parameter_validation() {
        let algorithm = VwapAlgorithm::new().unwrap();
        
        let mut valid_params = HashMap::new();
        valid_params.insert("participation_rate".to_string(), 0.15);
        valid_params.insert("market_impact_limit_bps".to_string(), 10.0);
        
        assert!(algorithm.validate_parameters(&valid_params).is_ok());
        
        let mut invalid_params = HashMap::new();
        invalid_params.insert("participation_rate".to_string(), 1.5);
        
        assert!(algorithm.validate_parameters(&invalid_params).is_err());
    }
    
    #[test] 
    fn test_adaptive_participation_rate() {
        let algorithm = VwapAlgorithm::new().unwrap();
        
        let market_conditions = MarketConditions {
            symbol: "BTCUSDT".to_string(),
            timestamp: Utc::now(),
            mid_price: 50000.0,
            bid_price: 49995.0,
            ask_price: 50005.0,
            spread_bps: 2.0,
            tick_size: 0.1,
            bid_size: 1000.0,
            ask_size: 1000.0,
            market_depth: MarketDepth {
                bids: vec![],
                asks: vec![],
                total_bid_volume: 5000.0,
                total_ask_volume: 5000.0,
            },
            average_daily_volume: 1000000.0,
            current_volume: 50000.0,
            volume_profile: vec![],
            realized_volatility: 0.2,
            implied_volatility: 0.25,
            price_momentum: 0.1,
            short_term_trend: 0.05,
            order_book_imbalance: 0.0,
            queue_position_estimate: 0.5,
            toxic_flow_indicator: 0.1,
            informed_trading_probability: 0.2,
            time_to_close: 3600,
            intraday_period: IntradayPeriod::MorningSession,
            is_auction_period: false,
            trading_session: TradingSession::NewYork,
        };
        
        let execution_params = ExecutionParams::default();
        
        let rate = algorithm.calculate_adaptive_participation_rate(
            0.10,
            &market_conditions,
            &execution_params,
        ).unwrap();
        
        assert!(rate > 0.0 && rate <= 1.0);
    }
}