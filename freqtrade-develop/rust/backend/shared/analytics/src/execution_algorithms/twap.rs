use super::*;
use super::market_impact::{AlmgrenChrissModel, MarketImpactResult, ImpactExecutionStrategy};
use anyhow::{Result, Context};
use chrono::{Duration, Utc};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// TWAP (Time Weighted Average Price) 执行算法
/// 
/// 该算法通过均匀的时间分片来执行大额订单，目标是最小化时间风险
/// 并通过市场影响模型优化执行时机
#[derive(Debug)]
pub struct TwapAlgorithm {
    config: TwapConfig,
    statistics: AlgorithmStatistics,
    market_impact_model: AlmgrenChrissModel,
    adaptive_params: TwapAdaptiveParams,
}

/// TWAP算法配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwapConfig {
    pub slice_duration_minutes: u32,        // 切片持续时间（分钟）
    pub min_slice_count: u32,               // 最小切片数量
    pub max_slice_count: u32,               // 最大切片数量
    pub slice_size_variance: f64,           // 切片大小方差 (0.0-1.0)
    pub time_variance_minutes: i64,         // 时间方差（分钟）
    pub market_impact_threshold_bps: f64,   // 市场冲击阈值
    pub adaptive_slicing: bool,             // 启用自适应切片
    pub volume_participation_limit: f64,    // 成交量参与限制
    pub price_improvement_target_bps: f64,  // 价格改善目标
    pub enable_aggressive_slicing: bool,    // 启用激进切片模式
    pub liquidity_detection_window: u32,    // 流动性检测窗口（分钟）
    pub cross_venue_optimization: bool,     // 跨场所优化
}

/// TWAP自适应参数
#[derive(Debug, Clone)]
struct TwapAdaptiveParams {
    current_slice_duration: u32,
    dynamic_size_adjustment: f64,
    urgency_multiplier: f64,
    market_timing_score: f64,
    execution_quality_score: f64,
    last_adjustment_time: DateTime<Utc>,
}

/// TWAP执行计划
#[derive(Debug, Clone)]
struct TwapExecutionPlan {
    parent_order_id: String,
    total_slices: u32,
    slice_schedules: Vec<TwapSliceSchedule>,
    expected_twap: f64,
    total_market_impact_bps: f64,
    execution_risk_score: f64,
}

/// TWAP切片计划
#[derive(Debug, Clone)]
struct TwapSliceSchedule {
    slice_id: u32,
    scheduled_time: DateTime<Utc>,
    target_quantity: f64,
    time_window_seconds: i64,
    market_impact_estimate_bps: f64,
    venue_allocation: HashMap<String, f64>,
    price_constraints: TwapPriceConstraints,
    urgency_level: f64,
}

/// TWAP价格约束
#[derive(Debug, Clone)]
struct TwapPriceConstraints {
    limit_price: Option<f64>,
    max_deviation_bps: f64,
    timing_penalty_bps: f64,
    impact_buffer_bps: f64,
}

impl TwapAlgorithm {
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: TwapConfig::default(),
            statistics: AlgorithmStatistics {
                algorithm_name: "TWAP".to_string(),
                ..Default::default()
            },
            market_impact_model: AlmgrenChrissModel::new()?,
            adaptive_params: TwapAdaptiveParams::default(),
        })
    }
    
    pub fn with_config(config: TwapConfig) -> Result<Self> {
        let mut instance = Self::new()?;
        instance.config = config;
        Ok(instance)
    }
    
    pub fn with_market_impact_model(mut self, model: AlmgrenChrissModel) -> Self {
        self.market_impact_model = model;
        self
    }
    
    /// 创建TWAP执行计划
    fn create_execution_plan(
        &self,
        parent_order: &ParentOrder,
        market_conditions: &MarketConditions,
        execution_params: &ExecutionParams,
    ) -> Result<TwapExecutionPlan> {
        info!("Creating TWAP execution plan for order {}", parent_order.id);
        
        // 计算优化的时间切片
        let slice_schedules = self.calculate_optimal_time_slices(
            parent_order,
            market_conditions,
            execution_params,
        )?;
        
        // 估算期望TWAP
        let expected_twap = self.estimate_expected_twap(
            &slice_schedules,
            market_conditions,
            parent_order.side == OrderSide::Buy,
        )?;
        
        // 计算总市场冲击
        let total_impact = self.calculate_total_market_impact(
            parent_order,
            &slice_schedules,
            market_conditions,
        )?;
        
        // 评估执行风险
        let execution_risk = self.assess_execution_risk(
            parent_order,
            &slice_schedules,
            market_conditions,
        )?;
        
        let plan = TwapExecutionPlan {
            parent_order_id: parent_order.id.clone(),
            total_slices: slice_schedules.len() as u32,
            slice_schedules,
            expected_twap,
            total_market_impact_bps: total_impact.permanent_impact_bps + total_impact.temporary_impact_bps,
            execution_risk_score: execution_risk,
        };
        
        debug!("TWAP execution plan created: {} slices, expected TWAP: {:.4}, total impact: {:.1}bps", 
               plan.total_slices, plan.expected_twap, plan.total_market_impact_bps);
        
        Ok(plan)
    }
    
    /// 计算优化的时间切片
    fn calculate_optimal_time_slices(
        &self,
        parent_order: &ParentOrder,
        market_conditions: &MarketConditions,
        execution_params: &ExecutionParams,
    ) -> Result<Vec<TwapSliceSchedule>> {
        let total_time = parent_order.time_horizon;
        let total_quantity = parent_order.total_quantity;
        
        // 计算最优切片数量
        let optimal_slice_count = self.calculate_optimal_slice_count(
            parent_order,
            market_conditions,
            execution_params,
        )?;
        
        let mut slice_schedules = Vec::new();
        let start_time = Utc::now();
        
        // 基础时间间隔
        let base_interval = total_time / optimal_slice_count as i64;
        let base_quantity = total_quantity / optimal_slice_count as f64;
        
        for i in 0..optimal_slice_count {
            // 应用时间随机化以避免可预测性
            let time_variance = if self.config.time_variance_minutes > 0 {
                let variance_range = self.config.time_variance_minutes as f64;
                (fastrand::f64() - 0.5) * variance_range * 60.0 // 转换为秒
            } else {
                0.0
            };
            
            let scheduled_time = start_time + Duration::seconds(
                i as i64 * base_interval + time_variance as i64
            );
            
            // 应用数量随机化
            let quantity_variance = if self.config.slice_size_variance > 0.0 {
                let variance = (fastrand::f64() - 0.5) * self.config.slice_size_variance;
                base_quantity * (1.0 + variance)
            } else {
                base_quantity
            };
            
            // 确保最后一个切片包含剩余数量
            let target_quantity = if i == optimal_slice_count - 1 {
                total_quantity - slice_schedules.iter().map(|s| s.target_quantity).sum::<f64>()
            } else {
                quantity_variance.max(100.0) // 最小订单量
            };
            
            if target_quantity <= 0.0 {
                continue;
            }
            
            // 为该切片计算市场冲击
            let slice_impact = self.calculate_slice_market_impact(
                parent_order,
                target_quantity,
                &scheduled_time,
                market_conditions,
            )?;
            
            // 计算该切片的紧急度
            let urgency_level = self.calculate_slice_urgency(
                i,
                optimal_slice_count,
                parent_order.urgency,
                &slice_impact,
            );
            
            let schedule = TwapSliceSchedule {
                slice_id: i,
                scheduled_time,
                target_quantity,
                time_window_seconds: (self.config.slice_duration_minutes * 60) as i64,
                market_impact_estimate_bps: slice_impact.permanent_impact_bps + slice_impact.temporary_impact_bps,
                venue_allocation: self.calculate_slice_venue_allocation(
                    market_conditions,
                    execution_params,
                    urgency_level,
                )?,
                price_constraints: self.calculate_slice_price_constraints(
                    parent_order,
                    &slice_impact,
                    urgency_level,
                )?,
                urgency_level,
            };
            
            slice_schedules.push(schedule);
        }
        
        debug!("Calculated {} optimal time slices", slice_schedules.len());
        Ok(slice_schedules)
    }
    
    /// 计算最优切片数量
    fn calculate_optimal_slice_count(
        &self,
        parent_order: &ParentOrder,
        market_conditions: &MarketConditions,
        execution_params: &ExecutionParams,
    ) -> Result<u32> {
        let time_hours = parent_order.time_horizon as f64 / 3600.0;
        let base_slices = (time_hours * 60.0 / self.config.slice_duration_minutes as f64).ceil() as u32;
        
        // 基于市场冲击优化切片数量
        let impact_optimal_slices = self.calculate_impact_optimal_slice_count(
            parent_order,
            market_conditions,
        )?;
        
        // 基于流动性优化切片数量
        let liquidity_adjusted_slices = self.adjust_slices_for_liquidity(
            base_slices,
            parent_order,
            market_conditions,
        );
        
        // 选择最优方案
        let optimal_slices = if self.config.adaptive_slicing {
            // 使用加权平均
            let weights = [0.4, 0.4, 0.2]; // [base, impact, liquidity]
            let values = [base_slices as f64, impact_optimal_slices as f64, liquidity_adjusted_slices as f64];
            let weighted_avg = weights.iter().zip(values.iter())
                .map(|(w, v)| w * v)
                .sum::<f64>();
            weighted_avg.round() as u32
        } else {
            base_slices
        };
        
        let final_slices = optimal_slices
            .max(self.config.min_slice_count)
            .min(self.config.max_slice_count);
        
        debug!("Optimal slice count: base={}, impact_optimal={}, liquidity_adjusted={}, final={}",
               base_slices, impact_optimal_slices, liquidity_adjusted_slices, final_slices);
        
        Ok(final_slices)
    }
    
    /// 基于市场冲击计算最优切片数量
    fn calculate_impact_optimal_slice_count(
        &self,
        parent_order: &ParentOrder,
        market_conditions: &MarketConditions,
    ) -> Result<u32> {
        let mut optimal_count = self.config.min_slice_count;
        let mut min_total_cost = f64::INFINITY;
        
        // 测试不同的切片数量
        for slice_count in self.config.min_slice_count..=self.config.max_slice_count {
            let slice_quantity = parent_order.total_quantity / slice_count as f64;
            
            // 创建测试订单
            let test_order = ParentOrder {
                id: format!("{}_test", parent_order.id),
                symbol: parent_order.symbol.clone(),
                side: parent_order.side,
                total_quantity: slice_quantity,
                limit_price: parent_order.limit_price,
                time_horizon: (self.config.slice_duration_minutes * 60) as i64,
                urgency: parent_order.urgency,
                ..Default::default()
            };
            
            // 计算单个切片的市场冲击
            let slice_impact = self.market_impact_model.calculate_impact(
                &test_order,
                market_conditions,
                &ImpactExecutionStrategy::Twap,
            )?;
            
            // 估算总成本（冲击 + 时间风险）
            let impact_cost = slice_impact.permanent_impact_bps + slice_impact.temporary_impact_bps * 0.5;
            let time_risk_cost = parent_order.urgency * 2.0 / slice_count as f64; // 时间风险随切片数量降低
            let total_cost = impact_cost * slice_count as f64 + time_risk_cost;
            
            if total_cost < min_total_cost {
                min_total_cost = total_cost;
                optimal_count = slice_count;
            }
        }
        
        Ok(optimal_count)
    }
    
    /// 基于流动性调整切片数量
    fn adjust_slices_for_liquidity(
        &self,
        base_slices: u32,
        parent_order: &ParentOrder,
        market_conditions: &MarketConditions,
    ) -> u32 {
        let adv_ratio = parent_order.total_quantity / market_conditions.average_daily_volume;
        let current_liquidity_ratio = market_conditions.current_volume / market_conditions.average_daily_volume;
        
        let adjustment_factor = if adv_ratio > 0.1 {
            // 大单需要更多切片
            1.5
        } else if current_liquidity_ratio < 0.5 {
            // 低流动性时增加切片
            1.3
        } else if current_liquidity_ratio > 1.5 {
            // 高流动性时可以减少切片
            0.8
        } else {
            1.0
        };
        
        ((base_slices as f64 * adjustment_factor).round() as u32)
            .max(self.config.min_slice_count)
            .min(self.config.max_slice_count)
    }
    fn name(&self) -> &str {
        "TWAP"
    }

    fn calculate_child_orders(
        &self,
        parent_order: &ParentOrder,
        market_conditions: &MarketConditions,
        execution_params: &ExecutionParams,
    ) -> Result<Vec<ChildOrder>> {
        let execution_plan = self.create_execution_plan(
            parent_order,
            market_conditions,
            execution_params,
        )?;
        
        let mut child_orders = Vec::new();
        
        for schedule in execution_plan.slice_schedules {
            for (venue, allocation) in schedule.venue_allocation {
                let venue_quantity = schedule.target_quantity * allocation;
                
                if venue_quantity >= 100.0 { // 最小订单量
                    let child_order = ChildOrder {
                        id: format!("{}_twap_{}_{}", parent_order.id, schedule.slice_id, venue),
                        parent_id: parent_order.id.clone(),
                        sequence_number: child_orders.len() as u32,
                        quantity: venue_quantity,
                        price: schedule.price_constraints.limit_price,
                        venue,
                        order_type: if schedule.price_constraints.limit_price.is_some() {
                            OrderType::Limit
                        } else {
                            OrderType::Market
                        },
                        time_in_force: TimeInForce::GoodTillTime(
                            schedule.scheduled_time + Duration::seconds(schedule.time_window_seconds)
                        ),
                        scheduled_time: schedule.scheduled_time,
                        execution_window: schedule.time_window_seconds,
                        is_hidden: execution_params.hidden_order_ratio > 0.5,
                        display_quantity: if execution_params.iceberg_size_ratio > 0.0 {
                            Some(venue_quantity * execution_params.iceberg_size_ratio)
                        } else {
                            None
                        },
                        post_only: schedule.urgency_level < 0.3, // 低紧急度时使用被动订单
                        reduce_only: false,
                    };
                    
                    child_orders.push(child_order);
                }
            }
        }
        
        info!("TWAP algorithm generated {} child orders", child_orders.len());
        Ok(child_orders)
    }
    
    /// 计算切片的市场冲击
    fn calculate_slice_market_impact(
        &self,
        parent_order: &ParentOrder,
        slice_quantity: f64,
        _scheduled_time: &DateTime<Utc>,
        market_conditions: &MarketConditions,
    ) -> Result<MarketImpactResult> {
        let slice_order = ParentOrder {
            id: format!("{}_slice", parent_order.id),
            symbol: parent_order.symbol.clone(),
            side: parent_order.side,
            total_quantity: slice_quantity,
            limit_price: parent_order.limit_price,
            time_horizon: (self.config.slice_duration_minutes * 60) as i64,
            urgency: parent_order.urgency,
            ..Default::default()
        };
        
        self.market_impact_model.calculate_impact(
            &slice_order,
            market_conditions,
            &ImpactExecutionStrategy::Twap,
        )
    }
    
    /// 计算切片紧急度
    fn calculate_slice_urgency(
        &self,
        slice_index: u32,
        total_slices: u32,
        parent_urgency: f64,
        slice_impact: &MarketImpactResult,
    ) -> f64 {
        // 基础紧急度随时间增加
        let time_urgency = (slice_index as f64 / total_slices as f64) * parent_urgency;
        
        // 基于市场冲击的紧急度调整
        let impact_adjustment = if slice_impact.permanent_impact_bps + slice_impact.temporary_impact_bps > 10.0 {
            -0.2 // 高冲击时降低紧急度，更加谨慎
        } else {
            0.1 // 低冲击时可以稍微积极一些
        };
        
        (time_urgency + impact_adjustment).max(0.0).min(1.0)
    }
    
    /// 计算切片场所分配
    fn calculate_slice_venue_allocation(
        &self,
        market_conditions: &MarketConditions,
        execution_params: &ExecutionParams,
        urgency_level: f64,
    ) -> Result<HashMap<String, f64>> {
        let mut allocation = HashMap::new();
        
        if !execution_params.venue_preferences.is_empty() {
            // 使用配置的场所偏好
            let total_weight: f64 = execution_params.venue_preferences.values().sum();
            for (venue, weight) in &execution_params.venue_preferences {
                allocation.insert(venue.clone(), weight / total_weight);
            }
        } else {
            // 基于紧急度的默认分配策略
            if urgency_level > 0.7 {
                // 高紧急度：主要使用主要场所
                allocation.insert("PRIMARY".to_string(), 0.8);
                allocation.insert("DARK_POOL_1".to_string(), 0.2);
            } else if urgency_level > 0.3 {
                // 中等紧急度：平衡分配
                allocation.insert("PRIMARY".to_string(), 0.5);
                allocation.insert("DARK_POOL_1".to_string(), 0.3);
                allocation.insert("DARK_POOL_2".to_string(), 0.2);
            } else {
                // 低紧急度：优先使用暗池
                allocation.insert("PRIMARY".to_string(), 0.3);
                allocation.insert("DARK_POOL_1".to_string(), 0.4);
                allocation.insert("DARK_POOL_2".to_string(), 0.3);
            }
        }
        
        // 基于流动性调整
        if market_conditions.current_volume < market_conditions.average_daily_volume * 0.5 {
            // 低流动性时增加暗池比例
            if let Some(primary) = allocation.get_mut("PRIMARY") {
                *primary *= 0.7;
                let reduction = *primary * 0.3;
                allocation.entry("DARK_POOL_1".to_string()).and_modify(|v| *v += reduction * 0.6).or_insert(reduction * 0.6);
                allocation.entry("DARK_POOL_2".to_string()).and_modify(|v| *v += reduction * 0.4).or_insert(reduction * 0.4);
            }
        }
        
        Ok(allocation)
    }
    
    /// 计算切片价格约束
    fn calculate_slice_price_constraints(
        &self,
        parent_order: &ParentOrder,
        slice_impact: &MarketImpactResult,
        urgency_level: f64,
    ) -> Result<TwapPriceConstraints> {
        let impact_buffer_bps = (slice_impact.permanent_impact_bps + slice_impact.temporary_impact_bps) * 1.1;
        let timing_penalty_bps = urgency_level * 5.0; // 紧急度越高，允许的时机损失越大
        let max_deviation_bps = self.config.price_improvement_target_bps * 3.0 + impact_buffer_bps + timing_penalty_bps;
        
        // 基于紧急度调整限价
        let adjusted_limit_price = if let Some(limit_price) = parent_order.limit_price {
            let urgency_adjustment = urgency_level * 0.5; // 最高50bps的紧急度调整
            let adjustment_factor = match parent_order.side {
                OrderSide::Buy => 1.0 + (impact_buffer_bps + urgency_adjustment * 100.0) / 10000.0,
                OrderSide::Sell => 1.0 - (impact_buffer_bps + urgency_adjustment * 100.0) / 10000.0,
            };
            Some(limit_price * adjustment_factor)
        } else {
            None
        };
        
        Ok(TwapPriceConstraints {
            limit_price: adjusted_limit_price,
            max_deviation_bps,
            timing_penalty_bps,
            impact_buffer_bps,
        })
    }
    
    /// 估算期望TWAP
    fn estimate_expected_twap(
        &self,
        slice_schedules: &[TwapSliceSchedule],
        market_conditions: &MarketConditions,
        is_buy: bool,
    ) -> Result<f64> {
        if slice_schedules.is_empty() {
            return Ok(market_conditions.mid_price);
        }
        
        let total_quantity: f64 = slice_schedules.iter().map(|s| s.target_quantity).sum();
        let mut weighted_price_sum = 0.0;
        
        for schedule in slice_schedules {
            // 预期执行价格 = 当前中间价 + 市场冲击
            let impact_adjustment = if is_buy {
                schedule.market_impact_estimate_bps / 10000.0
            } else {
                -schedule.market_impact_estimate_bps / 10000.0
            };
            
            let expected_price = market_conditions.mid_price * (1.0 + impact_adjustment);
            weighted_price_sum += expected_price * schedule.target_quantity;
        }
        
        Ok(weighted_price_sum / total_quantity)
    }
    
    /// 计算总市场冲击
    fn calculate_total_market_impact(
        &self,
        parent_order: &ParentOrder,
        slice_schedules: &[TwapSliceSchedule],
        market_conditions: &MarketConditions,
    ) -> Result<MarketImpactResult> {
        // 使用整体订单计算总体市场冲击
        self.market_impact_model.calculate_impact(
            parent_order,
            market_conditions,
            &ImpactExecutionStrategy::Twap,
        )
    }
    
    /// 评估执行风险
    fn assess_execution_risk(
        &self,
        parent_order: &ParentOrder,
        slice_schedules: &[TwapSliceSchedule],
        market_conditions: &MarketConditions,
    ) -> Result<f64> {
        let mut risk_score = 0.0;
        
        // 时间风险：执行时间越长风险越高
        let time_risk = (parent_order.time_horizon as f64 / 3600.0) * 0.1; // 每小时10%的时间风险
        risk_score += time_risk;
        
        // 市场冲击风险
        let avg_impact: f64 = slice_schedules.iter()
            .map(|s| s.market_impact_estimate_bps)
            .sum::<f64>() / slice_schedules.len() as f64;
        let impact_risk = avg_impact / 100.0; // 每100bps冲击对应100%风险
        risk_score += impact_risk;
        
        // 流动性风险
        let adv_ratio = parent_order.total_quantity / market_conditions.average_daily_volume;
        let liquidity_risk = if adv_ratio > 0.1 {
            adv_ratio * 2.0
        } else {
            adv_ratio * 0.5
        };
        risk_score += liquidity_risk;
        
        // 波动性风险
        let volatility_risk = market_conditions.realized_volatility * 0.5;
        risk_score += volatility_risk;
        
        // 规范化风险分数到0-1范围
        Ok(risk_score.min(1.0).max(0.0))
    }

    fn adapt_parameters(
        &mut self, 
        execution_state: &ExecutionState, 
        market_update: &MarketUpdate
    ) -> Result<()> {
        if !self.config.adaptive_slicing {
            return Ok(());
        }
        
        // 计算执行进度
        let completion_ratio = execution_state.filled_quantity / execution_state.total_quantity;
        let time_elapsed_ratio = execution_state.elapsed_time as f64 / 
                                (execution_state.elapsed_time + execution_state.remaining_time) as f64;
        
        // 基于执行进度调整
        if completion_ratio < time_elapsed_ratio - 0.15 {
            // 执行明显落后，需要调整策略
            self.adaptive_params.urgency_multiplier *= 1.3;
            self.adaptive_params.current_slice_duration = 
                (self.adaptive_params.current_slice_duration as f64 * 0.8) as u32;
            warn!("TWAP execution falling behind schedule, increasing urgency: {:.2}", 
                  self.adaptive_params.urgency_multiplier);
        } else if completion_ratio > time_elapsed_ratio + 0.1 {
            // 执行过快，可以放慢节奏
            self.adaptive_params.urgency_multiplier *= 0.95;
            self.adaptive_params.current_slice_duration = 
                (self.adaptive_params.current_slice_duration as f64 * 1.1) as u32;
            debug!("TWAP execution ahead of schedule, reducing urgency: {:.2}", 
                   self.adaptive_params.urgency_multiplier);
        }
        
        // 基于市场更新调整
        match market_update.update_type {
            MarketUpdateType::VolumeSpike => {
                // 成交量激增，可以加快执行
                self.adaptive_params.dynamic_size_adjustment *= 1.2;
                self.adaptive_params.market_timing_score += 0.1;
            },
            MarketUpdateType::VolatilityChange => {
                if market_update.volatility_change_ratio > 1.5 {
                    // 波动性大幅增加，需要更加谨慎
                    self.adaptive_params.dynamic_size_adjustment *= 0.8;
                    self.adaptive_params.urgency_multiplier *= 0.9;
                }
            },
            MarketUpdateType::LiquidityShock => {
                // 流动性冲击，显著降低执行速度
                self.adaptive_params.dynamic_size_adjustment *= 0.6;
                self.adaptive_params.current_slice_duration = 
                    (self.adaptive_params.current_slice_duration as f64 * 1.5) as u32;
            },
            MarketUpdateType::NewsEvent => {
                // 新闻事件，暂时减缓执行
                self.adaptive_params.urgency_multiplier *= 0.85;
            },
            _ => {}
        }
        
        // 应用边界约束
        self.adaptive_params.urgency_multiplier = self.adaptive_params.urgency_multiplier
            .max(0.1).min(3.0);
        self.adaptive_params.dynamic_size_adjustment = self.adaptive_params.dynamic_size_adjustment
            .max(0.2).min(2.0);
        self.adaptive_params.current_slice_duration = self.adaptive_params.current_slice_duration
            .max(1).min(60); // 1-60分钟
        
        // 更新执行质量评分
        self.update_execution_quality_score(execution_state)?;
        
        self.adaptive_params.last_adjustment_time = Utc::now();
        
        Ok(())
    }
    
    /// 更新执行质量评分
    fn update_execution_quality_score(&mut self, execution_state: &ExecutionState) -> Result<()> {
        let mut quality_score = 0.0;
        
        // 基于滑点的质量评分
        let slippage_score = if execution_state.slippage_bps.abs() < 2.0 {
            1.0
        } else if execution_state.slippage_bps.abs() < 5.0 {
            0.8
        } else if execution_state.slippage_bps.abs() < 10.0 {
            0.6
        } else {
            0.3
        };
        
        // 基于市场冲击的质量评分
        let impact_score = if execution_state.market_impact_bps < 3.0 {
            1.0
        } else if execution_state.market_impact_bps < 7.0 {
            0.8
        } else {
            0.5
        };
        
        // 基于时机成本的质量评分
        let timing_score = if execution_state.timing_cost_bps.abs() < 1.0 {
            1.0
        } else if execution_state.timing_cost_bps.abs() < 3.0 {
            0.8
        } else {
            0.6
        };
        
        // 综合质量评分
        quality_score = (slippage_score * 0.4 + impact_score * 0.4 + timing_score * 0.2);
        
        // 平滑更新
        self.adaptive_params.execution_quality_score = 
            self.adaptive_params.execution_quality_score * 0.7 + quality_score * 0.3;
        
        Ok(())
    }

    fn get_statistics(&self) -> AlgorithmStatistics {
        self.statistics.clone()
    }

    fn validate_parameters(&self, params: &HashMap<String, f64>) -> Result<()> {
        // 验证关键参数
        if let Some(&slice_duration) = params.get("slice_duration_minutes") {
            if slice_duration <= 0.0 || slice_duration > 120.0 {
                return Err(anyhow::anyhow!("Invalid slice_duration_minutes: must be between 0 and 120"));
            }
        }
        
        if let Some(&slice_variance) = params.get("slice_size_variance") {
            if slice_variance < 0.0 || slice_variance > 1.0 {
                return Err(anyhow::anyhow!("Invalid slice_size_variance: must be between 0.0 and 1.0"));
            }
        }
        
        if let Some(&volume_participation) = params.get("volume_participation_limit") {
            if volume_participation <= 0.0 || volume_participation > 1.0 {
                return Err(anyhow::anyhow!("Invalid volume_participation_limit: must be between 0.0 and 1.0"));
            }
        }
        
        if let Some(&impact_threshold) = params.get("market_impact_threshold_bps") {
            if impact_threshold < 0.0 || impact_threshold > 100.0 {
                return Err(anyhow::anyhow!("Invalid market_impact_threshold_bps: must be between 0.0 and 100.0"));
            }
        }
        
        Ok(())
    }
} // 结束TwapAlgorithm impl块

// 默认实现
impl Default for TwapConfig {
    fn default() -> Self {
        Self {
            slice_duration_minutes: 10,
            min_slice_count: 4,
            max_slice_count: 48,
            slice_size_variance: 0.1,
            time_variance_minutes: 2,
            market_impact_threshold_bps: 5.0,
            adaptive_slicing: true,
            volume_participation_limit: 0.15,
            price_improvement_target_bps: 1.0,
            enable_aggressive_slicing: false,
            liquidity_detection_window: 30,
            cross_venue_optimization: true,
        }
    }
}

impl Default for TwapAdaptiveParams {
    fn default() -> Self {
        Self {
            current_slice_duration: 10,
            dynamic_size_adjustment: 1.0,
            urgency_multiplier: 1.0,
            market_timing_score: 0.5,
            execution_quality_score: 0.8,
            last_adjustment_time: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twap_algorithm_creation() {
        let algorithm = TwapAlgorithm::new().unwrap();
        assert_eq!(algorithm.name(), "TWAP");
    }
    
    #[test]
    fn test_optimal_slice_count_calculation() {
        let algorithm = TwapAlgorithm::new().unwrap();
        
        let parent_order = ParentOrder {
            id: "test_order".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            total_quantity: 10000.0,
            order_type: OrderType::Limit,
            time_horizon: 3600, // 1 hour
            urgency: 0.5,
            limit_price: Some(50000.0),
            arrival_price: 50000.0,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        };
        
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
        
        let slice_count = algorithm.calculate_optimal_slice_count(
            &parent_order,
            &market_conditions,
            &execution_params,
        ).unwrap();
        
        assert!(slice_count >= algorithm.config.min_slice_count);
        assert!(slice_count <= algorithm.config.max_slice_count);
    }
    
    #[test]
    fn test_parameter_validation() {
        let algorithm = TwapAlgorithm::new().unwrap();
        
        let mut valid_params = HashMap::new();
        valid_params.insert("slice_duration_minutes".to_string(), 15.0);
        valid_params.insert("slice_size_variance".to_string(), 0.2);
        valid_params.insert("volume_participation_limit".to_string(), 0.1);
        
        assert!(algorithm.validate_parameters(&valid_params).is_ok());
        
        let mut invalid_params = HashMap::new();
        invalid_params.insert("slice_duration_minutes".to_string(), -5.0);
        
        assert!(algorithm.validate_parameters(&invalid_params).is_err());
    }
}