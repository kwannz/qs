use super::*;
use super::market_impact::{AlmgrenChrissModel, MarketImpactResult, ImpactExecutionStrategy};
use anyhow::{Result, Context};
use chrono::{Duration, Utc, Timelike};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// PoV (Percentage of Volume) 执行算法
/// 
/// 该算法根据实时市场成交量动态调整订单大小，
/// 以维持相对稳定的市场参与率来执行大额订单
#[derive(Debug)]
pub struct PovAlgorithm {
    config: PovConfig,
    statistics: AlgorithmStatistics,
    volume_tracker: VolumeTracker,
    adaptive_controller: AdaptiveController,
    risk_monitor: RiskMonitor,
    market_impact_model: AlmgrenChrissModel,
    dynamic_participation_controller: DynamicParticipationController,
}

/// 动态参与率控制器
#[derive(Debug)]
struct DynamicParticipationController {
    base_participation_rate: f64,
    impact_adjusted_rate: f64,
    momentum_adjusted_rate: f64,
    liquidity_adjusted_rate: f64,
    final_participation_rate: f64,
    
    // 历史参与率数据
    participation_history: Vec<ParticipationPoint>,
    impact_efficiency_score: f64,
    adaptation_speed: f64,
    
    last_update: DateTime<Utc>,
}

/// 参与率数据点
#[derive(Debug, Clone)]
struct ParticipationPoint {
    timestamp: DateTime<Utc>,
    target_rate: f64,
    actual_rate: f64,
    market_impact_bps: f64,
    execution_quality: f64,
    market_regime: String,
}

/// PoV算法配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PovConfig {
    pub target_participation_rate: f64,        // 目标参与率
    pub min_participation_rate: f64,           // 最小参与率
    pub max_participation_rate: f64,           // 最大参与率
    pub participation_rate_tolerance: f64,     // 参与率容忍度
    
    pub volume_measurement_window_minutes: u32, // 成交量测量窗口(分钟)
    pub volume_smoothing_factor: f64,          // 成交量平滑因子
    pub volume_forecast_horizon_minutes: u32,  // 成交量预测时间范围
    
    pub order_refresh_interval_seconds: u32,   // 订单刷新间隔(秒)
    pub max_order_life_minutes: u32,          // 最大订单生存时间(分钟)
    pub min_order_size: f64,                  // 最小订单大小
    pub max_order_size: f64,                  // 最大订单大小
    
    pub price_aggressiveness: f64,             // 价格激进程度 (0.0-1.0)
    pub liquidity_detection_threshold: f64,    // 流动性检测阈值
    pub market_impact_penalty: f64,           // 市场冲击惩罚系数
    
    pub enable_volume_prediction: bool,        // 启用成交量预测
    pub enable_cross_venue_tracking: bool,    // 启用跨场所追踪
    pub enable_adaptive_sizing: bool,          // 启用自适应大小调整
    pub enable_momentum_detection: bool,       // 启用动量检测
}

/// 成交量追踪器
#[derive(Debug)]
struct VolumeTracker {
    historical_windows: Vec<VolumeWindow>,
    current_window: VolumeWindow,
    smoothed_volume_rate: f64,
    volume_acceleration: f64,
    prediction_model: Option<VolumePredictionModel>,
    last_update: DateTime<Utc>,
}

/// 成交量窗口
#[derive(Debug, Clone)]
struct VolumeWindow {
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    total_volume: f64,
    our_volume: f64,
    venue_breakdown: HashMap<String, f64>,
    volume_rate: f64, // 每秒成交量
    participation_rate: f64,
}

/// 成交量预测模型
#[derive(Debug)]
struct VolumePredictionModel {
    ema_short: f64,      // 短期指数移动平均
    ema_long: f64,       // 长期指数移动平均
    trend_coefficient: f64,
    seasonality_factors: Vec<f64>,
    confidence_interval: f64,
}

/// 自适应控制器
#[derive(Debug)]
struct AdaptiveController {
    current_participation_rate: f64,
    participation_rate_error: f64,
    integral_error: f64,
    derivative_error: f64,
    last_error: f64,
    
    // PID控制参数
    kp: f64, // 比例系数
    ki: f64, // 积分系数
    kd: f64, // 微分系数
    
    // 自适应参数
    adaptation_rate: f64,
    stability_threshold: f64,
    last_adjustment: DateTime<Utc>,
}

/// 风险监控器
#[derive(Debug)]
struct RiskMonitor {
    current_exposure: f64,
    max_exposure: f64,
    venue_concentration: HashMap<String, f64>,
    market_impact_estimate: f64,
    liquidity_consumption_rate: f64,
    warning_flags: Vec<RiskWarning>,
}

/// 风险警告
#[derive(Debug, Clone)]
enum RiskWarning {
    ExcessiveParticipation { rate: f64, threshold: f64 },
    VenueConcentration { venue: String, concentration: f64 },
    MarketImpact { impact_bps: f64 },
    LiquidityShortage { available_liquidity: f64 },
    VolatilitySpike { volatility_ratio: f64 },
}

/// PoV执行状态
#[derive(Debug, Clone)]
struct PovExecutionState {
    parent_order_id: String,
    target_quantity: f64,
    filled_quantity: f64,
    current_orders: Vec<ActiveOrder>,
    
    // 实时指标
    realized_participation_rate: f64,
    average_fill_price: f64,
    market_impact_bps: f64,
    slippage_bps: f64,
    
    // 控制状态
    next_order_size: f64,
    next_refresh_time: DateTime<Utc>,
    execution_urgency: f64,
    
    last_updated: DateTime<Utc>,
}

/// 活跃订单
#[derive(Debug, Clone)]
struct ActiveOrder {
    child_order: ChildOrder,
    submission_time: DateTime<Utc>,
    expected_fill_rate: f64,
    actual_fill_quantity: f64,
    market_share_consumed: f64,
}

impl PovAlgorithm {
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: PovConfig::default(),
            statistics: AlgorithmStatistics {
                algorithm_name: "POV".to_string(),
                ..Default::default()
            },
            volume_tracker: VolumeTracker::new(),
            adaptive_controller: AdaptiveController::new(),
            risk_monitor: RiskMonitor::new(),
            market_impact_model: AlmgrenChrissModel::new()?,
            dynamic_participation_controller: DynamicParticipationController::new(),
        })
    }
    
    pub fn with_config(config: PovConfig) -> Result<Self> {
        let mut instance = Self::new()?;
        instance.config = config;
        Ok(instance)
    }
    
    pub fn with_market_impact_model(mut self, model: AlmgrenChrissModel) -> Self {
        self.market_impact_model = model;
        self
    }
    
    /// 计算下一个订单大小（市场冲击感知）
    fn calculate_next_order_size(
        &self,
        execution_state: &PovExecutionState,
        market_conditions: &MarketConditions,
    ) -> Result<f64> {
        // 获取当前市场成交量率
        let current_volume_rate = self.estimate_current_volume_rate(market_conditions)?;
        
        // 计算市场冲击优化的参与率
        let impact_optimal_participation_rate = self.calculate_impact_optimal_participation_rate(
            execution_state,
            market_conditions,
        )?;
        
        // 计算目标订单大小
        let target_order_size = current_volume_rate 
            * self.config.order_refresh_interval_seconds as f64
            * impact_optimal_participation_rate;
        
        // 应用动态调整
        let adjusted_size = self.apply_dynamic_adjustments(
            target_order_size,
            execution_state,
            market_conditions,
        )?;
        
        // 市场冲击验证与调整
        let impact_validated_size = self.validate_and_adjust_for_market_impact(
            adjusted_size,
            execution_state,
            market_conditions,
        )?;
        
        // 应用约束
        let final_size = impact_validated_size
            .max(self.config.min_order_size)
            .min(self.config.max_order_size)
            .min(execution_state.target_quantity - execution_state.filled_quantity);
        
        debug!("Calculated impact-aware order size: {:.2} (target: {:.2}, adjusted: {:.2}, validated: {:.2})", 
               final_size, target_order_size, adjusted_size, impact_validated_size);
        
        Ok(final_size)
    }
    
    /// 计算市场冲击优化的参与率
    fn calculate_impact_optimal_participation_rate(
        &self,
        execution_state: &PovExecutionState,
        market_conditions: &MarketConditions,
    ) -> Result<f64> {
        let base_rate = self.config.target_participation_rate;
        
        // 创建测试订单来评估不同参与率下的市场冲击
        let remaining_quantity = execution_state.target_quantity - execution_state.filled_quantity;
        let test_volume = remaining_quantity * 0.1; // 使用10%作为测试量
        
        let test_order = ParentOrder {
            id: "pov_impact_test".to_string(),
            symbol: market_conditions.symbol.clone(),
            side: OrderSide::Buy, // 默认买单
            total_quantity: test_volume,
            limit_price: Some(market_conditions.mid_price),
            time_horizon: self.config.order_refresh_interval_seconds as i64,
            urgency: execution_state.execution_urgency,
            ..Default::default()
        };
        
        // 计算基础市场冲击
        let base_impact = self.market_impact_model.calculate_impact(
            &test_order,
            market_conditions,
            &ImpactExecutionStrategy::Pov,
        )?;
        
        // 根据冲击调整参与率
        let total_impact_bps = base_impact.permanent_impact_bps + base_impact.temporary_impact_bps;
        
        let impact_adjustment = if total_impact_bps > 15.0 {
            0.6  // 高冲击时大幅降低参与率
        } else if total_impact_bps > 8.0 {
            0.8  // 中等冲击时中度降低
        } else if total_impact_bps > 3.0 {
            0.9  // 低冲击时轻微降低
        } else {
            1.0  // 极低冲击时保持原参与率
        };
        
        // 基于流动性状况进一步调整
        let liquidity_factor = (market_conditions.current_volume / market_conditions.average_daily_volume)
            .min(2.0).max(0.3);
        
        let optimal_rate = (base_rate * impact_adjustment * liquidity_factor)
            .max(self.config.min_participation_rate)
            .min(self.config.max_participation_rate);
        
        debug!("Impact optimal participation rate: base={:.3}, impact_adj={:.2}, liquidity={:.2}, optimal={:.3}",
               base_rate, impact_adjustment, liquidity_factor, optimal_rate);
        
        Ok(optimal_rate)
    }
    
    /// 应用动态调整
    fn apply_dynamic_adjustments(
        &self,
        base_size: f64,
        execution_state: &PovExecutionState,
        market_conditions: &MarketConditions,
    ) -> Result<f64> {
        let mut adjusted_size = base_size;
        
        // 参与率误差调整（加强版）
        let participation_error = self.dynamic_participation_controller.final_participation_rate - 
            execution_state.realized_participation_rate;
        
        if participation_error.abs() > self.config.participation_rate_tolerance {
            let pid_adjustment = self.calculate_pid_adjustment(participation_error)?;
            adjusted_size *= (1.0 + pid_adjustment);
            
            debug!("Applied PID adjustment: error={:.4}, adjustment={:.3}", 
                   participation_error, pid_adjustment);
        }
        
        // 动量检测调整
        if self.config.enable_momentum_detection {
            let momentum_adjustment = self.calculate_momentum_adjustment(
                market_conditions,
                execution_state,
            )?;
            adjusted_size *= momentum_adjustment;
        }
        
        // 波动性自适应调整
        let volatility_adjustment = self.calculate_volatility_adjustment(
            market_conditions.realized_volatility,
            execution_state.execution_urgency,
        );
        adjusted_size *= volatility_adjustment;
        
        // 跨场所流动性调整
        if self.config.enable_cross_venue_tracking {
            let cross_venue_adjustment = self.calculate_cross_venue_adjustment(market_conditions)?;
            adjusted_size *= cross_venue_adjustment;
        }
        
        Ok(adjusted_size)
    }
    
    /// 计算PID调整
    fn calculate_pid_adjustment(&self, error: f64) -> Result<f64> {
        let kp = 0.8;
        let ki = 0.1;
        let kd = 0.05;
        
        // 简化PID计算（实际中需要维护积分和微分项）
        let proportional = kp * error;
        let integral = ki * error * 0.1; // 简化积分项
        let derivative = kd * error; // 简化微分项
        
        let adjustment = (proportional + integral + derivative).max(-0.5).min(0.5);
        Ok(adjustment)
    }
    
    /// 计算动量调整
    fn calculate_momentum_adjustment(
        &self,
        market_conditions: &MarketConditions,
        execution_state: &PovExecutionState,
    ) -> Result<f64> {
        let price_momentum = market_conditions.price_momentum;
        let volume_momentum = (market_conditions.current_volume / market_conditions.average_daily_volume - 1.0)
            .max(-0.5).min(1.0);
        
        // 顺势执行：价格上涨时买入更积极，价格下跌时卖出更积极
        let momentum_factor = if price_momentum.abs() > 0.02 {
            1.0 + price_momentum * 0.3 + volume_momentum * 0.2
        } else {
            1.0
        };
        
        Ok(momentum_factor.max(0.5).min(1.8))
    }
    
    /// 计算波动性调整
    fn calculate_volatility_adjustment(&self, realized_vol: f64, urgency: f64) -> f64 {
        let base_vol = 0.2; // 基准波动率
        let vol_ratio = realized_vol / base_vol;
        
        // 高波动性时减小订单，但紧急度高时传缾庅度减小
        let vol_adjustment = if vol_ratio > 2.0 {
            0.6 + urgency * 0.3
        } else if vol_ratio > 1.5 {
            0.8 + urgency * 0.15
        } else {
            1.0
        };
        
        vol_adjustment.max(0.3).min(1.2)
    }
    
    /// 计算跨场所调整
    fn calculate_cross_venue_adjustment(&self, _market_conditions: &MarketConditions) -> Result<f64> {
        // 简化实现，实际中需要真实的跨场所数据
        Ok(1.0)
    }
    
    /// 市场冲击验证与调整
    fn validate_and_adjust_for_market_impact(
        &self,
        proposed_size: f64,
        execution_state: &PovExecutionState,
        market_conditions: &MarketConditions,
    ) -> Result<f64> {
        // 创建测试订单
        let test_order = ParentOrder {
            id: "pov_validation_test".to_string(),
            symbol: market_conditions.symbol.clone(),
            side: OrderSide::Buy,
            total_quantity: proposed_size,
            limit_price: Some(market_conditions.mid_price),
            time_horizon: self.config.order_refresh_interval_seconds as i64,
            urgency: execution_state.execution_urgency,
            ..Default::default()
        };
        
        // 计算预期市场冲击
        let impact = self.market_impact_model.calculate_impact(
            &test_order,
            market_conditions,
            &ImpactExecutionStrategy::Pov,
        )?;
        
        let total_impact_bps = impact.permanent_impact_bps + impact.temporary_impact_bps;
        
        // 如果冲击过大，减小订单大小
        let validated_size = if total_impact_bps > self.config.market_impact_penalty * 10.0 {
            let reduction_factor = (self.config.market_impact_penalty * 10.0) / total_impact_bps;
            let reduced_size = proposed_size * reduction_factor;
            
            warn!("High market impact detected ({:.1}bps), reducing order size from {:.0} to {:.0}", 
                  total_impact_bps, proposed_size, reduced_size);
            
            reduced_size
        } else {
            proposed_size
        };
        
        Ok(validated_size)
    }
    
    /// 实时成交量追踪
    fn real_time_volume_tracking(&self, market_conditions: &MarketConditions) -> Result<f64> {
        self.estimate_current_volume_rate(market_conditions)
    }
    
    /// 估计当前成交量率
    fn estimate_current_volume_rate(&self, market_conditions: &MarketConditions) -> Result<f64> {
        // 基础成交量率
        let base_rate = market_conditions.current_volume / 
            (self.config.volume_measurement_window_minutes as f64 * 60.0);
        
        if !self.config.enable_volume_prediction {
            return Ok(base_rate);
        }
        
        // 使用预测模型调整
        if let Some(ref model) = self.volume_tracker.prediction_model {
            let trend_adjustment = 1.0 + model.trend_coefficient;
            let seasonal_adjustment = self.get_seasonal_factor(Utc::now())?;
            
            Ok(base_rate * trend_adjustment * seasonal_adjustment)
        } else {
            Ok(base_rate)
        }
    }
    
    /// 获取季节性因子
    fn get_seasonal_factor(&self, timestamp: DateTime<Utc>) -> Result<f64> {
        // 简化的日内季节性模式
        let hour = timestamp.hour();
        let seasonal_factor = match hour {
            9..=10 => 1.5,   // 开盘高峰
            11..=14 => 0.8,  // 中午低谷
            15..=16 => 1.3,  // 收盘前活跃
            _ => 1.0,
        };
        
        Ok(seasonal_factor)
    }
    
    /// 计算参与率调整
    fn calculate_participation_adjustment(
        &self,
        execution_state: &PovExecutionState,
        market_conditions: &MarketConditions,
    ) -> Result<f64> {
        let participation_error = self.adaptive_controller.current_participation_rate - 
            execution_state.realized_participation_rate;
        
        if participation_error.abs() > self.config.participation_rate_tolerance {
            Ok(1.0 + participation_error * 0.5)
        } else {
            Ok(1.0)
        }
    }
    
    /// 应用传统自适应调整（保留为兼容性）
    fn apply_adaptive_adjustments(
        &self,
        base_size: f64,
        execution_state: &PovExecutionState,
        market_conditions: &MarketConditions,
    ) -> Result<f64> {
        // 这个方法现在委托给更高级的apply_dynamic_adjustments
        self.apply_dynamic_adjustments(base_size, execution_state, market_conditions)
    }
    
    /// 计算订单价格
    fn calculate_order_price(
        &self,
        side: OrderSide,
        market_conditions: &MarketConditions,
        urgency: f64,
    ) -> Result<f64> {
        let mid_price = market_conditions.mid_price;
        let half_spread = (market_conditions.ask_price - market_conditions.bid_price) / 2.0;
        
        // 基于激进程度调整价格
        let aggressiveness = self.config.price_aggressiveness * (1.0 + urgency);
        
        let price = match side {
            OrderSide::Buy => {
                // 买单：从bid开始，根据激进程度向ask靠拢
                market_conditions.bid_price + half_spread * aggressiveness
            },
            OrderSide::Sell => {
                // 卖单：从ask开始，根据激进程度向bid靠拢
                market_conditions.ask_price - half_spread * aggressiveness
            },
        };
        
        Ok(price)
    }
    
    /// 更新成交量追踪器
    fn update_volume_tracker(
        &mut self,
        market_update: &MarketUpdate,
        our_fill_volume: f64,
    ) -> Result<()> {
        let now = Utc::now();
        
        // 更新当前窗口
        if now.signed_duration_since(self.volume_tracker.current_window.start_time).num_minutes() 
            >= self.config.volume_measurement_window_minutes as i64 {
            
            // 完成当前窗口，创建新窗口
            self.volume_tracker.historical_windows.push(self.volume_tracker.current_window.clone());
            
            // 保持历史窗口数量限制
            if self.volume_tracker.historical_windows.len() > 100 {
                self.volume_tracker.historical_windows.remove(0);
            }
            
            // 开始新窗口
            self.volume_tracker.current_window = VolumeWindow {
                start_time: now,
                end_time: now + Duration::minutes(self.config.volume_measurement_window_minutes as i64),
                total_volume: 0.0,
                our_volume: 0.0,
                venue_breakdown: HashMap::new(),
                volume_rate: 0.0,
                participation_rate: 0.0,
            };
        }
        
        // 更新当前窗口数据
        self.volume_tracker.current_window.total_volume += market_update.volume_change_ratio;
        self.volume_tracker.current_window.our_volume += our_fill_volume;
        
        // 计算成交量率
        let window_duration_seconds = now.signed_duration_since(
            self.volume_tracker.current_window.start_time
        ).num_seconds() as f64;
        
        if window_duration_seconds > 0.0 {
            self.volume_tracker.current_window.volume_rate = 
                self.volume_tracker.current_window.total_volume / window_duration_seconds;
                
            self.volume_tracker.current_window.participation_rate = 
                if self.volume_tracker.current_window.total_volume > 0.0 {
                    self.volume_tracker.current_window.our_volume / 
                    self.volume_tracker.current_window.total_volume
                } else {
                    0.0
                };
        }
        
        // 更新平滑成交量率
        let alpha = self.config.volume_smoothing_factor;
        self.volume_tracker.smoothed_volume_rate = 
            alpha * self.volume_tracker.current_window.volume_rate + 
            (1.0 - alpha) * self.volume_tracker.smoothed_volume_rate;
        
        self.volume_tracker.last_update = now;
        
        Ok(())
    }
    
    /// 更新自适应控制器
    fn update_adaptive_controller(
        &mut self,
        current_participation_rate: f64,
        target_participation_rate: f64,
        dt: f64,
    ) -> Result<()> {
        let error = target_participation_rate - current_participation_rate;
        
        // PID控制器更新
        self.adaptive_controller.participation_rate_error = error;
        self.adaptive_controller.integral_error += error * dt;
        self.adaptive_controller.derivative_error = 
            (error - self.adaptive_controller.last_error) / dt;
        
        // 计算控制输出
        let control_output = 
            self.adaptive_controller.kp * error +
            self.adaptive_controller.ki * self.adaptive_controller.integral_error +
            self.adaptive_controller.kd * self.adaptive_controller.derivative_error;
        
        // 更新当前参与率
        self.adaptive_controller.current_participation_rate = 
            (target_participation_rate + control_output)
            .max(self.config.min_participation_rate)
            .min(self.config.max_participation_rate);
        
        self.adaptive_controller.last_error = error;
        self.adaptive_controller.last_adjustment = Utc::now();
        
        debug!("Adaptive controller updated: error={:.4}, control_output={:.4}, new_rate={:.4}",
               error, control_output, self.adaptive_controller.current_participation_rate);
        
        Ok(())
    }
    
    /// 检查风险条件
    fn check_risk_conditions(
        &mut self,
        execution_state: &PovExecutionState,
        market_conditions: &MarketConditions,
    ) -> Result<Vec<RiskWarning>> {
        let mut warnings = Vec::new();
        
        // 检查过度参与
        if execution_state.realized_participation_rate > self.config.max_participation_rate {
            warnings.push(RiskWarning::ExcessiveParticipation {
                rate: execution_state.realized_participation_rate,
                threshold: self.config.max_participation_rate,
            });
        }
        
        // 检查市场冲击
        if execution_state.market_impact_bps > 10.0 {
            warnings.push(RiskWarning::MarketImpact {
                impact_bps: execution_state.market_impact_bps,
            });
        }
        
        // 检查流动性
        let available_liquidity = market_conditions.bid_size + market_conditions.ask_size;
        if available_liquidity < self.config.min_order_size * 5.0 {
            warnings.push(RiskWarning::LiquidityShortage {
                available_liquidity,
            });
        }
        
        // 检查波动性
        if market_conditions.realized_volatility > 0.5 {
            warnings.push(RiskWarning::VolatilitySpike {
                volatility_ratio: market_conditions.realized_volatility / 0.2,
            });
        }
        
        self.risk_monitor.warning_flags = warnings.clone();
        
        Ok(warnings)
    }
}

impl ExecutionAlgorithm for PovAlgorithm {
    fn name(&self) -> &str {
        "POV"
    }
    
    fn calculate_child_orders(
        &self,
        parent_order: &ParentOrder,
        market_conditions: &MarketConditions,
        execution_params: &ExecutionParams,
    ) -> Result<Vec<ChildOrder>> {
        let mut child_orders = Vec::new();
        
        // 创建执行状态
        let execution_state = PovExecutionState {
            parent_order_id: parent_order.id.clone(),
            target_quantity: parent_order.total_quantity,
            filled_quantity: 0.0,
            current_orders: Vec::new(),
            realized_participation_rate: 0.0,
            average_fill_price: 0.0,
            market_impact_bps: 0.0,
            slippage_bps: 0.0,
            next_order_size: 0.0,
            next_refresh_time: Utc::now(),
            execution_urgency: parent_order.urgency,
            last_updated: Utc::now(),
        };
        
        // 计算初始订单大小
        let order_size = self.calculate_next_order_size(&execution_state, market_conditions)?;
        
        if order_size > 0.0 {
            // 计算订单价格
            let order_price = self.calculate_order_price(
                parent_order.side.clone(),
                market_conditions,
                parent_order.urgency,
            )?;
            
            // 创建子订单
            let child_order = ChildOrder {
                id: format!("{}_pov_0", parent_order.id),
                parent_id: parent_order.id.clone(),
                sequence_number: 0,
                quantity: order_size,
                price: Some(order_price),
                venue: "PRIMARY".to_string(),
                order_type: OrderType::Limit,
                time_in_force: TimeInForce::GoodTillTime(
                    Utc::now() + Duration::minutes(self.config.max_order_life_minutes as i64)
                ),
                scheduled_time: Utc::now(),
                execution_window: self.config.order_refresh_interval_seconds as i64,
                is_hidden: execution_params.hidden_order_ratio > 0.5,
                display_quantity: if execution_params.iceberg_size_ratio > 0.0 {
                    Some(order_size * execution_params.iceberg_size_ratio)
                } else {
                    None
                },
                post_only: false,
                reduce_only: false,
            };
            
            child_orders.push(child_order);
        }
        
        info!("PoV algorithm generated {} initial child orders", child_orders.len());
        Ok(child_orders)
    }
    
    fn adapt_parameters(
        &mut self,
        execution_state: &ExecutionState,
        market_update: &MarketUpdate,
    ) -> Result<()> {
        // 更新成交量追踪器
        let our_fill_volume = if let Some(last_fill) = execution_state.active_child_orders
            .iter()
            .flat_map(|order| &order.fills)
            .max_by_key(|fill| fill.timestamp) {
            last_fill.quantity
        } else {
            0.0
        };
        
        self.update_volume_tracker(market_update, our_fill_volume)?;
        
        // 计算当前参与率
        let current_participation_rate = if self.volume_tracker.current_window.total_volume > 0.0 {
            self.volume_tracker.current_window.our_volume / 
            self.volume_tracker.current_window.total_volume
        } else {
            0.0
        };
        
        // 更新自适应控制器
        let dt = Utc::now().signed_duration_since(
            self.adaptive_controller.last_adjustment
        ).num_seconds() as f64;
        
        if dt > 0.0 {
            self.update_adaptive_controller(
                current_participation_rate,
                self.config.target_participation_rate,
                dt,
            )?;
        }
        
        // 检查风险条件
        let pov_execution_state = PovExecutionState {
            parent_order_id: execution_state.parent_order_id.clone(),
            target_quantity: execution_state.total_quantity,
            filled_quantity: execution_state.filled_quantity,
            current_orders: Vec::new(),
            realized_participation_rate: current_participation_rate,
            average_fill_price: execution_state.average_fill_price,
            market_impact_bps: execution_state.market_impact_bps,
            slippage_bps: execution_state.slippage_bps,
            next_order_size: 0.0,
            next_refresh_time: Utc::now(),
            execution_urgency: execution_state.current_risk_score,
            last_updated: execution_state.last_updated,
        };
        
        let warnings = self.check_risk_conditions(&pov_execution_state, &market_update.market_conditions)?;
        
        if !warnings.is_empty() {
            warn!("PoV algorithm detected {} risk warnings", warnings.len());
            for warning in &warnings {
                debug!("Risk warning: {:?}", warning);
            }
        }
        
        Ok(())
    }
    
    fn get_statistics(&self) -> AlgorithmStatistics {
        self.statistics.clone()
    }
    
    fn validate_parameters(&self, params: &HashMap<String, f64>) -> Result<()> {
        if let Some(&target_rate) = params.get("target_participation_rate") {
            if target_rate <= 0.0 || target_rate > 1.0 {
                return Err(anyhow::anyhow!("Invalid target_participation_rate: must be between 0.0 and 1.0"));
            }
        }
        
        if let Some(&min_rate) = params.get("min_participation_rate") {
            if min_rate <= 0.0 || min_rate > 0.5 {
                return Err(anyhow::anyhow!("Invalid min_participation_rate: must be between 0.0 and 0.5"));
            }
        }
        
        if let Some(&max_rate) = params.get("max_participation_rate") {
            if max_rate <= 0.0 || max_rate > 1.0 {
                return Err(anyhow::anyhow!("Invalid max_participation_rate: must be between 0.0 and 1.0"));
            }
        }
        
        Ok(())
    }
}

// 实现各个组件的构造函数
impl VolumeTracker {
    fn new() -> Self {
        Self {
            historical_windows: Vec::new(),
            current_window: VolumeWindow {
                start_time: Utc::now(),
                end_time: Utc::now() + Duration::minutes(5),
                total_volume: 0.0,
                our_volume: 0.0,
                venue_breakdown: HashMap::new(),
                volume_rate: 0.0,
                participation_rate: 0.0,
            },
            smoothed_volume_rate: 0.0,
            volume_acceleration: 0.0,
            prediction_model: None,
            last_update: Utc::now(),
        }
    }
}

impl AdaptiveController {
    fn new() -> Self {
        Self {
            current_participation_rate: 0.1,
            participation_rate_error: 0.0,
            integral_error: 0.0,
            derivative_error: 0.0,
            last_error: 0.0,
            kp: 1.0,
            ki: 0.1,
            kd: 0.01,
            adaptation_rate: 0.1,
            stability_threshold: 0.02,
            last_adjustment: Utc::now(),
        }
    }
}

impl RiskMonitor {
    fn new() -> Self {
        Self {
            current_exposure: 0.0,
            max_exposure: 1000000.0,
            venue_concentration: HashMap::new(),
            market_impact_estimate: 0.0,
            liquidity_consumption_rate: 0.0,
            warning_flags: Vec::new(),
        }
    }
}

impl DynamicParticipationController {
    fn new() -> Self {
        Self {
            base_participation_rate: 0.15,
            impact_adjusted_rate: 0.15,
            momentum_adjusted_rate: 0.15,
            liquidity_adjusted_rate: 0.15,
            final_participation_rate: 0.15,
            participation_history: Vec::new(),
            impact_efficiency_score: 0.8,
            adaptation_speed: 0.1,
            last_update: Utc::now(),
        }
    }
    
    /// 更新动态参与率控制
    fn update_participation_control(
        &mut self,
        current_participation: f64,
        target_participation: f64,
        market_impact_bps: f64,
        execution_quality: f64,
        market_conditions: &MarketConditions,
    ) -> Result<()> {
        // 记录历史数据点
        let participation_point = ParticipationPoint {
            timestamp: Utc::now(),
            target_rate: target_participation,
            actual_rate: current_participation,
            market_impact_bps,
            execution_quality,
            market_regime: self.classify_market_regime(market_conditions),
        };
        
        self.participation_history.push(participation_point);
        
        // 保持历史数据限制
        if self.participation_history.len() > 100 {
            self.participation_history.remove(0);
        }
        
        // 更新影响效率分数
        self.update_impact_efficiency_score()?;
        
        // 计算各项调整后的参与率
        self.base_participation_rate = target_participation;
        self.impact_adjusted_rate = self.calculate_impact_adjusted_rate(market_impact_bps)?;
        self.momentum_adjusted_rate = self.calculate_momentum_adjusted_rate(market_conditions)?;
        self.liquidity_adjusted_rate = self.calculate_liquidity_adjusted_rate(market_conditions)?;
        
        // 综合计算最终参与率
        self.final_participation_rate = self.calculate_final_participation_rate();
        
        self.last_update = Utc::now();
        
        debug!("Dynamic participation updated: base={:.3}, impact_adj={:.3}, final={:.3}",
               self.base_participation_rate, self.impact_adjusted_rate, self.final_participation_rate);
        
        Ok(())
    }
    
    /// 分类市场制度
    fn classify_market_regime(&self, market_conditions: &MarketConditions) -> String {
        let vol_ratio = market_conditions.realized_volatility / 0.2;
        let volume_ratio = market_conditions.current_volume / market_conditions.average_daily_volume;
        let spread_ratio = market_conditions.spread_bps / 5.0; // 基准5bps
        
        if vol_ratio > 2.0 && spread_ratio > 2.0 {
            "HighVolHighSpread".to_string()
        } else if vol_ratio > 1.5 {
            "HighVolatility".to_string()
        } else if volume_ratio < 0.3 {
            "LowLiquidity".to_string()
        } else if spread_ratio > 3.0 {
            "WideSpread".to_string()
        } else {
            "Normal".to_string()
        }
    }
    
    /// 更新影响效率分数
    fn update_impact_efficiency_score(&mut self) -> Result<()> {
        if self.participation_history.len() < 5 {
            return Ok(());
        }
        
        let recent_points = &self.participation_history[self.participation_history.len()-5..];
        
        // 计算平均执行质量和市场冲击
        let avg_quality: f64 = recent_points.iter().map(|p| p.execution_quality).sum::<f64>() / 5.0;
        let avg_impact: f64 = recent_points.iter().map(|p| p.market_impact_bps).sum::<f64>() / 5.0;
        
        // 效率分数 = 执行质量 / (1 + 市场冲击)
        let new_efficiency = avg_quality / (1.0 + avg_impact / 10.0);
        
        // 平滑更新
        self.impact_efficiency_score = self.impact_efficiency_score * 0.8 + new_efficiency * 0.2;
        
        Ok(())
    }
    
    /// 计算影响调整后的参与率
    fn calculate_impact_adjusted_rate(&self, market_impact_bps: f64) -> Result<f64> {
        let impact_penalty = if market_impact_bps > 20.0 {
            0.5
        } else if market_impact_bps > 10.0 {
            0.7
        } else if market_impact_bps > 5.0 {
            0.85
        } else {
            1.0
        };
        
        Ok(self.base_participation_rate * impact_penalty)
    }
    
    /// 计算动量调整后的参与率
    fn calculate_momentum_adjusted_rate(&self, market_conditions: &MarketConditions) -> Result<f64> {
        let momentum_factor = 1.0 + market_conditions.price_momentum * 0.2;
        Ok(self.impact_adjusted_rate * momentum_factor.max(0.5).min(1.5))
    }
    
    /// 计算流动性调整后的参与率
    fn calculate_liquidity_adjusted_rate(&self, market_conditions: &MarketConditions) -> Result<f64> {
        let liquidity_ratio = market_conditions.current_volume / market_conditions.average_daily_volume;
        let liquidity_factor = liquidity_ratio.max(0.3).min(2.0);
        Ok(self.momentum_adjusted_rate * liquidity_factor)
    }
    
    /// 计算最终参与率
    fn calculate_final_participation_rate(&self) -> f64 {
        // 加权平均：流动性调整 70%，影响调整 30%
        let weighted_rate = self.liquidity_adjusted_rate * 0.7 + self.impact_adjusted_rate * 0.3;
        
        // 基于历史效率调整
        weighted_rate * (0.8 + self.impact_efficiency_score * 0.4)
    }
}

impl Default for PovConfig {
    fn default() -> Self {
        Self {
            target_participation_rate: 0.15,
            min_participation_rate: 0.05,
            max_participation_rate: 0.30,
            participation_rate_tolerance: 0.02,
            volume_measurement_window_minutes: 5,
            volume_smoothing_factor: 0.2,
            volume_forecast_horizon_minutes: 15,
            order_refresh_interval_seconds: 30,
            max_order_life_minutes: 5,
            min_order_size: 100.0,
            max_order_size: 10000.0,
            price_aggressiveness: 0.5,
            liquidity_detection_threshold: 0.5,
            market_impact_penalty: 0.8,
            enable_volume_prediction: true,
            enable_cross_venue_tracking: true,
            enable_adaptive_sizing: true,
            enable_momentum_detection: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pov_algorithm_creation() {
        let algorithm = PovAlgorithm::new().unwrap();
        assert_eq!(algorithm.name(), "POV");
    }
    
    #[test]
    fn test_volume_rate_calculation() {
        let algorithm = PovAlgorithm::new().unwrap();
        
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
        
        let volume_rate = algorithm.estimate_current_volume_rate(&market_conditions).unwrap();
        assert!(volume_rate > 0.0);
    }
    
    #[test]
    fn test_parameter_validation() {
        let algorithm = PovAlgorithm::new().unwrap();
        
        let mut valid_params = HashMap::new();
        valid_params.insert("target_participation_rate".to_string(), 0.15);
        valid_params.insert("min_participation_rate".to_string(), 0.05);
        valid_params.insert("max_participation_rate".to_string(), 0.30);
        
        assert!(algorithm.validate_parameters(&valid_params).is_ok());
        
        let mut invalid_params = HashMap::new();
        invalid_params.insert("target_participation_rate".to_string(), 1.5);
        
        assert!(algorithm.validate_parameters(&invalid_params).is_err());
    }
}