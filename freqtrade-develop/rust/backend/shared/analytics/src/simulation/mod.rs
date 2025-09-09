pub mod realistic_constraints;
// pub mod market_simulation;
// pub mod slippage_model;
// pub mod liquidity_model;

use anyhow::Result;
use serde::{Deserialize, Serialize};
// Note: HashMap import removed as unused
use chrono::{DateTime, Utc};

// 临时模块结构体
pub mod market_simulation {
    use super::*;
    
    #[derive(Debug)]
    pub struct MarketSimulator;
    
    #[derive(Debug, Clone)]
    pub struct MarketState;
    
    impl MarketSimulator {
        pub fn new(_config: &super::SimulationConfig) -> Self {
            Self
        }
        
        pub fn simulate_market_response(
            &mut self,
            _order: &super::SimulationOrder,
            _market_data: &super::MarketData,
        ) -> anyhow::Result<MarketState> {
            Ok(MarketState)
        }
    }
}

pub mod slippage_model {
    use super::*;
    
    #[derive(Debug)]
    pub struct SlippageModel;
    
    #[derive(Debug, Clone)]
    pub struct SlippageInfo {
        pub total_slippage: f64,
        pub market_impact: f64,
        pub timing_cost: f64,
        pub delay_cost: f64,
    }
    
    impl SlippageModel {
        pub fn new() -> Self {
            Self
        }
        
        pub fn calculate_slippage(
            &self,
            _order: &super::SimulationOrder,
            _market_data: &super::MarketData,
            _liquidity_info: &super::liquidity_model::LiquidityInfo,
        ) -> anyhow::Result<SlippageInfo> {
            Ok(SlippageInfo {
                total_slippage: 0.01,
                market_impact: 0.005,
                timing_cost: 0.003,
                delay_cost: 0.002,
            })
        }
    }
    
    impl Default for SlippageInfo {
        fn default() -> Self {
            Self {
                total_slippage: 0.0,
                market_impact: 0.0,
                timing_cost: 0.0,
                delay_cost: 0.0,
            }
        }
    }
}

pub mod liquidity_model {
    use super::*;
    
    #[derive(Debug)]
    pub struct LiquidityModel;
    
    #[derive(Debug, Clone)]
    pub struct LiquidityInfo {
        pub sufficient: bool,
        pub available_ratio: f64,
        pub liquidity_score: f64,
    }
    
    impl LiquidityModel {
        pub fn new() -> Self {
            Self
        }
        
        pub fn analyze_liquidity(
            &self,
            _order: &super::SimulationOrder,
            _market_data: &super::MarketData,
        ) -> anyhow::Result<LiquidityInfo> {
            Ok(LiquidityInfo {
                sufficient: true,
                available_ratio: 1.0,
                liquidity_score: 0.8,
            })
        }
    }
}

/// AG3真实约束仿真引擎
pub struct RealisticSimulationEngine {
    constraints: realistic_constraints::ConstraintEngine,
    market_sim: market_simulation::MarketSimulator,
    slippage_model: slippage_model::SlippageModel,
    liquidity_model: liquidity_model::LiquidityModel,
    config: SimulationConfig,
}

/// 仿真配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub enable_partial_fills: bool,       // 启用部分成交
    pub enable_queue_position: bool,      // 启用队列位置模拟
    pub enable_min_lot_size: bool,        // 启用最小成交单位
    pub enable_tick_size: bool,           // 启用价格步长
    pub enable_rate_limits: bool,         // 启用频率限制
    pub enable_rejection_simulation: bool, // 启用拒单模拟
    pub enable_latency_simulation: bool,  // 启用延迟模拟
    pub realistic_matching: bool,         // 启用真实撮合
    pub iceberg_detection: bool,          // 启用冰山单检测
    pub market_impact_decay: f64,         // 市场冲击衰减率
    pub max_simulation_time_ms: u64,      // 最大仿真时间
}

/// 仿真订单
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationOrder {
    pub id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: f64,
    pub price: Option<f64>,
    pub venue: String,
    pub strategy_id: String,
    pub timestamp: DateTime<Utc>,
    pub time_in_force: TimeInForce,
    pub min_quantity: Option<f64>,
    pub display_quantity: Option<f64>,
    pub is_iceberg: bool,
    pub parent_order_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
    IcebergLimit,
    HiddenLimit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeInForce {
    GTC,  // Good Till Cancel
    IOC,  // Immediate Or Cancel
    FOK,  // Fill Or Kill
    GTD,  // Good Till Date
}

/// 仿真执行结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationExecution {
    pub order_id: String,
    pub execution_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub price: f64,
    pub commission: f64,
    pub venue: String,
    pub timestamp: DateTime<Utc>,
    pub liquidity_flag: LiquidityFlag,
    pub execution_quality: ExecutionQuality,
    pub constraint_violations: Vec<ConstraintViolation>,
    pub simulation_metadata: SimulationMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LiquidityFlag {
    Maker,
    Taker,
    Unknown,
}

/// 执行质量指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionQuality {
    pub effective_slippage: f64,          // 有效滑点
    pub implementation_shortfall: f64,    // 实施缺口
    pub market_impact: f64,               // 市场冲击
    pub timing_cost: f64,                 // 时机成本
    pub opportunity_cost: f64,            // 机会成本
    pub total_cost_bps: f64,              // 总成本（基点）
    pub fill_ratio: f64,                  // 成交比率
    pub speed_of_execution: f64,          // 执行速度
}

/// 约束违反
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    pub violation_type: String,
    pub description: String,
    pub severity: ViolationSeverity,
    pub impact: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// 仿真元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationMetadata {
    pub queue_position: Option<u32>,      // 队列位置
    pub queue_wait_time_ms: Option<u64>,  // 队列等待时间
    pub partial_fills: Vec<PartialFill>,  // 部分成交记录
    pub rejected_quantity: f64,           // 拒单数量
    pub latency_ms: u64,                  // 延迟时间
    pub venue_response: VenueResponse,    // 场所响应
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialFill {
    pub quantity: f64,
    pub price: f64,
    pub timestamp: DateTime<Utc>,
    pub sequence: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VenueResponse {
    Accepted,
    Rejected(String),
    PartiallyFilled,
    Queued,
    Expired,
}

impl RealisticSimulationEngine {
    pub fn new(config: SimulationConfig) -> Self {
        Self {
            constraints: realistic_constraints::ConstraintEngine::new(&config),
            market_sim: market_simulation::MarketSimulator::new(&config),
            slippage_model: slippage_model::SlippageModel::new(),
            liquidity_model: liquidity_model::LiquidityModel::new(),
            config,
        }
    }

    /// 执行仿真交易
    pub fn execute_order(
        &mut self,
        order: SimulationOrder,
        market_data: &MarketData,
    ) -> Result<Vec<SimulationExecution>> {
        let start_time = std::time::Instant::now();
        
        // 1. 约束检查
        let constraint_result = self.constraints.check_constraints(&order, market_data)?;
        if !constraint_result.passed {
            return Ok(vec![self.create_rejected_execution(order, constraint_result.violations)?]);
        }
        
        // 2. 流动性检查
        let liquidity_info = self.liquidity_model.analyze_liquidity(&order, market_data)?;
        if !liquidity_info.sufficient {
            return Ok(vec![self.create_liquidity_constrained_execution(order, liquidity_info)?]);
        }
        
        // 3. 市场仿真
        let market_state = self.market_sim.simulate_market_response(&order, market_data)?;
        
        // 4. 滑点计算
        let slippage_info = self.slippage_model.calculate_slippage(
            &order, 
            market_data, 
            &liquidity_info,
        )?;
        
        // 5. 执行仿真
        let executions = self.simulate_execution(
            order,
            market_data,
            &market_state,
            &slippage_info,
            &liquidity_info,
        )?;
        
        // 6. 记录仿真时间
        let simulation_time = start_time.elapsed().as_millis() as u64;
        if simulation_time > self.config.max_simulation_time_ms {
            log::warn!("Simulation took {}ms, exceeding limit", simulation_time);
        }
        
        Ok(executions)
    }

    /// 批量执行仿真
    pub fn execute_batch(
        &mut self,
        orders: Vec<SimulationOrder>,
        market_data: &MarketData,
    ) -> Result<Vec<Vec<SimulationExecution>>> {
        let mut all_executions = Vec::new();
        let mut updated_market_data = market_data.clone();
        
        for order in orders {
            // 执行单个订单
            let executions = self.execute_order(order, &updated_market_data)?;
            
            // 更新市场状态（考虑市场冲击）
            if let Some(last_execution) = executions.last() {
                self.update_market_impact(&mut updated_market_data, last_execution)?;
            }
            
            all_executions.push(executions);
        }
        
        Ok(all_executions)
    }

    /// 仿真具体执行过程
    fn simulate_execution(
        &mut self,
        order: SimulationOrder,
        market_data: &MarketData,
        market_state: &market_simulation::MarketState,
        slippage_info: &slippage_model::SlippageInfo,
        liquidity_info: &liquidity_model::LiquidityInfo,
    ) -> Result<Vec<SimulationExecution>> {
        let mut executions = Vec::new();
        let mut remaining_quantity = order.quantity;
        let mut sequence = 0u32;
        
        // 根据订单类型和市场状态决定执行方式
        match order.order_type {
            OrderType::Market => {
                executions.extend(self.simulate_market_order(
                    &order, 
                    market_data, 
                    market_state,
                    slippage_info,
                    &mut remaining_quantity,
                    &mut sequence,
                )?);
            }
            OrderType::Limit => {
                executions.extend(self.simulate_limit_order(
                    &order,
                    market_data,
                    market_state,
                    liquidity_info,
                    &mut remaining_quantity,
                    &mut sequence,
                )?);
            }
            OrderType::IcebergLimit => {
                executions.extend(self.simulate_iceberg_order(
                    &order,
                    market_data,
                    market_state,
                    liquidity_info,
                    &mut remaining_quantity,
                    &mut sequence,
                )?);
            }
            _ => {
                // 其他订单类型的简化处理
                executions.extend(self.simulate_market_order(
                    &order, 
                    market_data, 
                    market_state,
                    slippage_info,
                    &mut remaining_quantity,
                    &mut sequence,
                )?);
            }
        }
        
        Ok(executions)
    }

    /// 仿真市价单执行
    fn simulate_market_order(
        &self,
        order: &SimulationOrder,
        market_data: &MarketData,
        market_state: &market_simulation::MarketState,
        slippage_info: &slippage_model::SlippageInfo,
        remaining_quantity: &mut f64,
        sequence: &mut u32,
    ) -> Result<Vec<SimulationExecution>> {
        let mut executions = Vec::new();
        
        // 获取最优价格
        let best_price = match order.side {
            OrderSide::Buy => market_data.best_ask(),
            OrderSide::Sell => market_data.best_bid(),
        };
        
        if let Some(execution_price) = best_price {
            // 应用滑点
            let final_price = execution_price + slippage_info.total_slippage;
            
            // 计算实际成交量
            let executable_quantity = if self.config.enable_partial_fills {
                self.calculate_executable_quantity(
                    *remaining_quantity, 
                    market_data, 
                    &order.side,
                )?
            } else {
                *remaining_quantity
            };
            
            // 创建执行记录
            let execution = self.create_execution(
                order,
                executable_quantity,
                final_price,
                market_data,
                slippage_info,
                *sequence,
            )?;
            
            executions.push(execution);
            *remaining_quantity -= executable_quantity;
            *sequence += 1;
        }
        
        Ok(executions)
    }

    /// 仿真限价单执行
    fn simulate_limit_order(
        &self,
        order: &SimulationOrder,
        market_data: &MarketData,
        market_state: &market_simulation::MarketState,
        liquidity_info: &liquidity_model::LiquidityInfo,
        remaining_quantity: &mut f64,
        sequence: &mut u32,
    ) -> Result<Vec<SimulationExecution>> {
        let mut executions = Vec::new();
        
        if let Some(limit_price) = order.price {
            // 检查价格是否可以立即执行
            let can_execute = match order.side {
                OrderSide::Buy => {
                    if let Some(ask_price) = market_data.best_ask() {
                        limit_price >= ask_price
                    } else {
                        false
                    }
                }
                OrderSide::Sell => {
                    if let Some(bid_price) = market_data.best_bid() {
                        limit_price <= bid_price
                    } else {
                        false
                    }
                }
            };
            
            if can_execute {
                // 立即执行
                let executable_quantity = if self.config.enable_partial_fills {
                    self.calculate_executable_quantity(
                        *remaining_quantity,
                        market_data,
                        &order.side,
                    )?
                } else {
                    *remaining_quantity
                };
                
                let execution = self.create_execution(
                    order,
                    executable_quantity,
                    limit_price,
                    market_data,
                    &slippage_model::SlippageInfo::default(),
                    *sequence,
                )?;
                
                executions.push(execution);
                *remaining_quantity -= executable_quantity;
            } else {
                // 排队等待
                if self.config.enable_queue_position {
                    let queue_execution = self.simulate_queue_execution(
                        order,
                        market_data,
                        liquidity_info,
                        remaining_quantity,
                        sequence,
                    )?;
                    
                    if let Some(exec) = queue_execution {
                        executions.push(exec);
                    }
                }
            }
        }
        
        Ok(executions)
    }

    /// 仿真冰山单执行
    fn simulate_iceberg_order(
        &self,
        order: &SimulationOrder,
        market_data: &MarketData,
        market_state: &market_simulation::MarketState,
        liquidity_info: &liquidity_model::LiquidityInfo,
        remaining_quantity: &mut f64,
        sequence: &mut u32,
    ) -> Result<Vec<SimulationExecution>> {
        let mut executions = Vec::new();
        
        let display_quantity = order.display_quantity.unwrap_or(order.quantity * 0.1);
        
        while *remaining_quantity > 0.0 {
            let current_display = display_quantity.min(*remaining_quantity);
            
            // 创建显示部分的订单
            let mut display_order = order.clone();
            display_order.quantity = current_display;
            
            let display_executions = self.simulate_limit_order(
                &display_order,
                market_data,
                market_state,
                liquidity_info,
                &mut *remaining_quantity,
                sequence,
            )?;
            
            executions.extend(display_executions);
            
            // 如果没有成交或者部分成交，等待一段时间
            if *remaining_quantity > current_display * 0.9 {
                break; // 简化：不继续等待
            }
        }
        
        Ok(executions)
    }

    /// 仿真队列执行
    fn simulate_queue_execution(
        &self,
        order: &SimulationOrder,
        market_data: &MarketData,
        liquidity_info: &liquidity_model::LiquidityInfo,
        remaining_quantity: &mut f64,
        sequence: &mut u32,
    ) -> Result<Option<SimulationExecution>> {
        // 简化的队列模拟
        let queue_position = self.estimate_queue_position(order, market_data)?;
        let wait_time = self.estimate_queue_wait_time(queue_position, liquidity_info)?;
        
        // 如果等待时间过长，可能不会成交
        if wait_time > 30000 { // 30秒
            return Ok(None);
        }
        
        // 模拟部分成交
        let fill_probability = self.calculate_fill_probability(queue_position, liquidity_info)?;
        if rand::random::<f64>() < fill_probability {
            let execution_price = order.price.unwrap_or(0.0);
            let fill_quantity = *remaining_quantity * fill_probability;
            
            let mut execution = self.create_execution(
                order,
                fill_quantity,
                execution_price,
                market_data,
                &slippage_model::SlippageInfo::default(),
                *sequence,
            )?;
            
            // 设置队列信息
            execution.simulation_metadata.queue_position = Some(queue_position);
            execution.simulation_metadata.queue_wait_time_ms = Some(wait_time);
            
            *remaining_quantity -= fill_quantity;
            *sequence += 1;
            
            Ok(Some(execution))
        } else {
            Ok(None)
        }
    }

    /// 创建执行记录
    fn create_execution(
        &self,
        order: &SimulationOrder,
        quantity: f64,
        price: f64,
        market_data: &MarketData,
        slippage_info: &slippage_model::SlippageInfo,
        sequence: u32,
    ) -> Result<SimulationExecution> {
        let execution_id = format!("{}_{}", order.id, sequence);
        
        // 计算执行质量指标
        let execution_quality = self.calculate_execution_quality(
            order, quantity, price, market_data, slippage_info,
        )?;
        
        // 计算手续费
        let commission = self.calculate_commission(quantity, price, &order.venue)?;
        
        // 确定流动性标志
        let liquidity_flag = self.determine_liquidity_flag(order, price, market_data)?;
        
        Ok(SimulationExecution {
            order_id: order.id.clone(),
            execution_id,
            symbol: order.symbol.clone(),
            side: order.side.clone(),
            quantity,
            price,
            commission,
            venue: order.venue.clone(),
            timestamp: Utc::now(),
            liquidity_flag,
            execution_quality,
            constraint_violations: Vec::new(),
            simulation_metadata: SimulationMetadata {
                queue_position: None,
                queue_wait_time_ms: None,
                partial_fills: Vec::new(),
                rejected_quantity: 0.0,
                latency_ms: self.simulate_latency()?,
                venue_response: VenueResponse::Accepted,
            },
        })
    }

    // 辅助方法实现
    fn calculate_executable_quantity(
        &self,
        requested_quantity: f64,
        market_data: &MarketData,
        side: &OrderSide,
    ) -> Result<f64> {
        // 根据市场深度计算可执行数量
        let available_liquidity = match side {
            OrderSide::Buy => market_data.ask_size_at_best(),
            OrderSide::Sell => market_data.bid_size_at_best(),
        };
        
        Ok(requested_quantity.min(available_liquidity.unwrap_or(requested_quantity)))
    }

    fn estimate_queue_position(&self, order: &SimulationOrder, market_data: &MarketData) -> Result<u32> {
        // 简化的队列位置估计
        Ok(rand::random::<u32>() % 100 + 1)
    }

    fn estimate_queue_wait_time(&self, queue_position: u32, liquidity_info: &liquidity_model::LiquidityInfo) -> Result<u64> {
        // 基于队列位置和流动性的等待时间估计
        let base_time = queue_position as u64 * 100; // 每个位置100ms
        let liquidity_factor = 1.0 / liquidity_info.liquidity_score.max(0.1);
        
        Ok((base_time as f64 * liquidity_factor) as u64)
    }

    fn calculate_fill_probability(&self, queue_position: u32, liquidity_info: &liquidity_model::LiquidityInfo) -> Result<f64> {
        // 基于队列位置和流动性的成交概率
        let base_prob = 1.0 / (1.0 + queue_position as f64 * 0.1);
        let liquidity_boost = liquidity_info.liquidity_score;
        
        Ok((base_prob * liquidity_boost).min(1.0))
    }

    fn calculate_execution_quality(
        &self,
        order: &SimulationOrder,
        quantity: f64,
        price: f64,
        market_data: &MarketData,
        slippage_info: &slippage_model::SlippageInfo,
    ) -> Result<ExecutionQuality> {
        let mid_price = market_data.mid_price().unwrap_or(price);
        
        // 有效滑点
        let effective_slippage = ((price - mid_price) / mid_price * 10000.0).abs(); // 基点
        
        // 实施缺口
        let implementation_shortfall = slippage_info.total_slippage / mid_price * 10000.0;
        
        // 成交比率
        let fill_ratio = quantity / order.quantity;
        
        Ok(ExecutionQuality {
            effective_slippage,
            implementation_shortfall,
            market_impact: slippage_info.market_impact,
            timing_cost: slippage_info.timing_cost,
            opportunity_cost: (1.0 - fill_ratio) * 10.0, // 简化计算
            total_cost_bps: effective_slippage + implementation_shortfall,
            fill_ratio,
            speed_of_execution: 1.0 / (1.0 + slippage_info.delay_cost),
        })
    }

    fn calculate_commission(&self, quantity: f64, price: f64, venue: &str) -> Result<f64> {
        // 简化的手续费计算
        let commission_rate = match venue {
            "NYSE" => 0.0003,
            "NASDAQ" => 0.0005,
            "BINANCE" => 0.001,
            _ => 0.0005,
        };
        
        Ok(quantity * price * commission_rate)
    }

    fn determine_liquidity_flag(
        &self,
        order: &SimulationOrder,
        price: f64,
        market_data: &MarketData,
    ) -> Result<LiquidityFlag> {
        // 简化的流动性判断
        let mid_price = market_data.mid_price().unwrap_or(price);
        
        let flag = match order.side {
            OrderSide::Buy => {
                if price < mid_price { LiquidityFlag::Maker } else { LiquidityFlag::Taker }
            }
            OrderSide::Sell => {
                if price > mid_price { LiquidityFlag::Maker } else { LiquidityFlag::Taker }
            }
        };
        
        Ok(flag)
    }

    fn simulate_latency(&self) -> Result<u64> {
        // 简化的延迟模拟
        Ok(rand::random::<u64>() % 50 + 10) // 10-60ms
    }

    fn update_market_impact(
        &mut self, 
        market_data: &mut MarketData, 
        execution: &SimulationExecution,
    ) -> Result<()> {
        // 简化的市场冲击更新
        let impact_factor = execution.quantity / 1000.0; // 每1000股1个基点的冲击
        let price_impact = match execution.side {
            OrderSide::Buy => impact_factor,
            OrderSide::Sell => -impact_factor,
        };
        
        // 更新市场数据（这里需要根据实际MarketData结构实现）
        // market_data.apply_price_impact(price_impact);
        
        Ok(())
    }

    // 创建拒单和流动性受限执行的辅助方法
    fn create_rejected_execution(
        &self, 
        order: SimulationOrder, 
        violations: Vec<ConstraintViolation>,
    ) -> Result<SimulationExecution> {
        let mut execution = self.create_execution(
            &order, 0.0, 0.0, 
            &MarketData::default(), 
            &slippage_model::SlippageInfo::default(), 
            0,
        )?;
        
        execution.constraint_violations = violations;
        execution.simulation_metadata.venue_response = VenueResponse::Rejected("Constraint violation".to_string());
        
        Ok(execution)
    }

    fn create_liquidity_constrained_execution(
        &self, 
        order: SimulationOrder, 
        liquidity_info: liquidity_model::LiquidityInfo,
    ) -> Result<SimulationExecution> {
        let partial_quantity = order.quantity * liquidity_info.available_ratio;
        
        let mut execution = self.create_execution(
            &order, partial_quantity, order.price.unwrap_or(0.0),
            &MarketData::default(), 
            &slippage_model::SlippageInfo::default(), 
            0,
        )?;
        
        execution.simulation_metadata.venue_response = VenueResponse::PartiallyFilled;
        execution.simulation_metadata.rejected_quantity = order.quantity - partial_quantity;
        
        Ok(execution)
    }
}

/// 市场数据结构（简化版）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub bid_prices: Vec<f64>,
    pub bid_sizes: Vec<f64>,
    pub ask_prices: Vec<f64>,
    pub ask_sizes: Vec<f64>,
}

impl MarketData {
    pub fn best_bid(&self) -> Option<f64> {
        self.bid_prices.first().copied()
    }
    
    pub fn best_ask(&self) -> Option<f64> {
        self.ask_prices.first().copied()
    }
    
    pub fn mid_price(&self) -> Option<f64> {
        if let (Some(bid), Some(ask)) = (self.best_bid(), self.best_ask()) {
            Some((bid + ask) / 2.0)
        } else {
            None
        }
    }
    
    pub fn bid_size_at_best(&self) -> Option<f64> {
        self.bid_sizes.first().copied()
    }
    
    pub fn ask_size_at_best(&self) -> Option<f64> {
        self.ask_sizes.first().copied()
    }
}

impl Default for MarketData {
    fn default() -> Self {
        Self {
            symbol: "DEFAULT".to_string(),
            timestamp: Utc::now(),
            bid_prices: vec![100.0],
            bid_sizes: vec![1000.0],
            ask_prices: vec![100.05],
            ask_sizes: vec![1000.0],
        }
    }
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            enable_partial_fills: true,
            enable_queue_position: true,
            enable_min_lot_size: true,
            enable_tick_size: true,
            enable_rate_limits: true,
            enable_rejection_simulation: true,
            enable_latency_simulation: true,
            realistic_matching: true,
            iceberg_detection: true,
            market_impact_decay: 0.95,
            max_simulation_time_ms: 1000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_engine_creation() {
        let config = SimulationConfig::default();
        let engine = RealisticSimulationEngine::new(config);
        // 基本创建测试通过
    }

    #[test] 
    fn test_market_data() {
        let market_data = MarketData::default();
        assert_eq!(market_data.best_bid(), Some(100.0));
        assert_eq!(market_data.best_ask(), Some(100.05));
        assert_eq!(market_data.mid_price(), Some(100.025));
    }
}