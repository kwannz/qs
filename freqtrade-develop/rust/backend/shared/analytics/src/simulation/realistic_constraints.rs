use crate::simulation::{SimulationOrder, SimulationConfig, MarketData, ConstraintViolation, ViolationSeverity};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc, Timelike, Datelike};

/// 约束引擎 - AG3真实交易约束模拟
pub struct ConstraintEngine {
    config: SimulationConfig,
    lot_size_constraints: HashMap<String, f64>,
    tick_size_constraints: HashMap<String, f64>,
    rate_limits: RateLimiter,
    venue_constraints: HashMap<String, VenueConstraints>,
    position_limits: PositionLimits,
}

/// 约束检查结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintCheckResult {
    pub passed: bool,
    pub violations: Vec<ConstraintViolation>,
    pub warnings: Vec<ConstraintViolation>,
    pub adjusted_order: Option<SimulationOrder>,
}

/// 场所约束
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VenueConstraints {
    min_order_size: f64,
    max_order_size: f64,
    min_price: f64,
    max_price: f64,
    allowed_order_types: Vec<String>,
    trading_hours: TradingHours,
    circuit_breakers: CircuitBreakers,
}

/// 交易时间约束
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TradingHours {
    open_time: String,    // "09:30:00"
    close_time: String,   // "16:00:00"
    timezone: String,     // "America/New_York"
    trading_days: Vec<u8>, // 1=Monday, 7=Sunday
}

/// 熔断机制
#[derive(Debug, Clone, Serialize, Deserialize)]  
struct CircuitBreakers {
    price_deviation_threshold: f64,  // 价格偏离阈值
    volume_spike_threshold: f64,     // 成交量激增阈值
    halt_duration_seconds: u64,      // 停牌时长
}

/// 频率限制器
#[derive(Debug, Clone)]
struct RateLimiter {
    order_limits: HashMap<String, OrderRateLimit>, // 按symbol限制
    venue_limits: HashMap<String, VenueRateLimit>, // 按venue限制
    global_limit: GlobalRateLimit,                 // 全局限制
}

#[derive(Debug, Clone)]
struct OrderRateLimit {
    max_orders_per_second: u32,
    max_orders_per_minute: u32,  
    max_volume_per_minute: f64,
    recent_orders: Vec<DateTime<Utc>>,
    recent_volume: Vec<(DateTime<Utc>, f64)>,
}

#[derive(Debug, Clone)]
struct VenueRateLimit {
    max_messages_per_second: u32,
    recent_messages: Vec<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
struct GlobalRateLimit {
    max_orders_per_second: u32,
    max_total_volume_per_minute: f64,
    recent_orders: Vec<DateTime<Utc>>,
    recent_total_volume: Vec<(DateTime<Utc>, f64)>,
}

/// 持仓限制
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PositionLimits {
    max_position_per_symbol: HashMap<String, f64>,
    max_total_exposure: f64,
    max_concentration_ratio: f64, // 单个持仓占总资产比例
    sector_limits: HashMap<String, f64>,
}

impl ConstraintEngine {
    pub fn new(config: &SimulationConfig) -> Self {
        Self {
            config: config.clone(),
            lot_size_constraints: Self::init_lot_sizes(),
            tick_size_constraints: Self::init_tick_sizes(),
            rate_limits: RateLimiter::new(),
            venue_constraints: Self::init_venue_constraints(),
            position_limits: PositionLimits::default(),
        }
    }

    /// 检查所有约束
    pub fn check_constraints(
        &mut self, 
        order: &SimulationOrder, 
        market_data: &MarketData,
    ) -> Result<ConstraintCheckResult> {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        let mut adjusted_order = None;

        // 1. 最小成交单位约束
        if self.config.enable_min_lot_size {
            if let Some(violation) = self.check_lot_size_constraint(order)? {
                if violation.severity == ViolationSeverity::Error {
                    violations.push(violation);
                } else {
                    warnings.push(violation);
                    // 尝试调整订单
                    adjusted_order = self.adjust_for_lot_size(order);
                }
            }
        }

        // 2. 价格步长约束
        if self.config.enable_tick_size {
            if let Some(violation) = self.check_tick_size_constraint(order)? {
                if violation.severity == ViolationSeverity::Error {
                    violations.push(violation);
                } else {
                    warnings.push(violation);
                    adjusted_order = self.adjust_for_tick_size(order);
                }
            }
        }

        // 3. 频率限制约束
        if self.config.enable_rate_limits {
            if let Some(violation) = self.check_rate_limits(order)? {
                violations.push(violation);
            }
        }

        // 4. 场所约束检查
        if let Some(violation) = self.check_venue_constraints(order, market_data)? {
            violations.push(violation);
        }

        // 5. 交易时间约束
        if let Some(violation) = self.check_trading_hours(order)? {
            violations.push(violation);
        }

        // 6. 熔断检查
        if let Some(violation) = self.check_circuit_breakers(order, market_data)? {
            violations.push(violation);
        }

        // 7. 持仓限制检查
        if let Some(violation) = self.check_position_limits(order)? {
            violations.push(violation);
        }

        let passed = violations.is_empty();

        Ok(ConstraintCheckResult {
            passed,
            violations,
            warnings,
            adjusted_order,
        })
    }

    /// 检查最小成交单位
    fn check_lot_size_constraint(&self, order: &SimulationOrder) -> Result<Option<ConstraintViolation>> {
        if let Some(&lot_size) = self.lot_size_constraints.get(&order.symbol) {
            let remainder = order.quantity % lot_size;
            if remainder > 1e-8 { // 浮点数精度容差
                return Ok(Some(ConstraintViolation {
                    violation_type: "LOT_SIZE_VIOLATION".to_string(),
                    description: format!(
                        "Order quantity {:.6} is not a multiple of lot size {:.6} for {}",
                        order.quantity, lot_size, order.symbol
                    ),
                    severity: ViolationSeverity::Warning, // 可以调整
                    impact: remainder / order.quantity, // 相对影响
                }));
            }
        }
        Ok(None)
    }

    /// 检查价格步长
    fn check_tick_size_constraint(&self, order: &SimulationOrder) -> Result<Option<ConstraintViolation>> {
        if let Some(price) = order.price {
            if let Some(&tick_size) = self.tick_size_constraints.get(&order.symbol) {
                let remainder = price % tick_size;
                if remainder > 1e-6 { // 价格精度容差
                    return Ok(Some(ConstraintViolation {
                        violation_type: "TICK_SIZE_VIOLATION".to_string(),
                        description: format!(
                            "Order price {:.6} is not aligned with tick size {:.6} for {}",
                            price, tick_size, order.symbol
                        ),
                        severity: ViolationSeverity::Warning,
                        impact: remainder / price,
                    }));
                }
            }
        }
        Ok(None)
    }

    /// 检查频率限制
    fn check_rate_limits(&mut self, order: &SimulationOrder) -> Result<Option<ConstraintViolation>> {
        let now = Utc::now();
        
        // 检查symbol级别的限制
        if let Some(symbol_limit) = self.rate_limits.order_limits.get_mut(&order.symbol) {
            // 清理过期记录
            symbol_limit.recent_orders.retain(|&timestamp| {
                now.signed_duration_since(timestamp).num_seconds() < 60
            });
            symbol_limit.recent_volume.retain(|(timestamp, _)| {
                now.signed_duration_since(*timestamp).num_seconds() < 60
            });

            // 检查订单频率
            let orders_in_last_second = symbol_limit.recent_orders.iter()
                .filter(|&&timestamp| now.signed_duration_since(timestamp).num_seconds() < 1)
                .count() as u32;

            if orders_in_last_second >= symbol_limit.max_orders_per_second {
                return Ok(Some(ConstraintViolation {
                    violation_type: "RATE_LIMIT_ORDERS_PER_SECOND".to_string(),
                    description: format!(
                        "Exceeded max orders per second ({}) for symbol {}",
                        symbol_limit.max_orders_per_second, order.symbol
                    ),
                    severity: ViolationSeverity::Error,
                    impact: 1.0, // 完全阻止
                }));
            }

            // 检查成交量限制
            let volume_in_last_minute: f64 = symbol_limit.recent_volume.iter()
                .map(|(_, volume)| volume)
                .sum();

            if volume_in_last_minute + order.quantity > symbol_limit.max_volume_per_minute {
                return Ok(Some(ConstraintViolation {
                    violation_type: "RATE_LIMIT_VOLUME_PER_MINUTE".to_string(),
                    description: format!(
                        "Exceeded max volume per minute ({:.2}) for symbol {}",
                        symbol_limit.max_volume_per_minute, order.symbol
                    ),
                    severity: ViolationSeverity::Error,
                    impact: 1.0,
                }));
            }

            // 记录当前订单
            symbol_limit.recent_orders.push(now);
            symbol_limit.recent_volume.push((now, order.quantity));
        }

        Ok(None)
    }

    /// 检查场所约束
    fn check_venue_constraints(
        &self, 
        order: &SimulationOrder, 
        market_data: &MarketData,
    ) -> Result<Option<ConstraintViolation>> {
        if let Some(venue_constraints) = self.venue_constraints.get(&order.venue) {
            // 检查订单大小
            if order.quantity < venue_constraints.min_order_size {
                return Ok(Some(ConstraintViolation {
                    violation_type: "VENUE_MIN_SIZE".to_string(),
                    description: format!(
                        "Order size {:.2} below venue minimum {:.2} for {}",
                        order.quantity, venue_constraints.min_order_size, order.venue
                    ),
                    severity: ViolationSeverity::Error,
                    impact: 1.0,
                }));
            }

            if order.quantity > venue_constraints.max_order_size {
                return Ok(Some(ConstraintViolation {
                    violation_type: "VENUE_MAX_SIZE".to_string(),
                    description: format!(
                        "Order size {:.2} exceeds venue maximum {:.2} for {}",
                        order.quantity, venue_constraints.max_order_size, order.venue
                    ),
                    severity: ViolationSeverity::Error,
                    impact: 1.0,
                }));
            }

            // 检查价格范围
            if let Some(price) = order.price {
                if price < venue_constraints.min_price || price > venue_constraints.max_price {
                    return Ok(Some(ConstraintViolation {
                        violation_type: "VENUE_PRICE_RANGE".to_string(),
                        description: format!(
                            "Order price {:.6} outside venue range [{:.6}, {:.6}] for {}",
                            price, venue_constraints.min_price, venue_constraints.max_price, order.venue
                        ),
                        severity: ViolationSeverity::Error,
                        impact: 1.0,
                    }));
                }
            }
        }

        Ok(None)
    }

    /// 检查交易时间
    fn check_trading_hours(&self, order: &SimulationOrder) -> Result<Option<ConstraintViolation>> {
        if let Some(venue_constraints) = self.venue_constraints.get(&order.venue) {
            let trading_hours = &venue_constraints.trading_hours;
            let now = Utc::now();
            
            // 简化的交易时间检查（实际实现需要考虑时区转换）
            let current_hour = now.hour();
            let is_weekend = now.weekday().num_days_from_monday() >= 5;
            
            if is_weekend {
                return Ok(Some(ConstraintViolation {
                    violation_type: "OUTSIDE_TRADING_HOURS".to_string(),
                    description: "Market is closed on weekends".to_string(),
                    severity: ViolationSeverity::Error,
                    impact: 1.0,
                }));
            }

            // 简化的交易时间检查（9:30-16:00 EST）
            if current_hour < 9 || current_hour >= 16 {
                return Ok(Some(ConstraintViolation {
                    violation_type: "OUTSIDE_TRADING_HOURS".to_string(),
                    description: format!("Order submitted outside trading hours at {}", now),
                    severity: ViolationSeverity::Warning,
                    impact: 0.5,
                }));
            }
        }

        Ok(None)
    }

    /// 检查熔断机制
    fn check_circuit_breakers(
        &self, 
        order: &SimulationOrder, 
        market_data: &MarketData,
    ) -> Result<Option<ConstraintViolation>> {
        if let Some(venue_constraints) = self.venue_constraints.get(&order.venue) {
            let circuit_breakers = &venue_constraints.circuit_breakers;
            
            if let (Some(order_price), Some(mid_price)) = (order.price, market_data.mid_price()) {
                let price_deviation = ((order_price - mid_price) / mid_price).abs();
                
                if price_deviation > circuit_breakers.price_deviation_threshold {
                    return Ok(Some(ConstraintViolation {
                        violation_type: "CIRCUIT_BREAKER_PRICE".to_string(),
                        description: format!(
                            "Order price deviates {:.2}% from mid-price, exceeding {:.2}% threshold",
                            price_deviation * 100.0,
                            circuit_breakers.price_deviation_threshold * 100.0
                        ),
                        severity: ViolationSeverity::Critical,
                        impact: 1.0,
                    }));
                }
            }
        }

        Ok(None)
    }

    /// 检查持仓限制
    fn check_position_limits(&self, order: &SimulationOrder) -> Result<Option<ConstraintViolation>> {
        if let Some(&max_position) = self.position_limits.max_position_per_symbol.get(&order.symbol) {
            if order.quantity > max_position {
                return Ok(Some(ConstraintViolation {
                    violation_type: "POSITION_LIMIT".to_string(),
                    description: format!(
                        "Order quantity {:.2} exceeds position limit {:.2} for {}",
                        order.quantity, max_position, order.symbol
                    ),
                    severity: ViolationSeverity::Error,
                    impact: (order.quantity - max_position) / order.quantity,
                }));
            }
        }

        Ok(None)
    }

    // 订单调整方法
    fn adjust_for_lot_size(&self, order: &SimulationOrder) -> Option<SimulationOrder> {
        if let Some(&lot_size) = self.lot_size_constraints.get(&order.symbol) {
            let mut adjusted_order = order.clone();
            adjusted_order.quantity = (order.quantity / lot_size).floor() * lot_size;
            
            if adjusted_order.quantity > 0.0 {
                return Some(adjusted_order);
            }
        }
        None
    }

    fn adjust_for_tick_size(&self, order: &SimulationOrder) -> Option<SimulationOrder> {
        if let Some(price) = order.price {
            if let Some(&tick_size) = self.tick_size_constraints.get(&order.symbol) {
                let mut adjusted_order = order.clone();
                adjusted_order.price = Some((price / tick_size).round() * tick_size);
                return Some(adjusted_order);
            }
        }
        None
    }

    // 初始化方法
    fn init_lot_sizes() -> HashMap<String, f64> {
        let mut lot_sizes = HashMap::new();
        
        // 股票（通常1股为单位）
        lot_sizes.insert("AAPL".to_string(), 1.0);
        lot_sizes.insert("GOOGL".to_string(), 1.0);
        lot_sizes.insert("MSFT".to_string(), 1.0);
        
        // 期货（合约规格）
        lot_sizes.insert("ES".to_string(), 1.0);  // E-mini S&P 500
        lot_sizes.insert("NQ".to_string(), 1.0);  // E-mini NASDAQ
        
        // 加密货币（小数位）
        lot_sizes.insert("BTCUSDT".to_string(), 0.00001);
        lot_sizes.insert("ETHUSDT".to_string(), 0.0001);
        
        lot_sizes
    }

    fn init_tick_sizes() -> HashMap<String, f64> {
        let mut tick_sizes = HashMap::new();
        
        // 股票（1分钱）
        tick_sizes.insert("AAPL".to_string(), 0.01);
        tick_sizes.insert("GOOGL".to_string(), 0.01);
        tick_sizes.insert("MSFT".to_string(), 0.01);
        
        // 期货
        tick_sizes.insert("ES".to_string(), 0.25);
        tick_sizes.insert("NQ".to_string(), 0.25);
        
        // 加密货币
        tick_sizes.insert("BTCUSDT".to_string(), 0.01);
        tick_sizes.insert("ETHUSDT".to_string(), 0.01);
        
        tick_sizes
    }

    fn init_venue_constraints() -> HashMap<String, VenueConstraints> {
        let mut constraints = HashMap::new();

        // NYSE约束
        constraints.insert("NYSE".to_string(), VenueConstraints {
            min_order_size: 1.0,
            max_order_size: 1_000_000.0,
            min_price: 0.01,
            max_price: 100_000.0,
            allowed_order_types: vec![
                "MARKET".to_string(), 
                "LIMIT".to_string(), 
                "STOP".to_string()
            ],
            trading_hours: TradingHours {
                open_time: "09:30:00".to_string(),
                close_time: "16:00:00".to_string(),
                timezone: "America/New_York".to_string(),
                trading_days: vec![1, 2, 3, 4, 5], // Monday-Friday
            },
            circuit_breakers: CircuitBreakers {
                price_deviation_threshold: 0.10, // 10%
                volume_spike_threshold: 5.0,     // 5x normal
                halt_duration_seconds: 300,      // 5 minutes
            },
        });

        // BINANCE约束
        constraints.insert("BINANCE".to_string(), VenueConstraints {
            min_order_size: 0.00001,
            max_order_size: 100_000.0,
            min_price: 0.000001,
            max_price: 1_000_000.0,
            allowed_order_types: vec![
                "MARKET".to_string(), 
                "LIMIT".to_string(), 
                "STOP_LIMIT".to_string(),
                "ICEBERG".to_string(),
            ],
            trading_hours: TradingHours {
                open_time: "00:00:00".to_string(), // 24/7
                close_time: "23:59:59".to_string(),
                timezone: "UTC".to_string(),
                trading_days: vec![1, 2, 3, 4, 5, 6, 7], // All days
            },
            circuit_breakers: CircuitBreakers {
                price_deviation_threshold: 0.20, // 20%
                volume_spike_threshold: 10.0,
                halt_duration_seconds: 60,
            },
        });

        constraints
    }
}

impl RateLimiter {
    fn new() -> Self {
        Self {
            order_limits: HashMap::new(),
            venue_limits: HashMap::new(),
            global_limit: GlobalRateLimit {
                max_orders_per_second: 100,
                max_total_volume_per_minute: 1_000_000.0,
                recent_orders: Vec::new(),
                recent_total_volume: Vec::new(),
            },
        }
    }
}

impl Default for PositionLimits {
    fn default() -> Self {
        let mut max_position_per_symbol = HashMap::new();
        max_position_per_symbol.insert("AAPL".to_string(), 10_000.0);
        max_position_per_symbol.insert("GOOGL".to_string(), 5_000.0);
        max_position_per_symbol.insert("BTCUSDT".to_string(), 100.0);

        let mut sector_limits = HashMap::new();
        sector_limits.insert("TECH".to_string(), 500_000.0);
        sector_limits.insert("FINANCE".to_string(), 300_000.0);

        Self {
            max_position_per_symbol,
            max_total_exposure: 1_000_000.0,
            max_concentration_ratio: 0.20, // 20%
            sector_limits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::{OrderSide, OrderType, TimeInForce};

    fn create_test_order() -> SimulationOrder {
        SimulationOrder {
            id: "test_order_1".to_string(),
            symbol: "AAPL".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            quantity: 100.0,
            price: Some(150.00),
            venue: "NYSE".to_string(),
            strategy_id: "test_strategy".to_string(),
            timestamp: Utc::now(),
            time_in_force: TimeInForce::GTC,
            min_quantity: None,
            display_quantity: None,
            is_iceberg: false,
            parent_order_id: None,
        }
    }

    #[test]
    fn test_lot_size_constraint() {
        let config = SimulationConfig::default();
        let engine = ConstraintEngine::new(&config);
        
        let mut order = create_test_order();
        order.quantity = 100.5; // 不是整数股
        
        let violation = engine.check_lot_size_constraint(&order).unwrap();
        assert!(violation.is_some());
        assert_eq!(violation.unwrap().violation_type, "LOT_SIZE_VIOLATION");
    }

    #[test]
    fn test_tick_size_constraint() {
        let config = SimulationConfig::default();
        let engine = ConstraintEngine::new(&config);
        
        let mut order = create_test_order();
        order.price = Some(150.005); // 不符合0.01步长
        
        let violation = engine.check_tick_size_constraint(&order).unwrap();
        assert!(violation.is_some());
        assert_eq!(violation.unwrap().violation_type, "TICK_SIZE_VIOLATION");
    }

    #[test]
    fn test_order_adjustment() {
        let config = SimulationConfig::default();
        let engine = ConstraintEngine::new(&config);
        
        let mut order = create_test_order();
        order.quantity = 100.7; // 需要调整
        
        let adjusted = engine.adjust_for_lot_size(&order);
        assert!(adjusted.is_some());
        assert_eq!(adjusted.unwrap().quantity, 100.0);
    }
}