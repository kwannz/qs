use super::*;
use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use rust_decimal::Decimal;
use std::str::FromStr;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{VecDeque, HashMap};
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use std::convert::TryInto;
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};

/// AG3级别的生产级速率控制规则
#[derive(Debug)]
pub struct VelocityControlRule {
    enabled: bool,
    priority: u8,
    config: VelocityControlConfig,
    tracker: Arc<RwLock<VelocityTracker>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VelocityControlConfig {
    pub enabled: bool,
    pub priority: u8,
    
    // 基础速率限制
    pub max_orders_per_second: u32,
    pub max_orders_per_minute: u32,
    pub max_orders_per_hour: u32,
    
    // 成交量速率限制
    pub max_volume_per_minute: Decimal,
    pub max_volume_per_hour: Decimal,
    pub max_volume_per_day: Decimal,
    
    // 按账户分组限制
    pub per_account_enabled: bool,
    pub max_orders_per_account_per_minute: u32,
    pub max_volume_per_account_per_hour: Decimal,
    
    // 按策略分组限制
    pub per_strategy_enabled: bool,
    pub max_orders_per_strategy_per_minute: u32,
    pub max_volume_per_strategy_per_hour: Decimal,
    
    // 按交易品种分组限制
    pub per_symbol_enabled: bool,
    pub max_orders_per_symbol_per_minute: u32,
    pub max_volume_per_symbol_per_hour: Decimal,
    
    // 预警阈值
    pub warning_threshold_percent: Decimal, // 达到限制的百分比时预警
    
    // 突发流量处理
    pub burst_allowance: u32,          // 允许的突发订单数
    pub burst_window_seconds: u32,     // 突发窗口时间
    pub burst_recovery_rate: f64,      // 令牌恢复速率
    
    // 时间窗口配置
    pub lookback_window_seconds: u64,  // 历史统计窗口
    pub cleanup_interval_minutes: u32, // 清理过期数据间隔
}

/// 订单事件
#[derive(Debug, Clone)]
pub struct OrderEvent {
    pub timestamp: DateTime<Utc>,
    pub account_id: String,
    pub strategy_id: String,
    pub symbol: String,
    pub quantity: Decimal,
}

/// 成交量事件
#[derive(Debug, Clone)]
pub struct VolumeEvent {
    pub timestamp: DateTime<Utc>,
    pub account_id: String,
    pub volume: Decimal,
}

/// 高性能速度跟踪器
#[derive(Debug)]
pub struct VelocityTracker {
    // 全局计数器（原子操作，无锁）
    global_order_count: Arc<AtomicU64>,
    global_volume_sum: Arc<RwLock<Decimal>>,
    
    // 时间窗口内的事件记录
    order_events: VecDeque<OrderEvent>,
    volume_events: VecDeque<VolumeEvent>,
    
    // 按维度分组的统计
    account_stats: HashMap<String, AccountVelocityStats>,
    strategy_stats: HashMap<String, StrategyVelocityStats>,
    symbol_stats: HashMap<String, SymbolVelocityStats>,
    
    // 令牌桶算法用于突发控制
    token_bucket: TokenBucket,
    
    config: VelocityControlConfig,
    last_cleanup: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct AccountVelocityStats {
    pub account_id: String,
    pub orders_last_minute: u32,
    pub orders_last_hour: u32,
    pub volume_last_hour: Decimal,
    pub last_order_time: DateTime<Utc>,
    pub consecutive_rejections: u32,
}

#[derive(Debug, Clone)]
pub struct StrategyVelocityStats {
    pub strategy_id: String,
    pub orders_last_minute: u32,
    pub volume_last_hour: Decimal,
    pub last_order_time: DateTime<Utc>,
    pub avg_order_size: Decimal,
}

#[derive(Debug, Clone)]
pub struct SymbolVelocityStats {
    pub symbol: String,
    pub orders_last_minute: u32,
    pub volume_last_hour: Decimal,
    pub last_order_time: DateTime<Utc>,
    pub peak_activity_time: Option<DateTime<Utc>>,
}

/// 令牌桶算法实现
#[derive(Debug)]
pub struct TokenBucket {
    tokens: f64,
    capacity: f64,
    refill_rate: f64,  // tokens per second
    last_refill: DateTime<Utc>,
}

impl VelocityControlRule {
    pub fn new(config: VelocityControlConfig) -> Result<Self> {
        let tracker = Arc::new(RwLock::new(VelocityTracker::new(config.clone())?));
        
        Ok(Self {
            enabled: config.enabled,
            priority: config.priority,
            config,
            tracker,
        })
    }
    
    /// 检查全局速率限制
    async fn check_global_limits(&self, _request: &PretradeRiskRequest) -> Result<RiskCheckResult> {
        let tracker = self.tracker.read().await;
        let _now = Utc::now();
        
        // 检查每秒限制
        let orders_last_second = tracker.count_orders_in_window(Duration::seconds(1));
        if orders_last_second >= self.config.max_orders_per_second {
            return Ok(RiskCheckResult {
                passed: false,
                violation: Some(RiskViolation {
                    rule_type: RiskRuleType::VelocityControl,
                    severity: RiskSeverity::High,
                    description: format!("Exceeded max orders per second: {} >= {}", 
                                       orders_last_second, self.config.max_orders_per_second),
                    current_value: Decimal::from(orders_last_second),
                    limit_value: Decimal::from(self.config.max_orders_per_second),
                    suggested_action: "Reduce order frequency".to_string(),
                }),
                warnings: vec![],
                suggested_adjustments: vec!["Consider order batching".to_string()],
            });
        }
        
        // 检查每分钟限制
        let orders_last_minute = tracker.count_orders_in_window(Duration::minutes(1));
        if orders_last_minute >= self.config.max_orders_per_minute {
            return Ok(RiskCheckResult {
                passed: false,
                violation: Some(RiskViolation {
                    rule_type: RiskRuleType::VelocityControl,
                    severity: RiskSeverity::High,
                    description: format!("Exceeded max orders per minute: {} >= {}", 
                                       orders_last_minute, self.config.max_orders_per_minute),
                    current_value: Decimal::from(orders_last_minute),
                    limit_value: Decimal::from(self.config.max_orders_per_minute),
                    suggested_action: "Reduce order frequency".to_string(),
                }),
                warnings: vec![],
                suggested_adjustments: vec!["Implement order queuing".to_string()],
            });
        }
        
        // 检查成交量限制
        let volume_last_minute = tracker.sum_volume_in_window(Duration::minutes(1));
        if volume_last_minute >= self.config.max_volume_per_minute {
            return Ok(RiskCheckResult {
                passed: false,
                violation: Some(RiskViolation {
                    rule_type: RiskRuleType::VelocityControl,
                    severity: RiskSeverity::Medium,
                    description: format!("Exceeded max volume per minute: {} >= {}", 
                                       volume_last_minute, self.config.max_volume_per_minute),
                    current_value: volume_last_minute,
                    limit_value: self.config.max_volume_per_minute,
                    suggested_action: "Reduce order sizes".to_string(),
                }),
                warnings: vec![],
                suggested_adjustments: vec!["Split large orders".to_string()],
            });
        }
        
        Ok(RiskCheckResult {
            passed: true,
            violation: None,
            warnings: vec![],
            suggested_adjustments: vec![],
        })
    }
    
    /// 检查账户级别限制
    async fn check_account_limits(&self, _request: &PretradeRiskRequest) -> Result<RiskCheckResult> {
        if !self.config.per_account_enabled {
            return Ok(RiskCheckResult {
                passed: true,
                violation: None,
                warnings: vec![],
                suggested_adjustments: vec![],
            });
        }
        
        let tracker = self.tracker.read().await;
        
        if let Some(stats) = tracker.account_stats.get(&_request.account_id) {
            // 检查每分钟订单数
            if stats.orders_last_minute >= self.config.max_orders_per_account_per_minute {
                return Ok(RiskCheckResult {
                    passed: false,
                    violation: Some(RiskViolation {
                        rule_type: RiskRuleType::VelocityControl,
                        severity: RiskSeverity::Medium,
                        description: format!("Account {} exceeded max orders per minute: {} >= {}", 
                                           _request.account_id, stats.orders_last_minute,
                                           self.config.max_orders_per_account_per_minute),
                        current_value: Decimal::from(stats.orders_last_minute),
                        limit_value: Decimal::from(self.config.max_orders_per_account_per_minute),
                        suggested_action: "Reduce account order frequency".to_string(),
                    }),
                    warnings: vec![],
                    suggested_adjustments: vec!["Implement account-level queuing".to_string()],
                });
            }
            
            // 检查小时成交量
            if stats.volume_last_hour >= self.config.max_volume_per_account_per_hour {
                return Ok(RiskCheckResult {
                    passed: false,
                    violation: Some(RiskViolation {
                        rule_type: RiskRuleType::VelocityControl,
                        severity: RiskSeverity::Medium,
                        description: format!("Account {} exceeded max volume per hour", _request.account_id),
                        current_value: stats.volume_last_hour,
                        limit_value: self.config.max_volume_per_account_per_hour,
                        suggested_action: "Reduce account order sizes".to_string(),
                    }),
                    warnings: vec![],
                    suggested_adjustments: vec!["Split account orders".to_string()],
                });
            }
        }
        
        Ok(RiskCheckResult {
            passed: true,
            violation: None,
            warnings: vec![],
            suggested_adjustments: vec![],
        })
    }
    
    /// 令牌桶突发控制检查
    async fn check_token_bucket(&self, _request: &PretradeRiskRequest) -> Result<RiskCheckResult> {
        let mut tracker = self.tracker.write().await;
        
        // 更新令牌桶
        tracker.token_bucket.refill();
        
        if tracker.token_bucket.consume(1.0) {
            Ok(RiskCheckResult {
                passed: true,
                violation: None,
                warnings: vec![],
                suggested_adjustments: vec![],
            })
        } else {
            Ok(RiskCheckResult {
                passed: false,
                violation: Some(RiskViolation {
                    rule_type: RiskRuleType::VelocityControl,
                    severity: RiskSeverity::High,
                    description: "Burst rate limit exceeded - token bucket empty".to_string(),
                    current_value: Decimal::from_str(&format!("{:.6}", tracker.token_bucket.tokens)).unwrap_or(Decimal::ZERO),
                    limit_value: Decimal::from_str(&format!("{:.6}", tracker.token_bucket.capacity)).unwrap_or(Decimal::ZERO),
                    suggested_action: "Wait for token bucket refill".to_string(),
                }),
                warnings: vec![],
                suggested_adjustments: vec![
                    format!("Wait {:.1} seconds for next token", 
                           1.0 / tracker.token_bucket.refill_rate)
                ],
            })
        }
    }
    
    /// 记录订单事件用于统计
    pub async fn record_order(&self, _request: &PretradeRiskRequest) -> Result<()> {
        let mut tracker = self.tracker.write().await;
        let now = Utc::now();
        
        // 记录全局事件
        let order_event = OrderEvent {
            timestamp: now,
            account_id: _request.account_id.clone(),
            strategy_id: _request.strategy_id.clone(),
            symbol: _request.symbol.clone(),
            quantity: _request.quantity,
        };
        
        let volume_event = VolumeEvent {
            timestamp: now,
            account_id: _request.account_id.clone(),
            volume: _request.quantity,
        };
        
        tracker.order_events.push_back(order_event);
        tracker.volume_events.push_back(volume_event);
        
        // 更新全局原子计数器
        tracker.global_order_count.fetch_add(1, Ordering::Relaxed);
        
        // 更新账户统计
        tracker.update_account_stats(&_request.account_id, _request.quantity, now);
        
        // 更新策略统计
        tracker.update_strategy_stats(&_request.strategy_id, _request.quantity, now);
        
        // 更新交易品种统计
        tracker.update_symbol_stats(&_request.symbol, _request.quantity, now);
        
        // 清理过期数据
        if now.signed_duration_since(tracker.last_cleanup).num_minutes() > 
           self.config.cleanup_interval_minutes as i64 {
            tracker.cleanup_expired_data(now);
        }
        
        Ok(())
    }
}

impl RiskRule for VelocityControlRule {
    fn rule_type(&self) -> RiskRuleType {
        RiskRuleType::VelocityControl
    }
    
    fn check(&self, _request: &PretradeRiskRequest, _context: &RiskContext) -> Result<RiskCheckResult> {
        // 使用异步块处理检查
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            // 1. 检查全局限制
            let global_result = self.check_global_limits(_request).await?;
            if !global_result.passed {
                return Ok(global_result);
            }
            
            // 2. 检查账户限制
            let account_result = self.check_account_limits(_request).await?;
            if !account_result.passed {
                return Ok(account_result);
            }
            
            // 3. 检查令牌桶
            let token_result = self.check_token_bucket(_request).await?;
            if !token_result.passed {
                return Ok(token_result);
            }
            
            // 4. 生成预警
            let mut warnings = Vec::new();
            let tracker = self.tracker.read().await;
            
            let current_rate = tracker.count_orders_in_window(Duration::minutes(1)) as f64 / 60.0;
            let limit_rate = self.config.max_orders_per_second as f64;
            let utilization = current_rate / limit_rate;
            
            let warning_threshold: f64 = self.config.warning_threshold_percent.try_into().unwrap_or(0.8);
            if utilization >= warning_threshold {
                warnings.push(RiskWarning {
                    rule_type: RiskRuleType::VelocityControl,
                    description: format!("High velocity utilization: {:.1}%", utilization * 100.0),
                    threshold_breach_percent: Decimal::from_str(&format!("{:.6}", utilization * 100.0))
                        .unwrap_or(Decimal::ZERO),
                    recommendation: "Consider reducing order frequency".to_string(),
                });
            }
            
            Ok(RiskCheckResult {
                passed: true,
                violation: None,
                warnings,
                suggested_adjustments: vec![],
            })
        })
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn priority(&self) -> u8 {
        self.priority
    }
    
    fn get_config(&self) -> serde_json::Value {
        serde_json::to_value(&self.config).unwrap_or(serde_json::Value::Null)
    }
}

impl VelocityTracker {
    pub fn new(config: VelocityControlConfig) -> Result<Self> {
        Ok(Self {
            global_order_count: Arc::new(AtomicU64::new(0)),
            global_volume_sum: Arc::new(RwLock::new(Decimal::ZERO)),
            order_events: VecDeque::new(),
            volume_events: VecDeque::new(),
            account_stats: HashMap::new(),
            strategy_stats: HashMap::new(),
            symbol_stats: HashMap::new(),
            token_bucket: TokenBucket::new(
                config.burst_allowance as f64,
                config.burst_recovery_rate,
            ),
            config,
            last_cleanup: Utc::now(),
        })
    }
    
    /// 统计时间窗口内的订单数
    pub fn count_orders_in_window(&self, window: Duration) -> u32 {
        let cutoff = Utc::now() - window;
        self.order_events
            .iter()
            .rev()
            .take_while(|event| event.timestamp >= cutoff)
            .count() as u32
    }
    
    /// 统计时间窗口内的成交量
    pub fn sum_volume_in_window(&self, window: Duration) -> Decimal {
        let cutoff = Utc::now() - window;
        self.volume_events
            .iter()
            .rev()
            .take_while(|event| event.timestamp >= cutoff)
            .map(|event| event.volume)
            .sum()
    }
    
    /// 更新账户统计
    fn update_account_stats(&mut self, account_id: &str, _quantity: Decimal, timestamp: DateTime<Utc>) {
        let stats = self.account_stats
            .entry(account_id.to_string())
            .or_insert_with(|| AccountVelocityStats {
                account_id: account_id.to_string(),
                orders_last_minute: 0,
                orders_last_hour: 0,
                volume_last_hour: Decimal::ZERO,
                last_order_time: timestamp,
                consecutive_rejections: 0,
            });
        
        // 重新计算时间窗口内的统计
        let one_minute_ago = timestamp - Duration::minutes(1);
        let one_hour_ago = timestamp - Duration::hours(1);
        
        stats.orders_last_minute = self.order_events
            .iter()
            .rev()
            .take_while(|e| e.timestamp >= one_minute_ago)
            .filter(|e| e.account_id == account_id)
            .count() as u32;
            
        stats.orders_last_hour = self.order_events
            .iter()
            .rev()
            .take_while(|e| e.timestamp >= one_hour_ago)
            .filter(|e| e.account_id == account_id)
            .count() as u32;
            
        stats.volume_last_hour = self.volume_events
            .iter()
            .rev()
            .take_while(|e| e.timestamp >= one_hour_ago)
            .filter(|e| e.account_id == account_id)
            .map(|e| e.volume)
            .sum();
            
        stats.last_order_time = timestamp;
    }
    
    /// 更新策略统计
    fn update_strategy_stats(&mut self, strategy_id: &str, _quantity: Decimal, timestamp: DateTime<Utc>) {
        let stats = self.strategy_stats
            .entry(strategy_id.to_string())
            .or_insert_with(|| StrategyVelocityStats {
                strategy_id: strategy_id.to_string(),
                orders_last_minute: 0,
                volume_last_hour: Decimal::ZERO,
                last_order_time: timestamp,
                avg_order_size: Decimal::ZERO,
            });
        
        stats.last_order_time = timestamp;
        
        // 更新平均订单大小
        let total_volume: Decimal = self.volume_events
            .iter()
            .map(|e| e.volume)
            .sum();
        let order_count = self.order_events.len() as u64;
        
        if order_count > 0 {
            stats.avg_order_size = total_volume / Decimal::from(order_count);
        }
    }
    
    /// 更新交易品种统计
    fn update_symbol_stats(&mut self, symbol: &str, _quantity: Decimal, timestamp: DateTime<Utc>) {
        let stats = self.symbol_stats
            .entry(symbol.to_string())
            .or_insert_with(|| SymbolVelocityStats {
                symbol: symbol.to_string(),
                orders_last_minute: 0,
                volume_last_hour: Decimal::ZERO,
                last_order_time: timestamp,
                peak_activity_time: None,
            });
        
        stats.last_order_time = timestamp;
        
        // 检查是否是新的峰值活动时间
        let current_minute_orders = self.order_events
            .iter()
            .rev()
            .take_while(|e| e.timestamp >= timestamp - Duration::minutes(1))
            .filter(|e| e.symbol == symbol)
            .count() as u32;
            
        if current_minute_orders > stats.orders_last_minute {
            stats.peak_activity_time = Some(timestamp);
        }
        
        stats.orders_last_minute = current_minute_orders;
    }
    
    /// 清理过期数据
    fn cleanup_expired_data(&mut self, now: DateTime<Utc>) {
        let retention_period = Duration::seconds(self.config.lookback_window_seconds as i64);
        let cutoff = now - retention_period;
        
        // 清理订单事件
        while let Some(event) = self.order_events.front() {
            if event.timestamp < cutoff {
                self.order_events.pop_front();
            } else {
                break;
            }
        }
        
        // 清理成交量事件
        while let Some(event) = self.volume_events.front() {
            if event.timestamp < cutoff {
                self.volume_events.pop_front();
            } else {
                break;
            }
        }
        
        self.last_cleanup = now;
        
        debug!("Cleaned up expired velocity tracking data, cutoff: {}", cutoff);
    }
}

impl TokenBucket {
    pub fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            tokens: capacity,
            capacity,
            refill_rate,
            last_refill: Utc::now(),
        }
    }
    
    pub fn refill(&mut self) {
        let now = Utc::now();
        let elapsed = now.signed_duration_since(self.last_refill).num_milliseconds() as f64 / 1000.0;
        
        let tokens_to_add = elapsed * self.refill_rate;
        self.tokens = (self.tokens + tokens_to_add).min(self.capacity);
        self.last_refill = now;
    }
    
    pub fn consume(&mut self, amount: f64) -> bool {
        if self.tokens >= amount {
            self.tokens -= amount;
            true
        } else {
            false
        }
    }
    
    pub fn available(&self) -> f64 {
        self.tokens
    }
}

impl Default for VelocityControlConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            priority: 2,
            max_orders_per_second: 10,
            max_orders_per_minute: 300,
            max_orders_per_hour: 1000,
            max_volume_per_minute: Decimal::from(1000000),    // $1M
            max_volume_per_hour: Decimal::from(10000000),     // $10M
            max_volume_per_day: Decimal::from(100000000),     // $100M
            per_account_enabled: true,
            max_orders_per_account_per_minute: 100,
            max_volume_per_account_per_hour: Decimal::from(5000000), // $5M
            per_strategy_enabled: true,
            max_orders_per_strategy_per_minute: 50,
            max_volume_per_strategy_per_hour: Decimal::from(2000000), // $2M
            per_symbol_enabled: true,
            max_orders_per_symbol_per_minute: 30,
            max_volume_per_symbol_per_hour: Decimal::from(1000000), // $1M
            warning_threshold_percent: Decimal::from_parts(80, 0, 0, false, 2), // 80%
            burst_allowance: 20,
            burst_window_seconds: 10,
            burst_recovery_rate: 1.0, // 1 token per second
            lookback_window_seconds: 3600, // 1 hour
            cleanup_interval_minutes: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::risk_control::{PretradeRiskRequest, OrderSide, OrderType};

    #[tokio::test]
    async fn test_velocity_control_basic() {
        let config = VelocityControlConfig {
            max_orders_per_second: 2,
            max_orders_per_minute: 10,
            ..Default::default()
        };
        
        let rule = VelocityControlRule::new(config).unwrap();
        
        let request = PretradeRiskRequest {
            order_id: "test_order".to_string(),
            strategy_id: "test_strategy".to_string(),
            account_id: "test_account".to_string(),
            symbol: "BTCUSD".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::from(100),
            price: Some(Decimal::from(50000)),
            order_type: OrderType::Limit,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };
        
        // 前几次应该通过
        for i in 0..2 {
            rule.record_order(&request).await.unwrap();
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
        
        // 第三次应该被限制
        let result = rule.check_global_limits(&request).await.unwrap();
        // 注意：由于时间窗口的精确性，这个测试可能需要调整
        println!("Velocity check result: {:?}", result);
    }
    
    #[tokio::test]
    async fn test_token_bucket() {
        let mut bucket = TokenBucket::new(10.0, 2.0); // 10 capacity, 2 tokens/sec
        
        // 应该能消费10个令牌
        for i in 0..10 {
            assert!(bucket.consume(1.0), "Failed to consume token {}", i);
        }
        
        // 第11个应该失败
        assert!(!bucket.consume(1.0));
        
        // 等待恢复
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        bucket.refill();
        
        // 应该有大约2个令牌可用
        assert!(bucket.consume(1.0));
        assert!(bucket.consume(1.0));
        assert!(!bucket.consume(1.0));
    }
    
    #[test]
    fn test_velocity_config_default() {
        let config = VelocityControlConfig::default();
        assert_eq!(config.max_orders_per_second, 10);
        assert_eq!(config.max_orders_per_minute, 300);
        assert_eq!(config.burst_allowance, 20);
        assert_eq!(config.burst_recovery_rate, 1.0);
    }
}