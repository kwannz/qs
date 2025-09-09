//! 统一事件总线系统
//! 支持本地事件分发、Redis Pub/Sub、Kafka等多种消息传递机制

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock, broadcast};
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use chrono::{DateTime, Utc};
use anyhow::{Result, Context};
use async_trait::async_trait;
use uuid::Uuid;

/// 事件ID类型
pub type EventId = String;

/// 事件主题类型
pub type EventTopic = String;

/// 事件元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetadata {
    pub id: EventId,
    pub topic: EventTopic,
    pub timestamp: DateTime<Utc>,
    pub source_service: String,
    pub trace_id: Option<String>,
    pub correlation_id: Option<String>,
    pub retry_count: u32,
    pub headers: HashMap<String, String>,
}

impl EventMetadata {
    pub fn new(topic: EventTopic, source_service: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            topic,
            timestamp: Utc::now(),
            source_service,
            trace_id: None,
            correlation_id: None,
            retry_count: 0,
            headers: HashMap::new(),
        }
    }

    pub fn with_trace_id(mut self, trace_id: String) -> Self {
        self.trace_id = Some(trace_id);
        self
    }

    pub fn with_correlation_id(mut self, correlation_id: String) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    pub fn with_header(mut self, key: String, value: String) -> Self {
        self.headers.insert(key, value);
        self
    }
}

/// 泛化事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event<T>
where
    T: Serialize + DeserializeOwned,
{
    pub metadata: EventMetadata,
    pub payload: T,
}

impl<T> Event<T>
where
    T: Serialize + DeserializeOwned,
{
    pub fn new(topic: EventTopic, source_service: String, payload: T) -> Self {
        Self {
            metadata: EventMetadata::new(topic, source_service),
            payload,
        }
    }

    pub fn with_metadata(payload: T, metadata: EventMetadata) -> Self {
        Self { metadata, payload }
    }
}

/// 交易系统事件枚举
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradingEvent {
    // 市场数据事件
    MarketDataUpdate {
        symbol: String,
        price: f64,
        volume: f64,
        timestamp: DateTime<Utc>,
    },
    
    // 订单事件
    OrderSubmitted {
        order_id: String,
        symbol: String,
        side: String,
        quantity: f64,
        price: Option<f64>,
    },
    OrderExecuted {
        order_id: String,
        execution_id: String,
        filled_quantity: f64,
        fill_price: f64,
        remaining_quantity: f64,
    },
    OrderCancelled {
        order_id: String,
        reason: String,
    },
    OrderRejected {
        order_id: String,
        reason: String,
    },
    
    // 信号事件
    SignalGenerated {
        strategy_id: String,
        symbol: String,
        signal_type: String,
        strength: f64,
        confidence: f64,
        alpha: f64,
    },
    
    // 风险事件
    RiskLimitBreached {
        limit_type: String,
        current_value: f64,
        limit_value: f64,
        severity: String,
    },
    
    // 系统事件
    ServiceStarted {
        service_name: String,
        version: String,
    },
    ServiceStopped {
        service_name: String,
        reason: String,
    },
    HealthCheckFailed {
        service_name: String,
        check_name: String,
        error: String,
    },
}

/// 事件处理器接口
#[async_trait]
pub trait EventHandler<T>: Send + Sync
where
    T: Serialize + DeserializeOwned + Send + 'static,
{
    async fn handle(&self, event: Event<T>) -> Result<()>;
    fn name(&self) -> &str;
    fn topics(&self) -> Vec<EventTopic>;
}

/// 事件总线接口
#[async_trait]
pub trait EventBus: Send + Sync {
    async fn publish<T>(&self, event: Event<T>) -> Result<()>
    where
        T: Serialize + DeserializeOwned + Send + 'static;
    
    async fn subscribe<T, H>(&self, handler: Arc<H>) -> Result<()>
    where
        T: Serialize + DeserializeOwned + Send + 'static,
        H: EventHandler<T> + 'static;
    
    async fn unsubscribe(&self, handler_name: &str, topic: &EventTopic) -> Result<()>;
    
    async fn shutdown(&self) -> Result<()>;
}

/// 本地事件总线实现
pub struct LocalEventBus {
    // 每个主题对应多个处理器
    handlers: Arc<RwLock<HashMap<EventTopic, Vec<Arc<dyn LocalEventHandler>>>>>,
    // 广播通道用于通知处理器
    sender: broadcast::Sender<SerializedEvent>,
    _receiver: broadcast::Receiver<SerializedEvent>,
}

/// 序列化的事件（用于跨处理器传递）
#[derive(Debug, Clone)]
struct SerializedEvent {
    pub topic: EventTopic,
    pub data: Vec<u8>,
    pub metadata: EventMetadata,
}

/// 本地事件处理器包装
#[async_trait]
trait LocalEventHandler: Send + Sync {
    async fn handle_serialized(&self, event: SerializedEvent) -> Result<()>;
    fn name(&self) -> &str;
    fn topics(&self) -> Vec<EventTopic>;
}

/// 类型化处理器包装
struct TypedLocalHandler<T, H>
where
    T: Serialize + DeserializeOwned + Send + 'static,
    H: EventHandler<T>,
{
    handler: Arc<H>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, H> TypedLocalHandler<T, H>
where
    T: Serialize + DeserializeOwned + Send + 'static,
    H: EventHandler<T>,
{
    fn new(handler: Arc<H>) -> Self {
        Self {
            handler,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<T, H> LocalEventHandler for TypedLocalHandler<T, H>
where
    T: Serialize + DeserializeOwned + Send + 'static,
    H: EventHandler<T>,
{
    async fn handle_serialized(&self, event: SerializedEvent) -> Result<()> {
        let payload: T = serde_json::from_slice(&event.data)
            .context("Failed to deserialize event payload")?;
        
        let typed_event = Event {
            metadata: event.metadata,
            payload,
        };
        
        self.handler.handle(typed_event).await
    }

    fn name(&self) -> &str {
        self.handler.name()
    }

    fn topics(&self) -> Vec<EventTopic> {
        self.handler.topics()
    }
}

impl LocalEventBus {
    pub fn new(buffer_size: usize) -> Self {
        let (sender, receiver) = broadcast::channel(buffer_size);
        
        let bus = Self {
            handlers: Arc::new(RwLock::new(HashMap::new())),
            sender,
            _receiver: receiver,
        };
        
        // 启动事件分发循环
        bus.start_dispatch_loop();
        bus
    }

    fn start_dispatch_loop(&self) {
        let handlers = Arc::clone(&self.handlers);
        let mut receiver = self.sender.subscribe();

        tokio::spawn(async move {
            loop {
                match receiver.recv().await {
                    Ok(event) => {
                        let handlers_guard = handlers.read().await;
                        
                        if let Some(topic_handlers) = handlers_guard.get(&event.topic) {
                            // 并行处理所有订阅该主题的处理器
                            let futures = topic_handlers.iter().map(|handler| {
                                let event_clone = event.clone();
                                let handler_clone = Arc::clone(handler);
                                
                                tokio::spawn(async move {
                                    if let Err(e) = handler_clone.handle_serialized(event_clone).await {
                                        tracing::error!("Event handler '{}' failed: {}", handler_clone.name(), e);
                                    }
                                })
                            });
                            
                            // 等待所有处理器完成
                            for future in futures {
                                let _ = future.await;
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        tracing::info!("Event bus dispatch loop shutting down");
                        break;
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        tracing::warn!("Event bus receiver lagged, some events may have been dropped");
                    }
                }
            }
        });
    }
}

#[async_trait]
impl EventBus for LocalEventBus {
    async fn publish<T>(&self, event: Event<T>) -> Result<()>
    where
        T: Serialize + DeserializeOwned + Send + 'static,
    {
        let serialized_data = serde_json::to_vec(&event.payload)
            .context("Failed to serialize event payload")?;

        let serialized_event = SerializedEvent {
            topic: event.metadata.topic.clone(),
            data: serialized_data,
            metadata: event.metadata,
        };

        self.sender.send(serialized_event)
            .map_err(|_| anyhow::anyhow!("Failed to send event: no receivers"))?;

        Ok(())
    }

    async fn subscribe<T, H>(&self, handler: Arc<H>) -> Result<()>
    where
        T: Serialize + DeserializeOwned + Send + 'static,
        H: EventHandler<T> + 'static,
    {
        let typed_handler = Arc::new(TypedLocalHandler::new(handler));
        let topics = typed_handler.topics();
        let mut handlers_guard = self.handlers.write().await;

        for topic in topics {
            handlers_guard
                .entry(topic)
                .or_insert_with(Vec::new)
                .push(typed_handler.clone() as Arc<dyn LocalEventHandler>);
        }

        Ok(())
    }

    async fn unsubscribe(&self, handler_name: &str, topic: &EventTopic) -> Result<()> {
        let mut handlers_guard = self.handlers.write().await;
        
        if let Some(topic_handlers) = handlers_guard.get_mut(topic) {
            topic_handlers.retain(|handler| handler.name() != handler_name);
            
            // 如果该主题没有处理器了，移除该主题
            if topic_handlers.is_empty() {
                handlers_guard.remove(topic);
            }
        }

        Ok(())
    }

    async fn shutdown(&self) -> Result<()> {
        // 清空所有处理器
        self.handlers.write().await.clear();
        Ok(())
    }
}

/// 分布式事件总线实现（基于Redis Pub/Sub和Kafka）
pub struct DistributedEventBus {
    local_bus: Arc<LocalEventBus>,
    redis_publisher: Option<Arc<RedisPubSub>>,
    kafka_producer: Option<Arc<dyn crate::data::streaming::StreamProducer>>,
    nats_producer: Option<Arc<dyn crate::data::streaming::StreamProducer>>,    // 新增：NATS/JetStream 生产
    nats_consumer: Option<Arc<dyn crate::data::streaming::StreamConsumer>>,    // 新增：NATS/JetStream 消费
    service_name: String,
}

impl DistributedEventBus {
    pub fn new(
        local_bus: Arc<LocalEventBus>,
        redis_publisher: Option<Arc<RedisPubSub>>,
        kafka_producer: Option<Arc<dyn crate::data::streaming::StreamProducer>>,
        service_name: String,
    ) -> Self {
        Self {
            local_bus,
            redis_publisher,
            kafka_producer,
            nats_producer: None,
            nats_consumer: None,
            service_name,
        }
    }

    /// 构造函数（带 NATS/JetStream）
    pub fn new_with_nats(
        local_bus: Arc<LocalEventBus>,
        redis_publisher: Option<Arc<RedisPubSub>>,
        kafka_producer: Option<Arc<dyn crate::data::streaming::StreamProducer>>,
        nats_producer: Option<Arc<dyn crate::data::streaming::StreamProducer>>,
        nats_consumer: Option<Arc<dyn crate::data::streaming::StreamConsumer>>,
        service_name: String,
    ) -> Self {
        Self {
            local_bus,
            redis_publisher,
            kafka_producer,
            nats_producer,
            nats_consumer,
            service_name,
        }
    }

    /// 构建仅基于 NATS/JetStream 的事件总线（简化）
    pub async fn from_nats(url: &str, stream: &str, subjects: Vec<String>, service_name: String) -> Result<Self> {
        use crate::data::adapters::nats_js::{JetStreamProducer, JetStreamDurableConsumer};

        let local_bus = Arc::new(LocalEventBus::new(10_000));
        let nats_prod = JetStreamProducer::connect(url, stream, subjects).await?;
        // 使用 durable 名称与第一个 subject 建立 durable 消费者（可扩展为多 subject）
        let durable = format!("{}_durable", service_name.replace('-', "_"));
        let subject = format!("{}", "events.>");
        let nats_cons = JetStreamDurableConsumer::connect_durable(url, stream, &durable, &subject).await?;
        Ok(Self::new_with_nats(
            local_bus,
            None,
            None,
            Some(Arc::new(nats_prod)),
            Some(Arc::new(nats_cons)),
            service_name,
        ))
    }
}

#[async_trait]
impl EventBus for DistributedEventBus {
    async fn publish<T>(&self, mut event: Event<T>) -> Result<()>
    where
        T: Serialize + DeserializeOwned + Send + 'static,
    {
        // 设置源服务名称
        event.metadata.source_service = self.service_name.clone();

        // 本地发布
        self.local_bus.publish(event.clone()).await?;

        // Redis发布（用于实时通信）
        if let Some(redis) = &self.redis_publisher {
            redis.publish(&event.metadata.topic, &event).await?;
        }

        // Kafka发布（用于可靠的消息传递）
        if let Some(kafka) = &self.kafka_producer {
            let message = crate::data::streaming::StreamMessage {
                id: Uuid::new_v4(),
                topic: event.metadata.topic.clone(),
                partition: None,
                key: Some(event.metadata.id.clone()),
                payload: serde_json::to_vec(&event)?,
                headers: event.metadata.headers.clone(),
                timestamp: event.metadata.timestamp,
            };
            
            kafka.send(message).await?;
        }

        // NATS/JetStream 发布（用于事件广播/回放）
        if let Some(nats) = &self.nats_producer {
            let message = crate::data::streaming::StreamMessage {
                id: Uuid::new_v4(),
                topic: event.metadata.topic.clone(),
                partition: None,
                key: Some(event.metadata.id.clone()),
                payload: serde_json::to_vec(&event)?,
                headers: event.metadata.headers.clone(),
                timestamp: event.metadata.timestamp,
            };
            nats.send(message).await?;
        }

        Ok(())
    }

    async fn subscribe<T, H>(&self, handler: Arc<H>) -> Result<()>
    where
        T: Serialize + DeserializeOwned + Send + 'static,
        H: EventHandler<T> + 'static,
    {
        // 本地订阅
        self.local_bus.subscribe(handler).await?;

        // 分布式订阅 - NATS/JetStream：将消息桥接回本地总线
        if let Some(nats_consumer) = &self.nats_consumer {
            let topics = handler.topics();
            let local_bus = self.local_bus.clone();
            nats_consumer.subscribe(topics.clone()).await?;
            tokio::spawn(async move {
                loop {
                    match nats_consumer.poll(500).await {
                        Ok(msgs) => {
                            for m in msgs {
                                // 构造 SerializedEvent 并注入本地广播
                                let serialized = SerializedEvent {
                                    topic: m.topic.clone(),
                                    data: m.payload.clone(),
                                    metadata: EventMetadata::new(m.topic.clone(), "nats_bridge".to_string()),
                                };
                                let _ = local_bus.sender.send(serialized);
                            }
                        }
                        Err(e) => {
                            tracing::warn!("NATS consumer poll error: {}", e);
                            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                        }
                    }
                }
            });
        }

        Ok(())
    }

    async fn unsubscribe(&self, handler_name: &str, topic: &EventTopic) -> Result<()> {
        self.local_bus.unsubscribe(handler_name, topic).await
    }

    async fn shutdown(&self) -> Result<()> {
        self.local_bus.shutdown().await
    }
}

/// Redis Pub/Sub（占位：未启用真实依赖时返回 UNAVAILABLE）
pub struct RedisPubSub;

impl RedisPubSub {
    pub async fn new(_connection_string: &str) -> Result<Self> {
        anyhow::bail!("Redis Pub/Sub integration not enabled in this build")
    }

    pub async fn publish<T>(&self, _topic: &str, _event: &Event<T>) -> Result<()>
    where
        T: Serialize + DeserializeOwned,
    {
        anyhow::bail!("Redis Pub/Sub integration not enabled in this build")
    }

    pub async fn subscribe<T, F>(&self, _topics: Vec<String>, _handler: F) -> Result<()>
    where
        T: Serialize + DeserializeOwned,
        F: FnMut(Event<T>) -> Result<()> + Send + 'static,
    {
        anyhow::bail!("Redis Pub/Sub integration not enabled in this build")
    }
}

/// 示例事件处理器：订单状态更新处理器
pub struct OrderStatusHandler {
    name: String,
}

impl OrderStatusHandler {
    pub fn new() -> Self {
        Self {
            name: "OrderStatusHandler".to_string(),
        }
    }
}

#[async_trait]
impl EventHandler<TradingEvent> for OrderStatusHandler {
    async fn handle(&self, event: Event<TradingEvent>) -> Result<()> {
        match &event.payload {
            TradingEvent::OrderSubmitted { order_id, symbol, side, quantity, price } => {
                tracing::info!(
                    "Order submitted: {} {} {} @ {:?} ({})",
                    order_id, side, quantity, price, symbol
                );
            }
            TradingEvent::OrderExecuted { order_id, execution_id, filled_quantity, fill_price, .. } => {
                tracing::info!(
                    "Order executed: {} ({}) - {} @ {}",
                    order_id, execution_id, filled_quantity, fill_price
                );
            }
            TradingEvent::OrderCancelled { order_id, reason } => {
                tracing::info!("Order cancelled: {} - {}", order_id, reason);
            }
            TradingEvent::OrderRejected { order_id, reason } => {
                tracing::warn!("Order rejected: {} - {}", order_id, reason);
            }
            _ => {
                // 忽略其他事件类型
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn topics(&self) -> Vec<EventTopic> {
        vec![
            "trading.order.submitted".to_string(),
            "trading.order.executed".to_string(),
            "trading.order.cancelled".to_string(),
            "trading.order.rejected".to_string(),
        ]
    }
}

/// 示例事件处理器：风险监控处理器
pub struct RiskMonitorHandler {
    name: String,
    alert_threshold: f64,
}

impl RiskMonitorHandler {
    pub fn new(alert_threshold: f64) -> Self {
        Self {
            name: "RiskMonitorHandler".to_string(),
            alert_threshold,
        }
    }
}

#[async_trait]
impl EventHandler<TradingEvent> for RiskMonitorHandler {
    async fn handle(&self, event: Event<TradingEvent>) -> Result<()> {
        match &event.payload {
            TradingEvent::RiskLimitBreached { limit_type, current_value, limit_value, severity } => {
                if current_value / limit_value > self.alert_threshold {
                    tracing::error!(
                        "CRITICAL RISK ALERT: {} exceeded by {:.2}% - Current: {}, Limit: {}, Severity: {}",
                        limit_type,
                        ((current_value / limit_value) - 1.0) * 100.0,
                        current_value,
                        limit_value,
                        severity
                    );
                    
                    // 这里可以触发紧急停止、通知等操作
                    self.trigger_emergency_response(limit_type, *current_value, *limit_value).await?;
                } else {
                    tracing::warn!(
                        "Risk limit breached: {} - Current: {}, Limit: {}, Severity: {}",
                        limit_type, current_value, limit_value, severity
                    );
                }
            }
            _ => {
                // 忽略其他事件类型
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn topics(&self) -> Vec<EventTopic> {
        vec!["trading.risk.limit_breached".to_string()]
    }
}

impl RiskMonitorHandler {
    async fn trigger_emergency_response(&self, limit_type: &str, current: f64, limit: f64) -> Result<()> {
        // 示例紧急响应逻辑
        tracing::error!("Triggering emergency response for {} breach: {} > {}", limit_type, current, limit);
        
        // 可以在这里实现：
        // 1. 停止所有新订单
        // 2. 取消所有pending订单
        // 3. 发送紧急通知
        // 4. 触发风险缓解措施
        
        Ok(())
    }
}

/// 事件总线管理器
pub struct EventBusManager {
    bus: Arc<dyn EventBus>,
    handlers: Arc<RwLock<Vec<Box<dyn std::any::Any + Send + Sync>>>>,
}

impl EventBusManager {
    pub fn new(bus: Arc<dyn EventBus>) -> Self {
        Self {
            bus,
            handlers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn register_handler<T, H>(&self, handler: H) -> Result<()>
    where
        T: Serialize + DeserializeOwned + Send + 'static,
        H: EventHandler<T> + 'static,
    {
        let handler_arc = Arc::new(handler);
        self.bus.subscribe(handler_arc.clone()).await?;
        
        // 保存处理器引用以防止被drop
        let mut handlers_guard = self.handlers.write().await;
        handlers_guard.push(Box::new(handler_arc));
        
        Ok(())
    }

    pub async fn publish<T>(&self, event: Event<T>) -> Result<()>
    where
        T: Serialize + DeserializeOwned + Send + 'static,
    {
        self.bus.publish(event).await
    }

    pub async fn shutdown(&self) -> Result<()> {
        self.bus.shutdown().await?;
        self.handlers.write().await.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TestEvent {
        pub data: String,
        pub value: u64,
    }

    struct TestHandler {
        name: String,
        call_count: Arc<AtomicU64>,
    }

    impl TestHandler {
        fn new(name: String) -> (Self, Arc<AtomicU64>) {
            let counter = Arc::new(AtomicU64::new(0));
            (Self {
                name,
                call_count: counter.clone(),
            }, counter)
        }
    }

    #[async_trait]
    impl EventHandler<TestEvent> for TestHandler {
        async fn handle(&self, event: Event<TestEvent>) -> Result<()> {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            tracing::info!("Handler {} received: {:?}", self.name, event.payload);
            Ok(())
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn topics(&self) -> Vec<EventTopic> {
            vec!["test.event".to_string()]
        }
    }

    #[tokio::test]
    async fn test_local_event_bus() {
        let bus = Arc::new(LocalEventBus::new(1000));
        
        let (handler1, counter1) = TestHandler::new("handler1".to_string());
        let (handler2, counter2) = TestHandler::new("handler2".to_string());
        
        bus.subscribe::<TestEvent, _>(Arc::new(handler1)).await.unwrap();
        bus.subscribe::<TestEvent, _>(Arc::new(handler2)).await.unwrap();
        
        let test_event = Event::new(
            "test.event".to_string(),
            "test_service".to_string(),
            TestEvent {
                data: "Hello World".to_string(),
                value: 42,
            },
        );
        
        bus.publish(test_event).await.unwrap();
        
        // 给事件处理一些时间
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        
        assert_eq!(counter1.load(Ordering::Relaxed), 1);
        assert_eq!(counter2.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_event_metadata() {
        let metadata = EventMetadata::new(
            "test.topic".to_string(),
            "test_service".to_string(),
        )
        .with_trace_id("trace123".to_string())
        .with_correlation_id("corr456".to_string())
        .with_header("custom".to_string(), "value".to_string());

        assert_eq!(metadata.topic, "test.topic");
        assert_eq!(metadata.source_service, "test_service");
        assert_eq!(metadata.trace_id, Some("trace123".to_string()));
        assert_eq!(metadata.correlation_id, Some("corr456".to_string()));
        assert_eq!(metadata.headers.get("custom"), Some(&"value".to_string()));
    }

    #[tokio::test]
    async fn test_trading_events() {
        let bus = Arc::new(LocalEventBus::new(1000));
        let handler = Arc::new(OrderStatusHandler::new());
        
        bus.subscribe::<TradingEvent, _>(handler).await.unwrap();
        
        let order_event = Event::new(
            "trading.order.submitted".to_string(),
            "trading_engine".to_string(),
            TradingEvent::OrderSubmitted {
                order_id: "order123".to_string(),
                symbol: "BTCUSDT".to_string(),
                side: "BUY".to_string(),
                quantity: 1.5,
                price: Some(50000.0),
            },
        );
        
        bus.publish(order_event).await.unwrap();
        
        // 等待处理
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn test_risk_monitor_handler() {
        let bus = Arc::new(LocalEventBus::new(1000));
        let handler = Arc::new(RiskMonitorHandler::new(0.1)); // 10%阈值
        
        bus.subscribe::<TradingEvent, _>(handler).await.unwrap();
        
        let risk_event = Event::new(
            "trading.risk.limit_breached".to_string(),
            "risk_engine".to_string(),
            TradingEvent::RiskLimitBreached {
                limit_type: "position_value".to_string(),
                current_value: 1100000.0,
                limit_value: 1000000.0,
                severity: "high".to_string(),
            },
        );
        
        bus.publish(risk_event).await.unwrap();
        
        // 等待处理
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn test_event_bus_manager() {
        let bus = Arc::new(LocalEventBus::new(1000));
        let manager = EventBusManager::new(bus as Arc<dyn EventBus>);
        
        let (handler, counter) = TestHandler::new("managed_handler".to_string());
        manager.register_handler::<TestEvent, _>(handler).await.unwrap();
        
        let test_event = Event::new(
            "test.event".to_string(),
            "test_service".to_string(),
            TestEvent {
                data: "Managed Event".to_string(),
                value: 123,
            },
        );
        
        manager.publish(test_event).await.unwrap();
        
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        
        assert_eq!(counter.load(Ordering::Relaxed), 1);
        
        manager.shutdown().await.unwrap();
    }
}

// 已移除所有模拟 Redis 依赖。请在生产构建中启用真实依赖并替换上方占位实现。
