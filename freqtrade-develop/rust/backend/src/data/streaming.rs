//! 流式数据处理抽象
//! 注：本仓库不再包含任何模拟实现。请在生产环境对接真实消息总线（如 NATS JetStream / Kafka）。

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use anyhow::Result;
use async_trait::async_trait;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMessage {
    pub id: Uuid,
    pub topic: String,
    pub partition: Option<u32>,
    pub key: Option<String>,
    pub payload: Vec<u8>,
    pub headers: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct StreamOffset {
    pub topic: String,
    pub partition: u32,
    pub offset: u64,
}

#[derive(Debug, Clone)]
pub struct ConsumerConfig {
    pub group_id: String,
    pub topics: Vec<String>,
    pub auto_offset_reset: OffsetReset,
    pub enable_auto_commit: bool,
    pub max_poll_records: usize,
    pub session_timeout_ms: u64,
}

#[derive(Debug, Clone)]
pub enum OffsetReset { Earliest, Latest, None }

#[derive(Debug, Clone)]
pub struct ProducerConfig {
    pub acks: AcknowledgmentMode,
    pub retries: u32,
    pub batch_size: usize,
    pub linger_ms: u64,
    pub compression_type: CompressionType,
    pub max_in_flight_requests: u32,
}

#[derive(Debug, Clone)]
pub enum AcknowledgmentMode { None, Leader, All }

#[derive(Debug, Clone)]
pub enum CompressionType { None, Gzip, Snappy, Lz4, Zstd }

#[async_trait]
pub trait StreamProducer: Send + Sync {
    async fn send(&self, message: StreamMessage) -> Result<StreamOffset>;
    async fn send_batch(&self, messages: Vec<StreamMessage>) -> Result<Vec<StreamOffset>>;
    async fn flush(&self) -> Result<()>;
    async fn close(&self) -> Result<()>;
}

#[async_trait]
pub trait StreamConsumer: Send + Sync {
    async fn subscribe(&self, topics: Vec<String>) -> Result<()>;
    async fn poll(&self, timeout_ms: u64) -> Result<Vec<StreamMessage>>;
    async fn commit(&self, offsets: Vec<StreamOffset>) -> Result<()>;
    async fn seek(&self, offset: StreamOffset) -> Result<()>;
    async fn close(&self) -> Result<()>;
}

/// 简单的生产者/消费者注册器（占位以便上层注入真实实现）
#[derive(Default)]
pub struct StreamRegistry {
    pub producers: HashMap<String, Arc<dyn StreamProducer>>,
    pub consumers: HashMap<String, Arc<dyn StreamConsumer>>,
    pub metrics: Arc<StreamMetrics>,
}

#[derive(Default)]
pub struct StreamMetrics {
    pub send_latency_ms: RwLock<Vec<u64>>, // 保留结构以便 Prometheus 导出
}

impl StreamRegistry {
    pub fn new() -> Self { Self::default() }
    pub fn add_producer(&mut self, name: String, producer: Arc<dyn StreamProducer>) {
        self.producers.insert(name, producer);
    }
    pub fn add_consumer(&mut self, name: String, consumer: Arc<dyn StreamConsumer>) {
        self.consumers.insert(name, consumer);
    }
}

