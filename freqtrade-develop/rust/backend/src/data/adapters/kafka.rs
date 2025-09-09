use std::sync::Arc;
use async_trait::async_trait;
use anyhow::Result;

use crate::data::streaming::{StreamProducer, StreamConsumer, StreamMessage, StreamOffset};

/// Kafka 适配占位实现：未启用真实依赖时返回 UNAVAILABLE
pub struct KafkaProducerAdapter;
pub struct KafkaConsumerAdapter;

#[async_trait]
impl StreamProducer for KafkaProducerAdapter {
    async fn send(&self, _message: StreamMessage) -> Result<StreamOffset> {
        anyhow::bail!("Kafka integration not enabled in this build (adapter stub)")
    }
    async fn send_batch(&self, _messages: Vec<StreamMessage>) -> Result<Vec<StreamOffset>> {
        anyhow::bail!("Kafka integration not enabled in this build (adapter stub)")
    }
    async fn flush(&self) -> Result<()> { Ok(()) }
    async fn close(&self) -> Result<()> { Ok(()) }
}

#[async_trait]
impl StreamConsumer for KafkaConsumerAdapter {
    async fn subscribe(&self, _topics: Vec<String>) -> Result<()> {
        anyhow::bail!("Kafka integration not enabled in this build (adapter stub)")
    }
    async fn poll(&self, _timeout_ms: u64) -> Result<Vec<StreamMessage>> {
        anyhow::bail!("Kafka integration not enabled in this build (adapter stub)")
    }
    async fn commit(&self, _offsets: Vec<StreamOffset>) -> Result<()> { Ok(()) }
    async fn seek(&self, _offset: StreamOffset) -> Result<()> { Ok(()) }
    async fn close(&self) -> Result<()> { Ok(()) }
}

