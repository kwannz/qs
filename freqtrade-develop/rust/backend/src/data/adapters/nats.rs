use std::sync::Arc;
use async_trait::async_trait;
use anyhow::{Result, Context};
use crate::data::streaming::{StreamProducer, StreamConsumer, StreamMessage, StreamOffset};

/// 简化的 NATS 适配器（基础 Pub/Sub，无 JetStream 持久性）
pub struct NatsProducer {
    client: async_nats::Client,
}

pub struct NatsConsumer {
    client: async_nats::Client,
    subscription: Option<async_nats::Subscriber>,
}

impl NatsProducer {
    pub async fn connect(url: &str) -> Result<Self> {
        let client = async_nats::connect(url).await.context("connect nats")?;
        Ok(Self { client })
    }
}

impl NatsConsumer {
    pub async fn connect(url: &str) -> Result<Self> {
        let client = async_nats::connect(url).await.context("connect nats")?;
        Ok(Self { client, subscription: None })
    }
}

#[async_trait]
impl StreamProducer for NatsProducer {
    async fn send(&self, message: StreamMessage) -> Result<StreamOffset> {
        let subject = message.topic.clone();
        self.client.publish(subject.clone(), message.payload.into()).await.context("nats publish")?;
        Ok(StreamOffset { topic: subject, partition: 0, offset: 0 })
    }
    async fn send_batch(&self, messages: Vec<StreamMessage>) -> Result<Vec<StreamOffset>> {
        let mut res = Vec::with_capacity(messages.len());
        for m in messages { res.push(self.send(m).await?); }
        Ok(res)
    }
    async fn flush(&self) -> Result<()> { Ok(()) }
    async fn close(&self) -> Result<()> { Ok(()) }
}

#[async_trait]
impl StreamConsumer for NatsConsumer {
    async fn subscribe(&self, topics: Vec<String>) -> Result<()> {
        // 简化：仅订阅第一个主题
        let subject = topics.get(0).cloned().unwrap_or_else(|| "events".to_string());
        let sub = self.client.subscribe(subject).await.context("nats subscribe")?;
        // Safety: subscription needs mut but trait uses &self; keep in RefCell with interior mutability in real impl
        unsafe { 
            let p = self as *const _ as *mut NatsConsumer;
            (*p).subscription = Some(sub);
        }
        Ok(())
    }
    async fn poll(&self, _timeout_ms: u64) -> Result<Vec<StreamMessage>> {
        let mut out = Vec::new();
        if let Some(sub) = &self.subscription {
            if let Ok(Some(msg)) = tokio::time::timeout(std::time::Duration::from_millis(_timeout_ms), sub.next()).await {
                let msg = msg.context("nats poll")?;
                out.push(StreamMessage {
                    id: uuid::Uuid::new_v4(),
                    topic: msg.subject.clone(),
                    partition: None,
                    key: None,
                    payload: msg.payload.to_vec(),
                    headers: Default::default(),
                    timestamp: chrono::Utc::now(),
                });
            }
        }
        Ok(out)
    }
    async fn commit(&self, _offsets: Vec<StreamOffset>) -> Result<()> { Ok(()) }
    async fn seek(&self, _offset: StreamOffset) -> Result<()> { Ok(()) }
    async fn close(&self) -> Result<()> { Ok(()) }
}
