use async_trait::async_trait;
use anyhow::{Result, Context};
use async_nats::{Client};
use async_nats::jetstream;
use crate::data::streaming::{StreamProducer, StreamConsumer, StreamMessage, StreamOffset};

/// JetStream 生产者（简化：确保 Stream 存在，然后发布消息）
pub struct JetStreamProducer {
    client: Client,
    js: jetstream::Context,
    stream_name: String,
}

impl JetStreamProducer {
    pub async fn connect(url: &str, stream_name: &str, subjects: Vec<String>) -> Result<Self> {
        let client = async_nats::connect(url).await.context("connect nats")?;
        let js = jetstream::new(client.clone());

        // 确保 Stream 存在
        if js.get_stream(stream_name).await.is_err() {
            let _ = js.create_stream(jetstream::stream::Config {
                name: stream_name.to_string(),
                subjects,
                ..Default::default()
            }).await.context("create jetstream stream")?;
        }

        Ok(Self { client, js, stream_name: stream_name.to_string() })
    }
}

#[async_trait]
impl StreamProducer for JetStreamProducer {
    async fn send(&self, message: StreamMessage) -> Result<StreamOffset> {
        // 直接对 subject 发布；要求 subject 匹配已配置的 stream subjects
        self.js.publish(message.topic.clone(), message.payload.into()).await
            .context("jetstream publish")?;
        Ok(StreamOffset { topic: message.topic, partition: 0, offset: 0 })
    }
    async fn send_batch(&self, messages: Vec<StreamMessage>) -> Result<Vec<StreamOffset>> {
        let mut res = Vec::with_capacity(messages.len());
        for m in messages { res.push(self.send(m).await?); }
        Ok(res)
    }
    async fn flush(&self) -> Result<()> { Ok(()) }
    async fn close(&self) -> Result<()> { Ok(()) }
}

/// JetStream 消费者（简化：暂用核心订阅，在后续按需切换 Pull Consumer）
pub struct JetStreamConsumer {
    client: Client,
    subscription: Option<async_nats::Subscriber>,
}

impl JetStreamConsumer {
    pub async fn connect(url: &str) -> Result<Self> {
        let client = async_nats::connect(url).await.context("connect nats")?;
        Ok(Self { client, subscription: None })
    }
}

#[async_trait]
impl StreamConsumer for JetStreamConsumer {
    async fn subscribe(&self, topics: Vec<String>) -> Result<()> {
        let subject = topics.get(0).cloned().unwrap_or_else(|| "events".to_string());
        let sub = self.client.subscribe(subject).await.context("nats subscribe")?;
        unsafe {
            let p = self as *const _ as *mut JetStreamConsumer;
            (*p).subscription = Some(sub);
        }
        Ok(())
    }
    async fn poll(&self, timeout_ms: u64) -> Result<Vec<StreamMessage>> {
        let mut out = Vec::new();
        if let Some(sub) = &self.subscription {
            if let Ok(Some(msg)) = tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), sub.next()).await {
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

/// JetStream Durable Pull Consumer（带 ack）
pub struct JetStreamDurableConsumer {
    js: jetstream::Context,
    stream: String,
    durable: String,
    subject: String,
}

impl JetStreamDurableConsumer {
    pub async fn connect_durable(url: &str, stream: &str, durable: &str, subject: &str) -> Result<Self> {
        let client = async_nats::connect(url).await.context("connect nats")?;
        let js = jetstream::new(client);

        // 确保 Stream 存在（如果未创建则创建一个匹配该 subject 的流）
        if js.get_stream(stream).await.is_err() {
            let _ = js.create_stream(jetstream::stream::Config {
                name: stream.to_string(),
                subjects: vec![subject.to_string()],
                ..Default::default()
            }).await.context("create jetstream stream")?;
        }

        // 确保 Durable Consumer 存在
        let stream_h = js.get_stream(stream).await.context("get stream")?;
        let _ = stream_h.create_consumer(jetstream::consumer::pull::Config {
            durable_name: Some(durable.to_string()),
            filter_subject: Some(subject.to_string()),
            ack_policy: jetstream::consumer::AckPolicy::Explicit,
            ..Default::default()
        }).await.context("create durable consumer")?;

        Ok(Self { js, stream: stream.to_string(), durable: durable.to_string(), subject: subject.to_string() })
    }
}

#[async_trait]
impl StreamConsumer for JetStreamDurableConsumer {
    async fn subscribe(&self, _topics: Vec<String>) -> Result<()> {
        Ok(()) // durable 已绑定到 subject
    }
    async fn poll(&self, timeout_ms: u64) -> Result<Vec<StreamMessage>> {
        let stream = self.js.get_stream(&self.stream).await.context("get stream")?;
        let consumer = stream.get_consumer::<jetstream::consumer::pull::Config>(&self.durable)
            .await.context("get consumer")?;

        // 拉取一批消息（最多 10 条）
        let mut out = Vec::new();
        match tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), consumer.fetch(10)).await {
            Ok(Ok(mut batch)) => {
                while let Some(msg) = batch.next().await {
                    let data = msg.payload.to_vec();
                    // ack，并忽略错误
                    let _ = msg.ack().await;
                    out.push(StreamMessage {
                        id: uuid::Uuid::new_v4(),
                        topic: self.subject.clone(),
                        partition: None,
                        key: None,
                        payload: data,
                        headers: Default::default(),
                        timestamp: chrono::Utc::now(),
                    });
                }
            }
            _ => {}
        }
        Ok(out)
    }
    async fn commit(&self, _offsets: Vec<StreamOffset>) -> Result<()> { Ok(()) }
    async fn seek(&self, _offset: StreamOffset) -> Result<()> { Ok(()) }
    async fn close(&self) -> Result<()> { Ok(()) }
}
