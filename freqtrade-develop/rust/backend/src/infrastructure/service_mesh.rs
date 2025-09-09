use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast, mpsc};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, instrument};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// AG3服务网格 - 微服务治理和通信框架
pub struct ServiceMesh {
    registry: Arc<RwLock<ServiceRegistry>>,
    discovery: ServiceDiscovery,
    load_balancer: LoadBalancer,
    circuit_breaker: ServiceCircuitBreaker,
    rate_limiter: RateLimiter,
    health_checker: HealthChecker,
    tracer: DistributedTracer,
    config: ServiceMeshConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMeshConfig {
    pub cluster_name: String,
    pub node_id: String,
    pub discovery_interval_sec: u64,
    pub health_check_interval_sec: u64,
    pub circuit_breaker_threshold: f64,
    pub rate_limit_rps: u64,
    pub trace_sampling_rate: f64,
    pub load_balancing_strategy: LoadBalancingStrategy,
    pub service_timeout_ms: u64,
    pub max_retries: u32,
}

/// 服务注册表
pub struct ServiceRegistry {
    services: HashMap<String, ServiceInstance>,
    service_versions: HashMap<String, Vec<ServiceVersion>>,
    dependencies: HashMap<String, Vec<ServiceDependency>>,
    health_status: HashMap<String, ServiceHealthStatus>,
}

/// 服务实例
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInstance {
    pub service_id: String,
    pub service_name: String,
    pub version: String,
    pub endpoint: ServiceEndpoint,
    pub metadata: HashMap<String, String>,
    pub tags: Vec<String>,
    pub health_check_url: Option<String>,
    pub registered_at: DateTime<Utc>,
    pub last_heartbeat: DateTime<Utc>,
    pub capabilities: Vec<ServiceCapability>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    pub protocol: ServiceProtocol,
    pub host: String,
    pub port: u16,
    pub path: Option<String>,
    pub tls_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceProtocol {
    Http,
    Https,
    Grpc,
    WebSocket,
    Tcp,
    Udp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceCapability {
    pub capability_name: String,
    pub version: String,
    pub schema_url: Option<String>,
}

/// 服务版本
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceVersion {
    pub version: String,
    pub release_notes: String,
    pub breaking_changes: Vec<String>,
    pub deprecated_features: Vec<String>,
    pub required_capabilities: Vec<String>,
}

/// 服务依赖
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceDependency {
    pub service_name: String,
    pub version_constraint: String,
    pub dependency_type: DependencyType,
    pub required: bool,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    Synchronous,    // 同步依赖
    Asynchronous,   // 异步依赖
    EventDriven,    // 事件驱动
    DataDependency, // 数据依赖
}

/// 服务健康状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceHealthStatus {
    pub service_id: String,
    pub status: HealthStatus,
    pub last_check: DateTime<Utc>,
    pub response_time_ms: u64,
    pub error_count: u32,
    pub success_count: u32,
    pub details: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Unhealthy,
    Degraded,
    Unknown,
    Maintenance,
}

/// 服务发现
pub struct ServiceDiscovery {
    discovery_channels: HashMap<String, broadcast::Sender<ServiceEvent>>,
    consul_client: Option<ConsulClient>,
    etcd_client: Option<EtcdClient>,
    k8s_client: Option<KubernetesClient>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceEvent {
    ServiceRegistered(ServiceInstance),
    ServiceDeregistered(String),
    ServiceHealthChanged { service_id: String, status: HealthStatus },
    ServiceUpdated(ServiceInstance),
}

/// 负载均衡器
pub struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    service_weights: HashMap<String, f64>,
    sticky_sessions: HashMap<String, String>, // session_id -> service_id
    health_aware: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    Random,
    ConsistentHashing,
    LatencyBased,
    ResourceBased,
}

/// 服务级熔断器
pub struct ServiceCircuitBreaker {
    breakers: HashMap<String, CircuitBreakerState>,
    config: CircuitBreakerConfig,
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerState {
    pub state: BreakerState,
    pub failure_count: u32,
    pub success_count: u32,
    pub last_failure_time: Option<DateTime<Utc>>,
    pub next_attempt_time: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BreakerState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout_duration_ms: u64,
    pub half_open_max_calls: u32,
}

/// 速率限制器
pub struct RateLimiter {
    limiters: HashMap<String, TokenBucket>,
    global_limiter: TokenBucket,
}

#[derive(Debug, Clone)]
pub struct TokenBucket {
    capacity: u64,
    tokens: Arc<RwLock<u64>>,
    refill_rate: u64,
    last_refill: Arc<RwLock<DateTime<Utc>>>,
}

/// 健康检查器
pub struct HealthChecker {
    check_strategies: HashMap<String, HealthCheckStrategy>,
    check_results: Arc<RwLock<HashMap<String, ServiceHealthStatus>>>,
    notification_channel: broadcast::Sender<ServiceEvent>,
}

#[derive(Debug, Clone)]
pub enum HealthCheckStrategy {
    HttpGet { url: String, expected_status: u16, timeout_ms: u64 },
    TcpConnect { host: String, port: u16, timeout_ms: u64 },
    GrpcHealthCheck { endpoint: String, service_name: String },
    CustomCheck { command: String, expected_exit_code: i32 },
}

/// 分布式追踪
pub struct DistributedTracer {
    trace_context: Arc<RwLock<HashMap<String, TraceContext>>>,
    span_exporter: SpanExporter,
    sampling_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub service_name: String,
    pub operation_name: String,
    pub start_time: DateTime<Utc>,
    pub duration_ns: Option<u64>,
    pub tags: HashMap<String, String>,
    pub logs: Vec<SpanLog>,
    pub status: TraceStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanLog {
    pub timestamp: DateTime<Utc>,
    pub level: LogLevel,
    pub message: String,
    pub fields: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraceStatus {
    Ok,
    Error(String),
    Timeout,
    Cancelled,
}

/// Span导出器
pub struct SpanExporter {
    jaeger_client: Option<JaegerClient>,
    zipkin_client: Option<ZipkinClient>,
    otlp_client: Option<OtlpClient>,
}

impl ServiceMesh {
    pub fn new(config: ServiceMeshConfig) -> Result<Self> {
        let registry = Arc::new(RwLock::new(ServiceRegistry::new()));
        let discovery = ServiceDiscovery::new(&config)?;
        let load_balancer = LoadBalancer::new(config.load_balancing_strategy.clone());
        let circuit_breaker = ServiceCircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 5,
            success_threshold: 3,
            timeout_duration_ms: 60000,
            half_open_max_calls: 3,
        });
        let rate_limiter = RateLimiter::new(config.rate_limit_rps);
        let health_checker = HealthChecker::new(config.health_check_interval_sec)?;
        let tracer = DistributedTracer::new(config.trace_sampling_rate)?;

        Ok(Self {
            registry,
            discovery,
            load_balancer,
            circuit_breaker,
            rate_limiter,
            health_checker,
            tracer,
            config,
        })
    }

    /// 启动服务网格
    #[instrument(skip(self))]
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting service mesh for cluster: {}", self.config.cluster_name);

        // 启动服务发现
        self.discovery.start().await?;

        // 启动健康检查
        self.health_checker.start().await?;

        // 启动分布式追踪
        self.tracer.start().await?;

        // 注册本节点
        self.register_self().await?;

        info!("Service mesh started successfully");
        Ok(())
    }

    /// 注册服务
    #[instrument(skip(self, instance))]
    pub async fn register_service(&self, instance: ServiceInstance) -> Result<()> {
        info!("Registering service: {} v{}", instance.service_name, instance.version);

        let mut registry = self.registry.write().await;
        registry.services.insert(instance.service_id.clone(), instance.clone());

        // 初始化健康状态
        registry.health_status.insert(
            instance.service_id.clone(),
            ServiceHealthStatus {
                service_id: instance.service_id.clone(),
                status: HealthStatus::Unknown,
                last_check: Utc::now(),
                response_time_ms: 0,
                error_count: 0,
                success_count: 0,
                details: HashMap::new(),
            },
        );

        drop(registry);

        // 通知服务发现
        self.discovery.notify_service_change(ServiceEvent::ServiceRegistered(instance)).await?;

        Ok(())
    }

    /// 发现服务
    #[instrument(skip(self))]
    pub async fn discover_service(
        &self,
        service_name: &str,
        version_constraint: Option<&str>,
    ) -> Result<Vec<ServiceInstance>> {
        let registry = self.registry.read().await;
        let mut matching_services = Vec::new();

        for service in registry.services.values() {
            if service.service_name == service_name {
                // 检查版本约束
                if let Some(constraint) = version_constraint {
                    if self.version_matches(&service.version, constraint) {
                        // 检查健康状态
                        if let Some(health) = registry.health_status.get(&service.service_id) {
                            if health.status == HealthStatus::Healthy {
                                matching_services.push(service.clone());
                            }
                        }
                    }
                } else {
                    matching_services.push(service.clone());
                }
            }
        }

        Ok(matching_services)
    }

    /// 调用服务
    #[instrument(skip(self, request))]
    pub async fn call_service<T, R>(
        &self,
        service_name: &str,
        operation: &str,
        request: T,
    ) -> Result<R>
    where
        T: Serialize,
        R: for<'de> Deserialize<'de>,
    {
        // 开始追踪
        let trace_ctx = self.tracer.start_span(service_name, operation).await?;

        // 服务发现
        let services = self.discover_service(service_name, None).await?;
        if services.is_empty() {
            return Err(anyhow::anyhow!("No healthy instances found for service: {}", service_name));
        }

        // 负载均衡选择实例
        let selected_service = self.load_balancer.select_service(&services).await?;

        // 检查熔断器状态
        if !self.circuit_breaker.can_execute(&selected_service.service_id).await? {
            return Err(anyhow::anyhow!("Circuit breaker is open for service: {}", service_name));
        }

        // 速率限制检查
        if !self.rate_limiter.acquire(&selected_service.service_id).await? {
            return Err(anyhow::anyhow!("Rate limit exceeded for service: {}", service_name));
        }

        // 执行调用
        let result = self.execute_service_call(&selected_service, operation, request, &trace_ctx).await;

        // 更新熔断器状态
        match &result {
            Ok(_) => {
                self.circuit_breaker.record_success(&selected_service.service_id).await?;
            }
            Err(_) => {
                self.circuit_breaker.record_failure(&selected_service.service_id).await?;
            }
        }

        // 结束追踪
        self.tracer.finish_span(trace_ctx, &result).await?;

        result
    }

    /// 执行实际的服务调用
    async fn execute_service_call<T, R>(
        &self,
        service: &ServiceInstance,
        operation: &str,
        request: T,
        trace_ctx: &TraceContext,
    ) -> Result<R>
    where
        T: Serialize,
        R: for<'de> Deserialize<'de>,
    {
        let start_time = std::time::Instant::now();

        let result = match service.endpoint.protocol {
            ServiceProtocol::Http | ServiceProtocol::Https => {
                self.http_call(service, operation, request, trace_ctx).await
            }
            ServiceProtocol::Grpc => {
                self.grpc_call(service, operation, request, trace_ctx).await
            }
            _ => {
                Err(anyhow::anyhow!("Unsupported protocol: {:?}", service.endpoint.protocol))
            }
        };

        let duration = start_time.elapsed();
        
        // 记录调用指标
        self.record_call_metrics(service, operation, duration, &result).await?;

        result
    }

    /// HTTP调用
    async fn http_call<T, R>(
        &self,
        service: &ServiceInstance,
        operation: &str,
        request: T,
        trace_ctx: &TraceContext,
    ) -> Result<R>
    where
        T: Serialize,
        R: for<'de> Deserialize<'de>,
    {
        let url = format!(
            "{}://{}:{}/{}",
            if service.endpoint.tls_enabled { "https" } else { "http" },
            service.endpoint.host,
            service.endpoint.port,
            operation
        );

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(self.config.service_timeout_ms))
            .build()?;

        let response = client
            .post(&url)
            .header("X-Trace-Id", &trace_ctx.trace_id)
            .header("X-Span-Id", &trace_ctx.span_id)
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let result: R = response.json().await?;
            Ok(result)
        } else {
            Err(anyhow::anyhow!("HTTP call failed with status: {}", response.status()))
        }
    }

    /// gRPC调用
    async fn grpc_call<T, R>(
        &self,
        service: &ServiceInstance,
        operation: &str,
        request: T,
        trace_ctx: &TraceContext,
    ) -> Result<R>
    where
        T: Serialize,
        R: for<'de> Deserialize<'de>,
    {
        // gRPC调用的简化实现
        // 实际实现需要根据具体的gRPC客户端库
        Err(anyhow::anyhow!("gRPC call not implemented"))
    }

    /// 记录调用指标
    async fn record_call_metrics<R>(
        &self,
        service: &ServiceInstance,
        operation: &str,
        duration: std::time::Duration,
        result: &Result<R>,
    ) -> Result<()> {
        // 更新健康状态
        let mut registry = self.registry.write().await;
        if let Some(health) = registry.health_status.get_mut(&service.service_id) {
            health.response_time_ms = duration.as_millis() as u64;
            health.last_check = Utc::now();
            
            match result {
                Ok(_) => {
                    health.success_count += 1;
                    health.status = HealthStatus::Healthy;
                }
                Err(_) => {
                    health.error_count += 1;
                    if health.error_count > 5 {
                        health.status = HealthStatus::Unhealthy;
                    }
                }
            }
        }

        Ok(())
    }

    /// 版本匹配检查
    fn version_matches(&self, version: &str, constraint: &str) -> bool {
        // 简化的版本匹配逻辑
        // 实际实现应该支持语义化版本约束
        version == constraint || constraint == "*"
    }

    /// 注册自身节点
    async fn register_self(&self) -> Result<()> {
        let self_instance = ServiceInstance {
            service_id: self.config.node_id.clone(),
            service_name: "service-mesh".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            endpoint: ServiceEndpoint {
                protocol: ServiceProtocol::Http,
                host: "localhost".to_string(),
                port: 8080,
                path: Some("/health".to_string()),
                tls_enabled: false,
            },
            metadata: HashMap::new(),
            tags: vec!["infrastructure".to_string()],
            health_check_url: Some("/health".to_string()),
            registered_at: Utc::now(),
            last_heartbeat: Utc::now(),
            capabilities: vec![],
        };

        self.register_service(self_instance).await
    }

    /// 获取服务拓扑
    pub async fn get_service_topology(&self) -> Result<ServiceTopology> {
        let registry = self.registry.read().await;
        let mut topology = ServiceTopology {
            services: Vec::new(),
            dependencies: Vec::new(),
        };

        // 收集服务信息
        for service in registry.services.values() {
            topology.services.push(ServiceNode {
                service_id: service.service_id.clone(),
                service_name: service.service_name.clone(),
                version: service.version.clone(),
                status: registry.health_status.get(&service.service_id)
                    .map(|h| h.status.clone())
                    .unwrap_or(HealthStatus::Unknown),
                endpoint: service.endpoint.clone(),
                metadata: service.metadata.clone(),
            });
        }

        // 收集依赖关系
        for (service_name, deps) in &registry.dependencies {
            for dep in deps {
                topology.dependencies.push(ServiceDependencyEdge {
                    from_service: service_name.clone(),
                    to_service: dep.service_name.clone(),
                    dependency_type: dep.dependency_type.clone(),
                    required: dep.required,
                });
            }
        }

        Ok(topology)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceTopology {
    pub services: Vec<ServiceNode>,
    pub dependencies: Vec<ServiceDependencyEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceNode {
    pub service_id: String,
    pub service_name: String,
    pub version: String,
    pub status: HealthStatus,
    pub endpoint: ServiceEndpoint,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceDependencyEdge {
    pub from_service: String,
    pub to_service: String,
    pub dependency_type: DependencyType,
    pub required: bool,
}

// 实现各个组件
impl ServiceRegistry {
    fn new() -> Self {
        Self {
            services: HashMap::new(),
            service_versions: HashMap::new(),
            dependencies: HashMap::new(),
            health_status: HashMap::new(),
        }
    }
}

impl ServiceDiscovery {
    fn new(config: &ServiceMeshConfig) -> Result<Self> {
        Ok(Self {
            discovery_channels: HashMap::new(),
            consul_client: None,
            etcd_client: None,
            k8s_client: None,
        })
    }

    async fn start(&mut self) -> Result<()> {
        info!("Starting service discovery");
        // 启动服务发现逻辑
        Ok(())
    }

    async fn notify_service_change(&self, event: ServiceEvent) -> Result<()> {
        // 通知所有监听者
        for channel in self.discovery_channels.values() {
            let _ = channel.send(event.clone());
        }
        Ok(())
    }
}

impl LoadBalancer {
    fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            service_weights: HashMap::new(),
            sticky_sessions: HashMap::new(),
            health_aware: true,
        }
    }

    async fn select_service(&self, services: &[ServiceInstance]) -> Result<&ServiceInstance> {
        if services.is_empty() {
            return Err(anyhow::anyhow!("No services available"));
        }

        match self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                // 简化的轮询实现
                Ok(&services[0])
            }
            LoadBalancingStrategy::Random => {
                let index = fastrand::usize(0..services.len());
                Ok(&services[index])
            }
            _ => {
                // 默认返回第一个服务
                Ok(&services[0])
            }
        }
    }
}

impl ServiceCircuitBreaker {
    fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            breakers: HashMap::new(),
            config,
        }
    }

    async fn can_execute(&self, service_id: &str) -> Result<bool> {
        // 简化的熔断器逻辑
        Ok(true)
    }

    async fn record_success(&mut self, service_id: &str) -> Result<()> {
        // 记录成功调用
        Ok(())
    }

    async fn record_failure(&mut self, service_id: &str) -> Result<()> {
        // 记录失败调用
        Ok(())
    }
}

impl RateLimiter {
    fn new(global_rps: u64) -> Self {
        Self {
            limiters: HashMap::new(),
            global_limiter: TokenBucket::new(global_rps, global_rps),
        }
    }

    async fn acquire(&self, service_id: &str) -> Result<bool> {
        // 简化的限流逻辑
        Ok(true)
    }
}

impl TokenBucket {
    fn new(capacity: u64, refill_rate: u64) -> Self {
        Self {
            capacity,
            tokens: Arc::new(RwLock::new(capacity)),
            refill_rate,
            last_refill: Arc::new(RwLock::new(Utc::now())),
        }
    }
}

impl HealthChecker {
    fn new(interval_sec: u64) -> Result<Self> {
        let (tx, _rx) = broadcast::channel(1000);
        Ok(Self {
            check_strategies: HashMap::new(),
            check_results: Arc::new(RwLock::new(HashMap::new())),
            notification_channel: tx,
        })
    }

    async fn start(&mut self) -> Result<()> {
        info!("Starting health checker");
        // 启动健康检查逻辑
        Ok(())
    }
}

impl DistributedTracer {
    fn new(sampling_rate: f64) -> Result<Self> {
        Ok(Self {
            trace_context: Arc::new(RwLock::new(HashMap::new())),
            span_exporter: SpanExporter::new()?,
            sampling_rate,
        })
    }

    async fn start(&mut self) -> Result<()> {
        info!("Starting distributed tracer");
        Ok(())
    }

    async fn start_span(&self, service_name: &str, operation: &str) -> Result<TraceContext> {
        let trace_id = Uuid::new_v4().to_string();
        let span_id = Uuid::new_v4().to_string();

        let ctx = TraceContext {
            trace_id,
            span_id,
            parent_span_id: None,
            service_name: service_name.to_string(),
            operation_name: operation.to_string(),
            start_time: Utc::now(),
            duration_ns: None,
            tags: HashMap::new(),
            logs: Vec::new(),
            status: TraceStatus::Ok,
        };

        Ok(ctx)
    }

    async fn finish_span<R>(&self, mut ctx: TraceContext, result: &Result<R>) -> Result<()> {
        ctx.duration_ns = Some(
            Utc::now()
                .signed_duration_since(ctx.start_time)
                .num_nanoseconds()
                .unwrap_or(0) as u64
        );

        ctx.status = match result {
            Ok(_) => TraceStatus::Ok,
            Err(e) => TraceStatus::Error(e.to_string()),
        };

        self.span_exporter.export_span(ctx).await
    }
}

impl SpanExporter {
    fn new() -> Result<Self> {
        Ok(Self {
            jaeger_client: None,
            zipkin_client: None,
            otlp_client: None,
        })
    }

    async fn export_span(&self, span: TraceContext) -> Result<()> {
        // 简化的span导出
        info!("Exporting span: {} for operation: {}", span.span_id, span.operation_name);
        Ok(())
    }
}

// 占位符客户端
pub struct ConsulClient;
pub struct EtcdClient;
pub struct KubernetesClient;
pub struct JaegerClient;
pub struct ZipkinClient;
pub struct OtlpClient;

impl Default for ServiceMeshConfig {
    fn default() -> Self {
        Self {
            cluster_name: "ag3-cluster".to_string(),
            node_id: Uuid::new_v4().to_string(),
            discovery_interval_sec: 30,
            health_check_interval_sec: 10,
            circuit_breaker_threshold: 0.5,
            rate_limit_rps: 1000,
            trace_sampling_rate: 0.1,
            load_balancing_strategy: LoadBalancingStrategy::RoundRobin,
            service_timeout_ms: 5000,
            max_retries: 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_mesh_creation() {
        let config = ServiceMeshConfig::default();
        let result = ServiceMesh::new(config);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_service_registration() {
        let config = ServiceMeshConfig::default();
        let mesh = ServiceMesh::new(config).unwrap();

        let service = ServiceInstance {
            service_id: "test-service-1".to_string(),
            service_name: "test-service".to_string(),
            version: "1.0.0".to_string(),
            endpoint: ServiceEndpoint {
                protocol: ServiceProtocol::Http,
                host: "localhost".to_string(),
                port: 8080,
                path: None,
                tls_enabled: false,
            },
            metadata: HashMap::new(),
            tags: Vec::new(),
            health_check_url: None,
            registered_at: Utc::now(),
            last_heartbeat: Utc::now(),
            capabilities: Vec::new(),
        };

        let result = mesh.register_service(service).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_service_discovery() {
        let config = ServiceMeshConfig::default();
        let mesh = ServiceMesh::new(config).unwrap();

        // 先注册一个服务
        let service = ServiceInstance {
            service_id: "test-service-1".to_string(),
            service_name: "test-service".to_string(),
            version: "1.0.0".to_string(),
            endpoint: ServiceEndpoint {
                protocol: ServiceProtocol::Http,
                host: "localhost".to_string(),
                port: 8080,
                path: None,
                tls_enabled: false,
            },
            metadata: HashMap::new(),
            tags: Vec::new(),
            health_check_url: None,
            registered_at: Utc::now(),
            last_heartbeat: Utc::now(),
            capabilities: Vec::new(),
        };

        mesh.register_service(service).await.unwrap();

        // 发现服务
        let discovered = mesh.discover_service("test-service", None).await.unwrap();
        assert_eq!(discovered.len(), 1);
        assert_eq!(discovered[0].service_name, "test-service");
    }
}