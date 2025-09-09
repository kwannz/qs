use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use std::collections::HashMap;

/// Strategy transmission and lifecycle management protocol
/// Defines how strategies are deployed, managed, and executed across the platform

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyManifest {
    pub id: String,
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub strategy_type: StrategyType,
    pub runtime: StrategyRuntime,
    pub parameters: StrategyParameters,
    pub dependencies: Vec<StrategyDependency>,
    pub permissions: StrategyPermissions,
    pub lifecycle: StrategyLifecycle,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrategyType {
    /// Pure algorithmic strategies
    Algorithmic {
        algorithm_type: AlgorithmType,
        complexity_score: u32,
    },
    /// Machine learning based strategies  
    MachineLearning {
        model_type: String,
        training_data_required: bool,
        inference_latency_ms: u32,
    },
    /// Hybrid strategies combining multiple approaches
    Hybrid {
        components: Vec<StrategyComponent>,
        orchestration_mode: OrchestrationMode,
    },
    /// Custom strategy implementations
    Custom {
        implementation_language: String,
        entry_point: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlgorithmType {
    TrendFollowing,
    MeanReversion,
    Arbitrage,
    MarketMaking,
    Momentum,
    Statistical,
    Technical,
    Fundamental,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyComponent {
    pub name: String,
    pub component_type: ComponentType,
    pub weight: f64,
    pub config: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComponentType {
    SignalGenerator,
    RiskManager,
    PositionSizer,
    ExecutionFilter,
    PerformanceTracker,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrchestrationMode {
    Sequential,
    Parallel,
    Conditional,
    Weighted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrategyRuntime {
    /// WebAssembly runtime for portable execution
    Wasm {
        module_path: String,
        memory_limit_mb: u32,
        execution_timeout_ms: u32,
    },
    /// Native Rust compiled strategy
    Native {
        binary_path: String,
        shared_library: bool,
    },
    /// Python script execution
    Python {
        script_path: String,
        python_version: String,
        virtual_env: Option<String>,
    },
    /// JavaScript/TypeScript execution
    JavaScript {
        script_path: String,
        node_version: String,
    },
    /// Container-based execution
    Container {
        image: String,
        tag: String,
        resource_limits: ContainerLimits,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerLimits {
    pub cpu_cores: f64,
    pub memory_mb: u32,
    pub disk_mb: u32,
    pub network_bandwidth_mbps: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyParameters {
    pub required: HashMap<String, ParameterDefinition>,
    pub optional: HashMap<String, ParameterDefinition>,
    pub presets: HashMap<String, ParameterPreset>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDefinition {
    pub param_type: ParameterType,
    pub description: String,
    pub default_value: Option<serde_json::Value>,
    pub validation: ParameterValidation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    Integer { min: Option<i64>, max: Option<i64> },
    Float { min: Option<f64>, max: Option<f64> },
    String { max_length: Option<usize> },
    Boolean,
    Array { element_type: Box<ParameterType> },
    Object { schema: serde_json::Value },
    Enum { values: Vec<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterValidation {
    pub required: bool,
    pub custom_validator: Option<String>,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterPreset {
    pub name: String,
    pub description: String,
    pub values: HashMap<String, serde_json::Value>,
    pub risk_profile: RiskProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskProfile {
    Conservative,
    Moderate,
    Aggressive,
    Custom { score: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyDependency {
    pub name: String,
    pub version_constraint: String,
    pub dependency_type: DependencyType,
    pub optional: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    MarketData { providers: Vec<String> },
    ExecutionVenue { exchanges: Vec<String> },
    ExternalApi { api_name: String, rate_limit: u32 },
    Database { db_type: String },
    Library { language: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyPermissions {
    pub market_data_access: Vec<String>,
    pub execution_permissions: ExecutionPermissions,
    pub external_api_access: Vec<String>,
    pub database_access: DatabaseAccess,
    pub network_access: NetworkAccess,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPermissions {
    pub can_place_orders: bool,
    pub can_cancel_orders: bool,
    pub can_modify_positions: bool,
    pub max_order_size: f64,
    pub max_daily_volume: f64,
    pub allowed_symbols: Vec<String>,
    pub allowed_exchanges: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseAccess {
    pub read_tables: Vec<String>,
    pub write_tables: Vec<String>,
    pub query_complexity_limit: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkAccess {
    pub allowed_hosts: Vec<String>,
    pub max_requests_per_minute: u32,
    pub timeout_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyLifecycle {
    pub phases: Vec<LifecyclePhase>,
    pub rollback_strategy: RollbackStrategy,
    pub health_checks: Vec<HealthCheck>,
    pub performance_thresholds: PerformanceThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecyclePhase {
    pub name: String,
    pub phase_type: PhaseType,
    pub duration_seconds: Option<u32>,
    pub success_criteria: Vec<SuccessCriteria>,
    pub failure_handling: FailureHandling,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhaseType {
    Initialization,
    Validation,
    WarmUp,
    Active,
    Cooldown,
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriteria {
    pub metric: String,
    pub operator: ComparisonOperator,
    pub threshold: f64,
    pub time_window_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    GreaterOrEqual,
    LessOrEqual,
    Within { tolerance: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureHandling {
    pub action: FailureAction,
    pub max_retries: u32,
    pub backoff_strategy: StrategyBackoffStrategy,
    pub notification_channels: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureAction {
    Retry,
    Rollback,
    Pause,
    Stop,
    Escalate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrategyBackoffStrategy {
    Linear { increment_seconds: u32 },
    Exponential { base_seconds: u32, multiplier: f64 },
    Fixed { delay_seconds: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RollbackStrategy {
    None,
    PreviousVersion,
    SafeDefault,
    Custom { rollback_procedure: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub check_type: HealthCheckType,
    pub interval_seconds: u32,
    pub timeout_seconds: u32,
    pub failure_threshold: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthCheckType {
    Heartbeat,
    PerformanceMetric { metric: String, threshold: f64 },
    CustomCheck { procedure: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub min_success_rate: f64,
    pub max_drawdown: f64,
    pub max_latency_ms: u32,
    pub min_sharpe_ratio: Option<f64>,
    pub max_var: Option<f64>,
}

// Strategy deployment and management messages

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyDeploymentRequest {
    pub manifest: StrategyManifest,
    pub deployment_config: DeploymentConfig,
    pub target_environment: Environment,
    pub rollout_strategy: RolloutStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfig {
    pub replicas: u32,
    pub resource_allocation: ResourceAllocation,
    pub environment_variables: HashMap<String, String>,
    pub secrets: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub cpu_cores: f64,
    pub memory_mb: u32,
    pub disk_mb: u32,
    pub gpu_units: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Environment {
    Development,
    Testing,
    Staging,
    Production,
    Custom { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RolloutStrategy {
    BlueGreen,
    Canary { percentage: u32 },
    RollingUpdate { batch_size: u32 },
    Immediate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyDeploymentResponse {
    pub deployment_id: String,
    pub status: DeploymentStatus,
    pub message: String,
    pub endpoints: Vec<ServiceEndpoint>,
    pub metrics_dashboard: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStatus {
    Pending,
    Deploying,
    Active,
    Failed { error: String },
    RollingBack,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    pub endpoint_type: EndpointType,
    pub url: String,
    pub authentication_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EndpointType {
    Api,
    Websocket,
    Metrics,
    Health,
    Admin,
}

// Strategy execution control messages

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyControlRequest {
    pub strategy_id: String,
    pub action: ControlAction,
    pub parameters: HashMap<String, serde_json::Value>,
    pub reason: Option<String>,
    pub requested_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlAction {
    Start,
    Stop,
    Pause,
    Resume,
    Restart,
    UpdateParameters { params: HashMap<String, serde_json::Value> },
    Scale { replicas: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyControlResponse {
    pub strategy_id: String,
    pub action: ControlAction,
    pub status: ControlStatus,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlStatus {
    Success,
    Failed { error: String },
    Pending,
}

// Strategy monitoring and reporting

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyStatusReport {
    pub strategy_id: String,
    pub instance_id: String,
    pub status: InstanceStatus,
    pub performance_metrics: StrategyPerformanceMetrics,
    pub resource_usage: ResourceUsage,
    pub health_status: HealthStatus,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InstanceStatus {
    Starting,
    Running,
    Paused,
    Stopping,
    Stopped,
    Error { details: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyPerformanceMetrics {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub total_trades: u64,
    pub avg_trade_duration: f64,
    pub profit_factor: f64,
    pub calmar_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: u32,
    pub disk_usage_mb: u32,
    pub network_io_mbps: f64,
    pub api_calls_per_minute: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub overall_status: Health,
    pub component_status: HashMap<String, Health>,
    pub last_health_check: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Health {
    Healthy,
    Degraded { reason: String },
    Unhealthy { reason: String },
    Unknown,
}

// Strategy lifecycle events

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyLifecycleEvent {
    pub event_id: String,
    pub strategy_id: String,
    pub event_type: LifecycleEventType,
    pub details: serde_json::Value,
    pub timestamp: DateTime<Utc>,
    pub severity: EventSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LifecycleEventType {
    Deployed,
    Started,
    Stopped,
    Paused,
    Resumed,
    Updated,
    Scaled,
    HealthCheckFailed,
    PerformanceThresholdBreached,
    Error,
    Warning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl StrategyManifest {
    pub fn new(name: String, version: String, author: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            version,
            author,
            description: String::new(),
            strategy_type: StrategyType::Custom {
                implementation_language: "rust".to_string(),
                entry_point: "main".to_string(),
            },
            runtime: StrategyRuntime::Native {
                binary_path: String::new(),
                shared_library: false,
            },
            parameters: StrategyParameters {
                required: HashMap::new(),
                optional: HashMap::new(),
                presets: HashMap::new(),
            },
            dependencies: Vec::new(),
            permissions: StrategyPermissions {
                market_data_access: Vec::new(),
                execution_permissions: ExecutionPermissions {
                    can_place_orders: false,
                    can_cancel_orders: false,
                    can_modify_positions: false,
                    max_order_size: 0.0,
                    max_daily_volume: 0.0,
                    allowed_symbols: Vec::new(),
                    allowed_exchanges: Vec::new(),
                },
                external_api_access: Vec::new(),
                database_access: DatabaseAccess {
                    read_tables: Vec::new(),
                    write_tables: Vec::new(),
                    query_complexity_limit: 100,
                },
                network_access: NetworkAccess {
                    allowed_hosts: Vec::new(),
                    max_requests_per_minute: 60,
                    timeout_seconds: 30,
                },
            },
            lifecycle: StrategyLifecycle {
                phases: Vec::new(),
                rollback_strategy: RollbackStrategy::None,
                health_checks: Vec::new(),
                performance_thresholds: PerformanceThresholds {
                    min_success_rate: 0.5,
                    max_drawdown: 0.2,
                    max_latency_ms: 1000,
                    min_sharpe_ratio: None,
                    max_var: None,
                },
            },
            created_at: now,
            updated_at: now,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("Strategy name cannot be empty".to_string());
        }

        if self.version.is_empty() {
            return Err("Strategy version cannot be empty".to_string());
        }

        // Validate parameters
        for (name, param) in &self.parameters.required {
            if let Err(e) = Self::validate_parameter(name, param) {
                return Err(format!("Invalid required parameter '{name}': {e}"));
            }
        }

        for (name, param) in &self.parameters.optional {
            if let Err(e) = Self::validate_parameter(name, param) {
                return Err(format!("Invalid optional parameter '{name}': {e}"));
            }
        }

        Ok(())
    }

    fn validate_parameter(name: &str, param: &ParameterDefinition) -> Result<(), String> {
        if name.is_empty() {
            return Err("Parameter name cannot be empty".to_string());
        }

        if param.description.is_empty() {
            return Err("Parameter description cannot be empty".to_string());
        }

        Ok(())
    }
}

// Utility functions for strategy protocol

pub fn create_basic_strategy_manifest(
    name: &str,
    version: &str,
    author: &str,
    strategy_type: StrategyType,
) -> StrategyManifest {
    let mut manifest = StrategyManifest::new(
        name.to_string(),
        version.to_string(),
        author.to_string(),
    );
    manifest.strategy_type = strategy_type;
    manifest
}

pub fn create_wasm_runtime_config(
    module_path: &str,
    memory_limit_mb: u32,
    execution_timeout_ms: u32,
) -> StrategyRuntime {
    StrategyRuntime::Wasm {
        module_path: module_path.to_string(),
        memory_limit_mb,
        execution_timeout_ms,
    }
}

pub fn create_container_runtime_config(
    image: &str,
    tag: &str,
    cpu_cores: f64,
    memory_mb: u32,
) -> StrategyRuntime {
    StrategyRuntime::Container {
        image: image.to_string(),
        tag: tag.to_string(),
        resource_limits: ContainerLimits {
            cpu_cores,
            memory_mb,
            disk_mb: 1024,
            network_bandwidth_mbps: 100,
        },
    }
}