//! CI/CD自动化管道
//! 支持GitHub Actions、GitLab CI、Jenkins等CI/CD系统

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context};
use chrono::{DateTime, Utc};

/// 管道阶段
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PipelineStage {
    Build,
    Test,
    SecurityScan,
    Deploy,
    PostDeploy,
}

/// 管道状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PipelineStatus {
    Pending,
    Running,
    Success,
    Failed,
    Cancelled,
    Skipped,
}

/// 部署环境
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentEnvironment {
    Development,
    Testing,
    Staging,
    Production,
    Custom(String),
}

impl DeploymentEnvironment {
    pub fn as_str(&self) -> &str {
        match self {
            DeploymentEnvironment::Development => "dev",
            DeploymentEnvironment::Testing => "test",
            DeploymentEnvironment::Staging => "staging",
            DeploymentEnvironment::Production => "prod",
            DeploymentEnvironment::Custom(name) => name,
        }
    }
}

/// 管道作业定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineJob {
    pub name: String,
    pub stage: PipelineStage,
    pub commands: Vec<String>,
    pub environment: HashMap<String, String>,
    pub dependencies: Vec<String>,
    pub artifacts: Vec<ArtifactConfig>,
    pub timeout_seconds: u32,
    pub retry_count: u32,
    pub when_condition: Option<String>,
}

/// 构建产物配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactConfig {
    pub name: String,
    pub paths: Vec<String>,
    pub expire_in: Option<String>,
    pub when: ArtifactWhen,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactWhen {
    Always,
    OnSuccess,
    OnFailure,
}

/// 管道配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub name: String,
    pub triggers: Vec<TriggerConfig>,
    pub variables: HashMap<String, String>,
    pub jobs: Vec<PipelineJob>,
    pub notifications: Vec<NotificationConfig>,
    pub deployment_environments: Vec<DeploymentEnvironmentConfig>,
}

/// 触发器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerConfig {
    pub trigger_type: TriggerType,
    pub branches: Vec<String>,
    pub paths: Vec<String>,
    pub schedule: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerType {
    Push,
    PullRequest,
    Tag,
    Schedule,
    Manual,
    Webhook,
}

/// 通知配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    pub notification_type: NotificationType,
    pub recipients: Vec<String>,
    pub when: Vec<PipelineStatus>,
    pub template: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationType {
    Email,
    Slack,
    Teams,
    Discord,
    Webhook,
}

/// 部署环境配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentEnvironmentConfig {
    pub name: DeploymentEnvironment,
    pub auto_deploy: bool,
    pub approval_required: bool,
    pub approvers: Vec<String>,
    pub deployment_strategy: DeploymentStrategy,
    pub health_checks: Vec<HealthCheckConfig>,
    pub rollback_on_failure: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStrategy {
    RollingUpdate,
    BlueGreen,
    Canary { percentage: u32 },
    Recreate,
}

/// 健康检查配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    pub name: String,
    pub endpoint: String,
    pub timeout_seconds: u32,
    pub interval_seconds: u32,
    pub retries: u32,
    pub expected_status: u16,
}

/// 管道执行结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineExecution {
    pub id: String,
    pub pipeline_name: String,
    pub status: PipelineStatus,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub duration_seconds: Option<u64>,
    pub commit_sha: String,
    pub branch: String,
    pub triggered_by: String,
    pub jobs: Vec<JobExecution>,
    pub artifacts: Vec<Artifact>,
}

/// 作业执行结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobExecution {
    pub name: String,
    pub stage: PipelineStage,
    pub status: PipelineStatus,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub duration_seconds: Option<u64>,
    pub log_output: String,
    pub exit_code: Option<i32>,
    pub artifacts: Vec<String>,
}

/// 构建产物
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
    pub checksum: String,
    pub created_at: DateTime<Utc>,
}

/// GitHub Actions生成器
pub struct GitHubActionsGenerator;

impl GitHubActionsGenerator {
    pub fn generate_workflow(config: &PipelineConfig) -> Result<String> {
        let mut workflow = format!(r#"name: {}

on:"#, config.name);

        // 添加触发器
        for trigger in &config.triggers {
            match trigger.trigger_type {
                TriggerType::Push => {
                    workflow.push_str(&format!(r#"
  push:
    branches: [{}]"#, trigger.branches.join(", ")));
                }
                TriggerType::PullRequest => {
                    workflow.push_str(&format!(r#"
  pull_request:
    branches: [{}]"#, trigger.branches.join(", ")));
                }
                TriggerType::Schedule => {
                    if let Some(schedule) = &trigger.schedule {
                        workflow.push_str(&format!(r#"
  schedule:
    - cron: '{}'"#, schedule));
                    }
                }
                _ => {}
            }
        }

        // 添加环境变量
        if !config.variables.is_empty() {
            workflow.push_str("\n\nenv:");
            for (key, value) in &config.variables {
                workflow.push_str(&format!("\n  {}: {}", key, value));
            }
        }

        workflow.push_str("\n\njobs:");

        // 按阶段分组作业
        let mut jobs_by_stage = HashMap::new();
        for job in &config.jobs {
            jobs_by_stage.entry(&job.stage).or_insert_with(Vec::new).push(job);
        }

        // 生成作业定义
        for (stage, jobs) in jobs_by_stage {
            for job in jobs {
                workflow.push_str(&Self::generate_job_yaml(job)?);
            }
        }

        Ok(workflow)
    }

    fn generate_job_yaml(job: &PipelineJob) -> Result<String> {
        let mut job_yaml = format!(r#"
  {}:
    runs-on: ubuntu-latest"#, job.name.replace(" ", "_"));

        // 添加依赖
        if !job.dependencies.is_empty() {
            job_yaml.push_str(&format!(r#"
    needs: [{}]"#, job.dependencies.join(", ")));
        }

        // 添加超时
        job_yaml.push_str(&format!(r#"
    timeout-minutes: {}"#, job.timeout_seconds / 60));

        // 添加环境变量
        if !job.environment.is_empty() {
            job_yaml.push_str("\n    env:");
            for (key, value) in &job.environment {
                job_yaml.push_str(&format!("\n      {}: {}", key, value));
            }
        }

        job_yaml.push_str("\n    steps:");
        job_yaml.push_str(r#"
    - name: Checkout code
      uses: actions/checkout@v4"#);

        // 添加命令步骤
        for (i, command) in job.commands.iter().enumerate() {
            job_yaml.push_str(&format!(r#"
    - name: Step {}
      run: {}"#, i + 1, command));
        }

        // 添加构建产物上传
        for artifact in &job.artifacts {
            job_yaml.push_str(&format!(r#"
    - name: Upload {}
      uses: actions/upload-artifact@v4
      with:
        name: {}
        path: {}"#, 
                artifact.name,
                artifact.name,
                artifact.paths.join("\n        ")
            ));
        }

        Ok(job_yaml)
    }
}

/// GitLab CI生成器
pub struct GitLabCIGenerator;

impl GitLabCIGenerator {
    pub fn generate_pipeline(config: &PipelineConfig) -> Result<String> {
        let mut pipeline = String::new();

        // 添加阶段定义
        let stages: Vec<String> = config.jobs.iter()
            .map(|job| Self::stage_to_string(&job.stage))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        pipeline.push_str(&format!("stages:\n{}\n", 
            stages.iter().map(|s| format!("  - {}", s)).collect::<Vec<_>>().join("\n")
        ));

        // 添加全局变量
        if !config.variables.is_empty() {
            pipeline.push_str("\nvariables:");
            for (key, value) in &config.variables {
                pipeline.push_str(&format!("\n  {}: {}", key, value));
            }
            pipeline.push('\n');
        }

        // 生成作业
        for job in &config.jobs {
            pipeline.push_str(&Self::generate_job_yaml(job)?);
        }

        Ok(pipeline)
    }

    fn generate_job_yaml(job: &PipelineJob) -> Result<String> {
        let mut job_yaml = format!(r#"
{}:
  stage: {}
  script:"#, job.name.replace(" ", "_"), Self::stage_to_string(&job.stage));

        for command in &job.commands {
            job_yaml.push_str(&format!("\n    - {}", command));
        }

        // 添加环境变量
        if !job.environment.is_empty() {
            job_yaml.push_str("\n  variables:");
            for (key, value) in &job.environment {
                job_yaml.push_str(&format!("\n    {}: {}", key, value));
            }
        }

        // 添加构建产物
        if !job.artifacts.is_empty() {
            job_yaml.push_str("\n  artifacts:");
            for artifact in &job.artifacts {
                job_yaml.push_str(&format!("\n    name: {}", artifact.name));
                job_yaml.push_str(&format!("\n    paths:\n{}", 
                    artifact.paths.iter().map(|p| format!("      - {}", p)).collect::<Vec<_>>().join("\n")
                ));
                if let Some(expire) = &artifact.expire_in {
                    job_yaml.push_str(&format!("\n    expire_in: {}", expire));
                }
            }
        }

        // 添加超时
        job_yaml.push_str(&format!("\n  timeout: {}m", job.timeout_seconds / 60));

        // 添加重试
        if job.retry_count > 0 {
            job_yaml.push_str(&format!("\n  retry: {}", job.retry_count));
        }

        // 添加条件
        if let Some(condition) = &job.when_condition {
            job_yaml.push_str(&format!("\n  only:\n    - {}", condition));
        }

        job_yaml.push('\n');
        Ok(job_yaml)
    }

    fn stage_to_string(stage: &PipelineStage) -> String {
        match stage {
            PipelineStage::Build => "build".to_string(),
            PipelineStage::Test => "test".to_string(),
            PipelineStage::SecurityScan => "security".to_string(),
            PipelineStage::Deploy => "deploy".to_string(),
            PipelineStage::PostDeploy => "post_deploy".to_string(),
        }
    }
}

/// 管道执行器
pub struct PipelineExecutor {
    config: PipelineConfig,
}

impl PipelineExecutor {
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    pub async fn execute(&self, commit_sha: String, branch: String, triggered_by: String) -> Result<PipelineExecution> {
        let execution_id = uuid::Uuid::new_v4().to_string();
        let start_time = Utc::now();

        tracing::info!("Starting pipeline execution: {}", execution_id);

        let mut job_executions = Vec::new();
        let mut overall_status = PipelineStatus::Success;

        // 按阶段顺序执行作业
        let stages = vec![
            PipelineStage::Build,
            PipelineStage::Test,
            PipelineStage::SecurityScan,
            PipelineStage::Deploy,
            PipelineStage::PostDeploy,
        ];

        for stage in stages {
            let stage_jobs: Vec<_> = self.config.jobs.iter()
                .filter(|job| job.stage == stage)
                .collect();

            if stage_jobs.is_empty() {
                continue;
            }

            tracing::info!("Executing stage: {:?}", stage);

            // 并行执行同一阶段的作业
            let mut stage_futures = Vec::new();
            for job in stage_jobs {
                stage_futures.push(self.execute_job(job.clone()));
            }

            let stage_results = futures::future::join_all(stage_futures).await;

            for result in stage_results {
                match result {
                    Ok(job_execution) => {
                        if job_execution.status == PipelineStatus::Failed {
                            overall_status = PipelineStatus::Failed;
                        }
                        job_executions.push(job_execution);
                    }
                    Err(e) => {
                        tracing::error!("Job execution failed: {}", e);
                        overall_status = PipelineStatus::Failed;
                    }
                }
            }

            // 如果阶段失败，停止执行
            if overall_status == PipelineStatus::Failed {
                break;
            }
        }

        let end_time = Utc::now();
        let duration = (end_time - start_time).num_seconds() as u64;

        let execution = PipelineExecution {
            id: execution_id,
            pipeline_name: self.config.name.clone(),
            status: overall_status,
            start_time,
            end_time: Some(end_time),
            duration_seconds: Some(duration),
            commit_sha,
            branch,
            triggered_by,
            jobs: job_executions,
            artifacts: Vec::new(), // TODO: 收集构建产物
        };

        // 发送通知
        self.send_notifications(&execution).await?;

        Ok(execution)
    }

    async fn execute_job(&self, job: PipelineJob) -> Result<JobExecution> {
        let start_time = Utc::now();
        tracing::info!("Executing job: {}", job.name);

        let mut log_output = String::new();
        let mut exit_code = 0;

        for (i, command) in job.commands.iter().enumerate() {
            tracing::debug!("Executing command {}: {}", i + 1, command);

            // 简化的命令执行（实际实现需要更复杂的逻辑）
            let output = tokio::process::Command::new("sh")
                .arg("-c")
                .arg(command)
                .envs(&job.environment)
                .output()
                .await?;

            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            log_output.push_str(&format!("Command {}: {}\n", i + 1, command));
            log_output.push_str(&format!("STDOUT:\n{}\n", stdout));
            if !stderr.is_empty() {
                log_output.push_str(&format!("STDERR:\n{}\n", stderr));
            }

            if !output.status.success() {
                exit_code = output.status.code().unwrap_or(-1);
                break;
            }
        }

        let end_time = Utc::now();
        let duration = (end_time - start_time).num_seconds() as u64;
        let status = if exit_code == 0 {
            PipelineStatus::Success
        } else {
            PipelineStatus::Failed
        };

        Ok(JobExecution {
            name: job.name,
            stage: job.stage,
            status,
            start_time,
            end_time: Some(end_time),
            duration_seconds: Some(duration),
            log_output,
            exit_code: Some(exit_code),
            artifacts: job.artifacts.iter().map(|a| a.name.clone()).collect(),
        })
    }

    async fn send_notifications(&self, execution: &PipelineExecution) -> Result<()> {
        for notification in &self.config.notifications {
            if notification.when.contains(&execution.status) {
                match notification.notification_type {
                    NotificationType::Email => {
                        tracing::info!("Sending email notification to: {:?}", notification.recipients);
                        // 实际实现需要集成邮件服务
                    }
                    NotificationType::Slack => {
                        tracing::info!("Sending Slack notification to: {:?}", notification.recipients);
                        // 实际实现需要集成Slack API
                    }
                    NotificationType::Webhook => {
                        tracing::info!("Sending webhook notification to: {:?}", notification.recipients);
                        // 实际实现需要HTTP请求
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }
}

/// 预定义的管道模板
pub struct PipelineTemplates;

impl PipelineTemplates {
    /// Rust应用的标准CI/CD管道
    pub fn rust_application() -> PipelineConfig {
        PipelineConfig {
            name: "Rust Application Pipeline".to_string(),
            triggers: vec![
                TriggerConfig {
                    trigger_type: TriggerType::Push,
                    branches: vec!["main".to_string(), "develop".to_string()],
                    paths: vec!["src/**".to_string(), "Cargo.toml".to_string()],
                    schedule: None,
                },
                TriggerConfig {
                    trigger_type: TriggerType::PullRequest,
                    branches: vec!["main".to_string()],
                    paths: vec!["src/**".to_string()],
                    schedule: None,
                },
            ],
            variables: [
                ("CARGO_TERM_COLOR".to_string(), "always".to_string()),
                ("RUST_BACKTRACE".to_string(), "1".to_string()),
            ].iter().cloned().collect(),
            jobs: vec![
                // Build job
                PipelineJob {
                    name: "build".to_string(),
                    stage: PipelineStage::Build,
                    commands: vec![
                        "cargo build --release".to_string(),
                        "cargo build --release --bin api_server".to_string(),
                    ],
                    environment: HashMap::new(),
                    dependencies: Vec::new(),
                    artifacts: vec![
                        ArtifactConfig {
                            name: "binary".to_string(),
                            paths: vec!["target/release/api_server".to_string()],
                            expire_in: Some("1 week".to_string()),
                            when: ArtifactWhen::OnSuccess,
                        },
                    ],
                    timeout_seconds: 1800,
                    retry_count: 1,
                    when_condition: None,
                },
                // Test job
                PipelineJob {
                    name: "test".to_string(),
                    stage: PipelineStage::Test,
                    commands: vec![
                        "cargo test --workspace --all-features".to_string(),
                        "cargo clippy --all-targets --all-features -- -D warnings".to_string(),
                        "cargo fmt -- --check".to_string(),
                    ],
                    environment: HashMap::new(),
                    dependencies: Vec::new(),
                    artifacts: vec![
                        ArtifactConfig {
                            name: "test-results".to_string(),
                            paths: vec!["target/test-results.xml".to_string()],
                            expire_in: Some("1 week".to_string()),
                            when: ArtifactWhen::Always,
                        },
                    ],
                    timeout_seconds: 1200,
                    retry_count: 1,
                    when_condition: None,
                },
                // Security scan job
                PipelineJob {
                    name: "security_scan".to_string(),
                    stage: PipelineStage::SecurityScan,
                    commands: vec![
                        "cargo audit".to_string(),
                        "cargo deny check".to_string(),
                    ],
                    environment: HashMap::new(),
                    dependencies: Vec::new(),
                    artifacts: Vec::new(),
                    timeout_seconds: 600,
                    retry_count: 0,
                    when_condition: None,
                },
                // Deploy job
                PipelineJob {
                    name: "deploy_staging".to_string(),
                    stage: PipelineStage::Deploy,
                    commands: vec![
                        "docker build -t app:$CI_COMMIT_SHA .".to_string(),
                        "docker tag app:$CI_COMMIT_SHA registry.example.com/app:$CI_COMMIT_SHA".to_string(),
                        "docker push registry.example.com/app:$CI_COMMIT_SHA".to_string(),
                        "kubectl set image deployment/app app=registry.example.com/app:$CI_COMMIT_SHA".to_string(),
                    ],
                    environment: [
                        ("KUBECONFIG".to_string(), "$KUBECONFIG_STAGING".to_string()),
                    ].iter().cloned().collect(),
                    dependencies: vec!["build".to_string(), "test".to_string()],
                    artifacts: Vec::new(),
                    timeout_seconds: 900,
                    retry_count: 2,
                    when_condition: Some("main".to_string()),
                },
            ],
            notifications: vec![
                NotificationConfig {
                    notification_type: NotificationType::Slack,
                    recipients: vec!["#ci-cd".to_string()],
                    when: vec![PipelineStatus::Failed, PipelineStatus::Success],
                    template: Some("Pipeline {{pipeline_name}} {{status}} for commit {{commit_sha}}".to_string()),
                },
            ],
            deployment_environments: vec![
                DeploymentEnvironmentConfig {
                    name: DeploymentEnvironment::Staging,
                    auto_deploy: true,
                    approval_required: false,
                    approvers: Vec::new(),
                    deployment_strategy: DeploymentStrategy::RollingUpdate,
                    health_checks: vec![
                        HealthCheckConfig {
                            name: "api_health".to_string(),
                            endpoint: "http://app.staging.example.com/health".to_string(),
                            timeout_seconds: 10,
                            interval_seconds: 30,
                            retries: 3,
                            expected_status: 200,
                        },
                    ],
                    rollback_on_failure: true,
                },
                DeploymentEnvironmentConfig {
                    name: DeploymentEnvironment::Production,
                    auto_deploy: false,
                    approval_required: true,
                    approvers: vec!["team-lead@example.com".to_string()],
                    deployment_strategy: DeploymentStrategy::BlueGreen,
                    health_checks: vec![
                        HealthCheckConfig {
                            name: "api_health".to_string(),
                            endpoint: "http://app.example.com/health".to_string(),
                            timeout_seconds: 10,
                            interval_seconds: 30,
                            retries: 5,
                            expected_status: 200,
                        },
                    ],
                    rollback_on_failure: true,
                },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_github_actions_generation() {
        let config = PipelineTemplates::rust_application();
        let workflow = GitHubActionsGenerator::generate_workflow(&config).unwrap();
        
        assert!(workflow.contains("name: Rust Application Pipeline"));
        assert!(workflow.contains("on:"));
        assert!(workflow.contains("jobs:"));
        assert!(workflow.contains("build:"));
        assert!(workflow.contains("test:"));
    }

    #[test]
    fn test_gitlab_ci_generation() {
        let config = PipelineTemplates::rust_application();
        let pipeline = GitLabCIGenerator::generate_pipeline(&config).unwrap();
        
        assert!(pipeline.contains("stages:"));
        assert!(pipeline.contains("- build"));
        assert!(pipeline.contains("- test"));
        assert!(pipeline.contains("script:"));
    }

    #[test]
    fn test_pipeline_config_creation() {
        let config = PipelineTemplates::rust_application();
        
        assert_eq!(config.name, "Rust Application Pipeline");
        assert_eq!(config.jobs.len(), 4);
        assert_eq!(config.deployment_environments.len(), 2);
        
        let build_job = config.jobs.iter().find(|j| j.name == "build").unwrap();
        assert_eq!(build_job.stage, PipelineStage::Build);
        assert_eq!(build_job.artifacts.len(), 1);
    }

    #[test]
    fn test_deployment_environment_string_conversion() {
        assert_eq!(DeploymentEnvironment::Development.as_str(), "dev");
        assert_eq!(DeploymentEnvironment::Testing.as_str(), "test");
        assert_eq!(DeploymentEnvironment::Staging.as_str(), "staging");
        assert_eq!(DeploymentEnvironment::Production.as_str(), "prod");
        assert_eq!(DeploymentEnvironment::Custom("custom".to_string()).as_str(), "custom");
    }

    #[tokio::test]
    async fn test_pipeline_execution() {
        let mut config = PipelineTemplates::rust_application();
        
        // 简化配置以便测试
        config.jobs = vec![
            PipelineJob {
                name: "simple_test".to_string(),
                stage: PipelineStage::Test,
                commands: vec!["echo 'Hello, World!'".to_string()],
                environment: HashMap::new(),
                dependencies: Vec::new(),
                artifacts: Vec::new(),
                timeout_seconds: 60,
                retry_count: 0,
                when_condition: None,
            },
        ];
        config.notifications.clear();

        let executor = PipelineExecutor::new(config);
        let result = executor.execute(
            "abc123".to_string(),
            "main".to_string(),
            "test_user".to_string(),
        ).await.unwrap();

        assert_eq!(result.status, PipelineStatus::Success);
        assert_eq!(result.jobs.len(), 1);
        assert_eq!(result.jobs[0].status, PipelineStatus::Success);
    }
}