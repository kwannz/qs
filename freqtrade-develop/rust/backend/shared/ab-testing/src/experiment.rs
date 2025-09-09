//! 实验定义和管理

use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// A/B测试实验
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub config: ExperimentConfig,
    pub status: ExperimentStatus,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub ended_at: Option<DateTime<Utc>>,
    pub created_by: String,
    pub variants: Vec<ExperimentVariant>,
    pub metrics: Vec<ExperimentMetric>,
    pub metadata: HashMap<String, String>,
}

/// 实验配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// 流量分配策略
    pub traffic_allocation: TrafficAllocation,
    
    /// 最小样本量
    pub minimum_sample_size: usize,
    
    /// 最大运行时间
    pub max_duration_days: u32,
    
    /// 统计显著性水平
    pub significance_level: f64,
    
    /// 统计功效
    pub statistical_power: f64,
    
    /// 最小可检测效应
    pub minimum_detectable_effect: f64,
    
    /// 是否启用自动停止
    pub auto_stop_enabled: bool,
    
    /// 自动停止配置
    pub auto_stop_config: Option<AutoStopConfig>,
    
    /// 实验类型
    pub experiment_type: ExperimentType,
    
    /// 目标人群
    pub target_audience: TargetAudience,
}

/// 流量分配策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficAllocation {
    pub strategy: AllocationStrategy,
    pub variant_weights: HashMap<String, f64>,
    pub sticky_bucketing: bool,
    pub holdout_percentage: f64,
}

/// 分配策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// 随机分配
    Random,
    /// 基于用户ID哈希
    UserIdHash,
    /// 基于会话ID哈希
    SessionIdHash,
    /// 自定义哈希键
    CustomHash { hash_key: String },
}

/// 自动停止配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoStopConfig {
    /// 检查间隔（小时）
    pub check_interval_hours: u32,
    
    /// 早停条件
    pub early_stop_conditions: Vec<EarlyStopCondition>,
    
    /// 风险控制阈值
    pub risk_threshold: f64,
    
    /// 最小运行时间（防止过早停止）
    pub minimum_runtime_hours: u32,
}

/// 早停条件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStopCondition {
    pub condition_type: EarlyStopType,
    pub threshold: f64,
    pub consecutive_checks: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EarlyStopType {
    /// 统计显著性达到
    StatisticalSignificance,
    /// 风险过高
    HighRisk,
    /// 效果太小
    NegligibleEffect,
    /// 样本量充足
    AdequateSampleSize,
}

/// 实验类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExperimentType {
    /// 模型A/B测试
    ModelAB {
        control_model_id: String,
        treatment_model_id: String,
        model_type: String,
    },
    /// 参数优化测试
    ParameterOptimization {
        base_model_id: String,
        parameter_variants: HashMap<String, serde_json::Value>,
    },
    /// 执行算法测试
    ExecutionAlgorithm {
        algorithms: HashMap<String, String>,
    },
    /// 特征实验
    FeatureExperiment {
        base_features: Vec<String>,
        experimental_features: HashMap<String, Vec<String>>,
    },
}

/// 目标人群
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetAudience {
    pub inclusion_criteria: Vec<AudienceCriterion>,
    pub exclusion_criteria: Vec<AudienceCriterion>,
    pub expected_daily_volume: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudienceCriterion {
    pub criterion_type: String,
    pub operator: String,
    pub value: serde_json::Value,
}

/// 实验状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExperimentStatus {
    /// 草稿状态
    Draft,
    /// 等待审批
    PendingApproval,
    /// 已批准，等待启动
    Approved,
    /// 正在运行
    Running,
    /// 已暂停
    Paused,
    /// 已完成
    Completed,
    /// 已停止（提前结束）
    Stopped,
    /// 失败
    Failed,
    /// 已归档
    Archived,
}

/// 实验变体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentVariant {
    pub id: String,
    pub name: String,
    pub description: String,
    pub is_control: bool,
    pub traffic_allocation: f64,
    pub configuration: HashMap<String, serde_json::Value>,
    pub model_reference: Option<ModelReference>,
}

/// 模型引用
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelReference {
    pub model_id: String,
    pub model_version: String,
    pub model_type: String,
    pub deployment_config: HashMap<String, serde_json::Value>,
}

/// 实验指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMetric {
    pub name: String,
    pub description: String,
    pub metric_type: MetricType,
    pub aggregation: AggregationType,
    pub is_primary: bool,
    pub higher_is_better: bool,
    pub unit: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    /// 转化率指标
    ConversionRate,
    /// 连续值指标
    Continuous,
    /// 计数指标
    Count,
    /// 比率指标
    Ratio,
    /// 自定义指标
    Custom { calculation: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationType {
    Sum,
    Mean,
    Median,
    P95,
    P99,
    Count,
    UniqueCount,
}

impl Experiment {
    /// 创建新实验
    pub fn new(
        name: String,
        description: String,
        config: ExperimentConfig,
        created_by: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            description,
            config,
            status: ExperimentStatus::Draft,
            created_at: Utc::now(),
            started_at: None,
            ended_at: None,
            created_by,
            variants: Vec::new(),
            metrics: Vec::new(),
            metadata: HashMap::new(),
        }
    }
    
    /// 添加变体
    pub fn add_variant(&mut self, variant: ExperimentVariant) -> Result<()> {
        // 验证变体配置
        self.validate_variant(&variant)?;
        
        // 检查控制组唯一性
        if variant.is_control
            && self.variants.iter().any(|v| v.is_control) {
                anyhow::bail!("只能有一个控制组变体");
            }
        
        self.variants.push(variant);
        Ok(())
    }
    
    /// 添加指标
    pub fn add_metric(&mut self, metric: ExperimentMetric) {
        self.metrics.push(metric);
    }
    
    /// 启动实验
    pub fn start(&mut self) -> Result<()> {
        if self.status != ExperimentStatus::Approved {
            anyhow::bail!("只有已批准的实验才能启动");
        }
        
        self.validate_for_start()?;
        
        self.status = ExperimentStatus::Running;
        self.started_at = Some(Utc::now());
        
        tracing::info!("实验 {} 已启动", self.name);
        Ok(())
    }
    
    /// 停止实验
    pub fn stop(&mut self, reason: Option<String>) -> Result<()> {
        if self.status != ExperimentStatus::Running {
            anyhow::bail!("只有正在运行的实验才能停止");
        }
        
        self.status = ExperimentStatus::Stopped;
        self.ended_at = Some(Utc::now());
        
        if let Some(reason) = reason {
            self.metadata.insert("stop_reason".to_string(), reason);
        }
        
        tracing::info!("实验 {} 已停止", self.name);
        Ok(())
    }
    
    /// 暂停实验
    pub fn pause(&mut self) -> Result<()> {
        if self.status != ExperimentStatus::Running {
            anyhow::bail!("只有正在运行的实验才能暂停");
        }
        
        self.status = ExperimentStatus::Paused;
        tracing::info!("实验 {} 已暂停", self.name);
        Ok(())
    }
    
    /// 恢复实验
    pub fn resume(&mut self) -> Result<()> {
        if self.status != ExperimentStatus::Paused {
            anyhow::bail!("只有暂停的实验才能恢复");
        }
        
        self.status = ExperimentStatus::Running;
        tracing::info!("实验 {} 已恢复", self.name);
        Ok(())
    }
    
    /// 完成实验
    pub fn complete(&mut self) -> Result<()> {
        if !matches!(self.status, ExperimentStatus::Running | ExperimentStatus::Paused) {
            anyhow::bail!("只有运行中或暂停的实验才能完成");
        }
        
        self.status = ExperimentStatus::Completed;
        self.ended_at = Some(Utc::now());
        
        tracing::info!("实验 {} 已完成", self.name);
        Ok(())
    }
    
    /// 检查实验是否过期
    pub fn is_expired(&self) -> bool {
        if let Some(started_at) = self.started_at {
            let max_duration = Duration::days(self.config.max_duration_days as i64);
            Utc::now() - started_at > max_duration
        } else {
            false
        }
    }
    
    /// 获取运行时长
    pub fn get_runtime_duration(&self) -> Option<Duration> {
        if let Some(started_at) = self.started_at {
            let end_time = self.ended_at.unwrap_or_else(Utc::now);
            Some(end_time - started_at)
        } else {
            None
        }
    }
    
    /// 获取控制组变体
    pub fn get_control_variant(&self) -> Option<&ExperimentVariant> {
        self.variants.iter().find(|v| v.is_control)
    }
    
    /// 获取处理组变体
    pub fn get_treatment_variants(&self) -> Vec<&ExperimentVariant> {
        self.variants.iter().filter(|v| !v.is_control).collect()
    }
    
    /// 获取主要指标
    pub fn get_primary_metrics(&self) -> Vec<&ExperimentMetric> {
        self.metrics.iter().filter(|m| m.is_primary).collect()
    }
    
    /// 验证实验配置
    pub fn validate(&self) -> Result<()> {
        // 检查基本配置
        if self.variants.is_empty() {
            anyhow::bail!("实验必须至少有一个变体");
        }
        
        if self.metrics.is_empty() {
            anyhow::bail!("实验必须至少有一个指标");
        }
        
        // 检查控制组
        let control_count = self.variants.iter().filter(|v| v.is_control).count();
        if control_count != 1 {
            anyhow::bail!("实验必须有且仅有一个控制组");
        }
        
        // 检查主要指标
        let primary_metrics_count = self.metrics.iter().filter(|m| m.is_primary).count();
        if primary_metrics_count == 0 {
            anyhow::bail!("实验必须至少有一个主要指标");
        }
        
        // 检查流量分配
        let total_weight: f64 = self.config.traffic_allocation.variant_weights.values().sum();
        if (total_weight - 1.0).abs() > 0.001 {
            anyhow::bail!("变体权重总和必须等于1.0");
        }
        
        // 检查统计参数
        if !(0.0..=1.0).contains(&self.config.significance_level) {
            anyhow::bail!("显著性水平必须在0-1之间");
        }
        
        if !(0.0..=1.0).contains(&self.config.statistical_power) {
            anyhow::bail!("统计功效必须在0-1之间");
        }
        
        Ok(())
    }
    
    fn validate_variant(&self, variant: &ExperimentVariant) -> Result<()> {
        if variant.id.is_empty() {
            anyhow::bail!("变体ID不能为空");
        }
        
        if variant.name.is_empty() {
            anyhow::bail!("变体名称不能为空");
        }
        
        if !(0.0..=1.0).contains(&variant.traffic_allocation) {
            anyhow::bail!("变体流量分配必须在0-1之间");
        }
        
        // 检查ID唯一性
        if self.variants.iter().any(|v| v.id == variant.id) {
            anyhow::bail!("变体ID必须唯一");
        }
        
        Ok(())
    }
    
    fn validate_for_start(&self) -> Result<()> {
        self.validate()?;
        
        // 检查是否有足够的配置
        if self.variants.len() < 2 {
            anyhow::bail!("实验至少需要2个变体才能启动");
        }
        
        // 检查自动停止配置
        if self.config.auto_stop_enabled
            && self.config.auto_stop_config.is_none() {
                anyhow::bail!("启用自动停止时必须配置自动停止参数");
            }
        
        Ok(())
    }
}