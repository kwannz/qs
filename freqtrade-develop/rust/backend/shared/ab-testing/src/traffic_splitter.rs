//! 流量分配器
//! 
//! 负责将用户流量分配到不同的实验变体

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use tracing::debug;

use crate::experiment::{Experiment, AllocationStrategy};
use crate::ABTestError;

/// 流量分配器
pub struct TrafficSplitter {
    experiments: HashMap<uuid::Uuid, Experiment>,
    user_assignments: HashMap<String, UserAssignment>,
}

/// 用户分配记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAssignment {
    pub user_id: String,
    pub experiment_id: uuid::Uuid,
    pub variant_id: String,
    pub assigned_at: chrono::DateTime<chrono::Utc>,
    pub hash_value: u64,
    pub assignment_context: HashMap<String, String>,
}

/// 分配结果
#[derive(Debug, Clone)]
pub struct AllocationResult {
    pub variant_id: Option<String>,
    pub experiment_id: uuid::Uuid,
    pub user_id: String,
    pub assignment_reason: AssignmentReason,
    pub hash_value: u64,
}

/// 分配原因
#[derive(Debug, Clone)]
pub enum AssignmentReason {
    /// 新分配
    NewAssignment,
    /// 粘性分组（重用之前的分配）
    StickyBucketing,
    /// 排除在实验外
    Excluded { reason: String },
    /// 在保留组中
    Holdout,
    /// 实验未运行
    ExperimentNotRunning,
}

/// 分配策略配置
#[derive(Debug, Clone)]
pub struct SplittingStrategy {
    pub hash_seed: u64,
    pub enable_sticky_bucketing: bool,
    pub holdout_percentage: f64,
    pub exclusion_rules: Vec<ExclusionRule>,
}

/// 排除规则
#[derive(Debug, Clone)]
pub struct ExclusionRule {
    pub rule_type: ExclusionRuleType,
    pub condition: String,
    pub values: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ExclusionRuleType {
    UserAttribute,
    UserSegment,
    GeographicLocation,
    DeviceType,
    Custom,
}

/// 分配上下文
#[derive(Debug, Clone)]
pub struct AllocationContext {
    pub user_id: String,
    pub session_id: Option<String>,
    pub user_attributes: HashMap<String, String>,
    pub request_time: chrono::DateTime<chrono::Utc>,
    pub custom_context: HashMap<String, String>,
}

impl TrafficSplitter {
    /// 创建新的流量分配器
    pub fn new() -> Self {
        Self {
            experiments: HashMap::new(),
            user_assignments: HashMap::new(),
        }
    }
    
    /// 添加实验
    pub fn add_experiment(&mut self, experiment: Experiment) {
        self.experiments.insert(experiment.id, experiment);
    }
    
    /// 移除实验
    pub fn remove_experiment(&mut self, experiment_id: &uuid::Uuid) -> Option<Experiment> {
        self.experiments.remove(experiment_id)
    }
    
    /// 为用户分配实验变体
    pub fn allocate_user(
        &mut self,
        experiment_id: &uuid::Uuid,
        context: &AllocationContext,
        strategy: &SplittingStrategy,
    ) -> Result<AllocationResult, ABTestError> {
        let experiment = self.experiments.get(experiment_id)
            .ok_or_else(|| ABTestError::ExperimentNotFound { 
                experiment_id: *experiment_id 
            })?;
        
        // 检查实验是否正在运行
        if !experiment.status.eq(&crate::experiment::ExperimentStatus::Running) {
            return Ok(AllocationResult {
                variant_id: None,
                experiment_id: *experiment_id,
                user_id: context.user_id.clone(),
                assignment_reason: AssignmentReason::ExperimentNotRunning,
                hash_value: 0,
            });
        }
        
        // 检查用户是否已有分配（粘性分组）
        if strategy.enable_sticky_bucketing {
            if let Some(existing_assignment) = self.get_existing_assignment(&context.user_id, experiment_id) {
                return Ok(AllocationResult {
                    variant_id: Some(existing_assignment.variant_id.clone()),
                    experiment_id: *experiment_id,
                    user_id: context.user_id.clone(),
                    assignment_reason: AssignmentReason::StickyBucketing,
                    hash_value: existing_assignment.hash_value,
                });
            }
        }
        
        // 检查排除规则
        if let Some(exclusion_reason) = self.check_exclusion_rules(context, &strategy.exclusion_rules) {
            return Ok(AllocationResult {
                variant_id: None,
                experiment_id: *experiment_id,
                user_id: context.user_id.clone(),
                assignment_reason: AssignmentReason::Excluded { reason: exclusion_reason },
                hash_value: 0,
            });
        }
        
        // 检查目标人群
        if !self.check_target_audience(context, &experiment.config.target_audience) {
            return Ok(AllocationResult {
                variant_id: None,
                experiment_id: *experiment_id,
                user_id: context.user_id.clone(),
                assignment_reason: AssignmentReason::Excluded { 
                    reason: "不符合目标人群条件".to_string() 
                },
                hash_value: 0,
            });
        }
        
        // 计算哈希值
        let hash_value = self.calculate_hash(
            &context.user_id,
            experiment_id,
            &experiment.config.traffic_allocation.strategy,
            context,
            strategy.hash_seed,
        );
        
        // 检查保留组
        if self.is_in_holdout(hash_value, strategy.holdout_percentage) {
            return Ok(AllocationResult {
                variant_id: None,
                experiment_id: *experiment_id,
                user_id: context.user_id.clone(),
                assignment_reason: AssignmentReason::Holdout,
                hash_value,
            });
        }
        
        // 分配变体
        let variant_id = self.assign_variant(hash_value, &experiment.config.traffic_allocation.variant_weights)?;
        
        // 记录分配
        let assignment = UserAssignment {
            user_id: context.user_id.clone(),
            experiment_id: *experiment_id,
            variant_id: variant_id.clone(),
            assigned_at: chrono::Utc::now(),
            hash_value,
            assignment_context: context.user_attributes.clone(),
        };
        
        self.user_assignments.insert(
            format!("{}_{}", context.user_id, experiment_id),
            assignment,
        );
        
        debug!(
            "用户 {} 被分配到实验 {} 的变体 {}",
            context.user_id, experiment_id, variant_id
        );
        
        Ok(AllocationResult {
            variant_id: Some(variant_id),
            experiment_id: *experiment_id,
            user_id: context.user_id.clone(),
            assignment_reason: AssignmentReason::NewAssignment,
            hash_value,
        })
    }
    
    /// 批量分配用户
    pub fn batch_allocate_users(
        &mut self,
        experiment_id: &uuid::Uuid,
        contexts: &[AllocationContext],
        strategy: &SplittingStrategy,
    ) -> Result<Vec<AllocationResult>, ABTestError> {
        let mut results = Vec::new();
        
        for context in contexts {
            let result = self.allocate_user(experiment_id, context, strategy)?;
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// 获取用户的所有实验分配
    pub fn get_user_experiments(&self, user_id: &str) -> Vec<&UserAssignment> {
        self.user_assignments
            .values()
            .filter(|assignment| assignment.user_id == user_id)
            .collect()
    }
    
    /// 获取实验的所有用户分配
    pub fn get_experiment_assignments(&self, experiment_id: &uuid::Uuid) -> Vec<&UserAssignment> {
        self.user_assignments
            .values()
            .filter(|assignment| assignment.experiment_id == *experiment_id)
            .collect()
    }
    
    /// 强制重新分配用户
    pub fn force_reallocate_user(
        &mut self,
        experiment_id: &uuid::Uuid,
        user_id: &str,
        variant_id: &str,
    ) -> Result<(), ABTestError> {
        let experiment = self.experiments.get(experiment_id)
            .ok_or_else(|| ABTestError::ExperimentNotFound { 
                experiment_id: *experiment_id 
            })?;
        
        // 验证变体存在
        if !experiment.variants.iter().any(|v| v.id == variant_id) {
            return Err(ABTestError::TrafficSplittingError {
                reason: format!("变体 {variant_id} 在实验 {experiment_id} 中不存在"),
            });
        }
        
        let assignment = UserAssignment {
            user_id: user_id.to_string(),
            experiment_id: *experiment_id,
            variant_id: variant_id.to_string(),
            assigned_at: chrono::Utc::now(),
            hash_value: 0, // 强制分配不使用哈希
            assignment_context: HashMap::new(),
        };
        
        self.user_assignments.insert(
            format!("{user_id}_{experiment_id}"),
            assignment,
        );
        
        Ok(())
    }
    
    /// 计算分配统计信息
    pub fn get_allocation_stats(&self, experiment_id: &uuid::Uuid) -> Option<AllocationStats> {
        let assignments: Vec<_> = self.get_experiment_assignments(experiment_id);
        
        if assignments.is_empty() {
            return None;
        }
        
        let mut variant_counts = HashMap::new();
        let mut total_assignments = 0;
        
        for assignment in &assignments {
            *variant_counts.entry(assignment.variant_id.clone()).or_insert(0) += 1;
            total_assignments += 1;
        }
        
        let variant_percentages: HashMap<String, f64> = variant_counts
            .iter()
            .map(|(variant_id, count)| {
                (variant_id.clone(), *count as f64 / total_assignments as f64)
            })
            .collect();
        
        Some(AllocationStats {
            total_assignments,
            variant_counts,
            variant_percentages,
            first_assignment: assignments.iter().map(|a| a.assigned_at).min(),
            last_assignment: assignments.iter().map(|a| a.assigned_at).max(),
        })
    }
    
    // 私有辅助方法
    
    fn get_existing_assignment(&self, user_id: &str, experiment_id: &uuid::Uuid) -> Option<&UserAssignment> {
        self.user_assignments.get(&format!("{user_id}_{experiment_id}"))
    }
    
    fn check_exclusion_rules(
        &self,
        context: &AllocationContext,
        exclusion_rules: &[ExclusionRule],
    ) -> Option<String> {
        for rule in exclusion_rules {
            if self.evaluate_exclusion_rule(context, rule) {
                return Some(format!("排除规则: {}", rule.condition));
            }
        }
        None
    }
    
    fn evaluate_exclusion_rule(&self, context: &AllocationContext, rule: &ExclusionRule) -> bool {
        match rule.rule_type {
            ExclusionRuleType::UserAttribute => {
                if let Some(attr_value) = context.user_attributes.get(&rule.condition) {
                    rule.values.contains(attr_value)
                } else {
                    false
                }
            }
            ExclusionRuleType::UserSegment => {
                // 实现用户分群逻辑
                false // 占位符
            }
            ExclusionRuleType::GeographicLocation => {
                // 实现地理位置排除逻辑
                false // 占位符
            }
            ExclusionRuleType::DeviceType => {
                // 实现设备类型排除逻辑
                false // 占位符
            }
            ExclusionRuleType::Custom => {
                // 实现自定义排除逻辑
                false // 占位符
            }
        }
    }
    
    fn check_target_audience(
        &self,
        context: &AllocationContext,
        target_audience: &crate::experiment::TargetAudience,
    ) -> bool {
        // 检查包含条件
        for criterion in &target_audience.inclusion_criteria {
            if !self.evaluate_audience_criterion(context, criterion) {
                return false;
            }
        }
        
        // 检查排除条件
        for criterion in &target_audience.exclusion_criteria {
            if self.evaluate_audience_criterion(context, criterion) {
                return false;
            }
        }
        
        true
    }
    
    fn evaluate_audience_criterion(
        &self,
        context: &AllocationContext,
        criterion: &crate::experiment::AudienceCriterion,
    ) -> bool {
        // 简化实现，实际应根据criterion_type和operator进行复杂判断
        if let Some(attr_value) = context.user_attributes.get(&criterion.criterion_type) {
            match criterion.operator.as_str() {
                "equals" => {
                    if let Some(expected) = criterion.value.as_str() {
                        attr_value == expected
                    } else {
                        false
                    }
                }
                "contains" => {
                    if let Some(expected) = criterion.value.as_str() {
                        attr_value.contains(expected)
                    } else {
                        false
                    }
                }
                "in" => {
                    if let Some(expected_array) = criterion.value.as_array() {
                        expected_array.iter().any(|v| {
                            v.as_str().is_some_and(|s| s == attr_value)
                        })
                    } else {
                        false
                    }
                }
                _ => false,
            }
        } else {
            false
        }
    }
    
    fn calculate_hash(
        &self,
        user_id: &str,
        experiment_id: &uuid::Uuid,
        strategy: &AllocationStrategy,
        context: &AllocationContext,
        seed: u64,
    ) -> u64 {
        let mut hasher = DefaultHasher::new();
        
        // 添加种子
        hasher.write_u64(seed);
        
        // 添加实验ID
        hasher.write(experiment_id.as_bytes());
        
        // 根据策略添加不同的键
        match strategy {
            AllocationStrategy::Random => {
                // 使用当前时间的纳秒作为随机种子
                hasher.write_u64(context.request_time.timestamp_nanos_opt().unwrap_or(0) as u64);
            }
            AllocationStrategy::UserIdHash => {
                hasher.write(user_id.as_bytes());
            }
            AllocationStrategy::SessionIdHash => {
                if let Some(session_id) = &context.session_id {
                    hasher.write(session_id.as_bytes());
                } else {
                    hasher.write(user_id.as_bytes()); // 回退到用户ID
                }
            }
            AllocationStrategy::CustomHash { hash_key } => {
                if let Some(custom_value) = context.custom_context.get(hash_key) {
                    hasher.write(custom_value.as_bytes());
                } else {
                    hasher.write(user_id.as_bytes()); // 回退到用户ID
                }
            }
        }
        
        hasher.finish()
    }
    
    fn is_in_holdout(&self, hash_value: u64, holdout_percentage: f64) -> bool {
        if holdout_percentage <= 0.0 {
            return false;
        }
        
        let hash_ratio = (hash_value % 100000) as f64 / 100000.0;
        hash_ratio < (holdout_percentage / 100.0)
    }
    
    fn assign_variant(&self, hash_value: u64, variant_weights: &HashMap<String, f64>) -> Result<String, ABTestError> {
        let hash_ratio = (hash_value % 100000) as f64 / 100000.0;
        
        // 构建累积分布
        let mut cumulative_weights = BTreeMap::new();
        let mut cumulative = 0.0;
        
        for (variant_id, weight) in variant_weights {
            cumulative += weight;
            cumulative_weights.insert((cumulative * 100000.0) as u64, variant_id.clone());
        }
        
        // 找到对应的变体
        let hash_scaled = (hash_ratio * 100000.0) as u64;
        
        for (threshold, variant_id) in cumulative_weights {
            if hash_scaled <= threshold {
                return Ok(variant_id);
            }
        }
        
        // 如果没有找到，返回第一个变体（不应该发生）
        variant_weights.keys().next()
            .cloned()
            .ok_or_else(|| ABTestError::TrafficSplittingError {
                reason: "没有可分配的变体".to_string(),
            })
    }
}

/// 分配统计信息
#[derive(Debug, Clone)]
pub struct AllocationStats {
    pub total_assignments: usize,
    pub variant_counts: HashMap<String, usize>,
    pub variant_percentages: HashMap<String, f64>,
    pub first_assignment: Option<chrono::DateTime<chrono::Utc>>,
    pub last_assignment: Option<chrono::DateTime<chrono::Utc>>,
}

impl Default for TrafficSplitter {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SplittingStrategy {
    fn default() -> Self {
        Self {
            hash_seed: 42,
            enable_sticky_bucketing: true,
            holdout_percentage: 0.0,
            exclusion_rules: Vec::new(),
        }
    }
}