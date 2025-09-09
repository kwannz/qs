use anyhow::{Result, Context};
use chrono::Timelike;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use tracing::debug;

use super::experiment_manager::{ExperimentConfig, AllocationStrategy, TrafficSplitter};

/// 流量分配器实现
#[derive(Debug)]
pub struct DefaultTrafficSplitter {
    // 用户分配缓存
    assignment_cache: Arc<RwLock<HashMap<String, UserAssignment>>>,
    // 配置
    sticky_sessions_enabled: bool,
    cache_ttl_seconds: u64,
}

use std::sync::Arc;
use tokio::sync::RwLock;

/// 用户分配记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAssignment {
    pub user_id: String,
    pub experiment_id: String,
    pub variant_id: String,
    pub assignment_time: i64,
    pub assignment_method: AssignmentMethod,
    pub user_attributes: HashMap<String, serde_json::Value>,
}

/// 分配方法
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AssignmentMethod {
    Random,
    DeterministicHash,
    Geographic,
    TimeBased,
    AttributeBased,
    Stratified,
}

impl DefaultTrafficSplitter {
    pub fn new(sticky_sessions: bool, cache_ttl_seconds: u64) -> Self {
        Self {
            assignment_cache: Arc::new(RwLock::new(HashMap::new())),
            sticky_sessions_enabled: sticky_sessions,
            cache_ttl_seconds,
        }
    }

    /// 基于确定性哈希的分配
    fn deterministic_assignment(&self, user_id: &str, experiment: &ExperimentConfig) -> Result<String> {
        // 创建一个稳定的哈希种子
        let mut hasher = DefaultHasher::new();
        user_id.hash(&mut hasher);
        experiment.id.hash(&mut hasher);
        let hash_value = hasher.finish();
        
        // 将哈希值转换为0-1之间的概率
        let probability = (hash_value % 10000) as f64 / 10000.0;
        
        // 根据权重分配变体
        let mut cumulative_weight = 0.0;
        for (variant_id, weight) in &experiment.traffic_allocation.variant_weights {
            cumulative_weight += weight;
            if probability <= cumulative_weight {
                return Ok(variant_id.clone());
            }
        }
        
        // 如果没有匹配到（不应该发生），返回控制组
        Ok(experiment.control_group.clone())
    }

    /// 基于地理位置的分配
    fn geographic_assignment(&self, _user_id: &str, _user_location: Option<&str>, experiment: &ExperimentConfig) -> Result<String> {
        // 简化实现：这里应该根据用户的地理位置进行分配
        // 例如：不同地区分配到不同变体
        
        // 暂时返回随机分配
        self.random_assignment(experiment)
    }

    /// 基于时间的分配
    fn time_based_assignment(&self, experiment: &ExperimentConfig) -> Result<String> {
        let current_hour = chrono::Utc::now().hour();
        
        // 根据时间段分配不同变体
        let variants: Vec<&String> = experiment.traffic_allocation.variant_weights.keys().collect();
        if variants.is_empty() {
            return Ok(experiment.control_group.clone());
        }
        
        let variant_index = (current_hour as usize) % variants.len();
        Ok(variants[variant_index].clone())
    }

    /// 基于用户属性的分配
    fn attribute_based_assignment(
        &self, 
        user_attributes: &HashMap<String, serde_json::Value>, 
        experiment: &ExperimentConfig
    ) -> Result<String> {
        // 根据用户属性进行分配
        // 例如：高价值用户分配到特定变体
        
        if let Some(user_type) = user_attributes.get("user_type") {
            if user_type.as_str() == Some("premium") {
                // 高价值用户优先分配到新功能变体
                for (variant_id, variant) in experiment.variants.iter()
                    .map(|v| (&v.id, v))
                    .filter(|(_, v)| !v.is_control) {
                    if experiment.traffic_allocation.variant_weights.contains_key(variant_id) {
                        return Ok(variant_id.clone());
                    }
                }
            }
        }
        
        // 默认随机分配
        self.random_assignment(experiment)
    }

    /// 分层抽样分配
    fn stratified_assignment(
        &self, 
        user_attributes: &HashMap<String, serde_json::Value>, 
        experiment: &ExperimentConfig
    ) -> Result<String> {
        // 根据用户分层进行均匀分配
        // 例如：按年龄段、地区、用户类型等分层
        
        let mut stratum_hash = 0u64;
        
        // 根据关键属性计算分层
        if let Some(age_group) = user_attributes.get("age_group") {
            if let Some(age_str) = age_group.as_str() {
                let mut hasher = DefaultHasher::new();
                age_str.hash(&mut hasher);
                stratum_hash ^= hasher.finish();
            }
        }
        
        if let Some(region) = user_attributes.get("region") {
            if let Some(region_str) = region.as_str() {
                let mut hasher = DefaultHasher::new();
                region_str.hash(&mut hasher);
                stratum_hash ^= hasher.finish();
            }
        }
        
        // 在分层内进行随机分配
        let probability = (stratum_hash % 10000) as f64 / 10000.0;
        let mut cumulative_weight = 0.0;
        
        for (variant_id, weight) in &experiment.traffic_allocation.variant_weights {
            cumulative_weight += weight;
            if probability <= cumulative_weight {
                return Ok(variant_id.clone());
            }
        }
        
        Ok(experiment.control_group.clone())
    }

    /// 随机分配
    fn random_assignment(&self, experiment: &ExperimentConfig) -> Result<String> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let random_value: f64 = rng.gen();
        
        let mut cumulative_weight = 0.0;
        for (variant_id, weight) in &experiment.traffic_allocation.variant_weights {
            cumulative_weight += weight;
            if random_value <= cumulative_weight {
                return Ok(variant_id.clone());
            }
        }
        
        Ok(experiment.control_group.clone())
    }

    /// 检查用户是否符合包含条件
    fn check_inclusion_criteria(
        &self, 
        user_attributes: &HashMap<String, serde_json::Value>, 
        experiment: &ExperimentConfig
    ) -> bool {
        if experiment.traffic_allocation.inclusion_criteria.is_empty() {
            return true;
        }
        
        for criterion in &experiment.traffic_allocation.inclusion_criteria {
            if !self.evaluate_criterion(user_attributes, criterion) {
                return false;
            }
        }
        
        true
    }

    /// 检查用户是否符合排除条件
    fn check_exclusion_criteria(
        &self, 
        user_attributes: &HashMap<String, serde_json::Value>, 
        experiment: &ExperimentConfig
    ) -> bool {
        for criterion in &experiment.traffic_allocation.exclusion_criteria {
            if self.evaluate_exclusion_criterion(user_attributes, criterion) {
                return true; // 用户应该被排除
            }
        }
        
        false // 用户不应该被排除
    }

    /// 评估包含条件
    fn evaluate_criterion(
        &self,
        user_attributes: &HashMap<String, serde_json::Value>,
        criterion: &super::experiment_manager::InclusionCriterion,
    ) -> bool {
        use super::experiment_manager::ComparisonOperator;
        
        let user_value = match user_attributes.get(&criterion.field) {
            Some(value) => value,
            None => return false,
        };
        
        match &criterion.operator {
            ComparisonOperator::Equal => user_value == &criterion.value,
            ComparisonOperator::NotEqual => user_value != &criterion.value,
            ComparisonOperator::GreaterThan => {
                if let (Some(user_num), Some(criterion_num)) = (user_value.as_f64(), criterion.value.as_f64()) {
                    user_num > criterion_num
                } else {
                    false
                }
            },
            ComparisonOperator::LessThan => {
                if let (Some(user_num), Some(criterion_num)) = (user_value.as_f64(), criterion.value.as_f64()) {
                    user_num < criterion_num
                } else {
                    false
                }
            },
            ComparisonOperator::GreaterThanOrEqual => {
                if let (Some(user_num), Some(criterion_num)) = (user_value.as_f64(), criterion.value.as_f64()) {
                    user_num >= criterion_num
                } else {
                    false
                }
            },
            ComparisonOperator::LessThanOrEqual => {
                if let (Some(user_num), Some(criterion_num)) = (user_value.as_f64(), criterion.value.as_f64()) {
                    user_num <= criterion_num
                } else {
                    false
                }
            },
            ComparisonOperator::In => {
                if let Some(array) = criterion.value.as_array() {
                    array.contains(user_value)
                } else {
                    false
                }
            },
            ComparisonOperator::NotIn => {
                if let Some(array) = criterion.value.as_array() {
                    !array.contains(user_value)
                } else {
                    true
                }
            },
            ComparisonOperator::Contains => {
                if let (Some(user_str), Some(criterion_str)) = (user_value.as_str(), criterion.value.as_str()) {
                    user_str.contains(criterion_str)
                } else {
                    false
                }
            },
            ComparisonOperator::StartsWith => {
                if let (Some(user_str), Some(criterion_str)) = (user_value.as_str(), criterion.value.as_str()) {
                    user_str.starts_with(criterion_str)
                } else {
                    false
                }
            },
            ComparisonOperator::EndsWith => {
                if let (Some(user_str), Some(criterion_str)) = (user_value.as_str(), criterion.value.as_str()) {
                    user_str.ends_with(criterion_str)
                } else {
                    false
                }
            },
        }
    }

    /// 评估排除条件
    fn evaluate_exclusion_criterion(
        &self,
        user_attributes: &HashMap<String, serde_json::Value>,
        criterion: &super::experiment_manager::ExclusionCriterion,
    ) -> bool {
        // 排除条件的评估逻辑与包含条件类似
        let user_value = match user_attributes.get(&criterion.field) {
            Some(value) => value,
            None => return false,
        };
        
        use super::experiment_manager::ComparisonOperator;
        match &criterion.operator {
            ComparisonOperator::Equal => user_value == &criterion.value,
            ComparisonOperator::NotEqual => user_value != &criterion.value,
            // ... 其他操作符的实现与包含条件相同
            _ => false, // 简化实现
        }
    }

    /// 获取用户属性（模拟实现）
    async fn get_user_attributes(&self, user_id: &str) -> Result<HashMap<String, serde_json::Value>> {
        // 在实际实现中，这里会从用户服务或数据库获取用户属性
        let mut attributes = HashMap::new();
        
        // 模拟一些用户属性
        attributes.insert("user_id".to_string(), serde_json::Value::String(user_id.to_string()));
        attributes.insert("user_type".to_string(), serde_json::Value::String("standard".to_string()));
        attributes.insert("region".to_string(), serde_json::Value::String("US".to_string()));
        attributes.insert("age_group".to_string(), serde_json::Value::String("25-34".to_string()));
        
        Ok(attributes)
    }

    /// 缓存用户分配
    async fn cache_assignment(&self, assignment: UserAssignment) -> Result<()> {
        if !self.sticky_sessions_enabled {
            return Ok(());
        }
        
        let cache_key = format!("{}:{}", assignment.user_id, assignment.experiment_id);
        let mut cache = self.assignment_cache.write().await;
        cache.insert(cache_key, assignment);
        
        Ok(())
    }

    /// 获取缓存的分配
    async fn get_cached_assignment(&self, user_id: &str, experiment_id: &str) -> Option<UserAssignment> {
        if !self.sticky_sessions_enabled {
            return None;
        }
        
        let cache_key = format!("{}:{}", user_id, experiment_id);
        let cache = self.assignment_cache.read().await;
        
        if let Some(assignment) = cache.get(&cache_key) {
            // 检查缓存是否过期
            let current_time = chrono::Utc::now().timestamp();
            let assignment_age = current_time - assignment.assignment_time;
            
            if assignment_age < self.cache_ttl_seconds as i64 {
                return Some(assignment.clone());
            }
        }
        
        None
    }

    /// 清理过期缓存
    async fn cleanup_expired_cache(&self) {
        if !self.sticky_sessions_enabled {
            return;
        }
        
        let current_time = chrono::Utc::now().timestamp();
        let mut cache = self.assignment_cache.write().await;
        
        cache.retain(|_, assignment| {
            let assignment_age = current_time - assignment.assignment_time;
            assignment_age < self.cache_ttl_seconds as i64
        });
    }
}

impl TrafficSplitter for DefaultTrafficSplitter {
    fn assign_variant(&self, user_id: &str, experiment: &ExperimentConfig) -> Result<String> {
        // 异步方法需要在运行时中执行
        let rt = tokio::runtime::Handle::try_current()
            .or_else(|_| tokio::runtime::Runtime::new().map(|rt| rt.handle().clone()))
            .context("Failed to get tokio runtime")?;
        
        rt.block_on(async {
            // 检查缓存的分配
            if let Some(cached) = self.get_cached_assignment(user_id, &experiment.id).await {
                debug!("Using cached assignment for user {} in experiment {}: {}", 
                       user_id, experiment.id, cached.variant_id);
                return Ok(cached.variant_id);
            }
            
            // 获取用户属性
            let user_attributes = self.get_user_attributes(user_id).await?;
            
            // 检查包含条件
            if !self.check_inclusion_criteria(&user_attributes, experiment) {
                return Err(anyhow::anyhow!("User does not meet inclusion criteria"));
            }
            
            // 检查排除条件
            if self.check_exclusion_criteria(&user_attributes, experiment) {
                return Err(anyhow::anyhow!("User meets exclusion criteria"));
            }
            
            // 根据分配策略进行分配
            let variant_id = match &experiment.traffic_allocation.allocation_strategy {
                AllocationStrategy::Random => self.random_assignment(experiment)?,
                AllocationStrategy::DeterministicHashing => self.deterministic_assignment(user_id, experiment)?,
                AllocationStrategy::GeographicSplit => {
                    let location = user_attributes.get("region").and_then(|v| v.as_str());
                    self.geographic_assignment(user_id, location, experiment)?
                },
                AllocationStrategy::TimeBased => self.time_based_assignment(experiment)?,
                AllocationStrategy::UserAttribute => self.attribute_based_assignment(&user_attributes, experiment)?,
                AllocationStrategy::Stratified => self.stratified_assignment(&user_attributes, experiment)?,
            };
            
            // 创建分配记录
            let assignment = UserAssignment {
                user_id: user_id.to_string(),
                experiment_id: experiment.id.clone(),
                variant_id: variant_id.clone(),
                assignment_time: chrono::Utc::now().timestamp(),
                assignment_method: match &experiment.traffic_allocation.allocation_strategy {
                    AllocationStrategy::Random => AssignmentMethod::Random,
                    AllocationStrategy::DeterministicHashing => AssignmentMethod::DeterministicHash,
                    AllocationStrategy::GeographicSplit => AssignmentMethod::Geographic,
                    AllocationStrategy::TimeBased => AssignmentMethod::TimeBased,
                    AllocationStrategy::UserAttribute => AssignmentMethod::AttributeBased,
                    AllocationStrategy::Stratified => AssignmentMethod::Stratified,
                },
                user_attributes,
            };
            
            // 缓存分配
            self.cache_assignment(assignment).await?;
            
            debug!("Assigned user {} to variant {} in experiment {} using {:?}", 
                   user_id, variant_id, experiment.id, experiment.traffic_allocation.allocation_strategy);
            
            Ok(variant_id)
        })
    }

    fn get_assignment(&self, user_id: &str, experiment_id: &str) -> Result<Option<String>> {
        let rt = tokio::runtime::Handle::try_current()
            .or_else(|_| tokio::runtime::Runtime::new().map(|rt| rt.handle().clone()))
            .context("Failed to get tokio runtime")?;
        
        rt.block_on(async {
            if let Some(assignment) = self.get_cached_assignment(user_id, experiment_id).await {
                Ok(Some(assignment.variant_id))
            } else {
                Ok(None)
            }
        })
    }
}

/// 流量分配统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficAllocationStats {
    pub experiment_id: String,
    pub total_assignments: u64,
    pub variant_assignments: HashMap<String, u64>,
    pub assignment_rate: f64,
    pub exclusion_rate: f64,
    pub cache_hit_rate: f64,
    pub assignment_methods: HashMap<AssignmentMethod, u64>,
}

impl DefaultTrafficSplitter {
    /// 获取流量分配统计
    pub async fn get_allocation_stats(&self, experiment_id: &str) -> Result<TrafficAllocationStats> {
        let cache = self.assignment_cache.read().await;
        
        let mut total_assignments = 0u64;
        let mut variant_assignments = HashMap::new();
        let mut assignment_methods = HashMap::new();
        
        for assignment in cache.values() {
            if assignment.experiment_id == experiment_id {
                total_assignments += 1;
                *variant_assignments.entry(assignment.variant_id.clone()).or_insert(0) += 1;
                *assignment_methods.entry(assignment.assignment_method.clone()).or_insert(0) += 1;
            }
        }
        
        Ok(TrafficAllocationStats {
            experiment_id: experiment_id.to_string(),
            total_assignments,
            variant_assignments,
            assignment_rate: 1.0, // 简化计算
            exclusion_rate: 0.0,  // 简化计算
            cache_hit_rate: 0.0,  // 简化计算
            assignment_methods,
        })
    }

    /// 启动缓存清理任务
    pub fn start_cache_cleanup_task(&self) {
        let cache_cleanup_interval = Duration::from_secs(self.cache_ttl_seconds / 10); // 每10%的TTL时间清理一次
        let cache = Arc::clone(&self.assignment_cache);
        let cache_ttl_seconds = self.cache_ttl_seconds;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(cache_cleanup_interval);
            loop {
                interval.tick().await;
                // 简化的缓存清理逻辑
                let mut cache_guard = cache.write().await;
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                cache_guard.retain(|_key, entry| {
                    // 简化实现：假设assignment_time是timestamp秒数
                    let assignment_time_secs = entry.assignment_time as u64;
                    now - assignment_time_secs < cache_ttl_seconds
                });
            }
        });
    }
}

use tokio::time::Duration;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ab_testing::experiment_manager::{
        ExperimentVariant, InclusionCriterion, ExclusionCriterion,
        TrafficAllocation, AllocationStrategy
    };

    fn create_test_experiment() -> ExperimentConfig {
        ExperimentConfig {
            id: "test_experiment".to_string(),
            name: "Test Experiment".to_string(),
            description: "Test experiment".to_string(),
            hypothesis: "Test hypothesis".to_string(),
            owner: "test_owner".to_string(),
            status: super::super::experiment_manager::ExperimentStatus::Running,
            traffic_allocation: TrafficAllocation {
                allocation_strategy: AllocationStrategy::DeterministicHashing,
                variant_weights: [
                    ("control".to_string(), 0.5),
                    ("treatment".to_string(), 0.5),
                ].into_iter().collect(),
                inclusion_criteria: vec![],
                exclusion_criteria: vec![],
                sticky_assignment: true,
            },
            variants: vec![
                ExperimentVariant {
                    id: "control".to_string(),
                    name: "Control".to_string(),
                    description: "Control variant".to_string(),
                    is_control: true,
                    traffic_percentage: 50.0,
                    config_overrides: HashMap::new(),
                    feature_flags: HashMap::new(),
                },
                ExperimentVariant {
                    id: "treatment".to_string(),
                    name: "Treatment".to_string(),
                    description: "Treatment variant".to_string(),
                    is_control: false,
                    traffic_percentage: 50.0,
                    config_overrides: HashMap::new(),
                    feature_flags: HashMap::new(),
                },
            ],
            control_group: "control".to_string(),
            target_metrics: vec![], // 简化测试
            start_time: chrono::Utc::now().timestamp_millis(),
            end_time: chrono::Utc::now().timestamp_millis() + 7 * 24 * 3600 * 1000,
            duration_days: 7,
            ramp_up_duration: Duration::from_secs(3600),
            ramp_down_duration: Duration::from_secs(3600),
            statistical_config: super::super::experiment_manager::StatisticalConfig {
                test_type: super::super::experiment_manager::StatisticalTest::TTest,
                multiple_testing_correction: super::super::experiment_manager::MultipleTesting::None,
                sequential_testing_enabled: false,
                bayesian_analysis_enabled: false,
                confidence_interval_type: super::super::experiment_manager::ConfidenceIntervalType::Normal,
                bootstrap_samples: 1000,
            },
            significance_level: 0.05,
            minimum_detectable_effect: 0.05,
            power: 0.8,
            safety_rules: vec![],
            circuit_breaker: super::super::experiment_manager::CircuitBreakerConfig {
                enabled: false,
                failure_threshold: 0.1,
                recovery_threshold: 0.05,
                check_interval: Duration::from_secs(60),
                half_open_max_requests: 100,
            },
            rollback_conditions: vec![],
            monitoring_config: super::super::experiment_manager::MonitoringConfig {
                metrics_collection_interval: Duration::from_secs(60),
                real_time_monitoring: true,
                dashboard_refresh_interval: Duration::from_secs(30),
                data_quality_checks: vec![],
            },
            alert_conditions: vec![],
        }
    }

    #[test]
    fn test_deterministic_assignment() {
        let splitter = DefaultTrafficSplitter::new(true, 3600);
        let experiment = create_test_experiment();
        
        // 同一用户应该始终分配到相同变体
        let user_id = "test_user_123";
        let variant1 = splitter.assign_variant(user_id, &experiment).unwrap();
        let variant2 = splitter.assign_variant(user_id, &experiment).unwrap();
        
        assert_eq!(variant1, variant2);
        assert!(variant1 == "control" || variant1 == "treatment");
    }

    #[tokio::test]
    async fn test_cache_functionality() {
        let splitter = DefaultTrafficSplitter::new(true, 3600);
        let experiment = create_test_experiment();
        let user_id = "test_user_456";
        
        // 第一次分配
        let variant1 = splitter.assign_variant(user_id, &experiment).unwrap();
        
        // 检查缓存
        let cached_variant = splitter.get_assignment(user_id, &experiment.id).unwrap();
        assert_eq!(Some(variant1.clone()), cached_variant);
        
        // 第二次分配应该使用缓存
        let variant2 = splitter.assign_variant(user_id, &experiment).unwrap();
        assert_eq!(variant1, variant2);
    }

    #[test]
    fn test_random_assignment_distribution() {
        let splitter = DefaultTrafficSplitter::new(false, 3600); // 禁用缓存
        let mut experiment = create_test_experiment();
        experiment.traffic_allocation.allocation_strategy = AllocationStrategy::Random;
        
        let mut control_count = 0;
        let mut treatment_count = 0;
        
        // 进行1000次分配
        for i in 0..1000 {
            let user_id = format!("user_{}", i);
            let variant = splitter.assign_variant(&user_id, &experiment).unwrap();
            
            match variant.as_str() {
                "control" => control_count += 1,
                "treatment" => treatment_count += 1,
                _ => panic!("Unexpected variant: {}", variant),
            }
        }
        
        // 检查分布是否合理（应该接近50:50）
        let total = control_count + treatment_count;
        let control_ratio = control_count as f64 / total as f64;
        
        assert!(control_ratio > 0.4 && control_ratio < 0.6, 
                "Control ratio {} is too far from 0.5", control_ratio);
    }
}