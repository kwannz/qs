//! 离线策略评估(OPE)实现
//! 支持IPS、DR、WIS等方法评估强化学习策略

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use anyhow::{Result, Context};

/// OPE方法枚举
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OPEMethod {
    /// Inverse Propensity Scoring
    IPS,
    /// Doubly Robust  
    DR,
    /// Weighted Importance Sampling
    WIS,
    /// Self-Normalized Importance Sampling
    SNIPS,
    /// Direct Method
    DM,
}

/// 历史记录项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalRecord {
    pub timestamp: DateTime<Utc>,
    pub context: Vec<f64>,
    pub action: String,
    pub reward: f64,
    pub propensity_score: f64, // 行为策略选择此动作的概率
}

/// 策略定义
pub trait Policy: Send + Sync {
    fn get_action_probability(&self, context: &[f64], action: &str) -> f64;
    fn get_all_action_probabilities(&self, context: &[f64]) -> HashMap<String, f64>;
}

/// 价值函数
pub trait ValueFunction: Send + Sync {
    fn estimate_value(&self, context: &[f64], action: &str) -> f64;
}

/// 评估结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub method: OPEMethod,
    pub estimated_value: f64,
    pub confidence_interval: (f64, f64),
    pub variance: f64,
    pub effective_sample_size: f64,
    pub bias_estimate: Option<f64>,
}

/// OPE评估器
pub struct OfflinePolicyEvaluator {
    historical_data: Vec<HistoricalRecord>,
    target_policy: Box<dyn Policy>,
    value_function: Option<Box<dyn ValueFunction>>,
}

impl OfflinePolicyEvaluator {
    pub fn new(
        historical_data: Vec<HistoricalRecord>,
        target_policy: Box<dyn Policy>,
    ) -> Self {
        Self {
            historical_data,
            target_policy,
            value_function: None,
        }
    }

    pub fn with_value_function(mut self, value_function: Box<dyn ValueFunction>) -> Self {
        self.value_function = Some(value_function);
        self
    }

    /// 评估策略价值
    pub fn evaluate(&self, method: OPEMethod) -> Result<EvaluationResult> {
        match method {
            OPEMethod::IPS => self.inverse_propensity_scoring(),
            OPEMethod::DR => self.doubly_robust(),
            OPEMethod::WIS => self.weighted_importance_sampling(),
            OPEMethod::SNIPS => self.self_normalized_importance_sampling(),
            OPEMethod::DM => self.direct_method(),
        }
    }

    /// Inverse Propensity Scoring
    fn inverse_propensity_scoring(&self) -> Result<EvaluationResult> {
        let mut weighted_rewards = Vec::new();
        let mut importance_weights = Vec::new();

        for record in &self.historical_data {
            let target_prob = self.target_policy.get_action_probability(&record.context, &record.action);
            let behavior_prob = record.propensity_score;
            
            if behavior_prob <= 0.0 {
                continue; // 跳过无效记录
            }

            let importance_weight = target_prob / behavior_prob;
            let weighted_reward = importance_weight * record.reward;
            
            weighted_rewards.push(weighted_reward);
            importance_weights.push(importance_weight);
        }

        if weighted_rewards.is_empty() {
            return Err(anyhow::anyhow!("No valid records for IPS evaluation"));
        }

        let estimated_value = weighted_rewards.iter().sum::<f64>() / weighted_rewards.len() as f64;
        let variance = self.calculate_variance(&weighted_rewards, estimated_value);
        let confidence_interval = self.calculate_confidence_interval(estimated_value, variance, weighted_rewards.len());
        let effective_sample_size = self.calculate_effective_sample_size(&importance_weights);

        Ok(EvaluationResult {
            method: OPEMethod::IPS,
            estimated_value,
            confidence_interval,
            variance,
            effective_sample_size,
            bias_estimate: None,
        })
    }

    /// Weighted Importance Sampling
    fn weighted_importance_sampling(&self) -> Result<EvaluationResult> {
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        let mut importance_weights = Vec::new();

        for record in &self.historical_data {
            let target_prob = self.target_policy.get_action_probability(&record.context, &record.action);
            let behavior_prob = record.propensity_score;
            
            if behavior_prob <= 0.0 {
                continue;
            }

            let importance_weight = target_prob / behavior_prob;
            numerator += importance_weight * record.reward;
            denominator += importance_weight;
            importance_weights.push(importance_weight);
        }

        if denominator == 0.0 {
            return Err(anyhow::anyhow!("Zero denominator in WIS evaluation"));
        }

        let estimated_value = numerator / denominator;
        
        // WIS方差计算更复杂，这里简化
        let variance = self.calculate_wis_variance(&importance_weights, estimated_value);
        let confidence_interval = self.calculate_confidence_interval(estimated_value, variance, importance_weights.len());
        let effective_sample_size = self.calculate_effective_sample_size(&importance_weights);

        Ok(EvaluationResult {
            method: OPEMethod::WIS,
            estimated_value,
            confidence_interval,
            variance,
            effective_sample_size,
            bias_estimate: None,
        })
    }

    /// Doubly Robust
    fn doubly_robust(&self) -> Result<EvaluationResult> {
        let value_function = self.value_function.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Value function required for DR method"))?;

        let mut dr_estimates = Vec::new();
        let mut importance_weights = Vec::new();

        for record in &self.historical_data {
            let target_prob = self.target_policy.get_action_probability(&record.context, &record.action);
            let behavior_prob = record.propensity_score;
            
            if behavior_prob <= 0.0 {
                continue;
            }

            let importance_weight = target_prob / behavior_prob;
            let predicted_value = value_function.estimate_value(&record.context, &record.action);
            
            // DR估计：预测值 + 重要性加权的残差
            let dr_estimate = predicted_value + importance_weight * (record.reward - predicted_value);
            
            dr_estimates.push(dr_estimate);
            importance_weights.push(importance_weight);
        }

        if dr_estimates.is_empty() {
            return Err(anyhow::anyhow!("No valid records for DR evaluation"));
        }

        let estimated_value = dr_estimates.iter().sum::<f64>() / dr_estimates.len() as f64;
        let variance = self.calculate_variance(&dr_estimates, estimated_value);
        let confidence_interval = self.calculate_confidence_interval(estimated_value, variance, dr_estimates.len());
        let effective_sample_size = self.calculate_effective_sample_size(&importance_weights);

        Ok(EvaluationResult {
            method: OPEMethod::DR,
            estimated_value,
            confidence_interval,
            variance,
            effective_sample_size,
            bias_estimate: Some(self.estimate_bias(&dr_estimates)),
        })
    }

    /// Self-Normalized Importance Sampling
    fn self_normalized_importance_sampling(&self) -> Result<EvaluationResult> {
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        let mut importance_weights = Vec::new();

        for record in &self.historical_data {
            let target_prob = self.target_policy.get_action_probability(&record.context, &record.action);
            let behavior_prob = record.propensity_score;
            
            if behavior_prob <= 0.0 {
                continue;
            }

            let importance_weight = target_prob / behavior_prob;
            let normalized_weight = importance_weight / self.historical_data.len() as f64;
            
            numerator += normalized_weight * record.reward;
            denominator += normalized_weight;
            importance_weights.push(importance_weight);
        }

        if denominator == 0.0 {
            return Err(anyhow::anyhow!("Zero denominator in SNIPS evaluation"));
        }

        let estimated_value = numerator / denominator;
        let variance = self.calculate_variance(&importance_weights, 1.0); // 简化的方差计算
        let confidence_interval = self.calculate_confidence_interval(estimated_value, variance, importance_weights.len());
        let effective_sample_size = self.calculate_effective_sample_size(&importance_weights);

        Ok(EvaluationResult {
            method: OPEMethod::SNIPS,
            estimated_value,
            confidence_interval,
            variance,
            effective_sample_size,
            bias_estimate: None,
        })
    }

    /// Direct Method
    fn direct_method(&self) -> Result<EvaluationResult> {
        let value_function = self.value_function.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Value function required for DM method"))?;

        let mut predicted_values = Vec::new();

        for record in &self.historical_data {
            let action_probs = self.target_policy.get_all_action_probabilities(&record.context);
            
            let expected_value: f64 = action_probs.iter()
                .map(|(action, prob)| {
                    prob * value_function.estimate_value(&record.context, action)
                })
                .sum();

            predicted_values.push(expected_value);
        }

        if predicted_values.is_empty() {
            return Err(anyhow::anyhow!("No records for DM evaluation"));
        }

        let estimated_value = predicted_values.iter().sum::<f64>() / predicted_values.len() as f64;
        let variance = self.calculate_variance(&predicted_values, estimated_value);
        let confidence_interval = self.calculate_confidence_interval(estimated_value, variance, predicted_values.len());

        Ok(EvaluationResult {
            method: OPEMethod::DM,
            estimated_value,
            confidence_interval,
            variance,
            effective_sample_size: predicted_values.len() as f64,
            bias_estimate: Some(self.estimate_dm_bias()),
        })
    }

    /// 计算方差
    fn calculate_variance(&self, values: &[f64], mean: f64) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }

        let sum_squared_diff: f64 = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum();

        sum_squared_diff / (values.len() - 1) as f64
    }

    /// WIS方差计算
    fn calculate_wis_variance(&self, weights: &[f64], estimated_value: f64) -> f64 {
        if weights.is_empty() {
            return 0.0;
        }

        let sum_weights: f64 = weights.iter().sum();
        let sum_squared_weights: f64 = weights.iter().map(|w| w.powi(2)).sum();
        
        // 简化的WIS方差估计
        let effective_n = sum_weights.powi(2) / sum_squared_weights;
        estimated_value.abs() / effective_n.sqrt()
    }

    /// 计算置信区间
    fn calculate_confidence_interval(&self, mean: f64, variance: f64, n: usize) -> (f64, f64) {
        if n == 0 {
            return (mean, mean);
        }

        let std_error = (variance / n as f64).sqrt();
        let margin = 1.96 * std_error; // 95% 置信区间

        (mean - margin, mean + margin)
    }

    /// 计算有效样本量
    fn calculate_effective_sample_size(&self, importance_weights: &[f64]) -> f64 {
        if importance_weights.is_empty() {
            return 0.0;
        }

        let sum_weights: f64 = importance_weights.iter().sum();
        let sum_squared_weights: f64 = importance_weights.iter().map(|w| w.powi(2)).sum();

        if sum_squared_weights == 0.0 {
            return 0.0;
        }

        sum_weights.powi(2) / sum_squared_weights
    }

    /// 估计偏差
    fn estimate_bias(&self, estimates: &[f64]) -> f64 {
        // 简化的偏差估计：使用交叉验证方法
        if estimates.len() < 10 {
            return 0.0;
        }

        let mid = estimates.len() / 2;
        let first_half: f64 = estimates[..mid].iter().sum::<f64>() / mid as f64;
        let second_half: f64 = estimates[mid..].iter().sum::<f64>() / (estimates.len() - mid) as f64;

        (first_half - second_half).abs()
    }

    /// DM方法偏差估计
    fn estimate_dm_bias(&self) -> f64 {
        // 模型偏差估计：比较预测值与实际奖励的差异
        let value_function = self.value_function.as_ref().unwrap();
        
        let mut prediction_errors = Vec::new();
        for record in &self.historical_data {
            let predicted = value_function.estimate_value(&record.context, &record.action);
            let error = (predicted - record.reward).abs();
            prediction_errors.push(error);
        }

        if prediction_errors.is_empty() {
            return 0.0;
        }

        prediction_errors.iter().sum::<f64>() / prediction_errors.len() as f64
    }
}

/// 策略比较器
pub struct PolicyComparator {
    evaluator: OfflinePolicyEvaluator,
    methods: Vec<OPEMethod>,
}

impl PolicyComparator {
    pub fn new(
        historical_data: Vec<HistoricalRecord>,
        target_policy: Box<dyn Policy>,
        value_function: Option<Box<dyn ValueFunction>>,
    ) -> Self {
        let mut evaluator = OfflinePolicyEvaluator::new(historical_data, target_policy);
        if let Some(vf) = value_function {
            evaluator = evaluator.with_value_function(vf);
        }

        Self {
            evaluator,
            methods: vec![OPEMethod::IPS, OPEMethod::WIS, OPEMethod::DR, OPEMethod::SNIPS],
        }
    }

    /// 使用多种方法评估策略
    pub fn comprehensive_evaluation(&self) -> Result<Vec<EvaluationResult>> {
        let mut results = Vec::new();

        for method in &self.methods {
            match self.evaluator.evaluate(method.clone()) {
                Ok(result) => results.push(result),
                Err(e) => {
                    tracing::warn!("Failed to evaluate with {:?}: {}", method, e);
                    continue;
                }
            }
        }

        if results.is_empty() {
            return Err(anyhow::anyhow!("All evaluation methods failed"));
        }

        Ok(results)
    }

    /// 获取聚合评估结果
    pub fn get_consensus_estimate(&self) -> Result<ConsensusResult> {
        let results = self.comprehensive_evaluation()?;
        
        if results.is_empty() {
            return Err(anyhow::anyhow!("No evaluation results available"));
        }

        // 根据有效样本量加权平均
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for result in &results {
            let weight = result.effective_sample_size / (1.0 + result.variance);
            weighted_sum += weight * result.estimated_value;
            total_weight += weight;
        }

        let consensus_estimate = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            results.iter().map(|r| r.estimated_value).sum::<f64>() / results.len() as f64
        };

        // 计算方法间的差异
        let values: Vec<f64> = results.iter().map(|r| r.estimated_value).collect();
        let method_variance = self.calculate_method_variance(&values, consensus_estimate);
        
        Ok(ConsensusResult {
            consensus_estimate,
            method_variance,
            individual_results: results,
            confidence_level: self.calculate_confidence_level(&values),
        })
    }

    fn calculate_method_variance(&self, values: &[f64], mean: f64) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }

        let sum_squared_diff: f64 = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum();

        sum_squared_diff / (values.len() - 1) as f64
    }

    fn calculate_confidence_level(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.5;
        }

        // 计算方法间的一致性
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std_dev = self.calculate_method_variance(values, mean).sqrt();
        
        if std_dev == 0.0 {
            return 1.0;
        }

        // 相对标准差越小，置信度越高
        let cv = std_dev / mean.abs();
        (1.0 / (1.0 + cv)).min(1.0).max(0.0)
    }
}

/// 共识结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    pub consensus_estimate: f64,
    pub method_variance: f64,
    pub individual_results: Vec<EvaluationResult>,
    pub confidence_level: f64,
}

/// 简单策略实现
pub struct SimplePolicy {
    action_probs: HashMap<String, f64>,
}

impl SimplePolicy {
    pub fn new(action_probs: HashMap<String, f64>) -> Self {
        Self { action_probs }
    }
}

impl Policy for SimplePolicy {
    fn get_action_probability(&self, _context: &[f64], action: &str) -> f64 {
        self.action_probs.get(action).copied().unwrap_or(0.0)
    }

    fn get_all_action_probabilities(&self, _context: &[f64]) -> HashMap<String, f64> {
        self.action_probs.clone()
    }
}

/// 线性价值函数
pub struct LinearValueFunction {
    weights: HashMap<String, Vec<f64>>,
    intercepts: HashMap<String, f64>,
}

impl LinearValueFunction {
    pub fn new(weights: HashMap<String, Vec<f64>>, intercepts: HashMap<String, f64>) -> Self {
        Self { weights, intercepts }
    }
}

impl ValueFunction for LinearValueFunction {
    fn estimate_value(&self, context: &[f64], action: &str) -> f64 {
        let intercept = self.intercepts.get(action).copied().unwrap_or(0.0);
        
        if let Some(weights) = self.weights.get(action) {
            let dot_product: f64 = context.iter()
                .zip(weights.iter())
                .map(|(c, w)| c * w)
                .sum();
            
            intercept + dot_product
        } else {
            intercept
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ips_evaluation() {
        let historical_data = vec![
            HistoricalRecord {
                timestamp: Utc::now(),
                context: vec![1.0, 0.5],
                action: "action1".to_string(),
                reward: 1.0,
                propensity_score: 0.5,
            },
            HistoricalRecord {
                timestamp: Utc::now(),
                context: vec![0.5, 1.0],
                action: "action2".to_string(),
                reward: 0.5,
                propensity_score: 0.3,
            },
        ];

        let target_policy = Box::new(SimplePolicy::new([
            ("action1".to_string(), 0.7),
            ("action2".to_string(), 0.3),
        ].iter().cloned().collect()));

        let evaluator = OfflinePolicyEvaluator::new(historical_data, target_policy);
        let result = evaluator.evaluate(OPEMethod::IPS).unwrap();

        assert!(result.estimated_value > 0.0);
        assert!(result.variance >= 0.0);
        assert!(result.effective_sample_size > 0.0);
    }

    #[test]
    fn test_policy_comparison() {
        let historical_data = vec![
            HistoricalRecord {
                timestamp: Utc::now(),
                context: vec![1.0, 0.5, 0.2],
                action: "low_rate".to_string(),
                reward: 0.8,
                propensity_score: 0.6,
            },
            HistoricalRecord {
                timestamp: Utc::now(),
                context: vec![0.5, 1.0, 0.8],
                action: "high_rate".to_string(),
                reward: 0.6,
                propensity_score: 0.4,
            },
        ];

        let target_policy = Box::new(SimplePolicy::new([
            ("low_rate".to_string(), 0.8),
            ("high_rate".to_string(), 0.2),
        ].iter().cloned().collect()));

        let value_function = Box::new(LinearValueFunction::new(
            [
                ("low_rate".to_string(), vec![0.5, 0.3, 0.2]),
                ("high_rate".to_string(), vec![0.2, 0.5, 0.3]),
            ].iter().cloned().collect(),
            [
                ("low_rate".to_string(), 0.1),
                ("high_rate".to_string(), 0.05),
            ].iter().cloned().collect(),
        ));

        let comparator = PolicyComparator::new(historical_data, target_policy, Some(value_function));
        let consensus = comparator.get_consensus_estimate().unwrap();

        assert!(consensus.consensus_estimate >= 0.0);
        assert!(consensus.confidence_level >= 0.0 && consensus.confidence_level <= 1.0);
        assert!(!consensus.individual_results.is_empty());
    }
}